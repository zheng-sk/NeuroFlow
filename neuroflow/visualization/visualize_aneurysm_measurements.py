#!/usr/bin/env python3
"""Create a measurement overlay figure for a selected aneurysm ROI.

The output is a PNG with axial/coronal/sagittal slices through the ROI
centroid, the aneurysm contour, bounding-box measurements, and principal-axis
measurement lines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/neuroflow_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import find_contours, label, marching_cubes, regionprops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize aneurysm ROI measurements over image slices.")
    p.add_argument("--roi", required=True, help="Aneurysm ROI NIfTI mask.")
    p.add_argument("--bg", required=True, help="Background magnitude NIfTI.")
    p.add_argument("--metrics-json", default="", help="Optional shape_metrics.json from calculate_aneurysm_shape_metrics.py.")
    p.add_argument("--out", default="", help="Output PNG. Default: <roi_parent>/<roi_stem>_measurements.png")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pad-vox", type=int, default=10, help="Voxel padding around the ROI in each view.")
    return p.parse_args()


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, np.asarray(img.affine, dtype=np.float64)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask > 0)
    props = regionprops(lab)
    if not props:
        raise ValueError("ROI mask is empty.")
    main = max(props, key=lambda p: p.area)
    return (lab == main.label)


def _voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return (np.concatenate([points, ones], axis=1) @ affine.T)[:, :3]


def _world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    inv = np.linalg.inv(affine)
    return (np.concatenate([points, ones], axis=1) @ inv.T)[:, :3]


def _principal_axes(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coords_ijk = np.argwhere(mask > 0).astype(np.float64)
    coords_world = _voxel_to_world(coords_ijk, affine)
    center_world = coords_world.mean(axis=0)
    centered = coords_world - center_world[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axes: list[dict[str, Any]] = []
    names = ["major", "intermediate", "minor"]
    colors = ["#ef4444", "#2563eb", "#16a34a"]
    for name, color, vec in zip(names, colors, vh):
        proj = centered @ vec
        p0 = center_world + float(proj.min()) * vec
        p1 = center_world + float(proj.max()) * vec
        length = float(np.linalg.norm(p1 - p0))
        axes.append({"name": name, "color": color, "p0": p0, "p1": p1, "length_mm": length})
    return center_world, axes


def _crop_limits(mask: np.ndarray, pad: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pts = np.argwhere(mask > 0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0) + 1
    shape = np.asarray(mask.shape)
    lo = np.maximum(0, lo - int(pad))
    hi = np.minimum(shape, hi + int(pad))
    return (int(lo[0]), int(hi[0])), (int(lo[1]), int(hi[1])), (int(lo[2]), int(hi[2]))


def _window_image(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = img[np.isfinite(img)]
    roi_vals = img[(mask > 0) & np.isfinite(img)]
    src = roi_vals if roi_vals.size >= 20 else vals
    if src.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(src, [1, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _draw_contours(ax, mask_slice: np.ndarray, color: str = "#f97316") -> None:
    for contour in find_contours(mask_slice.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.8)


def _draw_axis_projection(
    ax,
    axis_info: dict[str, Any],
    affine: np.ndarray,
    view: str,
    offset: tuple[int, int],
) -> None:
    pts_ijk = _world_to_voxel(np.vstack([axis_info["p0"], axis_info["p1"]]), affine)
    if view == "axial":
        x = pts_ijk[:, 0] - offset[0]
        y = pts_ijk[:, 1] - offset[1]
    elif view == "coronal":
        x = pts_ijk[:, 0] - offset[0]
        y = pts_ijk[:, 2] - offset[1]
    else:
        x = pts_ijk[:, 1] - offset[0]
        y = pts_ijk[:, 2] - offset[1]
    ax.plot(x, y, color=axis_info["color"], linewidth=2.2, alpha=0.95)


def _draw_automatic_neck_projection(
    ax,
    metrics: dict[str, Any],
    view: str,
    offset: tuple[int, int],
) -> None:
    voxels = metrics.get("automatic_neck_voxels_ijk")
    if not voxels:
        return
    pts = np.asarray(voxels, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return
    if view == "axial":
        x = pts[:, 0] - offset[0]
        y = pts[:, 1] - offset[1]
    elif view == "coronal":
        x = pts[:, 0] - offset[0]
        y = pts[:, 2] - offset[1]
    else:
        x = pts[:, 1] - offset[0]
        y = pts[:, 2] - offset[1]
    ax.scatter(
        x,
        y,
        s=30,
        color="#06b6d4",
        edgecolors="black",
        linewidths=0.35,
        alpha=0.92,
        zorder=6,
        label="automatic neck voxels",
    )


def _draw_bbox(ax, lo0: int, hi0: int, lo1: int, hi1: int, color: str = "#facc15") -> None:
    xs = [lo0, hi0, hi0, lo0, lo0]
    ys = [lo1, lo1, hi1, hi1, lo1]
    ax.plot(xs, ys, color=color, linewidth=1.5, linestyle="--")


def _read_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_text(metrics: dict[str, Any], axes: list[dict[str, Any]]) -> str:
    lines = ["PCA extent lines shown in image/3D"]
    for axis in axes:
        lines.append(f"{axis['name']:<13} {axis['length_mm']:>8.2f} mm")
    lines.append("")

    if metrics:
        lines.append("Shape metrics")
        wanted = [
            ("volume_mm3", "Volume", "mm3"),
            ("surface_area_mm2", "Surface area", "mm2"),
            ("bbox_width_mm", "BBox width", "mm"),
            ("bbox_height_mm", "BBox height", "mm"),
            ("bbox_depth_mm", "BBox depth", "mm"),
            ("axis_major_mm", "Axis major", "mm"),
            ("axis_intermediate_mm", "Axis intermediate", "mm"),
            ("axis_minor_mm", "Axis minor", "mm"),
            ("elongation", "Elongation", ""),
            ("flatness", "Flatness", ""),
            ("equivalent_sphere_diam_mm", "Equiv sphere diam", "mm"),
            ("neck_width_mm", "Neck width approx", "mm"),
            ("automatic_neck_voxel_count", "Auto neck voxels", ""),
            ("automatic_neck_area_mm2", "Auto neck area", "mm2"),
            ("automatic_neck_equivalent_diameter_mm", "Auto neck eq diam", "mm"),
            ("max_cross_section_diam_mm", "Max cross-section", "mm"),
            ("aspect_ratio_approx", "Aspect ratio approx", ""),
            ("bottleneck_factor_approx", "Bottleneck factor approx", ""),
            ("sphericity", "Sphericity", ""),
            ("non_sphericity_index", "NSI", ""),
            ("undulation_index", "UI", ""),
            ("num_connected_components", "Connected components", ""),
            ("centroid_x_mm", "Centroid x", "mm"),
            ("centroid_y_mm", "Centroid y", "mm"),
            ("centroid_z_mm", "Centroid z", "mm"),
        ]
        seen = set()
        for key, label_name, unit in wanted:
            if key in metrics and metrics[key] is not None:
                suffix = f" {unit}" if unit else ""
                lines.append(f"{label_name:<22} {metrics[key]}{suffix}")
                seen.add(key)
        for key in sorted(metrics):
            if key in seen or key.endswith("_path") or key == "seg_path":
                continue
            value = metrics[key]
            if isinstance(value, (int, float)) and value is not None:
                lines.append(f"{key:<22} {value}")
    else:
        lines.append("Metrics JSON not provided.")
    return "\n".join(lines)


def _draw_3d_roi(ax, mask: np.ndarray, affine: np.ndarray, axes: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    verts_ijk, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    verts = _voxel_to_world(verts_ijk, affine)
    if faces.shape[0] > 6000:
        stride = int(np.ceil(faces.shape[0] / 6000))
        faces = faces[::stride]
    mesh = Poly3DCollection(verts[faces], alpha=0.34, facecolor="#f97316", edgecolor="#c2410c", linewidths=0.08)
    ax.add_collection3d(mesh)

    for axis in axes:
        p0 = axis["p0"]
        p1 = axis["p1"]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=axis["color"],
            linewidth=3.0,
            label=f"{axis['name']} PCA extent",
        )

    neck_voxels = metrics.get("automatic_neck_voxels_ijk")
    if neck_voxels:
        neck_pts = _voxel_to_world(np.asarray(neck_voxels, dtype=np.float64), affine)
        ax.scatter(
            neck_pts[:, 0],
            neck_pts[:, 1],
            neck_pts[:, 2],
            color="#06b6d4",
            edgecolors="black",
            linewidths=0.25,
            s=26,
            depthshade=False,
            label="automatic neck voxels",
        )

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1.0)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_title("3D ROI surface with PCA extents")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def main() -> None:
    args = _parse_args()
    roi_path = Path(args.roi).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve()
    metrics_path = Path(args.metrics_json).expanduser().resolve() if args.metrics_json else None

    roi_raw, affine = _load_nifti(roi_path)
    bg, bg_affine = _load_nifti(bg_path)
    if roi_raw.shape != bg.shape:
        cropped = np.zeros(roi_raw.shape, dtype=np.float32)
        sx = min(roi_raw.shape[0], bg.shape[0])
        sy = min(roi_raw.shape[1], bg.shape[1])
        sz = min(roi_raw.shape[2], bg.shape[2])
        cropped[:sx, :sy, :sz] = bg[:sx, :sy, :sz]
        bg = cropped
    if not np.allclose(affine, bg_affine, atol=1e-3):
        print("Warning: ROI and background affines differ. Overlay assumes they are already registered.")

    mask = _largest_component(roi_raw >= float(args.threshold))
    center_world, axes = _principal_axes(mask, affine)
    center_ijk = np.rint(_world_to_voxel(center_world[None, :], affine)[0]).astype(int)
    center_ijk = np.clip(center_ijk, 0, np.asarray(mask.shape) - 1)

    (x0, x1), (y0, y1), (z0, z1) = _crop_limits(mask, int(args.pad_vox))
    vmin, vmax = _window_image(bg, mask)
    metrics = _read_metrics(metrics_path)

    z = int(center_ijk[2])
    y = int(center_ijk[1])
    x = int(center_ijk[0])

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05], height_ratios=[1.0, 1.0])
    ax_axial = fig.add_subplot(gs[0, 0])
    ax_coronal = fig.add_subplot(gs[0, 1])
    ax_sagittal = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")
    ax_text = fig.add_subplot(gs[1, 1:])

    panels = [
        (ax_axial, "axial", bg[x0:x1, y0:y1, z].T, mask[x0:x1, y0:y1, z].T, (x0, y0), (0, x1 - x0, 0, y1 - y0), "Axial"),
        (ax_coronal, "coronal", bg[x0:x1, y, z0:z1].T, mask[x0:x1, y, z0:z1].T, (x0, z0), (0, x1 - x0, 0, z1 - z0), "Coronal"),
        (ax_sagittal, "sagittal", bg[x, y0:y1, z0:z1].T, mask[x, y0:y1, z0:z1].T, (y0, z0), (0, y1 - y0, 0, z1 - z0), "Sagittal"),
    ]

    for ax, view, img_slice, mask_slice, offset, extent, title in panels:
        ax.imshow(img_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        _draw_contours(ax, mask_slice)
        _draw_bbox(ax, 0, int(extent[1]), 0, int(extent[3]))
        for axis in axes:
            _draw_axis_projection(ax, axis, affine, view=view, offset=offset)
        _draw_automatic_neck_projection(ax, metrics, view=view, offset=offset)
        ax.set_title(f"{title} through ROI centroid")
        ax.set_xlabel("voxel")
        ax.set_ylabel("voxel")
        if metrics.get("automatic_neck_voxels_ijk"):
            ax.legend(loc="lower right", fontsize=7, frameon=True)

    _draw_3d_roi(ax_3d, mask, affine, axes, metrics)

    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        _metric_text(metrics, axes),
        ha="left",
        va="top",
        fontsize=9.6,
        family="monospace",
        linespacing=1.25,
    )

    out_path = Path(args.out).expanduser().resolve() if args.out else roi_path.with_name(roi_path.name.replace(".nii.gz", "").replace(".nii", "") + "_measurements.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Aneurysm ROI measurement overlay - {roi_path.name}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Measurement overlay written: {out_path}")


if __name__ == "__main__":
    main()
