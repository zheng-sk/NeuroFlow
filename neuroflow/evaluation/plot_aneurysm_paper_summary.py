#!/usr/bin/env python3
"""Create a paper-style aneurysm morphology summary figure.

The figure is inspired by aneurysm morphology papers: it combines a 2D image
measurement view, a 3D ROI surface view, a regular/irregular shape cue, and a
convex-hull / undulation-index panel.
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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from skimage.measure import find_contours, label, marching_cubes, regionprops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a paper-style aneurysm morphology summary figure.")
    p.add_argument("--roi", required=True, help="Aneurysm ROI NIfTI mask.")
    p.add_argument("--bg", required=True, help="Background magnitude NIfTI.")
    p.add_argument("--metrics-json", required=True, help="shape_metrics.json from calculate_aneurysm_shape_metrics.py.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pad-vox", type=int, default=8)
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
    return lab == main.label


def _voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return (np.concatenate([points, ones], axis=1) @ affine.T)[:, :3]


def _principal_axes(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coords_ijk = np.argwhere(mask > 0).astype(np.float64)
    coords_world = _voxel_to_world(coords_ijk, affine)
    center = coords_world.mean(axis=0)
    centered = coords_world - center[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    names = ["Size", "Width", "Minor"]
    colors = ["white", "#111827", "#374151"]
    axes = []
    for name, color, vec in zip(names, colors, vh):
        proj = centered @ vec
        p0 = center + float(proj.min()) * vec
        p1 = center + float(proj.max()) * vec
        axes.append({"name": name, "color": color, "p0": p0, "p1": p1, "length_mm": float(np.linalg.norm(p1 - p0))})
    return center, axes


def _world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    inv = np.linalg.inv(affine)
    return (np.concatenate([points, ones], axis=1) @ inv.T)[:, :3]


def _crop(mask: np.ndarray, pad: int) -> tuple[slice, slice, slice]:
    pts = np.argwhere(mask > 0)
    lo = np.maximum(0, pts.min(axis=0) - int(pad))
    hi = np.minimum(np.asarray(mask.shape), pts.max(axis=0) + 1 + int(pad))
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2]))


def _window(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = img[(mask > 0) & np.isfinite(img)]
    if vals.size < 20:
        vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


def _surface(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts_ijk, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    return _voxel_to_world(verts_ijk, affine), faces


def _draw_contour(ax, mask_slice: np.ndarray, color: str = "white", lw: float = 1.8) -> None:
    for contour in find_contours(mask_slice.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=lw)


def _axis_2d(axis: dict[str, Any], affine: np.ndarray, view: str, offset: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pts = _world_to_voxel(np.vstack([axis["p0"], axis["p1"]]), affine)
    if view == "axial":
        x = pts[:, 0] - offset[0]
        y = pts[:, 1] - offset[1]
    elif view == "coronal":
        x = pts[:, 0] - offset[0]
        y = pts[:, 2] - offset[1]
    else:
        x = pts[:, 1] - offset[0]
        y = pts[:, 2] - offset[1]
    return x, y


def _point_vox_to_panel_xy(point_vox: np.ndarray, view: str, offset: tuple[int, int]) -> tuple[float, float]:
    p = np.asarray(point_vox, dtype=np.float64)
    if view == "axial":
        return float(p[0] - offset[0]), float(p[1] - offset[1])
    if view == "coronal":
        return float(p[0] - offset[0]), float(p[2] - offset[1])
    return float(p[1] - offset[0]), float(p[2] - offset[1])


def _slice_panel(
    bg: np.ndarray,
    mask: np.ndarray,
    crop: tuple[slice, slice, slice],
    view: str,
    slice_index: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    sx, sy, sz = crop
    if view == "axial":
        return bg[sx, sy, slice_index].T, mask[sx, sy, slice_index].T, (sx.start, sy.start)
    if view == "coronal":
        return bg[sx, slice_index, sz].T, mask[sx, slice_index, sz].T, (sx.start, sz.start)
    return bg[slice_index, sy, sz].T, mask[slice_index, sy, sz].T, (sy.start, sz.start)


def _load_manual_neck(metrics: dict[str, Any]) -> dict[str, Any] | None:
    neck_path = metrics.get("manual_neck_json")
    if not neck_path:
        return None
    path = Path(str(neck_path)).expanduser()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_height_line(mask: np.ndarray, affine: np.ndarray, neck_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return neck-plane midpoint and farthest sac point along the manual neck-plane normal."""
    neck_world = _voxel_to_world(np.asarray(neck_info["neck_points_vox"], dtype=np.float64), affine)
    neck_mid = neck_world.mean(axis=0)
    plane_point = np.asarray(neck_info["neck_plane_point_world_mm"], dtype=np.float64)
    plane_normal = np.asarray(neck_info["neck_plane_normal_world"], dtype=np.float64)
    plane_normal = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-12)
    sac_world = _voxel_to_world(np.argwhere(mask > 0).astype(np.float64), affine)
    signed = (sac_world - plane_point[None, :]) @ plane_normal
    dome = sac_world[int(np.argmax(signed))]
    return neck_mid, dome


def _style_box(ax, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(0.5, -0.08, label, transform=ax.transAxes, ha="center", va="top", fontsize=18, family="serif")
    else:
        ax.text(0.5, -0.08, label, transform=ax.transAxes, ha="center", va="top", fontsize=18, family="serif")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")


def _draw_3d_surface(ax, verts: np.ndarray, faces: np.ndarray, axes: list[dict[str, Any]], color: str = "#d6d3d1") -> None:
    f = faces
    if f.shape[0] > 7000:
        f = f[:: int(np.ceil(f.shape[0] / 7000))]
    coll = Poly3DCollection(verts[f], facecolor=color, edgecolor="#737373", linewidths=0.08, alpha=0.90)
    ax.add_collection3d(coll)
    for axis in axes[:2]:
        p0, p1 = axis["p0"], axis["p1"]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="black", linewidth=1.8)
    mins, maxs = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


def _draw_hull(ax, verts: np.ndarray) -> None:
    hull = ConvexHull(verts)
    simplices = hull.simplices
    if simplices.shape[0] > 5000:
        simplices = simplices[:: int(np.ceil(simplices.shape[0] / 5000))]
    coll = Poly3DCollection(verts[simplices], facecolor="#78716c", edgecolor="#57534e", linewidths=0.08, alpha=0.55)
    ax.add_collection3d(coll)
    mins, maxs = verts.min(axis=0), verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-55)
    ax.text2D(0.08, 0.16, "Convex hull", transform=ax.transAxes, fontsize=11, family="serif")
    ax.annotate("", xy=(0.25, 0.40), xytext=(0.10, 0.22), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"})


def main() -> None:
    args = _parse_args()
    roi_path = Path(args.roi).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve()
    metrics = json.loads(Path(args.metrics_json).expanduser().resolve().read_text(encoding="utf-8"))
    neck_info = _load_manual_neck(metrics)

    roi, affine = _load_nifti(roi_path)
    bg, _ = _load_nifti(bg_path)
    if bg.shape != roi.shape:
        tmp = np.zeros(roi.shape, dtype=np.float32)
        sx, sy, sz = min(bg.shape[0], roi.shape[0]), min(bg.shape[1], roi.shape[1]), min(bg.shape[2], roi.shape[2])
        tmp[:sx, :sy, :sz] = bg[:sx, :sy, :sz]
        bg = tmp

    mask = _largest_component(roi >= float(args.threshold))
    center, axes = _principal_axes(mask, affine)
    center_ijk = np.rint(_world_to_voxel(center[None, :], affine)[0]).astype(int)
    center_ijk = np.clip(center_ijk, 0, np.asarray(mask.shape) - 1)
    sx, sy, sz = _crop(mask, int(args.pad_vox))
    vmin, vmax = _window(bg, mask)
    verts, faces = _surface(mask, affine)

    fig = plt.figure(figsize=(10.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.15, hspace=0.10)
    ax_2d = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")
    ax_shape = fig.add_subplot(gs[1, 0])
    ax_hull = fig.add_subplot(gs[1, 2], projection="3d")
    ax_mid_top = fig.add_subplot(gs[0, 1])
    ax_mid_bottom = fig.add_subplot(gs[1, 1])

    if neck_info is not None:
        view = str(neck_info.get("view", "axial"))
        slice_index = int(neck_info.get("slice_index", center_ijk[2]))
    else:
        view = "axial"
        slice_index = int(center_ijk[2])

    img, m, offset = _slice_panel(bg, mask, (sx, sy, sz), view, slice_index)
    ax_2d.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    _draw_contour(ax_2d, m, color="white")
    neck_value = metrics.get("manual_neck_width_mm", metrics.get("neck_width_mm", float("nan")))
    size_value = metrics.get("height_from_neck_plane_mm", metrics.get("axis_major_mm", axes[0]["length_mm"]))
    if neck_info is not None:
        neck_pts = np.asarray(neck_info["neck_points_vox"], dtype=np.float64)
        sac_pt = np.asarray(neck_info["sac_side_point_vox"], dtype=np.float64)
        x0, y0 = _point_vox_to_panel_xy(neck_pts[0], view, offset)
        x1, y1 = _point_vox_to_panel_xy(neck_pts[1], view, offset)
        xs, ys = _point_vox_to_panel_xy(sac_pt, view, offset)
        height_p0_world, height_p1_world = _manual_height_line(mask, affine, neck_info)
        height_pts_vox = _world_to_voxel(np.vstack([height_p0_world, height_p1_world]), affine)
        hx0, hy0 = _point_vox_to_panel_xy(height_pts_vox[0], view, offset)
        hx1, hy1 = _point_vox_to_panel_xy(height_pts_vox[1], view, offset)
        ax_2d.plot([x0, x1], [y0, y1], color="white", linewidth=2.6)
        ax_2d.plot([hx0, hx1], [hy0, hy1], color="white", linewidth=2.0, linestyle="--")
        ax_2d.scatter([x0, x1], [y0, y1], color="white", s=24, edgecolors="black", linewidths=0.5, zorder=5)
        ax_2d.scatter([xs], [ys], color="#22c55e", s=32, edgecolors="white", linewidths=0.8, zorder=5)
    else:
        for axis in axes[:2]:
            xx, yy = _axis_2d(axis, affine, view, offset)
            ax_2d.plot(xx, yy, color="white", linewidth=1.4)
    ax_2d.text(0.06, 0.80, f"Size:\n{float(size_value):.2f} mm", color="white", transform=ax_2d.transAxes, fontsize=12, family="serif")
    ax_2d.text(0.58, 0.72, f"Neck:\n{float(neck_value):.2f} mm", color="white", transform=ax_2d.transAxes, fontsize=12, family="serif")
    ax_2d.set_xticks([])
    ax_2d.set_yticks([])
    _style_box(ax_2d, "2D slice")

    _draw_3d_surface(ax_3d, verts, faces, axes)
    if neck_info is not None:
        neck_world = _voxel_to_world(np.asarray(neck_info["neck_points_vox"], dtype=np.float64), affine)
        height_p0_world, height_p1_world = _manual_height_line(mask, affine, neck_info)
        ax_3d.plot(
            [neck_world[0, 0], neck_world[1, 0]],
            [neck_world[0, 1], neck_world[1, 1]],
            [neck_world[0, 2], neck_world[1, 2]],
            color="black",
            linewidth=3.0,
        )
        ax_3d.plot(
            [height_p0_world[0], height_p1_world[0]],
            [height_p0_world[1], height_p1_world[1]],
            [height_p0_world[2], height_p1_world[2]],
            color="black",
            linewidth=2.4,
            linestyle="--",
        )
    ax_3d.text2D(0.05, 0.20, f"Size: {float(size_value):.2f} mm\nNeck: {float(neck_value):.2f} mm", transform=ax_3d.transAxes, fontsize=12, family="serif")
    _style_box(ax_3d, "3D surface")

    mip = np.max(bg[sx, sy, sz], axis=2).T
    mmip = np.max(mask[sx, sy, sz], axis=2).T
    ax_shape.imshow(mip, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    _draw_contour(ax_shape, mmip, color="#f97316", lw=2.0)
    ui = metrics.get("undulation_index")
    nsi = metrics.get("non_sphericity_index")
    shape_label = "Irregular" if ((ui is not None and ui >= 0.10) or (nsi is not None and nsi >= 0.20)) else "Regular"
    ax_shape.text(0.04, 0.08, shape_label, transform=ax_shape.transAxes, fontsize=13, color="white", family="serif")
    ax_shape.set_xticks([])
    ax_shape.set_yticks([])
    _style_box(ax_shape, "Shape class cue")

    _draw_hull(ax_hull, verts)
    _style_box(ax_hull, "Convex hull")

    for ax, y_text in [(ax_mid_top, 0.5), (ax_mid_bottom, 0.5)]:
        ax.axis("off")
        ax.text(0.5, y_text + 0.13, "Compare", ha="center", va="center", fontsize=17, family="serif")
        arrow = FancyArrowPatch((0.08, y_text), (0.92, y_text), arrowstyle="<->", mutation_scale=28, lw=1.2, fill=False, transform=ax.transAxes)
        ax.add_patch(arrow)

    fig.suptitle("Aneurysm morphology summary", fontsize=17, family="serif", y=0.97)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Paper-style summary written: {out}")


if __name__ == "__main__":
    main()
