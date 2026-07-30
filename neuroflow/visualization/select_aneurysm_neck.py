#!/usr/bin/env python3
"""Manual aneurysm neck selector and sac-mask splitter.

This is intended as the second manual step after selecting a rough aneurysm ROI
that may include a small parent-vessel segment.

Workflow
--------
1. Load the rough ROI mask and background magnitude image.
2. Pick the neck in one of three simultaneous slice views.
3. Click two neck endpoints, then click one point on the aneurysm-sac side.
4. Save a sac-only mask and a JSON sidecar with the manual neck plane.

The neck plane is defined by the clicked neck line and the current image-slice
normal. The sac-side point selects which side of that plane to keep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/neuroflow_matplotlib")

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from skimage.measure import find_contours, label, regionprops

try:
    import pyvista as pv
    pv.OFF_SCREEN = False
except Exception:
    pv = None


VIEWS = ("axial", "coronal", "sagittal")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select aneurysm neck and export sac-only mask.")
    p.add_argument("--roi", required=True, help="Rough aneurysm ROI NIfTI mask.")
    p.add_argument("--bg", required=True, help="Background magnitude NIfTI.")
    p.add_argument("--out", required=True, help="Output sac-only NIfTI mask.")
    p.add_argument("--view", choices=VIEWS, default="axial", help="Preferred initial view for slice keys.")
    p.add_argument("--slice", type=int, default=None, help="Optional initial slice index for --view.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dilate", type=int, default=0, help="Optional dilation iterations before cutting, usually 0.")
    p.add_argument(
        "--mode",
        choices=["qt", "manual", "pyvista", "auto"],
        default="qt",
        help="qt opens an interactive 2D picker; manual writes a guide PNG and asks for terminal coordinates; pyvista opens an experimental 3D picker; auto saves the automatic 3D candidate without GUI.",
    )
    p.add_argument("--guide-out", default="", help="Manual-mode guide PNG path. Default: <out>_neck_guide.png")
    p.add_argument("--window-width", type=int, default=1400)
    p.add_argument("--window-height", type=int, default=900)
    return p.parse_args()


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, np.asarray(img.affine, dtype=np.float64), img


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask > 0)
    props = regionprops(lab)
    if not props:
        raise ValueError("ROI mask is empty.")
    main = max(props, key=lambda p: p.area)
    return lab == main.label


def _crop_limits(mask: np.ndarray, pad: int = 8) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pts = np.argwhere(mask > 0)
    lo = np.maximum(0, pts.min(axis=0) - int(pad))
    hi = np.minimum(np.asarray(mask.shape), pts.max(axis=0) + 1 + int(pad))
    return (int(lo[0]), int(hi[0])), (int(lo[1]), int(hi[1])), (int(lo[2]), int(hi[2]))


def _voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.concatenate([pts, ones], axis=1) @ affine.T)[:, :3]


def _world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    inv = np.linalg.inv(affine)
    return (np.concatenate([pts, ones], axis=1) @ inv.T)[:, :3]


def _window_image(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = img[(mask > 0) & np.isfinite(img)]
    if vals.size < 20:
        vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99.5])
    return float(lo), float(max(hi, lo + 1e-6))


def _build_grid(bg: np.ndarray, roi_mask: np.ndarray, affine: np.ndarray) -> pv.StructuredGrid:
    if pv is None:
        raise RuntimeError("PyVista is not available. Use --mode manual.")
    nx, ny, nz = bg.shape
    i, j, k = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    ijk1 = np.stack([i, j, k, np.ones_like(i)], axis=-1)
    xyz = ijk1 @ affine.T
    grid = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    grid.point_data["mag"] = bg.ravel(order="F")
    grid.point_data["roi"] = roi_mask.astype(np.uint8).ravel(order="F")
    return grid


def _slice_normal_world(view: str, affine: np.ndarray) -> np.ndarray:
    col = {"sagittal": 0, "coronal": 1, "axial": 2}[view]
    n = affine[:3, col].astype(np.float64)
    return n / max(np.linalg.norm(n), 1e-12)


def _slice_axis(view: str) -> int:
    return {"sagittal": 0, "coronal": 1, "axial": 2}[view]


def _default_slice(mask: np.ndarray, view: str) -> int:
    axis = _slice_axis(view)
    coords = np.argwhere(mask > 0)
    return int(np.rint(coords[:, axis].mean()))


def _clip_slice(value: int, shape: tuple[int, int, int], view: str) -> int:
    axis = _slice_axis(view)
    return int(max(0, min(int(value), int(shape[axis]) - 1)))


def _neck_side_cut(
    roi_mask: np.ndarray,
    affine: np.ndarray,
    neck_a_vox: np.ndarray,
    neck_b_vox: np.ndarray,
    sac_vox: np.ndarray,
    view: str,
    dilate: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    neck_a_vox, neck_b_vox, sac_vox = [np.asarray(p, dtype=np.float64) for p in (neck_a_vox, neck_b_vox, sac_vox)]
    neck_a, neck_b, sac_pt = _voxel_to_world(np.vstack([neck_a_vox, neck_b_vox, sac_vox]), affine)

    neck_dir = neck_b - neck_a
    neck_len = float(np.linalg.norm(neck_dir))
    if neck_len <= 1e-6:
        raise ValueError("Neck endpoints are too close together.")
    neck_dir = neck_dir / neck_len
    slice_normal = _slice_normal_world(view, affine)
    plane_normal = np.cross(neck_dir, slice_normal)
    norm = float(np.linalg.norm(plane_normal))
    if norm <= 1e-6:
        raise ValueError("Could not define neck plane from selected line and slice normal.")
    plane_normal = plane_normal / norm
    if float((sac_pt - neck_a) @ plane_normal) < 0:
        plane_normal = -plane_normal

    coords = np.argwhere(roi_mask > 0).astype(np.float64)
    coords_world = _voxel_to_world(coords, affine)
    signed = (coords_world - neck_a[None, :]) @ plane_normal
    candidate = np.zeros(roi_mask.shape, dtype=bool)
    keep_coords = coords[signed >= -1e-6].astype(int)
    candidate[keep_coords[:, 0], keep_coords[:, 1], keep_coords[:, 2]] = True

    lab = label(candidate)
    comp_label = 0
    sac_idx = np.rint(sac_vox).astype(int)
    if np.all((sac_idx >= 0) & (sac_idx < np.asarray(roi_mask.shape))):
        comp_label = int(lab[tuple(sac_idx)])
    if comp_label == 0:
        cc = np.argwhere(candidate)
        if cc.size == 0:
            raise ValueError("Neck plane removed all ROI voxels. Re-select the sac-side point.")
        tree = cKDTree(cc.astype(np.float64))
        _, nearest = tree.query(sac_vox[None, :], k=1)
        nearest_idx = cc[int(nearest[0])]
        comp_label = int(lab[tuple(nearest_idx)])
    sac_mask = lab == comp_label

    if int(dilate) > 0:
        sac_mask = binary_dilation(sac_mask, iterations=int(dilate)) & roi_mask

    sac_coords_world = _voxel_to_world(np.argwhere(sac_mask > 0).astype(np.float64), affine)
    sac_signed = (sac_coords_world - neck_a[None, :]) @ plane_normal
    height_mm = float(np.max(sac_signed)) if sac_signed.size else float("nan")
    info = {
        "view": view,
        "neck_points_vox": [[round(float(v), 3) for v in neck_a_vox], [round(float(v), 3) for v in neck_b_vox]],
        "sac_side_point_vox": [round(float(v), 3) for v in sac_vox],
        "neck_points_world_mm": [[round(float(v), 4) for v in neck_a], [round(float(v), 4) for v in neck_b]],
        "sac_side_point_world_mm": [round(float(v), 4) for v in sac_pt],
        "neck_plane_point_world_mm": [round(float(v), 4) for v in neck_a],
        "neck_plane_normal_world": [round(float(v), 8) for v in plane_normal],
        "manual_neck_width_mm": round(neck_len, 4),
        "height_from_neck_plane_mm": round(height_mm, 4),
        "roi_voxels": int(roi_mask.sum()),
        "sac_voxels": int(sac_mask.sum()),
    }
    return sac_mask.astype(np.uint8), info


def _automatic_neck_candidate(roi_mask: np.ndarray, affine: np.ndarray) -> dict[str, Any] | None:
    """Find the narrowest cross-section perpendicular to the longest 3D axis."""
    coords_ijk = np.argwhere(roi_mask > 0).astype(np.float64)
    if coords_ijk.shape[0] < 8:
        return None
    coords_world = _voxel_to_world(coords_ijk, affine)
    center_world = coords_world.mean(axis=0)
    centered = coords_world - center_world[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    major_dir = vh[0]
    projections = centered @ major_dir

    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    step = float(np.min(zooms))
    bins = np.arange(float(projections.min()), float(projections.max()) + step, step)
    if bins.size < 2:
        return None

    voxel_cross_area = step ** 2
    slice_infos: list[dict[str, Any]] = []
    total_voxels = int(coords_ijk.shape[0])
    min_side_count = max(20, int(round(0.03 * total_voxels)))
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_slice = (projections >= lo) & (projections < hi)
        n = int(in_slice.sum())
        if n < 4:
            continue
        below = int((projections < lo).sum())
        above = int((projections >= hi).sum())
        if min(below, above) < min_side_count:
            continue
        slice_infos.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "indices": np.flatnonzero(in_slice),
                "count": n,
                "below_count": below,
                "above_count": above,
                "diameter": 2.0 * math.sqrt(n * voxel_cross_area / math.pi),
            }
        )
    if not slice_infos:
        return None

    neck = min(slice_infos, key=lambda item: item["diameter"])
    neck_indices = np.asarray(neck["indices"], dtype=int)
    neck_voxels = coords_ijk[neck_indices]
    neck_world = coords_world[neck_indices]
    neck_center = neck_world.mean(axis=0)

    if neck_world.shape[0] >= 2:
        diffs = neck_world[:, None, :] - neck_world[None, :, :]
        dist2 = np.sum(diffs * diffs, axis=2)
        i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
        endpoint_voxels = np.vstack([neck_voxels[i], neck_voxels[j]])
    else:
        endpoint_voxels = np.vstack([neck_voxels[0], neck_voxels[0]])

    return {
        "method": "narrowest cross-section perpendicular to longest PCA axis",
        "axis_world": major_dir,
        "center_world": neck_center,
        "projection_range_mm": [float(neck["lo"]), float(neck["hi"])],
        "voxel_count": int(neck["count"]),
        "area_mm2": float(neck["count"] * voxel_cross_area),
        "equivalent_diameter_mm": float(neck["diameter"]),
        "voxels_ijk": neck_voxels.astype(int),
        "endpoint_voxels": endpoint_voxels.astype(np.float64),
    }


def _largest_component_from_mask(mask: np.ndarray) -> np.ndarray:
    lab = label(mask > 0)
    props = regionprops(lab)
    if not props:
        return np.zeros(mask.shape, dtype=bool)
    main = max(props, key=lambda p: p.area)
    return lab == main.label


def _automatic_neck_cut(
    roi_mask: np.ndarray,
    affine: np.ndarray,
    candidate: dict[str, Any],
    dilate: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.argwhere(roi_mask > 0).astype(np.float64)
    coords_world = _voxel_to_world(coords, affine)
    plane_point = np.asarray(candidate["center_world"], dtype=np.float64)
    plane_normal = np.asarray(candidate["axis_world"], dtype=np.float64)
    plane_normal = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-12)

    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    tol = float(np.min(zooms)) * 0.5
    signed = (coords_world - plane_point[None, :]) @ plane_normal

    side_masks = []
    for sign in (1.0, -1.0):
        keep = signed * sign >= -tol
        side = np.zeros(roi_mask.shape, dtype=bool)
        keep_coords = coords[keep].astype(int)
        if keep_coords.size:
            side[keep_coords[:, 0], keep_coords[:, 1], keep_coords[:, 2]] = True
        side = _largest_component_from_mask(side)
        side_coords = np.argwhere(side > 0).astype(np.float64)
        if side_coords.size:
            side_world = _voxel_to_world(side_coords, affine)
            rel = side_world - plane_point[None, :]
            axial = rel @ plane_normal
            radial = rel - axial[:, None] * plane_normal[None, :]
            max_radius = float(np.max(np.linalg.norm(radial, axis=1)))
        else:
            max_radius = 0.0
        # Prefer the wider/bulging side; volume breaks ties.
        side_masks.append((max_radius, int(side.sum()), sign, side))

    _, _, sign, sac_mask = max(side_masks, key=lambda item: (item[0], item[1]))
    if sign < 0:
        plane_normal = -plane_normal

    if int(dilate) > 0:
        sac_mask = binary_dilation(sac_mask, iterations=int(dilate)) & roi_mask

    sac_coords_world = _voxel_to_world(np.argwhere(sac_mask > 0).astype(np.float64), affine)
    sac_signed = (sac_coords_world - plane_point[None, :]) @ plane_normal
    height_mm = float(np.max(sac_signed)) if sac_signed.size else float("nan")
    sac_voxels = np.argwhere(sac_mask > 0).astype(np.float64)
    if sac_signed.size:
        dome_idx = int(np.argmax(sac_signed))
        sac_side_vox = sac_voxels[dome_idx]
        sac_side_world = sac_coords_world[dome_idx]
    else:
        sac_side_vox = np.asarray(candidate["endpoint_voxels"], dtype=np.float64)[0]
        sac_side_world = _voxel_to_world(sac_side_vox[None, :], affine)[0]
    endpoint_vox = np.asarray(candidate["endpoint_voxels"], dtype=np.float64)
    endpoint_world = _voxel_to_world(endpoint_vox, affine)
    eq_diam = float(candidate["equivalent_diameter_mm"])
    slice_index = int(np.rint(np.asarray(candidate["voxels_ijk"], dtype=np.float64)[:, 2].mean()))

    info = {
        "selection_mode": "automatic",
        "automatic_neck_method": candidate["method"],
        "view": "axial",
        "slice_index": slice_index,
        "neck_points_vox": [[round(float(v), 3) for v in endpoint_vox[0]], [round(float(v), 3) for v in endpoint_vox[1]]],
        "sac_side_point_vox": [round(float(v), 3) for v in sac_side_vox],
        "neck_points_world_mm": [[round(float(v), 4) for v in endpoint_world[0]], [round(float(v), 4) for v in endpoint_world[1]]],
        "sac_side_point_world_mm": [round(float(v), 4) for v in sac_side_world],
        "neck_plane_point_world_mm": [round(float(v), 4) for v in plane_point],
        "neck_plane_normal_world": [round(float(v), 8) for v in plane_normal],
        "manual_neck_width_mm": round(eq_diam, 4),
        "automatic_neck_equivalent_diameter_mm": round(eq_diam, 4),
        "automatic_neck_area_mm2": round(float(candidate["area_mm2"]), 6),
        "automatic_neck_voxel_count": int(candidate["voxel_count"]),
        "automatic_neck_voxels_ijk": np.asarray(candidate["voxels_ijk"], dtype=int).tolist(),
        "height_from_neck_plane_mm": round(height_mm, 4),
        "roi_voxels": int(roi_mask.sum()),
        "sac_voxels": int(sac_mask.sum()),
    }
    return sac_mask.astype(np.uint8), info


def _save_sac_outputs(
    sac_mask: np.ndarray,
    info: dict[str, Any],
    ref_img: nib.Nifti1Image,
    out_path: Path,
    source_roi_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(sac_mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))
    sidecar = out_path.with_suffix("")
    if sidecar.suffix == ".nii":
        sidecar = sidecar.with_suffix("")
    sidecar = sidecar.with_name(sidecar.name + "_neck.json")
    info = {
        "source_roi_path": str(source_roi_path),
        "out_path": str(out_path),
        **info,
        "neck_json": str(sidecar),
    }
    sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


def _draw_contours(ax, mask_slice: np.ndarray, x0: int, y0: int) -> None:
    for contour in find_contours(mask_slice.astype(float), 0.5):
        ax.plot(contour[:, 1] + x0, contour[:, 0] + y0, color="#f97316", linewidth=1.8)


def _manual_guide(
    roi_mask: np.ndarray,
    bg: np.ndarray,
    guide_path: Path,
) -> dict[str, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slice_indices = {view: _default_slice(roi_mask, view) for view in VIEWS}
    (x0, x1), (y0, y1), (z0, z1) = _crop_limits(roi_mask, pad=10)
    vmin, vmax = _window_image(bg, roi_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        ("axial", bg[x0:x1, y0:y1, slice_indices["axial"]].T, roi_mask[x0:x1, y0:y1, slice_indices["axial"]].T, (x0, x1, y0, y1), "x voxel", "y voxel"),
        ("coronal", bg[x0:x1, slice_indices["coronal"], z0:z1].T, roi_mask[x0:x1, slice_indices["coronal"], z0:z1].T, (x0, x1, z0, z1), "x voxel", "z voxel"),
        ("sagittal", bg[slice_indices["sagittal"], y0:y1, z0:z1].T, roi_mask[slice_indices["sagittal"], y0:y1, z0:z1].T, (y0, y1, z0, z1), "y voxel", "z voxel"),
    ]
    for ax, (view, img, mask_sl, extent, xlabel, ylabel) in zip(axes, panels):
        ax.imshow(img, cmap="gray", origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="equal")
        _draw_contours(ax, mask_sl, int(extent[0]), int(extent[2]))
        ax.set_title(f"{view} slice {slice_indices[view]}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color="yellow", alpha=0.28, linewidth=0.7)
    fig.suptitle("Manual neck selection guide: type coordinates from one panel only", fontsize=13)
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(guide_path, dpi=180)
    plt.close(fig)
    return slice_indices


def _parse_pair(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Enter exactly two numbers, e.g. 18.5 15.0")
    return float(parts[0]), float(parts[1])


def _manual_mode(
    roi_mask: np.ndarray,
    bg: np.ndarray,
    affine: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: Path,
    roi_path: Path,
    args: argparse.Namespace,
) -> None:
    default_guide = out_path.with_suffix("")
    if default_guide.suffix == ".nii":
        default_guide = default_guide.with_suffix("")
    guide_path = Path(args.guide_out).expanduser().resolve() if args.guide_out else default_guide.with_name(default_guide.name + "_neck_guide.png")
    slice_indices = _manual_guide(roi_mask, bg, guide_path)
    print(f"\nNeck guide written: {guide_path}")
    print("Open the PNG, choose ONE panel/view, and type coordinates from that panel's axes.")
    print("Views: axial uses x y, coronal uses x z, sagittal uses y z.")
    print("Press Ctrl+C to cancel.\n")

    while True:
        view = input("view [axial/coronal/sagittal]: ").strip().lower()
        if view in VIEWS:
            break
        print("Invalid view.")

    while True:
        p1 = _parse_pair(input("neck endpoint 1 coordinates: "))
        p2 = _parse_pair(input("neck endpoint 2 coordinates: "))
        p3 = _parse_pair(input("sac-side point coordinates: "))
        s = float(slice_indices[view])
        if view == "axial":
            neck_a = np.array([p1[0], p1[1], s], dtype=np.float64)
            neck_b = np.array([p2[0], p2[1], s], dtype=np.float64)
            sac = np.array([p3[0], p3[1], s], dtype=np.float64)
        elif view == "coronal":
            neck_a = np.array([p1[0], s, p1[1]], dtype=np.float64)
            neck_b = np.array([p2[0], s, p2[1]], dtype=np.float64)
            sac = np.array([p3[0], s, p3[1]], dtype=np.float64)
        else:
            neck_a = np.array([s, p1[0], p1[1]], dtype=np.float64)
            neck_b = np.array([s, p2[0], p2[1]], dtype=np.float64)
            sac = np.array([s, p3[0], p3[1]], dtype=np.float64)

        try:
            sac_mask, info = _neck_side_cut(
                roi_mask=roi_mask,
                affine=affine,
                neck_a_vox=neck_a,
                neck_b_vox=neck_b,
                sac_vox=sac,
                view=view,
                dilate=int(args.dilate),
            )
            break
        except ValueError as exc:
            print(f"\nSelection failed: {exc}")
            print("Try the same view again. Use a sac-side point clearly inside the aneurysm dome.\n")
    info["slice_index"] = int(slice_indices[view])
    info["guide_png"] = str(guide_path)
    _save_sac_outputs(sac_mask, info, ref_img, out_path, roi_path)


def _auto_mode(
    roi_mask: np.ndarray,
    affine: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: Path,
    roi_path: Path,
    args: argparse.Namespace,
) -> None:
    candidate = _automatic_neck_candidate(roi_mask, affine)
    if candidate is None:
        raise RuntimeError("Automatic neck candidate could not be found for this ROI.")
    sac_mask, info = _automatic_neck_cut(
        roi_mask=roi_mask,
        affine=affine,
        candidate=candidate,
        dilate=int(args.dilate),
    )
    _save_sac_outputs(sac_mask, info, ref_img, out_path, roi_path)


def _view_panel(
    bg: np.ndarray,
    roi_mask: np.ndarray,
    view: str,
    slice_index: int,
    limits: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str, str]:
    (x0, x1), (y0, y1), (z0, z1) = limits
    if view == "axial":
        return (
            bg[x0:x1, y0:y1, slice_index].T,
            roi_mask[x0:x1, y0:y1, slice_index].T,
            (x0, x1, y0, y1),
            "x voxel",
            "y voxel",
        )
    if view == "coronal":
        return (
            bg[x0:x1, slice_index, z0:z1].T,
            roi_mask[x0:x1, slice_index, z0:z1].T,
            (x0, x1, z0, z1),
            "x voxel",
            "z voxel",
        )
    return (
        bg[slice_index, y0:y1, z0:z1].T,
        roi_mask[slice_index, y0:y1, z0:z1].T,
        (y0, y1, z0, z1),
        "y voxel",
        "z voxel",
    )


class QtNeckSelector:
    def __init__(
        self,
        roi_mask: np.ndarray,
        bg: np.ndarray,
        affine: np.ndarray,
        ref_img: nib.Nifti1Image,
        out_path: Path,
        roi_path: Path,
        args: argparse.Namespace,
    ) -> None:
        import matplotlib

        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.Button = Button
        self.roi_mask = roi_mask.astype(bool)
        self.bg = bg.astype(np.float32)
        self.affine = affine
        self.ref_img = ref_img
        self.out_path = out_path
        self.roi_path = roi_path
        self.args = args
        self.vmin, self.vmax = _window_image(self.bg, self.roi_mask)
        self.limits = _crop_limits(self.roi_mask, pad=10)
        self.auto_candidate = _automatic_neck_candidate(self.roi_mask, self.affine)

        slice_indices = {view: _default_slice(self.roi_mask, view) for view in VIEWS}
        if self.auto_candidate is not None:
            auto_voxels = np.asarray(self.auto_candidate["voxels_ijk"], dtype=np.float64)
            slice_indices = {
                "axial": _clip_slice(int(np.rint(auto_voxels[:, 2].mean())), self.roi_mask.shape, "axial"),
                "coronal": _clip_slice(int(np.rint(auto_voxels[:, 1].mean())), self.roi_mask.shape, "coronal"),
                "sagittal": _clip_slice(int(np.rint(auto_voxels[:, 0].mean())), self.roi_mask.shape, "sagittal"),
            }
        if args.slice is not None:
            slice_indices[str(args.view)] = _clip_slice(int(args.slice), self.roi_mask.shape, str(args.view))
        message = "Automatic neck candidate shown in cyan. Press Auto Save/a to accept, or click 3 manual points to correct."
        if self.auto_candidate is None:
            message = "No automatic neck candidate found. Click neck endpoint 1 in the best slice view."
        self.state = NeckState(view=str(args.view), slice_indices=slice_indices, message=message)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14.5, 5.8), constrained_layout=False)
        self.fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.17, wspace=0.22)
        self.status = self.fig.text(0.045, 0.055, self._status_text(), ha="left", va="bottom", fontsize=10)
        self.help_text = self.fig.text(
            0.985,
            0.055,
            "Click 2 neck endpoints, then 1 point inside the aneurysm sac. "
            "a auto save | s manual save | r reset | 1/2/3 choose scroll view | arrows change slice | q quit",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#374151",
        )
        self.auto_ax = self.fig.add_axes([0.69, 0.895, 0.10, 0.055])
        self.save_ax = self.fig.add_axes([0.80, 0.895, 0.075, 0.055])
        self.reset_ax = self.fig.add_axes([0.885, 0.895, 0.075, 0.055])
        self.auto_button = Button(self.auto_ax, "Auto Save")
        self.save_button = Button(self.save_ax, "Save")
        self.reset_button = Button(self.reset_ax, "Reset")
        self.auto_button.on_clicked(lambda _event: self.accept_automatic())
        self.save_button.on_clicked(lambda _event: self.save())
        self.reset_button.on_clicked(lambda _event: self.reset_points())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def _status_text(self) -> str:
        count = len(self.state.points_vox)
        active = self.state.active_view or "none"
        return f"{self.state.message}  Points: {count}/3  Active view: {active}"

    def _point_from_click(self, view: str, x: float, y: float) -> np.ndarray:
        s = float(self.state.slice_indices[view])
        if view == "axial":
            return np.array([x, y, s], dtype=np.float64)
        if view == "coronal":
            return np.array([x, s, y], dtype=np.float64)
        return np.array([s, x, y], dtype=np.float64)

    def _point_to_panel_xy(self, point_vox: np.ndarray, view: str) -> tuple[float, float]:
        if view == "axial":
            return float(point_vox[0]), float(point_vox[1])
        if view == "coronal":
            return float(point_vox[0]), float(point_vox[2])
        return float(point_vox[1]), float(point_vox[2])

    def _candidate_xy_for_view(self, view: str) -> tuple[np.ndarray, np.ndarray]:
        if self.auto_candidate is None:
            return np.array([]), np.array([])
        pts = np.asarray(self.auto_candidate["voxels_ijk"], dtype=np.float64)
        slice_idx = float(self.state.slice_indices[view])
        if view == "axial":
            keep = np.abs(pts[:, 2] - slice_idx) <= 0.75
            return pts[keep, 0], pts[keep, 1]
        if view == "coronal":
            keep = np.abs(pts[:, 1] - slice_idx) <= 0.75
            return pts[keep, 0], pts[keep, 2]
        keep = np.abs(pts[:, 0] - slice_idx) <= 0.75
        return pts[keep, 1], pts[keep, 2]

    def draw(self) -> None:
        for ax, view in zip(self.axes, VIEWS):
            ax.clear()
            img, mask_sl, extent, xlabel, ylabel = _view_panel(
                self.bg,
                self.roi_mask,
                view,
                int(self.state.slice_indices[view]),
                self.limits,
            )
            ax.imshow(img, cmap="gray", origin="lower", extent=extent, vmin=self.vmin, vmax=self.vmax, aspect="equal")
            _draw_contours(ax, mask_sl, int(extent[0]), int(extent[2]))
            title = f"{view} slice {self.state.slice_indices[view]}"
            if view == self.state.active_view:
                title = f"ACTIVE: {title}"
            elif view == self.state.view:
                title = f"{title}  (scroll)"
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(color="yellow", alpha=0.28, linewidth=0.7)

            cx, cy = self._candidate_xy_for_view(view)
            if cx.size:
                ax.scatter(
                    cx,
                    cy,
                    c="#06b6d4",
                    s=48,
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=4,
                    label="auto neck",
                )
                ax.legend(loc="lower right", fontsize=7, frameon=True)

            if self.state.active_view == view:
                panel_pts = [self._point_to_panel_xy(p, view) for p in self.state.points_vox]
                if panel_pts:
                    xs, ys = zip(*panel_pts)
                    colors = ["#ef4444", "#ef4444", "#22c55e"][: len(panel_pts)]
                    ax.scatter(xs, ys, c=colors, s=70, edgecolors="white", linewidths=1.2, zorder=5)
                    for idx, (px, py) in enumerate(panel_pts, start=1):
                        ax.text(px + 0.25, py + 0.25, str(idx), color="white", fontsize=9, weight="bold", zorder=6)
                if len(panel_pts) >= 2:
                    ax.plot(
                        [panel_pts[0][0], panel_pts[1][0]],
                        [panel_pts[0][1], panel_pts[1][1]],
                        color="#ef4444",
                        linewidth=2.5,
                        zorder=4,
                    )

        self.fig.suptitle("Interactive 2D aneurysm neck selector", fontsize=14)
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        view = VIEWS[list(self.axes).index(event.inaxes)]
        if self.state.active_view is None:
            self.state.active_view = view
            self.state.view = view
        elif view != self.state.active_view:
            self.state.message = f"Continue in the active {self.state.active_view} panel, or press r to reset."
            self.draw()
            return
        if len(self.state.points_vox) >= 3:
            self.state.message = "Already have 3 points. Press r to restart or s to save."
            self.draw()
            return
        self.state.points_vox.append(self._point_from_click(view, float(event.xdata), float(event.ydata)))
        if len(self.state.points_vox) == 1:
            self.state.message = "Click neck endpoint 2 in the same panel."
        elif len(self.state.points_vox) == 2:
            self.state.message = "Click one point on the aneurysm sac side."
        else:
            self.state.message = "Ready. Press Save or s to export the sac-only mask."
        self.draw()

    def on_key(self, event) -> None:
        if event.key in ("q", "escape"):
            self.plt.close(self.fig)
        elif event.key == "s":
            self.save()
        elif event.key == "a":
            self.accept_automatic()
        elif event.key == "r":
            self.reset_points()
        elif event.key == "1":
            self.set_scroll_view("axial")
        elif event.key == "2":
            self.set_scroll_view("coronal")
        elif event.key == "3":
            self.set_scroll_view("sagittal")
        elif event.key in ("left", "down"):
            self.shift_slice(-1)
        elif event.key in ("right", "up"):
            self.shift_slice(1)

    def set_scroll_view(self, view: str) -> None:
        self.state.view = view
        self.state.message = f"{view} selected for slice scrolling. Press arrows to change slice."
        self.draw()

    def shift_slice(self, delta: int) -> None:
        view = self.state.view
        self.state.slice_indices[view] = _clip_slice(self.state.slice_indices[view] + int(delta), self.roi_mask.shape, view)
        if self.state.active_view == view:
            self.state.active_view = None
            self.state.points_vox = []
            self.state.message = f"{view} slice changed. Points reset; click neck endpoint 1."
        else:
            self.state.message = f"{view} slice changed."
        self.draw()

    def reset_points(self) -> None:
        self.state.points_vox = []
        self.state.active_view = None
        self.state.message = "Click neck endpoint 1 in the best slice view."
        self.draw()

    def save(self) -> None:
        if len(self.state.points_vox) != 3 or self.state.active_view is None:
            self.state.message = "Select exactly 2 neck endpoints and 1 sac-side point before saving."
            self.draw()
            return
        try:
            sac_mask, info = _neck_side_cut(
                roi_mask=self.roi_mask,
                affine=self.affine,
                neck_a_vox=self.state.points_vox[0],
                neck_b_vox=self.state.points_vox[1],
                sac_vox=self.state.points_vox[2],
                view=self.state.active_view,
                dilate=int(self.args.dilate),
            )
        except Exception as exc:
            self.state.message = f"Cannot save: {exc}"
            self.draw()
            return
        info["slice_index"] = int(self.state.slice_indices[self.state.active_view])
        _save_sac_outputs(sac_mask, info, self.ref_img, self.out_path, self.roi_path)
        self.state.message = f"Saved: {self.out_path}"
        self.draw()

    def accept_automatic(self) -> None:
        if self.auto_candidate is None:
            self.state.message = "No automatic neck candidate available. Use manual point selection."
            self.draw()
            return
        try:
            sac_mask, info = _automatic_neck_cut(
                roi_mask=self.roi_mask,
                affine=self.affine,
                candidate=self.auto_candidate,
                dilate=int(self.args.dilate),
            )
        except Exception as exc:
            self.state.message = f"Automatic save failed: {exc}. Use manual point selection."
            self.draw()
            return
        _save_sac_outputs(sac_mask, info, self.ref_img, self.out_path, self.roi_path)
        self.state.message = f"Auto neck accepted and saved: {self.out_path}"
        self.draw()

    def show(self) -> None:
        self.plt.show()


@dataclass
class NeckState:
    view: str
    slice_indices: dict[str, int]
    active_view: str | None = None
    points_vox: list[np.ndarray] = field(default_factory=list)
    message: str = "Click neck endpoint 1 in the best slice view."


class NeckSelector:
    def __init__(
        self,
        roi_mask: np.ndarray,
        bg: np.ndarray,
        affine: np.ndarray,
        ref_img: nib.Nifti1Image,
        out_path: Path,
        args: argparse.Namespace,
    ) -> None:
        self.roi_mask = roi_mask.astype(bool)
        self.bg = bg.astype(np.float32)
        self.affine = affine
        self.ref_img = ref_img
        self.out_path = out_path
        self.args = args
        self.vmin, self.vmax = _window_image(self.bg, self.roi_mask)
        self.grid = _build_grid(self.bg, self.roi_mask, self.affine)
        self.surface = self.grid.contour(isosurfaces=[0.5], scalars="roi")
        if self.surface.n_points == 0:
            raise ValueError("ROI surface is empty.")

        slice_indices = {view: _default_slice(self.roi_mask, view) for view in VIEWS}
        if args.slice is not None:
            slice_indices[str(args.view)] = _clip_slice(int(args.slice), self.roi_mask.shape, str(args.view))
        self.state = NeckState(view=str(args.view), slice_indices=slice_indices)

        self.plotter = pv.Plotter(
            shape=(1, 3),
            window_size=(int(args.window_width), int(args.window_height)),
            notebook=False,
            off_screen=False,
        )
        self.plotter.set_background("white")
        self.actors: dict[str, Any] = {}

    def _origin_world(self, view: str) -> np.ndarray:
        shape = np.asarray(self.roi_mask.shape, dtype=int)
        ijk = np.array([shape[0] / 2, shape[1] / 2, shape[2] / 2], dtype=np.float64)
        ijk[_slice_axis(view)] = float(self.state.slice_indices[view])
        return _voxel_to_world(ijk[None, :], self.affine)[0]

    def _slice_dataset(self, view: str) -> pv.DataSet:
        normal = {"sagittal": "x", "coronal": "y", "axial": "z"}[view]
        slc = self.grid.slice(normal=normal, origin=self._origin_world(view))
        masked = slc.threshold([0.5, 1.0], scalars="roi")
        return masked if masked.n_points > 0 else slc

    def _remove_actor(self, key: str) -> None:
        actor = self.actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor, reset_camera=False)

    def _point_vox_to_world(self, point_vox: np.ndarray) -> np.ndarray:
        return _voxel_to_world(np.asarray(point_vox, dtype=np.float64)[None, :], self.affine)[0]

    def _point_on_view_slice(self, point_vox: np.ndarray, view: str) -> bool:
        return abs(float(point_vox[_slice_axis(view)]) - float(self.state.slice_indices[view])) <= 0.75

    def draw(self, reset_camera: bool = False) -> None:
        self.plotter.clear()
        for idx, view in enumerate(("axial", "coronal", "sagittal")):
            self.plotter.subplot(0, idx)
            slc = self._slice_dataset(view)
            self.plotter.add_mesh(
                slc,
                scalars="mag",
                cmap="gray",
                clim=(self.vmin, self.vmax),
                show_scalar_bar=False,
                opacity=1.0,
            )
            self.plotter.add_mesh(
                self.surface,
                color="#f97316",
                opacity=0.18,
                smooth_shading=True,
                pickable=False,
            )

            visible_points = [p for p in self.state.points_vox if self._point_on_view_slice(p, view)]
            if visible_points:
                pts_world = np.vstack([self._point_vox_to_world(p) for p in visible_points])
                self.plotter.add_mesh(
                    pv.PolyData(pts_world),
                    color="#ef4444",
                    point_size=18,
                    render_points_as_spheres=True,
                    pickable=False,
                )
            if len(visible_points) >= 2:
                p0 = self._point_vox_to_world(visible_points[0])
                p1 = self._point_vox_to_world(visible_points[1])
                self.plotter.add_mesh(pv.Line(p0, p1), color="#ef4444", line_width=5, pickable=False)
            if len(visible_points) >= 3:
                sac_world = self._point_vox_to_world(visible_points[2])
                self.plotter.add_mesh(pv.Sphere(center=sac_world, radius=0.45), color="#22c55e", opacity=0.95, pickable=False)

            active = "ACTIVE " if self.state.active_view == view else ""
            self.plotter.add_text(
                f"{active}{view} slice {self.state.slice_indices[view]}",
                name=f"title_{view}",
                position="upper_edge",
                font_size=10,
                color="black",
            )
            self.plotter.add_axes()
            if view == "sagittal":
                self.plotter.view_yz()
            elif view == "coronal":
                self.plotter.view_xz()
            else:
                self.plotter.view_xy()
            try:
                self.plotter.enable_parallel_projection()
            except Exception:
                pass
            if reset_camera:
                self.plotter.reset_camera()

        self.plotter.subplot(0, 0)
        self.plotter.add_text(
            "\n".join(
                [
                    self.state.message,
                    "First click chooses active view. Keep all 3 clicks in that view.",
                    "s save | r reset | 1/2/3 choose slice keys | arrows move active slice | q quit",
                ]
            ),
            name="hud",
            position="upper_left",
            font_size=9,
            color="black",
        )
        self.plotter.render()

    def on_pick(self, point) -> None:
        if len(self.state.points_vox) >= 3:
            self.state.message = "Already have 3 points. Press r to pick again or s to save."
            self.draw()
            return
        pt = np.asarray(point, dtype=np.float64)
        vox = _world_to_voxel(pt[None, :], self.affine)[0]
        distances = {
            view: abs(float(vox[_slice_axis(view)]) - float(self.state.slice_indices[view]))
            for view in VIEWS
        }
        clicked_view = min(distances, key=distances.get)
        if self.state.active_view is None:
            self.state.active_view = clicked_view
            self.state.view = clicked_view
        elif clicked_view != self.state.active_view:
            self.state.message = f"Use the active {self.state.active_view} view, or press r to reset."
            self.draw()
            return
        vox[_slice_axis(self.state.active_view)] = float(self.state.slice_indices[self.state.active_view])
        self.state.points_vox.append(vox)
        if len(self.state.points_vox) == 1:
            self.state.message = "Click neck endpoint 2."
        elif len(self.state.points_vox) == 2:
            self.state.message = "Click one point on the aneurysm sac side of the neck line."
        else:
            self.state.message = "Ready. Press s or Save to export sac-only mask."
        self.draw()

    def set_view(self, view: str) -> None:
        self.state.view = view
        self.state.active_view = None
        self.state.points_vox = []
        self.state.message = f"{view} selected for slice scrolling. Click neck endpoint 1."
        self.draw(reset_camera=True)

    def shift_slice(self, delta: int) -> None:
        self.state.slice_indices[self.state.view] = _clip_slice(
            self.state.slice_indices[self.state.view] + int(delta), self.roi_mask.shape, self.state.view
        )
        self.state.active_view = None
        self.state.points_vox = []
        self.state.message = f"{self.state.view} slice changed. Points reset."
        self.draw(reset_camera=True)

    def reset_points(self) -> None:
        self.state.points_vox = []
        self.state.active_view = None
        self.state.message = "Click neck endpoint 1 in the best slice view."
        self.draw()

    def _make_sac_mask(self) -> tuple[np.ndarray, dict[str, Any]]:
        if len(self.state.points_vox) != 3:
            raise ValueError("Select exactly two neck endpoints and one sac-side point before saving.")
        if self.state.active_view is None:
            raise ValueError("No active view. Select points first.")
        neck_a_vox, neck_b_vox, sac_vox = [np.asarray(p, dtype=np.float64) for p in self.state.points_vox]
        sac_mask, info = _neck_side_cut(
            roi_mask=self.roi_mask,
            affine=self.affine,
            neck_a_vox=neck_a_vox,
            neck_b_vox=neck_b_vox,
            sac_vox=sac_vox,
            view=self.state.active_view,
            dilate=int(self.args.dilate),
        )
        info = {
            "source_roi_path": str(Path(self.args.roi).expanduser().resolve()),
            "out_path": str(self.out_path),
            "slice_index": int(self.state.slice_indices[self.state.active_view]),
            **info,
        }
        return sac_mask.astype(np.uint8), info

    def save(self) -> None:
        try:
            sac_mask, info = self._make_sac_mask()
        except Exception as exc:
            self.state.message = f"Cannot save: {exc}"
            self.draw()
            return

        _save_sac_outputs(
            sac_mask=sac_mask,
            info=info,
            ref_img=self.ref_img,
            out_path=self.out_path,
            source_roi_path=Path(self.args.roi).expanduser().resolve(),
        )
        self.state.message = f"Saved sac mask: {self.out_path.name}"
        self.draw()

    def show(self) -> None:
        self.draw(reset_camera=True)
        self.plotter.enable_point_picking(
            callback=self.on_pick,
            show_message=False,
            left_clicking=True,
            use_picker=False,
            show_point=False,
            color="black",
        )
        self.plotter.add_key_event("s", self.save)
        self.plotter.add_key_event("r", self.reset_points)
        self.plotter.add_key_event("1", lambda: self.set_view("axial"))
        self.plotter.add_key_event("2", lambda: self.set_view("coronal"))
        self.plotter.add_key_event("3", lambda: self.set_view("sagittal"))
        self.plotter.add_key_event("Left", lambda: self.shift_slice(-1))
        self.plotter.add_key_event("Right", lambda: self.shift_slice(1))
        self.plotter.add_key_event("Up", lambda: self.shift_slice(1))
        self.plotter.add_key_event("Down", lambda: self.shift_slice(-1))
        self.plotter.add_key_event("q", lambda: self.plotter.iren.terminate_app())
        self.plotter.add_key_event("Escape", lambda: self.plotter.iren.terminate_app())
        self.plotter.show()


def main() -> None:
    args = _parse_args()
    roi_path = Path(args.roi).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    roi_raw, affine, ref_img = _load_nifti(roi_path)
    bg, bg_affine, _ = _load_nifti(bg_path)
    if roi_raw.shape != bg.shape:
        tmp = np.zeros(roi_raw.shape, dtype=np.float32)
        sx = min(roi_raw.shape[0], bg.shape[0])
        sy = min(roi_raw.shape[1], bg.shape[1])
        sz = min(roi_raw.shape[2], bg.shape[2])
        tmp[:sx, :sy, :sz] = bg[:sx, :sy, :sz]
        bg = tmp
    if not np.allclose(affine, bg_affine, atol=1e-3):
        print("Warning: ROI and background affines differ. Overlay assumes they are registered.")

    roi_mask = _largest_component(roi_raw >= float(args.threshold))
    if str(args.mode) == "auto":
        _auto_mode(
            roi_mask=roi_mask,
            affine=affine,
            ref_img=ref_img,
            out_path=out_path,
            roi_path=roi_path,
            args=args,
        )
    elif str(args.mode) == "qt":
        selector = QtNeckSelector(
            roi_mask=roi_mask,
            bg=bg,
            affine=affine,
            ref_img=ref_img,
            out_path=out_path,
            roi_path=roi_path,
            args=args,
        )
        selector.show()
    elif str(args.mode) == "manual":
        _manual_mode(
            roi_mask=roi_mask,
            bg=bg,
            affine=affine,
            ref_img=ref_img,
            out_path=out_path,
            roi_path=roi_path,
            args=args,
        )
    else:
        if pv is None:
            raise RuntimeError("PyVista is not available. Use --mode manual.")
        selector = NeckSelector(roi_mask=roi_mask, bg=bg, affine=affine, ref_img=ref_img, out_path=out_path, args=args)
        selector.show()


if __name__ == "__main__":
    main()
