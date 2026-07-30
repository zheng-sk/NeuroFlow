#!/usr/bin/env python3
"""Compute morphological shape metrics for a segmented vascular zone (CoW / aneurysm).

Workflow
--------
1. Load binary NIfTI segmentation.
2. Extract the largest connected component (drops isolated noise voxels).
3. Run marching cubes → write VTK legacy surface PolyData (.vtk).
4. Write a VTK UnstructuredGrid volume mesh (voxels as hexahedral cells, .vtk text).
5. Compute morphological metrics and save to CSV / JSON.

Metrics
-------
- volume_mm3               : voxel count × voxel volume
- surface_area_mm2         : marching-cubes mesh surface area
- bbox_width_mm/height_mm/depth_mm : bounding box extents
- axis_major/minor/intermediate_mm : principal axes from inertia tensor
- elongation               : major / minor axis
- flatness                 : intermediate / minor axis
- equivalent_sphere_diam_mm: diameter of sphere with same volume
- sphericity               : ideal-sphere SA / actual SA  (1 = perfect sphere)
- non_sphericity_index     : 1 - (18π)^(1/3)·V^(2/3) / SA
- undulation_index         : 1 - V / V_convex_hull
- neck_width_mm            : narrowest cross-sectional diameter along the major axis
- max_cross_section_diam_mm: widest cross-sectional diameter along the major axis
- aspect_ratio_approx     : axis_major_mm / neck_width_mm
- bottleneck_factor_approx: max_cross_section_diam_mm / neck_width_mm
- num_connected_components : total blobs before filtering (noise indicator)
- centroid_x/y/z_mm        : physical centroid in mm from image origin
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import label, marching_cubes, mesh_surface_area, regionprops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape metrics for a binary vascular segmentation.")
    p.add_argument("--seg", required=True, help="Path to binary NIfTI segmentation (e.g. cow_seg_final.nii.gz)")
    p.add_argument("--out-dir", default="", help="Output directory. Default: <seg_parent>/shape_metrics")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarisation threshold (default 0.5)")
    p.add_argument(
        "--convex-hull-subsample",
        type=int,
        default=10,
        help="Subsample stride for foreground voxel corners when computing convex hull (speeds up large meshes, default 10)",
    )
    p.add_argument(
        "--neck-json",
        default="",
        help="Optional manual neck JSON from select_aneurysm_neck.py. Adds manual neck/height ratios.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# NIfTI loading
# ---------------------------------------------------------------------------

def _load_binary_mask(path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (binary_mask, voxel_axis_lengths_mm, affine)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    mask = (data >= threshold).astype(np.uint8)
    affine = np.array(img.affine, dtype=float)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    return mask, zooms, np.array(img.affine, dtype=float)


def _voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map Nx3 voxel coordinates to world/mm coordinates using the NIfTI affine."""
    points = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return (np.concatenate([points, ones], axis=1) @ affine.T)[:, :3]


def _foreground_voxel_corners_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return world-space corners of occupied voxels for convex-hull volume."""
    coords = np.argwhere(mask > 0).astype(np.float64)
    offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    corners = np.unique(corners, axis=0)
    return _voxel_to_world(corners, affine)


# ---------------------------------------------------------------------------
# VTK surface export (legacy ASCII PolyData, no VTK library required)
# ---------------------------------------------------------------------------

def _write_vtk_surface(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write triangulated surface as VTK legacy ASCII PolyData."""
    n_verts = len(verts)
    n_faces = len(faces)
    lines = [
        "# vtk DataFile Version 3.0",
        "Segmentation surface",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n_verts} float",
    ]
    for v in verts:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    lines.append(f"POLYGONS {n_faces} {n_faces * 4}")
    for f in faces:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# VTK volume mesh export (legacy ASCII UnstructuredGrid, hex voxels)
# ---------------------------------------------------------------------------

def _write_vtk_volume_mesh(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write occupied voxels as VTK legacy ASCII UnstructuredGrid (hexahedral cells)."""
    coords = np.argwhere(mask > 0).astype(float)  # (N, 3) in voxel index space
    n_cells = len(coords)
    # VTK_VOXEL (type 11) expects 8 corner points in this order.
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)

    all_points = []
    for c in coords:
        for off in offsets:
            all_points.append(c + off)
    all_points_arr = np.array(all_points)  # (N*8, 3)
    all_points_arr = _voxel_to_world(all_points_arr, affine)
    n_points = len(all_points_arr)

    lines = [
        "# vtk DataFile Version 3.0",
        "Segmentation volume mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_points} float",
    ]
    for p in all_points_arr:
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    lines.append(f"CELLS {n_cells} {n_cells * 9}")
    for i in range(n_cells):
        base = i * 8
        lines.append(f"8 {base} {base+1} {base+2} {base+3} {base+4} {base+5} {base+6} {base+7}")

    lines.append(f"CELL_TYPES {n_cells}")
    for _ in range(n_cells):
        lines.append("11")  # VTK_VOXEL

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# Shape metrics
# ---------------------------------------------------------------------------

def _compute_shape_metrics(
    mask: np.ndarray,
    zooms: np.ndarray,
    affine: np.ndarray,
    convex_hull_subsample: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    voxel_vol_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    # --- connected components ---
    labeled = label(mask)
    props_all = regionprops(labeled)
    num_cc = len(props_all)

    # use largest component
    main = max(props_all, key=lambda p: p.area)
    main_mask = (labeled == main.label).astype(np.uint8)

    # --- volume ---
    volume_mm3 = float(main.area) * voxel_vol_mm3

    # --- surface area via marching cubes ---
    verts_vox, faces, normals, _ = marching_cubes(main_mask.astype(float), level=0.5)
    verts = _voxel_to_world(verts_vox, affine)
    surface_area_mm2 = float(mesh_surface_area(verts, faces))

    # --- bounding box ---
    minr, minc, minz, maxr, maxc, maxz = main.bbox
    bbox_width_mm  = float((maxr - minr) * zooms[0])
    bbox_height_mm = float((maxc - minc) * zooms[1])
    bbox_depth_mm  = float((maxz - minz) * zooms[2])

    # --- principal axes from inertia tensor (physical lengths) ---
    # regionprops inertia_tensor_eigvals are in voxel² units; convert carefully
    # axis_major/minor_length are in voxels (assume isotropic); we scale to mm
    axis_major_mm        = float(main.axis_major_length) * float(zooms[0])
    axis_minor_mm        = float(main.axis_minor_length) * float(zooms[0])
    # intermediate axis: derive from inertia tensor eigenvalues
    eigs = np.sort(main.inertia_tensor_eigvals)  # ascending
    # eigenvalue of inertia tensor ≈ N·r² / 5 for each axis; larger eigenvalue = smaller axis
    # intermediate (middle eigenvalue) → estimated from major/minor by interpolation
    if eigs[1] > 0:
        axis_intermediate_mm = float(axis_minor_mm * math.sqrt(eigs[2] / eigs[1]))
    else:
        axis_intermediate_mm = axis_minor_mm

    elongation = axis_major_mm / axis_minor_mm if axis_minor_mm > 0 else float("nan")
    flatness   = axis_intermediate_mm / axis_minor_mm if axis_minor_mm > 0 else float("nan")

    # --- equivalent sphere diameter ---
    r_eq = (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    equiv_sphere_diam_mm = 2.0 * r_eq

    # --- sphericity: ratio of surface area of equiv sphere to actual SA ---
    sphere_sa = 4.0 * math.pi * r_eq ** 2
    sphericity = sphere_sa / surface_area_mm2 if surface_area_mm2 > 0 else float("nan")

    # --- Non-Sphericity Index (NSI) ---
    nsi = 1.0 - (18.0 * math.pi) ** (1.0 / 3.0) * volume_mm3 ** (2.0 / 3.0) / surface_area_mm2

    # --- Undulation Index (UI): 1 - V / V_convex ---
    # Use foreground voxel corners, not sparse surface vertices. This makes the
    # convex hull contain the voxelized sac volume used for volume_mm3.
    ui = float("nan")
    try:
        hull_points = _foreground_voxel_corners_world(main_mask, affine)
        stride = max(1, int(convex_hull_subsample))
        if len(hull_points) <= 50000:
            sub = hull_points
        else:
            sub = hull_points[::stride]
        if len(sub) >= 4:
            hull = ConvexHull(sub)
            vol_convex = hull.volume
            ui = 1.0 - volume_mm3 / vol_convex if vol_convex > 0 else float("nan")
    except Exception:
        pass

    # --- neck width and max cross-section diameter ---
    # Slice the mask perpendicular to the major axis at evenly-spaced positions.
    # For each slice, compute the equivalent circular diameter from the voxel count.
    # neck_width_mm          = diameter of the narrowest slice (minimum cross-section)
    # max_cross_section_diam_mm = diameter of the widest slice
    neck_width_mm: float = float("nan")
    max_cross_section_diam_mm: float = float("nan")
    auto_neck: dict[str, Any] = {}
    try:
        coords_ijk = np.argwhere(main_mask).astype(float)
        coords_mm = _voxel_to_world(coords_ijk, affine)
        coords_centered = coords_mm - coords_mm.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
        major_dir_world = vh[0]
        projections = coords_centered @ major_dir_world

        step = float(zooms.min())
        voxel_cross_area = step ** 2  # footprint of one voxel in the perpendicular plane
        bins = np.arange(projections.min(), projections.max() + step, step)

        slice_infos: list[dict[str, Any]] = []
        total_voxels = int(coords_ijk.shape[0])
        min_side_count = max(20, int(round(0.03 * total_voxels)))
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_slice = (projections >= lo) & (projections < hi)
            n = int(in_slice.sum())
            if n < 4:   # skip fringe slices with only 1-3 voxels
                continue
            below = int((projections < lo).sum())
            above = int((projections >= hi).sum())
            if min(below, above) < min_side_count:
                continue
            diameter = 2.0 * math.sqrt(n * voxel_cross_area / math.pi)
            slice_infos.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": n,
                    "below_count": below,
                    "above_count": above,
                    "diameter": float(diameter),
                    "indices": np.flatnonzero(in_slice),
                }
            )

        if slice_infos:
            neck_info = min(slice_infos, key=lambda item: item["diameter"])
            max_info = max(slice_infos, key=lambda item: item["diameter"])
            neck_width_mm = float(neck_info["diameter"])
            max_cross_section_diam_mm = float(max_info["diameter"])
            neck_indices = np.asarray(neck_info["indices"], dtype=int)
            neck_voxels = coords_ijk[neck_indices].astype(int)
            neck_world = coords_mm[neck_indices]
            neck_center = neck_world.mean(axis=0)
            auto_neck = {
                "automatic_neck_method": "narrowest cross-section perpendicular to longest PCA axis",
                "automatic_neck_voxel_count": int(neck_info["n"]),
                "automatic_neck_side_counts": [int(neck_info["below_count"]), int(neck_info["above_count"])],
                "automatic_neck_area_mm2": round(float(neck_info["n"] * voxel_cross_area), 6),
                "automatic_neck_equivalent_diameter_mm": round(neck_width_mm, 6),
                "automatic_neck_center_world_mm": [round(float(v), 6) for v in neck_center],
                "automatic_neck_axis_world": [round(float(v), 8) for v in major_dir_world],
                "automatic_neck_projection_range_mm": [round(float(neck_info["lo"]), 6), round(float(neck_info["hi"]), 6)],
                "automatic_neck_voxels_ijk": neck_voxels.tolist(),
            }
    except Exception:
        pass

    aspect_ratio_approx = (
        axis_major_mm / neck_width_mm
        if math.isfinite(neck_width_mm) and neck_width_mm > 0
        else float("nan")
    )
    bottleneck_factor_approx = (
        max_cross_section_diam_mm / neck_width_mm
        if math.isfinite(neck_width_mm) and neck_width_mm > 0 and math.isfinite(max_cross_section_diam_mm)
        else float("nan")
    )

    # --- centroid in physical mm (using affine) ---
    centroid_phys = _voxel_to_world(np.array(main.centroid, dtype=float)[None, :], affine)[0]
    centroid_x_mm, centroid_y_mm, centroid_z_mm = float(centroid_phys[0]), float(centroid_phys[1]), float(centroid_phys[2])

    return {
        "volume_mm3": round(volume_mm3, 4),
        "surface_area_mm2": round(surface_area_mm2, 4),
        "bbox_width_mm": round(bbox_width_mm, 4),
        "bbox_height_mm": round(bbox_height_mm, 4),
        "bbox_depth_mm": round(bbox_depth_mm, 4),
        "axis_major_mm": round(axis_major_mm, 4),
        "axis_minor_mm": round(axis_minor_mm, 4),
        "axis_intermediate_mm": round(axis_intermediate_mm, 4),
        "elongation": round(elongation, 6) if math.isfinite(elongation) else None,
        "flatness": round(flatness, 6) if math.isfinite(flatness) else None,
        "equivalent_sphere_diam_mm": round(equiv_sphere_diam_mm, 4),
        "sphericity": round(sphericity, 6) if math.isfinite(sphericity) else None,
        "non_sphericity_index": round(nsi, 6),
        "undulation_index": round(ui, 6) if math.isfinite(ui) else None,
        "neck_width_mm": round(neck_width_mm, 4) if math.isfinite(neck_width_mm) else None,
        "max_cross_section_diam_mm": round(max_cross_section_diam_mm, 4) if math.isfinite(max_cross_section_diam_mm) else None,
        "aspect_ratio_approx": round(aspect_ratio_approx, 6) if math.isfinite(aspect_ratio_approx) else None,
        "bottleneck_factor_approx": round(bottleneck_factor_approx, 6) if math.isfinite(bottleneck_factor_approx) else None,
        "num_connected_components": num_cc,
        "centroid_x_mm": round(centroid_x_mm, 4),
        "centroid_y_mm": round(centroid_y_mm, 4),
        "centroid_z_mm": round(centroid_z_mm, 4),
        **auto_neck,
    }, verts, faces, main_mask


def _add_manual_neck_metrics(metrics: dict[str, Any], main_mask: np.ndarray, affine: np.ndarray, neck_json: Path) -> dict[str, Any]:
    neck = json.loads(neck_json.read_text(encoding="utf-8"))
    neck_width = neck.get("manual_neck_width_mm")
    plane_point = neck.get("neck_plane_point_world_mm")
    plane_normal = neck.get("neck_plane_normal_world")

    if neck_width is None or plane_point is None or plane_normal is None:
        raise ValueError(f"Neck JSON is missing required manual neck fields: {neck_json}")

    neck_width = float(neck_width)
    p0 = np.asarray(plane_point, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)

    coords_world = _voxel_to_world(np.argwhere(main_mask > 0).astype(float), affine)
    signed = (coords_world - p0[None, :]) @ n
    positive = signed[signed >= 0]
    if positive.size == 0:
        height = float(np.max(np.abs(signed))) if signed.size else float("nan")
    else:
        height = float(np.max(positive))

    max_cross = metrics.get("max_cross_section_diam_mm")
    axis_major = metrics.get("axis_major_mm")
    metrics["manual_neck_width_mm"] = round(neck_width, 4)
    metrics["height_from_neck_plane_mm"] = round(height, 4) if math.isfinite(height) else None
    metrics["aspect_ratio_manual"] = round(height / neck_width, 6) if neck_width > 0 and math.isfinite(height) else None
    if max_cross is not None and neck_width > 0:
        metrics["bottleneck_factor_manual"] = round(float(max_cross) / neck_width, 6)
    elif axis_major is not None and neck_width > 0:
        metrics["bottleneck_factor_manual"] = round(float(axis_major) / neck_width, 6)
    else:
        metrics["bottleneck_factor_manual"] = None
    metrics["manual_neck_json"] = str(neck_json)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    seg_path = Path(args.seg).expanduser().resolve()
    if not seg_path.is_file():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else seg_path.parent / "shape_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask, zooms, affine = _load_binary_mask(seg_path, args.threshold)

    if mask.sum() == 0:
        raise ValueError("Segmentation mask is empty after thresholding.")

    metrics, verts, faces, main_mask = _compute_shape_metrics(mask, zooms, affine, args.convex_hull_subsample)
    if str(args.neck_json).strip():
        neck_json = Path(args.neck_json).expanduser().resolve()
        if not neck_json.is_file():
            raise FileNotFoundError(f"Manual neck JSON not found: {neck_json}")
        metrics = _add_manual_neck_metrics(metrics, main_mask, affine, neck_json)

    # write VTK surface
    vtk_surface_path = out_dir / "surface.vtk"
    _write_vtk_surface(verts, faces, vtk_surface_path)
    print(f"VTK surface written: {vtk_surface_path}")

    # write VTK volume mesh
    vtk_volume_path = out_dir / "volume_mesh.vtk"
    _write_vtk_volume_mesh(main_mask, affine, vtk_volume_path)
    print(f"VTK volume mesh written: {vtk_volume_path}")

    # save metrics CSV
    csv_path = out_dir / "shape_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Shape metrics CSV written: {csv_path}")

    # save metrics JSON
    json_path = out_dir / "shape_metrics.json"
    payload = {"seg_path": str(seg_path), **metrics}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Shape metrics JSON written: {json_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
