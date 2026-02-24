import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "semibold",
        "axes.grid": False,
        "font.size": 10,
    }
)

try:
    from scipy import stats
    from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, distance_transform_edt, gaussian_filter, label as ndi_label
    from scipy.spatial import cKDTree
except Exception as exc:
    raise RuntimeError(
        "This script requires scipy for statistical tests, interpolation helpers, and surface metrics."
    ) from exc

try:
    from skimage.measure import marching_cubes
    from skimage.morphology import skeletonize
    try:
        from skimage.morphology import skeletonize_3d
    except Exception:  # pragma: no cover
        skeletonize_3d = None
except Exception as exc:
    raise RuntimeError("This script requires scikit-image for surface extraction metrics.") from exc


def _load_payload(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _safe_skew(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    return float(stats.skew(x, bias=False))


def _safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return float("nan")
    return float(stats.kurtosis(x, fisher=True, bias=False))


def _relative_error(val: float, ref: float) -> float:
    denom = abs(float(ref))
    if denom < 1e-12:
        return float("nan")
    return abs(float(val) - float(ref)) / denom


def _wilcoxon_p(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 5:
        return float("nan")
    try:
        return float(stats.wilcoxon(x_arr, y_arr, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _extract_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def _vorticity_magnitude(uvw: np.ndarray, spacing_m: Tuple[float, float, float]) -> np.ndarray:
    # uvw shape: [3, X, Y, Z]
    u, v, w = uvw[0], uvw[1], uvw[2]
    du_dx, du_dy, du_dz = np.gradient(u, *spacing_m, edge_order=1)
    dv_dx, dv_dy, dv_dz = np.gradient(v, *spacing_m, edge_order=1)
    dw_dx, dw_dy, dw_dz = np.gradient(w, *spacing_m, edge_order=1)
    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy
    return np.sqrt(wx**2 + wy**2 + wz**2).astype(np.float32)


def _default_slice_triplet(valid_slices: List[int]) -> Tuple[List[int], List[str]]:
    if not valid_slices:
        return [], []
    arr = np.asarray(sorted(valid_slices), dtype=int)
    picks = [arr[int(round((len(arr) - 1) * q))] for q in (0.2, 0.5, 0.8)]
    labels = [f"slice_{int(s)}" for s in picks]
    return [int(x) for x in picks], labels


def _table2_rows(
    speed_ref: np.ndarray,
    speed_base: np.ndarray,
    speed_sr: np.ndarray,
    vort_ref: np.ndarray,
    vort_base: np.ndarray,
    vort_sr: np.ndarray,
    mask_ref: np.ndarray,
    flow_axis: int,
    min_voxels: int,
) -> Tuple[List[Dict[str, Any]], List[int], List[float], List[float]]:
    # Inputs are 4D: [T, X, Y, Z]. We aggregate slice statistics across all frames.
    if speed_ref.ndim != 4:
        raise ValueError(f"Expected speed_ref with shape [T,X,Y,Z], got {speed_ref.shape}")

    t_count = speed_ref.shape[0]
    slice_count = speed_ref.shape[flow_axis + 1]
    rows: List[Dict[str, Any]] = []
    valid_slices: List[int] = []
    re_base_all: List[float] = []
    re_sr_all: List[float] = []

    for s in range(slice_count):
        sv_ref_parts: List[np.ndarray] = []
        sv_base_parts: List[np.ndarray] = []
        sv_sr_parts: List[np.ndarray] = []
        vo_ref_parts: List[np.ndarray] = []
        vo_base_parts: List[np.ndarray] = []
        vo_sr_parts: List[np.ndarray] = []

        total_vox = 0
        for t in range(t_count):
            mask_sl = _extract_slice(mask_ref[t], flow_axis, s) > 0.5
            if int(mask_sl.sum()) == 0:
                continue

            sv_ref_parts.append(_extract_slice(speed_ref[t], flow_axis, s)[mask_sl])
            sv_base_parts.append(_extract_slice(speed_base[t], flow_axis, s)[mask_sl])
            sv_sr_parts.append(_extract_slice(speed_sr[t], flow_axis, s)[mask_sl])

            vo_ref_parts.append(_extract_slice(vort_ref[t], flow_axis, s)[mask_sl])
            vo_base_parts.append(_extract_slice(vort_base[t], flow_axis, s)[mask_sl])
            vo_sr_parts.append(_extract_slice(vort_sr[t], flow_axis, s)[mask_sl])
            total_vox += int(mask_sl.sum())

        if total_vox < int(min_voxels):
            continue

        valid_slices.append(s)

        sv_ref = np.concatenate(sv_ref_parts, axis=0)
        sv_base = np.concatenate(sv_base_parts, axis=0)
        sv_sr = np.concatenate(sv_sr_parts, axis=0)
        vo_ref = np.concatenate(vo_ref_parts, axis=0)
        vo_base = np.concatenate(vo_base_parts, axis=0)
        vo_sr = np.concatenate(vo_sr_parts, axis=0)

        metric_defs = [
            ("Mean velocity [m/s]", lambda a: float(np.mean(a)), sv_ref, sv_base, sv_sr),
            ("SD velocity [m/s]", lambda a: float(np.std(a, ddof=1)) if a.size > 1 else float("nan"), sv_ref, sv_base, sv_sr),
            ("Skewness velocity", _safe_skew, sv_ref, sv_base, sv_sr),
            ("Kurtosis velocity", _safe_kurtosis, sv_ref, sv_base, sv_sr),
            ("Mean vorticity [1/s]", lambda a: float(np.mean(a)), vo_ref, vo_base, vo_sr),
            ("SD vorticity [1/s]", lambda a: float(np.std(a, ddof=1)) if a.size > 1 else float("nan"), vo_ref, vo_base, vo_sr),
            ("Skewness vorticity", _safe_skew, vo_ref, vo_base, vo_sr),
            ("Kurtosis vorticity", _safe_kurtosis, vo_ref, vo_base, vo_sr),
        ]

        for var_name, fn, arr_ref, arr_base, arr_sr in metric_defs:
            ref_val = fn(arr_ref)
            base_val = fn(arr_base)
            sr_val = fn(arr_sr)
            re_base = _relative_error(base_val, ref_val)
            re_sr = _relative_error(sr_val, ref_val)
            rows.append(
                {
                    "slice_index": int(s),
                    "variable": var_name,
                    "ref": ref_val,
                    "baseline": base_val,
                    "sr": sr_val,
                    "re_baseline": re_base,
                    "re_sr": re_sr,
                }
            )
            re_base_all.append(re_base)
            re_sr_all.append(re_sr)

    return rows, valid_slices, re_base_all, re_sr_all


def _table2_rows_per_frame(
    speed_ref: np.ndarray,
    speed_base: np.ndarray,
    speed_sr: np.ndarray,
    vort_ref: np.ndarray,
    vort_base: np.ndarray,
    vort_sr: np.ndarray,
    mask_ref: np.ndarray,
    flow_axis: int,
    min_voxels: int,
    frame_source_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    if speed_ref.ndim != 4:
        raise ValueError(f"Expected speed_ref with shape [T,X,Y,Z], got {speed_ref.shape}")

    t_count = speed_ref.shape[0]
    if frame_source_indices.shape[0] != t_count:
        frame_source_indices = np.arange(t_count, dtype=np.int32)

    out_rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        t_rows, _, _, _ = _table2_rows(
            speed_ref=speed_ref[t : t + 1],
            speed_base=speed_base[t : t + 1],
            speed_sr=speed_sr[t : t + 1],
            vort_ref=vort_ref[t : t + 1],
            vort_base=vort_base[t : t + 1],
            vort_sr=vort_sr[t : t + 1],
            mask_ref=mask_ref[t : t + 1],
            flow_axis=flow_axis,
            min_voxels=min_voxels,
        )
        for row in t_rows:
            rr = dict(row)
            rr["frame_payload_index"] = int(t)
            rr["frame_source_index"] = int(frame_source_indices[t])
            out_rows.append(rr)

    return out_rows


def _flow_rate_curves(
    vel: np.ndarray,
    mask: np.ndarray,
    flow_axis: int,
    spacing_mm: Tuple[float, float, float],
) -> np.ndarray:
    # vel: [T, 3, X, Y, Z], mask: [T, X, Y, Z]
    t_count = vel.shape[0]
    n_slices = vel.shape[2 + flow_axis]

    axes = [0, 1, 2]
    axes.remove(flow_axis)
    area_mm2 = float(spacing_mm[axes[0]] * spacing_mm[axes[1]])

    q = np.zeros((t_count, n_slices), dtype=np.float32)
    for t in range(t_count):
        comp = vel[t, flow_axis]
        m = mask[t] > 0.5
        for s in range(n_slices):
            plane_v = _extract_slice(comp, flow_axis, s)
            plane_m = _extract_slice(m, flow_axis, s)
            q[t, s] = float(np.sum(plane_v[plane_m]) * area_mm2)
    return q


_N26_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)]


def _pick_centerline_mask(mask_txyz: np.ndarray, mode: str, frame_index: int) -> np.ndarray:
    if mode == "intersection":
        return np.all(mask_txyz > 0.5, axis=0).astype(np.uint8)
    if mode == "frame":
        f = int(np.clip(frame_index, 0, mask_txyz.shape[0] - 1))
        return (mask_txyz[f] > 0.5).astype(np.uint8)
    return np.any(mask_txyz > 0.5, axis=0).astype(np.uint8)


def _clean_centerline_mask(mask_xyz: np.ndarray, keep_components: int, closing_iters: int) -> np.ndarray:
    m = mask_xyz > 0.5
    if int(m.sum()) == 0:
        raise ValueError("Centerline mask is empty.")

    m = binary_fill_holes(m)
    iters = max(0, int(closing_iters))
    if iters > 0:
        m = binary_closing(m, structure=np.ones((3, 3, 3), dtype=bool), iterations=iters)
        m = binary_fill_holes(m)

    labels, n_comp = ndi_label(m)
    if n_comp <= 0:
        raise ValueError("Mask connectivity failed: no connected components found.")

    k = max(1, int(keep_components))
    if n_comp > k:
        sizes = np.asarray([int(np.count_nonzero(labels == i)) for i in range(1, n_comp + 1)], dtype=np.int64)
        keep = np.argsort(sizes)[::-1][:k] + 1
        m = np.isin(labels, keep)
        m = binary_fill_holes(m)
    return m.astype(np.uint8)


def _graph_components(adj: List[List[int]]) -> List[np.ndarray]:
    n = len(adj)
    seen = np.zeros((n,), dtype=bool)
    comps: List[np.ndarray] = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        nodes = []
        while q:
            u = q.popleft()
            nodes.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(np.asarray(nodes, dtype=np.int32))
    return comps


def _skeletonize_mask(mask_xyz: np.ndarray) -> np.ndarray:
    m = mask_xyz > 0
    if skeletonize_3d is not None:
        try:
            return skeletonize_3d(m)
        except Exception:
            pass
    try:
        return skeletonize(m, method="lee")
    except TypeError:
        return skeletonize(m)


def _build_skeleton_graph(mask_xyz: np.ndarray) -> Tuple[np.ndarray, List[List[int]], np.ndarray, List[np.ndarray]]:
    skel = _skeletonize_mask(mask_xyz)
    pts = np.argwhere(skel > 0)
    if pts.size == 0:
        raise ValueError("Skeletonization produced an empty centerline.")

    lut = {tuple(int(v) for v in p): i for i, p in enumerate(pts)}
    adj: List[List[int]] = [[] for _ in range(len(pts))]
    for i, p in enumerate(pts):
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        for dx, dy, dz in _N26_OFFSETS:
            j = lut.get((x + dx, y + dy, z + dz))
            if j is None or j <= i:
                continue
            adj[i].append(j)
            adj[j].append(i)
    deg = np.asarray([len(a) for a in adj], dtype=np.int32)
    comps = _graph_components(adj)
    return pts.astype(np.int32), adj, deg, comps


def _bfs_farthest(adj: List[List[int]], start: int, allowed: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    n = len(adj)
    parent = np.full((n,), -1, dtype=np.int32)
    dist = np.full((n,), -1, dtype=np.int32)
    q = deque([int(start)])
    dist[int(start)] = 0
    far = int(start)
    while q:
        u = q.popleft()
        if dist[u] > dist[far]:
            far = u
        for v in adj[u]:
            if (not allowed[v]) or dist[v] >= 0:
                continue
            dist[v] = dist[u] + 1
            parent[v] = u
            q.append(v)
    return far, parent, dist


def _reconstruct_path(parent: np.ndarray, end: int) -> np.ndarray:
    out = [int(end)]
    cur = int(end)
    while parent[cur] >= 0:
        cur = int(parent[cur])
        out.append(cur)
    out.reverse()
    return np.asarray(out, dtype=np.int32)


def _shortest_path_bfs(adj: List[List[int]], start: int, end: int, allowed: np.ndarray) -> np.ndarray:
    n = len(adj)
    parent = np.full((n,), -1, dtype=np.int32)
    seen = np.zeros((n,), dtype=bool)
    q = deque([int(start)])
    seen[int(start)] = True
    found = False
    while q:
        u = q.popleft()
        if u == int(end):
            found = True
            break
        for v in adj[u]:
            if (not allowed[v]) or seen[v]:
                continue
            seen[v] = True
            parent[v] = u
            q.append(v)
    if not found:
        raise ValueError("Could not find a valid centerline path between requested endpoints.")
    return _reconstruct_path(parent, int(end))


def _smooth_polyline_vox(points_xyz: np.ndarray, window: int) -> np.ndarray:
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.shape[0] < 3:
        return p
    w = max(1, int(window))
    if w <= 1:
        return p
    if w % 2 == 0:
        w += 1
    if p.shape[0] < w:
        return p
    k = np.ones((w,), dtype=np.float64) / float(w)
    pad = w // 2
    out = np.zeros_like(p)
    for a in range(3):
        arr = np.pad(p[:, a], (pad, pad), mode="edge")
        out[:, a] = np.convolve(arr, k, mode="valid")
    return out


def _sample_centerline_planes(
    centerline_vox: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    n_planes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    pts_mm = np.asarray(centerline_vox, dtype=np.float64) * spacing[None, :]
    if pts_mm.shape[0] < 2:
        raise ValueError("Centerline path is too short to define perpendicular planes.")

    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(2, int(n_planes))
    targets = np.linspace(0.0, total, n, dtype=np.float64)

    sampled = np.zeros((n, 3), dtype=np.float64)
    for a in range(3):
        sampled[:, a] = np.interp(targets, s, pts_mm[:, a])

    tang = np.gradient(sampled, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    nrm[nrm < 1e-8] = 1.0
    tang = tang / nrm
    return sampled.astype(np.float32), tang.astype(np.float32)


def _build_centerline_bundle(
    mask_txyz: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    mask_mode: str,
    mask_frame_index: int,
    keep_components: int,
    closing_iters: int,
    smooth_window: int,
    n_planes: int,
    start_xyz: Optional[Sequence[int]],
    end_xyz: Optional[Sequence[int]],
) -> Dict[str, Any]:
    mask_3d = _pick_centerline_mask(mask_txyz, mode=str(mask_mode), frame_index=int(mask_frame_index))
    clean_mask = _clean_centerline_mask(mask_3d, keep_components=int(keep_components), closing_iters=int(closing_iters))

    pts, adj, deg, comps = _build_skeleton_graph(clean_mask)
    comp_id = np.full((len(pts),), -1, dtype=np.int32)
    for ci, c in enumerate(comps):
        comp_id[c] = int(ci)

    if start_xyz is not None and end_xyz is not None:
        tree = cKDTree(pts.astype(np.float64))
        s_idx = int(tree.query(np.asarray(start_xyz, dtype=np.float64), k=1)[1])
        e_idx = int(tree.query(np.asarray(end_xyz, dtype=np.float64), k=1)[1])
        if comp_id[s_idx] != comp_id[e_idx]:
            raise ValueError("Provided centerline start/end are not in the same connected component.")
        allowed = np.zeros((len(pts),), dtype=bool)
        allowed[np.asarray(comps[int(comp_id[s_idx])], dtype=np.int32)] = True
        path_idx = _shortest_path_bfs(adj, start=s_idx, end=e_idx, allowed=allowed)
        path_mode = "user_endpoints"
    else:
        comp = max(comps, key=lambda x: int(x.shape[0]))
        allowed = np.zeros((len(pts),), dtype=bool)
        allowed[comp] = True
        endpoints = np.asarray([i for i in comp if deg[i] == 1], dtype=np.int32)
        start = int(endpoints[0]) if endpoints.size > 0 else int(comp[0])
        a, _, _ = _bfs_farthest(adj, start=start, allowed=allowed)
        b, parent, _ = _bfs_farthest(adj, start=a, allowed=allowed)
        path_idx = _reconstruct_path(parent, end=b)
        path_mode = "largest_component_longest_path"

    if path_idx.shape[0] < 2:
        raise ValueError("Centerline path extraction failed: path has fewer than 2 points.")

    path_vox = pts[path_idx].astype(np.float32)
    path_smooth_vox = _smooth_polyline_vox(path_vox, window=int(smooth_window)).astype(np.float32)
    plane_points_mm, plane_normals = _sample_centerline_planes(
        centerline_vox=path_smooth_vox,
        spacing_mm=spacing_mm,
        n_planes=int(n_planes),
    )
    plane_points_vox = (plane_points_mm / np.asarray(spacing_mm, dtype=np.float32)[None, :]).astype(np.float32)
    return {
        "mask_3d": clean_mask.astype(np.uint8),
        "path_mode": str(path_mode),
        "path_vox": path_vox,
        "path_smooth_vox": path_smooth_vox,
        "plane_points_mm": plane_points_mm,
        "plane_points_vox": plane_points_vox,
        "plane_normals": plane_normals,
    }


def _flow_rate_curves_centerline(
    vel: np.ndarray,
    mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    plane_points_mm: np.ndarray,
    plane_normals: np.ndarray,
    slab_thickness_mm: float,
    min_plane_voxels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if vel.ndim != 5 or mask.ndim != 4:
        raise ValueError(f"Expected vel [T,3,X,Y,Z] and mask [T,X,Y,Z], got {vel.shape} and {mask.shape}")
    t_count = vel.shape[0]
    n_planes = int(plane_points_mm.shape[0])
    q = np.full((t_count, n_planes), np.nan, dtype=np.float32)
    vox_counts = np.zeros((t_count, n_planes), dtype=np.int32)

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    voxel_vol_mm3 = float(np.prod(spacing))
    thickness = float(slab_thickness_mm)
    if thickness <= 0:
        thickness = float(np.mean(spacing))
    half_t = 0.5 * thickness

    for t in range(t_count):
        m = mask[t] > 0.5
        idx = np.argwhere(m)
        if idx.size == 0:
            continue
        coords_mm = (idx.astype(np.float64) + 0.5) * spacing[None, :]
        vel_vec = np.transpose(vel[t], (1, 2, 3, 0))[m].astype(np.float64)

        for p in range(n_planes):
            nvec = plane_normals[p].astype(np.float64)
            nrm = float(np.linalg.norm(nvec))
            if nrm < 1e-8:
                continue
            nvec = nvec / nrm
            d = np.dot(coords_mm - plane_points_mm[p].astype(np.float64)[None, :], nvec)
            sel = np.abs(d) <= half_t
            count = int(np.count_nonzero(sel))
            vox_counts[t, p] = count
            if count < int(min_plane_voxels):
                continue
            vn = np.dot(vel_vec[sel], nvec)
            q[t, p] = float(np.sum(vn) * voxel_vol_mm3 / thickness)
    return q, vox_counts


def _temporal_flow_from_sections(
    q_curves: np.ndarray,
    section_support: np.ndarray,
    min_support: int,
    aggregate: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if q_curves.ndim != 2:
        raise ValueError(f"Expected q_curves [T,S], got {q_curves.shape}")
    if section_support.shape != q_curves.shape:
        raise ValueError(f"section_support shape {section_support.shape} must match q_curves {q_curves.shape}")

    valid = np.where(np.sum(section_support, axis=0) >= int(min_support))[0]
    if valid.size == 0:
        finite_counts = np.sum(np.isfinite(q_curves), axis=0)
        valid = np.where(finite_counts > 0)[0]
    if valid.size == 0:
        return np.full((q_curves.shape[0],), np.nan, dtype=np.float32), valid.astype(np.int32)

    q_sel = q_curves[:, valid]
    agg = str(aggregate).strip().lower()
    if agg == "mean":
        q_time = np.nanmean(q_sel, axis=1).astype(np.float32)
    else:
        q_time = np.nanmedian(q_sel, axis=1).astype(np.float32)
    return q_time, valid.astype(np.int32)


def _aggregate_flow_sections(
    q_curves: np.ndarray,
    section_index: np.ndarray,
    aggregate: str,
) -> np.ndarray:
    if q_curves.ndim != 2:
        raise ValueError(f"Expected q_curves [T,S], got {q_curves.shape}")
    idx = np.asarray(section_index, dtype=np.int32)
    if idx.size == 0:
        return np.full((q_curves.shape[0],), np.nan, dtype=np.float32)
    q_sel = q_curves[:, idx]
    agg = str(aggregate).strip().lower()
    if agg == "mean":
        return np.nanmean(q_sel, axis=1).astype(np.float32)
    return np.nanmedian(q_sel, axis=1).astype(np.float32)


def _flow_rows_per_frame_slice(
    q_ref_curves: np.ndarray,
    q_base_curves: np.ndarray,
    q_sr_curves: np.ndarray,
    q_ref_scalar: float,
    frame_source_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    t_count, n_slices = q_ref_curves.shape
    if frame_source_indices.shape[0] != t_count:
        frame_source_indices = np.arange(t_count, dtype=np.int32)

    rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        for s in range(n_slices):
            q_ref = float(q_ref_curves[t, s])
            q_base = float(q_base_curves[t, s])
            q_sr = float(q_sr_curves[t, s])
            rows.append(
                {
                    "frame_payload_index": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "slice_index": int(s),
                    "q_ref_ml_s": q_ref,
                    "q_baseline_ml_s": q_base,
                    "q_sr_ml_s": q_sr,
                    "abs_err_baseline_vs_ref_ml_s": abs(q_base - q_ref),
                    "abs_err_sr_vs_ref_ml_s": abs(q_sr - q_ref),
                    "abs_err_baseline_vs_qref_ml_s": abs(q_base - float(q_ref_scalar)),
                    "abs_err_sr_vs_qref_ml_s": abs(q_sr - float(q_ref_scalar)),
                }
            )
    return rows


def _flow_summary_rows_per_frame(
    q_ref_curves: np.ndarray,
    q_base_curves: np.ndarray,
    q_sr_curves: np.ndarray,
    q_ref_scalar: float,
    frame_source_indices: np.ndarray,
    ref_label: str,
    baseline_label: str,
    sr_label: str,
) -> List[Dict[str, Any]]:
    t_count, _ = q_ref_curves.shape
    if frame_source_indices.shape[0] != t_count:
        frame_source_indices = np.arange(t_count, dtype=np.int32)

    rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        ref_t = q_ref_curves[t]
        base_t = q_base_curves[t]
        sr_t = q_sr_curves[t]

        def _append(method: str, q_method_t: np.ndarray) -> None:
            rows.append(
                {
                    "frame_payload_index": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "method": method,
                    "mean_Q_ml_s": float(np.mean(q_method_t)),
                    "MAD_Q_vs_qref_ml_s": float(np.mean(np.abs(q_method_t - float(q_ref_scalar)))),
                    "MAD_Q_vs_ref_profile_ml_s": float(np.mean(np.abs(q_method_t - ref_t))),
                }
            )

        _append(ref_label, ref_t)
        _append(baseline_label, base_t)
        _append(sr_label, sr_t)

    return rows


def _slice_voxel_counts_by_axis(mask_txyz: np.ndarray, flow_axis: int) -> np.ndarray:
    # mask_txyz: [T, X, Y, Z] -> counts [T, S]
    if mask_txyz.ndim != 4:
        raise ValueError(f"Expected mask_txyz [T,X,Y,Z], got {mask_txyz.shape}")
    t_count = mask_txyz.shape[0]
    n_slices = mask_txyz.shape[flow_axis + 1]
    out = np.zeros((t_count, n_slices), dtype=np.int32)
    for t in range(t_count):
        m = mask_txyz[t] > 0.5
        for s in range(n_slices):
            out[t, s] = int(np.count_nonzero(_extract_slice(m, flow_axis, s)))
    return out


def _temporal_flow_from_slices(
    q_curves: np.ndarray,
    slice_counts: np.ndarray,
    min_voxels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # q_curves: [T,S], slice_counts: [T,S]
    if q_curves.ndim != 2:
        raise ValueError(f"Expected q_curves [T,S], got {q_curves.shape}")
    if slice_counts.shape != q_curves.shape:
        raise ValueError(f"slice_counts shape {slice_counts.shape} must match q_curves {q_curves.shape}")

    valid_slices = np.where(slice_counts.sum(axis=0) >= int(min_voxels))[0]
    if valid_slices.size == 0:
        valid_slices = np.arange(q_curves.shape[1], dtype=np.int32)
    q_time = np.mean(q_curves[:, valid_slices], axis=1).astype(np.float32)
    return q_time, valid_slices.astype(np.int32)


def _correlation_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return {
            "n": float(x.size),
            "slope": float("nan"),
            "intercept": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "r2_linear": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }

    try:
        pearson_r, pearson_p = stats.pearsonr(x, y)
    except Exception:
        pearson_r, pearson_p = float("nan"), float("nan")
    try:
        spearman_rho, spearman_p = stats.spearmanr(x, y)
    except Exception:
        spearman_rho, spearman_p = float("nan"), float("nan")

    try:
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    except Exception:
        a, b = float("nan"), float("nan")
        r2 = float("nan")

    return {
        "n": float(x.size),
        "slope": float(a),
        "intercept": float(b),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "r2_linear": float(r2),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "bias": float(np.mean(y - x)),
    }


def _plot_correlation_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    color: str,
    seed: int,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return _correlation_stats(x, y)

    idx = np.arange(x.size)
    if x.size > 60000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=60000, replace=False)
    xs = x[idx]
    ys = y[idx]

    ax.scatter(xs, ys, s=6, alpha=0.22, color=color, edgecolors="none")
    lo, hi = _robust_range([x, y], symmetric=False, lower_q=0.2, upper_q=99.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#111827", linewidth=1.0, label="Identity")
    if x.size >= 2:
        try:
            a, b = np.polyfit(x, y, 1)
            ax.plot([lo, hi], [a * lo + b, a * hi + b], color="#7c3aed", linewidth=1.4, label="Linear fit")
        except Exception:
            pass

    st = _correlation_stats(x, y)
    txt = (
        f"n={int(st['n'])}\n"
        f"r={st['pearson_r']:.4f}\n"
        f"rho={st['spearman_rho']:.4f}\n"
        f"R2={st['r2_linear']:.4f}\n"
        f"RMSE={st['rmse']:.4f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8})
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=8, loc="lower right")
    return st


def _plot_bland_altman_panel(
    ax,
    ref_vals: np.ndarray,
    test_vals: np.ndarray,
    ref_label: str,
    test_label: str,
    seed: int,
) -> Dict[str, float]:
    ref_vals = np.asarray(ref_vals, dtype=np.float64)
    test_vals = np.asarray(test_vals, dtype=np.float64)
    m = np.isfinite(ref_vals) & np.isfinite(test_vals)
    ref_vals = ref_vals[m]
    test_vals = test_vals[m]
    if ref_vals.size < 3:
        ax.set_title(f"{test_label} vs {ref_label} (insufficient data)")
        return {
            "n": float(ref_vals.size),
            "bias": float("nan"),
            "loa_low": float("nan"),
            "loa_high": float("nan"),
            "sd_diff": float("nan"),
        }

    mean_v = 0.5 * (test_vals + ref_vals)
    diff_v = test_vals - ref_vals
    idx = np.arange(mean_v.size)
    if mean_v.size > 60000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(mean_v.size, size=60000, replace=False)
    ax.scatter(mean_v[idx], diff_v[idx], s=6, alpha=0.2, edgecolors="none")

    bias = float(np.mean(diff_v))
    sd = float(np.std(diff_v, ddof=1)) if diff_v.size > 1 else float("nan")
    loa_low = bias - 1.96 * sd if np.isfinite(sd) else float("nan")
    loa_high = bias + 1.96 * sd if np.isfinite(sd) else float("nan")
    ax.axhline(bias, color="#b91c1c", linestyle="-", linewidth=1.4, label=f"Bias={bias:.4f}")
    ax.axhline(loa_low, color="#111827", linestyle="--", linewidth=1.1, label=f"LoA low={loa_low:.4f}")
    ax.axhline(loa_high, color="#111827", linestyle="--", linewidth=1.1, label=f"LoA high={loa_high:.4f}")
    ax.set_title(f"{test_label} vs {ref_label}")
    ax.set_xlabel(f"Mean({test_label}, {ref_label}) [m/s]")
    ax.set_ylabel(f"{test_label} - {ref_label} [m/s]")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=8)
    return {
        "n": float(ref_vals.size),
        "bias": float(bias),
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "sd_diff": float(sd),
    }


def _suggest_flow_axis(
    vel_ref: np.ndarray,
    mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    # Heuristic:
    # - Lower temporal relative SD in per-slice flow
    # - Smoother mean flow profile along the axis
    # - Higher slice coverage with substantial flow
    details: Dict[int, Dict[str, float]] = {}

    for axis in (0, 1, 2):
        q = _flow_rate_curves(vel_ref, mask, flow_axis=axis, spacing_mm=spacing_mm)
        q_mean = q.mean(axis=0)
        q_sd = q.std(axis=0, ddof=1) if q.shape[0] > 1 else np.zeros_like(q_mean)

        abs_mean = np.abs(q_mean)
        peak = float(np.max(abs_mean)) if abs_mean.size > 0 else 0.0
        threshold = max(1e-8, 0.10 * peak)
        valid = abs_mean >= threshold
        coverage = float(np.mean(valid)) if valid.size > 0 else 0.0

        if np.any(valid):
            rel_sd = float(np.mean(q_sd[valid] / (abs_mean[valid] + 1e-8)))
        else:
            rel_sd = 1e6

        if q_mean.size >= 3:
            d1 = np.diff(q_mean)
            d2 = np.diff(q_mean, n=2)
            smoothness = float(np.mean(np.abs(d2)) / (np.mean(np.abs(d1)) + 1e-8))
        else:
            smoothness = 1e6

        score = rel_sd + 0.20 * smoothness - 0.15 * coverage
        details[axis] = {
            "score": float(score),
            "rel_sd": float(rel_sd),
            "smoothness": float(smoothness),
            "coverage": float(coverage),
        }

    best_axis = min(details.keys(), key=lambda a: details[a]["score"])
    return int(best_axis), details


def _wss_boundary_points(mask_ref: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    mask_bin = mask_ref > 0.5
    eroded = binary_erosion(mask_bin, iterations=1)
    boundary = mask_bin & (~eroded)

    sm = gaussian_filter(mask_bin.astype(np.float32), sigma=1.0)
    gx, gy, gz = np.gradient(sm)

    pts = np.argwhere(boundary)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if pts.shape[0] > int(max_points):
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    n = np.stack([gx[pts[:, 0], pts[:, 1], pts[:, 2]], gy[pts[:, 0], pts[:, 1], pts[:, 2]], gz[pts[:, 0], pts[:, 1], pts[:, 2]]], axis=1)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    valid = (n_norm[:, 0] > 1e-6)
    pts = pts[valid]
    n = n[valid] / n_norm[valid]

    return pts.astype(np.float32), n.astype(np.float32)


def _sample_trilinear_scalar(vol: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # vol: [X,Y,Z], pts: [N,3] in voxel coordinates
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    z0 = np.floor(z).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (z0 >= 0) & (x1 < vol.shape[0]) & (y1 < vol.shape[1]) & (z1 < vol.shape[2])
    out = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out, valid

    xv, yv, zv = x[valid], y[valid], z[valid]
    x0v, y0v, z0v = x0[valid], y0[valid], z0[valid]
    x1v, y1v, z1v = x1[valid], y1[valid], z1[valid]

    xd = xv - x0v
    yd = yv - y0v
    zd = zv - z0v

    c000 = vol[x0v, y0v, z0v]
    c001 = vol[x0v, y0v, z1v]
    c010 = vol[x0v, y1v, z0v]
    c011 = vol[x0v, y1v, z1v]
    c100 = vol[x1v, y0v, z0v]
    c101 = vol[x1v, y0v, z1v]
    c110 = vol[x1v, y1v, z0v]
    c111 = vol[x1v, y1v, z1v]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    out[valid] = c0 * (1 - zd) + c1 * zd
    return out, valid


def _sample_trilinear_vector(uvw: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = []
    valid_all = None
    for c in range(uvw.shape[0]):
        s, v = _sample_trilinear_scalar(uvw[c], pts)
        vals.append(s)
        valid_all = v if valid_all is None else (valid_all & v)
    arr = np.stack(vals, axis=1)
    return arr, valid_all


def _compute_wss_distribution(
    uvw_mean: np.ndarray,
    mask_ref: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    mu_pa_s: float,
    max_points: int,
    seed: int,
) -> np.ndarray:
    pts, normals = _wss_boundary_points(mask_ref=mask_ref, max_points=max_points, seed=seed)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    center = np.argwhere(mask_ref > 0.5).mean(axis=0)
    n_in = normals.copy()

    # Choose inward direction heuristically.
    p_plus = np.round(pts + n_in).astype(int)
    p_minus = np.round(pts - n_in).astype(int)

    def inside(p):
        valid = (
            (p[:, 0] >= 0)
            & (p[:, 1] >= 0)
            & (p[:, 2] >= 0)
            & (p[:, 0] < mask_ref.shape[0])
            & (p[:, 1] < mask_ref.shape[1])
            & (p[:, 2] < mask_ref.shape[2])
        )
        out = np.zeros((p.shape[0],), dtype=bool)
        out[valid] = mask_ref[p[valid, 0], p[valid, 1], p[valid, 2]] > 0.5
        return out

    in_plus = inside(p_plus)
    in_minus = inside(p_minus)

    both = in_plus & in_minus
    choose_minus = in_minus & (~in_plus)
    choose_plus = in_plus & (~in_minus)

    n_in[choose_minus] = -n_in[choose_minus]
    if np.any(both):
        toward_center = center[None, :] - pts[both]
        sgn = np.sum(toward_center * n_in[both], axis=1)
        flip = sgn < 0
        n_tmp = n_in[both]
        n_tmp[flip] = -n_tmp[flip]
        n_in[both] = n_tmp

    p1 = pts + n_in
    p2 = pts + 2.0 * n_in

    v1, valid1 = _sample_trilinear_vector(uvw_mean, p1)
    v2, valid2 = _sample_trilinear_vector(uvw_mean, p2)
    valid = valid1 & valid2
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64)

    n_valid = n_in[valid].astype(np.float64)
    v1 = v1[valid].astype(np.float64)
    v2 = v2[valid].astype(np.float64)

    spacing_m = np.asarray(spacing_mm, dtype=np.float64) / 1000.0
    n_phys = n_valid * spacing_m[None, :]
    d1 = np.linalg.norm(n_phys, axis=1)
    d1 = np.clip(d1, 1e-8, None)
    n_phys_u = n_phys / d1[:, None]

    v1n = np.sum(v1 * n_phys_u, axis=1)
    v2n = np.sum(v2 * n_phys_u, axis=1)

    vt1 = v1 - v1n[:, None] * n_phys_u
    vt2 = v2 - v2n[:, None] * n_phys_u

    vt1_mag = np.linalg.norm(vt1, axis=1)
    vt2_mag = np.linalg.norm(vt2, axis=1)

    shear_rate = np.abs((4.0 * vt1_mag - vt2_mag) / (2.0 * d1))
    tau = float(mu_pa_s) * shear_rate
    return tau.astype(np.float64)


def _wss_summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {k: float("nan") for k in ["Maximum", "Mean", "SD", "Quantile_97_5", "Median", "Quantile_2_5", "IQR_75_25"]}
    return {
        "Maximum": float(np.max(arr)),
        "Mean": float(np.mean(arr)),
        "SD": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        "Quantile_97_5": float(np.quantile(arr, 0.975)),
        "Median": float(np.median(arr)),
        "Quantile_2_5": float(np.quantile(arr, 0.025)),
        "IQR_75_25": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
    }


def _surface_distance_metrics(mask_a: np.ndarray, mask_b: np.ndarray, spacing_mm: Tuple[float, float, float]) -> Dict[str, float]:
    if int(mask_a.sum()) < 10 or int(mask_b.sum()) < 10:
        return {
            "mean_surface_distance_a_to_b_mm": float("nan"),
            "std_surface_distance_a_to_b_mm": float("nan"),
            "symmetric_mean_surface_distance_mm": float("nan"),
            "hausdorff_distance_mm": float("nan"),
        }

    va, _, _, _ = marching_cubes(mask_a.astype(np.float32), level=0.5, spacing=spacing_mm)
    vb, _, _, _ = marching_cubes(mask_b.astype(np.float32), level=0.5, spacing=spacing_mm)
    if va.shape[0] == 0 or vb.shape[0] == 0:
        return {
            "mean_surface_distance_a_to_b_mm": float("nan"),
            "std_surface_distance_a_to_b_mm": float("nan"),
            "symmetric_mean_surface_distance_mm": float("nan"),
            "hausdorff_distance_mm": float("nan"),
        }

    ta = cKDTree(va)
    tb = cKDTree(vb)
    d_ab = tb.query(va, k=1)[0]
    d_ba = ta.query(vb, k=1)[0]

    return {
        "mean_surface_distance_a_to_b_mm": float(np.mean(d_ab)),
        "std_surface_distance_a_to_b_mm": float(np.std(d_ab, ddof=1)) if d_ab.size > 1 else float("nan"),
        "symmetric_mean_surface_distance_mm": float(0.5 * (np.mean(d_ab) + np.mean(d_ba))),
        "hausdorff_distance_mm": float(max(np.max(d_ab), np.max(d_ba))),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _upsample_spatial(arr: np.ndarray, out_shape_xyz: Tuple[int, int, int], mode: str = "trilinear") -> np.ndarray:
    """Upsample [T,C,X,Y,Z] or [T,X,Y,Z] arrays to a target spatial shape."""
    if arr.ndim not in (4, 5):
        raise ValueError(f"Expected 4D/5D array, got shape {arr.shape}")

    if arr.ndim == 4:
        x = torch.from_numpy(arr[:, None, ...].astype(np.float32))
    else:
        x = torch.from_numpy(arr.astype(np.float32))

    if mode == "nearest":
        y = torch.nn.functional.interpolate(x, size=out_shape_xyz, mode=mode)
    else:
        y = torch.nn.functional.interpolate(x, size=out_shape_xyz, mode=mode, align_corners=False)

    out = y.numpy()
    if arr.ndim == 4:
        out = out[:, 0]
    return out.astype(np.float32)


def _clip_bbox_xyz(bbox_xyz: Sequence[int], shape_xyz: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    if len(bbox_xyz) != 6:
        raise ValueError(f"ROI bbox expects 6 integers [x0,x1,y0,y1,z0,z1), got {bbox_xyz}")
    x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_xyz]
    sx, sy, sz = [int(v) for v in shape_xyz]
    x0 = max(0, min(x0, sx - 1))
    y0 = max(0, min(y0, sy - 1))
    z0 = max(0, min(z0, sz - 1))
    x1 = max(x0 + 1, min(x1, sx))
    y1 = max(y0 + 1, min(y1, sy))
    z1 = max(z0 + 1, min(z1, sz))
    return x0, x1, y0, y1, z0, z1


def _resolve_roi_bbox(
    roi_bbox_cli: Optional[Sequence[int]],
    roi_json_path: str,
    shape_xyz: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    bbox_raw: Optional[Sequence[int]] = None

    if roi_json_path:
        p = Path(roi_json_path)
        if not p.exists():
            raise FileNotFoundError(f"ROI json not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        for key in ("bbox_hr_xyz", "bbox_xyz", "roi_bbox_hr_xyz", "roi_bbox_xyz"):
            val = data.get(key)
            if isinstance(val, list) and len(val) == 6:
                bbox_raw = [int(v) for v in val]
                break
        if bbox_raw is None:
            raise ValueError(
                f"ROI json {p} does not contain bbox field in one of "
                f"{['bbox_hr_xyz', 'bbox_xyz', 'roi_bbox_hr_xyz', 'roi_bbox_xyz']}"
            )

    if roi_bbox_cli is not None and len(roi_bbox_cli) == 6:
        bbox_raw = [int(v) for v in roi_bbox_cli]

    if bbox_raw is None:
        return None

    return _clip_bbox_xyz(bbox_raw, shape_xyz)


def _apply_roi_to_mask(
    mask_txyz: np.ndarray,
    bbox_xyz: Optional[Tuple[int, int, int, int, int, int]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if bbox_xyz is None:
        return mask_txyz.astype(np.float32), {"enabled": False}

    x0, x1, y0, y1, z0, z1 = bbox_xyz
    roi = np.zeros(mask_txyz.shape[1:], dtype=np.float32)
    roi[x0:x1, y0:y1, z0:z1] = 1.0
    out = (mask_txyz > 0.5).astype(np.float32) * roi[None, ...]
    info = {
        "enabled": True,
        "bbox_xyz": [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)],
        "bbox_size_xyz": [int(x1 - x0), int(y1 - y0), int(z1 - z0)],
    }
    return out.astype(np.float32), info


def _html_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    th = "".join(f"<th>{c}</th>" for c in columns)
    body_parts = []
    for row in rows:
        tds = "".join(f"<td>{row.get(c, '')}</td>" for c in columns)
        body_parts.append(f"<tr>{tds}</tr>")
    body = "\n".join(body_parts)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _autodetect_reference_label(metadata: Dict[str, Any]) -> str:
    try:
        hr_u = str(metadata.get("case_paths", {}).get("hr_u", "")).lower()
    except Exception:
        hr_u = ""
    if "cfd" in hr_u:
        return "CFD"
    if "7t" in hr_u:
        return "7T"
    return "Reference"


def _robust_range(
    arrays: Sequence[np.ndarray],
    symmetric: bool,
    lower_q: float = 0.5,
    upper_q: float = 99.5,
) -> Tuple[float, float]:
    vals: List[np.ndarray] = []
    for arr in arrays:
        v = np.asarray(arr, dtype=np.float32).ravel()
        v = v[np.isfinite(v)]
        if v.size > 0:
            vals.append(v)

    if not vals:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    merged = np.concatenate(vals, axis=0)
    if symmetric:
        abs_m = np.abs(merged)
        vmax = float(np.nanpercentile(abs_m, upper_q))
        if not np.isfinite(vmax) or vmax <= 1e-8:
            vmax = float(np.nanmax(abs_m))
        vmax = max(vmax, 1e-6)
        return -vmax, vmax

    vmin = float(np.nanpercentile(merged, lower_q))
    vmax = float(np.nanpercentile(merged, upper_q))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-8):
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))
    if vmax - vmin < 1e-8:
        pad = max(1e-3, 0.05 * max(abs(vmin), abs(vmax), 1.0))
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _subsample_for_plot(x: np.ndarray, max_samples: int = 350000, seed: int = 11) -> np.ndarray:
    if x.size <= int(max_samples):
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=int(max_samples), replace=False)
    return x[idx]


def _distribution_row(channel: str, method: str, values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "channel": channel,
            "method": method,
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "channel": channel,
        "method": method,
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "median": float(np.median(v)),
        "p05": float(np.quantile(v, 0.05)),
        "p95": float(np.quantile(v, 0.95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def _save_voxel_histograms(
    out_path: Path,
    baseline_4ch: np.ndarray,
    pred_4ch: np.ndarray,
    gt_4ch: np.ndarray,
    mask: np.ndarray,
    bins: int,
    baseline_label: str,
    sr_label: str,
    ref_label: str,
) -> List[Dict[str, Any]]:
    mask_bool = mask > 0.5
    if int(mask_bool.sum()) == 0:
        mask_bool = np.ones_like(mask, dtype=bool)

    channel_names = ["u", "v", "w", "mag"]
    colors = {"ref": "#111827", "base": "#1d4ed8", "sr": "#b91c1c"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_f = axes.ravel()
    rows: List[Dict[str, Any]] = []

    for c, ch in enumerate(channel_names):
        ax = axes_f[c]
        ref_vals = gt_4ch[:, c][mask_bool]
        base_vals = baseline_4ch[:, c][mask_bool]
        sr_vals = pred_4ch[:, c][mask_bool]

        rows.append(_distribution_row(ch, ref_label, ref_vals))
        rows.append(_distribution_row(ch, baseline_label, base_vals))
        rows.append(_distribution_row(ch, sr_label, sr_vals))

        sym = ch != "mag"
        vmin, vmax = _robust_range([ref_vals, base_vals, sr_vals], symmetric=sym, lower_q=0.5, upper_q=99.5)
        bin_edges = np.linspace(vmin, vmax, max(20, int(bins)) + 1)

        for vals, label, color, sseed in [
            (ref_vals, ref_label, colors["ref"], 21 + c),
            (base_vals, baseline_label, colors["base"], 31 + c),
            (sr_vals, sr_label, colors["sr"], 41 + c),
        ]:
            vv = np.asarray(vals, dtype=np.float64)
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                continue
            vv = vv[(vv >= vmin) & (vv <= vmax)]
            vv = _subsample_for_plot(vv, seed=sseed)
            if vv.size == 0:
                continue
            ax.hist(vv, bins=bin_edges, density=True, histtype="step", linewidth=1.7, alpha=0.95, color=color, label=label)

        ax.set_title(f"{ch.upper()} in-mask voxel distribution")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)

    fig.suptitle("Voxel-value distribution inside vessel mask", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return rows


def _save_channel_figure(
    out_path: Path,
    ch_name: str,
    lr_up: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    max_slices: int,
    n_cols: int = 4,
    bbox_xyz: Optional[Sequence[int]] = None,
) -> None:
    z_offset = 0
    if bbox_xyz is not None and len(bbox_xyz) == 6:
        x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_xyz]
        lr_view = lr_up[x0:x1, y0:y1, z0:z1]
        pred_view = pred[x0:x1, y0:y1, z0:z1]
        gt_view = gt[x0:x1, y0:y1, z0:z1]
        z_offset = int(z0)
    else:
        lr_view = lr_up
        pred_view = pred
        gt_view = gt

    n_slices = gt_view.shape[-1]
    if n_slices <= 0:
        raise ValueError("Channel figure received empty ROI after bbox cropping.")
    if n_slices <= max_slices:
        z_idx = list(range(n_slices))
    else:
        z_idx = np.linspace(0, n_slices - 1, max_slices).round().astype(int).tolist()

    is_mag = ch_name == "mag"
    cmap = "gray"
    vmin, vmax = _robust_range([lr_view, pred_view, gt_view], symmetric=(not is_mag), lower_q=0.5, upper_q=99.5)
    _, emax = _robust_range([np.abs(pred_view - gt_view)], symmetric=False, lower_q=0.0, upper_q=99.5)

    n_cols = max(1, min(int(n_cols), len(z_idx)))
    n_blocks = int(math.ceil(len(z_idx) / float(n_cols)))
    n_rows = 4 * n_blocks

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.5 * n_rows), squeeze=False)
    for rr in range(n_rows):
        for cc in range(n_cols):
            axes[rr, cc].axis("off")

    for j, z in enumerate(z_idx):
        blk = j // n_cols
        col = j % n_cols
        r0 = 4 * blk

        inp = lr_view[:, :, z]
        pd = pred_view[:, :, z]
        gtz = gt_view[:, :, z]
        err = np.abs(pd - gtz)

        axes[r0 + 0, col].imshow(inp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[r0 + 1, col].imshow(pd, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[r0 + 2, col].imshow(gtz, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[r0 + 3, col].imshow(err, cmap="magma", vmin=0.0, vmax=emax, interpolation="nearest")

        axes[r0 + 0, col].set_title(f"z={z + z_offset}", fontsize=10)
        for rr in range(4):
            axes[r0 + rr, col].axis("off")
            axes[r0 + rr, col].grid(False)

        if col == 0:
            axes[r0 + 0, col].set_ylabel("Input (LR up)", fontsize=10)
            axes[r0 + 1, col].set_ylabel("Prediction", fontsize=10)
            axes[r0 + 2, col].set_ylabel("Ground truth", fontsize=10)
            axes[r0 + 3, col].set_ylabel("|Error|", fontsize=10)

    region_name = "ROI bbox" if bbox_xyz is not None else "Full volume"
    fig.suptitle(f"{region_name} comparison: {ch_name}  |  value range [{vmin:.3f}, {vmax:.3f}]  |  error p99.5={emax:.3f}", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_centerline_overlay_figure(
    out_path: Path,
    mask_3d: np.ndarray,
    centerline_vox: np.ndarray,
    plane_points_vox: np.ndarray,
    valid_plane_index: np.ndarray,
) -> None:
    m = (np.asarray(mask_3d) > 0).astype(np.float32)
    cl = np.asarray(centerline_vox, dtype=np.float64)
    pp = np.asarray(plane_points_vox, dtype=np.float64)
    valid = set(int(v) for v in np.asarray(valid_plane_index, dtype=np.int32).tolist())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # X projection -> YZ plane (imshow x=Z, y=Y)
    p_yz = np.max(m, axis=0)
    axes[0].imshow(p_yz, origin="lower", cmap="gray")
    if cl.size > 0:
        axes[0].plot(cl[:, 2], cl[:, 1], color="#f59e0b", linewidth=2.0, label="centerline")
    if pp.size > 0:
        p_valid = np.asarray([i for i in range(pp.shape[0]) if i in valid], dtype=np.int32)
        p_invalid = np.asarray([i for i in range(pp.shape[0]) if i not in valid], dtype=np.int32)
        if p_invalid.size > 0:
            axes[0].scatter(pp[p_invalid, 2], pp[p_invalid, 1], s=18, c="#ef4444", label="plane (invalid)")
        if p_valid.size > 0:
            axes[0].scatter(pp[p_valid, 2], pp[p_valid, 1], s=20, c="#22c55e", label="plane (valid)")
    axes[0].set_title("Projection X (YZ)")
    axes[0].set_xlabel("Z [vox]")
    axes[0].set_ylabel("Y [vox]")

    # Y projection -> XZ plane (imshow x=Z, y=X)
    p_xz = np.max(m, axis=1)
    axes[1].imshow(p_xz, origin="lower", cmap="gray")
    if cl.size > 0:
        axes[1].plot(cl[:, 2], cl[:, 0], color="#f59e0b", linewidth=2.0)
    if pp.size > 0:
        p_valid = np.asarray([i for i in range(pp.shape[0]) if i in valid], dtype=np.int32)
        p_invalid = np.asarray([i for i in range(pp.shape[0]) if i not in valid], dtype=np.int32)
        if p_invalid.size > 0:
            axes[1].scatter(pp[p_invalid, 2], pp[p_invalid, 0], s=18, c="#ef4444")
        if p_valid.size > 0:
            axes[1].scatter(pp[p_valid, 2], pp[p_valid, 0], s=20, c="#22c55e")
    axes[1].set_title("Projection Y (XZ)")
    axes[1].set_xlabel("Z [vox]")
    axes[1].set_ylabel("X [vox]")

    # Z projection -> XY plane (imshow x=Y, y=X)
    p_xy = np.max(m, axis=2)
    axes[2].imshow(p_xy, origin="lower", cmap="gray")
    if cl.size > 0:
        axes[2].plot(cl[:, 1], cl[:, 0], color="#f59e0b", linewidth=2.0)
    if pp.size > 0:
        p_valid = np.asarray([i for i in range(pp.shape[0]) if i in valid], dtype=np.int32)
        p_invalid = np.asarray([i for i in range(pp.shape[0]) if i not in valid], dtype=np.int32)
        if p_invalid.size > 0:
            axes[2].scatter(pp[p_invalid, 1], pp[p_invalid, 0], s=18, c="#ef4444")
        if p_valid.size > 0:
            axes[2].scatter(pp[p_valid, 1], pp[p_valid, 0], s=20, c="#22c55e")
    axes[2].set_title("Projection Z (XY)")
    axes[2].set_xlabel("Y [vox]")
    axes[2].set_ylabel("X [vox]")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Centerline selection and sampled planes (valid={len(valid)}/{int(pp.shape[0])})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_centerline_3d_figure(
    out_path: Path,
    mask_3d: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    centerline_vox: np.ndarray,
    plane_points_vox: np.ndarray,
    valid_plane_index: np.ndarray,
    max_mask_points: int = 40000,
) -> None:
    m = np.asarray(mask_3d) > 0
    if int(np.count_nonzero(m)) == 0:
        return

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    cl_mm = np.asarray(centerline_vox, dtype=np.float64) * spacing[None, :]
    pp_mm = np.asarray(plane_points_vox, dtype=np.float64) * spacing[None, :]
    valid = set(int(v) for v in np.asarray(valid_plane_index, dtype=np.int32).tolist())

    verts = None
    faces = None
    try:
        v, f, _, _ = marching_cubes(m.astype(np.float32), level=0.5, spacing=spacing_mm)
        verts = np.asarray(v, dtype=np.float64)
        faces = np.asarray(f, dtype=np.int32)
        if faces.shape[0] > 90000:
            step = int(np.ceil(faces.shape[0] / 90000.0))
            faces = faces[::step]
    except Exception:
        verts = None
        faces = None

    pts_mm = None
    if verts is None or faces is None or faces.size == 0:
        boundary = m & (~binary_erosion(m, iterations=1))
        pts = np.argwhere(boundary)
        if pts.size == 0:
            pts = np.argwhere(m)
        if pts.size == 0:
            return
        if pts.shape[0] > int(max_mask_points):
            rng = np.random.default_rng(17)
            pick = rng.choice(pts.shape[0], size=int(max_mask_points), replace=False)
            pts = pts[pick]
        pts_mm = pts.astype(np.float64) * spacing[None, :]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if verts is not None and faces is not None and faces.size > 0:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = Poly3DCollection(verts[faces], alpha=0.10, facecolor="#93c5fd", edgecolor="none")
        ax.add_collection3d(mesh)
        boundary_label = "mask surface"
    else:
        ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2], s=1.0, c="#9ca3af", alpha=0.18, label="mask boundary")
        boundary_label = "mask boundary"

    if cl_mm.size > 0:
        ax.plot(cl_mm[:, 0], cl_mm[:, 1], cl_mm[:, 2], color="#f59e0b", linewidth=3.0, label="centerline")
    if pp_mm.size > 0:
        p_valid = np.asarray([i for i in range(pp_mm.shape[0]) if i in valid], dtype=np.int32)
        p_invalid = np.asarray([i for i in range(pp_mm.shape[0]) if i not in valid], dtype=np.int32)
        if p_invalid.size > 0:
            ax.scatter(pp_mm[p_invalid, 0], pp_mm[p_invalid, 1], pp_mm[p_invalid, 2], s=18, c="#ef4444", label="plane (invalid)")
        if p_valid.size > 0:
            ax.scatter(pp_mm[p_valid, 0], pp_mm[p_valid, 1], pp_mm[p_valid, 2], s=22, c="#22c55e", label="plane (valid)")

    if verts is not None and verts.size > 0:
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
    else:
        mins = np.min(pts_mm, axis=0)
        maxs = np.max(pts_mm, axis=0)
    ranges = np.maximum(maxs - mins, 1e-3)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((float(ranges[0]), float(ranges[1]), float(ranges[2])))
    except Exception:
        pass

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(f"3D Centerline Visualization ({boundary_label})")
    ax.view_init(elev=22, azim=42)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _orthonormal_basis_from_normal(nvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(nvec, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n = n / n_norm
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, a)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u_norm = float(np.linalg.norm(u))
    u = u / max(u_norm, 1e-8)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return u, v


def _centerline_section_geometry_rows(
    mask_3d: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    plane_points_mm: np.ndarray,
    plane_normals: np.ndarray,
    slab_thickness_mm: float,
) -> List[Dict[str, Any]]:
    m = np.asarray(mask_3d) > 0.5
    idx = np.argwhere(m)
    if idx.size == 0:
        return []

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    voxel_vol_mm3 = float(np.prod(spacing))
    thickness = float(slab_thickness_mm)
    if thickness <= 0:
        thickness = float(np.mean(spacing))
    half_t = 0.5 * thickness
    coords_mm = (idx.astype(np.float64) + 0.5) * spacing[None, :]

    dt_mm = distance_transform_edt(m, sampling=spacing_mm).astype(np.float32)
    rows: List[Dict[str, Any]] = []
    for p in range(int(plane_points_mm.shape[0])):
        r0 = np.asarray(plane_points_mm[p], dtype=np.float64)
        n = np.asarray(plane_normals[p], dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            continue
        n = n / n_norm
        d = np.dot(coords_mm - r0[None, :], n)
        sel = np.abs(d) <= half_t
        count = int(np.count_nonzero(sel))
        area_mm2 = float(count * voxel_vol_mm3 / thickness) if count > 0 else float("nan")

        row: Dict[str, Any] = {
            "plane_index": int(p),
            "support_voxels": count,
            "area_mm2": area_mm2,
            "eq_radius_mm": float(np.sqrt(max(area_mm2, 0.0) / np.pi)) if np.isfinite(area_mm2) else float("nan"),
            "centroid_offset_mm": float("nan"),
            "centroid_offset_norm": float("nan"),
            "major_sd_mm": float("nan"),
            "minor_sd_mm": float("nan"),
            "elongation_ratio": float("nan"),
            "compactness_ratio": float("nan"),
            "center_to_wall_mm": float("nan"),
            "center_to_wall_over_eq_radius": float("nan"),
        }

        pv = np.rint((r0 / spacing) - 0.5).astype(np.int32)
        sx, sy, sz = [int(v) for v in m.shape]
        if (0 <= int(pv[0]) < sx) and (0 <= int(pv[1]) < sy) and (0 <= int(pv[2]) < sz) and bool(m[int(pv[0]), int(pv[1]), int(pv[2])]):
            center_to_wall_mm = float(dt_mm[int(pv[0]), int(pv[1]), int(pv[2])])
            row["center_to_wall_mm"] = center_to_wall_mm

        if count >= 3:
            pts = coords_mm[sel]
            cen = np.mean(pts, axis=0)
            offset_mm = float(np.linalg.norm(cen - r0))
            row["centroid_offset_mm"] = offset_mm
            if np.isfinite(row["eq_radius_mm"]) and float(row["eq_radius_mm"]) > 1e-8:
                row["centroid_offset_norm"] = float(offset_mm / float(row["eq_radius_mm"]))
            if np.isfinite(row["center_to_wall_mm"]) and np.isfinite(row["eq_radius_mm"]) and float(row["eq_radius_mm"]) > 1e-8:
                row["center_to_wall_over_eq_radius"] = float(float(row["center_to_wall_mm"]) / float(row["eq_radius_mm"]))

            u, v = _orthonormal_basis_from_normal(n)
            rel = pts - r0[None, :]
            x2 = np.stack([np.dot(rel, u), np.dot(rel, v)], axis=1)
            cov = np.cov(x2, rowvar=False)
            eig = np.linalg.eigvalsh(cov)
            eig = np.sort(eig)
            ev1 = max(float(eig[-1]), 0.0)
            ev2 = max(float(eig[0]), 0.0)
            major_sd = float(np.sqrt(ev1))
            minor_sd = float(np.sqrt(ev2))
            row["major_sd_mm"] = major_sd
            row["minor_sd_mm"] = minor_sd
            if minor_sd > 1e-8:
                row["elongation_ratio"] = float(major_sd / minor_sd)
            if major_sd > 1e-8:
                row["compactness_ratio"] = float(minor_sd / major_sd)

        rows.append(row)
    return rows


def _centerline_tangent_qc(plane_normals: np.ndarray) -> Dict[str, Any]:
    n = np.asarray(plane_normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] < 2:
        return {
            "n_segments": 0,
            "mean_angle_deg": float("nan"),
            "p95_angle_deg": float("nan"),
            "max_angle_deg": float("nan"),
        }
    dots = np.sum(n[:-1] * n[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return {
        "n_segments": int(ang.size),
        "mean_angle_deg": float(np.mean(ang)) if ang.size > 0 else float("nan"),
        "p95_angle_deg": float(np.quantile(ang, 0.95)) if ang.size > 0 else float("nan"),
        "max_angle_deg": float(np.max(ang)) if ang.size > 0 else float("nan"),
    }


def _centerline_sign_qc(
    q_ref_time: np.ndarray,
    q_base_time: np.ndarray,
    q_sr_time: np.ndarray,
    baseline_label: str,
    sr_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eps = 1e-9
    for label, q in ((baseline_label, q_base_time), (sr_label, q_sr_time)):
        ref = np.asarray(q_ref_time, dtype=np.float64)
        tst = np.asarray(q, dtype=np.float64)
        valid = np.isfinite(ref) & np.isfinite(tst)
        if int(np.count_nonzero(valid)) == 0:
            rows.append(
                {
                    "method": str(label),
                    "n_frames": 0,
                    "pearson_r_vs_ref": float("nan"),
                    "pearson_p_vs_ref": float("nan"),
                    "sign_agreement_pct": float("nan"),
                    "flip_suspected": 0,
                }
            )
            continue
        ref_v = ref[valid]
        tst_v = tst[valid]
        signs_ok = (np.sign(ref_v + eps) == np.sign(tst_v + eps))
        cstats = _correlation_stats(ref_v, tst_v)
        corr = cstats.get("pearson_r", float("nan"))
        rows.append(
            {
                "method": str(label),
                "n_frames": int(ref_v.size),
                "pearson_r_vs_ref": float(corr),
                "pearson_p_vs_ref": float(cstats.get("pearson_p", float("nan"))),
                "sign_agreement_pct": float(100.0 * np.mean(signs_ok)),
                "flip_suspected": int(bool(np.isfinite(corr) and corr < -0.2)),
            }
        )
    return rows


def _centerline_error_pvalues(
    q_ref_curves: np.ndarray,
    q_base_curves: np.ndarray,
    q_sr_curves: np.ndarray,
    valid_sections: np.ndarray,
    peak_idx: int,
) -> Dict[str, float]:
    idx = np.asarray(valid_sections, dtype=np.int32)
    if idx.size == 0:
        idx = np.arange(q_ref_curves.shape[1], dtype=np.int32)

    # Peak-frame paired errors across sections
    ref_peak = np.asarray(q_ref_curves[peak_idx, idx], dtype=np.float64)
    base_peak = np.asarray(q_base_curves[peak_idx, idx], dtype=np.float64)
    sr_peak = np.asarray(q_sr_curves[peak_idx, idx], dtype=np.float64)
    e_base_peak = np.abs(base_peak - ref_peak)
    e_sr_peak = np.abs(sr_peak - ref_peak)
    p_peak = _wilcoxon_p(e_base_peak.tolist(), e_sr_peak.tolist())

    # Global paired errors across all frames/sections
    ref_all = np.asarray(q_ref_curves[:, idx], dtype=np.float64).reshape(-1)
    base_all = np.asarray(q_base_curves[:, idx], dtype=np.float64).reshape(-1)
    sr_all = np.asarray(q_sr_curves[:, idx], dtype=np.float64).reshape(-1)
    e_base_all = np.abs(base_all - ref_all)
    e_sr_all = np.abs(sr_all - ref_all)
    p_all = _wilcoxon_p(e_base_all.tolist(), e_sr_all.tolist())

    return {
        "peak_plane_abs_err_wilcoxon_p_baseline_vs_sr": float(p_peak),
        "all_planes_abs_err_wilcoxon_p_baseline_vs_sr": float(p_all),
        "n_peak_sections": int(np.isfinite(e_base_peak).sum() if e_base_peak.size > 0 else 0),
        "n_all_plane_samples": int(np.isfinite(e_base_all).sum() if e_base_all.size > 0 else 0),
    }


def _save_centerline_sections_figure(
    out_path: Path,
    mask_3d: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    plane_points_mm: np.ndarray,
    plane_normals: np.ndarray,
    slab_thickness_mm: float,
    section_rows: List[Dict[str, Any]],
    valid_plane_index: np.ndarray,
) -> None:
    m = np.asarray(mask_3d) > 0.5
    idx = np.argwhere(m)
    if idx.size == 0:
        return
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    coords_mm = (idx.astype(np.float64) + 0.5) * spacing[None, :]
    thickness = float(slab_thickness_mm) if float(slab_thickness_mm) > 0 else float(np.mean(spacing))
    half_t = 0.5 * thickness

    all_idx = np.arange(int(plane_points_mm.shape[0]), dtype=np.int32)
    valid = np.asarray(valid_plane_index, dtype=np.int32)
    base = valid if valid.size > 0 else all_idx
    if base.size == 0:
        return
    if base.size >= 3:
        sel_idx = np.asarray([base[0], base[base.size // 2], base[-1]], dtype=np.int32)
    elif base.size == 2:
        sel_idx = np.asarray([base[0], base[1]], dtype=np.int32)
    else:
        sel_idx = np.asarray([base[0]], dtype=np.int32)

    row_by_plane = {int(r["plane_index"]): r for r in section_rows}
    fig, axes = plt.subplots(1, int(sel_idx.size), figsize=(5.2 * int(sel_idx.size), 5), squeeze=False)
    for j, p in enumerate(sel_idx.tolist()):
        ax = axes[0, j]
        r0 = np.asarray(plane_points_mm[p], dtype=np.float64)
        n = np.asarray(plane_normals[p], dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            ax.set_title(f"Plane {int(p)} (invalid normal)")
            ax.axis("off")
            continue
        n = n / n_norm
        d = np.dot(coords_mm - r0[None, :], n)
        sel = np.abs(d) <= half_t
        pts = coords_mm[sel]
        if pts.shape[0] == 0:
            ax.set_title(f"Plane {int(p)} (no voxels)")
            ax.axis("off")
            continue
        u, v = _orthonormal_basis_from_normal(n)
        rel = pts - r0[None, :]
        x2 = np.dot(rel, u)
        y2 = np.dot(rel, v)
        ax.scatter(x2, y2, s=8, c="#9ca3af", alpha=0.7)
        ax.scatter([0.0], [0.0], s=38, c="#f59e0b", marker="x", label="centerline point")
        cen = np.mean(pts, axis=0)
        cen_rel = cen - r0
        ax.scatter([float(np.dot(cen_rel, u))], [float(np.dot(cen_rel, v))], s=28, c="#10b981", marker="+", label="section centroid")
        rr = row_by_plane.get(int(p), {})
        el = rr.get("elongation_ratio", float("nan"))
        off = rr.get("centroid_offset_norm", float("nan"))
        ax.set_title(f"Plane {int(p)}  n={int(pts.shape[0])}\nelong={el:.2f}  off/r={off:.2f}")
        ax.set_xlabel("u [mm]")
        ax.set_ylabel("v [mm]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Centerline plane sections (proximal / middle / distal)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_centerline_peak_qs_figure(
    out_path: Path,
    q_ref_curves: np.ndarray,
    q_base_curves: np.ndarray,
    q_sr_curves: np.ndarray,
    valid_sections: np.ndarray,
    peak_idx: int,
    ref_label: str,
    baseline_label: str,
    sr_label: str,
) -> None:
    q_ref = np.asarray(q_ref_curves[peak_idx], dtype=np.float64)
    q_base = np.asarray(q_base_curves[peak_idx], dtype=np.float64)
    q_sr = np.asarray(q_sr_curves[peak_idx], dtype=np.float64)
    idx = np.asarray(valid_sections, dtype=np.int32)
    if idx.size == 0:
        idx = np.arange(q_ref.shape[0], dtype=np.int32)
    x = idx.astype(np.int32)

    fig = plt.figure(figsize=(10, 4.8))
    plt.plot(x, q_ref[idx], marker="o", linewidth=2.0, label=ref_label)
    plt.plot(x, q_base[idx], marker="o", linewidth=1.8, label=baseline_label)
    plt.plot(x, q_sr[idx], marker="o", linewidth=1.8, label=sr_label)
    plt.xlabel("Centerline plane index")
    plt.ylabel("Flow rate [ml/s]")
    plt.title(f"Flow consistency along centerline at peak frame t={int(peak_idx)}")
    plt.grid(True, alpha=0.25, linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a professional uncertainty-quantification report from inference payload "
            "(full-volume prediction, visual inspection figures, and paper-style metrics)."
        )
    )
    parser.add_argument("--payload-npz", required=True, help="Path to analysis_payload.npz produced by run_sr_inference_case.py")
    parser.add_argument("--metadata-json", default="", help="Optional inference_metadata.json for richer report context")
    parser.add_argument("--out-dir", required=True, help="Output directory for report artifacts")

    parser.add_argument(
        "--flow-axis",
        type=str,
        default="auto",
        choices=["auto", "0", "1", "2"],
        help="Axis used for cross-sectional flow integration. Use 'auto' to select the best axis from reference flow consistency.",
    )
    parser.add_argument(
        "--flow-method",
        type=str,
        default="axis",
        choices=["axis", "centerline"],
        help="Flow integration mode: axis-based slices (legacy) or centerline-based orthogonal planes.",
    )
    parser.add_argument("--selected-frame", type=int, default=0, help="Frame index (within payload) used for visual panel")
    parser.add_argument("--max-display-slices", type=int, default=8, help="Max slices per visual panel")
    parser.add_argument("--panel-cols", type=int, default=4, help="Number of columns per visual panel block")
    parser.add_argument("--hist-bins", type=int, default=120, help="Bins for in-mask voxel distribution histograms")
    parser.add_argument(
        "--lr-mag-channel",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Which LR magnitude channel to display for 'mag' input (0=u, 1=v, 2=w).",
    )
    parser.add_argument(
        "--mask-min-slice-voxels",
        type=int,
        default=25,
        help="Min aggregated in-mask voxels per slice across all processed frames for slice-wise stats",
    )
    parser.add_argument(
        "--centerline-mask-mode",
        type=str,
        default="union",
        choices=["union", "intersection", "frame"],
        help="How to derive the 3D mask used for centerline extraction from temporal masks.",
    )
    parser.add_argument("--centerline-mask-frame-index", type=int, default=0, help="Frame index used when --centerline-mask-mode frame")
    parser.add_argument("--centerline-keep-components", type=int, default=1, help="Connected components kept after centerline mask cleanup")
    parser.add_argument("--centerline-closing-iters", type=int, default=1, help="Morphological closing iterations before skeletonization")
    parser.add_argument("--centerline-smooth-window", type=int, default=5, help="Moving-average window (voxels) used to smooth centerline path")
    parser.add_argument("--centerline-n-planes", type=int, default=7, help="Number of orthogonal planes sampled along centerline")
    parser.add_argument(
        "--centerline-slab-thickness-mm",
        type=float,
        default=0.0,
        help="Slab thickness used to integrate flow around each centerline plane (<=0 uses mean voxel spacing).",
    )
    parser.add_argument("--centerline-min-plane-voxels", type=int, default=10, help="Minimum in-mask voxels in a centerline slab to accept plane/frame flow")
    parser.add_argument("--centerline-min-valid-support", type=int, default=10, help="Minimum total support across time to keep a centerline plane")
    parser.add_argument("--centerline-aggregate", type=str, default="median", choices=["mean", "median"], help="Temporal aggregation across valid centerline planes")
    parser.add_argument(
        "--centerline-start-xyz",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional centerline start voxel (HR xyz). Use together with --centerline-end-xyz.",
    )
    parser.add_argument(
        "--centerline-end-xyz",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional centerline end voxel (HR xyz). Use together with --centerline-start-xyz.",
    )

    parser.add_argument("--q-ref", type=float, default=float("nan"), help="Reference flow rate in ml/s (paper uses calibrated reference).")
    parser.add_argument("--cca-range", type=str, default="", help="Optional temporal frame window start:end for flow stats.")

    parser.add_argument("--mu-pa-s", type=float, default=0.0035, help="Dynamic viscosity for WSS estimation (Pa*s)")
    parser.add_argument("--max-wall-points", type=int, default=30000, help="Max wall points sampled for WSS distribution")
    parser.add_argument(
        "--include-wss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable WSS metrics and plots. Default is disabled until WSS validation is finalized.",
    )
    parser.add_argument(
        "--roi-bbox",
        type=int,
        nargs=6,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="Optional ROI bbox in HR voxel indices [x0,x1,y0,y1,z0,z1) to restrict mask-based metrics.",
    )
    parser.add_argument(
        "--roi-json",
        default="",
        help="Optional ROI JSON (from interactive selector). Supports keys: bbox_hr_xyz / bbox_xyz.",
    )

    parser.add_argument("--baseline-label", default="3T", help="Label for baseline (native/LR) method")
    parser.add_argument("--sr-label", default="3T SR", help="Label for super-resolved method")
    parser.add_argument("--ref-label", default="auto", help="Label for reference method. Use 'auto' to infer from metadata.")
    parser.add_argument("--report-title", default="4D Flow SR Uncertainty Quantification Report", help="Report title")

    args = parser.parse_args()
    if (args.centerline_start_xyz is None) != (args.centerline_end_xyz is None):
        raise ValueError("Use --centerline-start-xyz and --centerline-end-xyz together, or omit both.")

    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(args.payload_npz)
    metadata = {}
    if args.metadata_json:
        mpath = Path(args.metadata_json)
        if mpath.exists():
            metadata = json.loads(mpath.read_text())

    ref_label = args.ref_label
    if str(ref_label).lower() == "auto":
        ref_label = _autodetect_reference_label(metadata)

    lr_norm = payload["lr_norm"].astype(np.float32)  # [T,6,X,Y,Z]
    pred_norm = payload["pred_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    gt_norm = payload["gt_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    mask = payload["mask"].astype(np.float32)  # [T,X,Y,Z]
    venc = payload["venc"].astype(np.float32)  # [T]
    hr_spacing = tuple(float(x) for x in payload["hr_spacing"].tolist())

    if pred_norm.shape[1] != 4 or gt_norm.shape[1] != 4:
        raise ValueError(f"Expected 4-channel pred/gt tensors. Got pred={pred_norm.shape}, gt={gt_norm.shape}")

    t_count = pred_norm.shape[0]
    if "frame_indices" in payload and np.asarray(payload["frame_indices"]).shape[0] == t_count:
        frame_source_indices = np.asarray(payload["frame_indices"], dtype=np.int32)
    else:
        frame_source_indices = np.arange(t_count, dtype=np.int32)
    fidx = int(np.clip(args.selected_frame, 0, t_count - 1))

    # Denormalize to physical units for velocity-related metrics
    pred_phys = pred_norm.copy()
    gt_phys = gt_norm.copy()
    lr_vel_phys = lr_norm[:, :3].copy()
    for t in range(t_count):
        pred_phys[t, :3] *= float(venc[t])
        gt_phys[t, :3] *= float(venc[t])
        lr_vel_phys[t] *= float(venc[t])

    # If LR baseline is spatially smaller (downsampled input), upsample to HR grid for fair metric comparison.
    if lr_vel_phys.shape[2:] != gt_phys.shape[2:]:
        lr_vel_phys_metrics = _upsample_spatial(lr_vel_phys, out_shape_xyz=tuple(int(v) for v in gt_phys.shape[2:]), mode="trilinear")
    else:
        lr_vel_phys_metrics = lr_vel_phys

    # Optional ROI bbox to restrict mask-based metrics.
    roi_bbox = _resolve_roi_bbox(
        roi_bbox_cli=args.roi_bbox,
        roi_json_path=str(args.roi_json),
        shape_xyz=tuple(int(v) for v in mask.shape[1:]),
    )
    mask_metrics, roi_info = _apply_roi_to_mask(mask_txyz=mask, bbox_xyz=roi_bbox)
    if int((mask_metrics > 0.5).sum()) == 0:
        raise ValueError("Selected ROI produced an empty in-mask region. Please adjust ROI bbox.")

    # Derive LR 4-channel display tensor (u,v,w,mag) using a single LR magnitude channel.
    lr_mag_single = lr_norm[:, 3 + int(args.lr_mag_channel)].astype(np.float32)
    lr_4ch = np.concatenate([lr_norm[:, :3], lr_mag_single[:, None]], axis=1).astype(np.float32)

    # Suggest best flow axis from reference consistency, then resolve selected axis.
    suggested_flow_axis, flow_axis_scores = _suggest_flow_axis(
        vel_ref=gt_phys[:, :3],
        mask=mask_metrics,
        spacing_mm=hr_spacing,
    )
    selected_flow_axis = int(suggested_flow_axis) if args.flow_axis == "auto" else int(args.flow_axis)

    # Upsample LR display to HR shape for visual side-by-side
    if lr_4ch.shape[2:] != gt_norm.shape[2:]:
        lr_t = torch.from_numpy(lr_4ch)
        lr_up = torch.nn.functional.interpolate(
            lr_t,
            size=gt_norm.shape[2:],
            mode="trilinear",
            align_corners=False,
        ).numpy()
    else:
        lr_up = lr_4ch

    # 1) Visual panels
    channel_names = ["u", "v", "w", "mag"]
    channel_figs: Dict[str, str] = {}
    for c, name in enumerate(channel_names):
        out_img = fig_dir / f"channel_{name}_comparison.png"
        _save_channel_figure(
            out_path=out_img,
            ch_name=name,
            lr_up=lr_up[fidx, c],
            pred=pred_norm[fidx, c],
            gt=gt_norm[fidx, c],
            max_slices=int(args.max_display_slices),
            n_cols=int(args.panel_cols),
            bbox_xyz=roi_bbox,
        )
        channel_figs[name] = str(out_img.name)

    fig_voxel_hist = fig_dir / "voxel_histogram_in_mask.png"
    voxel_dist_rows = _save_voxel_histograms(
        out_path=fig_voxel_hist,
        baseline_4ch=lr_up,
        pred_4ch=pred_norm,
        gt_4ch=gt_norm,
        mask=mask_metrics,
        bins=int(args.hist_bins),
        baseline_label=args.baseline_label,
        sr_label=args.sr_label,
        ref_label=ref_label,
    )
    voxel_dist_cols = ["channel", "method", "count", "mean", "std", "median", "p05", "p95", "min", "max"]
    _write_csv(metrics_dir / "voxel_distribution_stats.csv", voxel_dist_rows, voxel_dist_cols)

    # Multi-frame fields for paper-style stats
    mask_ref = (mask_metrics > 0.5).astype(np.float32)
    spacing_m = tuple(float(s) / 1000.0 for s in hr_spacing)
    speed_ref = np.sqrt((gt_phys[:, :3] ** 2).sum(axis=1)).astype(np.float32)  # [T,X,Y,Z]
    speed_base = np.sqrt((lr_vel_phys_metrics**2).sum(axis=1)).astype(np.float32)  # [T,X,Y,Z]
    speed_sr = np.sqrt((pred_phys[:, :3] ** 2).sum(axis=1)).astype(np.float32)  # [T,X,Y,Z]

    vort_ref = np.stack([_vorticity_magnitude(gt_phys[t, :3], spacing_m) for t in range(t_count)], axis=0)
    vort_base = np.stack([_vorticity_magnitude(lr_vel_phys_metrics[t], spacing_m) for t in range(t_count)], axis=0)
    vort_sr = np.stack([_vorticity_magnitude(pred_phys[t, :3], spacing_m) for t in range(t_count)], axis=0)

    table2_all, valid_slices, re_base_all, re_sr_all = _table2_rows(
        speed_ref=speed_ref,
        speed_base=speed_base,
        speed_sr=speed_sr,
        vort_ref=vort_ref,
        vort_base=vort_base,
        vort_sr=vort_sr,
        mask_ref=mask_ref,
        flow_axis=selected_flow_axis,
        min_voxels=int(args.mask_min_slice_voxels),
    )
    table2_per_frame = _table2_rows_per_frame(
        speed_ref=speed_ref,
        speed_base=speed_base,
        speed_sr=speed_sr,
        vort_ref=vort_ref,
        vort_base=vort_base,
        vort_sr=vort_sr,
        mask_ref=mask_ref,
        flow_axis=selected_flow_axis,
        min_voxels=int(args.mask_min_slice_voxels),
        frame_source_indices=frame_source_indices,
    )

    p_re = _wilcoxon_p(re_base_all, re_sr_all)

    # Build a Table-2-like compact subset (3 representative locations)
    pick_slices, pick_labels = _default_slice_triplet(valid_slices)
    label_map = {s: lab for s, lab in zip(pick_slices, pick_labels)}
    table2_compact = []
    for row in table2_all:
        s = int(row["slice_index"])
        if s in label_map:
            row2 = dict(row)
            row2["location"] = label_map[s]
            table2_compact.append(row2)

    # Save table2 CSVs
    t2_cols = ["slice_index", "variable", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    _write_csv(metrics_dir / "table2_like_all_slices.csv", table2_all, t2_cols)

    t2pf_cols = ["frame_payload_index", "frame_source_index", "slice_index", "variable", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    _write_csv(metrics_dir / "table2_like_per_frame_all_slices.csv", table2_per_frame, t2pf_cols)

    t2c_cols = ["location", "slice_index", "variable", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    _write_csv(metrics_dir / "table2_like_compact.csv", table2_compact, t2c_cols)

    # 2) Flow-rate metrics (temporal profile)
    flow_method = str(args.flow_method).strip().lower()
    flow_section_kind = "slice"
    flow_aggregate = "mean"
    flow_section_support = np.zeros((t_count, 0), dtype=np.int32)
    valid_flow_sections = np.zeros((0,), dtype=np.int32)
    centerline_bundle: Optional[Dict[str, Any]] = None
    centerline_summary: Dict[str, Any] = {}
    centerline_planes_rows: List[Dict[str, Any]] = []
    centerline_overlay_name = ""
    centerline_section_qc_rows: List[Dict[str, Any]] = []
    centerline_sign_rows: List[Dict[str, Any]] = []
    centerline_error_p: Dict[str, Any] = {}
    centerline_sections_name = ""
    centerline_peak_qs_name = ""
    centerline_3d_name = ""
    slab_thickness_mm = float("nan")

    if flow_method == "centerline":
        centerline_bundle = _build_centerline_bundle(
            mask_txyz=mask_metrics,
            spacing_mm=hr_spacing,
            mask_mode=str(args.centerline_mask_mode),
            mask_frame_index=int(args.centerline_mask_frame_index),
            keep_components=int(args.centerline_keep_components),
            closing_iters=int(args.centerline_closing_iters),
            smooth_window=int(args.centerline_smooth_window),
            n_planes=int(args.centerline_n_planes),
            start_xyz=args.centerline_start_xyz,
            end_xyz=args.centerline_end_xyz,
        )
        slab_thickness_mm = float(args.centerline_slab_thickness_mm)
        if slab_thickness_mm <= 0:
            slab_thickness_mm = float(np.mean(np.asarray(hr_spacing, dtype=np.float64)))

        q_ref_curves, support_ref = _flow_rate_curves_centerline(
            vel=gt_phys[:, :3],
            mask=mask_metrics,
            spacing_mm=hr_spacing,
            plane_points_mm=centerline_bundle["plane_points_mm"],
            plane_normals=centerline_bundle["plane_normals"],
            slab_thickness_mm=slab_thickness_mm,
            min_plane_voxels=int(args.centerline_min_plane_voxels),
        )
        q_base_curves, support_base = _flow_rate_curves_centerline(
            vel=lr_vel_phys_metrics,
            mask=mask_metrics,
            spacing_mm=hr_spacing,
            plane_points_mm=centerline_bundle["plane_points_mm"],
            plane_normals=centerline_bundle["plane_normals"],
            slab_thickness_mm=slab_thickness_mm,
            min_plane_voxels=int(args.centerline_min_plane_voxels),
        )
        q_sr_curves, support_sr = _flow_rate_curves_centerline(
            vel=pred_phys[:, :3],
            mask=mask_metrics,
            spacing_mm=hr_spacing,
            plane_points_mm=centerline_bundle["plane_points_mm"],
            plane_normals=centerline_bundle["plane_normals"],
            slab_thickness_mm=slab_thickness_mm,
            min_plane_voxels=int(args.centerline_min_plane_voxels),
        )
        flow_section_support = np.minimum(np.minimum(support_ref, support_base), support_sr).astype(np.int32)
        flow_aggregate = str(args.centerline_aggregate)
        q_ref_time, valid_flow_sections = _temporal_flow_from_sections(
            q_curves=q_ref_curves,
            section_support=flow_section_support,
            min_support=int(args.centerline_min_valid_support),
            aggregate=flow_aggregate,
        )
        q_base_time = _aggregate_flow_sections(q_base_curves, valid_flow_sections, aggregate=flow_aggregate)
        q_sr_time = _aggregate_flow_sections(q_sr_curves, valid_flow_sections, aggregate=flow_aggregate)
        flow_section_kind = "plane"

        centerline_summary = {
            "path_mode": str(centerline_bundle["path_mode"]),
            "n_path_points": int(centerline_bundle["path_vox"].shape[0]),
            "n_path_points_smoothed": int(centerline_bundle["path_smooth_vox"].shape[0]),
            "n_planes": int(centerline_bundle["plane_points_mm"].shape[0]),
            "n_valid_planes": int(valid_flow_sections.size),
            "mask_mode": str(args.centerline_mask_mode),
            "mask_frame_index": int(args.centerline_mask_frame_index),
            "closing_iters": int(args.centerline_closing_iters),
            "keep_components": int(args.centerline_keep_components),
            "slab_thickness_mm": float(slab_thickness_mm),
            "aggregate": str(flow_aggregate),
        }
    else:
        q_ref_curves = _flow_rate_curves(gt_phys[:, :3], mask_metrics, flow_axis=selected_flow_axis, spacing_mm=hr_spacing)
        q_base_curves = _flow_rate_curves(lr_vel_phys_metrics, mask_metrics, flow_axis=selected_flow_axis, spacing_mm=hr_spacing)
        q_sr_curves = _flow_rate_curves(pred_phys[:, :3], mask_metrics, flow_axis=selected_flow_axis, spacing_mm=hr_spacing)
        flow_section_support = _slice_voxel_counts_by_axis(mask_metrics, flow_axis=selected_flow_axis)
        q_ref_time, valid_flow_sections = _temporal_flow_from_slices(
            q_curves=q_ref_curves,
            slice_counts=flow_section_support,
            min_voxels=int(args.mask_min_slice_voxels),
        )
        q_base_time = _aggregate_flow_sections(q_base_curves, valid_flow_sections, aggregate=flow_aggregate)
        q_sr_time = _aggregate_flow_sections(q_sr_curves, valid_flow_sections, aggregate=flow_aggregate)

    if not np.isfinite(q_ref_time).any():
        raise ValueError("Flow integration produced no finite reference temporal samples. Adjust ROI/flow settings.")
    for q_arr in (q_ref_time, q_base_time, q_sr_time):
        finite = np.isfinite(q_arr)
        if finite.any():
            q_arr[~finite] = float(np.nanmedian(q_arr[finite]))

    if np.isfinite(float(args.q_ref)):
        q_ref_scalar = float(args.q_ref)
    else:
        q_ref_scalar = float(np.median(q_ref_time[np.isfinite(q_ref_time)]))

    if q_ref_time.size > 0:
        q_ref_abs = np.abs(np.asarray(q_ref_time, dtype=np.float64))
        if np.isfinite(q_ref_abs).any():
            peak_idx = int(np.nanargmax(q_ref_abs))
        else:
            peak_idx = 0
    else:
        peak_idx = 0
    peak_frame_src = int(frame_source_indices[peak_idx]) if frame_source_indices.size > peak_idx else peak_idx

    def flow_summary(name: str, q_time: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
        q_sel = q_time[idx]
        mad = float(np.mean(np.abs(q_sel - q_ref_scalar)))
        return {
            "method": name,
            "mean_Q_ml_s": float(np.mean(q_sel)),
            "temporal_SD_Q_ml_s": float(np.std(q_sel, ddof=1)) if q_sel.size > 1 else float("nan"),
            "MAD_Q_vs_qref_ml_s": mad,
            "MAD_Q_vs_qref_pct": 100.0 * mad / (abs(q_ref_scalar) + 1e-12),
        }

    all_idx = np.arange(q_ref_time.shape[0], dtype=np.int32)
    flow_rows = [
        flow_summary(ref_label, q_ref_time, all_idx),
        flow_summary(args.baseline_label, q_base_time, all_idx),
        flow_summary(args.sr_label, q_sr_time, all_idx),
    ]

    if args.cca_range:
        try:
            a, b = args.cca_range.split(":")
            a_i, b_i = int(a), int(b)
            win_idx = np.arange(max(0, a_i), min(q_ref_time.shape[0], b_i), dtype=np.int32)
            if win_idx.size > 0:
                flow_rows.extend(
                    [
                        {**flow_summary(ref_label, q_ref_time, win_idx), "method": f"{ref_label} (window)"},
                        {**flow_summary(args.baseline_label, q_base_time, win_idx), "method": f"{args.baseline_label} (window)"},
                        {**flow_summary(args.sr_label, q_sr_time, win_idx), "method": f"{args.sr_label} (window)"},
                    ]
                )
        except Exception:
            pass

    # Flow p-value comparing temporal absolute errors vs per-frame reference profile
    flow_err_base = np.abs(q_base_time - q_ref_time)
    flow_err_sr = np.abs(q_sr_time - q_ref_time)
    p_flow = _wilcoxon_p(flow_err_base.tolist(), flow_err_sr.tolist())

    flow_cols = ["method", "mean_Q_ml_s", "temporal_SD_Q_ml_s", "MAD_Q_vs_qref_ml_s", "MAD_Q_vs_qref_pct"]
    _write_csv(metrics_dir / "flow_metrics.csv", flow_rows, flow_cols)

    flow_time_rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        flow_time_rows.append(
            {
                "frame_payload_index": int(t),
                "frame_source_index": int(frame_source_indices[t]),
                "flow_method": str(flow_method),
                "q_ref_ml_s": float(q_ref_time[t]),
                "q_baseline_ml_s": float(q_base_time[t]),
                "q_sr_ml_s": float(q_sr_time[t]),
                "abs_err_baseline_vs_ref_ml_s": float(abs(q_base_time[t] - q_ref_time[t])),
                "abs_err_sr_vs_ref_ml_s": float(abs(q_sr_time[t] - q_ref_time[t])),
                "abs_err_baseline_vs_qref_ml_s": float(abs(q_base_time[t] - q_ref_scalar)),
                "abs_err_sr_vs_qref_ml_s": float(abs(q_sr_time[t] - q_ref_scalar)),
                "n_sections_aggregated": int(valid_flow_sections.size),
            }
        )
    flow_time_cols = [
        "frame_payload_index",
        "frame_source_index",
        "flow_method",
        "q_ref_ml_s",
        "q_baseline_ml_s",
        "q_sr_ml_s",
        "abs_err_baseline_vs_ref_ml_s",
        "abs_err_sr_vs_ref_ml_s",
        "abs_err_baseline_vs_qref_ml_s",
        "abs_err_sr_vs_qref_ml_s",
        "n_sections_aggregated",
    ]
    _write_csv(metrics_dir / "flow_rate_curves_per_frame.csv", flow_time_rows, flow_time_cols)

    if centerline_bundle is not None:
        raw_pts = np.asarray(centerline_bundle["path_vox"], dtype=np.float32)
        smooth_pts = np.asarray(centerline_bundle["path_smooth_vox"], dtype=np.float32)
        pcount = min(raw_pts.shape[0], smooth_pts.shape[0])
        centerline_point_rows: List[Dict[str, Any]] = []
        for i in range(pcount):
            centerline_point_rows.append(
                {
                    "point_index": int(i),
                    "raw_x_vox": float(raw_pts[i, 0]),
                    "raw_y_vox": float(raw_pts[i, 1]),
                    "raw_z_vox": float(raw_pts[i, 2]),
                    "smooth_x_vox": float(smooth_pts[i, 0]),
                    "smooth_y_vox": float(smooth_pts[i, 1]),
                    "smooth_z_vox": float(smooth_pts[i, 2]),
                    "smooth_x_mm": float(smooth_pts[i, 0] * float(hr_spacing[0])),
                    "smooth_y_mm": float(smooth_pts[i, 1] * float(hr_spacing[1])),
                    "smooth_z_mm": float(smooth_pts[i, 2] * float(hr_spacing[2])),
                }
            )
        _write_csv(
            metrics_dir / "centerline_points.csv",
            centerline_point_rows,
            [
                "point_index",
                "raw_x_vox",
                "raw_y_vox",
                "raw_z_vox",
                "smooth_x_vox",
                "smooth_y_vox",
                "smooth_z_vox",
                "smooth_x_mm",
                "smooth_y_mm",
                "smooth_z_mm",
            ],
        )

        plane_points_mm = np.asarray(centerline_bundle["plane_points_mm"], dtype=np.float32)
        plane_normals = np.asarray(centerline_bundle["plane_normals"], dtype=np.float32)
        valid_plane_set = set(int(v) for v in valid_flow_sections.tolist())
        for t in range(t_count):
            for p in range(int(plane_points_mm.shape[0])):
                centerline_planes_rows.append(
                    {
                        "frame_payload_index": int(t),
                        "frame_source_index": int(frame_source_indices[t]),
                        "plane_index": int(p),
                        "is_valid_plane": int(p in valid_plane_set),
                        "support_voxels": int(flow_section_support[t, p]),
                        "plane_x_mm": float(plane_points_mm[p, 0]),
                        "plane_y_mm": float(plane_points_mm[p, 1]),
                        "plane_z_mm": float(plane_points_mm[p, 2]),
                        "normal_x": float(plane_normals[p, 0]),
                        "normal_y": float(plane_normals[p, 1]),
                        "normal_z": float(plane_normals[p, 2]),
                        "q_ref_ml_s": float(q_ref_curves[t, p]),
                        "q_baseline_ml_s": float(q_base_curves[t, p]),
                        "q_sr_ml_s": float(q_sr_curves[t, p]),
                    }
                )
        _write_csv(
            metrics_dir / "flow_centerline_planes_per_frame.csv",
            centerline_planes_rows,
            [
                "frame_payload_index",
                "frame_source_index",
                "plane_index",
                "is_valid_plane",
                "support_voxels",
                "plane_x_mm",
                "plane_y_mm",
                "plane_z_mm",
                "normal_x",
                "normal_y",
                "normal_z",
                "q_ref_ml_s",
                "q_baseline_ml_s",
                "q_sr_ml_s",
            ],
        )
        fig_centerline = fig_dir / "centerline_overlay.png"
        _save_centerline_overlay_figure(
            out_path=fig_centerline,
            mask_3d=centerline_bundle["mask_3d"],
            centerline_vox=centerline_bundle["path_smooth_vox"],
            plane_points_vox=centerline_bundle["plane_points_vox"],
            valid_plane_index=valid_flow_sections,
        )
        if fig_centerline.exists():
            centerline_overlay_name = fig_centerline.name
        fig_centerline_3d = fig_dir / "centerline_3d.png"
        _save_centerline_3d_figure(
            out_path=fig_centerline_3d,
            mask_3d=centerline_bundle["mask_3d"],
            spacing_mm=hr_spacing,
            centerline_vox=centerline_bundle["path_smooth_vox"],
            plane_points_vox=centerline_bundle["plane_points_vox"],
            valid_plane_index=valid_flow_sections,
        )
        if fig_centerline_3d.exists():
            centerline_3d_name = fig_centerline_3d.name

        centerline_section_qc_rows = _centerline_section_geometry_rows(
            mask_3d=centerline_bundle["mask_3d"],
            spacing_mm=hr_spacing,
            plane_points_mm=centerline_bundle["plane_points_mm"],
            plane_normals=centerline_bundle["plane_normals"],
            slab_thickness_mm=float(slab_thickness_mm),
        )
        valid_plane_set = set(int(v) for v in np.asarray(valid_flow_sections, dtype=np.int32).tolist())
        for r in centerline_section_qc_rows:
            r["is_valid_plane"] = int(int(r.get("plane_index", -1)) in valid_plane_set)
        if centerline_section_qc_rows:
            _write_csv(
                metrics_dir / "centerline_section_qc.csv",
                centerline_section_qc_rows,
                [
                    "plane_index",
                    "is_valid_plane",
                    "support_voxels",
                    "area_mm2",
                    "eq_radius_mm",
                    "centroid_offset_mm",
                    "centroid_offset_norm",
                    "major_sd_mm",
                    "minor_sd_mm",
                    "elongation_ratio",
                    "compactness_ratio",
                    "center_to_wall_mm",
                    "center_to_wall_over_eq_radius",
                ],
            )

        centerline_sign_rows = _centerline_sign_qc(
            q_ref_time=q_ref_time,
            q_base_time=q_base_time,
            q_sr_time=q_sr_time,
            baseline_label=args.baseline_label,
            sr_label=args.sr_label,
        )
        if centerline_sign_rows:
            _write_csv(
                metrics_dir / "centerline_sign_qc.csv",
                centerline_sign_rows,
                ["method", "n_frames", "pearson_r_vs_ref", "pearson_p_vs_ref", "sign_agreement_pct", "flip_suspected"],
            )
        centerline_error_p = _centerline_error_pvalues(
            q_ref_curves=q_ref_curves,
            q_base_curves=q_base_curves,
            q_sr_curves=q_sr_curves,
            valid_sections=valid_flow_sections,
            peak_idx=int(peak_idx),
        )

        fig_sections = fig_dir / "centerline_plane_sections.png"
        _save_centerline_sections_figure(
            out_path=fig_sections,
            mask_3d=centerline_bundle["mask_3d"],
            spacing_mm=hr_spacing,
            plane_points_mm=centerline_bundle["plane_points_mm"],
            plane_normals=centerline_bundle["plane_normals"],
            slab_thickness_mm=float(slab_thickness_mm),
            section_rows=centerline_section_qc_rows,
            valid_plane_index=valid_flow_sections,
        )
        if fig_sections.exists():
            centerline_sections_name = fig_sections.name

        fig_qs = fig_dir / "centerline_flow_along_vessel_peak.png"
        _save_centerline_peak_qs_figure(
            out_path=fig_qs,
            q_ref_curves=q_ref_curves,
            q_base_curves=q_base_curves,
            q_sr_curves=q_sr_curves,
            valid_sections=valid_flow_sections,
            peak_idx=int(peak_idx),
            ref_label=ref_label,
            baseline_label=args.baseline_label,
            sr_label=args.sr_label,
        )
        if fig_qs.exists():
            centerline_peak_qs_name = fig_qs.name

        tangent_qc = _centerline_tangent_qc(centerline_bundle["plane_normals"])
        valid_geom = [r for r in centerline_section_qc_rows if int(r.get("is_valid_plane", 0)) == 1]
        def _arr(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
            return np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)

        off_norm = _arr(valid_geom, "centroid_offset_norm")
        elong = _arr(valid_geom, "elongation_ratio")
        comp = _arr(valid_geom, "compactness_ratio")
        area = _arr(valid_geom, "area_mm2")
        wall_ratio = _arr(valid_geom, "center_to_wall_over_eq_radius")
        wall_mm_all = _arr(centerline_section_qc_rows, "center_to_wall_mm")
        q_ref_peak = np.asarray(q_ref_curves[peak_idx, valid_flow_sections], dtype=np.float64) if valid_flow_sections.size > 0 else np.asarray([], dtype=np.float64)

        def _nmed(x: np.ndarray) -> float:
            return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")

        centerline_summary.update(
            {
                "tangent_mean_angle_deg": float(tangent_qc.get("mean_angle_deg", float("nan"))),
                "tangent_p95_angle_deg": float(tangent_qc.get("p95_angle_deg", float("nan"))),
                "tangent_max_angle_deg": float(tangent_qc.get("max_angle_deg", float("nan"))),
                "qc_center_offset_norm_median": _nmed(off_norm),
                "qc_elongation_ratio_p95": float(np.nanquantile(elong, 0.95)) if np.isfinite(elong).any() else float("nan"),
                "qc_compactness_ratio_median": _nmed(comp),
                "qc_center_to_wall_over_eq_radius_median": _nmed(wall_ratio),
                "qc_plane_points_inside_mask_pct": float(100.0 * np.mean(np.isfinite(wall_mm_all))) if wall_mm_all.size > 0 else float("nan"),
                "qc_area_cv_valid_planes": float(np.nanstd(area, ddof=1) / (abs(np.nanmean(area)) + 1e-12)) if np.isfinite(area).sum() > 1 else float("nan"),
                "qc_peak_q_cv_ref": float(np.nanstd(q_ref_peak, ddof=1) / (abs(np.nanmean(q_ref_peak)) + 1e-12)) if np.isfinite(q_ref_peak).sum() > 1 else float("nan"),
                "qc_peak_q_range_pct_ref": float((np.nanquantile(q_ref_peak, 0.95) - np.nanquantile(q_ref_peak, 0.05)) / (abs(np.nanmedian(q_ref_peak)) + 1e-12) * 100.0)
                if np.isfinite(q_ref_peak).sum() > 2
                else float("nan"),
                "qc_peak_plane_error_n": int(centerline_error_p.get("n_peak_sections", 0)),
                "qc_all_plane_error_n": int(centerline_error_p.get("n_all_plane_samples", 0)),
                "qc_peak_plane_abs_err_wilcoxon_p_baseline_vs_sr": float(centerline_error_p.get("peak_plane_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))),
                "qc_all_planes_abs_err_wilcoxon_p_baseline_vs_sr": float(centerline_error_p.get("all_planes_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))),
            }
        )
        if centerline_sign_rows:
            centerline_summary["sign_qc"] = centerline_sign_rows

    flow_per_frame_rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        ref_t = float(q_ref_time[t])
        for method_name, q_val in ((ref_label, ref_t), (args.baseline_label, float(q_base_time[t])), (args.sr_label, float(q_sr_time[t]))):
            flow_per_frame_rows.append(
                {
                    "frame_payload_index": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "method": method_name,
                    "flow_method": str(flow_method),
                    "Q_ml_s": float(q_val),
                    "abs_err_vs_qref_ml_s": float(abs(q_val - q_ref_scalar)),
                    "abs_err_vs_ref_profile_ml_s": float(abs(q_val - ref_t)),
                }
            )
    flow_per_frame_cols = [
        "frame_payload_index",
        "frame_source_index",
        "method",
        "flow_method",
        "Q_ml_s",
        "abs_err_vs_qref_ml_s",
        "abs_err_vs_ref_profile_ml_s",
    ]
    _write_csv(metrics_dir / "flow_metrics_per_frame.csv", flow_per_frame_rows, flow_per_frame_cols)

    # Flow figure (temporal)
    fig_flow = fig_dir / "flow_rate_profile.png"
    x_time = frame_source_indices.astype(np.int32)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_time, q_ref_time, label=ref_label, linewidth=2)
    plt.plot(x_time, q_base_time, label=args.baseline_label, linewidth=2)
    plt.plot(x_time, q_sr_time, label=args.sr_label, linewidth=2)
    plt.axhline(q_ref_scalar, linestyle="--", color="black", label=f"Qref={q_ref_scalar:.3f} ml/s")
    plt.xlabel("Temporal frame index")
    plt.ylabel("Flow rate [ml/s]")
    if flow_method == "centerline":
        flow_title = (
            f"Temporal flow profile (centerline, {int(valid_flow_sections.size)} valid planes, agg={flow_aggregate})"
            f"{' [ROI bbox]' if roi_bbox is not None else ''}"
        )
    else:
        flow_title = (
            f"Temporal flow profile (axis {selected_flow_axis}, aggregated over {int(valid_flow_sections.size)} slices)"
            f"{' [ROI bbox]' if roi_bbox is not None else ''}"
        )
    plt.title(
        flow_title
    )
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_flow, dpi=180)
    plt.close(fig)

    # 3) WSS statistics (optional)
    table3_rows: List[Dict[str, Any]] = []
    table3_per_frame_rows: List[Dict[str, Any]] = []
    p_wss = float("nan")
    fig_wss_name = ""

    if bool(args.include_wss):
        tau_ref_parts: List[np.ndarray] = []
        tau_base_parts: List[np.ndarray] = []
        tau_sr_parts: List[np.ndarray] = []
        wss_metric_names = ["Maximum", "Mean", "SD", "Quantile_97_5", "Median", "Quantile_2_5", "IQR_75_25"]

        for t in range(t_count):
            mask_t = (mask_metrics[t] > 0.5).astype(np.float32)
            if int(mask_t.sum()) < 25:
                continue

            tau_ref_t = _compute_wss_distribution(
                uvw_mean=gt_phys[t, :3],
                mask_ref=mask_t,
                spacing_mm=hr_spacing,
                mu_pa_s=float(args.mu_pa_s),
                max_points=int(args.max_wall_points),
                seed=7 + t,
            )
            tau_base_t = _compute_wss_distribution(
                uvw_mean=lr_vel_phys_metrics[t],
                mask_ref=mask_t,
                spacing_mm=hr_spacing,
                mu_pa_s=float(args.mu_pa_s),
                max_points=int(args.max_wall_points),
                seed=7 + t,
            )
            tau_sr_t = _compute_wss_distribution(
                uvw_mean=pred_phys[t, :3],
                mask_ref=mask_t,
                spacing_mm=hr_spacing,
                mu_pa_s=float(args.mu_pa_s),
                max_points=int(args.max_wall_points),
                seed=7 + t,
            )

            if tau_ref_t.size > 0 and tau_base_t.size > 0 and tau_sr_t.size > 0:
                tau_ref_parts.append(tau_ref_t)
                tau_base_parts.append(tau_base_t)
                tau_sr_parts.append(tau_sr_t)

                wss_ref_t = _wss_summary(tau_ref_t)
                wss_base_t = _wss_summary(tau_base_t)
                wss_sr_t = _wss_summary(tau_sr_t)
                for key in wss_metric_names:
                    ref_v_t = wss_ref_t[key]
                    base_v_t = wss_base_t[key]
                    sr_v_t = wss_sr_t[key]
                    table3_per_frame_rows.append(
                        {
                            "frame_payload_index": int(t),
                            "frame_source_index": int(frame_source_indices[t]),
                            "metric": key,
                            "ref": ref_v_t,
                            "baseline": base_v_t,
                            "sr": sr_v_t,
                            "re_baseline": _relative_error(base_v_t, ref_v_t),
                            "re_sr": _relative_error(sr_v_t, ref_v_t),
                            "n_ref": int(tau_ref_t.size),
                            "n_baseline": int(tau_base_t.size),
                            "n_sr": int(tau_sr_t.size),
                        }
                    )

        tau_ref = np.concatenate(tau_ref_parts, axis=0) if tau_ref_parts else np.zeros((0,), dtype=np.float64)
        tau_base = np.concatenate(tau_base_parts, axis=0) if tau_base_parts else np.zeros((0,), dtype=np.float64)
        tau_sr = np.concatenate(tau_sr_parts, axis=0) if tau_sr_parts else np.zeros((0,), dtype=np.float64)

        wss_ref = _wss_summary(tau_ref)
        wss_base = _wss_summary(tau_base)
        wss_sr = _wss_summary(tau_sr)

        for key in wss_metric_names:
            ref_v = wss_ref[key]
            base_v = wss_base[key]
            sr_v = wss_sr[key]
            table3_rows.append(
                {
                    "metric": key,
                    "ref": ref_v,
                    "baseline": base_v,
                    "sr": sr_v,
                    "re_baseline": _relative_error(base_v, ref_v),
                    "re_sr": _relative_error(sr_v, ref_v),
                }
            )

        _write_csv(metrics_dir / "table3_like_wss.csv", table3_rows, ["metric", "ref", "baseline", "sr", "re_baseline", "re_sr"])
        _write_csv(
            metrics_dir / "table3_like_wss_per_frame.csv",
            table3_per_frame_rows,
            [
                "frame_payload_index",
                "frame_source_index",
                "metric",
                "ref",
                "baseline",
                "sr",
                "re_baseline",
                "re_sr",
                "n_ref",
                "n_baseline",
                "n_sr",
            ],
        )

        # WSS p-value comparing pointwise absolute errors (paired by sampling seed/order)
        n_pair = min(tau_ref.size, tau_base.size, tau_sr.size)
        if n_pair >= 10:
            e_base = np.abs(tau_base[:n_pair] - tau_ref[:n_pair])
            e_sr = np.abs(tau_sr[:n_pair] - tau_ref[:n_pair])
            p_wss = _wilcoxon_p(e_base.tolist(), e_sr.tolist())
        else:
            p_wss = float("nan")

        fig_wss = fig_dir / "wss_distribution.png"
        fig = plt.figure(figsize=(9, 5))
        bins = 80
        if tau_ref.size > 0:
            plt.hist(tau_ref, bins=bins, alpha=0.4, density=True, label=ref_label)
        if tau_base.size > 0:
            plt.hist(tau_base, bins=bins, alpha=0.4, density=True, label=args.baseline_label)
        if tau_sr.size > 0:
            plt.hist(tau_sr, bins=bins, alpha=0.4, density=True, label=args.sr_label)
        plt.xlabel("WSS [Pa]")
        plt.ylabel("Density")
        plt.title("Wall shear stress distribution (boundary samples)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(fig_wss, dpi=180)
        plt.close(fig)
        fig_wss_name = fig_wss.name

    # 4) Geometry metrics (paper-like) from temporal mask variability
    geom_rows: List[Dict[str, Any]] = []
    geom_status = "not_available_single_frame"
    geom_note = "Computed across temporal masks when multiple frames are available."
    if mask_metrics.shape[0] > 1:
        mask_u8 = (mask_metrics > 0.5).astype(np.uint8)
        ref_mask = mask_u8[0]
        static_mask = bool(np.all(mask_u8 == ref_mask[None, ...]))
        if static_mask:
            geom_status = "not_applicable_static_mask"
            geom_note = (
                "Temporal geometry metrics marked as N/A because all frames share the same binary mask "
                "(e.g., registered sequence with frame-0 mask propagated)."
            )
            for t in range(mask_u8.shape[0]):
                geom_rows.append(
                    {
                        "frame": int(t),
                        "frame_source_index": int(frame_source_indices[t]),
                        "status": geom_status,
                        "mean_surface_distance_a_to_b_mm": float("nan"),
                        "std_surface_distance_a_to_b_mm": float("nan"),
                        "symmetric_mean_surface_distance_mm": float("nan"),
                        "hausdorff_distance_mm": float("nan"),
                    }
                )
            geom_summary = {
                "mean_surface_distance_mm": float("nan"),
                "std_surface_distance_mm": float("nan"),
                "mean_hausdorff_distance_mm": float("nan"),
            }
        else:
            geom_status = "computed"
            geom_note = "Computed from frame-wise mask surfaces against frame 0."
            per_frame = []
            for t in range(mask_u8.shape[0]):
                m_t = mask_u8[t]
                m = _surface_distance_metrics(m_t, ref_mask, hr_spacing)
                m["frame"] = int(t)
                m["frame_source_index"] = int(frame_source_indices[t])
                m["status"] = geom_status
                per_frame.append(m)

            for item in per_frame:
                geom_rows.append(item)

            msd = np.asarray([x["mean_surface_distance_a_to_b_mm"] for x in per_frame], dtype=np.float64)
            hd = np.asarray([x["hausdorff_distance_mm"] for x in per_frame], dtype=np.float64)
            geom_summary = {
                "mean_surface_distance_mm": float(np.nanmean(msd)),
                "std_surface_distance_mm": float(np.nanstd(msd, ddof=1)) if np.isfinite(msd).sum() > 1 else float("nan"),
                "mean_hausdorff_distance_mm": float(np.nanmean(hd)),
            }
    else:
        geom_note = "Temporal geometry metrics require at least two frames; marked as N/A."
        for t in range(mask_metrics.shape[0]):
            geom_rows.append(
                {
                    "frame": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "status": geom_status,
                    "mean_surface_distance_a_to_b_mm": float("nan"),
                    "std_surface_distance_a_to_b_mm": float("nan"),
                    "symmetric_mean_surface_distance_mm": float("nan"),
                    "hausdorff_distance_mm": float("nan"),
                }
            )
        geom_summary = {
            "mean_surface_distance_mm": float("nan"),
            "std_surface_distance_mm": float("nan"),
            "mean_hausdorff_distance_mm": float("nan"),
        }

    if geom_rows:
        _write_csv(
            metrics_dir / "geometry_temporal_surface_metrics.csv",
            geom_rows,
            [
                "frame",
                "frame_source_index",
                "status",
                "mean_surface_distance_a_to_b_mm",
                "std_surface_distance_a_to_b_mm",
                "symmetric_mean_surface_distance_mm",
                "hausdorff_distance_mm",
            ],
        )

    # 5) Additional diagnostic plots
    corr_rows: List[Dict[str, Any]] = []
    ba_rows: List[Dict[str, Any]] = []
    ba_speed_name = ""
    corr_speed_name = ""
    corr_flow_name = ""

    # Intraluminal speed diagnostics
    m_in = mask_ref > 0.5
    sp_ref = speed_ref[m_in]
    sp_base = speed_base[m_in]
    sp_sr = speed_sr[m_in]

    if sp_ref.size > 20:
        # Bland-Altman: baseline vs reference and SR vs reference
        fig_ba = fig_dir / "bland_altman_speed_dual.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ba_base = _plot_bland_altman_panel(
            ax=axes[0],
            ref_vals=sp_ref,
            test_vals=sp_base,
            ref_label=ref_label,
            test_label=args.baseline_label,
            seed=17,
        )
        ba_sr = _plot_bland_altman_panel(
            ax=axes[1],
            ref_vals=sp_ref,
            test_vals=sp_sr,
            ref_label=ref_label,
            test_label=args.sr_label,
            seed=29,
        )
        fig.suptitle("Bland-Altman: intraluminal speed", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_ba, dpi=180)
        plt.close(fig)
        ba_speed_name = fig_ba.name

        ba_rows.append({"domain": "speed_intraluminal", "method": args.baseline_label, **ba_base})
        ba_rows.append({"domain": "speed_intraluminal", "method": args.sr_label, **ba_sr})

        # Correlation: baseline/ref and SR/ref
        fig_corr_speed = fig_dir / "correlation_speed_intraluminal.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        c_base = _plot_correlation_panel(
            ax=axes[0],
            x=sp_ref,
            y=sp_base,
            x_label=f"{ref_label} speed [m/s]",
            y_label=f"{args.baseline_label} speed [m/s]",
            title=f"{args.baseline_label} vs {ref_label}",
            color="#1d4ed8",
            seed=41,
        )
        c_sr = _plot_correlation_panel(
            ax=axes[1],
            x=sp_ref,
            y=sp_sr,
            x_label=f"{ref_label} speed [m/s]",
            y_label=f"{args.sr_label} speed [m/s]",
            title=f"{args.sr_label} vs {ref_label}",
            color="#b91c1c",
            seed=43,
        )
        fig.suptitle("Correlation: intraluminal speed", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_corr_speed, dpi=180)
        plt.close(fig)
        corr_speed_name = fig_corr_speed.name

        corr_rows.append({"domain": "speed_intraluminal", "method": args.baseline_label, **c_base})
        corr_rows.append({"domain": "speed_intraluminal", "method": args.sr_label, **c_sr})

    # Temporal-flow correlation diagnostics
    if q_ref_time.size > 2:
        fig_corr_flow = fig_dir / "correlation_flow_temporal.png"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cf_base = _plot_correlation_panel(
            ax=axes[0],
            x=q_ref_time,
            y=q_base_time,
            x_label=f"{ref_label} flow [ml/s]",
            y_label=f"{args.baseline_label} flow [ml/s]",
            title=f"{args.baseline_label} vs {ref_label}",
            color="#1d4ed8",
            seed=53,
        )
        cf_sr = _plot_correlation_panel(
            ax=axes[1],
            x=q_ref_time,
            y=q_sr_time,
            x_label=f"{ref_label} flow [ml/s]",
            y_label=f"{args.sr_label} flow [ml/s]",
            title=f"{args.sr_label} vs {ref_label}",
            color="#b91c1c",
            seed=59,
        )
        fig.suptitle("Correlation: temporal flow profile", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_corr_flow, dpi=180)
        plt.close(fig)
        corr_flow_name = fig_corr_flow.name

        corr_rows.append({"domain": "flow_temporal", "method": args.baseline_label, **cf_base})
        corr_rows.append({"domain": "flow_temporal", "method": args.sr_label, **cf_sr})

    # Paper-like cerebrovascular metrics at peak flow (component-wise + core/wall)
    peak_mask = (mask_ref[peak_idx] > 0.5)
    core_mask = binary_erosion(peak_mask, iterations=1)
    if int(core_mask.sum()) == 0:
        core_mask = peak_mask.copy()
    wall_mask = peak_mask & (~core_mask)
    if int(wall_mask.sum()) == 0:
        wall_mask = peak_mask.copy()

    peak_ref_comp = {
        "u": gt_phys[peak_idx, 0],
        "v": gt_phys[peak_idx, 1],
        "w": gt_phys[peak_idx, 2],
        "mag": speed_ref[peak_idx],
    }
    peak_base_comp = {
        "u": lr_vel_phys_metrics[peak_idx, 0],
        "v": lr_vel_phys_metrics[peak_idx, 1],
        "w": lr_vel_phys_metrics[peak_idx, 2],
        "mag": speed_base[peak_idx],
    }
    peak_sr_comp = {
        "u": pred_phys[peak_idx, 0],
        "v": pred_phys[peak_idx, 1],
        "w": pred_phys[peak_idx, 2],
        "mag": speed_sr[peak_idx],
    }

    comp_corr_rows: List[Dict[str, Any]] = []
    comp_ba_rows: List[Dict[str, Any]] = []

    fig_comp_corr = fig_dir / "correlation_velocity_components_peak.png"
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for c_idx, comp_name in enumerate(("u", "v", "w", "mag")):
        rr = peak_ref_comp[comp_name][peak_mask]
        bb = peak_base_comp[comp_name][peak_mask]
        ss = peak_sr_comp[comp_name][peak_mask]
        st_base = _plot_correlation_panel(
            ax=axes[0, c_idx],
            x=rr,
            y=bb,
            x_label=f"{ref_label} {comp_name}",
            y_label=f"{args.baseline_label} {comp_name}",
            title=f"{args.baseline_label} vs {ref_label} ({comp_name})",
            color="#1d4ed8",
            seed=101 + c_idx,
        )
        st_sr = _plot_correlation_panel(
            ax=axes[1, c_idx],
            x=rr,
            y=ss,
            x_label=f"{ref_label} {comp_name}",
            y_label=f"{args.sr_label} {comp_name}",
            title=f"{args.sr_label} vs {ref_label} ({comp_name})",
            color="#b91c1c",
            seed=111 + c_idx,
        )
        comp_corr_rows.append(
            {
                "domain": "velocity_component_peak",
                "region": "intraluminal",
                "component": comp_name,
                "method": args.baseline_label,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                **st_base,
            }
        )
        comp_corr_rows.append(
            {
                "domain": "velocity_component_peak",
                "region": "intraluminal",
                "component": comp_name,
                "method": args.sr_label,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                **st_sr,
            }
        )
    fig.suptitle("Peak-flow component correlation (intraluminal)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_comp_corr, dpi=180)
    plt.close(fig)
    comp_corr_name = fig_comp_corr.name

    fig_comp_ba = fig_dir / "bland_altman_velocity_components_peak.png"
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for c_idx, comp_name in enumerate(("u", "v", "w", "mag")):
        rr = peak_ref_comp[comp_name][peak_mask]
        bb = peak_base_comp[comp_name][peak_mask]
        ss = peak_sr_comp[comp_name][peak_mask]
        ba_base = _plot_bland_altman_panel(
            ax=axes[0, c_idx],
            ref_vals=rr,
            test_vals=bb,
            ref_label=ref_label,
            test_label=args.baseline_label,
            seed=201 + c_idx,
        )
        ba_sr = _plot_bland_altman_panel(
            ax=axes[1, c_idx],
            ref_vals=rr,
            test_vals=ss,
            ref_label=ref_label,
            test_label=args.sr_label,
            seed=211 + c_idx,
        )
        comp_ba_rows.append(
            {
                "domain": "velocity_component_peak",
                "region": "intraluminal",
                "component": comp_name,
                "method": args.baseline_label,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                **ba_base,
            }
        )
        comp_ba_rows.append(
            {
                "domain": "velocity_component_peak",
                "region": "intraluminal",
                "component": comp_name,
                "method": args.sr_label,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                **ba_sr,
            }
        )
    fig.suptitle("Peak-flow component Bland-Altman (intraluminal)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_comp_ba, dpi=180)
    plt.close(fig)
    comp_ba_name = fig_comp_ba.name

    # Peak speed metrics in core and wall regions
    def _region_peak_speed_metrics(region_mask: np.ndarray, region_name: str, method_name: str, test_speed: np.ndarray) -> Dict[str, Any]:
        r = speed_ref[peak_idx][region_mask]
        t = test_speed[peak_idx][region_mask]
        if r.size == 0:
            return {
                "domain": "peak_velocity_magnitude",
                "region": region_name,
                "method": method_name,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "relative_error_pct": float("nan"),
                "cosine_similarity": float("nan"),
            }
        rel = float(np.mean(np.abs(t - r) / (np.abs(r) + 1e-12)) * 100.0)
        num = float(np.dot(r.astype(np.float64), t.astype(np.float64)))
        den = float(np.linalg.norm(r.astype(np.float64)) * np.linalg.norm(t.astype(np.float64)) + 1e-12)
        return {
            "domain": "peak_velocity_magnitude",
            "region": region_name,
            "method": method_name,
            "frame_payload_index": int(peak_idx),
            "frame_source_index": int(peak_frame_src),
            "n": int(r.size),
            "mae": float(np.mean(np.abs(t - r))),
            "rmse": float(np.sqrt(np.mean((t - r) ** 2))),
            "relative_error_pct": rel,
            "cosine_similarity": float(num / den),
        }

    peak_speed_rows = [
        _region_peak_speed_metrics(core_mask, "core", args.baseline_label, speed_base),
        _region_peak_speed_metrics(core_mask, "core", args.sr_label, speed_sr),
        _region_peak_speed_metrics(wall_mask, "wall", args.baseline_label, speed_base),
        _region_peak_speed_metrics(wall_mask, "wall", args.sr_label, speed_sr),
    ]

    # Flow peak-like metrics (temporal)
    flow_peak_rows = []
    for method_name, q_time in ((args.baseline_label, q_base_time), (args.sr_label, q_sr_time)):
        flow_peak_rows.append(
            {
                "domain": "flow_temporal",
                "method": method_name,
                "peak_frame_payload_index": int(peak_idx),
                "peak_frame_source_index": int(peak_frame_src),
                "peak_ref_flow_ml_s": float(q_ref_time[peak_idx]),
                "peak_method_flow_ml_s": float(q_time[peak_idx]),
                "peak_abs_err_ml_s": float(abs(q_time[peak_idx] - q_ref_time[peak_idx])),
                "peak_relative_err_pct": float(abs(q_time[peak_idx] - q_ref_time[peak_idx]) / (abs(q_ref_time[peak_idx]) + 1e-12) * 100.0),
                "rmse_over_time_ml_s": float(np.sqrt(np.mean((q_time - q_ref_time) ** 2))),
                "relative_err_over_time_pct": float(np.mean(np.abs(q_time - q_ref_time) / (np.abs(q_ref_time) + 1e-12)) * 100.0),
            }
        )

    corr_rows.extend(comp_corr_rows)
    ba_rows.extend(comp_ba_rows)

    corr_cols = [
        "domain",
        "region",
        "component",
        "method",
        "frame_payload_index",
        "frame_source_index",
        "n",
        "slope",
        "intercept",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "r2_linear",
        "rmse",
        "bias",
    ]
    if corr_rows:
        _write_csv(metrics_dir / "correlation_metrics.csv", corr_rows, corr_cols)
    _write_csv(
        metrics_dir / "peak_velocity_metrics.csv",
        peak_speed_rows,
        ["domain", "region", "method", "frame_payload_index", "frame_source_index", "n", "mae", "rmse", "relative_error_pct", "cosine_similarity"],
    )
    _write_csv(
        metrics_dir / "flow_peak_metrics.csv",
        flow_peak_rows,
        [
            "domain",
            "method",
            "peak_frame_payload_index",
            "peak_frame_source_index",
            "peak_ref_flow_ml_s",
            "peak_method_flow_ml_s",
            "peak_abs_err_ml_s",
            "peak_relative_err_pct",
            "rmse_over_time_ml_s",
            "relative_err_over_time_pct",
        ],
    )

    ba_cols = ["domain", "region", "component", "method", "frame_payload_index", "frame_source_index", "n", "bias", "sd_diff", "loa_low", "loa_high"]
    if ba_rows:
        _write_csv(metrics_dir / "bland_altman_stats.csv", ba_rows, ba_cols)

    summary = {
        "report_title": args.report_title,
        "labels": {
            "reference": ref_label,
            "baseline": args.baseline_label,
            "super_resolved": args.sr_label,
        },
        "payload_path": str(Path(args.payload_npz).resolve()),
        "metadata": metadata,
        "dimensions": {
            "T": int(t_count),
            "frame_source_indices": [int(x) for x in frame_source_indices.tolist()],
            "lr_shape_xyz": [int(x) for x in lr_norm.shape[2:]],
            "shape_XYZ": [int(x) for x in pred_norm.shape[2:]],
            "flow_method": str(flow_method),
            "flow_section_kind": str(flow_section_kind),
            "flow_axis": int(selected_flow_axis),
            "flow_axis_mode": str(args.flow_axis),
            "suggested_flow_axis": int(suggested_flow_axis),
        },
        "statistics": {
            "table2_wilcoxon_p_re_baseline_vs_sr": p_re,
            "flow_wilcoxon_p_abs_err": p_flow,
            "wss_wilcoxon_p_abs_err": p_wss,
            "flow_reference_q_ml_s": q_ref_scalar,
            "flow_temporal_points": int(q_ref_time.shape[0]),
            "flow_temporal_section_count": int(valid_flow_sections.size),
            "flow_peak_frame_payload_index": int(peak_idx),
            "flow_peak_frame_source_index": int(peak_frame_src),
            "flow_axis_scores": flow_axis_scores,
            "flow_aggregate": str(flow_aggregate),
            "centerline": centerline_summary,
            "wss_enabled": bool(args.include_wss),
            "relative_pressure_status": "pending_vwerp_implementation",
            "correlation_rows": corr_rows,
            "bland_altman_rows": ba_rows,
            "peak_velocity_rows": peak_speed_rows,
            "flow_peak_rows": flow_peak_rows,
            "geometry_status": geom_status,
            "geometry_note": geom_note,
            "geometry_summary": geom_summary,
            "roi": roi_info,
        },
        "visualization": {
            "lr_mag_channel_used": int(args.lr_mag_channel),
            "max_display_slices": int(args.max_display_slices),
            "panel_cols": int(args.panel_cols),
            "hist_bins": int(args.hist_bins),
            "centerline_overlay_figure": str(centerline_overlay_name),
            "centerline_3d_figure": str(centerline_3d_name),
            "centerline_sections_figure": str(centerline_sections_name),
            "centerline_peak_qs_figure": str(centerline_peak_qs_name),
        },
    }

    (metrics_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # HTML report
    def fmt_rows(rows: List[Dict[str, Any]], keys: Sequence[str], nd: int = 5) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    rr[k] = "nan" if not np.isfinite(v) else f"{v:.{nd}f}"
                else:
                    rr[k] = str(v)
            out.append(rr)
        return out

    t2_comp_cols = ["location", "slice_index", "variable", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    t2_comp_html = _html_table(fmt_rows(table2_compact, t2_comp_cols, nd=6), t2_comp_cols)

    t3_cols = ["metric", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    t3_html = _html_table(fmt_rows(table3_rows, t3_cols, nd=6), t3_cols) if table3_rows else ""

    corr_cols = [
        "domain",
        "region",
        "component",
        "method",
        "frame_payload_index",
        "frame_source_index",
        "n",
        "slope",
        "intercept",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "r2_linear",
        "rmse",
        "bias",
    ]
    corr_html = _html_table(fmt_rows(corr_rows, corr_cols, nd=6), corr_cols) if corr_rows else "<p class=\"muted\">No correlation rows available.</p>"
    ba_cols = ["domain", "region", "component", "method", "frame_payload_index", "frame_source_index", "n", "bias", "sd_diff", "loa_low", "loa_high"]
    ba_html = _html_table(fmt_rows(ba_rows, ba_cols, nd=6), ba_cols) if ba_rows else "<p class=\"muted\">No Bland-Altman rows available.</p>"
    peak_vel_cols = ["domain", "region", "method", "frame_payload_index", "frame_source_index", "n", "mae", "rmse", "relative_error_pct", "cosine_similarity"]
    peak_vel_html = _html_table(fmt_rows(peak_speed_rows, peak_vel_cols, nd=6), peak_vel_cols) if peak_speed_rows else "<p class=\"muted\">No peak velocity rows.</p>"
    flow_peak_cols = [
        "domain",
        "method",
        "peak_frame_payload_index",
        "peak_frame_source_index",
        "peak_ref_flow_ml_s",
        "peak_method_flow_ml_s",
        "peak_abs_err_ml_s",
        "peak_relative_err_pct",
        "rmse_over_time_ml_s",
        "relative_err_over_time_pct",
    ]
    flow_peak_html = _html_table(fmt_rows(flow_peak_rows, flow_peak_cols, nd=6), flow_peak_cols) if flow_peak_rows else "<p class=\"muted\">No peak flow rows.</p>"

    flow_html = _html_table(fmt_rows(flow_rows, flow_cols, nd=6), flow_cols)
    voxel_dist_html = _html_table(fmt_rows(voxel_dist_rows, voxel_dist_cols, nd=6), voxel_dist_cols)
    flow_axis_rows = []
    for axis in sorted(flow_axis_scores.keys()):
        d = flow_axis_scores[axis]
        flow_axis_rows.append(
            {
                "axis": int(axis),
                "score": d.get("score", float("nan")),
                "rel_sd": d.get("rel_sd", float("nan")),
                "smoothness": d.get("smoothness", float("nan")),
                "coverage": d.get("coverage", float("nan")),
            }
        )
    flow_axis_cols = ["axis", "score", "rel_sd", "smoothness", "coverage"]
    flow_axis_html = _html_table(fmt_rows(flow_axis_rows, flow_axis_cols, nd=6), flow_axis_cols)

    geom_html = _html_table(
        [
            {
                "mean_surface_distance_mm": "nan" if not np.isfinite(geom_summary["mean_surface_distance_mm"]) else f"{geom_summary['mean_surface_distance_mm']:.6f}",
                "std_surface_distance_mm": "nan" if not np.isfinite(geom_summary["std_surface_distance_mm"]) else f"{geom_summary['std_surface_distance_mm']:.6f}",
                "mean_hausdorff_distance_mm": "nan" if not np.isfinite(geom_summary["mean_hausdorff_distance_mm"]) else f"{geom_summary['mean_hausdorff_distance_mm']:.6f}",
            }
        ],
        ["mean_surface_distance_mm", "std_surface_distance_mm", "mean_hausdorff_distance_mm"],
    )

    ch_img_tags = "\n".join(
        f"<h4>Channel {k}</h4><img src='figures/{v}' alt='{k} comparison'/>" for k, v in channel_figs.items()
    )

    wss_summary_text = (
        f"{p_wss:.4g}" if (bool(args.include_wss) and np.isfinite(p_wss)) else ("disabled" if not bool(args.include_wss) else "nan")
    )
    wss_img_tag = f'<img src="figures/{fig_wss_name}" alt="WSS distribution"/>' if fig_wss_name else ""
    corr_speed_tag = f'<img src="figures/{corr_speed_name}" alt="Correlation speed"/>' if corr_speed_name else ""
    corr_flow_tag = f'<img src="figures/{corr_flow_name}" alt="Correlation temporal flow"/>' if corr_flow_name else ""
    comp_corr_tag = f'<img src="figures/{comp_corr_name}" alt="Correlation velocity components peak"/>' if comp_corr_name else ""
    comp_ba_tag = f'<img src="figures/{comp_ba_name}" alt="Bland-Altman velocity components peak"/>' if comp_ba_name else ""
    centerline_overlay_tag = f'<img src="figures/{centerline_overlay_name}" alt="Centerline overlay"/>' if centerline_overlay_name else ""
    centerline_3d_tag = f'<img src="figures/{centerline_3d_name}" alt="Centerline 3D"/>' if centerline_3d_name else ""
    centerline_sections_tag = f'<img src="figures/{centerline_sections_name}" alt="Centerline plane sections"/>' if centerline_sections_name else ""
    centerline_peak_qs_tag = f'<img src="figures/{centerline_peak_qs_name}" alt="Centerline peak flow along vessel"/>' if centerline_peak_qs_name else ""
    centerline_qc_cols = [
        "plane_index",
        "is_valid_plane",
        "support_voxels",
        "area_mm2",
        "centroid_offset_norm",
        "elongation_ratio",
        "compactness_ratio",
        "center_to_wall_over_eq_radius",
    ]
    centerline_qc_html = (
        _html_table(fmt_rows(centerline_section_qc_rows, centerline_qc_cols, nd=6), centerline_qc_cols)
        if centerline_section_qc_rows
        else "<p class=\"muted\">No centerline section QC rows.</p>"
    )
    centerline_sign_cols = ["method", "n_frames", "pearson_r_vs_ref", "pearson_p_vs_ref", "sign_agreement_pct", "flip_suspected"]
    centerline_sign_html = (
        _html_table(fmt_rows(centerline_sign_rows, centerline_sign_cols, nd=6), centerline_sign_cols)
        if centerline_sign_rows
        else "<p class=\"muted\">No centerline sign QC rows.</p>"
    )
    if bool(args.include_wss):
        wss_section = (
            "<h2>Paper-style Table 3 (WSS)</h2>"
            "<p class=\"muted\">WSS estimated from boundary-normal finite differences "
            "(2-point polynomial approximation), aggregated over all processed frames.</p>"
            f"{t3_html}"
            f"{wss_img_tag}"
        )
    else:
        wss_section = (
            "<h2>Paper-style Table 3 (WSS)</h2>"
            "<p class=\"muted\">WSS is disabled in this run (`--include-wss` not set).</p>"
        )

    corr_section = (
        "<h2>Correlation Diagnostics</h2>"
        f"{corr_speed_tag}"
        f"{corr_flow_tag}"
        f"{comp_corr_tag}"
        f"{corr_html}"
    )

    pressure_section = (
        "<h2>Relative Pressure Diagnostics</h2>"
        "<p class=\"muted\">Pending: this repository currently does not include a vWERP pipeline, "
        "so pressure metrics from the cerebrovascular paper are not yet computed.</p>"
    )

    saved_wss_items = (
        "<li><code>metrics/table3_like_wss.csv</code></li><li><code>metrics/table3_like_wss_per_frame.csv</code></li>"
        if bool(args.include_wss)
        else ""
    )
    if flow_method == "centerline":
        p_peak_planes = centerline_error_p.get("peak_plane_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))
        p_all_planes = centerline_error_p.get("all_planes_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))
        p_peak_txt = "nan" if not np.isfinite(float(p_peak_planes)) else f"{float(p_peak_planes):.4g}"
        p_all_txt = "nan" if not np.isfinite(float(p_all_planes)) else f"{float(p_all_planes):.4g}"
        centerline_exec_bullet = (
            f"<li>Centerline plane-error p-values (baseline vs SR): peak planes <b>{p_peak_txt}</b>, "
            f"all frame-plane samples <b>{p_all_txt}</b></li>"
        )
        flow_axis_bullet = (
            f"Flow axis used for slice-wise Table-2 metrics: <b>{selected_flow_axis}</b> "
            f"(mode: {args.flow_axis}, suggested: {suggested_flow_axis})"
        )
        flow_method_bullet = (
            f"Flow method: <b>centerline</b> ({int(valid_flow_sections.size)} valid planes, agg={flow_aggregate}, "
            f"path mode={centerline_summary.get('path_mode', 'n/a')})"
        )
        flow_temporal_bullet = (
            f"Temporal flow points: <b>{int(q_ref_time.shape[0])}</b>, aggregated planes: "
            f"<b>{int(valid_flow_sections.size)}</b>"
        )
        flow_diag_text = (
            "Temporal flow is computed per frame by integrating v·n in slabs around planes orthogonal to "
            "the extracted centerline, then aggregating valid planes."
        )
        flow_axis_section = ""
        centerline_section = (
            "<h3>Centerline Selection (Visual QC)</h3>"
            "<p class=\"muted\">Blue transparent shape: lumen mask. Orange line: centerline path. Green points: valid flow planes. "
            "Red points: sampled planes rejected by support threshold.</p>"
            f"{centerline_overlay_tag}"
            f"{centerline_3d_tag}"
            "<h3>Centerline Plane QC</h3>"
            "<p class=\"muted\">Check centeredness (offset_norm), compactness (elongation/compactness), and section-to-wall distance ratio.</p>"
            f"{centerline_sections_tag}"
            f"{centerline_qc_html}"
            "<h3>Centerline Flow Consistency (Peak Frame)</h3>"
            "<p class=\"muted\">Q(s) should vary smoothly along nearby planes in branch-free tubular segments. "
            f"Wilcoxon p (|err| baseline vs SR, peak frame planes): <b>{p_peak_txt}</b>; "
            f"all frame-plane samples: <b>{p_all_txt}</b>.</p>"
            f"{centerline_peak_qs_tag}"
            "<h3>Centerline Sign QC</h3>"
            "<p class=\"muted\">Negative temporal correlation with reference may indicate inverted proximal-distal orientation.</p>"
            f"{centerline_sign_html}"
        )
        saved_centerline_items = (
            "<li><code>metrics/centerline_points.csv</code></li>"
            "<li><code>metrics/flow_centerline_planes_per_frame.csv</code></li>"
            "<li><code>metrics/centerline_section_qc.csv</code></li>"
            "<li><code>metrics/centerline_sign_qc.csv</code></li>"
            "<li><code>figures/centerline_overlay.png</code></li>"
            "<li><code>figures/centerline_3d.png</code></li>"
            "<li><code>figures/centerline_plane_sections.png</code></li>"
            "<li><code>figures/centerline_flow_along_vessel_peak.png</code></li>"
        )
    else:
        centerline_exec_bullet = ""
        flow_axis_bullet = f"Flow axis used: <b>{selected_flow_axis}</b> (mode: {args.flow_axis}, suggested: {suggested_flow_axis})"
        flow_method_bullet = "Flow method: <b>axis slices</b>"
        flow_temporal_bullet = (
            f"Temporal flow points: <b>{int(q_ref_time.shape[0])}</b>, aggregated slices: "
            f"<b>{int(valid_flow_sections.size)}</b>"
        )
        flow_diag_text = "Temporal flow is computed per frame after aggregating cross-sectional flow across valid slices along the selected axis."
        flow_axis_section = (
            "<h3>Flow Axis Selection</h3>"
            "<p class=\"muted\">Lower score is better (lower temporal relative SD, smoother profile, higher valid-flow coverage).</p>"
            f"{flow_axis_html}"
        )
        centerline_section = ""
        saved_centerline_items = ""

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{args.report_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ color: #111827; }}
    .muted {{ color: #6b7280; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; margin: 8px 0 20px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    .pill {{ display: inline-block; background: #eef2ff; color: #3730a3; padding: 2px 8px; border-radius: 999px; margin-right: 8px; }}
  </style>
</head>
<body>
  <h1>{args.report_title}</h1>
  <p class=\"muted\">Generated from payload: <code>{Path(args.payload_npz).resolve()}</code></p>

  <p>
    <span class=\"pill\">Reference: {ref_label}</span>
    <span class=\"pill\">Baseline: {args.baseline_label}</span>
    <span class=\"pill\">Super-resolved: {args.sr_label}</span>
  </p>

  <h2>Executive Summary</h2>
  <ul>
    <li>Intraluminal statistics RE comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_re) else f'{p_re:.4g}'}</b></li>
    <li>Temporal flow absolute error comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_flow) else f'{p_flow:.4g}'}</b></li>
    <li>WSS absolute error comparison (Wilcoxon p): <b>{wss_summary_text}</b></li>
    <li>Flow reference value used (ml/s): <b>{q_ref_scalar:.6f}</b></li>
    {centerline_exec_bullet}
    <li>{flow_method_bullet}</li>
    <li>{flow_axis_bullet}</li>
    <li>{flow_temporal_bullet}</li>
    <li>LR magnitude channel used for visualization: <b>{args.lr_mag_channel}</b></li>
    <li>ROI mode: <b>{'enabled' if roi_info.get('enabled', False) else 'disabled'}</b>{'' if not roi_info.get('enabled', False) else f" (bbox xyz: {roi_info.get('bbox_xyz')})"}</li>
    <li>Baseline LR alignment for metrics: <b>{'upsampled to HR grid' if tuple(lr_norm.shape[2:]) != tuple(gt_norm.shape[2:]) else 'native HR size'}</b></li>
  </ul>

  <h2>{'Visual Inspection (ROI Bounding Box)' if roi_info.get('enabled', False) else 'Visual Inspection (Full Volume)'}</h2>
  {ch_img_tags}

  <h2>Voxel Distribution Inside Mask</h2>
  <p class=\"muted\">Histogram comparison over all in-mask voxels across all processed frames{'' if not roi_info.get('enabled', False) else ' (restricted to ROI bbox)'}.</p>
  <img src=\"figures/{fig_voxel_hist.name}\" alt=\"In-mask voxel histograms\"/>
  {voxel_dist_html}

  <h2>Flow-rate Diagnostics</h2>
  <img src=\"figures/{fig_flow.name}\" alt=\"Flow profile\"/>
  <p class=\"muted\">{flow_diag_text}</p>
  {flow_axis_section}
  {centerline_section}

  <h2>Paper-style Table 2 (Representative Slices)</h2>
  <p class=\"muted\">Variables: mean/SD/skewness/kurtosis of intraluminal velocity and vorticity (aggregated over all processed frames). Location labels are voxel slice IDs along the selected flow axis.</p>
  {t2_comp_html}

  <h2>Paper-style Flow Metrics</h2>
  <p class=\"muted\">Temporal summaries against reference flow (Qref).</p>
  {flow_html}
  <h3>Flow Peak Metrics (Paper-style)</h3>
  <p class=\"muted\">Peak-frame and temporal RMSE/relative-error summaries aligned with cerebrovascular SR comparisons.</p>
  {flow_peak_html}

  {wss_section}

  <h2>Geometry Uncertainty (Surface/Hausdorff)</h2>
  <p class=\"muted\">{geom_note}</p>
  {geom_html}

  {corr_section}

  {pressure_section}

  <h2>Bland-Altman</h2>
  {'' if not ba_speed_name else f'<img src="figures/{ba_speed_name}" alt="Bland-Altman speed"/>'}
  {comp_ba_tag}
  {ba_html}

  <h2>Peak Velocity Metrics (Core/Wall)</h2>
  <p class=\"muted\">Peak-flow velocity magnitude metrics (MAE, RMSE, relative error, cosine similarity), split by core and wall masks.</p>
  {peak_vel_html}

  <h2>Saved Artifacts</h2>
  <ul>
    <li><code>metrics/table2_like_all_slices.csv</code></li>
    <li><code>metrics/table2_like_per_frame_all_slices.csv</code></li>
    <li><code>metrics/table2_like_compact.csv</code></li>
    <li><code>metrics/flow_metrics.csv</code></li>
    <li><code>metrics/flow_metrics_per_frame.csv</code></li>
    <li><code>metrics/flow_rate_curves_per_frame.csv</code></li>
    {saved_centerline_items}
    {saved_wss_items}
    <li><code>metrics/geometry_temporal_surface_metrics.csv</code></li>
    <li><code>metrics/voxel_distribution_stats.csv</code></li>
    <li><code>metrics/correlation_metrics.csv</code></li>
    <li><code>metrics/bland_altman_stats.csv</code></li>
    <li><code>metrics/peak_velocity_metrics.csv</code></li>
    <li><code>metrics/flow_peak_metrics.csv</code></li>
    <li><code>metrics/summary_metrics.json</code></li>
  </ul>

</body>
</html>
"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    print("Report generated:")
    print(f"- HTML: {report_path}")
    print(f"- Figures: {fig_dir}")
    print(f"- Metrics: {metrics_dir}")


if __name__ == "__main__":
    main()
