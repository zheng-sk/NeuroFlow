import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional style dependency
    sns = None

REPORT_FIG_DPI = 320
REPORT_FONT_FAMILY = "Times New Roman"
REPORT_FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
REPORT_COLOR_REF = "#009E73"
REPORT_COLOR_BASELINE = "#D55E00"
REPORT_COLOR_SR = "#0072B2"
REPORT_COLOR_NEUTRAL = "#111827"
VIOLIN_UPPER_PERCENTILE = 95.0
REPORT_FILL_ALPHA = 0.74
REPORT_BOX_ALPHA = 0.48

if sns is not None:
    sns.set_theme(
        style="ticks",
        context="paper",
        font=REPORT_FONT_FAMILY,
        palette="colorblind",
    )

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "semibold",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "font.family": REPORT_FONT_FAMILY,
        "font.serif": REPORT_FONT_SERIF,
        "mathtext.fontset": "stix",
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "grid.color": "#d1d5db",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.8,
        "savefig.dpi": REPORT_FIG_DPI,
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


def _relative_error_eps(val: float, ref: float, eps: float) -> float:
    denom = max(abs(float(ref)), max(float(eps), 1e-12))
    return abs(float(val) - float(ref)) / denom


def _smape_ratio(val: float, ref: float, eps: float = 1e-12) -> float:
    denom = abs(float(val)) + abs(float(ref)) + max(float(eps), 1e-12)
    return 2.0 * abs(float(val) - float(ref)) / denom


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


def _finite_values(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _scipy_mean(x: Sequence[float] | np.ndarray) -> float:
    arr = _finite_values(x)
    if arr.size == 0:
        return float("nan")
    return float(stats.tmean(arr))


def _scipy_std(x: Sequence[float] | np.ndarray) -> float:
    arr = _finite_values(x)
    if arr.size < 2:
        return float("nan")
    return float(stats.tstd(arr))


def _scipy_percentile(x: Sequence[float] | np.ndarray, p: float) -> float:
    arr = _finite_values(x)
    if arr.size == 0:
        return float("nan")
    return float(stats.scoreatpercentile(arr, p))


def _scipy_median(x: Sequence[float] | np.ndarray) -> float:
    return _scipy_percentile(x, 50.0)


def _scipy_rmse(err: Sequence[float] | np.ndarray) -> float:
    arr = _finite_values(err)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(stats.tmean(arr**2)))


def _winsorize_upper_percentile(x: Sequence[float] | np.ndarray, upper_p: float = 95.0) -> np.ndarray:
    """Cap upper tails at a robust percentile to keep violin scales readable."""
    arr = _finite_values(x)
    if arr.size == 0:
        return arr
    p = float(np.clip(upper_p, 50.0, 100.0))
    hi = _scipy_percentile(arr, p)
    if not np.isfinite(hi):
        return arr
    return np.minimum(arr, hi)


def _table2_ref_eps_map(rows: List[Dict[str, Any]], ref_key: str = "ref", q: float = 5.0) -> Dict[str, float]:
    by_var: Dict[str, List[float]] = {}
    for r in rows:
        var = str(r.get("variable", "")).strip()
        ref = float(r.get(ref_key, float("nan")))
        if not var or not np.isfinite(ref):
            continue
        by_var.setdefault(var, []).append(abs(ref))
    out: Dict[str, float] = {}
    for var, vals in by_var.items():
        arr = _finite_values(vals)
        arr = arr[arr > 0]
        if arr.size == 0:
            out[var] = 1e-6
            continue
        eps = _scipy_percentile(arr, q)
        if not np.isfinite(eps) or eps <= 0:
            med = _scipy_median(arr)
            eps = 0.1 * med if np.isfinite(med) and med > 0 else 1e-6
        out[var] = max(float(eps), 1e-6)
    return out


def _augment_table2_rows_with_robust_relative(
    rows: List[Dict[str, Any]],
    eps_map: Dict[str, float],
    ref_key: str,
    baseline_key: str,
    sr_key: str,
) -> None:
    for r in rows:
        var = str(r.get("variable", "")).strip()
        ref = float(r.get(ref_key, float("nan")))
        base = float(r.get(baseline_key, float("nan")))
        sr = float(r.get(sr_key, float("nan")))
        eps = float(eps_map.get(var, 1e-6))
        if np.isfinite(ref) and np.isfinite(base):
            r["re_baseline_eps"] = _relative_error_eps(base, ref, eps)
            r["smape_baseline"] = _smape_ratio(base, ref, eps=eps)
        else:
            r["re_baseline_eps"] = float("nan")
            r["smape_baseline"] = float("nan")
        if np.isfinite(ref) and np.isfinite(sr):
            r["re_sr_eps"] = _relative_error_eps(sr, ref, eps)
            r["smape_sr"] = _smape_ratio(sr, ref, eps=eps)
        else:
            r["re_sr_eps"] = float("nan")
            r["smape_sr"] = float("nan")
        r["ref_eps"] = float(eps)


def _table2_relative_robust_summary(
    rows: List[Dict[str, Any]],
    ref_key: str,
    baseline_key: str,
    sr_key: str,
    re_base_key: str,
    re_sr_key: str,
    re_base_eps_key: str,
    re_sr_eps_key: str,
    smape_base_key: str,
    smape_sr_key: str,
    ref_eps_key: str = "ref_eps",
    baseline_method_label: str = "baseline",
    sr_method_label: str = "sr",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    methods = [
        (baseline_method_label, baseline_key, re_base_key, re_base_eps_key, smape_base_key),
        (sr_method_label, sr_key, re_sr_key, re_sr_eps_key, smape_sr_key),
    ]
    by_var = sorted(set(str(r.get("variable", "")).strip() for r in rows if str(r.get("variable", "")).strip()))
    for var in by_var:
        subset = [r for r in rows if str(r.get("variable", "")).strip() == var]
        ref_vals = _finite_values([float(r.get(ref_key, float("nan"))) for r in subset])
        eps_var = _scipy_median([float(r.get(ref_eps_key, float("nan"))) for r in subset])
        for method_name, pred_key, re_key, re_eps_key, smape_key in methods:
            pred_vals = _finite_values([float(r.get(pred_key, float("nan"))) for r in subset])
            abs_err_vals = _finite_values([abs(float(r.get(pred_key, float("nan"))) - float(r.get(ref_key, float("nan")))) for r in subset])
            re_vals = 100.0 * _finite_values([float(r.get(re_key, float("nan"))) for r in subset])
            re_eps_vals = 100.0 * _finite_values([float(r.get(re_eps_key, float("nan"))) for r in subset])
            smape_vals = 100.0 * _finite_values([float(r.get(smape_key, float("nan"))) for r in subset])
            denom_raw = float(np.sum(np.abs(ref_vals))) if ref_vals.size else float("nan")
            wape_raw = float(np.sum(abs_err_vals) / denom_raw * 100.0) if np.isfinite(denom_raw) and denom_raw > 1e-12 else float("nan")
            if ref_vals.size:
                denom_eps = float(np.sum(np.maximum(np.abs(ref_vals), max(float(eps_var), 1e-12))))
                wape_eps = float(np.sum(abs_err_vals) / denom_eps * 100.0) if denom_eps > 1e-12 else float("nan")
            else:
                wape_eps = float("nan")
            out.append(
                {
                    "variable": var,
                    "method": method_name,
                    "n": int(abs_err_vals.size),
                    "ref_eps": float(eps_var) if np.isfinite(eps_var) else float("nan"),
                    "mean_re_pct": _scipy_mean(re_vals),
                    "median_re_pct": _scipy_median(re_vals),
                    "p95_re_pct": _scipy_percentile(re_vals, 95.0),
                    "mean_re_eps_pct": _scipy_mean(re_eps_vals),
                    "median_re_eps_pct": _scipy_median(re_eps_vals),
                    "p95_re_eps_pct": _scipy_percentile(re_eps_vals, 95.0),
                    "mean_smape_pct": _scipy_mean(smape_vals),
                    "median_smape_pct": _scipy_median(smape_vals),
                    "p95_smape_pct": _scipy_percentile(smape_vals, 95.0),
                    "wape_pct": wape_raw,
                    "wape_eps_pct": wape_eps,
                }
            )
    return out


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
            ("Mean velocity [m/s]", _scipy_mean, sv_ref, sv_base, sv_sr),
            ("SD velocity [m/s]", _scipy_std, sv_ref, sv_base, sv_sr),
            ("Skewness velocity", _safe_skew, sv_ref, sv_base, sv_sr),
            ("Kurtosis velocity", _safe_kurtosis, sv_ref, sv_base, sv_sr),
            ("Mean vorticity [1/s]", _scipy_mean, vo_ref, vo_base, vo_sr),
            ("SD vorticity [1/s]", _scipy_std, vo_ref, vo_base, vo_sr),
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


def _table2_temporal_mean_rows(
    table2_per_frame_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    optional_fields = [f for f in ("re_baseline_eps", "re_sr_eps", "smape_baseline", "smape_sr", "ref_eps") if any(f in r for r in table2_per_frame_rows)]
    fields = ["ref", "baseline", "sr", "re_baseline", "re_sr"] + optional_fields
    grouped: Dict[Tuple[int, str], Dict[str, List[float]]] = {}
    for row in table2_per_frame_rows:
        slice_idx = int(row.get("slice_index", -1))
        var_name = str(row.get("variable", ""))
        key = (slice_idx, var_name)
        if key not in grouped:
            grouped[key] = {f: [] for f in fields}
        g = grouped[key]
        for field in fields:
            g[field].append(float(row.get(field, float("nan"))))

    out: List[Dict[str, Any]] = []
    for (slice_idx, var_name), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        ref_v = _finite_values(vals["ref"])
        base_v = _finite_values(vals["baseline"])
        sr_v = _finite_values(vals["sr"])
        re_base_v = _finite_values(vals["re_baseline"])
        re_sr_v = _finite_values(vals["re_sr"])
        re_base_eps_v = _finite_values(vals.get("re_baseline_eps", []))
        re_sr_eps_v = _finite_values(vals.get("re_sr_eps", []))
        smape_base_v = _finite_values(vals.get("smape_baseline", []))
        smape_sr_v = _finite_values(vals.get("smape_sr", []))
        ref_eps_v = _finite_values(vals.get("ref_eps", []))
        n_frames = int(max(ref_v.size, base_v.size, sr_v.size, re_base_v.size, re_sr_v.size))
        row_out = {
            "slice_index": int(slice_idx),
            "variable": str(var_name),
            "n_frames": int(n_frames),
            "ref_mean_over_frames": _scipy_mean(ref_v),
            "baseline_mean_over_frames": _scipy_mean(base_v),
            "sr_mean_over_frames": _scipy_mean(sr_v),
            "re_baseline_mean_over_frames": _scipy_mean(re_base_v),
            "re_sr_mean_over_frames": _scipy_mean(re_sr_v),
        }
        if re_base_eps_v.size or re_sr_eps_v.size:
            row_out["re_baseline_eps_mean_over_frames"] = _scipy_mean(re_base_eps_v)
            row_out["re_sr_eps_mean_over_frames"] = _scipy_mean(re_sr_eps_v)
        if smape_base_v.size or smape_sr_v.size:
            row_out["smape_baseline_mean_over_frames"] = _scipy_mean(smape_base_v)
            row_out["smape_sr_mean_over_frames"] = _scipy_mean(smape_sr_v)
        if ref_eps_v.size:
            row_out["ref_eps_mean_over_frames"] = _scipy_mean(ref_eps_v)
        out.append(row_out)
    return out


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
            abs_vs_qref = np.abs(q_method_t - float(q_ref_scalar))
            abs_vs_ref = np.abs(q_method_t - ref_t)
            rows.append(
                {
                    "frame_payload_index": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "method": method,
                    "mean_Q_ml_s": _scipy_mean(q_method_t),
                    "MAD_Q_vs_qref_ml_s": _scipy_mean(abs_vs_qref),
                    "MAD_Q_vs_ref_profile_ml_s": _scipy_mean(abs_vs_ref),
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
    q_time = np.asarray(stats.tmean(q_curves[:, valid_slices], axis=1), dtype=np.float32)
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
        lr = stats.linregress(x, y)
        a, b = float(lr.slope), float(lr.intercept)
        y_hat = a * x + b
        ss_res = float(np.sum((y - y_hat) ** 2))
        y_mean = _scipy_mean(y)
        ss_tot = float(np.sum((y - y_mean) ** 2))
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
        "rmse": _scipy_rmse(y - x),
        "bias": _scipy_mean(y - x),
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

    ax.scatter(xs, ys, s=20, marker="o", alpha=0.35, color=color, edgecolors="white", linewidths=0.25)
    lo, hi = _robust_range([x, y], symmetric=False, lower_q=0.2, upper_q=99.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=REPORT_COLOR_NEUTRAL, linewidth=1.0, label="Identity")
    if x.size >= 2:
        try:
            lr = stats.linregress(x, y)
            a, b = float(lr.slope), float(lr.intercept)
            ax.plot([lo, hi], [a * lo + b, a * hi + b], color="#374151", linewidth=1.4, label="Linear fit")
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
    ax.scatter(mean_v[idx], diff_v[idx], s=20, marker="o", alpha=0.38, color="#1f2937", edgecolors="white", linewidths=0.25)

    bias = _scipy_mean(diff_v)
    sd = _scipy_std(diff_v)
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


def _pad_limits(lo: float, hi: float, pad_ratio: float = 0.08) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    pad = max(1e-6, (hi - lo) * float(pad_ratio))
    return float(lo - pad), float(hi + pad)


def _bland_altman_joint_limits(
    ref_vals: np.ndarray,
    test_vals_list: Sequence[np.ndarray],
    pad_ratio: float = 0.08,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    means: List[np.ndarray] = []
    diffs: List[np.ndarray] = []
    ref0 = np.asarray(ref_vals, dtype=np.float64).ravel()
    for tv in test_vals_list:
        test0 = np.asarray(tv, dtype=np.float64).ravel()
        n = min(ref0.size, test0.size)
        if n < 3:
            continue
        rr = ref0[:n]
        tt = test0[:n]
        m = np.isfinite(rr) & np.isfinite(tt)
        rr = rr[m]
        tt = tt[m]
        if rr.size < 3:
            continue
        means.append(0.5 * (tt + rr))
        diffs.append(tt - rr)
    if not means or not diffs:
        return None
    x_lo, x_hi = _robust_range(means, symmetric=False, lower_q=0.2, upper_q=99.8)
    y_lo, y_hi = _robust_range(diffs, symmetric=True, lower_q=0.2, upper_q=99.8)
    x_lo, x_hi = _pad_limits(x_lo, x_hi, pad_ratio=pad_ratio)
    y_lo, y_hi = _pad_limits(y_lo, y_hi, pad_ratio=pad_ratio)
    return (x_lo, x_hi), (y_lo, y_hi)


def _correlation_joint_limits(
    ref_vals: np.ndarray,
    test_vals_list: Sequence[np.ndarray],
    pad_ratio: float = 0.05,
) -> Optional[Tuple[float, float]]:
    series: List[np.ndarray] = []
    r0 = np.asarray(ref_vals, dtype=np.float64).ravel()
    r0 = r0[np.isfinite(r0)]
    if r0.size:
        series.append(r0)
    for tv in test_vals_list:
        t0 = np.asarray(tv, dtype=np.float64).ravel()
        t0 = t0[np.isfinite(t0)]
        if t0.size:
            series.append(t0)
    if not series:
        return None
    lo, hi = _robust_range(series, symmetric=False, lower_q=0.2, upper_q=99.8)
    lo, hi = _pad_limits(lo, hi, pad_ratio=pad_ratio)
    return lo, hi


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
        coverage = _scipy_mean(valid.astype(np.float64)) if valid.size > 0 else 0.0

        if np.any(valid):
            rel_sd = _scipy_mean(q_sd[valid] / (abs_mean[valid] + 1e-8))
        else:
            rel_sd = 1e6

        if q_mean.size >= 3:
            d1 = np.diff(q_mean)
            d2 = np.diff(q_mean, n=2)
            smoothness = float(_scipy_mean(np.abs(d2)) / (_scipy_mean(np.abs(d1)) + 1e-8))
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
        "Mean": _scipy_mean(arr),
        "SD": _scipy_std(arr),
        "Quantile_97_5": _scipy_percentile(arr, 97.5),
        "Median": _scipy_median(arr),
        "Quantile_2_5": _scipy_percentile(arr, 2.5),
        "IQR_75_25": float(stats.iqr(arr, rng=(25, 75), nan_policy="omit")),
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
        "mean_surface_distance_a_to_b_mm": _scipy_mean(d_ab),
        "std_surface_distance_a_to_b_mm": _scipy_std(d_ab),
        "symmetric_mean_surface_distance_mm": float(0.5 * (_scipy_mean(d_ab) + _scipy_mean(d_ba))),
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


def _common_spatial_shape(*arrays: np.ndarray) -> Tuple[int, int, int]:
    shapes = [tuple(int(v) for v in np.asarray(a).shape[-3:]) for a in arrays]
    if not shapes:
        raise ValueError("No arrays provided to compute common spatial shape.")
    sx = min(s[0] for s in shapes)
    sy = min(s[1] for s in shapes)
    sz = min(s[2] for s in shapes)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"Invalid common spatial shape from {shapes}")
    return int(sx), int(sy), int(sz)


def _crop_spatial_to_shape(arr: np.ndarray, shape_xyz: Tuple[int, int, int]) -> np.ndarray:
    sx, sy, sz = [int(v) for v in shape_xyz]
    if arr.ndim < 3:
        raise ValueError(f"Expected array with >=3 dims, got shape {arr.shape}")
    slices = (slice(None),) * (arr.ndim - 3) + (slice(0, sx), slice(0, sy), slice(0, sz))
    return np.asarray(arr[slices], dtype=arr.dtype)


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


def _rel_fig_path(path: Path, fig_root: Path) -> str:
    try:
        return path.resolve().relative_to(fig_root.resolve()).as_posix()
    except Exception:
        return path.name


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


def _extract_res_increase(metadata: Dict[str, Any], payload: Dict[str, np.ndarray]) -> Optional[int]:
    try:
        if "res_increase" in metadata:
            res_meta = int(metadata["res_increase"])
            if res_meta >= 1:
                return res_meta
    except Exception:
        pass

    try:
        lr_shape = np.asarray(payload.get("lr_norm", np.empty((0, 0, 0, 0, 0)))).shape
        gt_shape = np.asarray(payload.get("gt_norm", np.empty((0, 0, 0, 0, 0)))).shape
        if len(lr_shape) >= 5 and len(gt_shape) >= 5:
            lr_xyz = np.asarray(lr_shape[-3:], dtype=np.float64)
            gt_xyz = np.asarray(gt_shape[-3:], dtype=np.float64)
            if np.all(lr_xyz > 0):
                ratios = gt_xyz / lr_xyz
                if np.all(np.isfinite(ratios)):
                    ratio_med = float(np.median(ratios))
                    ratio_round = int(np.round(ratio_med))
                    if ratio_round >= 1 and np.max(np.abs(ratios - ratio_round)) <= 0.15:
                        return ratio_round
    except Exception:
        pass
    return None


def _resolve_task_mode(
    task_mode_arg: str,
    metadata: Dict[str, Any],
    payload: Dict[str, np.ndarray],
) -> Tuple[str, str, Optional[int]]:
    arg = str(task_mode_arg or "auto").strip().lower()
    if arg not in {"auto", "denoising", "superresolution"}:
        arg = "auto"

    detected_res = _extract_res_increase(metadata=metadata, payload=payload)
    if arg == "denoising":
        mode_tag = "denoising"
    elif arg == "superresolution":
        mode_tag = "superresolution"
    else:
        mode_tag = "denoising" if detected_res == 1 else "superresolution"

    if mode_tag == "denoising":
        base_label = "Denoising"
    else:
        base_label = "Superresolution"

    if detected_res is not None:
        mode_label = f"{base_label} (res_increase={detected_res})"
    else:
        mode_label = f"{base_label} (res_increase=unknown)"
    return mode_tag, mode_label, detected_res


def _resolve_method_labels(task_mode_tag: str, baseline_label_arg: str, sr_label_arg: str) -> Tuple[str, str, str]:
    baseline_label = str(baseline_label_arg).strip() or "3T"
    sr_label = str(sr_label_arg).strip() or "3T SR"
    sr_role_label = "Super-resolved"

    sr_default_aliases = {
        "3t sr",
        "3t superresolution",
        "superresolution",
        "sr",
    }
    if str(task_mode_tag).strip().lower() == "denoising":
        sr_role_label = "Denoised"
        if sr_label.strip().lower() in sr_default_aliases:
            sr_label = "3T Denoised"

    return baseline_label, sr_label, sr_role_label


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


def _sync_axis_limits(
    axes: Sequence[Any],
    axis: str = "y",
    symmetric: bool = False,
    pad_ratio: float = 0.05,
) -> None:
    if axis not in ("x", "y"):
        raise ValueError(f"Unsupported axis: {axis}")

    bounds: List[Tuple[float, float]] = []
    for ax in axes:
        lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
        if np.isfinite(lo) and np.isfinite(hi):
            bounds.append((float(lo), float(hi)))
    if not bounds:
        return

    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    if symmetric:
        lim = max(abs(lo), abs(hi))
        lo, hi = -lim, lim
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0

    pad = max(1e-6, (hi - lo) * float(pad_ratio))
    lo -= pad
    hi += pad
    for ax in axes:
        if axis == "y":
            ax.set_ylim(lo, hi)
        else:
            ax.set_xlim(lo, hi)


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
        "mean": _scipy_mean(v),
        "std": _scipy_std(v),
        "median": _scipy_median(v),
        "p05": _scipy_percentile(v, 5.0),
        "p95": _scipy_percentile(v, 95.0),
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
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    mask_bool = mask > 0.5
    if int(mask_bool.sum()) == 0:
        mask_bool = np.ones_like(mask, dtype=bool)

    channel_names = ["u", "v", "w", "mag"]
    colors = {"ref": REPORT_COLOR_REF, "base": REPORT_COLOR_BASELINE, "sr": REPORT_COLOR_SR}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_f = axes.ravel()
    rows: List[Dict[str, Any]] = []
    channel_hist_figs: Dict[str, str] = {}

    def _plot_hist_panel(ax, ch: str, ref_vals: np.ndarray, base_vals: np.ndarray, sr_vals: np.ndarray, idx_seed: int) -> None:
        sym = ch != "mag"
        vmin, vmax = _robust_range([ref_vals, base_vals, sr_vals], symmetric=sym, lower_q=0.5, upper_q=99.5)
        bin_edges = np.linspace(vmin, vmax, max(20, int(bins)) + 1)
        for vals, label, color, sseed in [
            (ref_vals, ref_label, colors["ref"], 21 + idx_seed),
            (base_vals, baseline_label, colors["base"], 31 + idx_seed),
            (sr_vals, sr_label, colors["sr"], 41 + idx_seed),
        ]:
            vv = np.asarray(vals, dtype=np.float64)
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                continue
            vv = vv[(vv >= vmin) & (vv <= vmax)]
            vv = _subsample_for_plot(vv, seed=sseed)
            if vv.size == 0:
                continue
            if sns is not None:
                sns.histplot(
                    vv,
                    bins=bin_edges,
                    stat="density",
                    element="step",
                    fill=False,
                    linewidth=1.9,
                    color=color,
                    ax=ax,
                    label=label,
                )
            else:
                ax.hist(vv, bins=bin_edges, density=True, histtype="step", linewidth=1.9, alpha=0.95, color=color, label=label)
        ax.set_title(f"{ch.upper()} in-mask voxel distribution")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.28, linestyle="-")
        ax.legend(fontsize=8)

    for c, ch in enumerate(channel_names):
        ax = axes_f[c]
        ref_vals = gt_4ch[:, c][mask_bool]
        base_vals = baseline_4ch[:, c][mask_bool]
        sr_vals = pred_4ch[:, c][mask_bool]

        rows.append(_distribution_row(ch, ref_label, ref_vals))
        rows.append(_distribution_row(ch, baseline_label, base_vals))
        rows.append(_distribution_row(ch, sr_label, sr_vals))
        _plot_hist_panel(ax, ch, ref_vals, base_vals, sr_vals, c)

        fig_ch, ax_ch = plt.subplots(1, 1, figsize=(7.2, 4.8))
        _plot_hist_panel(ax_ch, ch, ref_vals, base_vals, sr_vals, c)
        fig_ch.tight_layout()
        ch_name = f"voxel_histogram_{ch}_in_mask.png"
        fig_ch.savefig(out_path.parent / ch_name, dpi=REPORT_FIG_DPI)
        plt.close(fig_ch)
        channel_hist_figs[ch] = ch_name

    fig.suptitle("Voxel-value distribution inside vessel mask", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
    plt.close(fig)
    return rows, channel_hist_figs


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

    common_xyz = _common_spatial_shape(lr_view, pred_view, gt_view)
    lr_view = _crop_spatial_to_shape(lr_view, common_xyz)
    pred_view = _crop_spatial_to_shape(pred_view, common_xyz)
    gt_view = _crop_spatial_to_shape(gt_view, common_xyz)

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
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
    plt.close(fig)


def _region_label(region_name: str) -> str:
    name = str(region_name).strip().lower()
    if name == "core":
        return "Core"
    if name == "wall":
        return "Wall"
    if name == "intraluminal":
        return "Intraluminal"
    return name.replace("_", " ").title()


def _metric_label(domain: str, component: str = "") -> str:
    d = str(domain or "").strip()
    c = str(component or "").strip()
    if d == "speed_intraluminal":
        base = "Speed (Intraluminal)"
    elif d == "flow_temporal":
        base = "Flow (Temporal)"
    elif d == "velocity_component_peak":
        base = "Velocity Peak"
    elif d == "velocity_component_all_frames":
        base = "Velocity Components (All Frames)"
    else:
        base = d.replace("_", " ").title()
    if c and c.lower() not in {"nan", "none"}:
        return f"{base} ({c})"
    return base


def _save_publication_figures(
    fig_dir: Path,
    baseline_label: str,
    sr_label: str,
    ref_label: str,
    peak_speed_rows: List[Dict[str, Any]],
    mean_speed_rows: List[Dict[str, Any]],
    table2_all_rows: List[Dict[str, Any]],
    table2_temporal_mean_rows: List[Dict[str, Any]],
    corr_rows: List[Dict[str, Any]],
    voxel_dist_rows: List[Dict[str, Any]],
    pvalue_rows: List[Dict[str, Any]],
    q_ref_time: np.ndarray,
    q_base_time: np.ndarray,
    q_sr_time: np.ndarray,
    frame_source_indices: np.ndarray,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    method_colors = {baseline_label: REPORT_COLOR_BASELINE, sr_label: REPORT_COLOR_SR}
    voxel_colors = {ref_label: REPORT_COLOR_REF, baseline_label: REPORT_COLOR_BASELINE, sr_label: REPORT_COLOR_SR}
    velocity_pref = ["Mean velocity [m/s]", "SD velocity [m/s]", "Skewness velocity", "Kurtosis velocity"]
    vorticity_pref = ["Mean vorticity [1/s]", "SD vorticity [1/s]", "Skewness vorticity", "Kurtosis vorticity"]
    moment_specs: List[Tuple[str, str, List[str]]] = [
        ("scale", "Mean & SD", ["Mean", "SD"]),
        ("shape", "Skewness & Kurtosis", ["Skewness", "Kurtosis"]),
    ]

    def _clean_var_name(v: str) -> str:
        return str(v).replace(" [m/s]", "").replace(" [1/s]", "")

    def _family_specs(by_var: Dict[str, Dict[str, np.ndarray]]) -> List[Tuple[str, str, List[str]]]:
        return [
            ("velocity", "Velocity", [v for v in velocity_pref if v in by_var]),
            ("vorticity", "Vorticity", [v for v in vorticity_pref if v in by_var]),
        ]

    def _box_points_plot(
        ax: plt.Axes,
        x_vals: List[str],
        y_vals: List[float],
        h_vals: List[str],
        order_vals: List[str],
    ) -> None:
        if sns is None:
            return
        xx = np.asarray(x_vals, dtype=object)
        yy = np.asarray(y_vals, dtype=np.float64)
        hh = np.asarray(h_vals, dtype=object)
        if yy.size == 0:
            return
        if yy.size > 26000:
            rng = np.random.default_rng(123)
            keep = rng.choice(yy.size, size=26000, replace=False)
            xx_plot = xx[keep]
            yy_plot = yy[keep]
            hh_plot = hh[keep]
        else:
            xx_plot = xx
            yy_plot = yy
            hh_plot = hh

        sns.boxplot(
            x=xx,
            y=yy,
            hue=hh,
            order=order_vals,
            palette=method_colors,
            linewidth=1.0,
            fliersize=0.0,
            saturation=0.62,
            dodge=True,
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_alpha(REPORT_BOX_ALPHA)
        sns.stripplot(
            x=xx_plot,
            y=yy_plot,
            hue=hh_plot,
            order=order_vals,
            palette=method_colors,
            dodge=True,
            jitter=0.18,
            size=2.4,
            alpha=0.28,
            linewidth=0,
            ax=ax,
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    def _save_table2_family_bar(
        by_var: Dict[str, Dict[str, np.ndarray]],
        temporal_mean: bool,
        metric_kind: str,
    ) -> None:
        if not by_var:
            return
        for family_tag, family_title, var_order in _family_specs(by_var):
            if not var_order:
                continue
            fig_w = max(8.4, 1.05 * float(len(var_order)) + 3.2)
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.6))
            x = np.arange(len(var_order), dtype=np.float64)
            width = 0.36
            for j, method_name in enumerate([baseline_label, sr_label]):
                vals = [_scipy_mean(by_var[v][method_name]) for v in var_order]
                bars = ax.bar(
                    x + (j - 0.5) * width,
                    [0.0 if not np.isfinite(v) else v for v in vals],
                    width=width,
                    color=method_colors[method_name],
                    edgecolor=REPORT_COLOR_NEUTRAL,
                    linewidth=0.8,
                    label=method_name,
                    alpha=REPORT_FILL_ALPHA,
                )
                for k, v in enumerate(vals):
                    if not np.isfinite(v):
                        bars[k].set_alpha(0.12)
            ax.set_xticks(x)
            ax.set_xticklabels([_clean_var_name(v) for v in var_order], rotation=15, ha="right")
            tm_prefix = "Temporal-Mean " if temporal_mean else ""
            if metric_kind == "relative":
                ax.set_ylabel("Mean Relative Error [%] vs Reference")
                ax.set_title(f"{tm_prefix}Relative Error by {family_title}")
                slug = "relative_error_pct"
            else:
                ax.set_ylabel("Mean Absolute Error vs Reference")
                ax.set_title(f"{tm_prefix}Absolute Error by {family_title}")
                slug = "abs_error"
            ax.set_xlabel("Variable")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if sns is not None:
                sns.despine(ax=ax)
            ax.legend(title="Method", frameon=False, loc="best")
            fig.tight_layout()
            tm_slug = "temporal_mean_" if temporal_mean else ""
            name = f"prof_table2_{tm_slug}{slug}_bar_{family_tag}.png"
            fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
            plt.close(fig)
            out[f"table2_{tm_slug}{slug}_bar_{family_tag}"] = name

    def _save_table2_family_violin(
        by_var: Dict[str, Dict[str, np.ndarray]],
        temporal_mean: bool,
        metric_kind: str,
    ) -> None:
        if sns is None or not by_var:
            return
        for family_tag, family_title, family_vars in _family_specs(by_var):
            if not family_vars:
                continue
            for moment_tag, moment_title, prefixes in moment_specs:
                var_order = [v for v in family_vars if any(v.startswith(p) for p in prefixes)]
                if not var_order:
                    continue
                display_names = {v: _clean_var_name(v) for v in var_order}
                x_vals: List[str] = []
                y_vals: List[float] = []
                h_vals: List[str] = []
                for var_name in var_order:
                    for method_name in (baseline_label, sr_label):
                        vals = _winsorize_upper_percentile(
                            by_var[var_name][method_name],
                            upper_p=VIOLIN_UPPER_PERCENTILE,
                        )
                        if vals.size == 0:
                            continue
                        x_vals.extend([display_names[var_name]] * int(vals.size))
                        y_vals.extend(vals.astype(np.float64).tolist())
                        h_vals.extend([method_name] * int(vals.size))
                if not y_vals:
                    continue
                fig_w = max(6.8, 1.35 * float(len(var_order)) + 1.8)
                fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.6))
                _box_points_plot(ax, x_vals, y_vals, h_vals, [display_names[v] for v in var_order])
                tm_prefix = "Temporal-Mean " if temporal_mean else ""
                if metric_kind == "relative":
                    ax.set_title(f"{tm_prefix}Relative Error Boxplot ({family_title}: {moment_title})")
                    ax.set_ylabel(f"Relative Error [%] vs Reference (winsorized at P{int(VIOLIN_UPPER_PERCENTILE)})")
                    slug = "relative_error_pct"
                else:
                    ax.set_title(f"{tm_prefix}Absolute Error Boxplot ({family_title}: {moment_title})")
                    ax.set_ylabel(f"Absolute Error vs Reference (winsorized at P{int(VIOLIN_UPPER_PERCENTILE)})")
                    slug = "abs_error"
                ax.set_xlabel("Variable")
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                sns.despine(ax=ax)
                ax.legend(handles=[Patch(facecolor=method_colors[m], edgecolor="#111827", label=m) for m in [baseline_label, sr_label]], title="Method", frameon=False, loc="upper right")
                ax.tick_params(axis="x", rotation=12)
                fig.tight_layout()
                tm_slug = "temporal_mean_" if temporal_mean else ""
                name = f"prof_table2_{tm_slug}{slug}_violin_{family_tag}_{moment_tag}.png"
                fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
                plt.close(fig)
                out[f"table2_{tm_slug}{slug}_violin_{family_tag}_{moment_tag}"] = name

    # 1) Combined panel: mean vs peak velocity errors (MAE, RMSE).
    speed_regions_pref = ["core", "wall", "intraluminal"]
    region_seen: List[str] = []
    for r in mean_speed_rows + peak_speed_rows:
        rr = str(r.get("region", "")).strip().lower()
        if rr and rr not in region_seen:
            region_seen.append(rr)
    region_order = [r for r in speed_regions_pref if r in region_seen] + [r for r in region_seen if r not in speed_regions_pref]
    metric_types: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("Temporal Mean (All Frames)", mean_speed_rows),
        ("Systolic Peak (Peak Frame)", peak_speed_rows),
    ]
    err_metrics: List[Tuple[str, str]] = [("mae", "MAE"), ("rmse", "RMSE")]
    if region_order:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharey="row")
        for col, (metric_type, source_rows) in enumerate(metric_types):
            for row_i, (metric_key, metric_name) in enumerate(err_metrics):
                ax = axes[row_i, col]
                x = np.arange(len(region_order), dtype=np.float64)
                width = 0.36
                any_valid = False
                for j, method_name in enumerate([baseline_label, sr_label]):
                    vals: List[float] = []
                    for region in region_order:
                        vv = [
                            float(rr.get(metric_key, float("nan")))
                            for rr in source_rows
                            if str(rr.get("method", "")) == method_name and str(rr.get("region", "")).strip().lower() == region
                        ]
                        vals.append(_scipy_mean(vv))
                    bars = ax.bar(
                        x + (j - 0.5) * width,
                        [0.0 if not np.isfinite(v) else v for v in vals],
                        width=width,
                        color=method_colors[method_name],
                        edgecolor="#111827",
                        linewidth=0.8,
                        label=method_name,
                        alpha=REPORT_FILL_ALPHA,
                    )
                    for k, v in enumerate(vals):
                        if not np.isfinite(v):
                            bars[k].set_alpha(0.12)
                    any_valid = any_valid or any(np.isfinite(v) for v in vals)
                if any_valid:
                    ax.set_xticks(x)
                    ax.set_xticklabels([_region_label(r) for r in region_order], rotation=0)
                    ax.set_title(f"{metric_name} | {metric_type}")
                    ax.set_ylabel("Error [m/s]")
                    ax.grid(axis="y", linestyle="--", alpha=0.35)
                    if sns is not None:
                        sns.despine(ax=ax)
                else:
                    ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle("Velocity Error: Peak vs Temporal Mean", fontsize=14, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        name = "prof_mean_vs_peak_velocity_errors.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["mean_vs_peak_velocity_errors"] = name
        # Standalone versions (no subplot): one figure per metric-type pair.
        for metric_type, source_rows in metric_types:
            for metric_key, metric_name in err_metrics:
                fig_s, ax_s = plt.subplots(1, 1, figsize=(7.6, 5.1))
                x = np.arange(len(region_order), dtype=np.float64)
                width = 0.36
                for j, method_name in enumerate([baseline_label, sr_label]):
                    vals: List[float] = []
                    for region in region_order:
                        vv = [
                            float(rr.get(metric_key, float("nan")))
                            for rr in source_rows
                            if str(rr.get("method", "")) == method_name and str(rr.get("region", "")).strip().lower() == region
                        ]
                        vals.append(_scipy_mean(vv))
                    bars = ax_s.bar(
                        x + (j - 0.5) * width,
                        [0.0 if not np.isfinite(v) else v for v in vals],
                        width=width,
                        color=method_colors[method_name],
                        edgecolor=REPORT_COLOR_NEUTRAL,
                        linewidth=0.8,
                        label=method_name,
                        alpha=REPORT_FILL_ALPHA,
                    )
                    for k, v in enumerate(vals):
                        if not np.isfinite(v):
                            bars[k].set_alpha(0.12)
                ax_s.set_xticks(x)
                ax_s.set_xticklabels([_region_label(r) for r in region_order], rotation=0)
                ax_s.set_title(f"{metric_name} | {metric_type}")
                ax_s.set_ylabel("Error [m/s]")
                ax_s.grid(axis="y", linestyle="--", alpha=0.35)
                if sns is not None:
                    sns.despine(ax=ax_s)
                ax_s.legend(title="Method", frameon=False, loc="best")
                fig_s.tight_layout()
                slug_type = metric_type.lower().replace(" ", "_").replace("(", "").replace(")", "")
                slug_metric = metric_name.lower().replace(" ", "_")
                name_s = f"prof_velocity_error_{slug_metric}_{slug_type}.png"
                fig_s.savefig(fig_dir / name_s, dpi=REPORT_FIG_DPI)
                plt.close(fig_s)
                out[f"velocity_error_{slug_metric}_{slug_type}"] = name_s

    # 2) Boxplot: slice-wise absolute errors for selected variables.
    vars_of_interest = ["Mean velocity [m/s]", "SD velocity [m/s]", "Mean vorticity [1/s]"]
    t2_by_var: Dict[str, Dict[str, np.ndarray]] = {}
    for v in vars_of_interest:
        base_vals = [
            abs(float(r.get("baseline", float("nan"))) - float(r.get("ref", float("nan"))))
            for r in table2_all_rows
            if str(r.get("variable", "")) == v
        ]
        sr_vals = [
            abs(float(r.get("sr", float("nan"))) - float(r.get("ref", float("nan"))))
            for r in table2_all_rows
            if str(r.get("variable", "")) == v
        ]
        b = _finite_values(base_vals)
        s = _finite_values(sr_vals)
        if b.size > 0 or s.size > 0:
            t2_by_var[v] = {baseline_label: b, sr_label: s}
    if t2_by_var:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5.8))
        vars_order = [v for v in vars_of_interest if v in t2_by_var]
        width = 0.32
        x0 = np.arange(len(vars_order), dtype=np.float64)
        for j, method_name in enumerate([baseline_label, sr_label]):
            pos = x0 + (j - 0.5) * width
            data = [t2_by_var[v][method_name] for v in vars_order]
            for p, arr in zip(pos, data):
                if arr.size == 0:
                    continue
                bp = ax.boxplot(
                    arr,
                    positions=[p],
                    widths=width * 0.92,
                    patch_artist=True,
                    showfliers=True,
                    medianprops={"color": "#111827", "linewidth": 1.2},
                    boxprops={"edgecolor": "#111827", "linewidth": 1.0},
                    whiskerprops={"color": "#111827", "linewidth": 1.0},
                    capprops={"color": "#111827", "linewidth": 1.0},
                    flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": method_colors[method_name], "markeredgecolor": "#111827", "alpha": 0.7},
                )
                for box in bp["boxes"]:
                    box.set_facecolor(method_colors[method_name])
                    box.set_alpha(REPORT_BOX_ALPHA)
        ax.set_xticks(x0)
        ax.set_xticklabels([v.replace(" [m/s]", "").replace(" [1/s]", "") for v in vars_order])
        ax.set_xlabel("Hemodynamic Parameter")
        ax.set_ylabel("Absolute Error (vs reference)")
        ax.set_title("Distribution of Absolute Errors Across All Slices")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
        legend_handles = [
            Patch(facecolor=method_colors[baseline_label], edgecolor="#111827", alpha=REPORT_BOX_ALPHA, label=baseline_label),
            Patch(facecolor=method_colors[sr_label], edgecolor="#111827", alpha=REPORT_BOX_ALPHA, label=sr_label),
        ]
        ax.legend(handles=legend_handles, title="Method", frameon=False, loc="upper right")
        fig.tight_layout()
        name = "prof_slice_relative_errors.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["slice_relative_errors"] = name

    # 2b) Table-2 all-slices error distributions by variable (absolute + relative%).
    t2_all_by_var: Dict[str, Dict[str, np.ndarray]] = {}
    t2_all_rel_by_var: Dict[str, Dict[str, np.ndarray]] = {}
    for row in table2_all_rows:
        var_name = str(row.get("variable", "")).strip()
        if not var_name:
            continue
        if var_name not in t2_all_by_var:
            t2_all_by_var[var_name] = {baseline_label: np.asarray([], dtype=np.float64), sr_label: np.asarray([], dtype=np.float64)}
        if var_name not in t2_all_rel_by_var:
            t2_all_rel_by_var[var_name] = {baseline_label: np.asarray([], dtype=np.float64), sr_label: np.asarray([], dtype=np.float64)}
        ref_v = float(row.get("ref", float("nan")))
        base_v = float(row.get("baseline", float("nan")))
        sr_v = float(row.get("sr", float("nan")))
        re_base = float(row.get("re_baseline_eps", row.get("re_baseline", float("nan"))))
        re_sr = float(row.get("re_sr_eps", row.get("re_sr", float("nan"))))
        b = _finite_values([abs(base_v - ref_v) if np.isfinite(base_v) and np.isfinite(ref_v) else float("nan")])
        s = _finite_values([abs(sr_v - ref_v) if np.isfinite(sr_v) and np.isfinite(ref_v) else float("nan")])
        rb = _finite_values([100.0 * re_base if np.isfinite(re_base) else float("nan")])
        rs = _finite_values([100.0 * re_sr if np.isfinite(re_sr) else float("nan")])
        if b.size > 0:
            t2_all_by_var[var_name][baseline_label] = np.concatenate([t2_all_by_var[var_name][baseline_label], b], axis=0)
        if s.size > 0:
            t2_all_by_var[var_name][sr_label] = np.concatenate([t2_all_by_var[var_name][sr_label], s], axis=0)
        if rb.size > 0:
            t2_all_rel_by_var[var_name][baseline_label] = np.concatenate([t2_all_rel_by_var[var_name][baseline_label], rb], axis=0)
        if rs.size > 0:
            t2_all_rel_by_var[var_name][sr_label] = np.concatenate([t2_all_rel_by_var[var_name][sr_label], rs], axis=0)

    if t2_all_by_var and sns is not None:
        # Strategy: split moments (Mean/SD vs Skewness/Kurtosis) to avoid mixed-scale distortion.
        family_specs: List[Tuple[str, str, List[str]]] = [
            ("velocity", "Velocity", [v for v in velocity_pref if v in t2_all_by_var]),
            ("vorticity", "Vorticity", [v for v in vorticity_pref if v in t2_all_by_var]),
        ]
        for family_tag, family_title, family_vars in family_specs:
            if not family_vars:
                continue
            for moment_tag, moment_title, prefixes in moment_specs:
                var_order = [v for v in family_vars if any(v.startswith(p) for p in prefixes)]
                if not var_order:
                    continue
                display_names = {v: v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order}
                x_vals: List[str] = []
                y_vals: List[float] = []
                h_vals: List[str] = []
                for var_name in var_order:
                    for method_name in (baseline_label, sr_label):
                        vals = _winsorize_upper_percentile(
                            t2_all_by_var[var_name][method_name],
                            upper_p=VIOLIN_UPPER_PERCENTILE,
                        )
                        if vals.size == 0:
                            continue
                        x_vals.extend([display_names[var_name]] * int(vals.size))
                        y_vals.extend(vals.astype(np.float64).tolist())
                        h_vals.extend([method_name] * int(vals.size))

                if not y_vals:
                    continue
                order_disp = [display_names[v] for v in var_order]
                fig_w = max(6.8, 1.35 * float(len(order_disp)) + 1.8)
                fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.6))
                _box_points_plot(ax, x_vals, y_vals, h_vals, order_disp)
                ax.set_title(f"Absolute Error Boxplot ({family_title}: {moment_title})")
                ax.set_xlabel("Variable")
                ax.set_ylabel(f"Absolute Error vs Reference (winsorized at P{int(VIOLIN_UPPER_PERCENTILE)})")
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                sns.despine(ax=ax)
                ax.legend(handles=[Patch(facecolor=method_colors[m], edgecolor="#111827", label=m) for m in [baseline_label, sr_label]], title="Method", frameon=False, loc="upper right")
                ax.tick_params(axis="x", rotation=12)
                fig.tight_layout()
                name = f"prof_table2_abs_error_violin_{family_tag}_{moment_tag}.png"
                fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
                plt.close(fig)
                out[f"table2_abs_error_violin_{family_tag}_{moment_tag}"] = name

    # 2c) Temporal-mean Table-2 plots (averaged over frames first, then compared across slices/variables).
    t2_tm_by_var: Dict[str, Dict[str, np.ndarray]] = {}
    t2_tm_rel_by_var: Dict[str, Dict[str, np.ndarray]] = {}
    for row in table2_temporal_mean_rows:
        var_name = str(row.get("variable", "")).strip()
        if not var_name:
            continue
        if var_name not in t2_tm_by_var:
            t2_tm_by_var[var_name] = {baseline_label: np.asarray([], dtype=np.float64), sr_label: np.asarray([], dtype=np.float64)}
        if var_name not in t2_tm_rel_by_var:
            t2_tm_rel_by_var[var_name] = {baseline_label: np.asarray([], dtype=np.float64), sr_label: np.asarray([], dtype=np.float64)}
        ref_v = float(row.get("ref_mean_over_frames", float("nan")))
        base_v = float(row.get("baseline_mean_over_frames", float("nan")))
        sr_v = float(row.get("sr_mean_over_frames", float("nan")))
        re_base = float(row.get("re_baseline_eps_mean_over_frames", row.get("re_baseline_mean_over_frames", float("nan"))))
        re_sr = float(row.get("re_sr_eps_mean_over_frames", row.get("re_sr_mean_over_frames", float("nan"))))
        b = _finite_values([abs(base_v - ref_v) if np.isfinite(base_v) and np.isfinite(ref_v) else float("nan")])
        s = _finite_values([abs(sr_v - ref_v) if np.isfinite(sr_v) and np.isfinite(ref_v) else float("nan")])
        rb = _finite_values([100.0 * re_base if np.isfinite(re_base) else float("nan")])
        rs = _finite_values([100.0 * re_sr if np.isfinite(re_sr) else float("nan")])
        if b.size > 0:
            t2_tm_by_var[var_name][baseline_label] = np.concatenate([t2_tm_by_var[var_name][baseline_label], b], axis=0)
        if s.size > 0:
            t2_tm_by_var[var_name][sr_label] = np.concatenate([t2_tm_by_var[var_name][sr_label], s], axis=0)
        if rb.size > 0:
            t2_tm_rel_by_var[var_name][baseline_label] = np.concatenate([t2_tm_rel_by_var[var_name][baseline_label], rb], axis=0)
        if rs.size > 0:
            t2_tm_rel_by_var[var_name][sr_label] = np.concatenate([t2_tm_rel_by_var[var_name][sr_label], rs], axis=0)

    if t2_tm_by_var:
        tm_var_order = sorted(
            list(t2_tm_by_var.keys()),
            key=lambda v: _scipy_mean(t2_tm_by_var[v][baseline_label]) if np.isfinite(_scipy_mean(t2_tm_by_var[v][baseline_label])) else -1e9,
            reverse=True,
        )

        # Summary bars: mean absolute error across slices from temporal-mean table.
        fig_h = max(5.6, 0.58 * float(len(tm_var_order)) + 1.4)
        fig, ax = plt.subplots(1, 1, figsize=(12.2, fig_h))
        y = np.arange(len(tm_var_order), dtype=np.float64)
        height = 0.36
        vals_base = [_scipy_mean(t2_tm_by_var[v][baseline_label]) for v in tm_var_order]
        vals_sr = [_scipy_mean(t2_tm_by_var[v][sr_label]) for v in tm_var_order]
        bars_base = ax.barh(
            y + 0.5 * height,
            [0.0 if not np.isfinite(v) else v for v in vals_base],
            height=height,
            color=method_colors[baseline_label],
            edgecolor=REPORT_COLOR_NEUTRAL,
            linewidth=0.8,
            alpha=REPORT_FILL_ALPHA,
            label=f"{baseline_label} mean absolute error (temporal-mean)",
        )
        bars_sr = ax.barh(
            y - 0.5 * height,
            [0.0 if not np.isfinite(v) else v for v in vals_sr],
            height=height,
            color=method_colors[sr_label],
            edgecolor=REPORT_COLOR_NEUTRAL,
            linewidth=0.8,
            alpha=REPORT_FILL_ALPHA,
            label=f"{sr_label} mean absolute error (temporal-mean)",
        )
        for k, v in enumerate(vals_base):
            if not np.isfinite(v):
                bars_base[k].set_alpha(0.12)
        for k, v in enumerate(vals_sr):
            if not np.isfinite(v):
                bars_sr[k].set_alpha(0.12)
        ax.set_yticks(y)
        ax.set_yticklabels(tm_var_order)
        ax.set_xlabel("Mean absolute error vs reference (temporal-mean across frames)")
        ax.set_title("Temporal-Mean Absolute Error by Variable")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        name = "prof_table2_temporal_mean_relative_error_bar.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["table2_temporal_mean_relative_error_bar"] = name

        # Violin over slices of temporal-mean absolute error split by moment family.
        if sns is not None:
            tm_family_specs: List[Tuple[str, str, List[str]]] = [
                ("velocity", "Velocity", [v for v in velocity_pref if v in tm_var_order]),
                ("vorticity", "Vorticity", [v for v in vorticity_pref if v in tm_var_order]),
            ]
            for family_tag, family_title, family_vars in tm_family_specs:
                if not family_vars:
                    continue
                for moment_tag, moment_title, prefixes in moment_specs:
                    var_order = [v for v in family_vars if any(v.startswith(p) for p in prefixes)]
                    if not var_order:
                        continue
                    display_names = {v: v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order}
                    x_vals_tm: List[str] = []
                    y_vals_tm: List[float] = []
                    h_vals_tm: List[str] = []
                    for var_name in var_order:
                        for method_name in (baseline_label, sr_label):
                            vals = _winsorize_upper_percentile(
                                t2_tm_by_var[var_name][method_name],
                                upper_p=VIOLIN_UPPER_PERCENTILE,
                            )
                            if vals.size == 0:
                                continue
                            x_vals_tm.extend([display_names[var_name]] * int(vals.size))
                            y_vals_tm.extend(vals.astype(np.float64).tolist())
                            h_vals_tm.extend([method_name] * int(vals.size))

                    if not y_vals_tm:
                        continue
                    order_disp_tm = [display_names[v] for v in var_order]
                    fig_w_vi = max(6.8, 1.35 * float(len(order_disp_tm)) + 1.8)
                    fig, ax = plt.subplots(1, 1, figsize=(fig_w_vi, 5.6))
                    _box_points_plot(ax, x_vals_tm, y_vals_tm, h_vals_tm, order_disp_tm)
                    ax.set_title(f"Temporal-Mean Absolute Error Boxplot ({family_title}: {moment_title})")
                    ax.set_xlabel("Variable")
                    ax.set_ylabel(
                        f"Absolute Error vs Reference (temporal-mean over frames, winsorized at P{int(VIOLIN_UPPER_PERCENTILE)})"
                    )
                    ax.grid(axis="y", linestyle="--", alpha=0.35)
                    sns.despine(ax=ax)
                    ax.legend(handles=[Patch(facecolor=method_colors[m], edgecolor="#111827", label=m) for m in [baseline_label, sr_label]], title="Method", frameon=False, loc="upper right")
                    ax.tick_params(axis="x", rotation=12)
                    fig.tight_layout()
                    name = f"prof_table2_temporal_mean_abs_error_violin_{family_tag}_{moment_tag}.png"
                    fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
                    plt.close(fig)
                    out[f"table2_temporal_mean_abs_error_violin_{family_tag}_{moment_tag}"] = name

    # 2d) Additional Table-2 family figures requested for paper: absolute/relative, bars + violins.
    _save_table2_family_bar(t2_all_by_var, temporal_mean=False, metric_kind="absolute")
    _save_table2_family_bar(t2_all_rel_by_var, temporal_mean=False, metric_kind="relative")
    _save_table2_family_violin(t2_all_rel_by_var, temporal_mean=False, metric_kind="relative")
    _save_table2_family_bar(t2_tm_by_var, temporal_mean=True, metric_kind="absolute")
    _save_table2_family_bar(t2_tm_rel_by_var, temporal_mean=True, metric_kind="relative")
    _save_table2_family_violin(t2_tm_rel_by_var, temporal_mean=True, metric_kind="relative")

    # 3) Correlation and RMSE barplots.
    corr_filtered: List[Dict[str, Any]] = []
    for r in corr_rows:
        method_name = str(r.get("method", ""))
        if method_name not in method_colors:
            continue
        n_val = float(r.get("n", float("nan")))
        if not np.isfinite(n_val) or n_val <= 100:
            continue
        corr_filtered.append(r)

    if corr_filtered:
        var_order: List[str] = []
        agg: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
        for r in corr_filtered:
            method_name = str(r.get("method", ""))
            label = _metric_label(str(r.get("domain", "")), str(r.get("component", "")))
            if label not in var_order:
                var_order.append(label)
            key = (label, method_name)
            agg.setdefault(key, {"pearson_r": [], "rmse": []})
            agg[key]["pearson_r"].append(float(r.get("pearson_r", float("nan"))))
            agg[key]["rmse"].append(float(r.get("rmse", float("nan"))))

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        for ax, metric_key, ylab, title in [
            (axes[0], "pearson_r", "Pearson r", "Pearson Correlation Coefficient"),
            (axes[1], "rmse", "RMSE [m/s]", "Root Mean Square Error (RMSE)"),
        ]:
            x = np.arange(len(var_order), dtype=np.float64)
            width = 0.36
            for j, method_name in enumerate([baseline_label, sr_label]):
                vals = [
                    _scipy_mean(agg.get((vv, method_name), {}).get(metric_key, []))
                    for vv in var_order
                ]
                bars = ax.bar(
                    x + (j - 0.5) * width,
                    [0.0 if not np.isfinite(v) else v for v in vals],
                    width=width,
                    color=method_colors[method_name],
                    edgecolor="#111827",
                    linewidth=0.8,
                    label=method_name,
                    alpha=REPORT_FILL_ALPHA,
                )
                for k, v in enumerate(vals):
                    if not np.isfinite(v):
                        bars[k].set_alpha(0.12)
            ax.set_xticks(x)
            ax.set_xticklabels(var_order, rotation=35, ha="right")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if sns is not None:
                sns.despine(ax=ax)
        axes[1].legend(title="Method", frameon=False, loc="best")
        fig.tight_layout()
        name = "prof_correlation_rmse.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["correlation_rmse"] = name
        # Standalone versions (no subplot)
        for metric_key, ylab, title, out_key, out_name in [
            ("pearson_r", "Pearson r", "Pearson Correlation Coefficient", "correlation_pearson_r", "prof_correlation_pearson_r.png"),
            ("rmse", "RMSE [m/s]", "Root Mean Square Error (RMSE)", "correlation_rmse_only", "prof_correlation_rmse_only.png"),
        ]:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(8.8, 5.1))
            x = np.arange(len(var_order), dtype=np.float64)
            width = 0.36
            for j, method_name in enumerate([baseline_label, sr_label]):
                vals = [_scipy_mean(agg.get((vv, method_name), {}).get(metric_key, [])) for vv in var_order]
                bars = ax_s.bar(
                    x + (j - 0.5) * width,
                    [0.0 if not np.isfinite(v) else v for v in vals],
                    width=width,
                    color=method_colors[method_name],
                    edgecolor=REPORT_COLOR_NEUTRAL,
                    linewidth=0.8,
                    label=method_name,
                    alpha=REPORT_FILL_ALPHA,
                )
                for k, v in enumerate(vals):
                    if not np.isfinite(v):
                        bars[k].set_alpha(0.12)
            ax_s.set_xticks(x)
            ax_s.set_xticklabels(var_order, rotation=35, ha="right")
            ax_s.set_ylabel(ylab)
            ax_s.set_title(title)
            ax_s.grid(axis="y", linestyle="--", alpha=0.35)
            if sns is not None:
                sns.despine(ax=ax_s)
            ax_s.legend(title="Method", frameon=False, loc="best")
            fig_s.tight_layout()
            fig_s.savefig(fig_dir / out_name, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            out[out_key] = out_name

    # 4) Voxel distribution standard deviation barplot.
    channel_pref = ["u", "v", "w", "mag"]
    channel_seen: List[str] = []
    vox_agg: Dict[Tuple[str, str], List[float]] = {}
    for r in voxel_dist_rows:
        ch = str(r.get("channel", "")).strip().lower()
        method_name = str(r.get("method", ""))
        std_val = float(r.get("std", float("nan")))
        if not ch:
            continue
        if method_name not in voxel_colors:
            continue
        if ch not in channel_seen:
            channel_seen.append(ch)
        vox_agg.setdefault((ch, method_name), []).append(std_val)
    channel_order = [c for c in channel_pref if c in channel_seen] + [c for c in channel_seen if c not in channel_pref]
    if channel_order:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.5))
        methods_in_vox = [m for m in [ref_label, baseline_label, sr_label] if any((c, m) in vox_agg for c in channel_order)]
        x = np.arange(len(channel_order), dtype=np.float64)
        width = 0.80 / max(1, len(methods_in_vox))
        center_shift = (len(methods_in_vox) - 1) * 0.5
        for j, method_name in enumerate(methods_in_vox):
            vals = [_scipy_mean(vox_agg.get((ch, method_name), [])) for ch in channel_order]
            bars = ax.bar(
                x + (j - center_shift) * width,
                [0.0 if not np.isfinite(v) else v for v in vals],
                width=width * 0.96,
                color=voxel_colors[method_name],
                edgecolor="#111827",
                linewidth=0.8,
                label=method_name,
                alpha=REPORT_FILL_ALPHA,
            )
            for k, v in enumerate(vals):
                if not np.isfinite(v):
                    bars[k].set_alpha(0.12)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in channel_order])
        ax.set_xlabel("Velocity Component Channel")
        ax.set_ylabel("Standard Deviation [m/s]")
        ax.set_title("Velocity Variance (Voxel Distribution Std Dev)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
        ax.legend(title="Method", frameon=False, loc="best")
        fig.tight_layout()
        name = "prof_voxel_distribution_std.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["voxel_distribution_std"] = name

    # 5) Relative error (%) by region for peak and temporal mean.
    if region_order:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharey=True)
        for ax, metric_type, source_rows in [
            (axes[0], "Peak Frame", peak_speed_rows),
            (axes[1], "Temporal Mean", mean_speed_rows),
        ]:
            x = np.arange(len(region_order), dtype=np.float64)
            width = 0.36
            for j, method_name in enumerate([baseline_label, sr_label]):
                vals: List[float] = []
                for region in region_order:
                    vv = [
                        float(rr.get("relative_error_pct", float("nan")))
                        for rr in source_rows
                        if str(rr.get("method", "")) == method_name and str(rr.get("region", "")).strip().lower() == region
                    ]
                    vals.append(_scipy_mean(vv))
                bars = ax.bar(
                    x + (j - 0.5) * width,
                    [0.0 if not np.isfinite(v) else v for v in vals],
                    width=width,
                    color=method_colors[method_name],
                    edgecolor=REPORT_COLOR_NEUTRAL,
                    linewidth=0.8,
                    alpha=REPORT_FILL_ALPHA,
                    label=method_name,
                )
                for k, v in enumerate(vals):
                    if not np.isfinite(v):
                        bars[k].set_alpha(0.12)
            ax.set_xticks(x)
            ax.set_xticklabels([_region_label(r) for r in region_order], rotation=0)
            ax.set_title(f"Relative Error (%) | {metric_type}")
            ax.set_ylabel("Relative Error [%]")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if sns is not None:
                sns.despine(ax=ax)
        axes[1].legend(title="Method", frameon=False, loc="best")
        fig.tight_layout()
        name = "prof_velocity_relative_error_pct.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["velocity_relative_error_pct"] = name
        # Standalone versions (no subplot)
        for metric_type, source_rows, out_key, out_name in [
            ("Peak Frame", peak_speed_rows, "velocity_relative_error_pct_peak", "prof_velocity_relative_error_pct_peak.png"),
            ("Temporal Mean", mean_speed_rows, "velocity_relative_error_pct_temporal_mean", "prof_velocity_relative_error_pct_temporal_mean.png"),
        ]:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(7.8, 4.9))
            x = np.arange(len(region_order), dtype=np.float64)
            width = 0.36
            for j, method_name in enumerate([baseline_label, sr_label]):
                vals: List[float] = []
                for region in region_order:
                    vv = [
                        float(rr.get("relative_error_pct", float("nan")))
                        for rr in source_rows
                        if str(rr.get("method", "")) == method_name and str(rr.get("region", "")).strip().lower() == region
                    ]
                    vals.append(_scipy_mean(vv))
                bars = ax_s.bar(
                    x + (j - 0.5) * width,
                    [0.0 if not np.isfinite(v) else v for v in vals],
                    width=width,
                    color=method_colors[method_name],
                    edgecolor=REPORT_COLOR_NEUTRAL,
                    linewidth=0.8,
                    alpha=REPORT_FILL_ALPHA,
                    label=method_name,
                )
                for k, v in enumerate(vals):
                    if not np.isfinite(v):
                        bars[k].set_alpha(0.12)
            ax_s.set_xticks(x)
            ax_s.set_xticklabels([_region_label(r) for r in region_order], rotation=0)
            ax_s.set_title(f"Relative Error (%) | {metric_type}")
            ax_s.set_ylabel("Relative Error [%]")
            ax_s.grid(axis="y", linestyle="--", alpha=0.35)
            if sns is not None:
                sns.despine(ax=ax_s)
            ax_s.legend(title="Method", frameon=False, loc="best")
            fig_s.tight_layout()
            fig_s.savefig(fig_dir / out_name, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            out[out_key] = out_name

    # 6) Temporal flow absolute error profile by frame.
    q_ref = np.asarray(q_ref_time, dtype=np.float64)
    q_base = np.asarray(q_base_time, dtype=np.float64)
    q_sr = np.asarray(q_sr_time, dtype=np.float64)
    frame_idx = np.asarray(frame_source_indices, dtype=np.int32)
    valid_flow = (
        q_ref.size > 0
        and q_ref.shape == q_base.shape
        and q_ref.shape == q_sr.shape
        and frame_idx.shape[0] == q_ref.shape[0]
    )
    if valid_flow:
        err_base = np.abs(q_base - q_ref)
        err_sr = np.abs(q_sr - q_ref)
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.7))
        ax.plot(frame_idx, err_base, marker="o", markersize=3.8, linewidth=2.0, color=REPORT_COLOR_BASELINE, label=f"{baseline_label} |Q-Qref|")
        ax.plot(frame_idx, err_sr, marker="o", markersize=3.8, linewidth=2.0, color=REPORT_COLOR_SR, label=f"{sr_label} |Q-Qref|")
        ax.set_xlabel("Temporal frame index")
        ax.set_ylabel("Absolute Error [ml/s]")
        ax.set_title("Temporal Flow Absolute Error Profile")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        name = "prof_flow_abs_error_over_time.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["flow_abs_error_over_time"] = name

    # 7) Statistical significance summary: -log10(p) by analysis and region.
    pval_agg: Dict[Tuple[str, str], List[float]] = {}
    analysis_order: List[str] = []
    pval_region_order: List[str] = []
    for row in pvalue_rows:
        analysis = str(row.get("analysis", "")).strip()
        region = str(row.get("region", "")).strip().lower()
        p_raw = float(row.get("wilcoxon_p_value", float("nan")))
        if not analysis or not region or not np.isfinite(p_raw):
            continue
        if analysis not in analysis_order:
            analysis_order.append(analysis)
        if region not in pval_region_order:
            pval_region_order.append(region)
        pval_agg.setdefault((analysis, region), []).append(p_raw)
    if pval_agg:
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0))
        x = np.arange(len(pval_region_order), dtype=np.float64)
        width = 0.82 / max(1, len(analysis_order))
        center_shift = (len(analysis_order) - 1) * 0.5
        analysis_palette = sns.color_palette("deep", n_colors=max(2, len(analysis_order))) if sns is not None else ["#4b5563", "#111827"]
        for j, analysis in enumerate(analysis_order):
            vals = []
            for region in pval_region_order:
                pp = _scipy_mean(pval_agg.get((analysis, region), []))
                vals.append(float(-np.log10(max(pp, 1e-300))) if np.isfinite(pp) and pp > 0 else float("nan"))
            bars = ax.bar(
                x + (j - center_shift) * width,
                [0.0 if not np.isfinite(v) else v for v in vals],
                width=width * 0.96,
                color=analysis_palette[j % len(analysis_palette)],
                edgecolor=REPORT_COLOR_NEUTRAL,
                linewidth=0.8,
                alpha=REPORT_FILL_ALPHA,
                label=analysis,
            )
            for k, v in enumerate(vals):
                if not np.isfinite(v):
                    bars[k].set_alpha(0.12)
        p_thr = -np.log10(0.05)
        ax.axhline(p_thr, color=REPORT_COLOR_NEUTRAL, linestyle="--", linewidth=1.1, label="p=0.05")
        ax.set_xticks(x)
        ax.set_xticklabels([_region_label(r) for r in pval_region_order])
        ax.set_ylabel("-log10(p-value)")
        ax.set_xlabel("Region")
        ax.set_title("Wilcoxon Significance Summary")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        name = "prof_significance_pvalues.png"
        fig.savefig(fig_dir / name, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        out["significance_pvalues"] = name

    return out


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
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
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
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
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
        "mean_angle_deg": _scipy_mean(ang) if ang.size > 0 else float("nan"),
        "p95_angle_deg": _scipy_percentile(ang, 95.0) if ang.size > 0 else float("nan"),
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
                "sign_agreement_pct": float(100.0 * _scipy_mean(signs_ok.astype(np.float64))),
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
        ax.scatter(x2, y2, s=14, marker="o", c="#9ca3af", alpha=0.75, edgecolors="white", linewidths=0.2)
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
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
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
    fig.savefig(out_path, dpi=REPORT_FIG_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a professional uncertainty-quantification report from inference payload "
            "(full-volume prediction, visual inspection figures, and paper-style metrics)."
        )
    )
    parser.add_argument("--payload-npz", required=True, help="Path to analysis_payload.npz produced by run_sr_inference_case.py")
    parser.add_argument(
        "--baseline-payload-npz",
        default="",
        help=(
            "Optional analysis_payload.npz used only for baseline 3T (lr_norm + venc). "
            "Useful to keep identical 3T baseline across workflows (e.g., use denoising baseline for superresolution)."
        ),
    )
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
    parser.add_argument(
        "--task-mode",
        type=str,
        default="auto",
        choices=["auto", "denoising", "superresolution"],
        help="Naming convention for outputs. auto infers from res_increase (1=denoising, >1=superresolution).",
    )
    parser.add_argument("--report-title", default="4D Flow SR Uncertainty Quantification Report", help="Report title")

    args = parser.parse_args()
    if (args.centerline_start_xyz is None) != (args.centerline_end_xyz is None):
        raise ValueError("Use --centerline-start-xyz and --centerline-end-xyz together, or omit both.")

    payload = _load_payload(args.payload_npz)
    baseline_payload = payload
    baseline_payload_path = str(Path(args.payload_npz).resolve())
    baseline_source_tag = "main_payload_lr"
    if str(args.baseline_payload_npz).strip():
        bpath = Path(args.baseline_payload_npz).resolve()
        baseline_payload = _load_payload(str(bpath))
        baseline_payload_path = str(bpath)
        baseline_source_tag = "external_payload_lr"
        if "lr_norm" not in baseline_payload:
            raise ValueError(f"Baseline payload missing 'lr_norm': {bpath}")
        if "venc" not in baseline_payload:
            raise ValueError(f"Baseline payload missing 'venc': {bpath}")
    metadata = {}
    if args.metadata_json:
        mpath = Path(args.metadata_json)
        if mpath.exists():
            metadata = json.loads(mpath.read_text())

    task_mode_tag, task_mode_label, detected_res_increase = _resolve_task_mode(
        task_mode_arg=args.task_mode,
        metadata=metadata,
        payload=payload,
    )
    baseline_label, sr_label, sr_role_label = _resolve_method_labels(
        task_mode_tag=task_mode_tag,
        baseline_label_arg=args.baseline_label,
        sr_label_arg=args.sr_label,
    )
    metrics_rel_prefix = f"metrics/{task_mode_tag}"

    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    metrics_root = out_dir / "metrics"
    fig_mode_dir = fig_dir / task_mode_tag
    metrics_dir = metrics_root / task_mode_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_mode_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_groups: Dict[str, Path] = {
        "comparison": fig_mode_dir / "comparison",
        "distribution": fig_mode_dir / "distribution",
        "flow": fig_mode_dir / "flow",
        "correlation": fig_mode_dir / "correlation",
        "bland_altman": fig_mode_dir / "bland_altman",
        "centerline": fig_mode_dir / "centerline",
        "wss": fig_mode_dir / "wss",
        "publication": fig_mode_dir / "publication",
    }
    for p in fig_groups.values():
        p.mkdir(parents=True, exist_ok=True)

    ref_label = args.ref_label
    if str(ref_label).lower() == "auto":
        ref_label = _autodetect_reference_label(metadata)

    lr_norm_inference = payload["lr_norm"].astype(np.float32)  # [T,6,X,Y,Z] from inference payload
    lr_norm = baseline_payload["lr_norm"].astype(np.float32)  # [T,6,X,Y,Z] baseline source used for metrics/plots
    pred_norm = payload["pred_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    gt_norm = payload["gt_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    mask = payload["mask"].astype(np.float32)  # [T,X,Y,Z]
    venc = payload["venc"].astype(np.float32)  # [T] for pred/gt denormalization
    venc_baseline = baseline_payload["venc"].astype(np.float32)  # [T] for baseline denormalization
    hr_spacing = tuple(float(x) for x in payload["hr_spacing"].tolist())
    if lr_norm.ndim != 5 or lr_norm.shape[1] < 3:
        raise ValueError(f"Baseline lr_norm must be [T,C,X,Y,Z] with C>=3. Got {lr_norm.shape}")

    spatial_shapes_original = {
        "pred_xyz": [int(x) for x in pred_norm.shape[-3:]],
        "gt_xyz": [int(x) for x in gt_norm.shape[-3:]],
        "mask_xyz": [int(x) for x in mask.shape[-3:]],
    }
    common_xyz = _common_spatial_shape(pred_norm, gt_norm, mask)
    pred_norm = _crop_spatial_to_shape(pred_norm, common_xyz)
    gt_norm = _crop_spatial_to_shape(gt_norm, common_xyz)
    mask = _crop_spatial_to_shape(mask, common_xyz)
    spatial_shapes_cropped = {
        "pred_xyz": [int(x) for x in pred_norm.shape[-3:]],
        "gt_xyz": [int(x) for x in gt_norm.shape[-3:]],
        "mask_xyz": [int(x) for x in mask.shape[-3:]],
    }
    spatial_crop_applied = any(
        spatial_shapes_original[k] != spatial_shapes_cropped[k] for k in spatial_shapes_original.keys()
    )

    if pred_norm.shape[1] != 4 or gt_norm.shape[1] != 4:
        raise ValueError(f"Expected 4-channel pred/gt tensors. Got pred={pred_norm.shape}, gt={gt_norm.shape}")

    t_count = pred_norm.shape[0]
    if "frame_indices" in payload and np.asarray(payload["frame_indices"]).shape[0] == t_count:
        frame_source_indices = np.asarray(payload["frame_indices"], dtype=np.int32)
    else:
        frame_source_indices = np.arange(t_count, dtype=np.int32)
    baseline_t = int(lr_norm.shape[0])
    if baseline_t <= 0:
        raise ValueError(f"Baseline lr_norm is empty. shape={lr_norm.shape}")
    baseline_frame_source_indices: np.ndarray
    if "frame_indices" in baseline_payload and np.asarray(baseline_payload["frame_indices"]).shape[0] == baseline_t:
        baseline_frame_source_indices = np.asarray(baseline_payload["frame_indices"], dtype=np.int32)
    else:
        baseline_frame_source_indices = np.arange(baseline_t, dtype=np.int32)

    # Align baseline temporal axis to the report payload frame order.
    if baseline_t != t_count or not np.array_equal(baseline_frame_source_indices, frame_source_indices):
        src_to_pos = {int(src): i for i, src in enumerate(baseline_frame_source_indices.tolist())}
        if not all(int(src) in src_to_pos for src in frame_source_indices.tolist()):
            raise ValueError(
                "Baseline payload frames do not cover report payload frame indices. "
                f"report={frame_source_indices.tolist()} baseline={baseline_frame_source_indices.tolist()}"
            )
        pick_idx = np.asarray([src_to_pos[int(src)] for src in frame_source_indices.tolist()], dtype=np.int64)
        lr_norm = lr_norm[pick_idx]
        if venc_baseline.shape[0] == baseline_t:
            venc_baseline = venc_baseline[pick_idx]
        baseline_t = int(lr_norm.shape[0])
        baseline_frame_source_indices = frame_source_indices.copy()
    if baseline_t != t_count:
        raise ValueError(
            f"Baseline payload temporal length ({baseline_t}) does not match report payload ({t_count}) after alignment."
        )
    if venc_baseline.shape[0] != t_count:
        raise ValueError(
            f"Baseline venc length ({venc_baseline.shape[0]}) does not match temporal length ({t_count})."
        )
    fidx = int(np.clip(args.selected_frame, 0, t_count - 1))

    # Denormalize to physical units for velocity-related metrics
    pred_phys = pred_norm.copy()
    gt_phys = gt_norm.copy()
    lr_vel_phys = lr_norm[:, :3].copy()
    for t in range(t_count):
        pred_phys[t, :3] *= float(venc[t])
        gt_phys[t, :3] *= float(venc[t])
        lr_vel_phys[t] *= float(venc_baseline[t])

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

    # Derive LR 4-channel display tensor (u,v,w,mag) from available baseline channels.
    if lr_norm.shape[1] >= 6:
        lr_mag_single = lr_norm[:, 3 + int(args.lr_mag_channel)].astype(np.float32)
    elif lr_norm.shape[1] >= 4:
        lr_mag_single = lr_norm[:, 3].astype(np.float32)
    else:
        lr_mag_single = np.sqrt(np.maximum(0.0, (lr_norm[:, 0] ** 2 + lr_norm[:, 1] ** 2 + lr_norm[:, 2] ** 2))).astype(np.float32)
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
        out_img = fig_groups["comparison"] / f"channel_{name}_comparison.png"
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
        channel_figs[name] = _rel_fig_path(out_img, fig_dir)

    fig_voxel_hist = fig_groups["distribution"] / "voxel_histogram_in_mask.png"
    voxel_dist_rows, voxel_hist_channel_figs = _save_voxel_histograms(
        out_path=fig_voxel_hist,
        baseline_4ch=lr_up,
        pred_4ch=pred_norm,
        gt_4ch=gt_norm,
        mask=mask_metrics,
        bins=int(args.hist_bins),
        baseline_label=baseline_label,
        sr_label=sr_label,
        ref_label=ref_label,
    )
    voxel_hist_channel_figs = {
        k: _rel_fig_path(fig_voxel_hist.parent / Path(v), fig_dir) for k, v in voxel_hist_channel_figs.items()
    }
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
    ref_eps_map = _table2_ref_eps_map(table2_all, ref_key="ref", q=5.0)
    _augment_table2_rows_with_robust_relative(
        table2_all,
        eps_map=ref_eps_map,
        ref_key="ref",
        baseline_key="baseline",
        sr_key="sr",
    )
    _augment_table2_rows_with_robust_relative(
        table2_per_frame,
        eps_map=ref_eps_map,
        ref_key="ref",
        baseline_key="baseline",
        sr_key="sr",
    )
    table2_temporal_mean = _table2_temporal_mean_rows(table2_per_frame)
    table2_relative_robust_summary = _table2_relative_robust_summary(
        table2_all,
        ref_key="ref",
        baseline_key="baseline",
        sr_key="sr",
        re_base_key="re_baseline",
        re_sr_key="re_sr",
        re_base_eps_key="re_baseline_eps",
        re_sr_eps_key="re_sr_eps",
        smape_base_key="smape_baseline",
        smape_sr_key="smape_sr",
        ref_eps_key="ref_eps",
        baseline_method_label=baseline_label,
        sr_method_label=sr_label,
    )
    table2_temporal_mean_relative_robust_summary = _table2_relative_robust_summary(
        table2_temporal_mean,
        ref_key="ref_mean_over_frames",
        baseline_key="baseline_mean_over_frames",
        sr_key="sr_mean_over_frames",
        re_base_key="re_baseline_mean_over_frames",
        re_sr_key="re_sr_mean_over_frames",
        re_base_eps_key="re_baseline_eps_mean_over_frames",
        re_sr_eps_key="re_sr_eps_mean_over_frames",
        smape_base_key="smape_baseline_mean_over_frames",
        smape_sr_key="smape_sr_mean_over_frames",
        ref_eps_key="ref_eps_mean_over_frames",
        baseline_method_label=baseline_label,
        sr_method_label=sr_label,
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
    t2_cols = [
        "slice_index",
        "variable",
        "ref",
        "baseline",
        "sr",
        "re_baseline",
        "re_sr",
        "re_baseline_eps",
        "re_sr_eps",
        "smape_baseline",
        "smape_sr",
        "ref_eps",
    ]
    _write_csv(metrics_dir / "table2_like_all_slices.csv", table2_all, t2_cols)

    t2pf_cols = [
        "frame_payload_index",
        "frame_source_index",
        "slice_index",
        "variable",
        "ref",
        "baseline",
        "sr",
        "re_baseline",
        "re_sr",
        "re_baseline_eps",
        "re_sr_eps",
        "smape_baseline",
        "smape_sr",
        "ref_eps",
    ]
    _write_csv(metrics_dir / "table2_like_per_frame_all_slices.csv", table2_per_frame, t2pf_cols)

    t2tm_cols = [
        "slice_index",
        "variable",
        "n_frames",
        "ref_mean_over_frames",
        "baseline_mean_over_frames",
        "sr_mean_over_frames",
        "re_baseline_mean_over_frames",
        "re_sr_mean_over_frames",
        "re_baseline_eps_mean_over_frames",
        "re_sr_eps_mean_over_frames",
        "smape_baseline_mean_over_frames",
        "smape_sr_mean_over_frames",
        "ref_eps_mean_over_frames",
    ]
    _write_csv(metrics_dir / "table2_like_temporal_mean.csv", table2_temporal_mean, t2tm_cols)

    t2c_cols = [
        "location",
        "slice_index",
        "variable",
        "ref",
        "baseline",
        "sr",
        "re_baseline",
        "re_sr",
        "re_baseline_eps",
        "re_sr_eps",
        "smape_baseline",
        "smape_sr",
        "ref_eps",
    ]
    _write_csv(metrics_dir / "table2_like_compact.csv", table2_compact, t2c_cols)
    _write_csv(
        metrics_dir / "table2_relative_robust_summary.csv",
        table2_relative_robust_summary,
        [
            "variable",
            "method",
            "n",
            "ref_eps",
            "mean_re_pct",
            "median_re_pct",
            "p95_re_pct",
            "mean_re_eps_pct",
            "median_re_eps_pct",
            "p95_re_eps_pct",
            "mean_smape_pct",
            "median_smape_pct",
            "p95_smape_pct",
            "wape_pct",
            "wape_eps_pct",
        ],
    )
    _write_csv(
        metrics_dir / "table2_temporal_mean_relative_robust_summary.csv",
        table2_temporal_mean_relative_robust_summary,
        [
            "variable",
            "method",
            "n",
            "ref_eps",
            "mean_re_pct",
            "median_re_pct",
            "p95_re_pct",
            "mean_re_eps_pct",
            "median_re_eps_pct",
            "p95_re_eps_pct",
            "mean_smape_pct",
            "median_smape_pct",
            "p95_smape_pct",
            "wape_pct",
            "wape_eps_pct",
        ],
    )

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
            q_arr[~finite] = _scipy_median(q_arr[finite])

    if np.isfinite(float(args.q_ref)):
        q_ref_scalar = float(args.q_ref)
    else:
        q_ref_scalar = _scipy_median(q_ref_time[np.isfinite(q_ref_time)])

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
        mad = _scipy_mean(np.abs(q_sel - q_ref_scalar))
        return {
            "method": name,
            "mean_Q_ml_s": _scipy_mean(q_sel),
            "temporal_SD_Q_ml_s": _scipy_std(q_sel),
            "MAD_Q_vs_qref_ml_s": mad,
            "MAD_Q_vs_qref_pct": 100.0 * mad / (abs(q_ref_scalar) + 1e-12),
        }

    all_idx = np.arange(q_ref_time.shape[0], dtype=np.int32)
    flow_rows = [
        flow_summary(ref_label, q_ref_time, all_idx),
        flow_summary(baseline_label, q_base_time, all_idx),
        flow_summary(sr_label, q_sr_time, all_idx),
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
                        {**flow_summary(baseline_label, q_base_time, win_idx), "method": f"{baseline_label} (window)"},
                        {**flow_summary(sr_label, q_sr_time, win_idx), "method": f"{sr_label} (window)"},
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
        fig_centerline = fig_groups["centerline"] / "centerline_overlay.png"
        _save_centerline_overlay_figure(
            out_path=fig_centerline,
            mask_3d=centerline_bundle["mask_3d"],
            centerline_vox=centerline_bundle["path_smooth_vox"],
            plane_points_vox=centerline_bundle["plane_points_vox"],
            valid_plane_index=valid_flow_sections,
        )
        if fig_centerline.exists():
            centerline_overlay_name = _rel_fig_path(fig_centerline, fig_dir)
        fig_centerline_3d = fig_groups["centerline"] / "centerline_3d.png"
        _save_centerline_3d_figure(
            out_path=fig_centerline_3d,
            mask_3d=centerline_bundle["mask_3d"],
            spacing_mm=hr_spacing,
            centerline_vox=centerline_bundle["path_smooth_vox"],
            plane_points_vox=centerline_bundle["plane_points_vox"],
            valid_plane_index=valid_flow_sections,
        )
        if fig_centerline_3d.exists():
            centerline_3d_name = _rel_fig_path(fig_centerline_3d, fig_dir)

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
            baseline_label=baseline_label,
            sr_label=sr_label,
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

        fig_sections = fig_groups["centerline"] / "centerline_plane_sections.png"
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
            centerline_sections_name = _rel_fig_path(fig_sections, fig_dir)

        fig_qs = fig_groups["centerline"] / "centerline_flow_along_vessel_peak.png"
        _save_centerline_peak_qs_figure(
            out_path=fig_qs,
            q_ref_curves=q_ref_curves,
            q_base_curves=q_base_curves,
            q_sr_curves=q_sr_curves,
            valid_sections=valid_flow_sections,
            peak_idx=int(peak_idx),
            ref_label=ref_label,
            baseline_label=baseline_label,
            sr_label=sr_label,
        )
        if fig_qs.exists():
            centerline_peak_qs_name = _rel_fig_path(fig_qs, fig_dir)

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
                "qc_plane_points_inside_mask_pct": float(100.0 * _scipy_mean(np.isfinite(wall_mm_all).astype(np.float64))) if wall_mm_all.size > 0 else float("nan"),
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
        for method_name, q_val in ((ref_label, ref_t), (baseline_label, float(q_base_time[t])), (sr_label, float(q_sr_time[t]))):
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
    fig_flow = fig_groups["flow"] / "flow_rate_profile.png"
    x_time = frame_source_indices.astype(np.int32)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_time, q_ref_time, label=ref_label, linewidth=2.0, marker="o", markersize=3.6, color=REPORT_COLOR_REF)
    plt.plot(x_time, q_base_time, label=baseline_label, linewidth=2.0, marker="o", markersize=3.6, color=REPORT_COLOR_BASELINE)
    plt.plot(x_time, q_sr_time, label=sr_label, linewidth=2.0, marker="o", markersize=3.6, color=REPORT_COLOR_SR)
    plt.axhline(q_ref_scalar, linestyle="--", color=REPORT_COLOR_NEUTRAL, linewidth=1.2, label=f"Qref={q_ref_scalar:.3f} ml/s")
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
    fig.savefig(fig_flow, dpi=REPORT_FIG_DPI)
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

        fig_wss = fig_groups["wss"] / "wss_distribution.png"
        fig = plt.figure(figsize=(9, 5))
        bins = 80
        if tau_ref.size > 0:
            plt.hist(tau_ref, bins=bins, alpha=0.4, density=True, label=ref_label)
        if tau_base.size > 0:
            plt.hist(tau_base, bins=bins, alpha=0.4, density=True, label=baseline_label)
        if tau_sr.size > 0:
            plt.hist(tau_sr, bins=bins, alpha=0.4, density=True, label=sr_label)
        plt.xlabel("WSS [Pa]")
        plt.ylabel("Density")
        plt.title("Wall shear stress distribution (boundary samples)")
        plt.legend()
        plt.tight_layout()
        fig.savefig(fig_wss, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        fig_wss_name = _rel_fig_path(fig_wss, fig_dir)

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
    ba_speed_single_names: Dict[str, str] = {}
    corr_speed_name = ""
    corr_speed_single_names: Dict[str, str] = {}
    corr_flow_name = ""
    corr_flow_single_names: Dict[str, str] = {}

    # Intraluminal speed diagnostics
    m_in = mask_ref > 0.5
    sp_ref = speed_ref[m_in]
    sp_base = speed_base[m_in]
    sp_sr = speed_sr[m_in]

    if sp_ref.size > 20:
        # Bland-Altman: baseline vs reference and SR vs reference
        fig_ba = fig_groups["bland_altman"] / "bland_altman_speed_dual.png"
        ba_limits_speed = _bland_altman_joint_limits(sp_ref, [sp_base, sp_sr], pad_ratio=0.08)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ba_base = _plot_bland_altman_panel(
            ax=axes[0],
            ref_vals=sp_ref,
            test_vals=sp_base,
            ref_label=ref_label,
            test_label=baseline_label,
            seed=17,
        )
        ba_sr = _plot_bland_altman_panel(
            ax=axes[1],
            ref_vals=sp_ref,
            test_vals=sp_sr,
            ref_label=ref_label,
            test_label=sr_label,
            seed=29,
        )
        if ba_limits_speed is not None:
            (x_lo, x_hi), (y_lo, y_hi) = ba_limits_speed
            for ax in np.ravel(axes):
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
        else:
            _sync_axis_limits(list(np.ravel(axes)), axis="y", symmetric=True, pad_ratio=0.08)
        fig.suptitle("Bland-Altman: intraluminal speed", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_ba, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        ba_speed_name = _rel_fig_path(fig_ba, fig_dir)
        # Standalone versions (no subplot)
        for method_name, test_vals, seed_val in (
            (baseline_label, sp_base, 17),
            (sr_label, sp_sr, 29),
        ):
            fig_single = fig_groups["bland_altman"] / f"bland_altman_speed_intraluminal_{method_name.lower().replace(' ', '_')}.png"
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.8, 5.0))
            _plot_bland_altman_panel(
                ax=ax_s,
                ref_vals=sp_ref,
                test_vals=test_vals,
                ref_label=ref_label,
                test_label=method_name,
                seed=seed_val,
            )
            if ba_limits_speed is not None:
                (x_lo, x_hi), (y_lo, y_hi) = ba_limits_speed
                ax_s.set_xlim(x_lo, x_hi)
                ax_s.set_ylim(y_lo, y_hi)
            fig_s.tight_layout()
            fig_s.savefig(fig_single, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            ba_speed_single_names[method_name] = _rel_fig_path(fig_single, fig_dir)

        ba_rows.append({"domain": "speed_intraluminal", "method": baseline_label, **ba_base})
        ba_rows.append({"domain": "speed_intraluminal", "method": sr_label, **ba_sr})

        # Correlation: baseline/ref and SR/ref
        fig_corr_speed = fig_groups["correlation"] / "correlation_speed_intraluminal.png"
        corr_limits_speed = _correlation_joint_limits(sp_ref, [sp_base, sp_sr], pad_ratio=0.05)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        c_base = _plot_correlation_panel(
            ax=axes[0],
            x=sp_ref,
            y=sp_base,
            x_label=f"{ref_label} speed [m/s]",
            y_label=f"{baseline_label} speed [m/s]",
            title=f"{baseline_label} vs {ref_label}",
            color=REPORT_COLOR_BASELINE,
            seed=41,
        )
        c_sr = _plot_correlation_panel(
            ax=axes[1],
            x=sp_ref,
            y=sp_sr,
            x_label=f"{ref_label} speed [m/s]",
            y_label=f"{sr_label} speed [m/s]",
            title=f"{sr_label} vs {ref_label}",
            color=REPORT_COLOR_SR,
            seed=43,
        )
        if corr_limits_speed is not None:
            lo_c, hi_c = corr_limits_speed
            for ax in np.ravel(axes):
                ax.set_xlim(lo_c, hi_c)
                ax.set_ylim(lo_c, hi_c)
        fig.suptitle("Correlation: intraluminal speed", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_corr_speed, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        corr_speed_name = _rel_fig_path(fig_corr_speed, fig_dir)
        # Standalone versions (no subplot)
        for method_name, y_vals, color_val, seed_val in (
            (baseline_label, sp_base, REPORT_COLOR_BASELINE, 41),
            (sr_label, sp_sr, REPORT_COLOR_SR, 43),
        ):
            fig_single = fig_groups["correlation"] / f"correlation_speed_intraluminal_{method_name.lower().replace(' ', '_')}.png"
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.8, 5.0))
            _plot_correlation_panel(
                ax=ax_s,
                x=sp_ref,
                y=y_vals,
                x_label=f"{ref_label} speed [m/s]",
                y_label=f"{method_name} speed [m/s]",
                title=f"{method_name} vs {ref_label}",
                color=color_val,
                seed=seed_val,
            )
            if corr_limits_speed is not None:
                lo_c, hi_c = corr_limits_speed
                ax_s.set_xlim(lo_c, hi_c)
                ax_s.set_ylim(lo_c, hi_c)
            fig_s.tight_layout()
            fig_s.savefig(fig_single, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            corr_speed_single_names[method_name] = _rel_fig_path(fig_single, fig_dir)

        corr_rows.append({"domain": "speed_intraluminal", "method": baseline_label, **c_base})
        corr_rows.append({"domain": "speed_intraluminal", "method": sr_label, **c_sr})

    # Temporal-flow correlation diagnostics
    if q_ref_time.size > 2:
        fig_corr_flow = fig_groups["correlation"] / "correlation_flow_temporal.png"
        corr_limits_flow = _correlation_joint_limits(q_ref_time, [q_base_time, q_sr_time], pad_ratio=0.05)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        cf_base = _plot_correlation_panel(
            ax=axes[0],
            x=q_ref_time,
            y=q_base_time,
            x_label=f"{ref_label} flow [ml/s]",
            y_label=f"{baseline_label} flow [ml/s]",
            title=f"{baseline_label} vs {ref_label}",
            color=REPORT_COLOR_BASELINE,
            seed=53,
        )
        cf_sr = _plot_correlation_panel(
            ax=axes[1],
            x=q_ref_time,
            y=q_sr_time,
            x_label=f"{ref_label} flow [ml/s]",
            y_label=f"{sr_label} flow [ml/s]",
            title=f"{sr_label} vs {ref_label}",
            color=REPORT_COLOR_SR,
            seed=59,
        )
        if corr_limits_flow is not None:
            lo_c, hi_c = corr_limits_flow
            for ax in np.ravel(axes):
                ax.set_xlim(lo_c, hi_c)
                ax.set_ylim(lo_c, hi_c)
        fig.suptitle("Correlation: temporal flow profile", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_corr_flow, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        corr_flow_name = _rel_fig_path(fig_corr_flow, fig_dir)
        # Standalone versions (no subplot)
        for method_name, y_vals, color_val, seed_val in (
            (baseline_label, q_base_time, REPORT_COLOR_BASELINE, 53),
            (sr_label, q_sr_time, REPORT_COLOR_SR, 59),
        ):
            fig_single = fig_groups["correlation"] / f"correlation_flow_temporal_{method_name.lower().replace(' ', '_')}.png"
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.8, 5.0))
            _plot_correlation_panel(
                ax=ax_s,
                x=q_ref_time,
                y=y_vals,
                x_label=f"{ref_label} flow [ml/s]",
                y_label=f"{method_name} flow [ml/s]",
                title=f"{method_name} vs {ref_label}",
                color=color_val,
                seed=seed_val,
            )
            if corr_limits_flow is not None:
                lo_c, hi_c = corr_limits_flow
                ax_s.set_xlim(lo_c, hi_c)
                ax_s.set_ylim(lo_c, hi_c)
            fig_s.tight_layout()
            fig_s.savefig(fig_single, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            corr_flow_single_names[method_name] = _rel_fig_path(fig_single, fig_dir)

        corr_rows.append({"domain": "flow_temporal", "method": baseline_label, **cf_base})
        corr_rows.append({"domain": "flow_temporal", "method": sr_label, **cf_sr})

    # Paper-like cerebrovascular metrics: peak-frame summaries + all-frame component BA diagnostics.
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
    comp_ba_names: Dict[str, str] = {}
    comp_corr_single_names: Dict[str, str] = {}
    comp_ba_single_names: Dict[str, str] = {}

    fig_comp_corr = fig_groups["correlation"] / "correlation_velocity_components_peak.png"
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for c_idx, comp_name in enumerate(("u", "v", "w", "mag")):
        rr = peak_ref_comp[comp_name][peak_mask]
        bb = peak_base_comp[comp_name][peak_mask]
        ss = peak_sr_comp[comp_name][peak_mask]
        corr_limits_comp = _correlation_joint_limits(rr, [bb, ss], pad_ratio=0.05)
        st_base = _plot_correlation_panel(
            ax=axes[0, c_idx],
            x=rr,
            y=bb,
            x_label=f"{ref_label} {comp_name}",
            y_label=f"{baseline_label} {comp_name}",
            title=f"{baseline_label} vs {ref_label} ({comp_name})",
            color=REPORT_COLOR_BASELINE,
            seed=101 + c_idx,
        )
        st_sr = _plot_correlation_panel(
            ax=axes[1, c_idx],
            x=rr,
            y=ss,
            x_label=f"{ref_label} {comp_name}",
            y_label=f"{sr_label} {comp_name}",
            title=f"{sr_label} vs {ref_label} ({comp_name})",
            color=REPORT_COLOR_SR,
            seed=111 + c_idx,
        )
        if corr_limits_comp is not None:
            lo_c, hi_c = corr_limits_comp
            axes[0, c_idx].set_xlim(lo_c, hi_c)
            axes[0, c_idx].set_ylim(lo_c, hi_c)
            axes[1, c_idx].set_xlim(lo_c, hi_c)
            axes[1, c_idx].set_ylim(lo_c, hi_c)
        # Standalone versions (no subplot)
        for method_name, yy, color_val, seed_val in (
            (baseline_label, bb, REPORT_COLOR_BASELINE, 101 + c_idx),
            (sr_label, ss, REPORT_COLOR_SR, 111 + c_idx),
        ):
            fig_single = fig_groups["correlation"] / (
                f"correlation_velocity_component_peak_{comp_name}_{method_name.lower().replace(' ', '_')}.png"
            )
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.6, 5.0))
            _plot_correlation_panel(
                ax=ax_s,
                x=rr,
                y=yy,
                x_label=f"{ref_label} {comp_name}",
                y_label=f"{method_name} {comp_name}",
                title=f"{method_name} vs {ref_label} ({comp_name})",
                color=color_val,
                seed=seed_val,
            )
            if corr_limits_comp is not None:
                lo_c, hi_c = corr_limits_comp
                ax_s.set_xlim(lo_c, hi_c)
                ax_s.set_ylim(lo_c, hi_c)
            fig_s.tight_layout()
            fig_s.savefig(fig_single, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            comp_corr_single_names[f"{comp_name}_{method_name}"] = _rel_fig_path(fig_single, fig_dir)
        comp_corr_rows.append(
            {
                "domain": "velocity_component_peak",
                "region": "intraluminal",
                "component": comp_name,
                "method": baseline_label,
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
                "method": sr_label,
                "frame_payload_index": int(peak_idx),
                "frame_source_index": int(peak_frame_src),
                **st_sr,
            }
        )
    fig.suptitle("Peak-flow component correlation (intraluminal)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_comp_corr, dpi=REPORT_FIG_DPI)
    plt.close(fig)
    comp_corr_name = _rel_fig_path(fig_comp_corr, fig_dir)

    comp_ref_all = {
        "u": gt_phys[:, 0],
        "v": gt_phys[:, 1],
        "w": gt_phys[:, 2],
        "mag": speed_ref,
    }
    comp_base_all = {
        "u": lr_vel_phys_metrics[:, 0],
        "v": lr_vel_phys_metrics[:, 1],
        "w": lr_vel_phys_metrics[:, 2],
        "mag": speed_base,
    }
    comp_sr_all = {
        "u": pred_phys[:, 0],
        "v": pred_phys[:, 1],
        "w": pred_phys[:, 2],
        "mag": speed_sr,
    }
    comp_title = {"u": "U component", "v": "V component", "w": "W component", "mag": "Speed magnitude"}

    for c_idx, comp_name in enumerate(("u", "v", "w", "mag")):
        rr = comp_ref_all[comp_name][m_in]
        bb = comp_base_all[comp_name][m_in]
        ss = comp_sr_all[comp_name][m_in]
        if rr.size < 20:
            continue

        fig_comp_ba = fig_groups["bland_altman"] / f"bland_altman_velocity_component_{comp_name}_allframes.png"
        ba_limits_comp = _bland_altman_joint_limits(rr, [bb, ss], pad_ratio=0.08)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ba_base = _plot_bland_altman_panel(
            ax=axes[0],
            ref_vals=rr,
            test_vals=bb,
            ref_label=ref_label,
            test_label=baseline_label,
            seed=201 + c_idx,
        )
        ba_sr = _plot_bland_altman_panel(
            ax=axes[1],
            ref_vals=rr,
            test_vals=ss,
            ref_label=ref_label,
            test_label=sr_label,
            seed=211 + c_idx,
        )
        if ba_limits_comp is not None:
            (x_lo, x_hi), (y_lo, y_hi) = ba_limits_comp
            for ax in np.ravel(axes):
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
        else:
            _sync_axis_limits(list(np.ravel(axes)), axis="y", symmetric=True, pad_ratio=0.08)
        fig.suptitle(
            f"Bland-Altman: {comp_title[comp_name]} (all frames, in-mask voxels)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(fig_comp_ba, dpi=REPORT_FIG_DPI)
        plt.close(fig)
        comp_ba_names[comp_name] = _rel_fig_path(fig_comp_ba, fig_dir)
        # Standalone versions (no subplot)
        for method_name, test_vals, seed_val in (
            (baseline_label, bb, 201 + c_idx),
            (sr_label, ss, 211 + c_idx),
        ):
            fig_single = fig_groups["bland_altman"] / (
                f"bland_altman_velocity_component_{comp_name}_{method_name.lower().replace(' ', '_')}.png"
            )
            fig_s, ax_s = plt.subplots(1, 1, figsize=(6.6, 5.0))
            _plot_bland_altman_panel(
                ax=ax_s,
                ref_vals=rr,
                test_vals=test_vals,
                ref_label=ref_label,
                test_label=method_name,
                seed=seed_val,
            )
            if ba_limits_comp is not None:
                (x_lo, x_hi), (y_lo, y_hi) = ba_limits_comp
                ax_s.set_xlim(x_lo, x_hi)
                ax_s.set_ylim(y_lo, y_hi)
            fig_s.tight_layout()
            fig_s.savefig(fig_single, dpi=REPORT_FIG_DPI)
            plt.close(fig_s)
            comp_ba_single_names[f"{comp_name}_{method_name}"] = _rel_fig_path(fig_single, fig_dir)

        comp_ba_rows.append(
            {
                "domain": "velocity_component_all_frames",
                "region": "intraluminal",
                "component": comp_name,
                "method": baseline_label,
                "frame_payload_index": -1,
                "frame_source_index": -1,
                **ba_base,
            }
        )
        comp_ba_rows.append(
            {
                "domain": "velocity_component_all_frames",
                "region": "intraluminal",
                "component": comp_name,
                "method": sr_label,
                "frame_payload_index": -1,
                "frame_source_index": -1,
                **ba_sr,
            }
        )

    # Peak speed metrics in core, wall, and full intraluminal regions.
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
        rel = float(_scipy_mean(np.abs(t - r) / (np.abs(r) + 1e-12)) * 100.0)
        num = float(np.dot(r.astype(np.float64), t.astype(np.float64)))
        den = float(np.linalg.norm(r.astype(np.float64)) * np.linalg.norm(t.astype(np.float64)) + 1e-12)
        return {
            "domain": "peak_velocity_magnitude",
            "region": region_name,
            "method": method_name,
            "frame_payload_index": int(peak_idx),
            "frame_source_index": int(peak_frame_src),
            "n": int(r.size),
            "mae": _scipy_mean(np.abs(t - r)),
            "rmse": _scipy_rmse(t - r),
            "relative_error_pct": rel,
            "cosine_similarity": float(num / den),
        }

    peak_speed_rows = [
        _region_peak_speed_metrics(core_mask, "core", baseline_label, speed_base),
        _region_peak_speed_metrics(core_mask, "core", sr_label, speed_sr),
        _region_peak_speed_metrics(wall_mask, "wall", baseline_label, speed_base),
        _region_peak_speed_metrics(wall_mask, "wall", sr_label, speed_sr),
        _region_peak_speed_metrics(peak_mask, "intraluminal", baseline_label, speed_base),
        _region_peak_speed_metrics(peak_mask, "intraluminal", sr_label, speed_sr),
    ]

    def _region_mean_speed_metrics(region_name: str, method_name: str, test_speed: np.ndarray) -> Dict[str, Any]:
        ref_parts: List[np.ndarray] = []
        test_parts: List[np.ndarray] = []
        for t in range(t_count):
            mask_t = mask_ref[t] > 0.5
            if int(mask_t.sum()) == 0:
                continue
            core_t = binary_erosion(mask_t, iterations=1)
            if int(core_t.sum()) == 0:
                core_t = mask_t.copy()
            wall_t = mask_t & (~core_t)
            if int(wall_t.sum()) == 0:
                wall_t = mask_t.copy()
            if region_name == "core":
                region_mask = core_t
            elif region_name == "wall":
                region_mask = wall_t
            else:
                region_mask = mask_t
            r_t = speed_ref[t][region_mask]
            s_t = test_speed[t][region_mask]
            if r_t.size == 0:
                continue
            ref_parts.append(np.asarray(r_t, dtype=np.float64))
            test_parts.append(np.asarray(s_t, dtype=np.float64))

        if not ref_parts:
            return {
                "domain": "mean_velocity_magnitude",
                "region": region_name,
                "method": method_name,
                "frame_payload_index": -1,
                "frame_source_index": -1,
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "relative_error_pct": float("nan"),
                "cosine_similarity": float("nan"),
            }

        r = np.concatenate(ref_parts, axis=0)
        t = np.concatenate(test_parts, axis=0)
        rel = float(_scipy_mean(np.abs(t - r) / (np.abs(r) + 1e-12)) * 100.0)
        num = float(np.dot(r, t))
        den = float(np.linalg.norm(r) * np.linalg.norm(t) + 1e-12)
        return {
            "domain": "mean_velocity_magnitude",
            "region": region_name,
            "method": method_name,
            "frame_payload_index": -1,
            "frame_source_index": -1,
            "n": int(r.size),
            "mae": _scipy_mean(np.abs(t - r)),
            "rmse": _scipy_rmse(t - r),
            "relative_error_pct": rel,
            "cosine_similarity": float(num / den),
        }

    mean_speed_rows = [
        _region_mean_speed_metrics("core", baseline_label, speed_base),
        _region_mean_speed_metrics("core", sr_label, speed_sr),
        _region_mean_speed_metrics("wall", baseline_label, speed_base),
        _region_mean_speed_metrics("wall", sr_label, speed_sr),
        _region_mean_speed_metrics("intraluminal", baseline_label, speed_base),
        _region_mean_speed_metrics("intraluminal", sr_label, speed_sr),
    ]

    # Statistical significance (paired Wilcoxon) for voxel-wise absolute errors: baseline vs SR.
    pvalue_rows: List[Dict[str, Any]] = []

    # 1) Peak velocity significance at peak frame (core/wall/intraluminal).
    for region_name, region_mask in (("core", core_mask), ("wall", wall_mask), ("intraluminal", peak_mask)):
        r_peak = speed_ref[peak_idx][region_mask]
        err_base_peak = np.abs(speed_base[peak_idx][region_mask] - r_peak)
        err_sr_peak = np.abs(speed_sr[peak_idx][region_mask] - r_peak)
        p_val = _wilcoxon_p(err_base_peak.tolist(), err_sr_peak.tolist())
        pvalue_rows.append(
            {
                "analysis": "Peak Velocity (Systolic)",
                "region": region_name,
                "wilcoxon_p_value": p_val,
                "n_voxels": int(r_peak.size),
            }
        )

    # 2) Mean velocity significance using all frames with per-frame dynamic masks.
    for region_name in ("core", "wall", "intraluminal"):
        err_base_mean_parts: List[np.ndarray] = []
        err_sr_mean_parts: List[np.ndarray] = []
        for t in range(t_count):
            mask_t = mask_ref[t] > 0.5
            if int(mask_t.sum()) == 0:
                continue
            core_t = binary_erosion(mask_t, iterations=1)
            if int(core_t.sum()) == 0:
                core_t = mask_t.copy()
            wall_t = mask_t & (~core_t)
            if int(wall_t.sum()) == 0:
                wall_t = mask_t.copy()

            if region_name == "core":
                region_mask_t = core_t
            elif region_name == "wall":
                region_mask_t = wall_t
            else:
                region_mask_t = mask_t
            r_t = speed_ref[t][region_mask_t]
            if r_t.size == 0:
                continue
            err_base_mean_parts.append(np.abs(speed_base[t][region_mask_t] - r_t))
            err_sr_mean_parts.append(np.abs(speed_sr[t][region_mask_t] - r_t))

        if err_base_mean_parts:
            err_base_mean = np.concatenate(err_base_mean_parts, axis=0)
            err_sr_mean = np.concatenate(err_sr_mean_parts, axis=0)
            p_val_mean = _wilcoxon_p(err_base_mean.tolist(), err_sr_mean.tolist())
            pvalue_rows.append(
                {
                    "analysis": "Mean Velocity (All frames)",
                    "region": region_name,
                    "wilcoxon_p_value": p_val_mean,
                    "n_voxels": int(err_base_mean.size),
                }
            )

    # Flow peak-like metrics (temporal)
    flow_peak_rows = []
    for method_name, q_time in ((baseline_label, q_base_time), (sr_label, q_sr_time)):
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
                "rmse_over_time_ml_s": _scipy_rmse(q_time - q_ref_time),
                "relative_err_over_time_pct": float(_scipy_mean(np.abs(q_time - q_ref_time) / (np.abs(q_ref_time) + 1e-12)) * 100.0),
            }
        )

    flow_average_rows = []
    for method_name, q_time in ((baseline_label, q_base_time), (sr_label, q_sr_time)):
        abs_err = np.abs(q_time - q_ref_time)
        rel_err = np.abs(q_time - q_ref_time) / (np.abs(q_ref_time) + 1e-12)
        flow_average_rows.append(
            {
                "domain": "flow_temporal_mean",
                "method": method_name,
                "mean_ref_flow_ml_s": _scipy_mean(q_ref_time),
                "mean_method_flow_ml_s": _scipy_mean(q_time),
                "mean_abs_err_ml_s": _scipy_mean(abs_err),
                "mean_relative_err_pct": float(_scipy_mean(rel_err) * 100.0),
                "rmse_over_time_ml_s": _scipy_rmse(q_time - q_ref_time),
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
        metrics_dir / "mean_velocity_metrics.csv",
        mean_speed_rows,
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
    _write_csv(
        metrics_dir / "flow_average_metrics.csv",
        flow_average_rows,
        [
            "domain",
            "method",
            "mean_ref_flow_ml_s",
            "mean_method_flow_ml_s",
            "mean_abs_err_ml_s",
            "mean_relative_err_pct",
            "rmse_over_time_ml_s",
        ],
    )
    _write_csv(
        metrics_dir / "significance_pvalues.csv",
        pvalue_rows,
        ["analysis", "region", "wilcoxon_p_value", "n_voxels"],
    )
    publication_figs_raw = _save_publication_figures(
        fig_dir=fig_groups["publication"],
        baseline_label=baseline_label,
        sr_label=sr_label,
        ref_label=ref_label,
        peak_speed_rows=peak_speed_rows,
        mean_speed_rows=mean_speed_rows,
        table2_all_rows=table2_all,
        table2_temporal_mean_rows=table2_temporal_mean,
        corr_rows=corr_rows,
        voxel_dist_rows=voxel_dist_rows,
        pvalue_rows=pvalue_rows,
        q_ref_time=q_ref_time,
        q_base_time=q_base_time,
        q_sr_time=q_sr_time,
        frame_source_indices=frame_source_indices,
    )
    publication_figs = {
        k: _rel_fig_path(fig_groups["publication"] / str(v), fig_dir)
        for k, v in publication_figs_raw.items()
    }

    ba_cols = ["domain", "region", "component", "method", "frame_payload_index", "frame_source_index", "n", "bias", "sd_diff", "loa_low", "loa_high"]
    if ba_rows:
        _write_csv(metrics_dir / "bland_altman_stats.csv", ba_rows, ba_cols)

    _write_csv(
        metrics_dir / "run_context.csv",
        [
            {
                "task_mode": task_mode_tag,
                "task_mode_label": task_mode_label,
                "res_increase_detected": None if detected_res_increase is None else int(detected_res_increase),
                "reference_label": str(ref_label),
                "baseline_label": str(baseline_label),
                "sr_label": str(sr_label),
                "sr_role_label": str(sr_role_label),
                "baseline_source": str(baseline_source_tag),
                "baseline_payload_path": str(baseline_payload_path),
                "baseline_lr_shape_xyz": json.dumps([int(x) for x in lr_norm.shape[2:]]),
                "inference_lr_shape_xyz": json.dumps([int(x) for x in lr_norm_inference.shape[2:]]),
                "spatial_shape_original_xyz": json.dumps(spatial_shapes_original),
                "spatial_shape_used_xyz": json.dumps(spatial_shapes_cropped),
                "spatial_crop_applied": bool(spatial_crop_applied),
            }
        ],
        [
            "task_mode",
            "task_mode_label",
            "res_increase_detected",
            "reference_label",
            "baseline_label",
            "sr_label",
            "sr_role_label",
            "baseline_source",
            "baseline_payload_path",
            "baseline_lr_shape_xyz",
            "inference_lr_shape_xyz",
            "spatial_shape_original_xyz",
            "spatial_shape_used_xyz",
            "spatial_crop_applied",
        ],
    )

    summary = {
        "report_title": args.report_title,
        "task_mode": task_mode_tag,
        "task_mode_label": task_mode_label,
        "res_increase_detected": None if detected_res_increase is None else int(detected_res_increase),
        "labels": {
            "reference": ref_label,
            "baseline": baseline_label,
            "super_resolved": sr_label,
            "enhanced_role": sr_role_label,
        },
        "payload_path": str(Path(args.payload_npz).resolve()),
        "baseline_payload_path": str(baseline_payload_path),
        "baseline_source": str(baseline_source_tag),
        "metadata": metadata,
        # Keep ROI at top-level for easier downstream tooling compatibility.
        "roi": roi_info,
        "dimensions": {
            "T": int(t_count),
            "frame_source_indices": [int(x) for x in frame_source_indices.tolist()],
            "lr_shape_xyz": [int(x) for x in lr_norm.shape[2:]],
            "inference_lr_shape_xyz": [int(x) for x in lr_norm_inference.shape[2:]],
            "shape_XYZ": [int(x) for x in pred_norm.shape[2:]],
            "spatial_shape_original": spatial_shapes_original,
            "spatial_shape_used": spatial_shapes_cropped,
            "spatial_crop_applied": bool(spatial_crop_applied),
            "flow_method": str(flow_method),
            "flow_section_kind": str(flow_section_kind),
            "flow_axis": int(selected_flow_axis),
            "flow_axis_mode": str(args.flow_axis),
            "suggested_flow_axis": int(suggested_flow_axis),
        },
        "statistics": {
            "table2_wilcoxon_p_re_baseline_vs_sr": p_re,
            "table2_temporal_mean_rows": table2_temporal_mean,
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
            "task_mode": task_mode_tag,
            "res_increase_detected": None if detected_res_increase is None else int(detected_res_increase),
            "correlation_rows": corr_rows,
            "bland_altman_rows": ba_rows,
            "peak_velocity_rows": peak_speed_rows,
            "mean_velocity_rows": mean_speed_rows,
            "significance_pvalue_rows": pvalue_rows,
            "flow_peak_rows": flow_peak_rows,
            "flow_average_rows": flow_average_rows,
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
            "voxel_histogram_overview_figure": str(_rel_fig_path(fig_voxel_hist, fig_dir)),
            "voxel_histogram_channel_figures": {k: str(v) for k, v in voxel_hist_channel_figs.items()},
            "component_bland_altman_figures": {k: str(v) for k, v in comp_ba_names.items()},
            "publication_style_figures": {k: str(v) for k, v in publication_figs.items()},
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

    t2_comp_cols = [
        "location",
        "slice_index",
        "variable",
        "ref",
        "baseline",
        "sr",
        "re_baseline",
        "re_sr",
        "re_baseline_eps",
        "re_sr_eps",
        "smape_baseline",
        "smape_sr",
    ]
    t2_comp_html = _html_table(fmt_rows(table2_compact, t2_comp_cols, nd=6), t2_comp_cols)
    t2_tm_cols = [
        "slice_index",
        "variable",
        "n_frames",
        "ref_mean_over_frames",
        "baseline_mean_over_frames",
        "sr_mean_over_frames",
        "re_baseline_mean_over_frames",
        "re_sr_mean_over_frames",
        "re_baseline_eps_mean_over_frames",
        "re_sr_eps_mean_over_frames",
        "smape_baseline_mean_over_frames",
        "smape_sr_mean_over_frames",
    ]
    t2_tm_html = _html_table(fmt_rows(table2_temporal_mean, t2_tm_cols, nd=6), t2_tm_cols) if table2_temporal_mean else "<p class=\"muted\">No temporal-mean Table-2 rows.</p>"

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
    mean_vel_cols = ["domain", "region", "method", "frame_payload_index", "frame_source_index", "n", "mae", "rmse", "relative_error_pct", "cosine_similarity"]
    mean_vel_html = _html_table(fmt_rows(mean_speed_rows, mean_vel_cols, nd=6), mean_vel_cols) if mean_speed_rows else "<p class=\"muted\">No mean velocity rows.</p>"
    pvalue_cols = ["analysis", "region", "wilcoxon_p_value", "n_voxels"]
    pvalue_html = _html_table(fmt_rows(pvalue_rows, pvalue_cols, nd=6), pvalue_cols) if pvalue_rows else "<p class=\"muted\">No statistical significance rows.</p>"
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
    flow_avg_cols = [
        "domain",
        "method",
        "mean_ref_flow_ml_s",
        "mean_method_flow_ml_s",
        "mean_abs_err_ml_s",
        "mean_relative_err_pct",
        "rmse_over_time_ml_s",
    ]
    flow_avg_html = _html_table(fmt_rows(flow_average_rows, flow_avg_cols, nd=6), flow_avg_cols) if flow_average_rows else "<p class=\"muted\">No mean flow rows.</p>"

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

    def _img_with_caption(rel_path: str, alt: str, desc: str) -> str:
        return f"<img src='figures/{rel_path}' alt='{alt}'/><p class=\"muted\">{desc}</p>"

    def _titled_img(title: str, rel_path: str, alt: str, desc: str) -> str:
        return f"<h4>{title}</h4>{_img_with_caption(rel_path, alt, desc)}"

    ch_img_tags = "\n".join(
        _titled_img(
            title=f"Channel {k}",
            rel_path=v,
            alt=f"{k} comparison",
            desc=f"Compara input LR interpolado, predicción {sr_role_label}, referencia y error absoluto en cortes representativos.",
        )
        for k, v in channel_figs.items()
    )
    voxel_hist_channel_tags = "\n".join(
        _titled_img(
            title=f"Intensity Histogram {k.upper()}",
            rel_path=v,
            alt=f"Intensity histogram {k.upper()}",
            desc=f"Distribución de intensidades dentro de la máscara para referencia, baseline y {sr_role_label} en este canal.",
        )
        for k, v in voxel_hist_channel_figs.items()
    )

    wss_summary_text = (
        f"{p_wss:.4g}" if (bool(args.include_wss) and np.isfinite(p_wss)) else ("disabled" if not bool(args.include_wss) else "nan")
    )
    wss_img_tag = _img_with_caption(
        fig_wss_name,
        "WSS distribution",
        f"Distribución de WSS en pared; permite comparar sesgo y dispersión entre baseline y {sr_role_label} frente a la referencia.",
    ) if fig_wss_name else ""
    corr_speed_tag = _img_with_caption(
        corr_speed_name,
        "Correlation speed",
        "Correlación voxel a voxel de la magnitud de velocidad intraluminal contra la referencia.",
    ) if corr_speed_name else ""
    corr_flow_tag = _img_with_caption(
        corr_flow_name,
        "Correlation temporal flow",
        "Correlación temporal del flujo por frame respecto a la referencia.",
    ) if corr_flow_name else ""
    comp_corr_tag = _img_with_caption(
        comp_corr_name,
        "Correlation velocity components peak",
        "Correlación por componente de velocidad (u,v,w) en el frame pico frente a la referencia.",
    ) if comp_corr_name else ""
    corr_speed_tags = (
        "\n".join(
            [
                _titled_img(
                    title=f"Correlation speed ({m})",
                    rel_path=p,
                    alt=f"Correlation speed {m}",
                    desc="Nube de dispersión y ajuste lineal de velocidad intraluminal vs referencia.",
                )
                for m, p in corr_speed_single_names.items()
            ]
        )
        if corr_speed_single_names
        else corr_speed_tag
    )
    corr_flow_tags = (
        "\n".join(
            [
                _titled_img(
                    title=f"Correlation flow temporal ({m})",
                    rel_path=p,
                    alt=f"Correlation flow temporal {m}",
                    desc="Comparación temporal frame a frame del flujo integrado frente a la referencia.",
                )
                for m, p in corr_flow_single_names.items()
            ]
        )
        if corr_flow_single_names
        else corr_flow_tag
    )
    ba_speed_tags = (
        "\n".join(
            [
                _titled_img(
                    title=f"Bland-Altman intraluminal speed ({m})",
                    rel_path=p,
                    alt=f"Bland-Altman intraluminal speed {m}",
                    desc="Diferencia vs media para velocidad intraluminal; muestra sesgo y límites de acuerdo.",
                )
                for m, p in ba_speed_single_names.items()
            ]
        )
        if ba_speed_single_names
        else (
            _img_with_caption(
                ba_speed_name,
                "Bland-Altman speed",
                "Diferencia vs media para velocidad intraluminal; permite ver sesgo y variabilidad.",
            )
            if ba_speed_name
            else ""
        )
    )
    comp_corr_tags = (
        "\n".join(
            [
                _titled_img(
                    title=f"Peak component correlation {k.replace('_', ' | ')}",
                    rel_path=v,
                    alt=f"Peak component correlation {k}",
                    desc="Correlación por componente en frame pico para evaluar fidelidad direccional.",
                )
                for k, v in comp_corr_single_names.items()
            ]
        )
        if comp_corr_single_names
        else comp_corr_tag
    )
    comp_ba_tags = "\n".join(
        [
            (
                _titled_img(
                    title=f"Bland-Altman {k.upper()} (all frames, in-mask voxels)",
                    rel_path=v,
                    alt=f"Bland-Altman {k.upper()} all frames in-mask voxels",
                    desc="Bland-Altman por componente usando todos los voxeles intraluminales en todos los frames.",
                )
            )
            for k, v in comp_ba_names.items()
        ]
    )
    if comp_ba_single_names:
        comp_ba_tags = "\n".join(
            [
                _titled_img(
                    title=f"Bland-Altman {k.replace('_', ' | ')}",
                    rel_path=v,
                    alt=f"Bland-Altman {k}",
                    desc="Diferencia vs media para componente/método específico.",
                )
                for k, v in comp_ba_single_names.items()
            ]
        )
    saved_comp_ba_items = "".join([f"<li><code>figures/{v}</code></li>" for _, v in comp_ba_names.items()])
    saved_comp_ba_single_items = "".join([f"<li><code>figures/{v}</code></li>" for _, v in comp_ba_single_names.items()])
    saved_corr_single_items = "".join([f"<li><code>figures/{v}</code></li>" for _, v in corr_speed_single_names.items()])
    saved_corr_single_items += "".join([f"<li><code>figures/{v}</code></li>" for _, v in corr_flow_single_names.items()])
    saved_corr_single_items += "".join([f"<li><code>figures/{v}</code></li>" for _, v in comp_corr_single_names.items()])
    saved_ba_speed_single_items = "".join([f"<li><code>figures/{v}</code></li>" for _, v in ba_speed_single_names.items()])
    saved_voxel_hist_items = "".join([f"<li><code>figures/{v}</code></li>" for _, v in voxel_hist_channel_figs.items()])
    publication_fig_order = [
        ("velocity_error_mae_systolic_peak_peak_frame", "Velocity Error MAE (Peak Frame)"),
        ("velocity_error_rmse_systolic_peak_peak_frame", "Velocity Error RMSE (Peak Frame)"),
        ("velocity_error_mae_temporal_mean_all_frames", "Velocity Error MAE (Temporal Mean)"),
        ("velocity_error_rmse_temporal_mean_all_frames", "Velocity Error RMSE (Temporal Mean)"),
        ("velocity_relative_error_pct_peak", "Velocity Relative Error (%) Peak"),
        ("velocity_relative_error_pct_temporal_mean", "Velocity Relative Error (%) Temporal Mean"),
        ("slice_relative_errors", "Slice-wise Absolute Errors"),
        ("table2_abs_error_bar_velocity", "Absolute Error Bar (Velocity)"),
        ("table2_abs_error_bar_vorticity", "Absolute Error Bar (Vorticity)"),
        ("table2_relative_error_pct_bar_velocity", "Relative Error Bar [%] (Velocity)"),
        ("table2_relative_error_pct_bar_vorticity", "Relative Error Bar [%] (Vorticity)"),
        ("table2_abs_error_violin_velocity_scale", "Absolute Error Boxplot+Points (Velocity: Mean & SD)"),
        ("table2_abs_error_violin_velocity_shape", "Absolute Error Boxplot+Points (Velocity: Skewness & Kurtosis)"),
        ("table2_abs_error_violin_vorticity_scale", "Absolute Error Boxplot+Points (Vorticity: Mean & SD)"),
        ("table2_abs_error_violin_vorticity_shape", "Absolute Error Boxplot+Points (Vorticity: Skewness & Kurtosis)"),
        ("table2_relative_error_pct_violin_velocity_scale", "Relative Error Boxplot+Points [%] (Velocity: Mean & SD)"),
        ("table2_relative_error_pct_violin_velocity_shape", "Relative Error Boxplot+Points [%] (Velocity: Skewness & Kurtosis)"),
        ("table2_relative_error_pct_violin_vorticity_scale", "Relative Error Boxplot+Points [%] (Vorticity: Mean & SD)"),
        ("table2_relative_error_pct_violin_vorticity_shape", "Relative Error Boxplot+Points [%] (Vorticity: Skewness & Kurtosis)"),
        ("table2_temporal_mean_relative_error_bar", "Temporal-Mean Absolute Error (Bar)"),
        ("table2_temporal_mean_abs_error_bar_velocity", "Temporal-Mean Absolute Error Bar (Velocity)"),
        ("table2_temporal_mean_abs_error_bar_vorticity", "Temporal-Mean Absolute Error Bar (Vorticity)"),
        ("table2_temporal_mean_relative_error_pct_bar_velocity", "Temporal-Mean Relative Error Bar [%] (Velocity)"),
        ("table2_temporal_mean_relative_error_pct_bar_vorticity", "Temporal-Mean Relative Error Bar [%] (Vorticity)"),
        ("table2_temporal_mean_abs_error_violin_velocity_scale", "Temporal-Mean Absolute Error Boxplot+Points (Velocity: Mean & SD)"),
        ("table2_temporal_mean_abs_error_violin_velocity_shape", "Temporal-Mean Absolute Error Boxplot+Points (Velocity: Skewness & Kurtosis)"),
        ("table2_temporal_mean_abs_error_violin_vorticity_scale", "Temporal-Mean Absolute Error Boxplot+Points (Vorticity: Mean & SD)"),
        ("table2_temporal_mean_abs_error_violin_vorticity_shape", "Temporal-Mean Absolute Error Boxplot+Points (Vorticity: Skewness & Kurtosis)"),
        ("table2_temporal_mean_relative_error_pct_violin_velocity_scale", "Temporal-Mean Relative Error Boxplot+Points [%] (Velocity: Mean & SD)"),
        ("table2_temporal_mean_relative_error_pct_violin_velocity_shape", "Temporal-Mean Relative Error Boxplot+Points [%] (Velocity: Skewness & Kurtosis)"),
        ("table2_temporal_mean_relative_error_pct_violin_vorticity_scale", "Temporal-Mean Relative Error Boxplot+Points [%] (Vorticity: Mean & SD)"),
        ("table2_temporal_mean_relative_error_pct_violin_vorticity_shape", "Temporal-Mean Relative Error Boxplot+Points [%] (Vorticity: Skewness & Kurtosis)"),
        ("flow_abs_error_over_time", "Temporal Flow Absolute Error"),
        ("correlation_pearson_r", "Correlation (Pearson r)"),
        ("correlation_rmse_only", "Correlation (RMSE)"),
        ("voxel_distribution_std", "Voxel Distribution Std Dev"),
        ("significance_pvalues", "Wilcoxon Significance Summary"),
    ]
    publication_desc = {
        "velocity_error_mae_systolic_peak_peak_frame": "Error absoluto medio de velocidad en el frame pico.",
        "velocity_error_rmse_systolic_peak_peak_frame": "RMSE de velocidad en el frame pico.",
        "velocity_error_mae_temporal_mean_all_frames": "Error absoluto medio de velocidad agregando todos los frames.",
        "velocity_error_rmse_temporal_mean_all_frames": "RMSE de velocidad agregando todos los frames.",
        "velocity_relative_error_pct_peak": "Error relativo porcentual por región en frame pico.",
        "velocity_relative_error_pct_temporal_mean": "Error relativo porcentual por región en promedio temporal.",
        "slice_relative_errors": "Distribución de error absoluto entre slices para variables seleccionadas.",
        "table2_abs_error_bar_velocity": "Barras del error absoluto medio por variable de velocidad.",
        "table2_abs_error_bar_vorticity": "Barras del error absoluto medio por variable de vorticidad.",
        "table2_relative_error_pct_bar_velocity": "Barras del error relativo medio (%) por variable de velocidad.",
        "table2_relative_error_pct_bar_vorticity": "Barras del error relativo medio (%) por variable de vorticidad.",
        "table2_abs_error_violin_velocity_scale": "Boxplot de error absoluto con puntos semitransparentes para velocidad (Mean/SD), con cola superior robusta al P95.",
        "table2_abs_error_violin_velocity_shape": "Boxplot de error absoluto con puntos semitransparentes para velocidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_abs_error_violin_vorticity_scale": "Boxplot de error absoluto con puntos semitransparentes para vorticidad (Mean/SD), con cola superior robusta al P95.",
        "table2_abs_error_violin_vorticity_shape": "Boxplot de error absoluto con puntos semitransparentes para vorticidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_relative_error_pct_violin_velocity_scale": "Boxplot de error relativo (%) con puntos semitransparentes para velocidad (Mean/SD), con cola superior robusta al P95.",
        "table2_relative_error_pct_violin_velocity_shape": "Boxplot de error relativo (%) con puntos semitransparentes para velocidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_relative_error_pct_violin_vorticity_scale": "Boxplot de error relativo (%) con puntos semitransparentes para vorticidad (Mean/SD), con cola superior robusta al P95.",
        "table2_relative_error_pct_violin_vorticity_shape": "Boxplot de error relativo (%) con puntos semitransparentes para vorticidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_temporal_mean_relative_error_bar": "Promedio temporal del error absoluto por variable.",
        "table2_temporal_mean_abs_error_bar_velocity": "Barras temporal-mean del error absoluto medio por variable de velocidad.",
        "table2_temporal_mean_abs_error_bar_vorticity": "Barras temporal-mean del error absoluto medio por variable de vorticidad.",
        "table2_temporal_mean_relative_error_pct_bar_velocity": "Barras temporal-mean del error relativo medio (%) por variable de velocidad.",
        "table2_temporal_mean_relative_error_pct_bar_vorticity": "Barras temporal-mean del error relativo medio (%) por variable de vorticidad.",
        "table2_temporal_mean_abs_error_violin_velocity_scale": "Boxplot temporal-mean de error absoluto con puntos para velocidad (Mean/SD), con cola superior robusta al P95.",
        "table2_temporal_mean_abs_error_violin_velocity_shape": "Boxplot temporal-mean de error absoluto con puntos para velocidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_temporal_mean_abs_error_violin_vorticity_scale": "Boxplot temporal-mean de error absoluto con puntos para vorticidad (Mean/SD), con cola superior robusta al P95.",
        "table2_temporal_mean_abs_error_violin_vorticity_shape": "Boxplot temporal-mean de error absoluto con puntos para vorticidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_temporal_mean_relative_error_pct_violin_velocity_scale": "Boxplot temporal-mean del error relativo (%) con puntos para velocidad (Mean/SD), con cola superior robusta al P95.",
        "table2_temporal_mean_relative_error_pct_violin_velocity_shape": "Boxplot temporal-mean del error relativo (%) con puntos para velocidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "table2_temporal_mean_relative_error_pct_violin_vorticity_scale": "Boxplot temporal-mean del error relativo (%) con puntos para vorticidad (Mean/SD), con cola superior robusta al P95.",
        "table2_temporal_mean_relative_error_pct_violin_vorticity_shape": "Boxplot temporal-mean del error relativo (%) con puntos para vorticidad (Skewness/Kurtosis), con cola superior robusta al P95.",
        "flow_abs_error_over_time": "Error absoluto temporal del flujo por frame.",
        "correlation_pearson_r": "Comparación de Pearson r entre métricas y métodos.",
        "correlation_rmse_only": "Comparación de RMSE entre métricas y métodos.",
        "voxel_distribution_std": "Desviación estándar de la distribución voxel a voxel por canal.",
        "significance_pvalues": "Resumen de significancia estadística (Wilcoxon) por análisis y región.",
    }
    publication_fig_tags = "\n".join(
        [
            f"<h3>{title}</h3>{_img_with_caption(publication_figs[key], title, publication_desc.get(key, 'Figura resumen de métricas para comparación de métodos.'))}"
            for key, title in publication_fig_order
            if key in publication_figs
        ]
    )
    publication_fig_section = (
        "<h2>Additional Publication-style Figures</h2>"
        "<p class=\"muted\">Automated summary figures generated from report metrics with a colorblind-safe palette "
        f"(reference=green, baseline=orange, {sr_role_label}=blue).</p>"
        f"{publication_fig_tags}"
        if publication_fig_tags
        else ""
    )
    saved_publication_fig_items = "".join(
        [f"<li><code>figures/{publication_figs[key]}</code></li>" for key, _ in publication_fig_order if key in publication_figs]
    )
    centerline_overlay_tag = _img_with_caption(
        centerline_overlay_name,
        "Centerline overlay",
        "Proyecciones ortogonales de la máscara con trayectoria del centerline y puntos de planos válidos/no válidos.",
    ) if centerline_overlay_name else ""
    centerline_3d_tag = _img_with_caption(
        centerline_3d_name,
        "Centerline 3D",
        "Vista 3D del lumen, centerline y planos muestreados para integración de flujo.",
    ) if centerline_3d_name else ""
    centerline_sections_tag = _img_with_caption(
        centerline_sections_name,
        "Centerline plane sections",
        "Cortes ortogonales del centerline para control de calidad geométrico de secciones.",
    ) if centerline_sections_name else ""
    centerline_peak_qs_tag = _img_with_caption(
        centerline_peak_qs_name,
        "Centerline peak flow along vessel",
        "Perfil Q(s) a lo largo del vaso en el frame pico para evaluar consistencia espacial del flujo.",
    ) if centerline_peak_qs_name else ""
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
        f"{corr_speed_tags}"
        f"{corr_flow_tags}"
        f"{comp_corr_tags}"
        f"{corr_html}"
    )

    pressure_section = (
        "<h2>Relative Pressure Diagnostics</h2>"
        "<p class=\"muted\">Pending: this repository currently does not include a vWERP pipeline, "
        "so pressure metrics from the cerebrovascular paper are not yet computed.</p>"
    )

    saved_wss_items = (
        f"<li><code>{metrics_rel_prefix}/table3_like_wss.csv</code></li>"
        f"<li><code>{metrics_rel_prefix}/table3_like_wss_per_frame.csv</code></li>"
        if bool(args.include_wss)
        else ""
    )
    if flow_method == "centerline":
        p_peak_planes = centerline_error_p.get("peak_plane_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))
        p_all_planes = centerline_error_p.get("all_planes_abs_err_wilcoxon_p_baseline_vs_sr", float("nan"))
        p_peak_txt = "nan" if not np.isfinite(float(p_peak_planes)) else f"{float(p_peak_planes):.4g}"
        p_all_txt = "nan" if not np.isfinite(float(p_all_planes)) else f"{float(p_all_planes):.4g}"
        centerline_exec_bullet = (
            f"<li>Centerline plane-error p-values (baseline vs {sr_role_label.lower()}): peak planes <b>{p_peak_txt}</b>, "
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
            f"Wilcoxon p (|err| baseline vs {sr_role_label.lower()}, peak frame planes): <b>{p_peak_txt}</b>; "
            f"all frame-plane samples: <b>{p_all_txt}</b>.</p>"
            f"{centerline_peak_qs_tag}"
            "<h3>Centerline Sign QC</h3>"
            "<p class=\"muted\">Negative temporal correlation with reference may indicate inverted proximal-distal orientation.</p>"
            f"{centerline_sign_html}"
        )
        saved_centerline_parts = [
            f"<li><code>{metrics_rel_prefix}/centerline_points.csv</code></li>",
            f"<li><code>{metrics_rel_prefix}/flow_centerline_planes_per_frame.csv</code></li>",
            f"<li><code>{metrics_rel_prefix}/centerline_section_qc.csv</code></li>",
            f"<li><code>{metrics_rel_prefix}/centerline_sign_qc.csv</code></li>",
        ]
        for fig_rel in (centerline_overlay_name, centerline_3d_name, centerline_sections_name, centerline_peak_qs_name):
            if fig_rel:
                saved_centerline_parts.append(f"<li><code>figures/{fig_rel}</code></li>")
        saved_centerline_items = "".join(saved_centerline_parts)
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

    report_title_with_mode = f"{args.report_title} [{task_mode_label}]"
    detected_res_text = "unknown" if detected_res_increase is None else str(int(detected_res_increase))
    if spatial_crop_applied:
        spatial_crop_bullet = (
            f"<li>Spatial shape harmonization: <b>cropped to {spatial_shapes_cropped['pred_xyz']}</b> "
            f"(from pred={spatial_shapes_original['pred_xyz']}, gt={spatial_shapes_original['gt_xyz']}, "
            f"mask={spatial_shapes_original['mask_xyz']})</li>"
        )
    else:
        spatial_crop_bullet = "<li>Spatial shape harmonization: <b>not needed</b></li>"
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{report_title_with_mode}</title>
  <style>
    body {{ font-family: 'Times New Roman', Times, serif; margin: 24px; color: #1f2937; }}
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
  <h1>{report_title_with_mode}</h1>
  <p class=\"muted\">Generated from payload: <code>{Path(args.payload_npz).resolve()}</code></p>
  <p class=\"muted\">Baseline source payload: <code>{baseline_payload_path}</code> ({baseline_source_tag})</p>
  <p class=\"muted\">Mode-specific outputs are saved under <code>figures/{task_mode_tag}/</code> and <code>{metrics_rel_prefix}/</code>.</p>

  <p>
    <span class=\"pill\">Task mode: {task_mode_label}</span>
    <span class=\"pill\">Reference: {ref_label}</span>
    <span class=\"pill\">Baseline: {baseline_label}</span>
    <span class=\"pill\">{sr_role_label}: {sr_label}</span>
  </p>

  <h2>Executive Summary</h2>
  <ul>
    <li>Intraluminal statistics RE comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_re) else f'{p_re:.4g}'}</b></li>
    <li>Temporal flow absolute error comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_flow) else f'{p_flow:.4g}'}</b></li>
    <li>WSS absolute error comparison (Wilcoxon p): <b>{wss_summary_text}</b></li>
    <li>Flow reference value used (ml/s): <b>{q_ref_scalar:.6f}</b></li>
    <li>Task naming mode: <b>{task_mode_tag}</b> (res_increase detected: <b>{detected_res_text}</b>)</li>
    {centerline_exec_bullet}
    <li>{flow_method_bullet}</li>
    <li>{flow_axis_bullet}</li>
    <li>{flow_temporal_bullet}</li>
    {spatial_crop_bullet}
    <li>LR magnitude channel used for visualization: <b>{args.lr_mag_channel}</b></li>
    <li>ROI mode: <b>{'enabled' if roi_info.get('enabled', False) else 'disabled'}</b>{'' if not roi_info.get('enabled', False) else f" (bbox xyz: {roi_info.get('bbox_xyz')})"}</li>
    <li>Baseline source: <b>{baseline_source_tag}</b> ({Path(baseline_payload_path).name})</li>
    <li>Baseline LR alignment for metrics: <b>{'upsampled to HR grid' if tuple(lr_norm.shape[2:]) != tuple(gt_norm.shape[2:]) else 'native HR size'}</b></li>
  </ul>

  <h2>{'Visual Inspection (ROI Bounding Box)' if roi_info.get('enabled', False) else 'Visual Inspection (Full Volume)'}</h2>
  {ch_img_tags}

  <h2>Voxel Distribution Inside Mask</h2>
  <p class=\"muted\">Histogram comparison over all in-mask voxels across all processed frames{'' if not roi_info.get('enabled', False) else ' (restricted to ROI bbox)'}.</p>
  <img src=\"figures/{_rel_fig_path(fig_voxel_hist, fig_dir)}\" alt=\"In-mask voxel histograms\"/>
  <p class=\"muted\">Histograma global dentro de la máscara: compara la distribución de valores entre referencia, baseline y {sr_role_label} usando todos los frames.</p>
  {voxel_hist_channel_tags}
  {voxel_dist_html}

  <h2>Flow-rate Diagnostics</h2>
  <img src=\"figures/{_rel_fig_path(fig_flow, fig_dir)}\" alt=\"Flow profile\"/>
  <p class=\"muted\">Perfil temporal de flujo integrado por frame para referencia, baseline y {sr_role_label}.</p>
  <p class=\"muted\">{flow_diag_text}</p>
  {flow_axis_section}
  {centerline_section}

  <h2>Paper-style Table 2 (Representative Slices)</h2>
  <p class=\"muted\">Variables: mean/SD/skewness/kurtosis of intraluminal velocity and vorticity (aggregated over all processed frames). Location labels are voxel slice IDs along the selected flow axis.</p>
  {t2_comp_html}
  <h3>Table 2 Temporal Mean (Averaged Over Frames)</h3>
  <p class=\"muted\">Frame-wise Table-2 metrics averaged over time for each slice and variable.</p>
  {t2_tm_html}

  <h2>Paper-style Flow Metrics</h2>
  <p class=\"muted\">Temporal summaries against reference flow (Qref).</p>
  {flow_html}
  <h3>Flow Peak Metrics (Paper-style)</h3>
  <p class=\"muted\">Peak-frame and temporal RMSE/relative-error summaries aligned with cerebrovascular {sr_role_label.lower()} comparisons.</p>
  {flow_peak_html}
  <h3>Flow Average Metrics (Paper-style)</h3>
  <p class=\"muted\">Temporal mean-flow and average-error summaries using all frames.</p>
  {flow_avg_html}

  {wss_section}

  <h2>Geometry Uncertainty (Surface/Hausdorff)</h2>
  <p class=\"muted\">{geom_note}</p>
  {geom_html}

  {corr_section}

  {pressure_section}

  <h2>Bland-Altman</h2>
  {ba_speed_tags}
  <p class=\"muted\">Component plots below use all in-mask voxels aggregated over all processed frames.</p>
  {comp_ba_tags}
  {ba_html}

  <h2>Peak Velocity Metrics (Core/Wall/Intraluminal)</h2>
  <p class=\"muted\">Peak-flow velocity magnitude metrics (MAE, RMSE, relative error, cosine similarity), reported for core, wall, and full intraluminal masks.</p>
  {peak_vel_html}
  <h2>Mean Velocity Metrics (Core/Wall/Intraluminal)</h2>
  <p class=\"muted\">All-frames in-mask velocity magnitude metrics (MAE, RMSE, relative error, cosine similarity), reported for core, wall, and full intraluminal masks.</p>
  {mean_vel_html}

  <h2>Statistical Significance (Baseline vs {sr_role_label})</h2>
  <p class=\"muted\">Paired Wilcoxon tests on voxel-wise absolute errors against reference (peak frame and all-frame mean analyses).</p>
  {pvalue_html}

  {publication_fig_section}

  <h2>Saved Artifacts</h2>
  <ul>
    <li><code>{metrics_rel_prefix}/table2_like_all_slices.csv</code></li>
    <li><code>{metrics_rel_prefix}/table2_like_per_frame_all_slices.csv</code></li>
    <li><code>{metrics_rel_prefix}/table2_like_temporal_mean.csv</code></li>
    <li><code>{metrics_rel_prefix}/table2_like_compact.csv</code></li>
    <li><code>{metrics_rel_prefix}/table2_relative_robust_summary.csv</code></li>
    <li><code>{metrics_rel_prefix}/table2_temporal_mean_relative_robust_summary.csv</code></li>
    <li><code>{metrics_rel_prefix}/flow_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/flow_metrics_per_frame.csv</code></li>
    <li><code>{metrics_rel_prefix}/flow_rate_curves_per_frame.csv</code></li>
    {saved_centerline_items}
    {saved_wss_items}
    <li><code>{metrics_rel_prefix}/geometry_temporal_surface_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/voxel_distribution_stats.csv</code></li>
    <li><code>{metrics_rel_prefix}/correlation_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/bland_altman_stats.csv</code></li>
    <li><code>{metrics_rel_prefix}/peak_velocity_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/mean_velocity_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/significance_pvalues.csv</code></li>
    <li><code>{metrics_rel_prefix}/flow_peak_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/flow_average_metrics.csv</code></li>
    <li><code>{metrics_rel_prefix}/run_context.csv</code></li>
    {saved_voxel_hist_items}
    {saved_ba_speed_single_items}
    {saved_corr_single_items}
    {saved_comp_ba_items}
    {saved_comp_ba_single_items}
    {saved_publication_fig_items}
    <li><code>{metrics_rel_prefix}/summary_metrics.json</code></li>
  </ul>

</body>
</html>
"""

    report_mode_path = out_dir / f"report_{task_mode_tag}.html"
    report_mode_path.write_text(html, encoding="utf-8")
    report_path = out_dir / "report.html"
    if report_path != report_mode_path:
        report_path.write_text(html, encoding="utf-8")

    print("Report generated:")
    print(f"- HTML (mode-specific): {report_mode_path}")
    if report_path != report_mode_path:
        print(f"- HTML (latest alias): {report_path}")
    print(f"- Figures: {fig_mode_dir}")
    print(f"- Metrics: {metrics_dir}")


if __name__ == "__main__":
    main()
