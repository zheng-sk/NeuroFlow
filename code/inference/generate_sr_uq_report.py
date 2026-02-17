import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy import stats
    from scipy.ndimage import binary_erosion, gaussian_filter
    from scipy.spatial import cKDTree
except Exception as exc:
    raise RuntimeError(
        "This script requires scipy for statistical tests, interpolation helpers, and surface metrics."
    ) from exc

try:
    from skimage.measure import marching_cubes
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
    labels = ["CCA", "Bifurcation", "ICA/ECA"]
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
    slice_count = speed_ref.shape[flow_axis]
    rows: List[Dict[str, Any]] = []
    valid_slices: List[int] = []
    re_base_all: List[float] = []
    re_sr_all: List[float] = []

    for s in range(slice_count):
        mask_sl = _extract_slice(mask_ref, flow_axis, s) > 0.5
        if int(mask_sl.sum()) < int(min_voxels):
            continue

        valid_slices.append(s)

        sv_ref = _extract_slice(speed_ref, flow_axis, s)[mask_sl]
        sv_base = _extract_slice(speed_base, flow_axis, s)[mask_sl]
        sv_sr = _extract_slice(speed_sr, flow_axis, s)[mask_sl]

        vo_ref = _extract_slice(vort_ref, flow_axis, s)[mask_sl]
        vo_base = _extract_slice(vort_base, flow_axis, s)[mask_sl]
        vo_sr = _extract_slice(vort_sr, flow_axis, s)[mask_sl]

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


def _html_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    th = "".join(f"<th>{c}</th>" for c in columns)
    body_parts = []
    for row in rows:
        tds = "".join(f"<td>{row.get(c, '')}</td>" for c in columns)
        body_parts.append(f"<tr>{tds}</tr>")
    body = "\n".join(body_parts)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _save_channel_figure(
    out_path: Path,
    ch_name: str,
    lr_up: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    max_slices: int,
) -> None:
    n_slices = gt.shape[-1]
    if n_slices <= max_slices:
        z_idx = list(range(n_slices))
    else:
        z_idx = np.linspace(0, n_slices - 1, max_slices).round().astype(int).tolist()

    is_mag = ch_name == "mag"
    cmap = "gray" if is_mag else "coolwarm"
    vmin, vmax = ((0.0, 1.0) if is_mag else (-1.0, 1.0))

    fig, axes = plt.subplots(4, len(z_idx), figsize=(3.0 * len(z_idx), 10))
    if len(z_idx) == 1:
        axes = axes.reshape(4, 1)

    for j, z in enumerate(z_idx):
        inp = lr_up[:, :, z]
        pd = pred[:, :, z]
        gtz = gt[:, :, z]
        err = np.abs(pd - gtz)

        im0 = axes[0, j].imshow(inp, cmap=cmap, vmin=vmin, vmax=vmax)
        im1 = axes[1, j].imshow(pd, cmap=cmap, vmin=vmin, vmax=vmax)
        im2 = axes[2, j].imshow(gtz, cmap=cmap, vmin=vmin, vmax=vmax)
        im3 = axes[3, j].imshow(err, cmap="magma")

        axes[0, j].set_title(f"z={z}", fontsize=10)
        for r in range(4):
            axes[r, j].axis("off")

    axes[0, 0].set_ylabel("Input (LR up)", fontsize=11)
    axes[1, 0].set_ylabel("Prediction", fontsize=11)
    axes[2, 0].set_ylabel("Ground truth", fontsize=11)
    axes[3, 0].set_ylabel("|Error|", fontsize=11)

    plt.colorbar(im0, ax=axes[0, :], fraction=0.012, pad=0.01)
    plt.colorbar(im1, ax=axes[1, :], fraction=0.012, pad=0.01)
    plt.colorbar(im2, ax=axes[2, :], fraction=0.012, pad=0.01)
    plt.colorbar(im3, ax=axes[3, :], fraction=0.012, pad=0.01)

    fig.suptitle(f"Full-volume comparison: {ch_name}", fontsize=14)
    fig.tight_layout()
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

    parser.add_argument("--flow-axis", type=int, default=2, choices=[0, 1, 2], help="Axis used for cross-sectional flow integration")
    parser.add_argument("--selected-frame", type=int, default=0, help="Frame index (within payload) used for visual panel")
    parser.add_argument("--max-display-slices", type=int, default=12, help="Max slices per visual panel")
    parser.add_argument("--mask-min-slice-voxels", type=int, default=25, help="Min in-mask voxels per slice for slice-wise stats")

    parser.add_argument("--q-ref", type=float, default=float("nan"), help="Reference flow rate in ml/s (paper uses calibrated reference).")
    parser.add_argument("--cca-range", type=str, default="", help="Optional slice range start:end for CCA-only flow stats.")

    parser.add_argument("--mu-pa-s", type=float, default=0.0035, help="Dynamic viscosity for WSS estimation (Pa*s)")
    parser.add_argument("--max-wall-points", type=int, default=30000, help="Max wall points sampled for WSS distribution")

    parser.add_argument("--baseline-label", default="3T", help="Label for baseline (native/LR) method")
    parser.add_argument("--sr-label", default="3T SR", help="Label for super-resolved method")
    parser.add_argument("--ref-label", default="CFD", help="Label for reference method")
    parser.add_argument("--report-title", default="4D Flow SR Uncertainty Quantification Report", help="Report title")

    args = parser.parse_args()

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

    lr_norm = payload["lr_norm"].astype(np.float32)  # [T,6,X,Y,Z]
    pred_norm = payload["pred_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    gt_norm = payload["gt_norm"].astype(np.float32)  # [T,4,X,Y,Z]
    mask = payload["mask"].astype(np.float32)  # [T,X,Y,Z]
    venc = payload["venc"].astype(np.float32)  # [T]
    hr_spacing = tuple(float(x) for x in payload["hr_spacing"].tolist())

    if pred_norm.shape[1] != 4 or gt_norm.shape[1] != 4:
        raise ValueError(f"Expected 4-channel pred/gt tensors. Got pred={pred_norm.shape}, gt={gt_norm.shape}")

    t_count = pred_norm.shape[0]
    fidx = int(np.clip(args.selected_frame, 0, t_count - 1))

    # Denormalize to physical units for velocity-related metrics
    pred_phys = pred_norm.copy()
    gt_phys = gt_norm.copy()
    lr_vel_phys = lr_norm[:, :3].copy()
    for t in range(t_count):
        pred_phys[t, :3] *= float(venc[t])
        gt_phys[t, :3] *= float(venc[t])
        lr_vel_phys[t] *= float(venc[t])

    # Derive LR 4-channel display tensor (u,v,w,mag proxy)
    lr_mag_proxy = np.sqrt(np.maximum(lr_norm[:, 3], 0.0) ** 2 + np.maximum(lr_norm[:, 4], 0.0) ** 2 + np.maximum(lr_norm[:, 5], 0.0) ** 2)
    lr_4ch = np.concatenate([lr_norm[:, :3], lr_mag_proxy[:, None]], axis=1).astype(np.float32)

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
        )
        channel_figs[name] = str(out_img.name)

    # Mean fields for paper-style stats
    mask_ref = (mask.mean(axis=0) >= 0.5).astype(np.float32)
    pred_vel_mean = pred_phys[:, :3].mean(axis=0)
    gt_vel_mean = gt_phys[:, :3].mean(axis=0)
    lr_vel_mean = lr_vel_phys.mean(axis=0)

    spacing_m = tuple(float(s) / 1000.0 for s in hr_spacing)
    speed_ref = np.sqrt((gt_vel_mean**2).sum(axis=0)).astype(np.float32)
    speed_base = np.sqrt((lr_vel_mean**2).sum(axis=0)).astype(np.float32)
    speed_sr = np.sqrt((pred_vel_mean**2).sum(axis=0)).astype(np.float32)

    vort_ref = _vorticity_magnitude(gt_vel_mean, spacing_m)
    vort_base = _vorticity_magnitude(lr_vel_mean, spacing_m)
    vort_sr = _vorticity_magnitude(pred_vel_mean, spacing_m)

    table2_all, valid_slices, re_base_all, re_sr_all = _table2_rows(
        speed_ref=speed_ref,
        speed_base=speed_base,
        speed_sr=speed_sr,
        vort_ref=vort_ref,
        vort_base=vort_base,
        vort_sr=vort_sr,
        mask_ref=mask_ref,
        flow_axis=int(args.flow_axis),
        min_voxels=int(args.mask_min_slice_voxels),
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

    t2c_cols = ["location", "slice_index", "variable", "ref", "baseline", "sr", "re_baseline", "re_sr"]
    _write_csv(metrics_dir / "table2_like_compact.csv", table2_compact, t2c_cols)

    # 2) Flow-rate metrics
    q_ref_curves = _flow_rate_curves(gt_phys[:, :3], mask, flow_axis=int(args.flow_axis), spacing_mm=hr_spacing)
    q_base_curves = _flow_rate_curves(lr_vel_phys, mask, flow_axis=int(args.flow_axis), spacing_mm=hr_spacing)
    q_sr_curves = _flow_rate_curves(pred_phys[:, :3], mask, flow_axis=int(args.flow_axis), spacing_mm=hr_spacing)

    q_ref_mean = q_ref_curves.mean(axis=0)
    q_ref_sd = q_ref_curves.std(axis=0, ddof=1) if q_ref_curves.shape[0] > 1 else np.zeros_like(q_ref_mean)
    q_base_mean = q_base_curves.mean(axis=0)
    q_base_sd = q_base_curves.std(axis=0, ddof=1) if q_base_curves.shape[0] > 1 else np.zeros_like(q_base_mean)
    q_sr_mean = q_sr_curves.mean(axis=0)
    q_sr_sd = q_sr_curves.std(axis=0, ddof=1) if q_sr_curves.shape[0] > 1 else np.zeros_like(q_sr_mean)

    if np.isfinite(float(args.q_ref)):
        q_ref_scalar = float(args.q_ref)
    else:
        q_ref_scalar = float(np.median(q_ref_mean[np.isfinite(q_ref_mean)]))

    def flow_summary(name: str, q_mean: np.ndarray, q_sd: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
        mad = float(np.mean(np.abs(q_mean[idx] - q_ref_scalar)))
        mean_sd = float(np.mean(q_sd[idx]))
        mean_q = float(np.mean(q_mean[idx]))
        return {
            "method": name,
            "mean_Q_ml_s": mean_q,
            "MAD_Q_ml_s": mad,
            "MAD_Q_pct_ref": 100.0 * mad / (abs(q_ref_scalar) + 1e-12),
            "mean_SD_Q_ml_s": mean_sd,
            "mean_SD_Q_pct_ref": 100.0 * mean_sd / (abs(q_ref_scalar) + 1e-12),
        }

    all_idx = np.arange(q_ref_mean.shape[0])
    flow_rows = [
        flow_summary(args.ref_label, q_ref_mean, q_ref_sd, all_idx),
        flow_summary(args.baseline_label, q_base_mean, q_base_sd, all_idx),
        flow_summary(args.sr_label, q_sr_mean, q_sr_sd, all_idx),
    ]

    if args.cca_range:
        try:
            a, b = args.cca_range.split(":")
            a_i, b_i = int(a), int(b)
            cca_idx = np.arange(max(0, a_i), min(q_ref_mean.shape[0], b_i))
            if cca_idx.size > 0:
                flow_rows.extend(
                    [
                        {**flow_summary(args.ref_label, q_ref_mean, q_ref_sd, cca_idx), "method": f"{args.ref_label} (CCA)"},
                        {
                            **flow_summary(args.baseline_label, q_base_mean, q_base_sd, cca_idx),
                            "method": f"{args.baseline_label} (CCA)",
                        },
                        {**flow_summary(args.sr_label, q_sr_mean, q_sr_sd, cca_idx), "method": f"{args.sr_label} (CCA)"},
                    ]
                )
        except Exception:
            pass

    # Flow p-value comparing absolute errors vs reference scalar
    flow_err_base = np.abs(q_base_mean - q_ref_scalar)
    flow_err_sr = np.abs(q_sr_mean - q_ref_scalar)
    p_flow = _wilcoxon_p(flow_err_base.tolist(), flow_err_sr.tolist())

    flow_cols = ["method", "mean_Q_ml_s", "MAD_Q_ml_s", "MAD_Q_pct_ref", "mean_SD_Q_ml_s", "mean_SD_Q_pct_ref"]
    _write_csv(metrics_dir / "flow_metrics.csv", flow_rows, flow_cols)

    # Flow figure
    fig_flow = fig_dir / "flow_rate_profile.png"
    x = np.arange(q_ref_mean.shape[0])
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, q_ref_mean, label=args.ref_label, linewidth=2)
    plt.fill_between(x, q_ref_mean - q_ref_sd, q_ref_mean + q_ref_sd, alpha=0.2)
    plt.plot(x, q_base_mean, label=args.baseline_label, linewidth=2)
    plt.fill_between(x, q_base_mean - q_base_sd, q_base_mean + q_base_sd, alpha=0.2)
    plt.plot(x, q_sr_mean, label=args.sr_label, linewidth=2)
    plt.fill_between(x, q_sr_mean - q_sr_sd, q_sr_mean + q_sr_sd, alpha=0.2)
    plt.axhline(q_ref_scalar, linestyle="--", color="black", label=f"Qref={q_ref_scalar:.3f} ml/s")
    plt.xlabel(f"Slice index along axis {args.flow_axis}")
    plt.ylabel("Flow rate [ml/s]")
    plt.title("Flow-rate profile (mean ± SD across frames)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_flow, dpi=180)
    plt.close(fig)

    # 3) WSS statistics (Table-3-like)
    tau_ref = _compute_wss_distribution(
        uvw_mean=gt_vel_mean,
        mask_ref=mask_ref,
        spacing_mm=hr_spacing,
        mu_pa_s=float(args.mu_pa_s),
        max_points=int(args.max_wall_points),
        seed=7,
    )
    tau_base = _compute_wss_distribution(
        uvw_mean=lr_vel_mean,
        mask_ref=mask_ref,
        spacing_mm=hr_spacing,
        mu_pa_s=float(args.mu_pa_s),
        max_points=int(args.max_wall_points),
        seed=7,
    )
    tau_sr = _compute_wss_distribution(
        uvw_mean=pred_vel_mean,
        mask_ref=mask_ref,
        spacing_mm=hr_spacing,
        mu_pa_s=float(args.mu_pa_s),
        max_points=int(args.max_wall_points),
        seed=7,
    )

    wss_ref = _wss_summary(tau_ref)
    wss_base = _wss_summary(tau_base)
    wss_sr = _wss_summary(tau_sr)

    table3_rows = []
    for key in ["Maximum", "Mean", "SD", "Quantile_97_5", "Median", "Quantile_2_5", "IQR_75_25"]:
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
        plt.hist(tau_ref, bins=bins, alpha=0.4, density=True, label=args.ref_label)
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

    # 4) Geometry metrics (paper-like) from temporal mask variability
    geom_rows: List[Dict[str, Any]] = []
    if mask.shape[0] > 1:
        ref_mask = (mask[0] > 0.5).astype(np.uint8)
        per_frame = []
        for t in range(mask.shape[0]):
            m_t = (mask[t] > 0.5).astype(np.uint8)
            m = _surface_distance_metrics(m_t, ref_mask, hr_spacing)
            m["frame"] = int(t)
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
                "mean_surface_distance_a_to_b_mm",
                "std_surface_distance_a_to_b_mm",
                "symmetric_mean_surface_distance_mm",
                "hausdorff_distance_mm",
            ],
        )

    # 5) Additional diagnostic plots
    # Bland-Altman for intraluminal speed
    m_in = mask_ref > 0.5
    sp_ref = speed_ref[m_in]
    sp_sr = speed_sr[m_in]
    if sp_ref.size > 20:
        mean_sp = 0.5 * (sp_sr + sp_ref)
        diff_sp = sp_sr - sp_ref
        bias = float(np.mean(diff_sp))
        sd = float(np.std(diff_sp, ddof=1))
        loa_l = bias - 1.96 * sd
        loa_h = bias + 1.96 * sd

        fig_ba = fig_dir / "bland_altman_speed.png"
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(mean_sp, diff_sp, s=4, alpha=0.2)
        plt.axhline(bias, color="red", linestyle="-", label=f"Bias={bias:.4f}")
        plt.axhline(loa_l, color="black", linestyle="--", label=f"LoA low={loa_l:.4f}")
        plt.axhline(loa_h, color="black", linestyle="--", label=f"LoA high={loa_h:.4f}")
        plt.xlabel("Mean(speed_SR, speed_ref) [m/s]")
        plt.ylabel("speed_SR - speed_ref [m/s]")
        plt.title("Bland-Altman: intraluminal speed")
        plt.legend()
        plt.tight_layout()
        fig.savefig(fig_ba, dpi=180)
        plt.close(fig)
        ba_speed_name = fig_ba.name
    else:
        ba_speed_name = ""

    summary = {
        "report_title": args.report_title,
        "labels": {
            "reference": args.ref_label,
            "baseline": args.baseline_label,
            "super_resolved": args.sr_label,
        },
        "payload_path": str(Path(args.payload_npz).resolve()),
        "metadata": metadata,
        "dimensions": {
            "T": int(t_count),
            "shape_XYZ": [int(x) for x in pred_norm.shape[2:]],
            "flow_axis": int(args.flow_axis),
        },
        "statistics": {
            "table2_wilcoxon_p_re_baseline_vs_sr": p_re,
            "flow_wilcoxon_p_abs_err": p_flow,
            "wss_wilcoxon_p_abs_err": p_wss,
            "flow_reference_q_ml_s": q_ref_scalar,
            "geometry_summary": geom_summary,
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
    t3_html = _html_table(fmt_rows(table3_rows, t3_cols, nd=6), t3_cols)

    flow_html = _html_table(fmt_rows(flow_rows, flow_cols, nd=6), flow_cols)

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
    <span class=\"pill\">Reference: {args.ref_label}</span>
    <span class=\"pill\">Baseline: {args.baseline_label}</span>
    <span class=\"pill\">Super-resolved: {args.sr_label}</span>
  </p>

  <h2>Executive Summary</h2>
  <ul>
    <li>Intraluminal statistics RE comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_re) else f'{p_re:.4g}'}</b></li>
    <li>Flow profile absolute error comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_flow) else f'{p_flow:.4g}'}</b></li>
    <li>WSS absolute error comparison (Wilcoxon p): <b>{'nan' if not np.isfinite(p_wss) else f'{p_wss:.4g}'}</b></li>
    <li>Flow reference value used (ml/s): <b>{q_ref_scalar:.6f}</b></li>
  </ul>

  <h2>Visual Inspection (Full Volume)</h2>
  {ch_img_tags}

  <h2>Flow-rate Diagnostics</h2>
  <img src=\"figures/{fig_flow.name}\" alt=\"Flow profile\"/>

  <h2>Paper-style Table 2 (Representative Locations)</h2>
  <p class=\"muted\">Variables: mean/SD/skewness/kurtosis of intraluminal velocity and vorticity.</p>
  {t2_comp_html}

  <h2>Paper-style Flow Metrics</h2>
  <p class=\"muted\">MAD and SD summaries against reference flow (Qref).</p>
  {flow_html}

  <h2>Paper-style Table 3 (WSS)</h2>
  <p class=\"muted\">WSS estimated from boundary-normal finite differences (2-point polynomial approximation).</p>
  {t3_html}
  <img src=\"figures/{fig_wss.name}\" alt=\"WSS distribution\"/>

  <h2>Geometry Uncertainty (Surface/Hausdorff)</h2>
  <p class=\"muted\">Computed across temporal masks when multiple frames are available.</p>
  {geom_html}

  <h2>Bland-Altman</h2>
  {'' if not ba_speed_name else f'<img src="figures/{ba_speed_name}" alt="Bland-Altman speed"/>'}

  <h2>Saved Artifacts</h2>
  <ul>
    <li><code>metrics/table2_like_all_slices.csv</code></li>
    <li><code>metrics/table2_like_compact.csv</code></li>
    <li><code>metrics/flow_metrics.csv</code></li>
    <li><code>metrics/table3_like_wss.csv</code></li>
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
