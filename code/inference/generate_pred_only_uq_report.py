import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats

import generate_sr_uq_report as uq


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "semibold",
        "axes.grid": False,
        "font.size": 10,
    }
)


def _load_nifti_time_first(path: str, time_axis: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim == 4:
        if time_axis < 0:
            time_axis = data.ndim + time_axis
        if time_axis < 0 or time_axis >= data.ndim:
            raise ValueError(f"Invalid time_axis={time_axis} for data shape {data.shape}")
        if time_axis != 0:
            data = np.moveaxis(data, time_axis, 0)
    else:
        raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape} for {path}")

    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data.astype(np.float32), img.affine.astype(np.float32), spacing


def _align_mask_txyz(mask_raw: np.ndarray, t_count: int, shape_xyz: Tuple[int, int, int]) -> np.ndarray:
    if mask_raw.ndim == 3:
        if tuple(mask_raw.shape) != tuple(shape_xyz):
            raise ValueError(f"3D mask shape {mask_raw.shape} does not match prediction shape {shape_xyz}")
        return np.repeat(mask_raw[None, ...], t_count, axis=0).astype(np.float32)

    if mask_raw.ndim != 4:
        raise ValueError(f"Mask must be 3D or 4D, got shape {mask_raw.shape}")

    if tuple(mask_raw.shape[1:]) != tuple(shape_xyz):
        raise ValueError(f"4D mask spatial shape {mask_raw.shape[1:]} does not match prediction shape {shape_xyz}")

    if mask_raw.shape[0] == t_count:
        return mask_raw.astype(np.float32)
    if mask_raw.shape[0] == 1:
        return np.repeat(mask_raw, t_count, axis=0).astype(np.float32)

    raise ValueError(f"Mask temporal size {mask_raw.shape[0]} does not match prediction frames {t_count}")


def _table2_rows_single(
    speed: np.ndarray,
    vort: np.ndarray,
    mask_ref: np.ndarray,
    flow_axis: int,
    min_voxels: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if speed.ndim != 4:
        raise ValueError(f"Expected speed with shape [T,X,Y,Z], got {speed.shape}")

    t_count = speed.shape[0]
    slice_count = speed.shape[flow_axis + 1]
    rows: List[Dict[str, Any]] = []
    valid_slices: List[int] = []

    for s in range(slice_count):
        sv_parts: List[np.ndarray] = []
        vo_parts: List[np.ndarray] = []
        total_vox = 0

        for t in range(t_count):
            mask_sl = uq._extract_slice(mask_ref[t], flow_axis, s) > 0.5
            if int(mask_sl.sum()) == 0:
                continue

            sv_parts.append(uq._extract_slice(speed[t], flow_axis, s)[mask_sl])
            vo_parts.append(uq._extract_slice(vort[t], flow_axis, s)[mask_sl])
            total_vox += int(mask_sl.sum())

        if total_vox < int(min_voxels):
            continue

        valid_slices.append(int(s))
        sv = np.concatenate(sv_parts, axis=0)
        vo = np.concatenate(vo_parts, axis=0)

        metric_defs = [
            ("Mean velocity [m/s]", lambda a: float(np.mean(a)), sv),
            ("SD velocity [m/s]", lambda a: float(np.std(a, ddof=1)) if a.size > 1 else float("nan"), sv),
            ("Skewness velocity", lambda a: float(stats.skew(a, bias=False)) if a.size >= 3 else float("nan"), sv),
            ("Kurtosis velocity", lambda a: float(stats.kurtosis(a, fisher=True, bias=False)) if a.size >= 4 else float("nan"), sv),
            ("Mean vorticity [1/s]", lambda a: float(np.mean(a)), vo),
            ("SD vorticity [1/s]", lambda a: float(np.std(a, ddof=1)) if a.size > 1 else float("nan"), vo),
            ("Skewness vorticity", lambda a: float(stats.skew(a, bias=False)) if a.size >= 3 else float("nan"), vo),
            ("Kurtosis vorticity", lambda a: float(stats.kurtosis(a, fisher=True, bias=False)) if a.size >= 4 else float("nan"), vo),
        ]

        for var_name, fn, arr in metric_defs:
            rows.append(
                {
                    "slice_index": int(s),
                    "variable": var_name,
                    "pred": fn(arr),
                }
            )

    return rows, valid_slices


def _table2_rows_single_per_frame(
    speed: np.ndarray,
    vort: np.ndarray,
    mask_ref: np.ndarray,
    flow_axis: int,
    min_voxels: int,
    frame_source_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t_count = speed.shape[0]

    for t in range(t_count):
        t_rows, _ = _table2_rows_single(
            speed=speed[t : t + 1],
            vort=vort[t : t + 1],
            mask_ref=mask_ref[t : t + 1],
            flow_axis=flow_axis,
            min_voxels=min_voxels,
        )
        for r in t_rows:
            rr = dict(r)
            rr["frame_payload_index"] = int(t)
            rr["frame_source_index"] = int(frame_source_indices[t])
            rows.append(rr)

    return rows


def _save_voxel_hist_single(
    out_path: Path,
    pred_4ch: np.ndarray,
    mask: np.ndarray,
    bins: int,
    method_label: str,
) -> List[Dict[str, Any]]:
    mask_bool = mask > 0.5
    if int(mask_bool.sum()) == 0:
        mask_bool = np.ones_like(mask, dtype=bool)

    channel_names = ["u", "v", "w", "mag"]
    color = "#b91c1c"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_f = axes.ravel()
    rows: List[Dict[str, Any]] = []

    for c, ch in enumerate(channel_names):
        ax = axes_f[c]
        vals = pred_4ch[:, c][mask_bool]
        rows.append(uq._distribution_row(ch, method_label, vals))

        sym = ch != "mag"
        vmin, vmax = uq._robust_range([vals], symmetric=sym, lower_q=0.5, upper_q=99.5)
        bin_edges = np.linspace(vmin, vmax, max(20, int(bins)) + 1)

        vv = np.asarray(vals, dtype=np.float64)
        vv = vv[np.isfinite(vv)]
        vv = vv[(vv >= vmin) & (vv <= vmax)]
        vv = uq._subsample_for_plot(vv, seed=19 + c)
        if vv.size > 0:
            ax.hist(vv, bins=bin_edges, density=True, histtype="step", linewidth=1.7, alpha=0.95, color=color, label=method_label)

        ax.set_title(f"{ch.upper()} in-mask voxel distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)

    fig.suptitle("Voxel-value distribution inside vessel mask (prediction-only)", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return rows


def _flow_rows_per_frame_slice_single(
    q_curves: np.ndarray,
    q_ref_scalar: float,
    frame_source_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    t_count, n_slices = q_curves.shape
    rows: List[Dict[str, Any]] = []
    for t in range(t_count):
        for s in range(n_slices):
            q_val = float(q_curves[t, s])
            rows.append(
                {
                    "frame_payload_index": int(t),
                    "frame_source_index": int(frame_source_indices[t]),
                    "slice_index": int(s),
                    "q_pred_ml_s": q_val,
                    "abs_err_pred_vs_qref_ml_s": abs(q_val - float(q_ref_scalar)),
                }
            )
    return rows


def _flow_summary_rows_per_frame_single(
    q_curves: np.ndarray,
    q_ref_scalar: float,
    frame_source_indices: np.ndarray,
    method_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in range(q_curves.shape[0]):
        q_t = q_curves[t]
        rows.append(
            {
                "frame_payload_index": int(t),
                "frame_source_index": int(frame_source_indices[t]),
                "method": method_label,
                "mean_Q_ml_s": float(np.mean(q_t)),
                "MAD_Q_vs_qref_ml_s": float(np.mean(np.abs(q_t - float(q_ref_scalar)))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate prediction-only 4D-flow metrics/report from NIfTI files (u,v,w + optional mag/mask). "
            "No reference or baseline required."
        )
    )
    parser.add_argument("--u-path", required=True, help="Predicted U/Vx NIfTI path.")
    parser.add_argument("--v-path", required=True, help="Predicted V/Vy NIfTI path.")
    parser.add_argument("--w-path", required=True, help="Predicted W/Vz NIfTI path.")
    parser.add_argument("--mag-path", default="", help="Optional predicted magnitude NIfTI path.")
    parser.add_argument("--mask-path", default="", help="Optional vessel mask NIfTI path (3D/4D).")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis in NIfTI when data is 4D.")

    parser.add_argument("--out-dir", required=True, help="Output directory for report artifacts.")
    parser.add_argument("--method-label", default="Prediction", help="Label used in tables/plots.")

    parser.add_argument("--flow-axis", type=str, default="auto", choices=["auto", "0", "1", "2"])
    parser.add_argument("--hist-bins", type=int, default=120)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-min-slice-voxels", type=int, default=25)
    parser.add_argument("--q-ref", type=float, default=float("nan"))
    parser.add_argument("--mu-pa-s", type=float, default=0.0035)
    parser.add_argument("--max-wall-points", type=int, default=30000)

    parser.add_argument(
        "--roi-bbox",
        type=int,
        nargs=6,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="Optional ROI bbox in voxel indices [x0,x1,y0,y1,z0,z1).",
    )
    parser.add_argument(
        "--roi-json",
        default="",
        help="Optional ROI JSON with key bbox_hr_xyz/bbox_xyz.",
    )

    parser.add_argument("--report-title", default="4D Flow Prediction-Only Metrics Report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    u_txyz, aff_u, spacing_mm = _load_nifti_time_first(args.u_path, int(args.time_axis))
    v_txyz, aff_v, _ = _load_nifti_time_first(args.v_path, int(args.time_axis))
    w_txyz, aff_w, _ = _load_nifti_time_first(args.w_path, int(args.time_axis))

    if not (u_txyz.shape == v_txyz.shape == w_txyz.shape):
        raise ValueError(f"u/v/w shapes must match. u={u_txyz.shape}, v={v_txyz.shape}, w={w_txyz.shape}")

    if not (np.allclose(aff_u, aff_v) and np.allclose(aff_u, aff_w)):
        raise ValueError("u/v/w affines do not match after canonicalization.")

    t_count = int(u_txyz.shape[0])
    shape_xyz = tuple(int(v) for v in u_txyz.shape[1:])
    frame_source_indices = np.arange(t_count, dtype=np.int32)

    if args.mag_path:
        mag_txyz, aff_m, _ = _load_nifti_time_first(args.mag_path, int(args.time_axis))
        if mag_txyz.shape != u_txyz.shape:
            raise ValueError(f"mag shape {mag_txyz.shape} does not match velocity shape {u_txyz.shape}")
        if not np.allclose(aff_u, aff_m):
            raise ValueError("mag affine does not match velocity affine after canonicalization.")
        mag_source = "nifti"
    else:
        mag_txyz = np.sqrt(u_txyz**2 + v_txyz**2 + w_txyz**2).astype(np.float32)
        mag_source = "derived_from_velocity"

    if args.mask_path:
        mask_raw, aff_mask, _ = _load_nifti_time_first(args.mask_path, int(args.time_axis))
        if mask_raw.ndim == 4:
            mask_txyz = _align_mask_txyz(mask_raw, t_count=t_count, shape_xyz=shape_xyz)
        else:
            mask_txyz = _align_mask_txyz(mask_raw, t_count=t_count, shape_xyz=shape_xyz)
        if not np.allclose(aff_u, aff_mask):
            raise ValueError("mask affine does not match velocity affine after canonicalization.")
    else:
        mask_txyz = np.ones((t_count, *shape_xyz), dtype=np.float32)

    mask_txyz = (mask_txyz >= float(args.mask_threshold)).astype(np.float32)

    roi_bbox = uq._resolve_roi_bbox(
        roi_bbox_cli=args.roi_bbox,
        roi_json_path=str(args.roi_json),
        shape_xyz=shape_xyz,
    )
    mask_metrics, roi_info = uq._apply_roi_to_mask(mask_txyz=mask_txyz, bbox_xyz=roi_bbox)
    if int((mask_metrics > 0.5).sum()) == 0:
        raise ValueError("Selected ROI produced an empty in-mask region. Please adjust ROI bbox.")

    pred_vel_t = np.stack([u_txyz, v_txyz, w_txyz], axis=1).astype(np.float32)  # [T,3,X,Y,Z]
    pred_4ch_t = np.stack([u_txyz, v_txyz, w_txyz, mag_txyz], axis=1).astype(np.float32)  # [T,4,X,Y,Z]

    suggested_flow_axis, flow_axis_scores = uq._suggest_flow_axis(
        vel_ref=pred_vel_t,
        mask=mask_metrics,
        spacing_mm=spacing_mm,
    )
    selected_flow_axis = int(suggested_flow_axis) if args.flow_axis == "auto" else int(args.flow_axis)

    # 1) Voxel distributions
    fig_voxel_hist = fig_dir / "pred_only_voxel_histogram_in_mask.png"
    voxel_dist_rows = _save_voxel_hist_single(
        out_path=fig_voxel_hist,
        pred_4ch=pred_4ch_t,
        mask=mask_metrics,
        bins=int(args.hist_bins),
        method_label=args.method_label,
    )
    voxel_dist_cols = ["channel", "method", "count", "mean", "std", "median", "p05", "p95", "min", "max"]
    uq._write_csv(metrics_dir / "pred_only_voxel_distribution_stats.csv", voxel_dist_rows, voxel_dist_cols)

    # 2) Table2-like (single method)
    spacing_m = tuple(float(s) / 1000.0 for s in spacing_mm)
    speed_pred = np.sqrt((pred_vel_t**2).sum(axis=1)).astype(np.float32)
    vort_pred = np.stack([uq._vorticity_magnitude(pred_vel_t[t], spacing_m) for t in range(t_count)], axis=0)

    table2_all, valid_slices = _table2_rows_single(
        speed=speed_pred,
        vort=vort_pred,
        mask_ref=mask_metrics,
        flow_axis=selected_flow_axis,
        min_voxels=int(args.mask_min_slice_voxels),
    )
    table2_per_frame = _table2_rows_single_per_frame(
        speed=speed_pred,
        vort=vort_pred,
        mask_ref=mask_metrics,
        flow_axis=selected_flow_axis,
        min_voxels=int(args.mask_min_slice_voxels),
        frame_source_indices=frame_source_indices,
    )

    t2_cols = ["slice_index", "variable", "pred"]
    uq._write_csv(metrics_dir / "pred_only_table2_all_slices.csv", table2_all, t2_cols)

    t2pf_cols = ["frame_payload_index", "frame_source_index", "slice_index", "variable", "pred"]
    uq._write_csv(metrics_dir / "pred_only_table2_per_frame_all_slices.csv", table2_per_frame, t2pf_cols)

    # Compact representative slices
    pick_slices, pick_labels = uq._default_slice_triplet(valid_slices)
    label_map = {s: lab for s, lab in zip(pick_slices, pick_labels)}
    table2_compact = []
    for row in table2_all:
        s = int(row["slice_index"])
        if s in label_map:
            rr = dict(row)
            rr["location"] = label_map[s]
            table2_compact.append(rr)
    uq._write_csv(
        metrics_dir / "pred_only_table2_compact.csv",
        table2_compact,
        ["location", "slice_index", "variable", "pred"],
    )

    # 3) Flow metrics
    q_curves = uq._flow_rate_curves(pred_vel_t, mask_metrics, flow_axis=selected_flow_axis, spacing_mm=spacing_mm)
    q_mean = q_curves.mean(axis=0)
    q_sd = q_curves.std(axis=0, ddof=1) if q_curves.shape[0] > 1 else np.zeros_like(q_mean)

    if np.isfinite(float(args.q_ref)):
        q_ref_scalar = float(args.q_ref)
    else:
        q_ref_scalar = float(np.median(q_mean[np.isfinite(q_mean)]))

    flow_rows = [
        {
            "method": args.method_label,
            "mean_Q_ml_s": float(np.mean(q_mean)),
            "MAD_Q_ml_s": float(np.mean(np.abs(q_mean - q_ref_scalar))),
            "MAD_Q_pct_ref": 100.0 * float(np.mean(np.abs(q_mean - q_ref_scalar))) / (abs(q_ref_scalar) + 1e-12),
            "mean_SD_Q_ml_s": float(np.mean(q_sd)),
            "mean_SD_Q_pct_ref": 100.0 * float(np.mean(q_sd)) / (abs(q_ref_scalar) + 1e-12),
        }
    ]
    flow_cols = ["method", "mean_Q_ml_s", "MAD_Q_ml_s", "MAD_Q_pct_ref", "mean_SD_Q_ml_s", "mean_SD_Q_pct_ref"]
    uq._write_csv(metrics_dir / "pred_only_flow_metrics.csv", flow_rows, flow_cols)

    flow_slice_rows = _flow_rows_per_frame_slice_single(
        q_curves=q_curves,
        q_ref_scalar=q_ref_scalar,
        frame_source_indices=frame_source_indices,
    )
    uq._write_csv(
        metrics_dir / "pred_only_flow_rate_curves_per_frame.csv",
        flow_slice_rows,
        ["frame_payload_index", "frame_source_index", "slice_index", "q_pred_ml_s", "abs_err_pred_vs_qref_ml_s"],
    )

    flow_per_frame_rows = _flow_summary_rows_per_frame_single(
        q_curves=q_curves,
        q_ref_scalar=q_ref_scalar,
        frame_source_indices=frame_source_indices,
        method_label=args.method_label,
    )
    uq._write_csv(
        metrics_dir / "pred_only_flow_metrics_per_frame.csv",
        flow_per_frame_rows,
        ["frame_payload_index", "frame_source_index", "method", "mean_Q_ml_s", "MAD_Q_vs_qref_ml_s"],
    )

    fig_flow = fig_dir / "pred_only_flow_rate_profile.png"
    x = np.arange(q_mean.shape[0])
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, q_mean, label=args.method_label, linewidth=2)
    plt.fill_between(x, q_mean - q_sd, q_mean + q_sd, alpha=0.2)
    plt.axhline(q_ref_scalar, linestyle="--", color="black", label=f"Qref={q_ref_scalar:.3f} ml/s")
    plt.xlabel(f"Slice index along axis {selected_flow_axis}")
    plt.ylabel("Flow rate [ml/s]")
    plt.title("Prediction-only flow-rate profile (mean ± SD across frames)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_flow, dpi=180)
    plt.close(fig)

    # 4) WSS metrics (single method)
    tau_parts: List[np.ndarray] = []
    table3_per_frame_rows: List[Dict[str, Any]] = []
    wss_metric_names = ["Maximum", "Mean", "SD", "Quantile_97_5", "Median", "Quantile_2_5", "IQR_75_25"]

    for t in range(t_count):
        mask_t = (mask_metrics[t] > 0.5).astype(np.float32)
        if int(mask_t.sum()) < 25:
            continue

        tau_t = uq._compute_wss_distribution(
            uvw_mean=pred_vel_t[t],
            mask_ref=mask_t,
            spacing_mm=spacing_mm,
            mu_pa_s=float(args.mu_pa_s),
            max_points=int(args.max_wall_points),
            seed=7 + t,
        )
        if tau_t.size > 0:
            tau_parts.append(tau_t)
            wss_t = uq._wss_summary(tau_t)
            for key in wss_metric_names:
                table3_per_frame_rows.append(
                    {
                        "frame_payload_index": int(t),
                        "frame_source_index": int(frame_source_indices[t]),
                        "metric": key,
                        "pred": wss_t[key],
                        "n_pred": int(tau_t.size),
                    }
                )

    tau_pred = np.concatenate(tau_parts, axis=0) if tau_parts else np.zeros((0,), dtype=np.float64)
    wss_pred = uq._wss_summary(tau_pred)

    table3_rows = []
    for key in wss_metric_names:
        table3_rows.append({"metric": key, "pred": wss_pred[key]})

    uq._write_csv(metrics_dir / "pred_only_table3_wss.csv", table3_rows, ["metric", "pred"])
    uq._write_csv(
        metrics_dir / "pred_only_table3_wss_per_frame.csv",
        table3_per_frame_rows,
        ["frame_payload_index", "frame_source_index", "metric", "pred", "n_pred"],
    )

    fig_wss = fig_dir / "pred_only_wss_distribution.png"
    fig = plt.figure(figsize=(9, 5))
    if tau_pred.size > 0:
        plt.hist(tau_pred, bins=80, alpha=0.45, density=True, label=args.method_label)
    plt.xlabel("WSS [Pa]")
    plt.ylabel("Density")
    plt.title("Prediction-only wall shear stress distribution")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_wss, dpi=180)
    plt.close(fig)

    # 5) Geometry temporal uncertainty from prediction mask
    geom_rows: List[Dict[str, Any]] = []
    geom_status = "not_available_single_frame"
    geom_note = "Temporal geometry metrics require at least two frames; marked as N/A."

    if mask_metrics.shape[0] > 1:
        mask_u8 = (mask_metrics > 0.5).astype(np.uint8)
        ref_mask = mask_u8[0]
        static_mask = bool(np.all(mask_u8 == ref_mask[None, ...]))

        if static_mask:
            geom_status = "not_applicable_static_mask"
            geom_note = "Temporal geometry metrics marked as N/A because all frames share the same binary mask."
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
                m = uq._surface_distance_metrics(m_t, ref_mask, spacing_mm)
                m["frame"] = int(t)
                m["frame_source_index"] = int(frame_source_indices[t])
                m["status"] = geom_status
                per_frame.append(m)

            geom_rows.extend(per_frame)
            msd = np.asarray([x["mean_surface_distance_a_to_b_mm"] for x in per_frame], dtype=np.float64)
            hd = np.asarray([x["hausdorff_distance_mm"] for x in per_frame], dtype=np.float64)
            geom_summary = {
                "mean_surface_distance_mm": float(np.nanmean(msd)),
                "std_surface_distance_mm": float(np.nanstd(msd, ddof=1)) if np.isfinite(msd).sum() > 1 else float("nan"),
                "mean_hausdorff_distance_mm": float(np.nanmean(hd)),
            }
    else:
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

    uq._write_csv(
        metrics_dir / "pred_only_geometry_temporal_surface_metrics.csv",
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

    # Summary + HTML
    summary = {
        "report_title": args.report_title,
        "mode": "prediction_only",
        "label": args.method_label,
        "inputs": {
            "u_path": str(Path(args.u_path).resolve()),
            "v_path": str(Path(args.v_path).resolve()),
            "w_path": str(Path(args.w_path).resolve()),
            "mag_path": str(Path(args.mag_path).resolve()) if args.mag_path else "",
            "mask_path": str(Path(args.mask_path).resolve()) if args.mask_path else "",
            "mag_source": mag_source,
        },
        "dimensions": {
            "T": int(t_count),
            "shape_XYZ": [int(v) for v in shape_xyz],
            "spacing_mm": [float(v) for v in spacing_mm],
            "flow_axis": int(selected_flow_axis),
            "flow_axis_mode": str(args.flow_axis),
            "suggested_flow_axis": int(suggested_flow_axis),
            "frame_source_indices": [int(x) for x in frame_source_indices.tolist()],
        },
        "statistics": {
            "flow_reference_q_ml_s": float(q_ref_scalar),
            "flow_axis_scores": flow_axis_scores,
            "wss_summary": wss_pred,
            "geometry_status": geom_status,
            "geometry_note": geom_note,
            "geometry_summary": geom_summary,
            "roi": roi_info,
        },
    }
    (metrics_dir / "pred_only_summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt_rows(rows: List[Dict[str, Any]], keys: Sequence[str], nd: int = 6) -> List[Dict[str, Any]]:
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

    t2_comp_cols = ["location", "slice_index", "variable", "pred"]
    t2_comp_html = uq._html_table(fmt_rows(table2_compact, t2_comp_cols), t2_comp_cols)

    t3_cols = ["metric", "pred"]
    t3_html = uq._html_table(fmt_rows(table3_rows, t3_cols), t3_cols)

    flow_html = uq._html_table(fmt_rows(flow_rows, flow_cols), flow_cols)
    voxel_dist_html = uq._html_table(fmt_rows(voxel_dist_rows, voxel_dist_cols), voxel_dist_cols)

    geom_html = uq._html_table(
        [
            {
                "mean_surface_distance_mm": "nan" if not np.isfinite(geom_summary["mean_surface_distance_mm"]) else f"{geom_summary['mean_surface_distance_mm']:.6f}",
                "std_surface_distance_mm": "nan" if not np.isfinite(geom_summary["std_surface_distance_mm"]) else f"{geom_summary['std_surface_distance_mm']:.6f}",
                "mean_hausdorff_distance_mm": "nan" if not np.isfinite(geom_summary["mean_hausdorff_distance_mm"]) else f"{geom_summary['mean_hausdorff_distance_mm']:.6f}",
            }
        ],
        ["mean_surface_distance_mm", "std_surface_distance_mm", "mean_hausdorff_distance_mm"],
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
    .pill {{ display: inline-block; background: #eef2ff; color: #3730a3; padding: 2px 8px; border-radius: 999px; margin-right: 8px; }}
  </style>
</head>
<body>
  <h1>{args.report_title}</h1>
  <p>
    <span class=\"pill\">Mode: prediction-only</span>
    <span class=\"pill\">Method: {args.method_label}</span>
  </p>
  <ul>
    <li>Flow axis used: <b>{selected_flow_axis}</b> (mode: {args.flow_axis}, suggested: {suggested_flow_axis})</li>
    <li>Qref used (ml/s): <b>{q_ref_scalar:.6f}</b></li>
    <li>Magnitude source: <b>{mag_source}</b></li>
    <li>ROI mode: <b>{'enabled' if roi_info.get('enabled', False) else 'disabled'}</b>{'' if not roi_info.get('enabled', False) else f" (bbox xyz: {roi_info.get('bbox_xyz')})"}</li>
  </ul>

  <h2>Voxel Distribution Inside Mask</h2>
  <img src=\"figures/{fig_voxel_hist.name}\" alt=\"Prediction-only in-mask voxel histograms\"/>
  {voxel_dist_html}

  <h2>Flow-rate Diagnostics</h2>
  <img src=\"figures/{fig_flow.name}\" alt=\"Prediction-only flow profile\"/>
  {flow_html}

  <h2>Table 2-like (Prediction-only)</h2>
  <p class=\"muted\">Mean/SD/skewness/kurtosis of intraluminal velocity and vorticity.</p>
  {t2_comp_html}

  <h2>Table 3-like WSS (Prediction-only)</h2>
  {t3_html}
  <img src=\"figures/{fig_wss.name}\" alt=\"Prediction-only WSS distribution\"/>

  <h2>Geometry Temporal Summary</h2>
  <p class=\"muted\">{geom_note}</p>
  {geom_html}

  <h2>Saved Artifacts</h2>
  <ul>
    <li><code>metrics/pred_only_table2_all_slices.csv</code></li>
    <li><code>metrics/pred_only_table2_per_frame_all_slices.csv</code></li>
    <li><code>metrics/pred_only_table2_compact.csv</code></li>
    <li><code>metrics/pred_only_flow_metrics.csv</code></li>
    <li><code>metrics/pred_only_flow_metrics_per_frame.csv</code></li>
    <li><code>metrics/pred_only_flow_rate_curves_per_frame.csv</code></li>
    <li><code>metrics/pred_only_table3_wss.csv</code></li>
    <li><code>metrics/pred_only_table3_wss_per_frame.csv</code></li>
    <li><code>metrics/pred_only_geometry_temporal_surface_metrics.csv</code></li>
    <li><code>metrics/pred_only_voxel_distribution_stats.csv</code></li>
    <li><code>metrics/pred_only_summary_metrics.json</code></li>
  </ul>
</body>
</html>
"""

    report_path = out_dir / "pred_only_report.html"
    report_path.write_text(html, encoding="utf-8")

    print("Prediction-only report generated:")
    print(f"- HTML: {report_path}")
    print(f"- Figures: {fig_dir}")
    print(f"- Metrics: {metrics_dir}")


if __name__ == "__main__":
    main()
