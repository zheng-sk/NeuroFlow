#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute velocity metrics from analysis_payload.npz produced by "
            "run_sr_inference_case.py. Uses pred_norm and gt_norm so prediction "
            "and reference are compared in the same normalized range."
        )
    )
    p.add_argument("--payload-npz", required=True, help="Path to analysis_payload.npz")
    p.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory for CSV/JSON metrics. Default: <payload_parent>/metrics/velocity"
        ),
    )
    p.add_argument(
        "--non-fluid-loss-weight",
        type=float,
        default=0.3,
        help="Weight used to reproduce training-style weighted velocity loss.",
    )
    p.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize payload mask.",
    )
    return p.parse_args()


def _safe_region_mean(vol_txyz: np.ndarray, weight_txyz: np.ndarray) -> np.ndarray:
    num = (vol_txyz * weight_txyz).sum(axis=(1, 2, 3))
    den = np.maximum(weight_txyz.sum(axis=(1, 2, 3)), 1.0)
    return (num / den).astype(np.float64)


def _component_metrics(
    sq_err_txyz: np.ndarray,
    mask_txyz: np.ndarray,
    outside_txyz: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "masked": _safe_region_mean(sq_err_txyz, mask_txyz),
        "background": _safe_region_mean(sq_err_txyz, outside_txyz),
        "global": sq_err_txyz.mean(axis=(1, 2, 3)).astype(np.float64),
    }


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _align_spatial_shapes(
    pred_tcxyz: np.ndarray,
    gt_tcxyz: np.ndarray,
    mask_txyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pred_xyz = tuple(int(v) for v in pred_tcxyz.shape[2:5])
    gt_xyz = tuple(int(v) for v in gt_tcxyz.shape[2:5])
    mask_xyz = tuple(int(v) for v in mask_txyz.shape[1:4])

    target_xyz = (
        min(pred_xyz[0], gt_xyz[0], mask_xyz[0]),
        min(pred_xyz[1], gt_xyz[1], mask_xyz[1]),
        min(pred_xyz[2], gt_xyz[2], mask_xyz[2]),
    )
    if min(target_xyz) <= 0:
        raise ValueError(
            f"Invalid spatial shapes after alignment request: pred={pred_xyz}, gt={gt_xyz}, mask={mask_xyz}"
        )

    changed = (pred_xyz != target_xyz) or (gt_xyz != target_xyz) or (mask_xyz != target_xyz)
    if changed:
        pred_tcxyz = pred_tcxyz[:, :, : target_xyz[0], : target_xyz[1], : target_xyz[2]]
        gt_tcxyz = gt_tcxyz[:, :, : target_xyz[0], : target_xyz[1], : target_xyz[2]]
        mask_txyz = mask_txyz[:, : target_xyz[0], : target_xyz[1], : target_xyz[2]]

    info = {
        "pred_xyz_original": list(pred_xyz),
        "gt_xyz_original": list(gt_xyz),
        "mask_xyz_original": list(mask_xyz),
        "aligned_xyz": list(target_xyz),
        "front_cropped": bool(changed),
    }
    return pred_tcxyz, gt_tcxyz, mask_txyz, info


def main() -> None:
    args = _parse_args()

    payload_path = Path(args.payload_npz).resolve()
    if not payload_path.is_file():
        raise FileNotFoundError(f"Payload not found: {payload_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (payload_path.parent / "metrics" / "velocity")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(payload_path)
    pred = np.asarray(d["pred_norm"][:, :3], dtype=np.float64)  # [T, 3, X, Y, Z]
    gt = np.asarray(d["gt_norm"][:, :3], dtype=np.float64)      # [T, 3, X, Y, Z]
    mask = (np.asarray(d["mask"], dtype=np.float64) >= float(args.mask_threshold)).astype(np.float64)  # [T, X, Y, Z]
    frame_indices = np.asarray(d["frame_indices"], dtype=np.int32)
    venc = np.asarray(d["venc"], dtype=np.float64)

    if pred.shape[0] != mask.shape[0]:
        raise ValueError(f"Time dimension mismatch: pred T={pred.shape[0]} vs mask T={mask.shape[0]}")

    pred, gt, mask, align_info = _align_spatial_shapes(pred, gt, mask)

    outside = 1.0 - mask
    sq_err = (pred - gt) ** 2
    vel_sq_err = sq_err.sum(axis=1)  # [T, X, Y, Z], same training-style sum over u,v,w

    vel_masked = _safe_region_mean(vel_sq_err, mask)
    vel_background = _safe_region_mean(vel_sq_err, outside)
    vel_global = vel_sq_err.mean(axis=(1, 2, 3)).astype(np.float64)
    vel_weighted = vel_masked + (float(args.non_fluid_loss_weight) * vel_background)

    u_metrics = _component_metrics(sq_err[:, 0], mask, outside)
    v_metrics = _component_metrics(sq_err[:, 1], mask, outside)
    w_metrics = _component_metrics(sq_err[:, 2], mask, outside)

    per_frame_rows: list[dict[str, Any]] = []
    for i in range(pred.shape[0]):
        per_frame_rows.append(
            {
                "frame_idx": int(i),
                "frame_source_index": int(frame_indices[i]) if i < frame_indices.size else int(i),
                "venc": float(venc[i]) if i < venc.size else float("nan"),
                "mask_voxels": int(mask[i].sum()),
                "background_voxels": int(outside[i].sum()),
                "velocity_mse_masked": float(vel_masked[i]),
                "velocity_mse_background": float(vel_background[i]),
                "velocity_mse_global": float(vel_global[i]),
                "velocity_loss_weighted": float(vel_weighted[i]),
                "u_mse_masked": float(u_metrics["masked"][i]),
                "u_mse_background": float(u_metrics["background"][i]),
                "u_mse_global": float(u_metrics["global"][i]),
                "v_mse_masked": float(v_metrics["masked"][i]),
                "v_mse_background": float(v_metrics["background"][i]),
                "v_mse_global": float(v_metrics["global"][i]),
                "w_mse_masked": float(w_metrics["masked"][i]),
                "w_mse_background": float(w_metrics["background"][i]),
                "w_mse_global": float(w_metrics["global"][i]),
            }
        )

    summary = {
        "payload_path": str(payload_path),
        "n_frames": int(pred.shape[0]),
        "mask_threshold": float(args.mask_threshold),
        "non_fluid_loss_weight": float(args.non_fluid_loss_weight),
        "range_note": (
            "Metrics are computed from pred_norm and gt_norm in analysis_payload.npz, "
            "so prediction and reference are compared in the same venc-normalized range."
        ),
        "spatial_alignment": align_info,
        "velocity_mse_masked": _summary_stats(vel_masked),
        "velocity_mse_background": _summary_stats(vel_background),
        "velocity_mse_global": _summary_stats(vel_global),
        "velocity_loss_weighted": _summary_stats(vel_weighted),
        "u_mse_masked": _summary_stats(u_metrics["masked"]),
        "u_mse_background": _summary_stats(u_metrics["background"]),
        "u_mse_global": _summary_stats(u_metrics["global"]),
        "v_mse_masked": _summary_stats(v_metrics["masked"]),
        "v_mse_background": _summary_stats(v_metrics["background"]),
        "v_mse_global": _summary_stats(v_metrics["global"]),
        "w_mse_masked": _summary_stats(w_metrics["masked"]),
        "w_mse_background": _summary_stats(w_metrics["background"]),
        "w_mse_global": _summary_stats(w_metrics["global"]),
    }

    per_frame_csv = out_dir / "velocity_metrics_per_frame.csv"
    with per_frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    summary_json = out_dir / "velocity_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    compact_csv = out_dir / "velocity_metrics_compact.csv"
    compact_rows = [
        {"metric": "velocity_mse_masked_mean", "value": summary["velocity_mse_masked"]["mean"]},
        {"metric": "velocity_mse_background_mean", "value": summary["velocity_mse_background"]["mean"]},
        {"metric": "velocity_mse_global_mean", "value": summary["velocity_mse_global"]["mean"]},
        {"metric": "velocity_loss_weighted_mean", "value": summary["velocity_loss_weighted"]["mean"]},
        {"metric": "u_mse_masked_mean", "value": summary["u_mse_masked"]["mean"]},
        {"metric": "v_mse_masked_mean", "value": summary["v_mse_masked"]["mean"]},
        {"metric": "w_mse_masked_mean", "value": summary["w_mse_masked"]["mean"]},
        {"metric": "u_mse_background_mean", "value": summary["u_mse_background"]["mean"]},
        {"metric": "v_mse_background_mean", "value": summary["v_mse_background"]["mean"]},
        {"metric": "w_mse_background_mean", "value": summary["w_mse_background"]["mean"]},
    ]
    with compact_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(compact_rows)

    print(f"Saved per-frame metrics: {per_frame_csv}")
    print(f"Saved compact metrics:   {compact_csv}")
    print(f"Saved summary metrics:   {summary_json}")
    if align_info["front_cropped"]:
        print(
            "Warning: pred/gt/mask spatial shapes differed and were aligned by front-cropping "
            f"to {tuple(align_info['aligned_xyz'])} from "
            f"pred={tuple(align_info['pred_xyz_original'])}, "
            f"gt={tuple(align_info['gt_xyz_original'])}, "
            f"mask={tuple(align_info['mask_xyz_original'])}."
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
