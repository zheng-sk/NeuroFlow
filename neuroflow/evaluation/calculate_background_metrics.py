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
            "Compute background MSE outside the mask from analysis_payload.npz "
            "produced by run_sr_inference_case.py."
        )
    )
    p.add_argument("--payload-npz", required=True, help="Path to analysis_payload.npz")
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for background metrics. Default: <payload_parent>/metrics/background",
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

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (payload_path.parent / "metrics" / "background")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(payload_path)
    pred = np.asarray(d["pred_norm"], dtype=np.float64)  # [T, C, X, Y, Z]
    gt = np.asarray(d["gt_norm"], dtype=np.float64)      # [T, C, X, Y, Z]
    mask = (np.asarray(d["mask"], dtype=np.float64) >= float(args.mask_threshold)).astype(np.float64)  # [T, X, Y, Z]
    frame_indices = np.asarray(d["frame_indices"], dtype=np.int32)
    venc = np.asarray(d["venc"], dtype=np.float64)

    if pred.shape[0] != mask.shape[0]:
        raise ValueError(f"Time dimension mismatch: pred T={pred.shape[0]} vs mask T={mask.shape[0]}")

    pred, gt, mask, align_info = _align_spatial_shapes(pred, gt, mask)
    outside = 1.0 - mask

    sq_err = (pred - gt) ** 2

    vel_sq_err_sum = sq_err[:, :3].sum(axis=1)  # training-style sum over u,v,w outside mask
    velocity_background = _safe_region_mean(vel_sq_err_sum, outside)

    u_background = _safe_region_mean(sq_err[:, 0], outside)
    v_background = _safe_region_mean(sq_err[:, 1], outside)
    w_background = _safe_region_mean(sq_err[:, 2], outside)

    mag_background = None
    if pred.shape[1] >= 4 and gt.shape[1] >= 4:
        mag_background = _safe_region_mean(sq_err[:, 3], outside)

    per_frame_rows: list[dict[str, Any]] = []
    for i in range(pred.shape[0]):
        row = {
            "frame_idx": int(i),
            "frame_source_index": int(frame_indices[i]) if i < frame_indices.size else int(i),
            "venc": float(venc[i]) if i < venc.size else float("nan"),
            "background_voxels": int(outside[i].sum()),
            "background_mse_velocity": float(velocity_background[i]),
            "background_mse_u": float(u_background[i]),
            "background_mse_v": float(v_background[i]),
            "background_mse_w": float(w_background[i]),
        }
        if mag_background is not None:
            row["background_mse_mag"] = float(mag_background[i])
        per_frame_rows.append(row)

    summary = {
        "payload_path": str(payload_path),
        "n_frames": int(pred.shape[0]),
        "mask_threshold": float(args.mask_threshold),
        "range_note": (
            "Background MSE is computed from pred_norm and gt_norm in analysis_payload.npz, "
            "so prediction and reference are compared in the same normalized range."
        ),
        "spatial_alignment": align_info,
        "background_mse_velocity": _summary_stats(velocity_background),
        "background_mse_u": _summary_stats(u_background),
        "background_mse_v": _summary_stats(v_background),
        "background_mse_w": _summary_stats(w_background),
    }
    if mag_background is not None:
        summary["background_mse_mag"] = _summary_stats(mag_background)

    per_frame_csv = out_dir / "background_metrics_per_frame.csv"
    with per_frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    summary_json = out_dir / "background_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    compact_rows = [
        {"metric": "background_mse_velocity_mean", "value": summary["background_mse_velocity"]["mean"]},
        {"metric": "background_mse_u_mean", "value": summary["background_mse_u"]["mean"]},
        {"metric": "background_mse_v_mean", "value": summary["background_mse_v"]["mean"]},
        {"metric": "background_mse_w_mean", "value": summary["background_mse_w"]["mean"]},
    ]
    if mag_background is not None:
        compact_rows.append({"metric": "background_mse_mag_mean", "value": summary["background_mse_mag"]["mean"]})

    compact_csv = out_dir / "background_metrics_compact.csv"
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
