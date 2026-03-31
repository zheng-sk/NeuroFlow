#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sr_uq_common import (
    _align_mask_to_target,
    _predict_with_sliding_window,
    _prepare_frame,
    _read_case_volumes,
    choose_frame_indices,
    load_case_table,
    load_sr_model,
    save_metadata_json,
    save_payload_npz,
    save_predicted_nifti,
)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def _parse_frame_indices(values: Optional[List[int]]) -> Optional[List[int]]:
    if not values:
        return None
    return [int(v) for v in values]


def _load_mask_3d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D/4D mask, got shape {data.shape} for {path}")
    return (data >= 0.5).astype(np.float32)


def _segment_stage1(stage1_dir: Path, seg_model_dir: Path, env: dict[str, str] | None) -> Path:
    seg_tmp_root = stage1_dir / "_segmentation_tmp"
    if seg_tmp_root.exists():
        shutil.rmtree(seg_tmp_root)

    seg_script = THIS_DIR.parent / "segmentation" / "segment_cow_patient_pipeline.py"
    _run(
        [
            sys.executable,
            str(seg_script),
            "--patient-dir",
            str(stage1_dir),
            "--output-dir",
            str(seg_tmp_root),
            "--model-dir",
            str(seg_model_dir),
            "--mag-name",
            "nifti/pred_mag.nii.gz",
            "--vx-name",
            "nifti/pred_u.nii.gz",
            "--vy-name",
            "nifti/pred_v.nii.gz",
            "--vz-name",
            "nifti/pred_w.nii.gz",
            "--venc",
            "0.90",
            "--time-axis",
            "-1",
            "--angio-mode",
            "mag_only",
            "--mag-projection-method",
            "percentile",
            "--mag-projection-percentile",
            "100",
            "--ensemble-mode",
            "union",
            "--classic-percentile",
            "97.5",
            "--post-min-component-size",
            "30",
            "--classic-sigmas",
            "0.8,1.2,1.6,2.0",
            "--save-intermediates",
        ],
        env=env,
    )
    src_mask = seg_tmp_root / stage1_dir.name / "cow_seg_final.nii.gz"
    if not src_mask.exists():
        raise FileNotFoundError(f"Missing stage-1 segmented mask: {src_mask}")
    dst_mask = stage1_dir / "cow_seg_final.nii.gz"
    dst_mask.write_bytes(src_mask.read_bytes())
    return dst_mask


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Cascade inference with stage-1 auto-segmentation mask: "
            "run stage1 SR, segment CoW on stage1 output, mask stage1 prediction, "
            "then run stage2 denoising from the same cascade checkpoint."
        )
    )
    p.add_argument("--case-csv", required=True)
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seg-model-dir", default=str(THIS_DIR.parents[2] / "models" / "topcow-claim-models"))
    p.add_argument("--output-prefix", default="pred")
    p.add_argument("--time-axis", type=int, default=-1)
    p.add_argument("--frame-index", type=int, nargs="*", default=None)
    p.add_argument("--use-csv-frame-selection", action="store_true")
    p.add_argument("--stage1-patch-size", type=int, default=24)
    p.add_argument("--stage1-sw-batch-size", type=int, default=1)
    p.add_argument("--stage2-patch-size", type=int, default=48)
    p.add_argument("--stage2-sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--res-increase", type=int, default=2)
    p.add_argument("--low-resblock", type=int, default=8)
    p.add_argument("--hi-resblock", type=int, default=4)
    p.add_argument("--predict-mag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--raw-phase-input", dest="lr_raw_phase_input", action="store_true", default=True)
    p.add_argument("--already-velocity-input", dest="lr_raw_phase_input", action="store_false")
    p.add_argument("--hr-raw-phase-target", dest="hr_raw_phase_input", action="store_true", default=None)
    p.add_argument("--hr-already-velocity-target", dest="hr_raw_phase_input", action="store_false")
    p.add_argument("--legacy-invert-uv-sign-on-raw", action="store_true")
    p.add_argument("--raw-center", type=float, default=2048.0)
    p.add_argument("--raw-scale", type=float, default=2048.0)
    p.add_argument("--mag-scale", type=float, default=4095.0)
    p.add_argument("--mag-norm-mode", type=str, default="monai_minmax", choices=["monai_minmax", "divisor"])
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--gpu", default="", help="Optional CUDA_VISIBLE_DEVICES override for segmentation only.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = out_dir / "stage1_superresolution"
    stage1_nifti_dir = stage1_dir / "nifti"
    stage1_nifti_dir.mkdir(parents=True, exist_ok=True)
    final_nifti_dir = out_dir / "nifti"
    final_nifti_dir.mkdir(parents=True, exist_ok=True)

    cases = load_case_table(args.case_csv, include_hr_mag=True)
    if not cases:
        raise RuntimeError(f"No rows found in CSV: {args.case_csv}")
    case_idx = int(args.case_index) % len(cases)
    case = cases[case_idx]

    preview = _read_case_volumes(case=case, time_axis=int(args.time_axis))
    explicit = _parse_frame_indices(args.frame_index)
    if explicit:
        frame_indices = explicit
        frame_mode = "explicit"
    elif args.use_csv_frame_selection:
        frame_indices = choose_frame_indices(case=case, t_count=int(preview["t_count"]), explicit_indices=None)
        frame_mode = "csv"
    else:
        frame_indices = list(range(int(preview["t_count"])))
        frame_mode = "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, predict_mag_flag = load_sr_model(
        model_path=args.model_path,
        res_increase=int(args.res_increase),
        low_resblock=int(args.low_resblock),
        hi_resblock=int(args.hi_resblock),
        device=device,
        predict_mag=args.predict_mag,
        model_variant=None,
    )
    if getattr(model, "model_variant", "") != "cascade_sr_dn_masked":
        raise ValueError(
            f"Expected cascade_sr_dn_masked checkpoint, got model_variant={getattr(model, 'model_variant', '')!r}"
        )
    if not hasattr(model, "stage1") or not hasattr(model, "stage2"):
        raise ValueError("Loaded cascade checkpoint does not expose stage1/stage2 modules.")
    if not predict_mag_flag:
        raise ValueError("Cascade workflow expects predict_mag=True.")

    hr_raw_phase_input = bool(args.lr_raw_phase_input) if args.hr_raw_phase_input is None else bool(args.hr_raw_phase_input)

    print(f"Device: {device}")
    print(f"Model variant: {getattr(model, 'model_variant', 'unknown')}")
    print(f"Selected case: {case_idx}/{len(cases)-1}")
    print(f"Frame selection mode: {frame_mode}")
    print(f"Frames: {frame_indices}")

    volumes = _read_case_volumes(case=case, time_axis=int(args.time_axis))
    lr_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []
    stage1_pred_all: list[np.ndarray] = []
    stage1_seg_all: list[np.ndarray] = []
    ref_mask_all: list[np.ndarray] = []
    venc_all: list[float] = []

    for i, frame_idx in enumerate(frame_indices):
        lr_input_norm, gt_4ch_norm, mask_bin, venc_scalar = _prepare_frame(
            case=case,
            volumes=volumes,
            frame_idx=frame_idx,
            lr_raw_phase_input=bool(args.lr_raw_phase_input),
            hr_raw_phase_input=bool(hr_raw_phase_input),
            legacy_invert_uv_sign_on_raw=bool(args.legacy_invert_uv_sign_on_raw),
            raw_center=float(args.raw_center),
            raw_scale=float(args.raw_scale),
            mag_scale=float(args.mag_scale),
            mag_norm_mode=str(args.mag_norm_mode),
            mask_threshold=float(args.mask_threshold),
            apply_mask_to_lr_inputs=False,
            apply_mask_to_lr_magnitude=True,
        )
        stage1_pred, stage1_seg = _predict_with_sliding_window(
            model=model.stage1,
            lr_input_norm=lr_input_norm,
            mask_hr=None,
            roi_size=(int(args.stage1_patch_size), int(args.stage1_patch_size), int(args.stage1_patch_size)),
            sw_batch_size=int(args.stage1_sw_batch_size),
            overlap=float(args.overlap),
            device=device,
        )
        if stage1_pred.shape[0] != 4:
            raise ValueError(f"Expected 4 output channels from stage1, got shape {stage1_pred.shape}")

        lr_all.append(lr_input_norm)
        gt_all.append(gt_4ch_norm)
        stage1_pred_all.append(stage1_pred.astype(np.float32))
        if stage1_seg is not None:
            stage1_seg_all.append(stage1_seg.astype(np.float32))
        ref_mask_all.append(mask_bin.astype(np.float32))
        venc_all.append(float(venc_scalar))
        print(f"Stage 1 processed frame {i + 1}/{len(frame_indices)} (t={frame_idx})")

    stage1_payload = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "lr_norm": np.stack(lr_all, axis=0).astype(np.float32),
        "gt_norm": np.stack(gt_all, axis=0).astype(np.float32),
        "pred_norm": np.stack(stage1_pred_all, axis=0).astype(np.float32),
        "mask": np.stack(ref_mask_all, axis=0).astype(np.float32),
        "venc": np.asarray(venc_all, dtype=np.float32),
        "lr_affine": np.asarray(preview["lr_img"].affine, dtype=np.float32),
        "hr_affine": np.asarray(preview["hr_img"].affine, dtype=np.float32),
        "lr_spacing": np.asarray(preview["lr_img"].header.get_zooms()[:3], dtype=np.float32),
        "hr_spacing": np.asarray(preview["hr_img"].header.get_zooms()[:3], dtype=np.float32),
        "t_count": int(preview["t_count"]),
    }
    if stage1_seg_all:
        stage1_payload["seg_pred"] = np.stack(stage1_seg_all, axis=0).astype(np.float32)
    stage1_nifti_paths = save_predicted_nifti(
        pred_norm=stage1_payload["pred_norm"],
        venc_per_frame=stage1_payload["venc"],
        mag_scale=float(args.mag_scale),
        mag_norm_mode=str(args.mag_norm_mode),
        out_prefix=str((stage1_nifti_dir / args.output_prefix).resolve()),
        lr_affine=stage1_payload["lr_affine"],
        res_increase=int(args.res_increase),
        save_uvw=True,
    )
    save_payload_npz(str((stage1_dir / "analysis_payload.npz").resolve()), stage1_payload)

    seg_env = os.environ.copy()
    if str(args.gpu).strip():
        seg_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    stage1_mask_path = _segment_stage1(stage1_dir, Path(args.seg_model_dir).resolve(), seg_env)
    stage1_mask = _load_mask_3d(stage1_mask_path)

    final_pred_all: list[np.ndarray] = []
    stage1_mask_all: list[np.ndarray] = []
    for i, stage1_pred in enumerate(stage1_pred_all):
        mask_stage1_aligned = _align_mask_to_target(stage1_mask, tuple(int(v) for v in stage1_pred.shape[1:4]))
        sr_u = stage1_pred[0:1] * mask_stage1_aligned[None, ...]
        sr_v = stage1_pred[1:2] * mask_stage1_aligned[None, ...]
        sr_w = stage1_pred[2:3] * mask_stage1_aligned[None, ...]
        sr_mag = np.clip(stage1_pred[3:4], 0.0, 1.0) * mask_stage1_aligned[None, ...]
        stage2_input = np.concatenate([sr_u, sr_v, sr_w, sr_mag, sr_mag, sr_mag], axis=0).astype(np.float32)

        final_pred, _stage2_seg = _predict_with_sliding_window(
            model=model.stage2,
            lr_input_norm=stage2_input,
            mask_hr=None,
            roi_size=(int(args.stage2_patch_size), int(args.stage2_patch_size), int(args.stage2_patch_size)),
            sw_batch_size=int(args.stage2_sw_batch_size),
            overlap=float(args.overlap),
            device=device,
        )
        if final_pred.shape[0] != 4:
            raise ValueError(f"Expected 4 output channels from stage2, got shape {final_pred.shape}")
        final_pred_all.append(final_pred.astype(np.float32))
        stage1_mask_all.append(mask_stage1_aligned.astype(np.float32))
        print(f"Stage 2 processed frame {i + 1}/{len(stage1_pred_all)}")

    final_payload = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "lr_norm": np.stack(stage1_pred_all, axis=0).astype(np.float32),
        "gt_norm": np.stack(gt_all, axis=0).astype(np.float32),
        "pred_norm": np.stack(final_pred_all, axis=0).astype(np.float32),
        # Keep original 7T mask for final metric evaluation consistency.
        "mask": np.stack(ref_mask_all, axis=0).astype(np.float32),
        "venc": np.asarray(venc_all, dtype=np.float32),
        "lr_affine": np.asarray(preview["hr_img"].affine, dtype=np.float32),
        "hr_affine": np.asarray(preview["hr_img"].affine, dtype=np.float32),
        "lr_spacing": np.asarray(preview["hr_img"].header.get_zooms()[:3], dtype=np.float32),
        "hr_spacing": np.asarray(preview["hr_img"].header.get_zooms()[:3], dtype=np.float32),
        "t_count": int(preview["t_count"]),
        "stage1_mask": np.stack(stage1_mask_all, axis=0).astype(np.float32),
    }
    final_nifti_paths = save_predicted_nifti(
        pred_norm=final_payload["pred_norm"],
        venc_per_frame=final_payload["venc"],
        mag_scale=float(args.mag_scale),
        mag_norm_mode=str(args.mag_norm_mode),
        out_prefix=str((final_nifti_dir / args.output_prefix).resolve()),
        lr_affine=final_payload["lr_affine"],
        res_increase=1,
        save_uvw=True,
    )

    payload_path = out_dir / "analysis_payload.npz"
    save_payload_npz(str(payload_path.resolve()), final_payload)
    meta = {
        "case_csv": str(Path(args.case_csv).resolve()),
        "case_index": int(case_idx),
        "frame_indices": [int(x) for x in frame_indices],
        "frame_selection_mode": frame_mode,
        "model_path": str(Path(args.model_path).resolve()),
        "model_variant": "cascade_sr_dn_masked_stage1mask_inference",
        "checkpoint_model_variant": str(getattr(model, "model_variant", "")),
        "predict_mag": bool(predict_mag_flag),
        "lr_raw_phase_input": bool(args.lr_raw_phase_input),
        "hr_raw_phase_input": bool(hr_raw_phase_input),
        "legacy_invert_uv_sign_on_raw": bool(args.legacy_invert_uv_sign_on_raw),
        "raw_center": float(args.raw_center),
        "raw_scale": float(args.raw_scale),
        "mag_scale": float(args.mag_scale),
        "mag_norm_mode": str(args.mag_norm_mode),
        "mask_threshold": float(args.mask_threshold),
        "res_increase": int(args.res_increase),
        "stage1_patch_size": int(args.stage1_patch_size),
        "stage1_sw_batch_size": int(args.stage1_sw_batch_size),
        "stage2_patch_size": int(args.stage2_patch_size),
        "stage2_sw_batch_size": int(args.stage2_sw_batch_size),
        "overlap": float(args.overlap),
        "payload_path": str(payload_path.resolve()),
        "stage1_dir": str(stage1_dir.resolve()),
        "stage1_mask_path": str(stage1_mask_path.resolve()),
        "stage1_nifti_outputs": stage1_nifti_paths,
        "nifti_outputs": final_nifti_paths,
        "case_paths": {
            "lr_u": case.get("lr_u", ""),
            "lr_v": case.get("lr_v", ""),
            "lr_w": case.get("lr_w", ""),
            "lr_mag_u": case.get("lr_mag_u", ""),
            "lr_mag_v": case.get("lr_mag_v", ""),
            "lr_mag_w": case.get("lr_mag_w", ""),
            "hr_u": case.get("hr_u", ""),
            "hr_v": case.get("hr_v", ""),
            "hr_w": case.get("hr_w", ""),
            "hr_mag": case.get("hr_mag", ""),
            "mask": case.get("mask", ""),
        },
    }
    save_metadata_json(str((out_dir / "inference_metadata.json").resolve()), meta)

    print("\nSaved artifacts:")
    print(f"- Final payload: {payload_path}")
    print(f"- Metadata:      {out_dir / 'inference_metadata.json'}")
    print(f"- Stage 1 dir:   {stage1_dir}")
    print(f"- Stage 1 mask:  {stage1_mask_path}")
    for key, val in final_nifti_paths.items():
        print(f"- Final NIfTI ({key}): {val}")


if __name__ == "__main__":
    main()
