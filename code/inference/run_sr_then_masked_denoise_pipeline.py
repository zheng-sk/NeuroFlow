#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def _resolve_model_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        matches = sorted(candidate.glob("*-best.pt"))
        if not matches:
            raise FileNotFoundError(f"No *-best.pt checkpoint found under: {candidate}")
        return matches[0].resolve()

    matches = sorted(root.glob(f"models/{value}_*/{value}-best.pt"))
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve model path from {value!r}. "
            "Pass a .pt path, a model directory, or an experiment prefix."
        )
    return matches[-1].resolve()


def _load_case_row(case_csv: Path, case_index: int) -> Dict[str, Any]:
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from Network.NiftiPatchDataset import load_nifti_case_table

    cases = load_nifti_case_table(str(case_csv), include_hr_mag=True)
    if not cases:
        raise RuntimeError(f"No rows found in CSV: {case_csv}")
    idx = int(case_index) % len(cases)
    return cases[idx]


def _load_mask_3d(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D/4D mask, got shape {data.shape} for {path}")
    return (data >= 0.5).astype(np.float32), img


def _apply_mask_to_nifti(
    input_path: Path,
    mask_3d: np.ndarray,
    out_path: Path,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> None:
    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    if data.ndim == 3:
        if tuple(data.shape) != tuple(mask_3d.shape):
            raise ValueError(f"Shape mismatch: {input_path} has {data.shape}, mask has {mask_3d.shape}")
        masked = data * mask_3d
    elif data.ndim == 4:
        if tuple(data.shape[:3]) != tuple(mask_3d.shape):
            raise ValueError(f"Shape mismatch: {input_path} has {data.shape[:3]}, mask has {mask_3d.shape}")
        masked = data * mask_3d[..., None]
    else:
        raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape} for {input_path}")

    if clip_min is not None or clip_max is not None:
        lo = float(clip_min) if clip_min is not None else None
        hi = float(clip_max) if clip_max is not None else None
        if lo is not None and hi is not None:
            masked = np.clip(masked, lo, hi)
        elif lo is not None:
            masked = np.maximum(masked, lo)
        else:
            masked = np.minimum(masked, hi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(masked.astype(np.float32), img.affine, img.header), str(out_path))


def _write_single_case_csv(case: Dict[str, Any], masked_dir: Path, out_csv: Path) -> None:
    row = {
        "lr_u": str((masked_dir / "pred_u_masked.nii.gz").resolve()),
        "lr_v": str((masked_dir / "pred_v_masked.nii.gz").resolve()),
        "lr_w": str((masked_dir / "pred_w_masked.nii.gz").resolve()),
        "lr_mag_u": str((masked_dir / "pred_mag_masked.nii.gz").resolve()),
        "lr_mag_v": str((masked_dir / "pred_mag_masked.nii.gz").resolve()),
        "lr_mag_w": str((masked_dir / "pred_mag_masked.nii.gz").resolve()),
        "hr_u": str(case["hr_u"]),
        "hr_v": str(case["hr_v"]),
        "hr_w": str(case["hr_w"]),
        "hr_mag": str(case.get("hr_mag", "")),
        # Evaluation mask should remain the original 7T mask; the stage-1 SR mask
        # is only used to build the denoising input files above.
        "mask": str(case.get("mask", "")),
        "venc": float(case.get("venc", 0.0)),
        "venc_u": float(case.get("venc_u", 0.0)),
        "venc_v": float(case.get("venc_v", 0.0)),
        "venc_w": float(case.get("venc_w", 0.0)),
        "time_start": int(case.get("time_start", -1)),
        "time_end": int(case.get("time_end", -1)),
        "time_index": int(case.get("time_index", -1)),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _segment_prediction(patient_dir: Path, out_root: Path, model_dir: Path, env: dict[str, str] | None) -> Path:
    seg_script = REPO_ROOT / "code" / "segmentation" / "segment_cow_patient_pipeline.py"
    _run(
        [
            sys.executable,
            str(seg_script),
            "--patient-dir",
            str(patient_dir),
            "--output-dir",
            str(out_root),
            "--model-dir",
            str(model_dir),
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
    return out_root / patient_dir.name / "cow_seg_final.nii.gz"


def _copy_final_mask(seg_mask: Path, dst: Path) -> None:
    dst.write_bytes(seg_mask.read_bytes())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Two-stage pipeline: 0.53T -> SR -> CoW segmentation -> mask SR outputs -> "
            "masked denoising -> final metrics."
        )
    )
    p.add_argument("--case-csv", required=True, help="Original 0.53T x2 case CSV.")
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--sr-model", required=True, help="SR model prefix, dir, or .pt path.")
    p.add_argument("--dn-model", required=True, help="Denoising model prefix, dir, or .pt path.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seg-model-dir", default=str(REPO_ROOT / "models" / "topcow-claim-models"))
    p.add_argument("--gpu", default="", help="Optional CUDA_VISIBLE_DEVICES override.")
    p.add_argument("--sr-patch-size", type=int, default=24)
    p.add_argument("--sr-sw-batch-size", type=int, default=2)
    p.add_argument("--dn-patch-size", type=int, default=48)
    p.add_argument("--dn-sw-batch-size", type=int, default=2)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = REPO_ROOT
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if str(args.gpu).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    case_csv = Path(args.case_csv).resolve()
    case = _load_case_row(case_csv, int(args.case_index))
    sr_model_path = _resolve_model_path(root, args.sr_model)
    dn_model_path = _resolve_model_path(root, args.dn_model)
    seg_model_dir = Path(args.seg_model_dir).resolve()

    stage1_dir = out_dir / "stage1_superresolution"
    stage2_dir = out_dir / "stage2_denoised"
    masked_inputs_dir = stage1_dir / "masked_inputs" / "nifti"

    infer_script = root / "code" / "inference" / "run_sr_inference_case.py"
    vel_metrics_script = root / "code" / "inference" / "calculate_velocity_metrics.py"
    bg_metrics_script = root / "code" / "inference" / "calculate_background_metrics.py"
    geom_metrics_script = root / "code" / "inference" / "calculate_geometry_metrics.py"

    # Stage 1: SR from original 0.53T case.
    _run(
        [
            sys.executable,
            str(infer_script),
            "--case-csv",
            str(case_csv),
            "--case-index",
            str(int(args.case_index)),
            "--model-path",
            str(sr_model_path),
            "--out-dir",
            str(stage1_dir),
            "--patch-size",
            str(int(args.sr_patch_size)),
            "--sw-batch-size",
            str(int(args.sr_sw_batch_size)),
            "--res-increase",
            "2",
            "--predict-mag",
        ],
        env=env,
    )

    # Stage 1 segmentation for CoW mask.
    seg_tmp_root = stage1_dir / "_segmentation_tmp"
    seg_mask = _segment_prediction(stage1_dir, seg_tmp_root, seg_model_dir, env=env)
    final_stage1_mask = stage1_dir / "cow_seg_final.nii.gz"
    _copy_final_mask(seg_mask, final_stage1_mask)

    # Mask SR outputs to build denoising input.
    mask_3d, _ = _load_mask_3d(final_stage1_mask)
    sr_nifti_dir = stage1_dir / "nifti"
    _apply_mask_to_nifti(sr_nifti_dir / "pred_u.nii.gz", mask_3d, masked_inputs_dir / "pred_u_masked.nii.gz")
    _apply_mask_to_nifti(sr_nifti_dir / "pred_v.nii.gz", mask_3d, masked_inputs_dir / "pred_v_masked.nii.gz")
    _apply_mask_to_nifti(sr_nifti_dir / "pred_w.nii.gz", mask_3d, masked_inputs_dir / "pred_w_masked.nii.gz")
    # Keep SR magnitude scale, but clip slight model overshoots before denoising.
    _apply_mask_to_nifti(
        sr_nifti_dir / "pred_mag.nii.gz",
        mask_3d,
        masked_inputs_dir / "pred_mag_masked.nii.gz",
        clip_min=0.0,
        clip_max=1.0,
    )

    # Build one-row CSV for stage-2 denoising.
    stage2_case_csv = stage1_dir / "denoising_input_case.csv"
    _write_single_case_csv(case, masked_inputs_dir, stage2_case_csv)

    # Stage 2: denoising over masked SR output.
    _run(
        [
            sys.executable,
            str(infer_script),
            "--case-csv",
            str(stage2_case_csv),
            "--case-index",
            "0",
            "--model-path",
            str(dn_model_path),
            "--out-dir",
            str(stage2_dir),
            "--patch-size",
            str(int(args.dn_patch_size)),
            "--sw-batch-size",
            str(int(args.dn_sw_batch_size)),
            "--res-increase",
            "1",
            "--predict-mag",
            "--already-velocity-input",
            "--mag-norm-mode",
            "divisor",
            "--mag-scale",
            "1.0",
            "--mask-threshold",
            str(float(args.mask_threshold)),
        ],
        env=env,
    )

    # Final metrics: velocity + background.
    final_payload = stage2_dir / "analysis_payload.npz"
    _run(
        [
            sys.executable,
            str(vel_metrics_script),
            "--payload-npz",
            str(final_payload),
            "--out-dir",
            str(stage2_dir / "metrics" / "velocity"),
        ]
    )
    _run(
        [
            sys.executable,
            str(bg_metrics_script),
            "--payload-npz",
            str(final_payload),
            "--out-dir",
            str(stage2_dir / "metrics" / "background"),
        ]
    )

    # Final geometry: segment final denoised output and compare to original 7T mask.
    final_seg_tmp_root = stage2_dir / "_segmentation_tmp"
    final_seg_mask = _segment_prediction(stage2_dir, final_seg_tmp_root, seg_model_dir, env=env)
    final_stage2_mask = stage2_dir / "cow_seg_final.nii.gz"
    _copy_final_mask(final_seg_mask, final_stage2_mask)

    _run(
        [
            sys.executable,
            str(geom_metrics_script),
            "--pred-mask",
            str(final_stage2_mask),
            "--ref-mask",
            str(case["mask"]),
            "--out-dir",
            str(stage2_dir / "metrics" / "geometry"),
        ]
    )

    summary = {
        "case_csv": str(case_csv),
        "case_index": int(args.case_index),
        "sr_model_path": str(sr_model_path),
        "dn_model_path": str(dn_model_path),
        "seg_model_dir": str(seg_model_dir),
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage1_mask": str(final_stage1_mask),
        "stage2_mask": str(final_stage2_mask),
        "stage2_case_csv": str(stage2_case_csv),
        "masked_inputs_dir": str(masked_inputs_dir),
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPipeline finished.")
    print(f"- Stage 1 SR:      {stage1_dir}")
    print(f"- Stage 2 denoise: {stage2_dir}")
    print(f"- Summary:         {out_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
