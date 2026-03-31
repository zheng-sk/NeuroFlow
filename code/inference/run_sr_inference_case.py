import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"The cuda\.cudart module is deprecated and will be removed in a future release",
    category=FutureWarning,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sr_uq_common import (
    _read_case_volumes,
    choose_frame_indices,
    load_case_table,
    load_sr_model,
    run_case_inference,
    save_segmentation_nifti,
    save_metadata_json,
    save_payload_npz,
    save_predicted_nifti,
)


def _parse_frame_indices(values: Optional[List[int]]) -> Optional[List[int]]:
    if not values:
        return None
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-volume 4DFlowNet inference from a case listed in a NIfTI case CSV and "
            "save predicted NIfTI outputs plus an analysis payload for reporting."
        )
    )
    parser.add_argument("--case-csv", required=True, help="CSV with paired LR/HR NIfTI paths.")
    parser.add_argument("--case-index", type=int, default=0, help="Case index in CSV.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path (.pt).")
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help=(
            "Optional model architecture variant override "
            "(e.g., original, residual_skip, pre_res_attention, pre_upsample_attention, phase1_attention, phase2_attention). "
            "If omitted, use checkpoint metadata when available."
        ),
    )

    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--output-prefix", default="pred", help="Output NIfTI prefix name.")

    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis in input NIfTI (default last axis).")
    parser.add_argument(
        "--frame-index",
        type=int,
        nargs="*",
        default=None,
        help="Optional frame indices to process. If omitted, all available frames are processed.",
    )
    parser.add_argument(
        "--use-csv-frame-selection",
        action="store_true",
        help="If set and --frame-index is not provided, use CSV time_index/time_start/time_end instead of all frames.",
    )

    parser.add_argument("--patch-size", type=int, default=16, help="Sliding-window ROI size per axis.")
    parser.add_argument("--sw-batch-size", type=int, default=2, help="Sliding-window batch size.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding-window overlap [0,1).")

    parser.add_argument("--res-increase", type=int, default=2, help="Upsampling ratio used by the model.")
    parser.add_argument("--low-resblock", type=int, default=8, help="Number of low-res residual blocks.")
    parser.add_argument("--hi-resblock", type=int, default=4, help="Number of high-res residual blocks.")
    parser.add_argument(
        "--predict-mag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for predict_mag model head. Default: inferred from checkpoint.",
    )
    parser.add_argument(
        "--cascade-use-seg-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for cascade stage-1 segmentation head. Default: inferred from checkpoint.",
    )
    parser.add_argument(
        "--cascade-use-seg-mask-at-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override to use cascade seg-head mask at inference. Default: inferred from checkpoint.",
    )

    parser.add_argument(
        "--raw-phase-input",
        dest="lr_raw_phase_input",
        action="store_true",
        default=True,
        help="Assume LR/input velocity NIfTI values are raw phase-like and convert to velocity.",
    )
    parser.add_argument(
        "--already-velocity-input",
        dest="lr_raw_phase_input",
        action="store_false",
        help="Disable RAW conversion for LR/input velocity (input already physical).",
    )
    parser.add_argument(
        "--hr-raw-phase-target",
        dest="hr_raw_phase_input",
        action="store_true",
        default=None,
        help="Assume HR/target velocity NIfTI values are raw phase-like and convert to velocity.",
    )
    parser.add_argument(
        "--hr-already-velocity-target",
        dest="hr_raw_phase_input",
        action="store_false",
        help="Disable RAW conversion for HR/target velocity (target already physical).",
    )
    parser.add_argument(
        "--legacy-invert-uv-sign-on-raw",
        action="store_true",
        help="Legacy mode: invert U/V signs after RAW->velocity conversion.",
    )
    parser.add_argument("--raw-center", type=float, default=2048.0, help="Raw phase center.")
    parser.add_argument("--raw-scale", type=float, default=2048.0, help="Raw phase scale.")
    parser.add_argument(
        "--mag-scale",
        type=float,
        default=4095.0,
        help="Magnitude normalization divisor (used only with --mag-norm-mode divisor).",
    )
    parser.add_argument(
        "--mag-norm-mode",
        type=str,
        default="monai_minmax",
        choices=["monai_minmax", "divisor"],
        help="Magnitude normalization mode. monai_minmax applies MONAI ScaleIntensity to [0,1] per frame.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument(
        "--apply-mask-to-lr-inputs",
        action="store_true",
        help="Apply the CSV/reference mask to LR inputs before inference, matching masked-input training.",
    )
    parser.add_argument(
        "--apply-mask-to-lr-magnitude",
        dest="apply_mask_to_lr_magnitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --apply-mask-to-lr-inputs is enabled, also mask LR magnitude channels.",
    )
    parser.add_argument(
        "--use-reference-mask-for-cascade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Provide CSV/reference mask to cascade model during inference when available.",
    )
    parser.add_argument(
        "--save-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save predicted segmentation probability/binary NIfTI when the model emits seg_map.",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize saved segmentation predictions.",
    )

    args = parser.parse_args()
    hr_raw_phase_input = bool(args.lr_raw_phase_input) if args.hr_raw_phase_input is None else bool(args.hr_raw_phase_input)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti_dir = out_dir / "nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)

    cases = load_case_table(args.case_csv, include_hr_mag=True)
    if not cases:
        raise RuntimeError(f"No rows found in CSV: {args.case_csv}")

    case_idx = int(args.case_index) % len(cases)
    case = cases[case_idx]

    # Determine frame indices with a first-pass load to know t_count.
    preview = _read_case_volumes(case=case, time_axis=int(args.time_axis))
    explicit = _parse_frame_indices(args.frame_index)
    if explicit:
        frame_indices = explicit
        frame_mode = "explicit"
    elif args.use_csv_frame_selection:
        frame_indices = choose_frame_indices(
            case=case,
            t_count=int(preview["t_count"]),
            explicit_indices=None,
        )
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
        model_variant=args.model_variant,
        cascade_use_seg_head=args.cascade_use_seg_head,
        cascade_use_seg_mask_at_inference=args.cascade_use_seg_mask_at_inference,
    )

    if not predict_mag_flag:
        raise ValueError(
            "This workflow expects a 4-channel model output (predict_mag=True), "
            "but the checkpoint reports predict_mag=False."
        )

    print(f"Device: {device}")
    print(f"Model variant: {getattr(model, 'model_variant', 'original')}")
    print(f"Selected case: {case_idx}/{len(cases)-1}")
    print(f"Frame selection mode: {frame_mode}")
    print(f"Frames: {frame_indices}")

    payload = run_case_inference(
        case=case,
        model=model,
        frame_indices=frame_indices,
        patch_size=int(args.patch_size),
        sw_batch_size=int(args.sw_batch_size),
        overlap=float(args.overlap),
        device=device,
        lr_raw_phase_input=bool(args.lr_raw_phase_input),
        hr_raw_phase_input=bool(hr_raw_phase_input),
        legacy_invert_uv_sign_on_raw=bool(args.legacy_invert_uv_sign_on_raw),
        raw_center=float(args.raw_center),
        raw_scale=float(args.raw_scale),
        mag_scale=float(args.mag_scale),
        mag_norm_mode=str(args.mag_norm_mode),
        mask_threshold=float(args.mask_threshold),
        time_axis=int(args.time_axis),
        apply_mask_to_lr_inputs=bool(args.apply_mask_to_lr_inputs),
        apply_mask_to_lr_magnitude=bool(args.apply_mask_to_lr_magnitude),
        use_reference_mask_for_model=bool(args.use_reference_mask_for_cascade),
    )

    output_prefix = str((nifti_dir / args.output_prefix).resolve())
    nifti_paths = save_predicted_nifti(
        pred_norm=payload["pred_norm"],
        venc_per_frame=payload["venc"],
        mag_scale=float(args.mag_scale),
        mag_norm_mode=str(args.mag_norm_mode),
        out_prefix=output_prefix,
        lr_affine=payload["lr_affine"],
        res_increase=int(args.res_increase),
        save_uvw=True,
    )
    seg_nifti_paths = {}
    if bool(args.save_segmentation) and "seg_pred" in payload:
        seg_nifti_paths = save_segmentation_nifti(
            seg_pred=payload["seg_pred"],
            out_prefix=output_prefix,
            lr_affine=payload["lr_affine"],
            res_increase=int(args.res_increase),
            threshold=float(args.seg_threshold),
        )

    payload_path = out_dir / "analysis_payload.npz"
    save_payload_npz(str(payload_path), payload)

    meta = {
        "case_csv": str(Path(args.case_csv).resolve()),
        "case_index": int(case_idx),
        "frame_indices": [int(x) for x in frame_indices],
        "frame_selection_mode": frame_mode,
        "model_path": str(Path(args.model_path).resolve()),
        "model_variant": str(getattr(model, "model_variant", "original")),
        "predict_mag": bool(predict_mag_flag),
        "lr_raw_phase_input": bool(args.lr_raw_phase_input),
        "hr_raw_phase_input": bool(hr_raw_phase_input),
        "legacy_invert_uv_sign_on_raw": bool(args.legacy_invert_uv_sign_on_raw),
        "raw_center": float(args.raw_center),
        "raw_scale": float(args.raw_scale),
        "mag_scale": float(args.mag_scale),
        "mag_norm_mode": str(args.mag_norm_mode),
        "mask_threshold": float(args.mask_threshold),
        "apply_mask_to_lr_inputs": bool(args.apply_mask_to_lr_inputs),
        "apply_mask_to_lr_magnitude": bool(args.apply_mask_to_lr_magnitude),
        "use_reference_mask_for_cascade": bool(args.use_reference_mask_for_cascade),
        "patch_size": int(args.patch_size),
        "sw_batch_size": int(args.sw_batch_size),
        "overlap": float(args.overlap),
        "res_increase": int(args.res_increase),
        "low_resblock": int(args.low_resblock),
        "hi_resblock": int(args.hi_resblock),
        "time_axis": int(args.time_axis),
        "nifti_outputs": nifti_paths,
        "segmentation_outputs": seg_nifti_paths,
        "payload_path": str(payload_path.resolve()),
        "cascade_use_seg_head": bool(getattr(model, "cascade_use_seg_head", False)),
        "cascade_use_seg_mask_at_inference": bool(getattr(model, "cascade_use_seg_mask_at_inference", False)),
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
    meta_path = out_dir / "inference_metadata.json"
    save_metadata_json(str(meta_path), meta)

    print("\nSaved artifacts:")
    print(f"- Payload: {payload_path}")
    print(f"- Metadata: {meta_path}")
    for key, val in nifti_paths.items():
        print(f"- NIfTI ({key}): {val}")
    for key, val in seg_nifti_paths.items():
        print(f"- Segmentation ({key}): {val}")


if __name__ == "__main__":
    main()
