#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_SCRIPT = REPO_ROOT / "code" / "segmentation" / "segment_cow_patient_pipeline.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "topcow-claim-models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "cow_segmentation_predictions"


def _iter_metadata(root: Path, recursive: bool, metadata_name: str) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(metadata_name)
    else:
        yield from root.glob(f"*/{metadata_name}")
        candidate = root / metadata_name
        if candidate.is_file():
            yield candidate


def discover_prediction_cases(input_root: Path, recursive: bool, metadata_name: str) -> List[Path]:
    case_dirs = []
    seen = set()
    for meta_path in _iter_metadata(input_root, recursive=recursive, metadata_name=metadata_name):
        case_dir = meta_path.resolve().parent
        if case_dir in seen:
            continue
        seen.add(case_dir)
        case_dirs.append(case_dir)
    return sorted(case_dirs)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_output_path(case_dir: Path, configured_path: str, fallback_rel: str) -> str:
    raw = str(configured_path or "").strip()
    if not raw:
        return fallback_rel
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((case_dir / p).resolve())


def _resolve_reference_mask(meta: dict) -> str:
    case_paths = meta.get("case_paths", {}) or {}
    return str(case_paths.get("mask", "") or "").strip()


def build_segment_command(args: argparse.Namespace, case_dir: Path) -> List[str]:
    meta_path = case_dir / args.metadata_name
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    meta = _load_json(meta_path)
    nifti_outputs = meta.get("nifti_outputs", {}) or {}

    mag_path = _resolve_output_path(case_dir, nifti_outputs.get("mag", ""), args.pred_mag_rel)
    u_path = _resolve_output_path(case_dir, nifti_outputs.get("u", ""), args.pred_u_rel)
    v_path = _resolve_output_path(case_dir, nifti_outputs.get("v", ""), args.pred_v_rel)
    w_path = _resolve_output_path(case_dir, nifti_outputs.get("w", ""), args.pred_w_rel)

    required = [Path(mag_path), Path(u_path), Path(v_path), Path(w_path)]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction file(s) for {case_dir.name}: {missing}")

    case_time_axis = int(meta.get("time_axis", args.time_axis))
    case_venc = float(args.venc)
    mask_path = ""
    if args.mask_from_metadata:
        mask_path = _resolve_reference_mask(meta)

    cmd = [
        sys.executable,
        str(args.segment_script),
        "--patient-dir",
        str(case_dir),
        "--output-dir",
        str(args.output_dir),
        "--model-dir",
        str(args.model_dir),
        "--mag-name",
        mag_path,
        "--vx-name",
        u_path,
        "--vy-name",
        v_path,
        "--vz-name",
        w_path,
        "--venc",
        str(case_venc),
        "--time-axis",
        str(case_time_axis),
        "--mask-threshold",
        str(args.mask_threshold),
        "--angio-mode",
        str(args.angio_mode),
        "--speed-percentile",
        str(args.speed_percentile),
        "--mag-projection-method",
        str(args.mag_projection_method),
        "--mag-projection-percentile",
        str(args.mag_projection_percentile),
        "--mag-projection-topk",
        str(args.mag_projection_topk),
        "--classic-sigmas",
        args.classic_sigmas,
        "--classic-percentile",
        str(args.classic_percentile),
        "--classic-morph-radius",
        str(args.classic_morph_radius),
        "--classic-min-component-size",
        str(args.classic_min_component_size),
        "--classic-z-min-frac",
        str(args.classic_z_min_frac),
        "--classic-z-max-frac",
        str(args.classic_z_max_frac),
        "--post-close-radius",
        str(args.post_close_radius),
        "--post-open-radius",
        str(args.post_open_radius),
        "--post-min-component-size",
        str(args.post_min_component_size),
    ]

    if mask_path:
        cmd.extend(["--mask-path", mask_path])
    if args.ai_only:
        cmd.extend(["--no-classic-cow", "--ensemble-mode", "ai"])
    elif args.classic_only:
        cmd.extend(["--classic-only", "--ensemble-mode", "classic"])
    else:
        cmd.extend(["--ensemble-mode", args.ensemble_mode])

    if args.no_postprocess:
        cmd.append("--no-postprocess")
    if args.no_fill_holes:
        cmd.append("--no-fill-holes")
    if args.classic_use_morph_open:
        cmd.append("--classic-use-morph-open")
    if args.classic_no_morph_close:
        cmd.append("--classic-no-morph-close")
    if args.save_intermediates:
        cmd.append("--save-intermediates")
    if args.keep_temp:
        cmd.append("--keep-temp")

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-segment CoW from prediction folders that contain inference_metadata.json "
            "and NIfTI outputs such as nifti/pred_u.nii.gz, pred_v.nii.gz, pred_w.nii.gz, pred_mag.nii.gz."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder containing prediction case folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for CoW masks.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="nnU-Net CoW model directory.")
    parser.add_argument("--segment-script", type=Path, default=DEFAULT_SEGMENT_SCRIPT, help="Path to segment script.")
    parser.add_argument("--metadata-name", type=str, default="inference_metadata.json", help="Metadata filename.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under --input-root.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases that already have cow_seg_final.nii.gz.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after first failure.")

    parser.add_argument("--pred-mag-rel", default="nifti/pred_mag.nii.gz", help="Fallback MAG path relative to case dir.")
    parser.add_argument("--pred-u-rel", default="nifti/pred_u.nii.gz", help="Fallback U path relative to case dir.")
    parser.add_argument("--pred-v-rel", default="nifti/pred_v.nii.gz", help="Fallback V path relative to case dir.")
    parser.add_argument("--pred-w-rel", default="nifti/pred_w.nii.gz", help="Fallback W path relative to case dir.")

    parser.add_argument(
        "--mask-from-metadata",
        action="store_true",
        help="Pass case_paths.mask from inference_metadata.json as analysis mask to the segmentation pipeline.",
    )
    parser.add_argument("--venc", type=float, default=0.90, help="VENC fallback value in m/s.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Fallback time axis if metadata is missing it.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold passed to segmentation.")
    parser.add_argument("--angio-mode", choices=["mag_speed", "mag_only"], default="mag_speed")
    parser.add_argument("--speed-percentile", type=float, default=90.0)
    parser.add_argument("--mag-projection-method", choices=["median", "max", "percentile", "topk_mean"], default="percentile")
    parser.add_argument("--mag-projection-percentile", type=float, default=95.0)
    parser.add_argument("--mag-projection-topk", type=int, default=3)

    parser.add_argument("--classic-sigmas", default="1,2,3,4")
    parser.add_argument("--classic-percentile", type=float, default=95.0)
    parser.add_argument("--classic-morph-radius", type=int, default=1)
    parser.add_argument("--classic-use-morph-open", action="store_true")
    parser.add_argument("--classic-no-morph-close", action="store_true")
    parser.add_argument("--classic-min-component-size", type=int, default=80)
    parser.add_argument("--classic-z-min-frac", type=float, default=0.15)
    parser.add_argument("--classic-z-max-frac", type=float, default=0.95)

    parser.add_argument("--ensemble-mode", choices=["union", "intersection", "ai", "classic"], default="union")
    parser.add_argument("--ai-only", action="store_true", help="Disable classic branch and use only AI mask.")
    parser.add_argument("--classic-only", action="store_true", help="Disable AI branch and use only classic mask.")

    parser.add_argument("--no-postprocess", action="store_true", help="Disable final postprocessing.")
    parser.add_argument("--post-close-radius", type=int, default=1)
    parser.add_argument("--post-open-radius", type=int, default=0)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--post-min-component-size", type=int, default=30)

    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ai_only and args.classic_only:
        raise ValueError("--ai-only and --classic-only are mutually exclusive.")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    case_dirs = discover_prediction_cases(args.input_root, recursive=args.recursive, metadata_name=args.metadata_name)
    if not case_dirs:
        print(f"No prediction cases found under {args.input_root} with metadata {args.metadata_name}.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0
    skipped = 0

    for case_dir in case_dirs:
        final_mask = args.output_dir / case_dir.name / "cow_seg_final.nii.gz"
        if args.skip_existing and final_mask.is_file():
            print(f"[SKIP] {case_dir.name}: {final_mask}")
            skipped += 1
            continue

        try:
            cmd = build_segment_command(args, case_dir)
            print(f"[RUN] {case_dir.name}")
            print("      " + " ".join(cmd))
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                failed += 1
                print(f"[FAIL] {case_dir.name} exited with code {completed.returncode}")
                if args.stop_on_error:
                    return completed.returncode
            else:
                ok += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {case_dir.name}: {exc}")
            if args.stop_on_error:
                raise

    print(f"Completed. ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
