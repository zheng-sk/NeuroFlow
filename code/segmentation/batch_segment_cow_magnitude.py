#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_SCRIPT = REPO_ROOT / "code" / "segmentation" / "segment_cow_patient_pipeline.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "topcow-claim-models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cow_segmentation_patient_batch"


def _iter_glob(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    return root.rglob(pattern) if recursive else root.glob(pattern)


def discover_patient_dirs(input_root: Path, recursive: bool, mag_patterns: List[str]) -> List[Path]:
    patient_dirs = set()
    for pattern in mag_patterns:
        for path in _iter_glob(input_root, pattern, recursive):
            if path.is_file():
                patient_dirs.add(path.resolve().parent)
    return sorted(patient_dirs)


def missing_required_files(patient_dir: Path, mag_name: str, vx_name: str, vy_name: str, vz_name: str) -> List[Path]:
    expected = [
        patient_dir / mag_name,
        patient_dir / vx_name,
        patient_dir / vy_name,
        patient_dir / vz_name,
    ]
    return [p for p in expected if not p.is_file()]


def run_case(args: argparse.Namespace, patient_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(args.segment_script),
        "--patient-dir",
        str(patient_dir),
        "--output-dir",
        str(args.output_dir),
        "--model-dir",
        str(args.model_dir),
        "--mag-name",
        args.mag_name,
        "--vx-name",
        args.vx_name,
        "--vy-name",
        args.vy_name,
        "--vz-name",
        args.vz_name,
        "--venc",
        str(args.venc),
        "--time-axis",
        str(args.time_axis),
        "--mask-threshold",
        str(args.mask_threshold),
        "--speed-percentile",
        str(args.speed_percentile),
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

    if args.mask_path is not None:
        cmd.extend(["--mask-path", str(args.mask_path)])

    if args.ai_only:
        cmd.extend(["--no-classic-cow", "--ensemble-mode", "ai"])
    elif args.classic_only:
        cmd.extend(["--classic-only", "--ensemble-mode", "classic"])
    else:
        cmd.extend(["--ensemble-mode", args.ensemble_mode])

    if args.classic_use_morph_open:
        cmd.append("--classic-use-morph-open")
    if args.classic_no_morph_close:
        cmd.append("--classic-no-morph-close")

    if args.no_postprocess:
        cmd.append("--no-postprocess")
    if args.no_fill_holes:
        cmd.append("--no-fill-holes")

    if args.save_intermediates:
        cmd.append("--save-intermediates")
    if args.keep_temp:
        cmd.append("--keep-temp")

    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch CoW segmentation case-by-case using patient folders. "
            "Each case is processed with segment_cow_patient_pipeline.py, "
            "which builds angiography from magnitude + velocity (speed)."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder containing patient subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"nnU-Net model folder (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--segment-script",
        type=Path,
        default=DEFAULT_SEGMENT_SCRIPT,
        help=f"Path to segment_cow_patient_pipeline.py (default: {DEFAULT_SEGMENT_SCRIPT}).",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search patient folders recursively using magnitude file patterns.",
    )
    parser.add_argument(
        "--mag-pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern used only to discover patient folders. Repeat for multiple patterns. "
            "Defaults: mag_7T_in_3T.nii.gz, input_mag_raw.nii.gz"
        ),
    )

    parser.add_argument("--mag-name", default="mag_7T_in_3T.nii.gz", help="Magnitude filename inside each patient folder.")
    parser.add_argument("--vx-name", default="phaseX_7T_in_3T.nii.gz", help="Velocity X filename inside each patient folder.")
    parser.add_argument("--vy-name", default="phaseY_7T_in_3T.nii.gz", help="Velocity Y filename inside each patient folder.")
    parser.add_argument("--vz-name", default="phaseZ_7T_in_3T.nii.gz", help="Velocity Z filename inside each patient folder.")

    parser.add_argument("--venc", type=float, default=0.90, help="VENC in m/s for RAW->velocity conversion in angiography build.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis in input NIfTI.")
    parser.add_argument("--mask-path", type=Path, default=None, help="Optional external mask path.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument("--speed-percentile", type=float, default=90.0, help="Percentile over speed(t) for angiography.")

    parser.add_argument("--classic-sigmas", default="1,2,3,4", help="Comma-separated sigmas for frangi/sato.")
    parser.add_argument("--classic-percentile", type=float, default=95.0, help="Percentile threshold over vesselness.")
    parser.add_argument("--classic-morph-radius", type=int, default=1, help="Morphology radius for classic branch.")
    parser.add_argument("--classic-use-morph-open", action="store_true", help="Enable opening in classic branch.")
    parser.add_argument("--classic-no-morph-close", action="store_true", help="Disable closing in classic branch.")
    parser.add_argument("--classic-min-component-size", type=int, default=80, help="Min component size in classic branch.")
    parser.add_argument("--classic-z-min-frac", type=float, default=0.15, help="Lower z fraction for classic ROI filtering.")
    parser.add_argument("--classic-z-max-frac", type=float, default=0.95, help="Upper z fraction for classic ROI filtering.")

    parser.add_argument(
        "--ensemble-mode",
        choices=["union", "intersection", "ai", "classic"],
        default="union",
        help="How to combine AI and classical masks when classic branch is enabled.",
    )
    parser.add_argument(
        "--ai-only",
        action="store_true",
        help="Disable classic branch and force AI-only output.",
    )
    parser.add_argument(
        "--classic-only",
        action="store_true",
        help="Skip AI inference and force classic-only output.",
    )

    parser.add_argument("--no-postprocess", action="store_true", help="Disable final mask postprocessing.")
    parser.add_argument("--post-close-radius", type=int, default=1, help="Final closing radius.")
    parser.add_argument("--post-open-radius", type=int, default=0, help="Final opening radius.")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable final hole filling.")
    parser.add_argument("--post-min-component-size", type=int, default=30, help="Final min component size.")

    parser.add_argument("--save-intermediates", action="store_true", help="Save intermediate maps/masks from patient pipeline.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary nnU-Net folders from patient pipeline.")

    parser.add_argument("--stop-on-error", action="store_true", help="Stop execution when one case fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ai_only and args.classic_only:
        raise ValueError("--ai-only and --classic-only are mutually exclusive.")
    if not args.mag_pattern:
        args.mag_pattern = ["mag_7T_in_3T.nii.gz", "input_mag_raw.nii.gz"]

    if not args.input_root.exists():
        raise FileNotFoundError(f"input-root does not exist: {args.input_root}")
    if not args.segment_script.exists():
        raise FileNotFoundError(f"segment script does not exist: {args.segment_script}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")

    patient_dirs = discover_patient_dirs(args.input_root, args.recursive, args.mag_pattern)
    if not patient_dirs:
        raise RuntimeError(
            "No patient folders were found from magnitude patterns. "
            f"input_root={args.input_root}, patterns={args.mag_pattern}, recursive={args.recursive}"
        )

    print(f"Found {len(patient_dirs)} patient folder(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    skipped = []

    for i, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        print(f"\n[{i}/{len(patient_dirs)}] Case: {patient_id}")
        print(f"Patient dir: {patient_dir}")

        missing = missing_required_files(
            patient_dir,
            mag_name=args.mag_name,
            vx_name=args.vx_name,
            vy_name=args.vy_name,
            vz_name=args.vz_name,
        )
        if missing:
            skipped.append((patient_id, [str(p) for p in missing]))
            print("[SKIP] missing required files:")
            for p in missing:
                print(f"  - {p}")
            if args.stop_on_error:
                break
            continue

        code = run_case(args, patient_dir)
        if code != 0:
            failed.append((patient_id, str(patient_dir), code))
            print(f"[ERROR] case={patient_id} returncode={code}")
            if args.stop_on_error:
                break

    if skipped:
        print("\nSkipped cases (missing required files):")
        for patient_id, missing in skipped:
            print(f"- {patient_id}")
            for path in missing:
                print(f"  {path}")

    if failed:
        print("\nFailed cases:")
        for patient_id, patient_dir, code in failed:
            print(f"- {patient_id}: {patient_dir} (code={code})")
        return 1

    mode = "AI-only" if args.ai_only else f"AI+classic ({args.ensemble_mode})"
    post = "disabled" if args.no_postprocess else "enabled"
    print(
        "\nDone: all discovered cases were segmented via patient pipeline "
        f"(angiography built from MAG+velocity, {mode}, postprocessing {post})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
