#!/usr/bin/env python3
"""Run temporal registration first, then batch 7T->3T registration."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CasePair:
    case_key: str
    fixed_dir: str
    moving_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full registration pipeline: temporal registration for all folders, "
            "then inter-scan 7T->3T batch registration."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Root directory with paired 3T/7T folders")
    parser.add_argument("--output-dir", required=True, help="Final output root (registered 7T in 3T space)")
    parser.add_argument(
        "--temporal-dir",
        default=None,
        help="Intermediate temporal output directory (default: <output-dir>/_temporal_registered)",
    )

    parser.add_argument("--fixed-suffix", default="_3T", help="Suffix used by fixed (3T) folders")
    parser.add_argument("--moving-suffix", default="_7T", help="Suffix used by moving (7T) folders")

    parser.add_argument("--fixed-mag-name", default="input_mag_raw.nii.gz")
    parser.add_argument("--moving-mag-name", default="input_mag_raw.nii.gz")
    parser.add_argument("--phase-x-name", default="Vx.nii.gz")
    parser.add_argument("--phase-y-name", default="Vy.nii.gz")
    parser.add_argument("--phase-z-name", default="Vz.nii.gz")

    parser.add_argument("--temporal-reg-type", default="Rigid", help="Temporal registration transform type")
    parser.add_argument(
        "--show-temporal-frame-progress",
        action="store_true",
        help="Show per-frame progress messages during temporal registration",
    )
    parser.add_argument(
        "--temporal-timing-report",
        default=None,
        help="Optional CSV path for temporal per-patient timing report",
    )
    parser.add_argument("--interscan-reg-type", default="antsRegistrationSyN[a]")
    parser.add_argument("--mask-method", choices=["ants", "hdbet"], default="ants")
    parser.add_argument("--device", default="cpu", help="For HD-BET: cpu/mps/cuda")
    parser.add_argument("--use-tta", action="store_true", help="Enable test-time augmentation in HD-BET")
    parser.add_argument("--interpolator-mag", default="bSpline")
    parser.add_argument("--phase-warp-mode", choices=["direct", "complex"], default="direct")
    parser.add_argument("--interpolator-phase", default="nearestNeighbor")
    parser.add_argument("--hist-match", action="store_true")
    parser.add_argument("--qc-frames", default="0,50,99")
    parser.add_argument("--strict", action="store_true", help="Stop batch inter-scan stage on first failed case")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-case final outputs")

    parser.add_argument("--keep-temporal", action="store_true", help="Keep intermediate temporal outputs")
    parser.add_argument("--keep-qc", action="store_true", help="Keep per-case QC folders in final output")
    parser.add_argument(
        "--keep-transforms",
        action="store_true",
        help="Keep per-case transform files in final output",
    )
    parser.add_argument("--brain-mask-subdir", default="BrainMasks", help="Per-case folder for saved brain masks")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Validate that every case has all final registered outputs before cleanup",
    )
    parser.add_argument(
        "--force-temporal",
        action="store_true",
        help="Always run temporal registration even if temporal outputs already exist and look complete",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print commands and planned cleanup only")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs in child scripts")
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def remove_named_dirs(root_dir: str, names: set[str]) -> int:
    removed = 0
    for current_root, dirs, _files in os.walk(root_dir, topdown=True):
        for dirname in list(dirs):
            if dirname not in names:
                continue
            target = os.path.join(current_root, dirname)
            shutil.rmtree(target, ignore_errors=True)
            dirs.remove(dirname)
            removed += 1
    return removed


def discover_case_pairs(input_dir: str, fixed_suffix: str, moving_suffix: str) -> list[CasePair]:
    pairs: list[CasePair] = []

    for root, dirs, _files in os.walk(input_dir):
        for dirname in dirs:
            if fixed_suffix and not dirname.endswith(fixed_suffix):
                continue

            case_base = dirname[:-len(fixed_suffix)] if fixed_suffix else dirname
            moving_name = case_base + moving_suffix

            fixed_dir = os.path.join(root, dirname)
            moving_dir = os.path.join(root, moving_name)

            if not os.path.isdir(moving_dir):
                continue

            rel_parent = os.path.relpath(root, input_dir)
            case_key = case_base if rel_parent == "." else os.path.join(rel_parent, case_base)
            pairs.append(CasePair(case_key=case_key, fixed_dir=fixed_dir, moving_dir=moving_dir))

    unique_pairs = {pair.case_key: pair for pair in pairs}
    return [unique_pairs[key] for key in sorted(unique_pairs)]


def validate_final_outputs(output_dir: str, case_keys: list[str]) -> list[tuple[str, list[str]]]:
    required_outputs = [
        "mag_7T_in_3T.nii.gz",
        "phaseX_7T_in_3T.nii.gz",
        "phaseY_7T_in_3T.nii.gz",
        "phaseZ_7T_in_3T.nii.gz",
    ]
    failures: list[tuple[str, list[str]]] = []

    for case_key in case_keys:
        case_dir = os.path.join(output_dir, case_key)
        missing = [
            filename for filename in required_outputs if not os.path.isfile(os.path.join(case_dir, filename))
        ]
        if missing:
            failures.append((case_key, missing))

    return failures


def build_pair_dirs(temporal_dir: str, case_key: str, fixed_suffix: str, moving_suffix: str) -> tuple[str, str]:
    case_path = Path(case_key)
    parent = "" if str(case_path.parent) == "." else str(case_path.parent)
    case_base = case_path.name

    fixed_dir = os.path.join(temporal_dir, parent, case_base + fixed_suffix)
    moving_dir = os.path.join(temporal_dir, parent, case_base + moving_suffix)
    return fixed_dir, moving_dir


def temporal_outputs_ready(args: argparse.Namespace, temporal_dir: str, case_keys: list[str]) -> tuple[bool, list[str]]:
    missing_details: list[str] = []

    for case_key in case_keys:
        fixed_dir, moving_dir = build_pair_dirs(temporal_dir, case_key, args.fixed_suffix, args.moving_suffix)
        expected = [
            ("fixed_mag", os.path.join(fixed_dir, args.fixed_mag_name)),
            ("moving_mag", os.path.join(moving_dir, args.moving_mag_name)),
            ("phase_x", os.path.join(moving_dir, args.phase_x_name)),
            ("phase_y", os.path.join(moving_dir, args.phase_y_name)),
            ("phase_z", os.path.join(moving_dir, args.phase_z_name)),
        ]

        for label, path in expected:
            if not os.path.isfile(path):
                missing_details.append(f"{case_key}:{label}:{path}")

    return len(missing_details) == 0, missing_details


def main() -> None:
    args = parse_args()

    temporal_dir = args.temporal_dir or os.path.join(args.output_dir, "_temporal_registered")
    temporal_preexisting = os.path.isdir(temporal_dir)
    discovered_pairs = discover_case_pairs(args.input_dir, args.fixed_suffix, args.moving_suffix)
    case_keys = [pair.case_key for pair in discovered_pairs]
    if not case_keys:
        raise SystemExit(
            f"No paired folders found in {args.input_dir} with suffixes "
            f"'{args.fixed_suffix}' and '{args.moving_suffix}'."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.dry_run:
        os.makedirs(temporal_dir, exist_ok=True)

    reg_dir = Path(__file__).parent
    temporal_script = reg_dir / "batch_register_magnitude.py"
    interscan_script = reg_dir / "batch_register_7T_to_3T.py"

    temporal_cmd = [
        sys.executable,
        str(temporal_script),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        temporal_dir,
        "--reg-type",
        args.temporal_reg_type,
    ]
    if args.show_temporal_frame_progress:
        temporal_cmd.append("--show-frame-progress")
    if args.temporal_timing_report:
        temporal_cmd.extend(["--timing-report", args.temporal_timing_report])
    if args.verbose:
        temporal_cmd.append("--verbose")
    run_temporal = True
    if not args.force_temporal and temporal_preexisting:
        ready, missing_items = temporal_outputs_ready(args, temporal_dir, case_keys)
        if ready:
            run_temporal = False
            print("== Stage 1/2: Temporal registration ==")
            print("Temporal outputs already detected. Skipping temporal stage.")
        elif args.verbose:
            print(f"Temporal outputs incomplete ({len(missing_items)} missing files). Re-running temporal stage.")

    if run_temporal:
        print("== Stage 1/2: Temporal registration ==")
        rc = run_command(temporal_cmd, args.dry_run)
        if rc != 0:
            raise SystemExit(f"Temporal stage failed with exit code {rc}")

    interscan_cmd = [
        sys.executable,
        str(interscan_script),
        "--input-dir",
        temporal_dir,
        "--output-dir",
        args.output_dir,
        "--fixed-suffix",
        args.fixed_suffix,
        "--moving-suffix",
        args.moving_suffix,
        "--fixed-mag-name",
        args.fixed_mag_name,
        "--moving-mag-name",
        args.moving_mag_name,
        "--phase-x-name",
        args.phase_x_name,
        "--phase-y-name",
        args.phase_y_name,
        "--phase-z-name",
        args.phase_z_name,
        "--mask-method",
        args.mask_method,
        "--device",
        args.device,
        "--reg-type",
        args.interscan_reg_type,
        "--interpolator-mag",
        args.interpolator_mag,
        "--phase-warp-mode",
        args.phase_warp_mode,
        "--interpolator-phase",
        args.interpolator_phase,
        "--qc-frames",
        args.qc_frames,
        "--save-brain-masks",
        "--brain-mask-subdir",
        args.brain_mask_subdir,
    ]

    if args.use_tta:
        interscan_cmd.append("--use-tta")
    if args.hist_match:
        interscan_cmd.append("--hist-match")
    if args.strict:
        interscan_cmd.append("--strict")
    if args.overwrite:
        interscan_cmd.append("--overwrite")
    if args.dry_run:
        interscan_cmd.append("--dry-run")
    if args.verbose:
        interscan_cmd.append("--verbose")

    print("== Stage 2/2: Inter-scan 7T->3T registration ==")
    rc = run_command(interscan_cmd, args.dry_run)
    if rc != 0:
        raise SystemExit(f"Inter-scan stage failed with exit code {rc}")

    if args.dry_run:
        print("Dry run complete: no files were written or removed.")
        return

    if args.final_only:
        failures = validate_final_outputs(args.output_dir, case_keys)
        if failures:
            print("\nFinal-output validation failed. Missing files:")
            for case_key, missing in failures:
                print(f" - {case_key}: {', '.join(missing)}")
            raise SystemExit("Aborting cleanup because some final outputs are incomplete.")
        print(f"Final-output validation passed for {len(case_keys)} case(s).")

    removed_qc = 0
    removed_tx = 0

    if not args.keep_qc:
        removed_qc = remove_named_dirs(args.output_dir, {"QC"})
    if not args.keep_transforms:
        removed_tx = remove_named_dirs(args.output_dir, {"Transforms_7T_to_3T"})

    removed_temporal = False
    if not args.keep_temporal:
        temporal_abs = os.path.abspath(temporal_dir)
        input_abs = os.path.abspath(args.input_dir)
        if temporal_abs == input_abs:
            raise SystemExit("Refusing to delete temporal directory because it matches --input-dir.")
        if temporal_preexisting and not run_temporal:
            if args.verbose:
                print("Keeping pre-existing temporal directory (was reused in this run).")
        elif os.path.isdir(temporal_abs):
            shutil.rmtree(temporal_abs, ignore_errors=True)
            removed_temporal = True

    print("\nFull registration pipeline complete")
    print(f"Final outputs: {args.output_dir}")
    print(f"Brain masks folder name: {args.brain_mask_subdir}")
    print(f"Removed QC folders: {removed_qc}")
    print(f"Removed transform folders: {removed_tx}")
    print(f"Removed temporal directory: {'yes' if removed_temporal else 'no'}")


if __name__ == "__main__":
    main()
