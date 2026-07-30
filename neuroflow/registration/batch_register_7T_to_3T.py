#!/usr/bin/env python3
"""Batch 7T->3T registration by pairing folders and running per-case registration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass(frozen=True)
class CasePair:
    case_key: str
    fixed_dir: str
    moving_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch registration of 7T data into 3T space.")
    parser.add_argument("--input-dir", required=True, help="Root directory containing paired 3T/7T folders")
    parser.add_argument("--output-dir", required=True, help="Root directory for per-case outputs")

    parser.add_argument("--fixed-suffix", default="_3T", help="Suffix used by fixed (3T) folders")
    parser.add_argument("--moving-suffix", default="_7T", help="Suffix used by moving (7T) folders")

    parser.add_argument("--fixed-mag-name", default="input_mag_raw.nii.gz", help="Magnitude file inside 3T folder")
    parser.add_argument("--moving-mag-name", default="input_mag_raw.nii.gz", help="Magnitude file inside 7T folder")
    parser.add_argument("--phase-x-name", default="Vx.nii.gz", help="Phase/velocity X file inside 7T folder")
    parser.add_argument("--phase-y-name", default="Vy.nii.gz", help="Phase/velocity Y file inside 7T folder")
    parser.add_argument("--phase-z-name", default="Vz.nii.gz", help="Phase/velocity Z file inside 7T folder")

    parser.add_argument("--mask-method", choices=["ants", "hdbet", "none"], default="ants")
    parser.add_argument("--device", default="cpu", help="For HD-BET: cpu/mps/cuda")
    parser.add_argument("--use-tta", action="store_true", help="Enable test-time augmentation in HD-BET")

    parser.add_argument("--reg-type", default="antsRegistrationSyN[a]")
    parser.add_argument("--interpolator-mag", default="bSpline")
    parser.add_argument("--phase-warp-mode", choices=["direct", "complex"], default="direct")
    parser.add_argument("--interpolator-phase", default="nearestNeighbor")
    parser.add_argument("--hist-match", action="store_true")
    parser.add_argument("--qc-frames", default="0,50,99")
    parser.add_argument(
        "--save-brain-masks",
        action="store_true",
        help="Persist per-case 3T/7T brain masks estimated during inter-scan registration",
    )
    parser.add_argument(
        "--brain-mask-subdir",
        default=None,
        help="Optional per-case subdirectory name where masks are saved (default uses registration script default)",
    )

    parser.add_argument("--overwrite", action="store_true", help="Re-run cases even when output already exists")
    parser.add_argument("--strict", action="store_true", help="Stop on first failed case")
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without running registration")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
            if rel_parent == ".":
                case_key = case_base
            else:
                case_key = os.path.join(rel_parent, case_base)

            pairs.append(CasePair(case_key=case_key, fixed_dir=fixed_dir, moving_dir=moving_dir))

    # De-duplicate by case key while preserving deterministic ordering.
    unique_pairs = {pair.case_key: pair for pair in pairs}
    return [unique_pairs[key] for key in sorted(unique_pairs)]


def required_inputs(pair: CasePair, args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    def resolve_existing(base_dir: str, primary: str, aliases: list[str]) -> str:
        for candidate in [primary] + aliases:
            full_path = os.path.join(base_dir, candidate)
            if os.path.isfile(full_path):
                return full_path
        return os.path.join(base_dir, primary)

    paths = {
        "fixed_mag_3t": resolve_existing(
            pair.fixed_dir,
            args.fixed_mag_name,
            ["mag_3T_in_3T.nii.gz", "magnitude.nii.gz"],
        ),
        "moving_mag_7t": resolve_existing(
            pair.moving_dir,
            args.moving_mag_name,
            ["mag_7T_in_3T.nii.gz", "magnitude.nii.gz"],
        ),
        "moving_phase_x": resolve_existing(
            pair.moving_dir,
            args.phase_x_name,
            ["input_phase_x_raw.nii.gz", "vx.nii.gz", "flow_x.nii.gz", "phaseX_7T_in_3T.nii.gz"],
        ),
        "moving_phase_y": resolve_existing(
            pair.moving_dir,
            args.phase_y_name,
            ["input_phase_y_raw.nii.gz", "vy.nii.gz", "flow_y.nii.gz", "phaseY_7T_in_3T.nii.gz"],
        ),
        "moving_phase_z": resolve_existing(
            pair.moving_dir,
            args.phase_z_name,
            ["input_phase_z_raw.nii.gz", "vz.nii.gz", "flow_z.nii.gz", "phaseZ_7T_in_3T.nii.gz"],
        ),
    }
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    return paths, missing


def build_case_command(
    case_inputs: dict[str, str],
    out_dir: str,
    qc_dir: str,
    args: argparse.Namespace,
) -> list[str]:
    register_script = Path(__file__).with_name("register_7T_to_3T_with_qc.py")
    cmd = [
        sys.executable,
        str(register_script),
        "--fixed_mag_3t",
        case_inputs["fixed_mag_3t"],
        "--moving_mag_7t",
        case_inputs["moving_mag_7t"],
        "--moving_phase_x",
        case_inputs["moving_phase_x"],
        "--moving_phase_y",
        case_inputs["moving_phase_y"],
        "--moving_phase_z",
        case_inputs["moving_phase_z"],
        "--out_dir",
        out_dir,
        "--qc_dir",
        qc_dir,
        "--mask_method",
        args.mask_method,
        "--device",
        args.device,
        "--reg_type",
        args.reg_type,
        "--interpolator_mag",
        args.interpolator_mag,
        "--phase_warp_mode",
        args.phase_warp_mode,
        "--interpolator_phase",
        args.interpolator_phase,
        "--qc_frames",
        args.qc_frames,
    ]

    if args.use_tta:
        cmd.append("--use_tta")
    if args.hist_match:
        cmd.append("--hist_match")
    if args.save_brain_masks:
        cmd.append("--save_brain_masks")
    if args.brain_mask_subdir:
        cmd.extend(["--brain_mask_dir", os.path.join(out_dir, args.brain_mask_subdir)])
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def main() -> None:
    args = parse_args()

    pairs = discover_case_pairs(
        input_dir=args.input_dir,
        fixed_suffix=args.fixed_suffix,
        moving_suffix=args.moving_suffix,
    )
    if not pairs:
        print(
            f"No paired folders found in {args.input_dir} with suffixes "
            f"'{args.fixed_suffix}' and '{args.moving_suffix}'."
        )
        return

    print(f"Found {len(pairs)} paired cases.")
    print(f"Input root:  {args.input_dir}")
    print(f"Output root: {args.output_dir}")

    success = 0
    skipped = 0
    failed: list[str] = []

    for pair in tqdm(pairs, desc="Registering 7T->3T cases", unit="case"):
        out_dir = os.path.join(args.output_dir, pair.case_key)
        qc_dir = os.path.join(out_dir, "QC")
        expected_output = os.path.join(out_dir, "mag_7T_in_3T.nii.gz")

        if os.path.exists(expected_output) and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"[SKIP] {pair.case_key} (already exists)")
            continue

        inputs, missing = required_inputs(pair, args)
        if missing:
            failed.append(pair.case_key)
            print(f"[FAIL] {pair.case_key}: missing required files: {', '.join(missing)}")
            if args.strict:
                break
            continue

        cmd = build_case_command(inputs, out_dir, qc_dir, args)
        if args.dry_run:
            print(f"[DRY RUN] {pair.case_key}")
            print(" ", " ".join(cmd))
            success += 1
            continue

        os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            success += 1
            if args.verbose:
                print(f"[OK] {pair.case_key}")
        else:
            failed.append(pair.case_key)
            print(f"[FAIL] {pair.case_key}: registration exited with code {result.returncode}")
            if args.strict:
                break

    print("\nBatch 7T->3T registration complete")
    print(f"Total pairs: {len(pairs)}")
    print(f"Succeeded:   {success}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {len(failed)}")

    if failed:
        print("Failed cases:")
        for case in failed:
            print(f" - {case}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
