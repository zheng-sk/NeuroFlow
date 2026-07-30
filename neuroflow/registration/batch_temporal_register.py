#!/usr/bin/env python3
"""Batch temporal registration for 4D NIfTI files."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

try:
    from .core import register_4d_nifti
except ImportError:  # pragma: no cover - allows direct script execution from this folder
    from core import register_4d_nifti

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch temporal registration for 4D NIfTI files.")
    parser.add_argument("--input-dir", required=True, help="Root directory containing input NIfTI files")
    parser.add_argument("--output-dir", required=True, help="Root directory for registered outputs")
    parser.add_argument("--pattern", default="*.nii.gz", help="File pattern to match (default: *.nii.gz)")
    parser.add_argument("--reg-type", default="Rigid", help="ANTs transform type (Rigid, Affine, SyN, ...)")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under input directory")
    parser.add_argument("--ref-t", type=int, default=0, help="Reference frame index (default: 0)")
    parser.add_argument(
        "--qc-frames",
        default="0,50,99",
        help="Percentiles to sample for QC overlays (default: 0,50,99)",
    )
    parser.add_argument("--interpolator", default="linear", help="Interpolator for warping frames")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose registration logs")
    return parser.parse_args()


def find_files(input_dir: str, pattern: str, recursive: bool) -> list[str]:
    search_path = os.path.join(input_dir, "**", pattern) if recursive else os.path.join(input_dir, pattern)
    files = glob.glob(search_path, recursive=recursive)
    return sorted(files)


def main() -> None:
    args = parse_args()

    files = find_files(args.input_dir, args.pattern, args.recursive)
    if not files:
        print(f"No files found matching '{args.pattern}' in {args.input_dir}")
        return

    print(f"Found {len(files)} files")
    print(f"Output directory: {args.output_dir}")
    print(f"Registration: {args.reg_type} | ref_t={args.ref_t}")

    success_count = 0
    fail_count = 0

    for input_path in tqdm(files, desc="Registering files", unit="file"):
        relative_path = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, relative_path)

        base = os.path.basename(input_path)
        stem = base[:-7] if base.endswith(".nii.gz") else Path(base).stem
        qc_dir = os.path.join(os.path.dirname(output_path), f"QC_{stem}")

        transforms = register_4d_nifti(
            input_path=input_path,
            output_path=output_path,
            qc_dir=qc_dir,
            ref_t=args.ref_t,
            reg_type=args.reg_type,
            interpolator=args.interpolator,
            qc_frames_str=args.qc_frames,
            verbose=args.verbose,
        )

        if transforms is not None:
            success_count += 1
        else:
            fail_count += 1

    print("\nBatch temporal registration complete")
    print(f"Total: {len(files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
