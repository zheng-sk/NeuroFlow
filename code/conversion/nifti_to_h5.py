#!/usr/bin/env python3
"""Batch convert processed NIfTI patient folders to HDF5 files."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "processed_inputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "h5_inputs"
DEFAULT_SCRIPT_PATH = REPO_ROOT / "src" / "prepare_data" / "prepare_nifti_data.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert processed NIfTIs to HDF5 for prediction.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Folder with processed patient subfolders")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder to save .h5 files")
    parser.add_argument("--script-path", type=Path, default=DEFAULT_SCRIPT_PATH, help="Path to prepare_nifti_data.py")
    parser.add_argument("--venc", type=str, default="0.90", help="VENC value passed to prepare_nifti_data.py")
    return parser.parse_args()


def required_files(patient_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        patient_dir / "Vx.nii.gz",
        patient_dir / "Vy.nii.gz",
        patient_dir / "Vz.nii.gz",
        patient_dir / "input_mag_raw.nii.gz",
    )


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        print(f"Error: input directory not found: {args.input_root}")
        return
    if not args.script_path.exists():
        print(f"Error: preparation script not found: {args.script_path}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    print(f"Found {len(patient_dirs)} patients in {args.input_root}")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"\n--- Converting: {patient_id} ---")

        u_path, v_path, w_path, mag_path = required_files(patient_dir)
        if not all(path.exists() for path in (u_path, v_path, w_path, mag_path)):
            print(f"Skipping {patient_id}: missing required NIfTI files in {patient_dir}")
            continue

        output_h5 = f"{patient_id}.h5"
        cmd = [
            sys.executable,
            str(args.script_path),
            "--u",
            str(u_path),
            "--v",
            str(v_path),
            "--w",
            str(w_path),
            "--mag",
            str(mag_path),
            "--venc",
            args.venc,
            "--output-dir",
            str(args.output_dir),
            "--output-filename",
            output_h5,
        ]

        print("Running preparation script...")
        try:
            subprocess.run(cmd, check=True)
            print(f"--> Created: {args.output_dir / output_h5}")
        except subprocess.CalledProcessError as exc:
            print(f"Error processing {patient_id}: {exc}")

    print("\nBatch conversion completed.")


if __name__ == "__main__":
    main()
