#!/usr/bin/env python3
"""Batch temporal registration using magnitude as transform source."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

try:
    from .core import apply_transforms_to_file, apply_transforms_to_velocity_triplet, register_4d_nifti
except ImportError:  # pragma: no cover - allows direct script execution from this folder
    from core import apply_transforms_to_file, apply_transforms_to_velocity_triplet, register_4d_nifti

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


FORBIDDEN_MAG_TOKENS = {"phase", "vx", "vy", "vz", "velocity", "flow_x", "flow_y", "flow_z"}


def identify_images(folder: str) -> tuple[str | None, tuple[str, str, str] | None, list[str]]:
    """Identify magnitude, complete velocity triplet (if present), and remaining scalar files."""
    files = glob.glob(os.path.join(folder, "*.nii.gz")) or glob.glob(os.path.join(folder, "*.nii"))

    mag_file = None
    vx = vy = vz = None
    other_files: list[str] = []

    for file_path in files:
        name = os.path.basename(file_path).lower()

        is_mag = ("magnitude" in name or "mag" in name) and not any(token in name for token in FORBIDDEN_MAG_TOKENS)
        if is_mag:
            if mag_file is None:
                mag_file = file_path
            else:
                other_files.append(file_path)
            continue

        if "vx" in name or "flow_x" in name:
            vx = file_path
        elif "vy" in name or "flow_y" in name:
            vy = file_path
        elif "vz" in name or "flow_z" in name:
            vz = file_path
        else:
            other_files.append(file_path)

    velocity_triplet = (vx, vy, vz) if vx and vy and vz else None
    if velocity_triplet is None:
        if vx:
            other_files.append(vx)
        if vy:
            other_files.append(vy)
        if vz:
            other_files.append(vz)

    return mag_file, velocity_triplet, other_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register magnitude per folder and propagate transforms.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reg-type", default="Rigid")
    return parser.parse_args()


def find_patient_dirs(input_dir: str) -> list[str]:
    patient_dirs: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        if any(name.endswith(".nii") or name.endswith(".nii.gz") for name in files):
            patient_dirs.append(root)
    return sorted(patient_dirs)


def main() -> None:
    args = parse_args()

    patient_dirs = find_patient_dirs(args.input_dir)
    if not patient_dirs:
        print("No folders with NIfTI files found.")
        return

    print(f"Found {len(patient_dirs)} folders containing NIfTI files.")

    for patient_dir in tqdm(patient_dirs, desc="Processing folders"):
        mag_path, velocity_triplet, other_paths = identify_images(patient_dir)
        if not mag_path:
            continue

        rel_dir = os.path.relpath(patient_dir, args.input_dir)
        target_dir = os.path.join(args.output_dir, rel_dir)

        folder_name = Path(patient_dir).name
        mag_out = os.path.join(target_dir, os.path.basename(mag_path))
        qc_dir = os.path.join(target_dir, "QC_Registration")

        transforms = register_4d_nifti(
            input_path=mag_path,
            output_path=mag_out,
            qc_dir=qc_dir,
            ref_t=0,
            reg_type=args.reg_type,
        )
        if transforms is None:
            continue

        if velocity_triplet:
            vx_in, vy_in, vz_in = velocity_triplet
            ok = apply_transforms_to_velocity_triplet(
                vx_path=vx_in,
                vy_path=vy_in,
                vz_path=vz_in,
                out_vx=os.path.join(target_dir, os.path.basename(vx_in)),
                out_vy=os.path.join(target_dir, os.path.basename(vy_in)),
                out_vz=os.path.join(target_dir, os.path.basename(vz_in)),
                transforms_map=transforms,
                fixed_ref_mag_path=mag_path,
                ref_t=0,
            )
            if not ok:
                print(f"Warning: velocity vector transform failed for {folder_name}")

        for file_path in other_paths:
            apply_transforms_to_file(
                input_path=file_path,
                output_path=os.path.join(target_dir, os.path.basename(file_path)),
                transforms_map=transforms,
                ref_img_path=mag_path,
            )

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
