#!/usr/bin/env python3
"""Batch temporal registration using magnitude as transform source."""

from __future__ import annotations

import argparse
import glob
import os
import time
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
    parser.add_argument("--show-frame-progress", action="store_true", help="Show per-frame temporal progress")
    parser.add_argument("--timing-report", default=None, help="Optional CSV path for per-patient timing summary")
    parser.add_argument("--verbose", action="store_true", help="Print per-patient timing details")
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

    csv_rows: list[str] = []
    total_patient_seconds = 0.0
    total_images_processed = 0
    total_images_failed = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing folders", unit="folder"):
        patient_start = time.perf_counter()
        mag_path, velocity_triplet, other_paths = identify_images(patient_dir)
        if not mag_path:
            total_images_failed += 1
            print(f"[SKIP] {patient_dir}: magnitude file not found.")
            continue

        rel_dir = os.path.relpath(patient_dir, args.input_dir)
        target_dir = os.path.join(args.output_dir, rel_dir)

        folder_name = Path(patient_dir).name
        mag_out = os.path.join(target_dir, os.path.basename(mag_path))
        qc_dir = os.path.join(target_dir, "QC_Registration")

        patient_images_processed = 0
        patient_images_failed = 0
        mag_seconds = 0.0
        velocity_seconds = 0.0
        others_seconds = 0.0

        mag_start = time.perf_counter()
        transforms = register_4d_nifti(
            input_path=mag_path,
            output_path=mag_out,
            qc_dir=qc_dir,
            ref_t=0,
            reg_type=args.reg_type,
            show_frame_progress=args.show_frame_progress,
            progress_label=folder_name,
        )
        mag_seconds = time.perf_counter() - mag_start
        if transforms is None:
            patient_images_failed += 1
            total_images_failed += 1
            print(f"[FAIL] {folder_name}: temporal registration failed for magnitude.")
            continue

        patient_images_processed += 1
        total_images_processed += 1

        if velocity_triplet:
            vx_in, vy_in, vz_in = velocity_triplet
            velocity_start = time.perf_counter()
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
            velocity_seconds = time.perf_counter() - velocity_start
            if not ok:
                print(f"Warning: velocity vector transform failed for {folder_name}")
                patient_images_failed += 3
                total_images_failed += 3
            else:
                patient_images_processed += 3
                total_images_processed += 3

        for file_path in other_paths:
            other_start = time.perf_counter()
            ok = apply_transforms_to_file(
                input_path=file_path,
                output_path=os.path.join(target_dir, os.path.basename(file_path)),
                transforms_map=transforms,
                ref_img_path=mag_path,
            )
            others_seconds += time.perf_counter() - other_start
            if ok:
                patient_images_processed += 1
                total_images_processed += 1
            else:
                patient_images_failed += 1
                total_images_failed += 1

        patient_seconds = time.perf_counter() - patient_start
        total_patient_seconds += patient_seconds
        avg_seconds = patient_seconds / max(1, patient_images_processed + patient_images_failed)

        csv_rows.append(
            ",".join(
                [
                    folder_name,
                    rel_dir.replace(",", "_"),
                    str(patient_images_processed),
                    str(patient_images_failed),
                    f"{mag_seconds:.4f}",
                    f"{velocity_seconds:.4f}",
                    f"{others_seconds:.4f}",
                    f"{patient_seconds:.4f}",
                    f"{avg_seconds:.4f}",
                ]
            )
        )

        if args.verbose:
            print(
                f"[PATIENT] {folder_name}: total={patient_seconds:.2f}s "
                f"(mag={mag_seconds:.2f}s, velocity={velocity_seconds:.2f}s, other={others_seconds:.2f}s), "
                f"images ok={patient_images_processed}, failed={patient_images_failed}, "
                f"avg/image={avg_seconds:.2f}s"
            )

    print("\nBatch processing complete.")
    print(f"Folders processed: {len(patient_dirs)}")
    print(f"Images processed:  {total_images_processed}")
    print(f"Images failed:     {total_images_failed}")
    print(f"Total wall time:   {total_patient_seconds:.2f}s")
    if total_images_processed + total_images_failed > 0:
        print(
            f"Average/image:     "
            f"{total_patient_seconds / (total_images_processed + total_images_failed):.2f}s"
        )

    if args.timing_report:
        report_dir = os.path.dirname(args.timing_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.timing_report, "w", encoding="utf-8") as handle:
            handle.write(
                "patient,relative_dir,images_processed,images_failed,"
                "seconds_mag,seconds_velocity,seconds_other,seconds_total,seconds_per_image\n"
            )
            for row in csv_rows:
                handle.write(row + "\n")
        print(f"Timing report written to: {args.timing_report}")


if __name__ == "__main__":
    main()
