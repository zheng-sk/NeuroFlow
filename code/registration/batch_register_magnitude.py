#!/usr/bin/env python3
"""Batch temporal registration using magnitude as transform source."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


FORBIDDEN_MAG_TOKENS = {"phase", "vx", "vy", "vz", "velocity", "flow_x", "flow_y", "flow_z"}


def _nframes(path: str) -> int:
    """Return number of time frames in a NIfTI file without loading data."""
    import nibabel as nib
    shape = nib.load(path).shape
    return int(shape[3]) if len(shape) == 4 else 1


def identify_images(folder: str) -> tuple[str | None, tuple[str, str, str] | None, list[str]]:
    """Identify magnitude, complete velocity triplet (if present), and remaining scalar files.

    When multiple magnitude candidates exist, the one with the most time frames is
    preferred so that single-frame anatomical references (fl3d1r_t70, fl3d1_ns, etc.)
    are never chosen over the actual 4D flow magnitude.
    """
    files = glob.glob(os.path.join(folder, "*.nii.gz")) or glob.glob(os.path.join(folder, "*.nii"))

    mag_candidates: list[str] = []
    vx = vy = vz = None
    other_files: list[str] = []

    for file_path in files:
        name = os.path.basename(file_path).lower()

        is_mag = ("magnitude" in name or "mag" in name) and not any(token in name for token in FORBIDDEN_MAG_TOKENS)
        if is_mag:
            mag_candidates.append(file_path)
            continue

        if "vx" in name or "flow_x" in name:
            vx = file_path
        elif "vy" in name or "flow_y" in name:
            vy = file_path
        elif "vz" in name or "flow_z" in name:
            vz = file_path
        else:
            other_files.append(file_path)

    # Pick the magnitude with the most frames; put the rest in other_files.
    mag_file = None
    if mag_candidates:
        mag_candidates.sort(key=_nframes, reverse=True)
        mag_file = mag_candidates[0]
        other_files.extend(mag_candidates[1:])

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
    parser.add_argument("--skip-existing", action="store_true", help="Skip folders whose output already exists")
    parser.add_argument("--workers", type=int, default=1, help="Number of patients to process in parallel")
    parser.add_argument("--itk-threads", type=int, default=0,
                        help="ITK/ANTs threads per worker (0 = auto: cpu_count // workers)")
    return parser.parse_args()


def find_patient_dirs(input_dir: str) -> list[str]:
    patient_dirs: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        if any(name.endswith(".nii") or name.endswith(".nii.gz") for name in files):
            patient_dirs.append(root)
    return sorted(patient_dirs)


def _process_patient(
    patient_dir: str,
    input_dir: str,
    output_dir: str,
    reg_type: str,
    show_frame_progress: bool,
    skip_existing: bool,
    itk_threads: int,
) -> dict:
    """Process one patient folder. Safe to call from a subprocess."""
    if itk_threads > 0:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)

    # Deferred import so ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is set first.
    try:
        from code.registration.core import apply_transforms_to_file, apply_transforms_to_velocity_triplet, register_4d_nifti
    except ImportError:
        from core import apply_transforms_to_file, apply_transforms_to_velocity_triplet, register_4d_nifti

    folder_name = Path(patient_dir).name
    rel_dir = os.path.relpath(patient_dir, input_dir)
    target_dir = os.path.join(output_dir, rel_dir)

    result = dict(
        folder_name=folder_name, rel_dir=rel_dir,
        images_processed=0, images_failed=0,
        mag_seconds=0.0, velocity_seconds=0.0, others_seconds=0.0, patient_seconds=0.0,
    )

    mag_path, velocity_triplet, other_paths = identify_images(patient_dir)
    if not mag_path:
        print(f"[SKIP] {folder_name}: magnitude file not found.")
        result["images_failed"] += 1
        return result

    mag_out = os.path.join(target_dir, os.path.basename(mag_path))
    if skip_existing and os.path.exists(mag_out):
        print(f"[SKIP] {folder_name}: output already exists.")
        return result

    qc_dir = os.path.join(target_dir, "QC_Registration")
    patient_start = time.perf_counter()

    mag_start = time.perf_counter()
    transforms = register_4d_nifti(
        input_path=mag_path, output_path=mag_out, qc_dir=qc_dir,
        ref_t=0, reg_type=reg_type,
        show_frame_progress=show_frame_progress, progress_label=folder_name,
    )
    result["mag_seconds"] = time.perf_counter() - mag_start

    if transforms is None:
        result["images_failed"] += 1
        print(f"[FAIL] {folder_name}: temporal registration failed for magnitude.")
        result["patient_seconds"] = time.perf_counter() - patient_start
        return result

    result["images_processed"] += 1

    if velocity_triplet:
        vx_in, vy_in, vz_in = velocity_triplet
        vel_start = time.perf_counter()
        ok = apply_transforms_to_velocity_triplet(
            vx_path=vx_in, vy_path=vy_in, vz_path=vz_in,
            out_vx=os.path.join(target_dir, os.path.basename(vx_in)),
            out_vy=os.path.join(target_dir, os.path.basename(vy_in)),
            out_vz=os.path.join(target_dir, os.path.basename(vz_in)),
            transforms_map=transforms, fixed_ref_mag_path=mag_path, ref_t=0,
        )
        result["velocity_seconds"] = time.perf_counter() - vel_start
        if not ok:
            print(f"Warning: velocity vector transform failed for {folder_name}")
            result["images_failed"] += 3
        else:
            result["images_processed"] += 3

    for file_path in other_paths:
        oth_start = time.perf_counter()
        ok = apply_transforms_to_file(
            input_path=file_path,
            output_path=os.path.join(target_dir, os.path.basename(file_path)),
            transforms_map=transforms, ref_img_path=mag_path,
        )
        result["others_seconds"] += time.perf_counter() - oth_start
        if ok:
            result["images_processed"] += 1
        else:
            result["images_failed"] += 1

    result["patient_seconds"] = time.perf_counter() - patient_start
    print(f"[OK] {folder_name}: {result['patient_seconds']:.1f}s")
    return result


def main() -> None:
    args = parse_args()

    patient_dirs = find_patient_dirs(args.input_dir)
    if not patient_dirs:
        print("No folders with NIfTI files found.")
        return

    print(f"Found {len(patient_dirs)} folders containing NIfTI files.")

    workers = max(1, args.workers)
    itk_threads = args.itk_threads if args.itk_threads > 0 else max(1, os.cpu_count() // workers)
    print(f"Workers: {workers}  |  ITK threads/worker: {itk_threads}")

    import functools
    import multiprocessing as mp

    task = functools.partial(
        _process_patient,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reg_type=args.reg_type,
        show_frame_progress=args.show_frame_progress,
        skip_existing=args.skip_existing,
        itk_threads=itk_threads,
    )

    if workers == 1:
        results = [task(d) for d in tqdm(patient_dirs, desc="Processing folders", unit="folder")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(task, patient_dirs), total=len(patient_dirs),
                                desc="Processing folders", unit="folder"))

    csv_rows: list[str] = []
    total_patient_seconds = 0.0
    total_images_processed = 0
    total_images_failed = 0

    for r in results:
        total_patient_seconds += r["patient_seconds"]
        total_images_processed += r["images_processed"]
        total_images_failed += r["images_failed"]
        n = r["images_processed"] + r["images_failed"]
        avg = r["patient_seconds"] / max(1, n)
        csv_rows.append(",".join([
            r["folder_name"], r["rel_dir"].replace(",", "_"),
            str(r["images_processed"]), str(r["images_failed"]),
            f"{r['mag_seconds']:.4f}", f"{r['velocity_seconds']:.4f}",
            f"{r['others_seconds']:.4f}", f"{r['patient_seconds']:.4f}", f"{avg:.4f}",
        ]))
        if args.verbose:
            print(
                f"[PATIENT] {r['folder_name']}: total={r['patient_seconds']:.2f}s "
                f"(mag={r['mag_seconds']:.2f}s, velocity={r['velocity_seconds']:.2f}s, "
                f"other={r['others_seconds']:.2f}s), "
                f"images ok={r['images_processed']}, failed={r['images_failed']}, "
                f"avg/image={avg:.2f}s"
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
