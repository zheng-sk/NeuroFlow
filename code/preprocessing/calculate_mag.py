#!/usr/bin/env python3
"""Generate derived magnitude features from 4D flow velocity components."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

CSV_COLUMNS = {
    "case_id": "Case_ID",
    "magnitude": "Path_Mag",
    "vx": "Path_Vx",
    "vy": "Path_Vy",
    "vz": "Path_Vz",
}


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else root / path


def save_nifti(data: np.ndarray, affine: np.ndarray, header, out_path: Path) -> None:
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), affine, header)
    nib.save(image, str(out_path))


def process_case(vx_path: Path, vy_path: Path, vz_path: Path, mag_path: Path, output_dir: Path, case_id: str) -> bool:
    print(f"\n--- Processing case: {case_id} ---")
    start = time.time()

    missing = []
    for name, path in (("Vx", vx_path), ("Vy", vy_path), ("Vz", vz_path), ("Magnitude", mag_path)):
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        print("Missing input files:")
        for item in missing:
            print(f"- {item}")
        return False

    try:
        vx_img = nib.load(str(vx_path))
        vy_img = nib.load(str(vy_path))
        vz_img = nib.load(str(vz_path))
        mag_img = nib.load(str(mag_path))
    except Exception as exc:  # pragma: no cover - I/O failure path
        print(f"Failed to read NIfTI files: {exc}")
        return False

    vx = vx_img.get_fdata(dtype=np.float32)
    vy = vy_img.get_fdata(dtype=np.float32)
    vz = vz_img.get_fdata(dtype=np.float32)
    mag = mag_img.get_fdata(dtype=np.float32)

    if not (vx.shape == vy.shape == vz.shape == mag.shape):
        print("Shape mismatch detected:")
        print(f"- Vx: {vx.shape}")
        print(f"- Vy: {vy.shape}")
        print(f"- Vz: {vz.shape}")
        print(f"- Magnitude: {mag.shape}")
        return False

    speed = np.sqrt(vx * vx + vy * vy + vz * vz, dtype=np.float32)
    pcmra = mag * speed

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(vx_path, output_dir / "Vx.nii.gz")
    shutil.copy2(vy_path, output_dir / "Vy.nii.gz")
    shutil.copy2(vz_path, output_dir / "Vz.nii.gz")

    affine = vx_img.affine
    header = vx_img.header.copy()
    save_nifti(mag, affine, header, output_dir / "input_mag_raw.nii.gz")
    save_nifti(speed, affine, header, output_dir / "input_speed_raw.nii.gz")
    save_nifti(pcmra, affine, header, output_dir / "input_pcmra_raw.nii.gz")

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute speed and PC-MRA volumes from dataset.csv")
    parser.add_argument("--csv", required=True, type=Path, help="Path to dataset CSV")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        type=Path,
        help="Root folder used to resolve relative paths inside the CSV",
    )
    parser.add_argument(
        "--output-subdir",
        default="processed_inputs",
        help="Output subdirectory inside --data-root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        return

    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        required = [CSV_COLUMNS[key] for key in CSV_COLUMNS]
        missing_columns = [name for name in required if name not in (reader.fieldnames or [])]
        if missing_columns:
            print(f"Missing CSV columns: {missing_columns}")
            return

        rows = list(reader)

    total = len(rows)
    success = 0
    failed = 0
    print(f"Found {total} cases")

    output_root = args.data_root / args.output_subdir

    for index, row in enumerate(rows, start=1):
        print(f"Progress [{index}/{total}]")
        case_id = row[CSV_COLUMNS["case_id"]].strip()

        out_dir = output_root / case_id
        ok = process_case(
            vx_path=resolve_path(args.data_root, row[CSV_COLUMNS["vx"]]),
            vy_path=resolve_path(args.data_root, row[CSV_COLUMNS["vy"]]),
            vz_path=resolve_path(args.data_root, row[CSV_COLUMNS["vz"]]),
            mag_path=resolve_path(args.data_root, row[CSV_COLUMNS["magnitude"]]),
            output_dir=out_dir,
            case_id=case_id,
        )

        if ok:
            success += 1
        else:
            failed += 1

    print("\n--- Summary ---")
    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
