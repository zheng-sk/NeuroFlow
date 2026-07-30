#!/usr/bin/env python3
"""Attach CoW segmentation masks to the `mask` column of a paired training CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate or overwrite the `mask` column in a paired LR/HR CSV using CoW "
            "segmentation outputs (for example, cow_seg_final.nii.gz)."
        )
    )
    parser.add_argument("--csv-in", required=True, help="Input CSV path.")
    parser.add_argument("--csv-out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--masks-root",
        required=True,
        help="Root folder containing per-case CoW segmentation outputs.",
    )
    parser.add_argument(
        "--mask-name",
        default="cow_seg_final.nii.gz",
        help="Mask filename expected inside each case folder.",
    )
    parser.add_argument(
        "--hr-col",
        default="hr_u",
        help="CSV column used to infer case path (default: hr_u).",
    )
    parser.add_argument(
        "--hr-root-name",
        default="hr_7t_in_3t",
        help=(
            "Folder name in hr path used to recover the case relative path "
            "(default: hr_7t_in_3t)."
        ),
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative-to-cwd", "relative-to-csv-out"],
        default="relative-to-cwd",
        help="How to write paths into the mask column.",
    )
    parser.add_argument(
        "--only-empty",
        action="store_true",
        help="Only fill rows where `mask` is empty. Existing values are kept.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row cannot be mapped to an existing mask file.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-row details.")
    return parser.parse_args()


def infer_case_rel_path(hr_path_value: str, hr_root_name: str) -> Path:
    hr_path = Path(hr_path_value)
    parent = hr_path.parent
    parts = parent.parts
    if hr_root_name in parts:
        idx = parts.index(hr_root_name)
        rel_parts = parts[idx + 1 :]
        if rel_parts:
            return Path(*rel_parts)
    return Path(parent.name)


def format_mask_path(mask_path: Path, mode: str, csv_out_path: Path) -> str:
    mask_abs = mask_path.resolve()
    if mode == "absolute":
        return str(mask_abs)
    if mode == "relative-to-csv-out":
        base = csv_out_path.resolve().parent
        return os.path.relpath(mask_abs, start=base)
    return os.path.relpath(mask_abs, start=Path.cwd().resolve())


def insert_mask_field(fieldnames: list[str]) -> list[str]:
    if "mask" in fieldnames:
        return fieldnames
    updated = list(fieldnames)
    if "hr_w" in updated:
        idx = updated.index("hr_w") + 1
        updated.insert(idx, "mask")
    else:
        updated.append("mask")
    return updated


def main() -> int:
    args = parse_args()
    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    masks_root = Path(args.masks_root)

    if not csv_in.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Masks root folder not found: {masks_root}")

    with csv_in.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_in}")
        fieldnames = insert_mask_field(reader.fieldnames)
        rows = list(reader)

    if args.hr_col not in fieldnames:
        raise KeyError(f"Column `{args.hr_col}` not found in CSV header.")

    total = len(rows)
    filled = 0
    kept = 0
    missing = 0

    for idx, row in enumerate(rows, start=1):
        current_mask = (row.get("mask") or "").strip()
        if args.only_empty and current_mask:
            kept += 1
            continue

        hr_value = (row.get(args.hr_col) or "").strip()
        if not hr_value:
            missing += 1
            if args.verbose:
                print(f"[MISS] row={idx}: empty `{args.hr_col}`")
            continue

        case_rel = infer_case_rel_path(hr_value, args.hr_root_name)
        candidates = [masks_root / case_rel / args.mask_name]
        if case_rel != Path(case_rel.name):
            candidates.append(masks_root / case_rel.name / args.mask_name)

        match = next((c for c in candidates if c.is_file()), None)
        if match is None:
            missing += 1
            if args.verbose:
                print(f"[MISS] row={idx}: case={case_rel}")
            continue

        row["mask"] = format_mask_path(match, args.path_mode, csv_out)
        filled += 1
        if args.verbose:
            print(f"[OK] row={idx}: case={case_rel} -> {row['mask']}")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("CoW mask CSV update complete")
    print(f"Input CSV:   {csv_in}")
    print(f"Output CSV:  {csv_out}")
    print(f"Rows:        {total}")
    print(f"Masks set:   {filled}")
    print(f"Masks kept:  {kept}")
    print(f"Masks miss:  {missing}")

    if missing and args.strict:
        print("Strict mode: missing masks detected.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
