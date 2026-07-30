#!/usr/bin/env python3
"""
Remap LR columns in a paired NIfTI case CSV to a new downsampled root (x0.5 -> SR x2).
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List


LR_COLS = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w"]


def _resolve(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _rel_or_abs(path: Path, out_csv_dir: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path)
    return os.path.relpath(str(path), str(out_csv_dir))


def _map_cols(mode: str) -> Dict[str, str]:
    if mode == "lr_from_lr":
        return {
            "lr_u": "lr_u",
            "lr_v": "lr_v",
            "lr_w": "lr_w",
            "lr_mag_u": "lr_mag_u",
            "lr_mag_v": "lr_mag_v",
            "lr_mag_w": "lr_mag_w",
        }
    if mode == "lr_from_hr":
        return {
            "lr_u": "hr_u",
            "lr_v": "hr_v",
            "lr_w": "hr_w",
            "lr_mag_u": "hr_mag",
            "lr_mag_v": "hr_mag",
            "lr_mag_w": "hr_mag",
        }
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new paired case CSV for SR x2 experiments.")
    parser.add_argument("--in-csv", required=True, help="Input paired case CSV.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["lr_from_lr", "lr_from_hr"],
        help="How to generate new LR columns: from existing LR files or from HR files.",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory used by the source columns in --mode (e.g., data/paired_dataset/lr_3t or hr_7t_in_3t).",
    )
    parser.add_argument(
        "--new-lr-root",
        required=True,
        help="Root directory containing downsampled files with the same relative tree as --source-root.",
    )
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute paths in output CSV.")
    args = parser.parse_args()

    in_csv = Path(args.in_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    in_base = in_csv.parent
    out_base = out_csv.parent
    out_base.mkdir(parents=True, exist_ok=True)

    source_root = Path(args.source_root).resolve()
    new_lr_root = Path(args.new_lr_root).resolve()
    col_map = _map_cols(str(args.mode))

    with in_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames: List[str] = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No header found in CSV: {in_csv}")

        rows = []
        for i, row in enumerate(reader, start=1):
            out_row = dict(row)

            for dst_col in LR_COLS:
                src_col = col_map[dst_col]
                src_value = (row.get(src_col, "") or "").strip()
                if not src_value:
                    raise ValueError(f"Row {i}: missing source column '{src_col}' for mode '{args.mode}'")

                src_abs = _resolve(src_value, in_base)
                try:
                    rel = src_abs.relative_to(source_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Row {i}: source path is not under --source-root.\n"
                        f"  src={src_abs}\n"
                        f"  source_root={source_root}\n"
                        f"  (set correct --source-root)"
                    ) from exc

                new_abs = (new_lr_root / rel).resolve()
                out_row[dst_col] = _rel_or_abs(new_abs, out_base, bool(args.absolute_paths))

            rows.append(out_row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Mode: {args.mode}")
    print(f"Source root: {source_root}")
    print(f"New LR root: {new_lr_root}")


if __name__ == "__main__":
    main()
