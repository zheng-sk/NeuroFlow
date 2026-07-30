#!/usr/bin/env python3
"""Randomly sample validation patients from a paired NIfTI CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a validation CSV by randomly sampling N patients (case-level) "
            "from a paired LR/HR CSV. Optionally writes a train CSV with the rest."
        )
    )
    parser.add_argument("--csv-in", required=True, help="Input paired CSV.")
    parser.add_argument("--val-csv-out", required=True, help="Output validation CSV.")
    parser.add_argument("--train-csv-out", default=None, help="Optional output train CSV (remaining cases).")
    parser.add_argument(
        "--num-patients",
        type=int,
        default=1,
        help="Number of unique patients/cases to sample for validation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--hr-col", default="hr_u", help="CSV column used to infer case id.")
    parser.add_argument(
        "--hr-root-name",
        default="hr_7t_in_3t",
        help="Folder token in hr path used to recover relative case id.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a row has empty hr column or case id cannot be inferred.",
    )
    return parser.parse_args()


def infer_case_id(hr_value: str, hr_root_name: str) -> str:
    hr_path = Path(hr_value.strip())
    if not hr_path.parts:
        return ""
    parent = hr_path.parent
    parts = parent.parts
    if hr_root_name in parts:
        idx = parts.index(hr_root_name)
        rel = parts[idx + 1 :]
        if rel:
            return str(Path(*rel))
    return parent.name


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    csv_in = Path(args.csv_in)
    val_out = Path(args.val_csv_out)
    train_out = Path(args.train_csv_out) if args.train_csv_out else None

    if not csv_in.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")
    if args.num_patients <= 0:
        raise ValueError("--num-patients must be >= 1")

    with csv_in.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_in}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if args.hr_col not in fieldnames:
        raise KeyError(f"Column `{args.hr_col}` not found in CSV header.")

    case_by_row: list[str] = []
    unique_cases: list[str] = []
    seen: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        hr_val = (row.get(args.hr_col) or "").strip()
        case_id = infer_case_id(hr_val, args.hr_root_name) if hr_val else ""
        if not case_id:
            if args.strict:
                raise ValueError(f"Row {idx} has invalid `{args.hr_col}` value: {hr_val!r}")
            case_id = f"__row_{idx}"
        case_by_row.append(case_id)
        if case_id not in seen:
            unique_cases.append(case_id)
            seen.add(case_id)

    if args.num_patients > len(unique_cases):
        raise ValueError(
            f"Requested {args.num_patients} patients but CSV has only {len(unique_cases)} unique cases."
        )

    rng = random.Random(args.seed)
    selected_cases = set(rng.sample(unique_cases, args.num_patients))

    val_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []

    for row, case_id in zip(rows, case_by_row):
        if case_id in selected_cases:
            val_rows.append(row)
        else:
            train_rows.append(row)

    write_csv(val_out, fieldnames, val_rows)
    if train_out is not None:
        write_csv(train_out, fieldnames, train_rows)

    print("Validation split CSV created")
    print(f"Input CSV:         {csv_in}")
    print(f"Unique cases:      {len(unique_cases)}")
    print(f"Selected for val:  {len(selected_cases)}")
    print(f"Validation rows:   {len(val_rows)} -> {val_out}")
    if train_out is not None:
        print(f"Training rows:     {len(train_rows)} -> {train_out}")
    print("Selected cases:")
    for case in sorted(selected_cases):
        print(f"- {case}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
