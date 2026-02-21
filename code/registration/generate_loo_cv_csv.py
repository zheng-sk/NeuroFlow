#!/usr/bin/env python3
"""Generate leave-one-out train/val/test CSV splits at patient level."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leave-one-out cross-validation splits from a paired NIfTI CSV. "
            "Each fold holds out one patient as test; validation patients are "
            "sampled from remaining patients."
        )
    )
    parser.add_argument("--csv-in", required=True, help="Input paired CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory for folds.")
    parser.add_argument("--hr-col", default="hr_u", help="CSV column used to infer patient/case id.")
    parser.add_argument(
        "--hr-root-name",
        default="hr_7t_in_3t",
        help="Folder token in hr path used to recover relative case id.",
    )
    parser.add_argument(
        "--val-num-patients",
        type=int,
        default=1,
        help="Number of validation patients per fold (0 disables validation split).",
    )
    parser.add_argument(
        "--train-val-only",
        action="store_true",
        help=(
            "Generate only train/val folds (no test.csv). "
            "Each fold uses leave-one-out validation: 1 patient in val, rest in train."
        ),
    )
    parser.add_argument(
        "--val-strategy",
        choices=["cyclic", "random"],
        default="cyclic",
        help="How to choose validation patients from the non-test pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --val-strategy random.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a row has missing/invalid hr column and case id cannot be inferred.",
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


def sanitize_case_id(case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id.strip())
    return cleaned.strip("._-") or "case"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_val_cases(
    remaining_cases: list[str],
    fold_index: int,
    val_num: int,
    strategy: str,
    seed: int,
) -> set[str]:
    if val_num <= 0 or not remaining_cases:
        return set()
    if val_num > len(remaining_cases):
        raise ValueError(
            f"Requested val_num_patients={val_num} but only {len(remaining_cases)} non-test patients remain."
        )

    if strategy == "random":
        rng = random.Random(seed + fold_index)
        return set(rng.sample(remaining_cases, val_num))

    # cyclic strategy (deterministic, balanced over folds)
    start = fold_index % len(remaining_cases)
    val_cases = []
    for offset in range(val_num):
        idx = (start + offset) % len(remaining_cases)
        val_cases.append(remaining_cases[idx])
    return set(val_cases)


def main() -> int:
    args = parse_args()
    csv_in = Path(args.csv_in)
    out_dir = Path(args.out_dir)

    if not csv_in.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")
    if args.val_num_patients < 0:
        raise ValueError("--val-num-patients must be >= 0")

    with csv_in.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_in}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if args.hr_col not in fieldnames:
        raise KeyError(f"Column `{args.hr_col}` not found in CSV header.")

    row_case_ids: list[str] = []
    unique_cases: list[str] = []
    seen = set()
    for idx, row in enumerate(rows, start=1):
        hr_val = (row.get(args.hr_col) or "").strip()
        case_id = infer_case_id(hr_val, args.hr_root_name) if hr_val else ""
        if not case_id:
            if args.strict:
                raise ValueError(f"Row {idx} has invalid `{args.hr_col}` value: {hr_val!r}")
            case_id = f"__row_{idx}"
        row_case_ids.append(case_id)
        if case_id not in seen:
            seen.add(case_id)
            unique_cases.append(case_id)

    unique_cases = sorted(unique_cases)
    if len(unique_cases) < 2:
        raise ValueError("Leave-one-out requires at least 2 unique patients/cases.")
    if args.train_val_only and args.val_num_patients != 1:
        raise ValueError("--train-val-only requires --val-num-patients=1.")
    if args.val_num_patients >= len(unique_cases):
        raise ValueError(
            f"--val-num-patients={args.val_num_patients} must be smaller than total cases={len(unique_cases)}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for fold_index, fold_case in enumerate(unique_cases):
        if args.train_val_only:
            val_cases = {fold_case}
            train_cases = set(unique_cases) - val_cases
            test_case = None
        else:
            test_case = fold_case
            remaining = [c for c in unique_cases if c != test_case]
            val_cases = choose_val_cases(
                remaining_cases=remaining,
                fold_index=fold_index,
                val_num=args.val_num_patients,
                strategy=args.val_strategy,
                seed=args.seed,
            )
            train_cases = set(remaining) - val_cases

        train_rows: list[dict[str, str]] = []
        val_rows: list[dict[str, str]] = []
        test_rows: list[dict[str, str]] = []

        for row, case_id in zip(rows, row_case_ids):
            if test_case is not None and case_id == test_case:
                test_rows.append(row)
            elif case_id in val_cases:
                val_rows.append(row)
            elif case_id in train_cases:
                train_rows.append(row)

        fold_tag = next(iter(sorted(val_cases))) if args.train_val_only else str(test_case)
        fold_name = f"fold_{fold_index:03d}_{sanitize_case_id(fold_tag)}"
        fold_dir = out_dir / fold_name
        write_csv(fold_dir / "train.csv", fieldnames, train_rows)
        write_csv(fold_dir / "val.csv", fieldnames, val_rows)
        if not args.train_val_only:
            write_csv(fold_dir / "test.csv", fieldnames, test_rows)

        summary_rows.append(
            {
                "fold": fold_name,
                "test_case": "" if test_case is None else str(test_case),
                "num_train_rows": str(len(train_rows)),
                "num_val_rows": str(len(val_rows)),
                "num_test_rows": str(len(test_rows)),
                "num_train_cases": str(len(train_cases)),
                "num_val_cases": str(len(val_cases)),
                "num_test_cases": "0" if test_case is None else "1",
                "val_cases": ";".join(sorted(val_cases)),
            }
        )

    summary_path = out_dir / "loo_folds_summary.csv"
    summary_fields = [
        "fold",
        "test_case",
        "num_train_rows",
        "num_val_rows",
        "num_test_rows",
        "num_train_cases",
        "num_val_cases",
        "num_test_cases",
        "val_cases",
    ]
    write_csv(summary_path, summary_fields, summary_rows)

    print("LOO split generation complete")
    print(f"Input CSV:      {csv_in}")
    print(f"Unique cases:   {len(unique_cases)}")
    print(f"Output folder:  {out_dir}")
    print(f"Summary CSV:    {summary_path}")
    if args.train_val_only:
        print("Mode: train/val only (leave-one-out validation), no test.csv generated.")
    else:
        print(f"Validation per fold (cases): {args.val_num_patients} [{args.val_strategy}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
