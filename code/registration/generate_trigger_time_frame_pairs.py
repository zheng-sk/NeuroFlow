#!/usr/bin/env python3
"""Expand case-level paired CSVs into TriggerTime-guided frame-pair CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Sequence


NEW_COLUMNS = [
    "lr_time_index",
    "hr_time_index",
    "lr_trigger_time_ms",
    "hr_trigger_time_ms",
    "pairing_method",
]
TRIGGER_TIME_DECIMALS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read TriggerTime from sorted-patient DICOM folders and expand a case-level paired "
            "CSV into one row per 3T->7T frame pair using nearest normalized cardiac phase."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Case-level CSV with 4D LR/HR NIfTI paths.")
    parser.add_argument("--output-csv", required=True, help="Expanded frame-paired CSV to write.")
    parser.add_argument(
        "--sorted-patients-root",
        default="data/sorted_patients",
        help="Root containing <CASE>_3T and <CASE>_7T DICOM folders.",
    )
    parser.add_argument("--lr-case-suffix", default="_3T", help="Suffix for LR DICOM case folders.")
    parser.add_argument("--hr-case-suffix", default="_7T", help="Suffix for HR DICOM case folders.")
    parser.add_argument(
        "--phase-vx-glob",
        default="Phase_Vx_*",
        help="Glob used to find the DICOM phase-X series inside each case folder.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case pairing summary.")
    return parser.parse_args()


def _resolve_path(path_value: str, base_dir: str) -> str:
    if path_value is None:
        return ""
    path_value = str(path_value).strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value

    base_abs = os.path.abspath(base_dir)
    search_roots = [base_abs, os.path.abspath(os.getcwd())]
    cursor = base_abs
    while True:
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        search_roots.append(parent)
        cursor = parent

    seen = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = os.path.abspath(os.path.join(root, path_value))
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(os.path.join(base_abs, path_value))


def _frame_count(nifti_path: str) -> int:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to validate NIfTI frame counts.") from exc

    img = nib.load(nifti_path)
    if img.ndim <= 3:
        return 1
    return int(img.shape[-1])


def _unique_sorted(values: Iterable[float], tol: float = 1e-3) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
    return out


def find_single_series(case_dir: Path, phase_vx_glob: str) -> Path:
    matches = sorted(path for path in case_dir.glob(phase_vx_glob) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"No series matching {phase_vx_glob!r} found under {case_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Expected one {phase_vx_glob!r} series under {case_dir}, found: {names}")
    return matches[0]


def read_unique_trigger_times(series_dir: Path) -> list[float]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required to read TriggerTime from DICOM.") from exc

    values: list[float] = []
    for path in sorted(series_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        trigger_time = getattr(ds, "TriggerTime", None)
        if trigger_time is None:
            continue
        try:
            values.append(float(trigger_time))
        except (TypeError, ValueError):
            continue

    unique_values = _unique_sorted(values)
    if not unique_values:
        raise RuntimeError(f"No TriggerTime values found in DICOM series {series_dir}")
    return unique_values


def normalize_trigger_times(trigger_times: Sequence[float]) -> list[float]:
    if not trigger_times:
        raise ValueError("TriggerTime list cannot be empty.")
    # DICOM TriggerTime values often carry tiny floating-point jitter between
    # studies. Stabilize pairing before normalization so physiologically equal
    # phases do not flip to the neighboring frame due to sub-millisecond noise.
    trigger_times = [round(float(value), TRIGGER_TIME_DECIMALS) for value in trigger_times]
    if len(trigger_times) == 1:
        return [0.0]
    start = float(trigger_times[0])
    end = float(trigger_times[-1])
    if end <= start:
        raise ValueError(f"TriggerTime values must be strictly increasing: {list(trigger_times)}")
    scale = end - start
    return [(float(value) - start) / scale for value in trigger_times]


def build_nearest_frame_mapping(lr_trigger_times: Sequence[float], hr_trigger_times: Sequence[float]) -> list[int]:
    lr_phases = normalize_trigger_times(lr_trigger_times)
    hr_phases = normalize_trigger_times(hr_trigger_times)
    if not hr_phases:
        raise ValueError("HR TriggerTime list cannot be empty.")

    mapping: list[int] = []
    for phase in lr_phases:
        best_index = min(range(len(hr_phases)), key=lambda idx: (abs(hr_phases[idx] - phase), idx))
        mapping.append(int(best_index))
    return mapping


def infer_case_key(row: dict[str, str], base_dir: str) -> str:
    lr_u = _resolve_path(row.get("lr_u", ""), base_dir)
    hr_u = _resolve_path(row.get("hr_u", ""), base_dir)
    lr_case = Path(lr_u).parent.name
    hr_case = Path(hr_u).parent.name
    if not lr_case:
        raise ValueError("Could not infer case key from lr_u path.")
    if hr_case and lr_case != hr_case:
        raise ValueError(f"CSV row mixes LR case {lr_case!r} and HR case {hr_case!r}.")
    return lr_case


def expand_case_row(
    row: dict[str, str],
    lr_trigger_times: Sequence[float],
    hr_trigger_times: Sequence[float],
) -> list[dict[str, str]]:
    hr_mapping = build_nearest_frame_mapping(lr_trigger_times=lr_trigger_times, hr_trigger_times=hr_trigger_times)
    expanded_rows: list[dict[str, str]] = []

    for lr_index, hr_index in enumerate(hr_mapping):
        expanded = dict(row)
        if "time_start" in expanded:
            expanded["time_start"] = ""
        if "time_end" in expanded:
            expanded["time_end"] = ""
        if "time_index" in expanded:
            expanded["time_index"] = str(lr_index)
        expanded["lr_time_index"] = str(lr_index)
        expanded["hr_time_index"] = str(hr_index)
        expanded["lr_trigger_time_ms"] = f"{round(float(lr_trigger_times[lr_index]), TRIGGER_TIME_DECIMALS):.3f}"
        expanded["hr_trigger_time_ms"] = f"{round(float(hr_trigger_times[hr_index]), TRIGGER_TIME_DECIMALS):.3f}"
        expanded["pairing_method"] = "trigger_time_nearest"
        expanded_rows.append(expanded)

    return expanded_rows


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    sorted_patients_root = Path(args.sorted_patients_root).resolve()
    csv_base_dir = str(input_csv.parent)

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise SystemExit(f"Input CSV has no header: {input_csv}")
    for column in NEW_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    expanded_rows: list[dict[str, str]] = []
    case_summaries: list[tuple[str, int, int, int]] = []

    for row in rows:
        case_key = infer_case_key(row, csv_base_dir)
        lr_case_dir = sorted_patients_root / f"{case_key}{args.lr_case_suffix}"
        hr_case_dir = sorted_patients_root / f"{case_key}{args.hr_case_suffix}"
        if not lr_case_dir.is_dir():
            raise SystemExit(f"Missing LR DICOM case folder: {lr_case_dir}")
        if not hr_case_dir.is_dir():
            raise SystemExit(f"Missing HR DICOM case folder: {hr_case_dir}")

        lr_series = find_single_series(lr_case_dir, args.phase_vx_glob)
        hr_series = find_single_series(hr_case_dir, args.phase_vx_glob)
        lr_trigger_times = read_unique_trigger_times(lr_series)
        hr_trigger_times = read_unique_trigger_times(hr_series)

        lr_frame_count = _frame_count(_resolve_path(row["lr_u"], csv_base_dir))
        hr_frame_count = _frame_count(_resolve_path(row["hr_u"], csv_base_dir))

        if len(lr_trigger_times) != lr_frame_count:
            raise SystemExit(
                f"{case_key}: 3T TriggerTime count ({len(lr_trigger_times)}) does not match "
                f"LR NIfTI frames ({lr_frame_count})."
            )
        if len(hr_trigger_times) != hr_frame_count:
            raise SystemExit(
                f"{case_key}: 7T TriggerTime count ({len(hr_trigger_times)}) does not match "
                f"HR NIfTI frames ({hr_frame_count})."
            )

        case_rows = expand_case_row(row=row, lr_trigger_times=lr_trigger_times, hr_trigger_times=hr_trigger_times)
        expanded_rows.extend(case_rows)
        case_summaries.append((case_key, len(lr_trigger_times), len(hr_trigger_times), len(case_rows)))

        if args.verbose:
            mapping = [int(case_row["hr_time_index"]) for case_row in case_rows]
            print(
                f"{case_key}: 3T={len(lr_trigger_times)} frame(s), "
                f"7T={len(hr_trigger_times)} frame(s), hr_mapping={mapping}"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded_rows)

    print(f"Wrote {len(expanded_rows)} paired frame row(s) to {output_csv}")
    for case_key, lr_count, hr_count, out_count in case_summaries:
        print(f"- {case_key}: 3T={lr_count}, 7T={hr_count}, output_rows={out_count}")


if __name__ == "__main__":
    main()
