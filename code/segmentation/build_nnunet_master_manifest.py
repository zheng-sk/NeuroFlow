#!/usr/bin/env python3
"""Build a case-level master manifest for nnU-Net CoW experiments on 3T inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a case-level manifest for nnU-Net segmentation experiments using 3T inputs "
            "and registered 7T-derived CoW masks as ground truth."
        )
    )
    parser.add_argument("--csv-in", required=True, help="Input paired CSV (one or more rows per case).")
    parser.add_argument("--csv-out", required=True, help="Output master manifest CSV.")
    parser.add_argument("--lr-mag-col", default="lr_mag_u", help="CSV column used as primary 3T magnitude input.")
    parser.add_argument("--lr-u-col", default="lr_u", help="CSV column for 3T velocity X.")
    parser.add_argument("--lr-v-col", default="lr_v", help="CSV column for 3T velocity Y.")
    parser.add_argument("--lr-w-col", default="lr_w", help="CSV column for 3T velocity Z.")
    parser.add_argument("--mask-col", default="mask", help="CSV column for GT mask path.")
    parser.add_argument(
        "--case-id-col",
        default="",
        help="Optional CSV column containing an explicit case id. If omitted, infer from paths.",
    )
    parser.add_argument(
        "--case-from-col",
        default="hr_u",
        help="CSV column used to infer case id when --case-id-col is absent.",
    )
    parser.add_argument(
        "--root-token",
        default="hr_7t_in_3t",
        help="Folder token used to recover relative case id from path-based inference.",
    )
    parser.add_argument(
        "--masks-root",
        default="",
        help=(
            "Optional root folder containing per-case masks. "
            "When provided, <masks-root>/<case_id>/<mask-name> is preferred over the CSV mask path."
        ),
    )
    parser.add_argument(
        "--mask-name",
        default="cow_seg_final.nii.gz",
        help="Mask filename expected under <masks-root>/<case_id>/ when inferring missing masks.",
    )
    parser.add_argument(
        "--mask-time-index",
        type=int,
        default=0,
        help="Frame index to inspect when mask NIfTI is 4D.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative-to-csv-out", "relative-to-cwd"],
        default="relative-to-csv-out",
        help="How paths are written to the output CSV.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing files or inconsistent duplicate rows.")
    parser.add_argument("--verbose", action="store_true", help="Print per-case details.")
    return parser.parse_args()


def infer_case_id(path_value: str, root_token: str) -> str:
    path = Path(path_value.strip())
    if not path.parts:
        return ""
    parent = path.parent
    parts = parent.parts
    if root_token in parts:
        idx = parts.index(root_token)
        rel = parts[idx + 1 :]
        if rel:
            return str(Path(*rel))
    return parent.name


def resolve_path(value: str, csv_parent: Path) -> Path:
    raw_value = (value or "").strip()
    if not raw_value:
        return Path("")
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (csv_parent / path).resolve()


def format_path(path: Path, mode: str, csv_out: Path) -> str:
    path = path.resolve()
    if mode == "absolute":
        return str(path)
    if mode == "relative-to-cwd":
        return os.path.relpath(path, start=Path.cwd().resolve())
    return os.path.relpath(path, start=csv_out.resolve().parent)


def select_mask_3d(mask_data, time_index: int):
    import numpy as np

    if mask_data.ndim == 3:
        return np.asarray(mask_data, dtype=np.float32)
    if mask_data.ndim != 4:
        raise ValueError(f"Unsupported mask ndim={mask_data.ndim}; expected 3D or 4D.")
    if not (0 <= time_index < mask_data.shape[3]):
        raise ValueError(f"Invalid mask time index {time_index} for mask shape {mask_data.shape}.")
    return np.asarray(mask_data[..., time_index], dtype=np.float32)


def inspect_case(lr_mag: Path, mask_path: Path, mask_time_index: int) -> dict[str, str]:
    import nibabel as nib
    import numpy as np

    mag_img = nib.load(str(lr_mag))
    mag_shape = tuple(int(x) for x in mag_img.shape)
    mag_spatial_shape = tuple(int(x) for x in mag_img.shape[:3])
    mag_spacing = tuple(float(x) for x in mag_img.header.get_zooms()[:3])

    mask_img = nib.load(str(mask_path))
    mask_shape_raw = tuple(int(x) for x in mask_img.shape)
    mask_3d = select_mask_3d(np.asarray(mask_img.dataobj), time_index=mask_time_index)
    mask_spatial_shape = tuple(int(x) for x in mask_3d.shape[:3])
    same_shape = mask_spatial_shape == mag_spatial_shape
    same_affine = np.allclose(mask_img.affine[:3, :], mag_img.affine[:3, :], atol=1e-4)

    num_frames = 1 if len(mag_shape) <= 3 else int(mag_shape[3])
    return {
        "lr_shape": json.dumps(mag_shape),
        "lr_spatial_shape": json.dumps(mag_spatial_shape),
        "mask_shape_raw": json.dumps(mask_shape_raw),
        "mask_spatial_shape": json.dumps(mask_spatial_shape),
        "num_frames": str(num_frames),
        "spacing_xyz": json.dumps([round(x, 6) for x in mag_spacing]),
        "shape_match": "1" if same_shape else "0",
        "affine_match": "1" if same_affine else "0",
        "ready_for_nnunet": "1" if (same_shape and same_affine) else "0",
    }


def main() -> int:
    args = parse_args()
    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    csv_parent = csv_in.resolve().parent
    masks_root = Path(args.masks_root).resolve() if args.masks_root else None

    if not csv_in.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")

    with csv_in.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_in}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required_cols = [args.lr_mag_col, args.lr_u_col, args.lr_v_col, args.lr_w_col]
    for col in required_cols:
        if col not in fieldnames:
            raise KeyError(f"Required column not found: {col}")
    if args.mask_col not in fieldnames and not masks_root:
        raise KeyError(f"Mask column `{args.mask_col}` not found and --masks-root was not provided.")
    if args.case_id_col and args.case_id_col not in fieldnames:
        raise KeyError(f"Explicit case id column not found: {args.case_id_col}")
    if not args.case_id_col and args.case_from_col not in fieldnames:
        raise KeyError(f"Case inference column not found: {args.case_from_col}")

    case_rows: dict[str, dict[str, str]] = {}
    duplicate_counts: dict[str, int] = {}
    missing_cases: list[str] = []
    not_ready_cases: list[str] = []

    out_fields = [
        "case_id",
        "source_rows",
        "lr_mag",
        "lr_u",
        "lr_v",
        "lr_w",
        "mask_gt",
        "lr_shape",
        "lr_spatial_shape",
        "mask_shape_raw",
        "mask_spatial_shape",
        "num_frames",
        "spacing_xyz",
        "shape_match",
        "affine_match",
        "ready_for_nnunet",
    ]

    for row_idx, row in enumerate(rows, start=1):
        if args.case_id_col:
            case_id = (row.get(args.case_id_col) or "").strip()
        else:
            case_id = infer_case_id((row.get(args.case_from_col) or "").strip(), args.root_token)
        if not case_id:
            case_id = f"row_{row_idx:04d}"

        lr_mag = resolve_path(row.get(args.lr_mag_col, ""), csv_parent)
        lr_u = resolve_path(row.get(args.lr_u_col, ""), csv_parent)
        lr_v = resolve_path(row.get(args.lr_v_col, ""), csv_parent)
        lr_w = resolve_path(row.get(args.lr_w_col, ""), csv_parent)
        mask_value = row.get(args.mask_col, "") if args.mask_col in row else ""
        csv_mask_path = resolve_path(mask_value, csv_parent) if mask_value else Path("")
        mask_path = csv_mask_path

        if masks_root is not None:
            candidate = (masks_root / case_id / args.mask_name).resolve()
            if candidate.is_file():
                mask_path = candidate

        paths = {
            "lr_mag": lr_mag,
            "lr_u": lr_u,
            "lr_v": lr_v,
            "lr_w": lr_w,
            "mask_gt": mask_path,
        }
        missing = [name for name, path in paths.items() if str(path) == "." or not path.is_file()]
        if missing:
            missing_cases.append(case_id)
            if args.verbose:
                print(f"[MISS] {case_id}: missing {', '.join(missing)}")
            if args.strict:
                raise FileNotFoundError(f"{case_id}: missing files: {', '.join(missing)}")
            continue

        inspected = inspect_case(lr_mag=lr_mag, mask_path=mask_path, mask_time_index=args.mask_time_index)
        if inspected["ready_for_nnunet"] != "1":
            not_ready_cases.append(case_id)

        current = {
            "case_id": case_id,
            "source_rows": str(row_idx),
            "lr_mag": format_path(lr_mag, args.path_mode, csv_out),
            "lr_u": format_path(lr_u, args.path_mode, csv_out),
            "lr_v": format_path(lr_v, args.path_mode, csv_out),
            "lr_w": format_path(lr_w, args.path_mode, csv_out),
            "mask_gt": format_path(mask_path, args.path_mode, csv_out),
            **inspected,
        }

        if case_id in case_rows:
            duplicate_counts[case_id] = duplicate_counts.get(case_id, 1) + 1
            prev = case_rows[case_id]
            compare_keys = ["lr_mag", "lr_u", "lr_v", "lr_w", "mask_gt"]
            if any(prev[key] != current[key] for key in compare_keys):
                message = f"{case_id}: duplicate case rows resolve to different files."
                if args.strict:
                    raise ValueError(message)
                if args.verbose:
                    print(f"[WARN] {message} Keeping first occurrence.")
                continue
            prev["source_rows"] = f"{prev['source_rows']};{row_idx}"
            continue

        case_rows[case_id] = current
        duplicate_counts.setdefault(case_id, 1)

        if args.verbose:
            status = "READY" if current["ready_for_nnunet"] == "1" else "CHECK"
            print(f"[{status}] {case_id} | frames={current['num_frames']} | lr_mag={lr_mag.name}")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    ordered_cases = [case_rows[key] for key in sorted(case_rows)]
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(ordered_cases)

    print("nnU-Net master manifest complete")
    print(f"Input CSV:         {csv_in}")
    print(f"Output CSV:        {csv_out}")
    print(f"Cases exported:    {len(ordered_cases)}")
    print(f"Missing cases:     {len(set(missing_cases))}")
    print(f"Need alignment:    {len(set(not_ready_cases))}")
    dup_case_count = sum(1 for v in duplicate_counts.values() if v > 1)
    print(f"Duplicate cases:   {dup_case_count}")

    if not_ready_cases:
        print("Cases requiring shape/affine review:")
        for case_id in sorted(set(not_ready_cases)):
            print(f"  - {case_id}")

    if missing_cases and args.strict:
        return 1
    if not_ready_cases and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
