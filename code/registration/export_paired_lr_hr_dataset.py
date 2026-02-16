#!/usr/bin/env python3
"""Export paired LR/HR NIfTI folders plus CSV for training/prediction workflows."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairCase:
    case_key: str
    lr_dir: str
    hr_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paired 3T (LR) and 7T->3T (HR target) datasets.")
    parser.add_argument("--temporal-dir", required=True, help="Root with temporally registered folders")
    parser.add_argument("--registered-dir", required=True, help="Root with final 7T->3T registered outputs")
    parser.add_argument("--output-root", required=True, help="Root where lr/hr paired folders are written")

    parser.add_argument("--fixed-suffix", default="_3T", help="Suffix for LR folders in temporal-dir")
    parser.add_argument("--moving-suffix", default="_7T", help="Suffix for HR source folders in temporal-dir")

    parser.add_argument("--lr-u-name", default="Vx.nii.gz")
    parser.add_argument("--lr-v-name", default="Vy.nii.gz")
    parser.add_argument("--lr-w-name", default="Vz.nii.gz")
    parser.add_argument("--lr-mag-name", default="input_mag_raw.nii.gz")

    parser.add_argument("--hr-u-name", default="phaseX_7T_in_3T.nii.gz")
    parser.add_argument("--hr-v-name", default="phaseY_7T_in_3T.nii.gz")
    parser.add_argument("--hr-w-name", default="phaseZ_7T_in_3T.nii.gz")
    parser.add_argument("--hr-mag-name", default="mag_7T_in_3T.nii.gz")

    parser.add_argument("--out-u-name", default="Vx.nii.gz", help="Filename used in exported lr/hr folders")
    parser.add_argument("--out-v-name", default="Vy.nii.gz", help="Filename used in exported lr/hr folders")
    parser.add_argument("--out-w-name", default="Vz.nii.gz", help="Filename used in exported lr/hr folders")
    parser.add_argument("--out-mag-name", default="input_mag_raw.nii.gz", help="Filename used in exported lr/hr folders")

    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on first missing case/file")

    parser.add_argument("--csv-path", default=None, help="Output CSV for NIfTI paired cases")
    parser.add_argument("--venc", type=float, default=0.9, help="VENC value stored in CSV")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def discover_pairs(temporal_dir: str, registered_dir: str, fixed_suffix: str, moving_suffix: str) -> list[PairCase]:
    pairs: list[PairCase] = []
    seen: set[str] = set()

    for root, dirs, _files in os.walk(temporal_dir):
        for dirname in dirs:
            if fixed_suffix and not dirname.endswith(fixed_suffix):
                continue

            case_base = dirname[:-len(fixed_suffix)] if fixed_suffix else dirname
            moving_name = case_base + moving_suffix

            lr_dir = os.path.join(root, dirname)
            moving_dir = os.path.join(root, moving_name)
            if not os.path.isdir(moving_dir):
                continue

            rel_parent = os.path.relpath(root, temporal_dir)
            case_key = case_base if rel_parent == "." else os.path.join(rel_parent, case_base)
            if case_key in seen:
                continue

            hr_dir = os.path.join(registered_dir, case_key)
            pairs.append(PairCase(case_key=case_key, lr_dir=lr_dir, hr_dir=hr_dir))
            seen.add(case_key)

    return sorted(pairs, key=lambda item: item.case_key)


def safe_remove(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    os.remove(path)


def transfer_file(src: str, dst: str, mode: str, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if not overwrite:
            return
        safe_remove(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def resolve_time_end(lr_u: str, hr_u: str) -> int | None:
    try:
        import nibabel as nib
    except ImportError:
        return None

    def frame_count(path: str) -> int:
        img = nib.load(path)
        if img.ndim <= 3:
            return 1
        return int(img.shape[-1])

    try:
        return min(frame_count(lr_u), frame_count(hr_u)) - 1
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    pairs = discover_pairs(
        temporal_dir=args.temporal_dir,
        registered_dir=args.registered_dir,
        fixed_suffix=args.fixed_suffix,
        moving_suffix=args.moving_suffix,
    )
    if not pairs:
        raise SystemExit("No paired cases found to export.")

    lr_root = os.path.join(args.output_root, "lr_3t")
    hr_root = os.path.join(args.output_root, "hr_7t_in_3t")
    os.makedirs(lr_root, exist_ok=True)
    os.makedirs(hr_root, exist_ok=True)

    csv_path = args.csv_path or os.path.join(args.output_root, "paired_nifti_cases.csv")
    rows: list[dict[str, str]] = []
    success = 0
    failed = 0

    for pair in pairs:
        lr_src = {
            "u": os.path.join(pair.lr_dir, args.lr_u_name),
            "v": os.path.join(pair.lr_dir, args.lr_v_name),
            "w": os.path.join(pair.lr_dir, args.lr_w_name),
            "mag": os.path.join(pair.lr_dir, args.lr_mag_name),
        }
        hr_src = {
            "u": os.path.join(pair.hr_dir, args.hr_u_name),
            "v": os.path.join(pair.hr_dir, args.hr_v_name),
            "w": os.path.join(pair.hr_dir, args.hr_w_name),
            "mag": os.path.join(pair.hr_dir, args.hr_mag_name),
        }

        missing = [name for name, path in {**{f"lr_{k}": v for k, v in lr_src.items()}, **{f"hr_{k}": v for k, v in hr_src.items()}}.items() if not os.path.isfile(path)]
        if missing:
            failed += 1
            print(f"[FAIL] {pair.case_key}: missing {', '.join(missing)}")
            if args.strict:
                raise SystemExit(1)
            continue

        lr_case = os.path.join(lr_root, pair.case_key)
        hr_case = os.path.join(hr_root, pair.case_key)
        os.makedirs(lr_case, exist_ok=True)
        os.makedirs(hr_case, exist_ok=True)

        lr_dst = {
            "u": os.path.join(lr_case, args.out_u_name),
            "v": os.path.join(lr_case, args.out_v_name),
            "w": os.path.join(lr_case, args.out_w_name),
            "mag": os.path.join(lr_case, args.out_mag_name),
        }
        hr_dst = {
            "u": os.path.join(hr_case, args.out_u_name),
            "v": os.path.join(hr_case, args.out_v_name),
            "w": os.path.join(hr_case, args.out_w_name),
            "mag": os.path.join(hr_case, args.out_mag_name),
        }

        for key in ("u", "v", "w", "mag"):
            transfer_file(lr_src[key], lr_dst[key], args.mode, args.overwrite)
            transfer_file(hr_src[key], hr_dst[key], args.mode, args.overwrite)

        time_end = resolve_time_end(lr_dst["u"], hr_dst["u"])
        rows.append(
            {
                "lr_u": os.path.relpath(lr_dst["u"]),
                "lr_v": os.path.relpath(lr_dst["v"]),
                "lr_w": os.path.relpath(lr_dst["w"]),
                "lr_mag_u": os.path.relpath(lr_dst["mag"]),
                "lr_mag_v": os.path.relpath(lr_dst["mag"]),
                "lr_mag_w": os.path.relpath(lr_dst["mag"]),
                "hr_u": os.path.relpath(hr_dst["u"]),
                "hr_v": os.path.relpath(hr_dst["v"]),
                "hr_w": os.path.relpath(hr_dst["w"]),
                "hr_mag": os.path.relpath(hr_dst["mag"]),
                "mask": "",
                "venc": f"{args.venc}",
                "time_start": "0",
                "time_end": str(time_end) if time_end is not None else "",
            }
        )

        success += 1
        if args.verbose:
            print(f"[OK] {pair.case_key}")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "lr_u",
            "lr_v",
            "lr_w",
            "lr_mag_u",
            "lr_mag_v",
            "lr_mag_w",
            "hr_u",
            "hr_v",
            "hr_w",
            "hr_mag",
            "mask",
            "venc",
            "time_start",
            "time_end",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nPaired export complete")
    print(f"Cases found:   {len(pairs)}")
    print(f"Cases exported:{success}")
    print(f"Cases failed:  {failed}")
    print(f"LR folder:     {lr_root}")
    print(f"HR folder:     {hr_root}")
    print(f"CSV:           {csv_path}")

    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
