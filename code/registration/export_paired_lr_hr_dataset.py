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


CSV_FIELDNAMES = [
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
    "venc_u",
    "venc_v",
    "venc_w",
    "time_start",
    "time_end",
    "time_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paired 3T (LR) and 7T->3T (HR target) datasets.")
    parser.add_argument("--temporal-dir", default=None, help="Root with temporally registered folders")
    parser.add_argument("--registered-dir", default=None, help="Root with final 7T->3T registered outputs")
    parser.add_argument("--output-root", required=True, help="Root where lr/hr paired folders are written")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only (re)generate the paired CSV from existing output folders, without copying/symlinking files again.",
    )
    parser.add_argument(
        "--existing-lr-root",
        default=None,
        help="When --csv-only is enabled, LR root to scan (default: <output-root>/lr_3t).",
    )
    parser.add_argument(
        "--existing-hr-root",
        default=None,
        help="When --csv-only is enabled, HR root to scan (default: <output-root>/hr_7t_in_3t).",
    )

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
    parser.add_argument(
        "--apply-hr-mask",
        action="store_true",
        help="Apply a binary mask to exported HR files (u/v/w/mag) before writing CSV rows.",
    )
    parser.add_argument(
        "--hr-mask-root",
        default=None,
        help="Root folder containing per-case masks (required with --apply-hr-mask).",
    )
    parser.add_argument(
        "--hr-mask-name",
        default="cow_seg_final.nii.gz",
        help="Mask filename expected under <hr-mask-root>/<case_key>/ (default: cow_seg_final.nii.gz).",
    )
    parser.add_argument(
        "--hr-mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize mask values.",
    )
    parser.add_argument(
        "--hr-mask-time-index",
        type=int,
        default=0,
        help="Mask time frame index when mask is 4D and HR volume is 3D.",
    )
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


def discover_existing_pairs(hr_root: str, lr_root: str, hr_u_name: str) -> list[PairCase]:
    """
    Discover already-exported paired case folders from existing LR/HR roots.
    We index cases from HR folders that contain `hr_u_name`.
    """
    hr_root_p = Path(hr_root)
    lr_root_p = Path(lr_root)
    pairs: list[PairCase] = []

    for hr_u_path in hr_root_p.rglob(hr_u_name):
        case_hr_dir = hr_u_path.parent
        try:
            rel_case = case_hr_dir.relative_to(hr_root_p)
        except ValueError:
            continue
        case_key = rel_case.as_posix()
        case_lr_dir = lr_root_p / rel_case
        pairs.append(PairCase(case_key=case_key, lr_dir=str(case_lr_dir), hr_dir=str(case_hr_dir)))

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


def _select_3d_mask(mask_data: np.ndarray, time_index: int) -> np.ndarray:
    import numpy as np

    if mask_data.ndim == 3:
        return mask_data
    if mask_data.ndim != 4:
        raise ValueError(f"Unsupported mask shape {mask_data.shape}: expected 3D or 4D.")
    if not (0 <= time_index < mask_data.shape[3]):
        raise ValueError(
            f"Invalid --hr-mask-time-index={time_index} for 4D mask shape={mask_data.shape}."
        )
    return np.asarray(mask_data[..., time_index], dtype=np.float32)


def _broadcast_mask(mask_3d: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    import numpy as np

    if len(target_shape) == 3:
        return mask_3d
    if len(target_shape) == 4:
        return np.broadcast_to(mask_3d[..., np.newaxis], target_shape)
    raise ValueError(f"Unsupported HR shape {target_shape}: expected 3D or 4D.")


def _load_case_mask(mask_path: str, hr_reference_path: str, threshold: float, time_index: int) -> np.ndarray:
    import nibabel as nib
    import nibabel.processing as nibproc
    import numpy as np

    mask_img = nib.load(mask_path)
    hr_ref_img = nib.load(hr_reference_path)
    mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)
    mask_data = _select_3d_mask(mask_data, time_index=time_index)
    mask_img_3d = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)

    same_shape = tuple(mask_img_3d.shape[:3]) == tuple(hr_ref_img.shape[:3])
    same_affine = np.allclose(mask_img_3d.affine, hr_ref_img.affine, atol=1e-4)
    if not (same_shape and same_affine):
        mask_img_3d = nibproc.resample_from_to(mask_img_3d, (hr_ref_img.shape[:3], hr_ref_img.affine), order=0)

    mask_3d = np.asarray(mask_img_3d.dataobj, dtype=np.float32)
    return (mask_3d >= float(threshold)).astype(np.float32)


def apply_mask_to_hr_file(hr_path: str, mask_3d: np.ndarray) -> None:
    import nibabel as nib
    import numpy as np

    hr_img = nib.load(hr_path)
    hr_data = np.asarray(hr_img.dataobj, dtype=np.float32)
    mask = _broadcast_mask(mask_3d, hr_data.shape)
    masked = (hr_data * mask).astype(np.float32)
    out = nib.Nifti1Image(masked, hr_img.affine, hr_img.header)
    nib.save(out, hr_path)


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


def make_csv_row(lr_paths: dict[str, str], hr_paths: dict[str, str], venc: float, time_end: int | None) -> dict[str, str]:
    venc_text = f"{venc}"
    return {
        "lr_u": os.path.relpath(lr_paths["u"]),
        "lr_v": os.path.relpath(lr_paths["v"]),
        "lr_w": os.path.relpath(lr_paths["w"]),
        "lr_mag_u": os.path.relpath(lr_paths["mag"]),
        "lr_mag_v": os.path.relpath(lr_paths["mag"]),
        "lr_mag_w": os.path.relpath(lr_paths["mag"]),
        "hr_u": os.path.relpath(hr_paths["u"]),
        "hr_v": os.path.relpath(hr_paths["v"]),
        "hr_w": os.path.relpath(hr_paths["w"]),
        "hr_mag": os.path.relpath(hr_paths["mag"]),
        "mask": "",
        "venc": venc_text,
        "venc_u": venc_text,
        "venc_v": venc_text,
        "venc_w": venc_text,
        "time_start": "0",
        "time_end": str(time_end) if time_end is not None else "",
        "time_index": "",
    }


def main() -> None:
    args = parse_args()
    lr_root = args.existing_lr_root or os.path.join(args.output_root, "lr_3t")
    hr_root = args.existing_hr_root or os.path.join(args.output_root, "hr_7t_in_3t")

    if args.csv_only:
        if not os.path.isdir(lr_root):
            raise SystemExit(f"LR root not found for --csv-only: {lr_root}")
        if not os.path.isdir(hr_root):
            raise SystemExit(f"HR root not found for --csv-only: {hr_root}")
        pairs = discover_existing_pairs(hr_root=hr_root, lr_root=lr_root, hr_u_name=args.out_u_name)
    else:
        if not args.temporal_dir or not args.registered_dir:
            raise SystemExit("--temporal-dir and --registered-dir are required unless --csv-only is set.")
        pairs = discover_pairs(
            temporal_dir=args.temporal_dir,
            registered_dir=args.registered_dir,
            fixed_suffix=args.fixed_suffix,
            moving_suffix=args.moving_suffix,
        )
        os.makedirs(lr_root, exist_ok=True)
        os.makedirs(hr_root, exist_ok=True)

    if args.apply_hr_mask and not args.hr_mask_root:
        raise SystemExit("--hr-mask-root is required when --apply-hr-mask is enabled.")
    if args.apply_hr_mask:
        try:
            import nibabel  # noqa: F401
            import numpy  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "--apply-hr-mask requires nibabel and numpy in the active environment."
            ) from exc
    mask_root = os.path.abspath(args.hr_mask_root) if args.hr_mask_root else None

    if not pairs:
        raise SystemExit("No paired cases found.")

    csv_path = args.csv_path or os.path.join(args.output_root, "paired_nifti_cases.csv")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows: list[dict[str, str]] = []
    success = 0
    failed = 0

    for pair in pairs:
        if args.csv_only:
            lr_paths = {
                "u": os.path.join(pair.lr_dir, args.out_u_name),
                "v": os.path.join(pair.lr_dir, args.out_v_name),
                "w": os.path.join(pair.lr_dir, args.out_w_name),
                "mag": os.path.join(pair.lr_dir, args.out_mag_name),
            }
            hr_paths = {
                "u": os.path.join(pair.hr_dir, args.out_u_name),
                "v": os.path.join(pair.hr_dir, args.out_v_name),
                "w": os.path.join(pair.hr_dir, args.out_w_name),
                "mag": os.path.join(pair.hr_dir, args.out_mag_name),
            }
        else:
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

            lr_case = os.path.join(lr_root, pair.case_key)
            hr_case = os.path.join(hr_root, pair.case_key)
            os.makedirs(lr_case, exist_ok=True)
            os.makedirs(hr_case, exist_ok=True)

            lr_paths = {
                "u": os.path.join(lr_case, args.out_u_name),
                "v": os.path.join(lr_case, args.out_v_name),
                "w": os.path.join(lr_case, args.out_w_name),
                "mag": os.path.join(lr_case, args.out_mag_name),
            }
            hr_paths = {
                "u": os.path.join(hr_case, args.out_u_name),
                "v": os.path.join(hr_case, args.out_v_name),
                "w": os.path.join(hr_case, args.out_w_name),
                "mag": os.path.join(hr_case, args.out_mag_name),
            }

        check_paths = {**{f"lr_{k}": v for k, v in lr_paths.items()}, **{f"hr_{k}": v for k, v in hr_paths.items()}}
        if not args.csv_only:
            check_paths = {**{f"lr_{k}": v for k, v in lr_src.items()}, **{f"hr_{k}": v for k, v in hr_src.items()}}
        missing = [name for name, path in check_paths.items() if not os.path.isfile(path)]
        if missing:
            failed += 1
            print(f"[FAIL] {pair.case_key}: missing {', '.join(missing)}")
            if args.strict:
                raise SystemExit(1)
            continue

        if not args.csv_only:
            for key in ("u", "v", "w", "mag"):
                transfer_file(lr_src[key], lr_paths[key], args.mode, args.overwrite)
                hr_mode = "copy" if args.apply_hr_mask else args.mode
                transfer_file(hr_src[key], hr_paths[key], hr_mode, args.overwrite)

        if args.apply_hr_mask:
            if args.csv_only:
                symlinked = [p for p in hr_paths.values() if os.path.islink(p)]
                if symlinked:
                    raise SystemExit(
                        "Refusing to apply mask in --csv-only mode because HR files are symlinks. "
                        "Use a copied dataset or rerun export without symlinks."
                    )

            mask_path = os.path.join(mask_root, pair.case_key, args.hr_mask_name)
            if not os.path.isfile(mask_path):
                failed += 1
                print(f"[FAIL] {pair.case_key}: missing hr mask at {mask_path}")
                if args.strict:
                    raise SystemExit(1)
                continue

            try:
                case_mask = _load_case_mask(
                    mask_path=mask_path,
                    hr_reference_path=hr_paths["u"],
                    threshold=args.hr_mask_threshold,
                    time_index=args.hr_mask_time_index,
                )
                for key in ("u", "v", "w", "mag"):
                    apply_mask_to_hr_file(hr_paths[key], case_mask)
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {pair.case_key}: hr mask application failed ({exc})")
                if args.strict:
                    raise SystemExit(1)
                continue

        time_end = resolve_time_end(lr_paths["u"], hr_paths["u"])
        rows.append(make_csv_row(lr_paths=lr_paths, hr_paths=hr_paths, venc=args.venc, time_end=time_end))

        success += 1
        if args.verbose:
            print(f"[OK] {pair.case_key}")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
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
