#!/usr/bin/env python3
"""Detect a common CoW ROI per 3T/7T patient pair and crop all pair files with the same bbox."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class CasePair:
    case_key: str
    fixed_dir: Path
    moving_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per patient pair (_3T/_7T), run YOLO on both magnitude volumes, build one common "
            "bbox (enclosing both), and crop all NIfTI files in both folders with that same bbox."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Root with paired folders (e.g., data/processed_inputs)")
    parser.add_argument("--output-dir", required=True, help="Root where cropped folders are written")
    parser.add_argument("--yolo-model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--yolo-device", default=None, help="YOLO device override (e.g., cpu, mps, cuda:0)")
    parser.add_argument("--fixed-suffix", default="_3T", help="Suffix for fixed folders")
    parser.add_argument("--moving-suffix", default="_7T", help="Suffix for moving folders")
    parser.add_argument("--magnitude-name", default="input_mag_raw.nii.gz", help="Magnitude filename in both folders")
    parser.add_argument(
        "--margin-xy",
        type=int,
        default=20,
        help="Legacy fallback margin for X/Y when --margin-x/--margin-y are not set",
    )
    parser.add_argument("--margin-x", type=int, default=None, help="Extra margin for X axis (overrides --margin-xy)")
    parser.add_argument("--margin-y", type=int, default=None, help="Extra margin for Y axis (overrides --margin-xy)")
    parser.add_argument(
        "--margin-x-side",
        choices=["both", "min", "max"],
        default="both",
        help="Where to apply X margin: both borders, only lower border (min), or only upper border (max)",
    )
    parser.add_argument(
        "--margin-y-side",
        choices=["both", "min", "max"],
        default="both",
        help="Where to apply Y margin: both borders, only lower border (min), or only upper border (max)",
    )
    parser.add_argument(
        "--margin-z",
        type=int,
        default=8,
        help="Deprecated (ignored). Z crop always keeps the full axis.",
    )
    parser.add_argument(
        "--shape-mismatch",
        choices=["error", "skip"],
        default="error",
        help="Behavior when 3T/7T magnitude 3D shapes differ",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Summary JSON path (default: <output-dir>/plans.json)",
    )
    parser.add_argument(
        "--write-folder-plans",
        action="store_true",
        help="Also write plans.json inside each cropped patient folder",
    )
    return parser.parse_args()


def discover_case_pairs(input_dir: Path, fixed_suffix: str, moving_suffix: str) -> list[CasePair]:
    pairs: list[CasePair] = []
    for root, dirs, _files in os.walk(input_dir):
        root_path = Path(root)
        for dirname in dirs:
            if fixed_suffix and not dirname.endswith(fixed_suffix):
                continue
            case_base = dirname[:-len(fixed_suffix)] if fixed_suffix else dirname
            moving_name = case_base + moving_suffix
            moving_dir = root_path / moving_name
            if not moving_dir.is_dir():
                continue
            fixed_dir = root_path / dirname
            rel_parent = fixed_dir.parent.relative_to(input_dir)
            case_key = str((rel_parent / case_base).as_posix()) if str(rel_parent) != "." else case_base
            pairs.append(CasePair(case_key=case_key, fixed_dir=fixed_dir, moving_dir=moving_dir))

    unique = {pair.case_key: pair for pair in pairs}
    return [unique[k] for k in sorted(unique)]


def maybe_mip_4d(data: np.ndarray) -> tuple[np.ndarray, bool]:
    if data.ndim == 4:
        return np.max(data, axis=-1), True
    if data.ndim == 3:
        return data, False
    raise ValueError(f"Unsupported NIfTI shape {data.shape}. Expected 3D or 4D.")


def detect_bbox_with_yolo(volume_3d: np.ndarray, model, yolo_device: str | None):
    bbox_centers: list[tuple[int, int, int]] = []
    bbox_widths: list[int] = []
    bbox_heights: list[int] = []
    slices_with_roi: list[int] = []

    for slice_idx in range(volume_3d.shape[2]):
        slice_data = volume_3d[:, :, slice_idx]
        smin = float(np.min(slice_data))
        smax = float(np.max(slice_data))
        if smax > smin:
            norm_slice = (slice_data - smin) / (smax - smin) * 255.0
        else:
            norm_slice = np.zeros_like(slice_data, dtype=np.float32)

        norm_slice_u8 = norm_slice.astype(np.uint8, copy=False)
        # Avoid OpenCV dependency: build 3-channel uint8 image directly.
        slice_rgb = np.repeat(norm_slice_u8[..., np.newaxis], 3, axis=2)
        results = model.predict(source=slice_rgb, verbose=False, device=yolo_device)

        for result in results:
            boxes = result.boxes.xyxy
            if boxes is None or len(boxes) == 0:
                continue
            slices_with_roi.append(slice_idx)
            for bbox in boxes:
                x_min, y_min, x_max, y_max = map(int, bbox.tolist())
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = max(0, x_max - x_min)
                height = max(0, y_max - y_min)
                bbox_centers.append((center_x, center_y, slice_idx))
                bbox_widths.append(width)
                bbox_heights.append(height)

    return bbox_centers, bbox_widths, bbox_heights, slices_with_roi


def compute_crop_dict(
    volume_shape: tuple[int, int, int],
    bbox_centers,
    bbox_widths,
    bbox_heights,
    slices_with_roi,
    margin_x: int,
    margin_y: int,
    margin_x_side: str,
    margin_y_side: str,
    margin_z: int,
) -> dict[str, object]:
    if not bbox_centers:
        return {
            "detected": False,
            "size": [int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])],
            "location": [0, 0, 0],
        }

    median_center_x = int(np.median([c[0] for c in bbox_centers]))
    median_center_y = int(np.median([c[1] for c in bbox_centers]))
    median_width = int(np.median(bbox_widths))
    median_height = int(np.median(bbox_heights))

    crop_x_min = max(0, median_center_x - median_width // 2)
    crop_x_max = min(volume_shape[1], median_center_x + median_width // 2)
    crop_y_min = max(0, median_center_y - median_height // 2)
    crop_y_max = min(volume_shape[0], median_center_y + median_height // 2)

    def apply_margin(min_v: int, max_v: int, limit: int, margin: int, side: str) -> tuple[int, int]:
        if side == "both":
            return max(0, min_v - margin), min(limit, max_v + margin)
        if side == "min":
            return max(0, min_v - margin), min(limit, max_v)
        if side == "max":
            return max(0, min_v), min(limit, max_v + margin)
        raise ValueError(f"Unsupported margin side: {side}")

    y_start, y_end = apply_margin(crop_y_min, crop_y_max, volume_shape[0], int(margin_y), margin_y_side)
    x_start, x_end = apply_margin(crop_x_min, crop_x_max, volume_shape[1], int(margin_x), margin_x_side)
    # Keep full Z extent; CoW ROI is only constrained in X/Y.
    _ = margin_z
    _ = slices_with_roi
    z_start = 0
    z_end = int(volume_shape[2])

    return {
        "detected": True,
        "size": [int(y_end - y_start), int(x_end - x_start), int(z_end - z_start)],
        "location": [int(y_start), int(x_start), int(z_start)],
    }


def crop_to_bounds(crop_dict: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    y0, x0, z0 = [int(v) for v in crop_dict["location"]]  # type: ignore[index]
    sy, sx, sz = [int(v) for v in crop_dict["size"]]  # type: ignore[index]
    return y0, y0 + sy, x0, x0 + sx, z0, z0 + sz


def merge_pair_crops(shape: tuple[int, int, int], crop_a: dict[str, object], crop_b: dict[str, object]) -> dict[str, object]:
    a_det = bool(crop_a.get("detected", False))
    b_det = bool(crop_b.get("detected", False))

    if not a_det and not b_det:
        return {
            "detected": False,
            "size": [int(shape[0]), int(shape[1]), int(shape[2])],
            "location": [0, 0, 0],
            "merge_mode": "full_volume_fallback",
        }
    if a_det and not b_det:
        out = dict(crop_a)
        out["merge_mode"] = "a_only"
        return out
    if b_det and not a_det:
        out = dict(crop_b)
        out["merge_mode"] = "b_only"
        return out

    ay0, ay1, ax0, ax1, az0, az1 = crop_to_bounds(crop_a)
    by0, by1, bx0, bx1, bz0, bz1 = crop_to_bounds(crop_b)

    y0 = max(0, min(ay0, by0))
    x0 = max(0, min(ax0, bx0))
    z0 = max(0, min(az0, bz0))
    y1 = min(shape[0], max(ay1, by1))
    x1 = min(shape[1], max(ax1, bx1))
    z1 = min(shape[2], max(az1, bz1))

    return {
        "detected": True,
        "size": [int(y1 - y0), int(x1 - x0), int(z1 - z0)],
        "location": [int(y0), int(x0), int(z0)],
        "merge_mode": "enclosing_union",
    }


def list_nifti_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.nii.gz"))
    files.extend(sorted(folder.glob("*.nii")))
    return sorted(set(files))


def crop_array(data: np.ndarray, crop_dict: dict[str, object]) -> np.ndarray:
    y0, x0, z0 = [int(v) for v in crop_dict["location"]]  # type: ignore[index]
    sy, sx, sz = [int(v) for v in crop_dict["size"]]  # type: ignore[index]
    return data[y0 : y0 + sy, x0 : x0 + sx, z0 : z0 + sz, ...]


def save_cropped_nifti(cropped: np.ndarray, out_path: Path, reference_img: nib.Nifti1Image, crop_location: list[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offset = np.array(crop_location, dtype=np.float64)
    affine_new = reference_img.affine.copy()
    affine_new[:3, 3] = reference_img.affine[:3, 3] + reference_img.affine[:3, :3] @ offset

    header = reference_img.header.copy()
    header.set_data_shape(cropped.shape)
    header.set_data_dtype(cropped.dtype)

    out_img = nib.Nifti1Image(cropped, affine_new, header)
    _qform, qcode = reference_img.get_qform(coded=True)
    _sform, scode = reference_img.get_sform(coded=True)
    out_img.set_qform(affine_new, int(qcode) if qcode is not None else 1)
    out_img.set_sform(affine_new, int(scode) if scode is not None else 1)
    nib.save(out_img, str(out_path))


def run_detection_for_mag(
    mag_path: Path,
    model,
    yolo_device: str | None,
    margin_x: int,
    margin_y: int,
    margin_x_side: str,
    margin_y_side: str,
    margin_z: int,
):
    img = nib.load(str(mag_path))
    data = np.asarray(img.dataobj)
    vol3d, used_mip = maybe_mip_4d(data)
    centers, widths, heights, slices = detect_bbox_with_yolo(vol3d, model=model, yolo_device=yolo_device)
    crop = compute_crop_dict(
        volume_shape=tuple(int(v) for v in vol3d.shape),
        bbox_centers=centers,
        bbox_widths=widths,
        bbox_heights=heights,
        slices_with_roi=slices,
        margin_x=margin_x,
        margin_y=margin_y,
        margin_x_side=margin_x_side,
        margin_y_side=margin_y_side,
        margin_z=margin_z,
    )
    return img, vol3d, used_mip, crop


def _resolve_magnitude(folder: Path, exact_name: str) -> Path | None:
    """Return the magnitude path for a folder.

    Tries the exact name first. If not found, falls back to discovering NIfTI
    files whose name contains 'magnitude' or 'mag' (case-insensitive) and picks
    the one with the most time frames so that single-frame anatomical references
    are never chosen over the actual 4D flow magnitude.
    """
    import nibabel as nib

    exact = folder / exact_name
    if exact.exists():
        return exact

    candidates = [
        p for p in sorted(folder.glob("*.nii.gz"))
        if any(tok in p.name.lower() for tok in ("magnitude", "mag"))
        and not any(tok in p.name.lower() for tok in ("phase", "vx", "vy", "vz"))
    ]
    if not candidates:
        return None

    def _nframes(p: Path) -> int:
        shape = nib.load(str(p)).shape
        return int(shape[3]) if len(shape) == 4 else 1

    candidates.sort(key=_nframes, reverse=True)
    chosen = candidates[0]
    print(f"  [auto-mag] {folder.name}: using {chosen.name}")
    return chosen


def main() -> None:
    args = parse_args()

    margin_x = args.margin_x if args.margin_x is not None else args.margin_xy
    margin_y = args.margin_y if args.margin_y is not None else args.margin_xy

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required. Install it in your environment before running this script.") from exc

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else (output_dir / "plans.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pairs = discover_case_pairs(input_dir, args.fixed_suffix, args.moving_suffix)
    if not pairs:
        raise SystemExit(
            f"No pairs found in {input_dir} with suffixes "
            f"'{args.fixed_suffix}' and '{args.moving_suffix}'."
        )

    model = YOLO(args.yolo_model)
    summary: list[dict[str, object]] = []

    for pair in pairs:
        fixed_mag = _resolve_magnitude(pair.fixed_dir, args.magnitude_name)
        moving_mag = _resolve_magnitude(pair.moving_dir, args.magnitude_name)
        if fixed_mag is None or moving_mag is None:
            print(f"[SKIP] {pair.case_key}: missing magnitude file(s).")
            summary.append(
                {
                    "case_key": pair.case_key,
                    "status": "skipped_missing_magnitude",
                    "fixed_magnitude": str(pair.fixed_dir / args.magnitude_name),
                    "moving_magnitude": str(pair.moving_dir / args.magnitude_name),
                }
            )
            continue

        print(f"\n=== {pair.case_key} ===")
        _fixed_img, fixed_3d, fixed_used_mip, bbox_fixed = run_detection_for_mag(
            fixed_mag,
            model=model,
            yolo_device=args.yolo_device,
            margin_x=margin_x,
            margin_y=margin_y,
            margin_x_side=args.margin_x_side,
            margin_y_side=args.margin_y_side,
            margin_z=args.margin_z,
        )
        _moving_img, moving_3d, moving_used_mip, bbox_moving = run_detection_for_mag(
            moving_mag,
            model=model,
            yolo_device=args.yolo_device,
            margin_x=margin_x,
            margin_y=margin_y,
            margin_x_side=args.margin_x_side,
            margin_y_side=args.margin_y_side,
            margin_z=args.margin_z,
        )

        fixed_shape = tuple(int(v) for v in fixed_3d.shape)
        moving_shape = tuple(int(v) for v in moving_3d.shape)
        if fixed_shape != moving_shape:
            msg = f"3D magnitude shape mismatch: {fixed_shape} vs {moving_shape}"
            if args.shape_mismatch == "error":
                raise SystemExit(f"[ERROR] {pair.case_key}: {msg}")
            print(f"[SKIP]  {pair.case_key}: {msg}")
            summary.append(
                {
                    "case_key": pair.case_key,
                    "status": "skipped_shape_mismatch",
                    "fixed_shape_3d": list(fixed_shape),
                    "moving_shape_3d": list(moving_shape),
                    "fixed_bbox": bbox_fixed,
                    "moving_bbox": bbox_moving,
                }
            )
            continue

        common_bbox = merge_pair_crops(fixed_shape, bbox_fixed, bbox_moving)
        print(f" fixed bbox : {bbox_fixed}")
        print(f" moving bbox: {bbox_moving}")
        print(f" common bbox: {common_bbox}")

        cropped_files: list[str] = []
        skipped_files: list[dict[str, str]] = []

        for folder in (pair.fixed_dir, pair.moving_dir):
            for src_path in list_nifti_files(folder):
                rel_path = src_path.relative_to(input_dir)
                dst_path = output_dir / rel_path

                src_img = nib.load(str(src_path))
                src_data = np.asarray(src_img.dataobj)
                if src_data.ndim < 3:
                    skipped_files.append({"file": str(rel_path), "reason": f"unsupported_ndim_{src_data.ndim}"})
                    continue
                if tuple(int(v) for v in src_data.shape[:3]) != fixed_shape:
                    skipped_files.append(
                        {
                            "file": str(rel_path),
                            "reason": (
                                f"spatial_shape_mismatch_{tuple(int(v) for v in src_data.shape[:3])}"
                                f"_expected_{fixed_shape}"
                            ),
                        }
                    )
                    continue

                cropped = crop_array(src_data, common_bbox)
                save_cropped_nifti(
                    cropped=cropped,
                    out_path=dst_path,
                    reference_img=src_img,
                    crop_location=[int(v) for v in common_bbox["location"]],  # type: ignore[index]
                )
                cropped_files.append(str(rel_path))

            if args.write_folder_plans:
                out_folder = output_dir / folder.relative_to(input_dir)
                out_folder.mkdir(parents=True, exist_ok=True)
                with (out_folder / "plans.json").open("w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "case_key": pair.case_key,
                            "folder": str(folder.relative_to(input_dir)),
                            "bbox": common_bbox,
                        },
                        pf,
                        indent=2,
                    )

        summary.append(
            {
                "case_key": pair.case_key,
                "status": "ok",
                "fixed_folder": str(pair.fixed_dir),
                "moving_folder": str(pair.moving_dir),
                "magnitude_file": args.magnitude_name,
                "margin_x": int(margin_x),
                "margin_y": int(margin_y),
                "margin_x_side": args.margin_x_side,
                "margin_y_side": args.margin_y_side,
                "fixed_shape_3d": list(fixed_shape),
                "fixed_used_mip_from_4d": bool(fixed_used_mip),
                "moving_used_mip_from_4d": bool(moving_used_mip),
                "fixed_bbox": bbox_fixed,
                "moving_bbox": bbox_moving,
                "common_bbox": common_bbox,
                "cropped_file_count": len(cropped_files),
                "skipped_file_count": len(skipped_files),
                "skipped_files": skipped_files,
            }
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {output_json}")


if __name__ == "__main__":
    main()
