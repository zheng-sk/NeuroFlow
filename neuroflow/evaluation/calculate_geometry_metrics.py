#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute geometry metrics between a predicted CoW segmentation and the "
            "reference 7T mask. If needed, masks are aligned by front-cropping, "
            "following the same convention used in other repo metrics."
        )
    )
    p.add_argument("--pred-mask", required=True, help="Path to predicted CoW mask, e.g. cow_seg_final.nii.gz")
    p.add_argument("--ref-mask", default="", help="Path to reference 7T mask. If omitted, resolved from --metadata-json.")
    p.add_argument(
        "--metadata-json",
        default="",
        help="Optional inference_metadata.json used to resolve case_paths.mask when --ref-mask is omitted.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory for geometry metrics. Default: <pred_mask_parent>/metrics/geometry",
    )
    p.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize both predicted and reference masks.",
    )
    return p.parse_args()


def _summary_stats_single(value: float) -> dict[str, float]:
    return {"mean": float(value), "std": 0.0, "min": float(value), "max": float(value)}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_ref_mask_path(ref_mask_arg: str, metadata_json_arg: str) -> Path:
    if str(ref_mask_arg).strip():
        ref_path = Path(ref_mask_arg).expanduser().resolve()
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference mask not found: {ref_path}")
        return ref_path

    if not str(metadata_json_arg).strip():
        raise ValueError("Provide either --ref-mask or --metadata-json.")

    meta_path = Path(metadata_json_arg).expanduser().resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

    meta = _load_json(meta_path)
    mask_path = str(((meta.get("case_paths") or {}).get("mask") or "")).strip()
    if not mask_path:
        raise ValueError(f"Metadata JSON does not contain case_paths.mask: {meta_path}")

    ref_path = Path(mask_path).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference mask from metadata not found: {ref_path}")
    return ref_path


def _load_mask_image(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
        header = img.header.copy()
        header.set_data_shape(data.shape)
        header.set_zooms(img.header.get_zooms()[:3])
        img = nib.Nifti1Image(data.astype(np.float32), img.affine, header)
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D or 4D mask NIfTI, got shape={img.shape} for {path}")
    return img


def _align_masks_by_front_crop(
    pred: np.ndarray,
    ref: np.ndarray,
    pred_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred_xyz = tuple(int(v) for v in pred.shape[:3])
    ref_xyz = tuple(int(v) for v in ref.shape[:3])
    target_xyz = (
        min(pred_xyz[0], ref_xyz[0]),
        min(pred_xyz[1], ref_xyz[1]),
        min(pred_xyz[2], ref_xyz[2]),
    )
    if min(target_xyz) <= 0:
        raise ValueError(f"Invalid shapes for front-crop alignment: pred={pred_xyz}, ref={ref_xyz}")

    pred_aligned = pred[: target_xyz[0], : target_xyz[1], : target_xyz[2]]
    ref_aligned = ref[: target_xyz[0], : target_xyz[1], : target_xyz[2]]

    info = {
        "pred_xyz_original": list(pred_xyz),
        "ref_xyz_original": list(ref_xyz),
        "aligned_xyz": list(target_xyz),
        "front_cropped": bool(pred_xyz != target_xyz or ref_xyz != target_xyz),
        "same_affine_within_tol": bool(np.allclose(pred_img.affine, ref_img.affine, atol=1e-4)),
        "pred_spacing_mm_original": [float(v) for v in pred_img.header.get_zooms()[:3]],
        "ref_spacing_mm": [float(v) for v in ref_img.header.get_zooms()[:3]],
    }
    return pred_aligned, ref_aligned, info


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return float((2.0 * inter) / (a_sum + b_sum))


def _finite_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _finite_std(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _surface_distance_metrics(mask_a: np.ndarray, mask_b: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, float]:
    if int(mask_a.sum()) < 10 or int(mask_b.sum()) < 10:
        return {
            "mean_surface_distance_a_to_b_mm": float("nan"),
            "std_surface_distance_a_to_b_mm": float("nan"),
            "symmetric_mean_surface_distance_mm": float("nan"),
            "hausdorff_distance_mm": float("nan"),
        }

    verts_a, _, _, _ = marching_cubes(mask_a.astype(np.float32), level=0.5, spacing=spacing_mm)
    verts_b, _, _, _ = marching_cubes(mask_b.astype(np.float32), level=0.5, spacing=spacing_mm)
    if verts_a.shape[0] == 0 or verts_b.shape[0] == 0:
        return {
            "mean_surface_distance_a_to_b_mm": float("nan"),
            "std_surface_distance_a_to_b_mm": float("nan"),
            "symmetric_mean_surface_distance_mm": float("nan"),
            "hausdorff_distance_mm": float("nan"),
        }

    tree_a = cKDTree(verts_a)
    tree_b = cKDTree(verts_b)
    dist_a_to_b = tree_b.query(verts_a, k=1)[0]
    dist_b_to_a = tree_a.query(verts_b, k=1)[0]

    mean_a_to_b = _finite_mean(dist_a_to_b)
    mean_b_to_a = _finite_mean(dist_b_to_a)
    return {
        "mean_surface_distance_a_to_b_mm": mean_a_to_b,
        "std_surface_distance_a_to_b_mm": _finite_std(dist_a_to_b),
        "symmetric_mean_surface_distance_mm": float(0.5 * (mean_a_to_b + mean_b_to_a)),
        "hausdorff_distance_mm": float(max(np.max(dist_a_to_b), np.max(dist_b_to_a))),
    }


def main() -> None:
    args = _parse_args()

    pred_mask_path = Path(args.pred_mask).expanduser().resolve()
    if not pred_mask_path.is_file():
        raise FileNotFoundError(f"Predicted mask not found: {pred_mask_path}")

    ref_mask_path = _resolve_ref_mask_path(args.ref_mask, args.metadata_json)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (pred_mask_path.parent / "metrics" / "geometry")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_img = _load_mask_image(pred_mask_path)
    ref_img = _load_mask_image(ref_mask_path)

    pred_raw = (np.asarray(pred_img.dataobj, dtype=np.float32) >= float(args.mask_threshold)).astype(np.uint8)
    ref_raw = (np.asarray(ref_img.dataobj, dtype=np.float32) >= float(args.mask_threshold)).astype(np.uint8)
    pred, ref, align_info = _align_masks_by_front_crop(pred_raw, ref_raw, pred_img, ref_img)

    pred_voxels = int(pred.sum())
    ref_voxels = int(ref.sum())
    intersection_voxels = int(np.logical_and(pred > 0, ref > 0).sum())
    spacing_mm = tuple(float(v) for v in ref_img.header.get_zooms()[:3])

    dice = _dice_score(pred, ref)
    surface = _surface_distance_metrics(pred, ref, spacing_mm)

    row = {
        "pred_mask_path": str(pred_mask_path),
        "ref_mask_path": str(ref_mask_path),
        "mask_threshold": float(args.mask_threshold),
        "pred_voxels": pred_voxels,
        "ref_voxels": ref_voxels,
        "intersection_voxels": intersection_voxels,
        "dice_score": float(dice),
        "mean_surface_distance_a_to_b_mm": float(surface["mean_surface_distance_a_to_b_mm"]),
        "std_surface_distance_a_to_b_mm": float(surface["std_surface_distance_a_to_b_mm"]),
        "symmetric_mean_surface_distance_mm": float(surface["symmetric_mean_surface_distance_mm"]),
        "hausdorff_distance_mm": float(surface["hausdorff_distance_mm"]),
    }

    summary = {
        "pred_mask_path": str(pred_mask_path),
        "ref_mask_path": str(ref_mask_path),
        "mask_threshold": float(args.mask_threshold),
        "alignment": align_info,
        "pred_voxels": pred_voxels,
        "ref_voxels": ref_voxels,
        "intersection_voxels": intersection_voxels,
        "dice_score": _summary_stats_single(float(dice)),
        "mean_surface_distance_a_to_b_mm": _summary_stats_single(float(surface["mean_surface_distance_a_to_b_mm"])),
        "std_surface_distance_a_to_b_mm": _summary_stats_single(float(surface["std_surface_distance_a_to_b_mm"])),
        "symmetric_mean_surface_distance_mm": _summary_stats_single(float(surface["symmetric_mean_surface_distance_mm"])),
        "hausdorff_distance_mm": _summary_stats_single(float(surface["hausdorff_distance_mm"])),
        "range_note": (
            "Geometry metrics are computed between the re-segmented prediction mask and the original 7T mask. "
            "If needed, masks are aligned by front-cropping to the common spatial extent."
        ),
    }

    single_csv = out_dir / "geometry_metrics_single.csv"
    with single_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    compact_csv = out_dir / "geometry_metrics_compact.csv"
    compact_rows = [
        {"metric": "dice_score", "value": float(dice)},
        {"metric": "hausdorff_distance_mm", "value": float(surface["hausdorff_distance_mm"])},
        {"metric": "symmetric_mean_surface_distance_mm", "value": float(surface["symmetric_mean_surface_distance_mm"])},
    ]
    with compact_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(compact_rows)

    summary_json = out_dir / "geometry_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved single-row metrics: {single_csv}")
    print(f"Saved compact metrics:    {compact_csv}")
    print(f"Saved summary metrics:    {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
