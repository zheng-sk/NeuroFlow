#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np


def ensure_time_last(data: np.ndarray, time_axis: int = -1) -> np.ndarray:
    if data.ndim == 3:
        return data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D/4D, got {data.ndim}D with shape={data.shape}")

    resolved = time_axis if time_axis >= 0 else data.ndim + time_axis
    if resolved < 0 or resolved >= data.ndim:
        raise ValueError(f"Invalid time_axis={time_axis} for shape={data.shape}")
    if resolved != 3:
        data = np.moveaxis(data, resolved, 3)
    return data


def convert_raw_to_velocity_repo_style(
    data: np.ndarray, venc: float, invert_sign: bool = False
) -> np.ndarray:
    arr = data.astype(np.float32, copy=False)
    mn, mx = float(np.min(arr)), float(np.max(arr))

    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (
        (mn < -500.0)
        and (mx > 500.0)
        and (max_abs <= 8192.0)
        and (centered_ratio < 0.25)
    )

    converted = arr
    if (is_unsigned_raw or is_signed_raw) and (venc is not None):
        if is_unsigned_raw:
            converted = (arr - 2048.0) / 2048.0 * float(venc)
        else:
            scale = 4096.0 if max_abs > 3000.0 else 2048.0
            converted = arr / scale * float(venc)

        if invert_sign:
            converted = -converted

    return converted.astype(np.float32, copy=False)


def robust_norm_in_mask(
    vol: np.ndarray, mask3d: np.ndarray, p_lo: float = 1, p_hi: float = 99, eps: float = 1e-8
) -> np.ndarray:
    vals = vol[mask3d > 0]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip(vol, lo, hi)
    out = (clipped - lo) / (hi - lo + eps)
    out = out * mask3d
    return out.astype(np.float32)


def save_3d_with_ref(vol: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    header = ref_img.header.copy()
    header.set_data_shape(vol.shape)
    header.set_zooms(ref_img.header.get_zooms()[:3])
    out_img = nib.Nifti1Image(vol.astype(np.float32), ref_img.affine, header)
    nib.save(out_img, str(out_path))


def load_mask_aligned(mask_path: Path, ref_img: nib.Nifti1Image, threshold: float) -> np.ndarray:
    mask_img = nib.load(str(mask_path))
    if len(mask_img.shape) == 4:
        mask_img = nib.Nifti1Image(
            np.asarray(mask_img.dataobj[..., 0], dtype=np.float32), mask_img.affine, mask_img.header
        )

    ref_space = (ref_img.shape[:3], ref_img.affine)
    same_shape = mask_img.shape[:3] == ref_img.shape[:3]
    same_affine = np.allclose(mask_img.affine, ref_img.affine, atol=1e-4)
    if not (same_shape and same_affine):
        mask_img = nibproc.resample_from_to(mask_img, ref_space, order=0)

    mask = (np.asarray(mask_img.dataobj, dtype=np.float32) > float(threshold)).astype(np.float32)
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Version E (angiogram CoW candidate) end-to-end on cropped CoW ROI."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/temporal_registered_cow_crop"),
        help="Root with case folders containing input_mag_raw.nii.gz, Vx.nii.gz, Vy.nii.gz, Vz.nii.gz",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        required=True,
        help="Case folder name, e.g. subject_001_7T",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/temporal_mip_tests"),
        help="Output root folder",
    )
    parser.add_argument("--venc", type=float, default=0.90, help="VENC in m/s")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis index in input NIfTI")

    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Optional external mask path (if omitted, full ROI is used)",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold for mask binarization")

    parser.add_argument("--angio-percentile-threshold", type=float, default=99.0)
    parser.add_argument("--angio-min-component-size", type=int, default=120)
    parser.add_argument("--angio-z-min-frac", type=float, default=0.18)
    parser.add_argument("--angio-z-max-frac", type=float, default=0.62)
    parser.add_argument("--angio-morph-radius", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from scipy.ndimage import label
        from skimage.morphology import ball, binary_closing, binary_opening, remove_small_objects
    except ImportError as exc:
        raise ImportError(
            "This script requires scipy + scikit-image. Install with: pip install scipy scikit-image"
        ) from exc

    base_dir = args.data_root / args.case_id
    if not base_dir.exists():
        available = sorted([p.name for p in args.data_root.iterdir() if p.is_dir()]) if args.data_root.exists() else []
        raise FileNotFoundError(
            f"Case folder not found: {base_dir}\nAvailable cases under {args.data_root}: {available}"
        )

    mag_path = base_dir / "input_mag_raw.nii.gz"
    vx_path = base_dir / "Vx.nii.gz"
    vy_path = base_dir / "Vy.nii.gz"
    vz_path = base_dir / "Vz.nii.gz"
    for p in [mag_path, vx_path, vy_path, vz_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    case_out = args.out_dir / args.case_id / "version_e"
    case_out.mkdir(parents=True, exist_ok=True)

    mag_img = nib.load(str(mag_path))
    vx_img = nib.load(str(vx_path))
    vy_img = nib.load(str(vy_path))
    vz_img = nib.load(str(vz_path))

    mag4d = ensure_time_last(np.asarray(mag_img.dataobj, dtype=np.float32), args.time_axis)
    vx4d_raw = ensure_time_last(np.asarray(vx_img.dataobj, dtype=np.float32), args.time_axis)
    vy4d_raw = ensure_time_last(np.asarray(vy_img.dataobj, dtype=np.float32), args.time_axis)
    vz4d_raw = ensure_time_last(np.asarray(vz_img.dataobj, dtype=np.float32), args.time_axis)

    if not (mag4d.shape == vx4d_raw.shape == vy4d_raw.shape == vz4d_raw.shape):
        raise ValueError(
            "Shape mismatch:\n"
            f"MAG {mag4d.shape}\n"
            f"Vx  {vx4d_raw.shape}\n"
            f"Vy  {vy4d_raw.shape}\n"
            f"Vz  {vz4d_raw.shape}"
        )

    vx4d = convert_raw_to_velocity_repo_style(vx4d_raw, args.venc, invert_sign=False)
    vy4d = convert_raw_to_velocity_repo_style(vy4d_raw, args.venc, invert_sign=False)
    vz4d = convert_raw_to_velocity_repo_style(vz4d_raw, args.venc, invert_sign=False)
    speed4d = np.sqrt(vx4d * vx4d + vy4d * vy4d + vz4d * vz4d).astype(np.float32)

    if args.mask_path is not None:
        mask3d = load_mask_aligned(args.mask_path, mag_img, args.mask_threshold)
    else:
        mask3d = np.ones(mag4d.shape[:3], dtype=np.float32)
    mask_bool = mask3d > 0

    mag_med = np.median(mag4d, axis=3).astype(np.float32)
    speed_p90 = np.percentile(speed4d, 90, axis=3).astype(np.float32)
    angio_3d = (
        robust_norm_in_mask(mag_med, mask3d) * robust_norm_in_mask(speed_p90, mask3d)
    ).astype(np.float32)

    thr_angio = np.percentile(angio_3d[mask_bool], args.angio_percentile_threshold)
    angio_bin = (angio_3d >= thr_angio) & mask_bool
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        angio_bin = binary_opening(angio_bin, ball(args.angio_morph_radius))
        angio_bin = binary_closing(angio_bin, ball(args.angio_morph_radius))
        angio_bin = remove_small_objects(angio_bin.astype(bool), min_size=args.angio_min_component_size)

    shape = angio_bin.shape
    z0 = int(shape[2] * args.angio_z_min_frac)
    z1 = int(shape[2] * args.angio_z_max_frac)
    angio_roi = np.zeros(shape, dtype=bool)
    angio_roi[:, :, z0:z1] = True
    angio_roi &= mask_bool

    labeled, n_comp = label(angio_bin)
    keep_ids: list[int] = []
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        if int(np.sum(comp)) < args.angio_min_component_size:
            continue
        if np.any(comp & angio_roi):
            keep_ids.append(comp_id)

    if len(keep_ids) == 0:
        cow_angio_seg = angio_bin.astype(np.uint8)
    else:
        cow_angio_seg = np.isin(labeled, keep_ids).astype(np.uint8)

    save_3d_with_ref(angio_3d, mag_img, case_out / "vE_cow_angiogram_3d.nii.gz")
    save_3d_with_ref(cow_angio_seg.astype(np.float32), mag_img, case_out / "vE_cow_angiogram_seg.nii.gz")
    save_3d_with_ref(angio_roi.astype(np.float32), mag_img, case_out / "vE_cow_angiogram_roi_mask.nii.gz")
    save_3d_with_ref(mask3d.astype(np.float32), mag_img, case_out / "vE_analysis_mask.nii.gz")

    print("Case:", args.case_id)
    print("Input:", base_dir.resolve())
    print("Output:", case_out.resolve())
    print("Shape (x,y,z,t):", mag4d.shape)
    print("Mask voxels:", int(np.sum(mask_bool)))
    print(f"Threshold p{args.angio_percentile_threshold}:", float(thr_angio))
    print("Components found:", int(n_comp))
    print("Components kept:", len(keep_ids))
    print("Final CoW angio voxels:", int(np.sum(cow_angio_seg > 0)))


if __name__ == "__main__":
    main()
