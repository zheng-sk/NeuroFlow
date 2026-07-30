#!/usr/bin/env python3
"""
Downsample a tree of NIfTI files by a spatial scale factor.

Use this to create synthetic lower-resolution inputs (e.g., x0.5) for SR experiments
with `res_increase=2`.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def _resolve_time_axis(ndim: int, axis: int) -> int:
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid time axis {axis} for ndim={ndim}")
    return axis


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_mask_file(path: Path, mask_keywords: Iterable[str]) -> bool:
    name = path.name.lower()
    return any(k.lower() in name for k in mask_keywords)


def _spatial_zoom_factors(shape: Tuple[int, ...], scale: float, time_axis: int) -> Tuple[float, ...]:
    z = [float(scale)] * len(shape)
    if len(shape) == 4:
        z[int(time_axis)] = 1.0
    return tuple(z)


def _downsample_one(
    in_path: Path,
    out_path: Path,
    scale: float,
    time_axis: int,
    mask_keywords: Iterable[str],
    force_linear: bool,
    force_nearest: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    img = nib.load(str(in_path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim not in (3, 4):
        raise ValueError(f"Expected 3D/4D NIfTI. Got {in_path}: shape={data.shape}")

    t_axis = _resolve_time_axis(data.ndim, time_axis) if data.ndim == 4 else -1
    is_mask = _is_mask_file(in_path, mask_keywords)
    if force_linear and force_nearest:
        raise ValueError("Choose either --force-linear or --force-nearest, not both.")

    if force_nearest:
        interp_order = 0
    elif force_linear:
        interp_order = 1
    else:
        interp_order = 0 if is_mask else 1

    zf = _spatial_zoom_factors(data.shape, float(scale), t_axis)
    out_data = zoom(data, zoom=zf, order=int(interp_order), mode="nearest")

    if interp_order == 0:
        # Keep mask-like nearest-neighbor outputs stable as integer if source was integer.
        src_dtype = img.get_data_dtype()
        if np.issubdtype(src_dtype, np.integer):
            out_data = np.rint(out_data).astype(src_dtype, copy=False)
        else:
            out_data = out_data.astype(np.float32, copy=False)
    else:
        out_data = out_data.astype(np.float32, copy=False)

    out_affine = img.affine.copy()
    out_affine[:3, :3] = out_affine[:3, :3] / float(scale)

    out_img = nib.Nifti1Image(out_data, out_affine, header=img.header.copy())
    out_img.set_qform(out_affine, int(img.header["qform_code"]))
    out_img.set_sform(out_affine, int(img.header["sform_code"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    return tuple(data.shape), tuple(out_data.shape), int(interp_order)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample all NIfTI files under --input-root and write them to --output-root "
            "preserving relative paths."
        )
    )
    parser.add_argument("--input-root", required=True, help="Root directory with input NIfTI files.")
    parser.add_argument("--output-root", required=True, help="Output root directory for downsampled NIfTI files.")
    parser.add_argument("--scale", type=float, default=0.5, help="Spatial scale factor. Use 0.5 for 2x downsample.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D files (default: last axis).")
    parser.add_argument(
        "--mask-keywords",
        type=str,
        default="mask,seg,label",
        help="Comma-separated keywords to detect mask files (nearest-neighbor interpolation).",
    )
    parser.add_argument("--force-linear", action="store_true", help="Force linear interpolation for all files.")
    parser.add_argument("--force-nearest", action="store_true", help="Force nearest-neighbor interpolation for all files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned operations without writing outputs.")
    args = parser.parse_args()

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {in_root}")
    if not (0.0 < float(args.scale) <= 1.0):
        raise ValueError(f"--scale must be in (0, 1]. Got {args.scale}")

    mask_keywords = [x.strip().lower() for x in str(args.mask_keywords).split(",") if x.strip()]
    files = [p for p in sorted(in_root.rglob("*")) if p.is_file() and _is_nifti(p)]
    if not files:
        raise RuntimeError(f"No NIfTI files found under {in_root}")

    print(f"Found {len(files)} NIfTI files under: {in_root}")
    print(f"Output root: {out_root}")
    print(f"Scale: {args.scale} (res_increase target ~ {int(round(1.0 / float(args.scale)))})")

    for i, in_path in enumerate(files, start=1):
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        if args.dry_run:
            print(f"[{i}/{len(files)}] {in_path} -> {out_path}")
            continue

        in_shape, out_shape, order = _downsample_one(
            in_path=in_path,
            out_path=out_path,
            scale=float(args.scale),
            time_axis=int(args.time_axis),
            mask_keywords=mask_keywords,
            force_linear=bool(args.force_linear),
            force_nearest=bool(args.force_nearest),
        )
        print(f"[{i}/{len(files)}] order={order} shape {in_shape} -> {out_shape} | {rel}")

    print("Done.")


if __name__ == "__main__":
    main()

