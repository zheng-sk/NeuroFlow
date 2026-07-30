#!/usr/bin/env python3
"""Resample per-case CoW masks to a reference image grid, typically 3T x0.5."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resample CoW masks to a reference image grid using nearest-neighbor interpolation."
    )
    p.add_argument("--source-masks-root", required=True, help="Folder containing source masks in <root>/<case_id>/<mask-name>.")
    p.add_argument("--reference-root", required=True, help="Folder containing reference images in <root>/<case_id>/<reference-name>.")
    p.add_argument("--output-root", required=True, help="Folder where resampled masks will be written.")
    p.add_argument("--mask-name", default="cow_seg_final.nii.gz")
    p.add_argument("--reference-name", default="input_mag_raw.nii.gz")
    p.add_argument("--binary-threshold", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    source_root = Path(args.source_masks_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source masks root not found: {source_root}")
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")

    cases = sorted(p.name for p in source_root.iterdir() if p.is_dir())
    written = 0
    skipped = 0
    missing = []

    for case_id in cases:
        src_mask = source_root / case_id / args.mask_name
        ref_img_path = reference_root / case_id / args.reference_name
        out_dir = output_root / case_id
        out_mask = out_dir / args.mask_name

        if not src_mask.is_file() or not ref_img_path.is_file():
            missing.append(case_id)
            if args.verbose:
                print(f"[MISS] {case_id}: source mask or reference image missing")
            if args.strict:
                raise FileNotFoundError(f"{case_id}: missing source mask or reference image")
            continue

        if out_mask.exists() and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"[SKIP] {case_id}: output exists")
            continue

        src_img = nib.load(str(src_mask))
        ref_img = nib.load(str(ref_img_path))

        if len(src_img.shape) == 4:
            src_img = nib.Nifti1Image(np.asarray(src_img.dataobj[..., 0], dtype=np.float32), src_img.affine, src_img.header)

        ref_space = (ref_img.shape[:3], ref_img.affine)
        resampled = nibproc.resample_from_to(src_img, ref_space, order=0)
        resampled_data = (np.asarray(resampled.dataobj, dtype=np.float32) > float(args.binary_threshold)).astype(np.uint8)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_header = ref_img.header.copy()
        out_header.set_data_shape(resampled_data.shape)
        out_header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(resampled_data, ref_img.affine, out_header), str(out_mask))

        written += 1
        if args.verbose:
            print(f"[OK] {case_id}: {out_mask}")

    print("Mask resampling complete")
    print(f"Source masks root: {source_root}")
    print(f"Reference root:    {reference_root}")
    print(f"Output root:       {output_root}")
    print(f"Masks written:     {written}")
    print(f"Skipped:           {skipped}")
    print(f"Missing cases:     {len(missing)}")
    return 0 if not (args.strict and missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
