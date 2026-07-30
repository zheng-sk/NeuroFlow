#!/usr/bin/env python3
"""Temporal registration (motion correction) for one 4D NIfTI file."""

from __future__ import annotations

import argparse
import os

import ants
import numpy as np

try:
    from .common import (
        ensure_dir,
        frame3d_from_4d,
        make_mask_reference,
        mean_abs_diff,
        normalized_cross_correlation,
        parse_qc_frames,
        stack_4d,
    )
    from .qc import save_absdiff_png, save_alpha_overlay, save_edge_overlay, save_metric_plot
except ImportError:  # pragma: no cover - allows direct script execution from this folder
    from common import (
        ensure_dir,
        frame3d_from_4d,
        make_mask_reference,
        mean_abs_diff,
        normalized_cross_correlation,
        parse_qc_frames,
        stack_4d,
    )
    from qc import save_absdiff_png, save_alpha_overlay, save_edge_overlay, save_metric_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal registration of 4D NIfTI frames to one reference frame.")
    parser.add_argument("--input", required=True, help="Input 4D NIfTI (.nii or .nii.gz)")
    parser.add_argument("--out", required=True, help="Output registered 4D NIfTI path")
    parser.add_argument("--qc_dir", required=True, help="Directory to write QC PNGs")
    parser.add_argument("--reg_type", default="Rigid", help="ANTs registration type_of_transform")
    parser.add_argument("--ref_t", type=int, default=0, help="Reference frame index (default: 0)")
    parser.add_argument(
        "--qc_frames",
        default="0,1,25,50,75,99",
        help="QC frame percentiles list in [0..99]",
    )
    parser.add_argument("--diff_frame", type=int, default=-1, help="Frame index for abs-diff QC (-1 uses last)")
    parser.add_argument("--interpolator", default="linear", help="Interpolator for apply_transforms")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.qc_dir)

    if args.verbose:
        print("Reading:", args.input)

    img4d = ants.image_read(args.input, dimension=4, reorient=False)
    total_frames = img4d.shape[3]

    if args.ref_t < 0 or args.ref_t >= total_frames:
        raise ValueError(f"--ref_t out of range: {args.ref_t} for T={total_frames}")

    reference = frame3d_from_4d(img4d, args.ref_t)
    reference_mask = make_mask_reference(reference)

    qc_indices = parse_qc_frames(args.qc_frames, total_frames, args.ref_t)
    diff_t = args.diff_frame if args.diff_frame >= 0 else total_frames - 1
    diff_t = int(np.clip(diff_t, 0, total_frames - 1))

    if args.verbose:
        print("img4d shape:", img4d.shape)
        print("ref_t:", args.ref_t)
        print("reg_type:", args.reg_type)
        print("QC indices:", qc_indices)
        print("diff frame:", diff_t)

    warped_frames = [None] * total_frames
    mad_before = np.zeros(total_frames, dtype=np.float64)
    mad_after = np.zeros(total_frames, dtype=np.float64)
    ncc_before = np.zeros(total_frames, dtype=np.float64)
    ncc_after = np.zeros(total_frames, dtype=np.float64)

    warped_frames[args.ref_t] = reference
    ncc_before[args.ref_t] = 1.0
    ncc_after[args.ref_t] = 1.0

    for t in range(total_frames):
        if t == args.ref_t:
            continue

        moving = frame3d_from_4d(img4d, t)

        mad_before[t] = mean_abs_diff(reference, moving, mask=reference_mask)
        ncc_before[t] = normalized_cross_correlation(reference, moving, mask=reference_mask)

        registration = ants.registration(
            fixed=reference,
            moving=moving,
            type_of_transform=args.reg_type,
            mask=reference_mask,
            moving_mask=None,
            mask_all_stages=True,
            verbose=args.verbose,
        )

        warped = ants.apply_transforms(
            fixed=reference,
            moving=moving,
            transformlist=registration["fwdtransforms"],
            interpolator=args.interpolator,
        )
        warped_frames[t] = warped

        mad_after[t] = mean_abs_diff(reference, warped, mask=reference_mask)
        ncc_after[t] = normalized_cross_correlation(reference, warped, mask=reference_mask)

        if args.verbose:
            print(
                f"t={t:03d}: MAD {mad_before[t]:.3f} -> {mad_after[t]:.3f} | "
                f"NCC {ncc_before[t]:.3f} -> {ncc_after[t]:.3f}"
            )

    ants.image_write(stack_4d(warped_frames, img4d), args.out)

    save_metric_plot(
        mad_before,
        mad_after,
        os.path.join(args.qc_dir, "qc_mad.png"),
        "Temporal registration QC (MAD lower is better)",
        "Mean absolute diff vs ref (mask)",
    )
    save_metric_plot(
        ncc_before,
        ncc_after,
        os.path.join(args.qc_dir, "qc_ncc.png"),
        "Temporal registration QC (NCC higher is better)",
        "Normalized cross-correlation vs ref (mask)",
    )

    for t in qc_indices:
        moving = frame3d_from_4d(img4d, t)
        warped = warped_frames[t]

        save_alpha_overlay(
            reference,
            moving,
            reference_mask,
            os.path.join(args.qc_dir, f"alpha_before_t{t:03d}.png"),
            f"ALPHA BEFORE (ref t={args.ref_t} vs frame t={t})",
            alpha=0.30,
        )
        save_alpha_overlay(
            reference,
            warped,
            reference_mask,
            os.path.join(args.qc_dir, f"alpha_after_t{t:03d}.png"),
            f"ALPHA AFTER (ref t={args.ref_t} vs warped t={t})",
            alpha=0.30,
        )
        save_edge_overlay(
            reference,
            moving,
            reference_mask,
            os.path.join(args.qc_dir, f"edge_before_t{t:03d}.png"),
            f"EDGE BEFORE (ref t={args.ref_t} vs frame t={t})",
            alpha_max=0.85,
            edge_thresh=0.20,
        )
        save_edge_overlay(
            reference,
            warped,
            reference_mask,
            os.path.join(args.qc_dir, f"edge_after_t{t:03d}.png"),
            f"EDGE AFTER (ref t={args.ref_t} vs warped t={t})",
            alpha_max=0.85,
            edge_thresh=0.20,
        )

    diff_moving = frame3d_from_4d(img4d, diff_t)
    save_absdiff_png(
        reference,
        diff_moving,
        os.path.join(args.qc_dir, f"absdiff_before_t{diff_t:03d}.png"),
        f"ABSDIFF BEFORE (ref t={args.ref_t} vs frame t={diff_t})",
    )
    save_absdiff_png(
        reference,
        warped_frames[diff_t],
        os.path.join(args.qc_dir, f"absdiff_after_t{diff_t:03d}.png"),
        f"ABSDIFF AFTER (ref t={args.ref_t} vs warped t={diff_t})",
    )

    summary_path = os.path.join(args.qc_dir, "qc_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Input: {args.input}\n")
        handle.write(f"Output: {args.out}\n")
        handle.write(f"T: {total_frames}\n")
        handle.write(f"Reference frame: {args.ref_t}\n")
        handle.write(f"Registration type: {args.reg_type}\n")
        handle.write(f"Interpolator: {args.interpolator}\n\n")
        handle.write(
            "MAD before: min/median/max = "
            f"{mad_before.min():.6f} / {np.median(mad_before):.6f} / {mad_before.max():.6f}\n"
        )
        handle.write(
            "MAD after:  min/median/max = "
            f"{mad_after.min():.6f} / {np.median(mad_after):.6f} / {mad_after.max():.6f}\n\n"
        )
        handle.write(
            "NCC before: min/median/max = "
            f"{ncc_before.min():.6f} / {np.median(ncc_before):.6f} / {ncc_before.max():.6f}\n"
        )
        handle.write(
            "NCC after:  min/median/max = "
            f"{ncc_after.min():.6f} / {np.median(ncc_after):.6f} / {ncc_after.max():.6f}\n\n"
        )
        handle.write("QC frame indices: " + ",".join(map(str, qc_indices)) + "\n")
        handle.write(f"Abs-diff frame: {diff_t}\n")

    if args.verbose:
        print("QC written to:", args.qc_dir)


if __name__ == "__main__":
    main()
