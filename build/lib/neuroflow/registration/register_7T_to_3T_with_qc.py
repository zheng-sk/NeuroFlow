#!/usr/bin/env python3
"""
register_7T_to_3T_with_qc.py

Goal:
- Leave everything in 3T space (fixed = 3T, moving = 7T)
- Inputs are 4D MAG for 3T and 7T (ideally already temporally motion-corrected)
- 7T velocities are in RAW PHASE (wrapped). Phase can be warped in two modes:
  1) direct (default): apply only spatial transforms (recommended when preserving raw levels is priority)
  2) complex: real=mag*cos(phase), imag=mag*sin(phase), warp real/imag, then atan2(imag,real)

Outputs:
- Registered 7T magnitude 4D in 3T space
- Registered 7T phase triplet 4D in 3T space
- QC PNGs written to qc_dir (overwritten each run)

Usage example:
python register_7T_to_3T_with_qc.py \
  --fixed_mag_3t  ../data/temporal_registered/subject_001_3T/input_mag_raw.nii.gz \
  --moving_mag_7t ../data/temporal_registered/subject_001_7T/input_mag_raw.nii.gz \
  --moving_phase_x ../data/temporal_registered/subject_001_7T/input_phase_x_raw.nii.gz \
  --moving_phase_y ../data/temporal_registered/subject_001_7T/input_phase_y_raw.nii.gz \
  --moving_phase_z ../data/temporal_registered/subject_001_7T/input_phase_z_raw.nii.gz \
  --out_dir ../data/registered_7T_in_3T/subject_001 \
  --qc_dir  ../data/registered_7T_in_3T/subject_001/QC \
  --mask_method hdbet --device mps \
  --reg_type "antsRegistrationSyN[a]" \
  --verbose

Notes:
- For inter-scan 3T<->7T, start with linear (Rigid/Affine). Only use deformable if needed.
- Registration is estimated on a robust temporal reference (median across time).
- Transform is applied to ORIGINAL 7T magnitude and RAW phases.
"""

import os
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import ants

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def norm01(x: np.ndarray, p_lo=1, p_hi=99) -> np.ndarray:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip(x, lo, hi)
    y = (y - lo) / (hi - lo)
    return y.astype(np.float32)


def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    arr = img4d.numpy()[..., t]
    return ants.from_numpy(
        arr,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )


def ref3d_temporal_median(img4d: "ants.ANTsImage") -> "ants.ANTsImage":
    arr = img4d.numpy()
    ref = np.median(arr, axis=3)  # (x,y,z)
    return ants.from_numpy(
        ref,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )


def winsorize_image(img3d: "ants.ANTsImage", mask3d=None, p_low=0.01, p_high=0.99) -> "ants.ANTsImage":
    a = img3d.numpy()
    if mask3d is not None:
        m = mask3d.numpy() > 0
        vals = a[m]
    else:
        vals = a[np.isfinite(a)]
    if vals.size == 0:
        return img3d

    lo = float(np.quantile(vals, p_low))
    hi = float(np.quantile(vals, p_high))
    a_clip = np.clip(a, lo, hi)

    return ants.from_numpy(
        a_clip,
        spacing=img3d.spacing,
        origin=img3d.origin,
        direction=img3d.direction
    )


# ----------------------------
# Masking
# ----------------------------
def mask_from_ants(img3d: "ants.ANTsImage") -> "ants.ANTsImage":
    m = ants.get_mask(img3d, cleanup=2)
    m = ants.iMath(m, "FillHoles")
    m = ants.iMath(m, "GetLargestComponent")
    return m


def mask_from_hdbet(img3d: "ants.ANTsImage", out_dir: str, device="cpu", use_tta=False) -> "ants.ANTsImage":
    """
    Runs hd-bet on a 3D image and returns the mask as ANTsImage.
    Requires `hd-bet` installed and on PATH.
    Overwrites outputs in out_dir.
    """
    ensure_dir(out_dir)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.nii.gz")
        ants.image_write(img3d, in_path)

        out_base = os.path.join(out_dir, "brain_mask.nii.gz")
        # hd-bet saves mask as <out>_bet.nii.gz when --save_bet_mask and --no_bet_image
        cmd = [
            "hd-bet",
            "-i", in_path,
            "-o", out_base,
            "-device", device,
            "--save_bet_mask",
            "--no_bet_image",
        ]
        if not use_tta:
            cmd.append("--disable_tta")

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"HD-BET failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

        mask_path = out_base.replace(".nii.gz", "_bet.nii.gz")
        if not os.path.exists(mask_path):
            raise RuntimeError(f"HD-BET mask not found at {mask_path}")

        m = ants.image_read(mask_path)
        return m


# ----------------------------
# QC saving
# ----------------------------
def save_alpha_overlay(ref3d, mov3d, mask3d, out_png, title, alpha=0.30):
    r = ref3d.numpy()
    m = mov3d.numpy()
    mk = (mask3d.numpy() > 0) if mask3d is not None else np.ones_like(r, dtype=bool)

    z = r.shape[2] // 2
    r2 = norm01(r[:, :, z].astype(np.float32)) * mk[:, :, z]
    m2 = norm01(m[:, :, z].astype(np.float32)) * mk[:, :, z]

    plt.figure(figsize=(6, 6))
    plt.imshow(r2, cmap="gray", interpolation="nearest")
    plt.imshow(m2, cmap="hot", interpolation="nearest", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_edge_overlay(ref3d, mov3d, mask3d, out_png, title,
                      alpha_max=0.95, smooth_sigma=1.0, edge_thresh=0.25, dilate_iters=1):
    try:
        mov_s = ants.smooth_image(mov3d, smooth_sigma) if smooth_sigma and smooth_sigma > 0 else mov3d
    except Exception:
        mov_s = mov3d

    r = ref3d.numpy()
    m = mov_s.numpy()
    mk = (mask3d.numpy() > 0) if mask3d is not None else np.ones_like(r, dtype=bool)

    z = r.shape[2] // 2
    r2 = norm01(r[:, :, z].astype(np.float32))
    m2 = norm01(m[:, :, z].astype(np.float32)) * mk[:, :, z]

    gx = np.abs(np.diff(m2, axis=1, prepend=m2[:, :1]))
    gy = np.abs(np.diff(m2, axis=0, prepend=m2[:1, :]))
    edge = gx + gy

    vals = edge[np.isfinite(edge)]
    if vals.size == 0:
        edge01 = np.zeros_like(edge, dtype=np.float32)
    else:
        e_hi = np.percentile(vals, 99.5)
        edge01 = np.zeros_like(edge, dtype=np.float32) if e_hi <= 1e-8 else np.clip(edge / (e_hi + 1e-8), 0, 1).astype(np.float32)

    edge01 = np.where(edge01 >= edge_thresh, edge01, 0.0).astype(np.float32)

    def dilate2d(a):
        shifts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifts.append(np.roll(np.roll(a, dy, axis=0), dx, axis=1))
        return np.maximum.reduce(shifts)

    for _ in range(max(0, int(dilate_iters))):
        edge01 = dilate2d(edge01)

    rgba = np.zeros((edge01.shape[0], edge01.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 1.0
    rgba[..., 2] = 0.0
    rgba[..., 3] = np.clip(edge01 * alpha_max, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(r2, cmap="gray", interpolation="nearest")
    plt.imshow(rgba, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def mad(a3d, b3d, mask3d=None) -> float:
    a = a3d.numpy()
    b = b3d.numpy()
    if mask3d is None:
        return float(np.mean(np.abs(a - b)))
    m = mask3d.numpy() > 0
    if not np.any(m):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[m] - b[m])))


def ncc(a3d, b3d, mask3d=None) -> float:
    a = a3d.numpy().astype(np.float64)
    b = b3d.numpy().astype(np.float64)
    if mask3d is not None:
        m = mask3d.numpy() > 0
        if np.any(m):
            a = a[m]
            b = b[m]
        else:
            a = a.ravel()
            b = b.ravel()
    else:
        a = a.ravel()
        b = b.ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)


# ----------------------------
# Phase warp strategies
# ----------------------------
def warp_phase_direct(
    phase3d: "ants.ANTsImage",
    fixed_ref3d: "ants.ANTsImage",
    txlist,
    interpolator="nearestNeighbor",
):
    """
    Apply only spatial transforms to raw phase values (no phase-to-complex conversion).
    This is the least intrusive mode in terms of intensity/value changes.
    """
    return ants.apply_transforms(
        fixed=fixed_ref3d,
        moving=phase3d,
        transformlist=txlist,
        interpolator=interpolator,
    )


def warp_phase_raw_as_complex(
    phase3d: "ants.ANTsImage",
    mag3d: "ants.ANTsImage",
    fixed_ref3d: "ants.ANTsImage",
    txlist,
    interpolator="linear",
):
    """
    Warp raw wrapped phase safely:
      real = mag*cos(phase), imag = mag*sin(phase)
      warp real/imag as scalars
      phase_w = atan2(imag_w, real_w)
    """
    p = phase3d.numpy().astype(np.float32)
    m = mag3d.numpy().astype(np.float32)

    real = m * np.cos(p)
    imag = m * np.sin(p)

    real_img = ants.from_numpy(real, spacing=phase3d.spacing, origin=phase3d.origin, direction=phase3d.direction)
    imag_img = ants.from_numpy(imag, spacing=phase3d.spacing, origin=phase3d.origin, direction=phase3d.direction)

    real_w = ants.apply_transforms(fixed=fixed_ref3d, moving=real_img, transformlist=txlist, interpolator=interpolator)
    imag_w = ants.apply_transforms(fixed=fixed_ref3d, moving=imag_img, transformlist=txlist, interpolator=interpolator)

    pw = np.arctan2(imag_w.numpy().astype(np.float32), real_w.numpy().astype(np.float32))
    out = ants.from_numpy(pw, spacing=fixed_ref3d.spacing, origin=fixed_ref3d.origin, direction=fixed_ref3d.direction)
    return out


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixed_mag_3t", required=True, help="3T magnitude 4D (fixed space)")
    p.add_argument("--moving_mag_7t", required=True, help="7T magnitude 4D (moving)")
    p.add_argument("--moving_phase_x", required=True, help="7T raw phase X 4D")
    p.add_argument("--moving_phase_y", required=True, help="7T raw phase Y 4D")
    p.add_argument("--moving_phase_z", required=True, help="7T raw phase Z 4D")

    p.add_argument("--out_dir", required=True, help="Output directory (overwritten)")
    p.add_argument("--qc_dir", required=True, help="QC output directory (overwritten)")

    p.add_argument("--mask_method", choices=["ants", "hdbet", "none"], default="ants")
    p.add_argument("--device", default="cpu", help="For HD-BET: cpu/mps/cuda")
    p.add_argument("--use_tta", action="store_true", help="HD-BET TTA (slower)")

    p.add_argument("--reg_type", default="antsRegistrationSyN[a]",
                   help="ANTs registration type, e.g. Rigid, Affine, SyNRA, antsRegistrationSyN[a]")
    p.add_argument("--interpolator_mag", default="bSpline", help="Interpolator for magnitude application")
    p.add_argument(
        "--phase_warp_mode",
        choices=["direct", "complex"],
        default="direct",
        help="How to warp phase: direct=apply transform only, complex=mag*cos/sin strategy",
    )
    p.add_argument(
        "--interpolator_phase",
        default="nearestNeighbor",
        help="Interpolator for phase warping (nearestNeighbor recommended for raw phase preservation)",
    )

    p.add_argument("--hist_match", action="store_true", help="Histogram match moving->fixed for registration estimation only")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--qc_frames", default="0,50,99", help="Which time frames to QC as comma-separated indices (e.g., 0,10,20)")
    p.add_argument(
        "--save_brain_masks",
        action="store_true",
        help="Persist the 3T and 7T brain masks estimated for registration",
    )
    p.add_argument(
        "--brain_mask_dir",
        default=None,
        help="Directory where brain masks are saved (default: <out_dir>/BrainMasks)",
    )
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.qc_dir)

    # Load 4D
    fixed4d = ants.image_read(args.fixed_mag_3t, dimension=4, reorient=False)
    mov4d   = ants.image_read(args.moving_mag_7t, dimension=4, reorient=False)

    phx4d = ants.image_read(args.moving_phase_x, dimension=4, reorient=False)
    phy4d = ants.image_read(args.moving_phase_y, dimension=4, reorient=False)
    phz4d = ants.image_read(args.moving_phase_z, dimension=4, reorient=False)

    T_fixed = fixed4d.shape[3]
    T_mov   = mov4d.shape[3]

    # Only require consistency within 7T
    if not (T_mov == phx4d.shape[3] == phy4d.shape[3] == phz4d.shape[3]):
        raise ValueError("7T mag and phase components must have same T.")

    if args.verbose:
        print("3T fixed shape:", fixed4d.shape, "spacing:", fixed4d.spacing)
        print("7T moving shape:", mov4d.shape, "spacing:", mov4d.spacing)

    # Parse QC frames
    qc_frames = []
    if args.qc_frames:
        try:
            qc_frames = sorted({int(x.strip()) for x in args.qc_frames.split(",") if x.strip() != ""})
        except ValueError as e:
            raise ValueError(f"Invalid --qc_frames value: {args.qc_frames}") from e

    # Robust temporal references (3D)
    fixed_ref  = ref3d_temporal_median(fixed4d)
    moving_ref = ref3d_temporal_median(mov4d)

    # Masks
    if args.mask_method == "hdbet":
        fixed_mask  = mask_from_hdbet(fixed_ref,  os.path.join(args.qc_dir, "hdbet_fixed_3t"), device=args.device, use_tta=args.use_tta)
        moving_mask = mask_from_hdbet(moving_ref, os.path.join(args.qc_dir, "hdbet_moving_7t"), device=args.device, use_tta=args.use_tta)
    elif args.mask_method == "ants":
        fixed_mask  = mask_from_ants(fixed_ref)
        moving_mask = mask_from_ants(moving_ref)
    else:
        fixed_mask = None
        moving_mask = None
        if args.verbose:
            print("Masking disabled (--mask_method none).")

    if args.save_brain_masks and fixed_mask is not None and moving_mask is not None:
        mask_dir = args.brain_mask_dir or os.path.join(args.out_dir, "BrainMasks")
        ensure_dir(mask_dir)
        ants.image_write(fixed_mask, os.path.join(mask_dir, "fixed_3T_mask_ref.nii.gz"))
        ants.image_write(moving_mask, os.path.join(mask_dir, "moving_7T_mask_ref.nii.gz"))
    elif args.save_brain_masks and args.verbose:
        print("Skipping mask save: no masks available in --mask_method none.")

    # Preprocess for estimation only
    if fixed_mask is not None and moving_mask is not None:
        fixed_n4 = ants.n4_bias_field_correction(fixed_ref,  mask=fixed_mask,  shrink_factor=4, rescale_intensities=True)
        mov_n4   = ants.n4_bias_field_correction(moving_ref, mask=moving_mask, shrink_factor=4, rescale_intensities=True)
    else:
        fixed_n4 = ants.n4_bias_field_correction(fixed_ref,  shrink_factor=4, rescale_intensities=True)
        mov_n4   = ants.n4_bias_field_correction(moving_ref, shrink_factor=4, rescale_intensities=True)

    fixed_dn = ants.denoise_image(fixed_n4, noise_model="Rician")
    mov_dn   = ants.denoise_image(mov_n4,   noise_model="Rician")

    fixed_w = winsorize_image(fixed_dn, mask3d=fixed_mask, p_low=0.01, p_high=0.99)
    mov_w   = winsorize_image(mov_dn,   mask3d=moving_mask, p_low=0.01, p_high=0.99)

    if args.hist_match:
        mov_w = ants.histogram_match_image(mov_w, fixed_w, number_of_histogram_bins=256, number_of_match_points=15)

    # Save estimation QC before
    save_alpha_overlay(fixed_w, mov_w, fixed_mask,
                       os.path.join(args.qc_dir, "est_before_alpha.png"),
                       "EST BEFORE (fixed 3T vs moving 7T) alpha")
    save_edge_overlay(fixed_w, mov_w, fixed_mask,
                      os.path.join(args.qc_dir, "est_before_edge.png"),
                      "EST BEFORE (fixed 3T vs moving 7T) edge")

    # Registration (estimate)
    reg_kwargs = {
        "fixed": fixed_w,
        "moving": mov_w,
        "type_of_transform": args.reg_type,
        "verbose": args.verbose,
    }
    if fixed_mask is not None and moving_mask is not None:
        reg_kwargs.update(
            {
                "mask": fixed_mask,
                "moving_mask": moving_mask,
                "mask_all_stages": True,
            }
        )
    reg = ants.registration(**reg_kwargs)
    txlist = reg["fwdtransforms"]  # moving -> fixed
    warped_est = reg["warpedmovout"]

    # Save estimation QC after
    save_alpha_overlay(fixed_w, warped_est, fixed_mask,
                       os.path.join(args.qc_dir, "est_after_alpha.png"),
                       "EST AFTER (fixed 3T vs warped 7T) alpha")
    save_edge_overlay(fixed_w, warped_est, fixed_mask,
                      os.path.join(args.qc_dir, "est_after_edge.png"),
                      "EST AFTER (fixed 3T vs warped 7T) edge")

    est_mad_before = mad(fixed_w, mov_w, fixed_mask)
    est_mad_after  = mad(fixed_w, warped_est, fixed_mask)
    est_ncc_before = ncc(fixed_w, mov_w, fixed_mask)
    est_ncc_after  = ncc(fixed_w, warped_est, fixed_mask)

    # Persist transforms (recommended)
    tx_dir = os.path.join(args.out_dir, "Transforms_7T_to_3T")
    ensure_dir(tx_dir)
    txlist_persist = []
    for i, tx in enumerate(txlist):
        ext = ".mat" if tx.endswith(".mat") else ".h5" if tx.endswith(".h5") else os.path.splitext(tx)[1]
        dst = os.path.join(tx_dir, f"reg_fwd_{i:02d}{ext}")
        shutil.copy(tx, dst)
        txlist_persist.append(dst)

    # Apply to ORIGINAL 7T magnitude (all frames) -> 3T space
    fixed_geom = frame3d_from_4d(fixed4d, 0)  # exact grid to write into (3T space)
    mov_mag_np = mov4d.numpy()

    warped_mag_frames = []
    warped_phx_frames = []
    warped_phy_frames = []
    warped_phz_frames = []

    # QC metrics per 7T frame using the 3T reference (fixed_ref) resampled to fixed_geom
    fixed_geom = frame3d_from_4d(fixed4d, 0)  # exact 3T grid to write into
    fixed_ref_geom = ants.resample_image_to_target(fixed_ref, fixed_geom, interp_type=1)
    mask_fixed_geom = (
        ants.resample_image_to_target(fixed_mask, fixed_geom, interp_type=0)
        if fixed_mask is not None
        else None
    )

    if args.save_brain_masks and mask_fixed_geom is not None:
        mask_dir = args.brain_mask_dir or os.path.join(args.out_dir, "BrainMasks")
        ants.image_write(mask_fixed_geom, os.path.join(mask_dir, "fixed_3T_mask_output_grid.nii.gz"))

    mad_before_list = []
    mad_after_list  = []
    ncc_before_list = []
    ncc_after_list  = []

    mov_mag_np = mov4d.numpy()

    for t in range(T_mov):
        mov_mag_t = ants.from_numpy(
            mov_mag_np[..., t],
            spacing=mov4d.spacing[:3],
            origin=mov4d.origin[:3],
            direction=mov4d.direction[:3, :3],
        )

        # BEFORE metrics: compare in 3T grid by resampling moving to fixed_geom WITHOUT registration
        mov_mag_t_rs = ants.resample_image_to_target(mov_mag_t, fixed_geom, interp_type=1)

        # AFTER: apply inter-scan transform
        mov_mag_w = ants.apply_transforms(
            fixed=fixed_geom,
            moving=mov_mag_t,
            transformlist=txlist_persist,
            interpolator=args.interpolator_mag,
        )
        warped_mag_frames.append(mov_mag_w)

        mad_before_list.append(mad(fixed_ref_geom, mov_mag_t_rs, mask_fixed_geom))
        mad_after_list.append(mad(fixed_ref_geom, mov_mag_w,    mask_fixed_geom))
        ncc_before_list.append(ncc(fixed_ref_geom, mov_mag_t_rs, mask_fixed_geom))
        ncc_after_list.append(ncc(fixed_ref_geom, mov_mag_w,    mask_fixed_geom))

        # Phase warps
        phx_t = frame3d_from_4d(phx4d, t)
        phy_t = frame3d_from_4d(phy4d, t)
        phz_t = frame3d_from_4d(phz4d, t)

        if args.phase_warp_mode == "direct":
            phx_w = warp_phase_direct(phx_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)
            phy_w = warp_phase_direct(phy_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)
            phz_w = warp_phase_direct(phz_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)
        else:
            phx_w = warp_phase_raw_as_complex(phx_t, mov_mag_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)
            phy_w = warp_phase_raw_as_complex(phy_t, mov_mag_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)
            phz_w = warp_phase_raw_as_complex(phz_t, mov_mag_t, fixed_geom, txlist_persist, interpolator=args.interpolator_phase)

        warped_phx_frames.append(phx_w)
        warped_phy_frames.append(phy_w)
        warped_phz_frames.append(phz_w)

        # QC overlays for selected 7T frames (against fixed_ref_geom)
        if t in qc_frames:
            save_alpha_overlay(fixed_ref_geom, mov_mag_t_rs, mask_fixed_geom,
                            os.path.join(args.qc_dir, f"t{t:03d}_alpha_before.png"),
                            f"t={t} BEFORE (3T-ref vs 7T-resampled) alpha")
            save_alpha_overlay(fixed_ref_geom, mov_mag_w, mask_fixed_geom,
                            os.path.join(args.qc_dir, f"t{t:03d}_alpha_after.png"),
                            f"t={t} AFTER (3T-ref vs warped 7T) alpha")
            save_edge_overlay(fixed_ref_geom, mov_mag_t_rs, mask_fixed_geom,
                            os.path.join(args.qc_dir, f"t{t:03d}_edge_before.png"),
                            f"t={t} BEFORE (3T-ref vs 7T-resampled) edge")
            save_edge_overlay(fixed_ref_geom, mov_mag_w, mask_fixed_geom,
                            os.path.join(args.qc_dir, f"t{t:03d}_edge_after.png"),
                            f"t={t} AFTER (3T-ref vs warped 7T) edge")
        # Stack 4D outputs in 3T geometry (use fixed4d header)
    def stack4d(frames, template4d):
        arr = np.stack([f.numpy() for f in frames], axis=-1).astype(np.float32)
        return ants.from_numpy(arr, origin=template4d.origin, spacing=template4d.spacing, direction=template4d.direction)

    out_mag = os.path.join(args.out_dir, "mag_7T_in_3T.nii.gz")
    out_phx = os.path.join(args.out_dir, "phaseX_7T_in_3T.nii.gz")
    out_phy = os.path.join(args.out_dir, "phaseY_7T_in_3T.nii.gz")
    out_phz = os.path.join(args.out_dir, "phaseZ_7T_in_3T.nii.gz")

    ants.image_write(stack4d(warped_mag_frames, fixed4d), out_mag)
    ants.image_write(stack4d(warped_phx_frames, fixed4d), out_phx)
    ants.image_write(stack4d(warped_phy_frames, fixed4d), out_phy)
    ants.image_write(stack4d(warped_phz_frames, fixed4d), out_phz)

    # QC plots: per-frame MAD/NCC + estimation summary
    plt.figure()
    plt.plot(mad_before_list, label="MAD before (per frame)")
    plt.plot(mad_after_list,  label="MAD after (per frame)")
    plt.xlabel("Frame t")
    plt.ylabel("MAD (mask)")
    plt.title("7T->3T registration QC on MAG (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.qc_dir, "qc_mad_per_frame.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(ncc_before_list, label="NCC before (per frame)")
    plt.plot(ncc_after_list,  label="NCC after (per frame)")
    plt.xlabel("Frame t")
    plt.ylabel("NCC (mask)")
    plt.title("7T->3T registration QC on MAG (higher is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.qc_dir, "qc_ncc_per_frame.png"), dpi=150)
    plt.close()

    # Write a summary
    with open(os.path.join(args.qc_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== Inputs ===\n")
        f.write(f"fixed_mag_3t:  {args.fixed_mag_3t}\n")
        f.write(f"moving_mag_7t: {args.moving_mag_7t}\n")
        f.write(f"moving_phase_x: {args.moving_phase_x}\n")
        f.write(f"moving_phase_y: {args.moving_phase_y}\n")
        f.write(f"moving_phase_z: {args.moving_phase_z}\n\n")

        f.write("=== Registration estimation ===\n")
        f.write(f"reg_type: {args.reg_type}\n")
        f.write(f"mask_method: {args.mask_method}\n")
        f.write(f"hist_match: {args.hist_match}\n")
        f.write(f"phase_warp_mode: {args.phase_warp_mode}\n")
        f.write(f"interpolator_phase: {args.interpolator_phase}\n")
        f.write(f"EST MAD before/after: {est_mad_before:.6f} / {est_mad_after:.6f}\n")
        f.write(f"EST NCC before/after: {est_ncc_before:.6f} / {est_ncc_after:.6f}\n\n")

        f.write("=== Apply-to-4D outputs (3T space) ===\n")
        f.write(f"mag:   {out_mag}\n")
        f.write(f"phaseX:{out_phx}\n")
        f.write(f"phaseY:{out_phy}\n")
        f.write(f"phaseZ:{out_phz}\n\n")

        f.write("=== Transforms persisted ===\n")
        for pth in txlist_persist:
            f.write(pth + "\n")
        f.write("\n")

        if args.save_brain_masks and fixed_mask is not None:
            mask_dir = args.brain_mask_dir or os.path.join(args.out_dir, "BrainMasks")
            f.write("=== Brain masks ===\n")
            f.write(os.path.join(mask_dir, "fixed_3T_mask_ref.nii.gz") + "\n")
            f.write(os.path.join(mask_dir, "moving_7T_mask_ref.nii.gz") + "\n")
            f.write(os.path.join(mask_dir, "fixed_3T_mask_output_grid.nii.gz") + "\n\n")
        elif args.save_brain_masks:
            f.write("=== Brain masks ===\n")
            f.write("Not saved because --mask_method none was used.\n\n")

        f.write("=== Per-frame QC (MAG) ===\n")
        f.write(f"MAD before: min/median/max = {np.min(mad_before_list):.6f} / {np.median(mad_before_list):.6f} / {np.max(mad_before_list):.6f}\n")
        f.write(f"MAD after:  min/median/max = {np.min(mad_after_list):.6f} / {np.median(mad_after_list):.6f} / {np.max(mad_after_list):.6f}\n")
        f.write(f"NCC before: min/median/max = {np.min(ncc_before_list):.6f} / {np.median(ncc_before_list):.6f} / {np.max(ncc_before_list):.6f}\n")
        f.write(f"NCC after:  min/median/max = {np.min(ncc_after_list):.6f} / {np.median(ncc_after_list):.6f} / {np.max(ncc_after_list):.6f}\n")
        f.write(f"QC frames saved: {qc_frames}\n")

    if args.verbose:
        print("Wrote outputs:")
        print(" ", out_mag)
        print(" ", out_phx)
        print(" ", out_phy)
        print(" ", out_phz)
        print("QC in:", args.qc_dir)


if __name__ == "__main__":
    main()
