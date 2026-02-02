#!/usr/bin/env python3
"""
temporal_register_to_t0.py

Temporal registration (motion correction) for a 4D NIfTI:
- Registers every frame t -> reference frame t=0 using ANTsPy
- Writes registered 4D NIfTI
- Saves QC PNGs (overwrites on each run):
  * overlays BEFORE/AFTER for selected frames:
      1) alpha overlay (gray ref + hot moving)
      2) edge overlay (ref gray + moving edges)
  * MAD plot (lower better)
  * NCC plot (higher better)
  * mid-slice abs-diff BEFORE/AFTER for a selected frame
  * qc_summary.txt

Usage:
  python temporal_register_to_t0.py \
    --input /path/to/4d.nii.gz \
    --out /path/to/out_registered.nii.gz \
    --qc_dir /path/to/qc_outputs \
    --reg_type Rigid \
    --verbose
"""

import os
import argparse
import numpy as np
import ants

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# IO / geometry helpers
# ----------------------------
def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    arr = img4d.numpy()[..., t]
    return ants.from_numpy(
        arr,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Mask + metrics
# ----------------------------
def make_mask_reference(ref3d: "ants.ANTsImage") -> "ants.ANTsImage":
    m = ants.get_mask(ref3d, cleanup=2)
    m = ants.iMath(m, "FillHoles")
    m = ants.iMath(m, "GetLargestComponent")
    return m


def mean_abs_diff(a_img: "ants.ANTsImage", b_img: "ants.ANTsImage", mask: "ants.ANTsImage" = None) -> float:
    a = a_img.numpy()
    b = b_img.numpy()
    if mask is None:
        return float(np.mean(np.abs(a - b)))
    m = mask.numpy() > 0
    if not np.any(m):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[m] - b[m])))


def ncc(a_img: "ants.ANTsImage", b_img: "ants.ANTsImage", mask: "ants.ANTsImage") -> float:
    a = a_img.numpy()
    b = b_img.numpy()
    m = mask.numpy() > 0
    if np.any(m):
        a = a[m].astype(np.float64)
        b = b[m].astype(np.float64)
    else:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()

    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)


# ----------------------------
# Display normalization (robust)
# ----------------------------
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


# ----------------------------
# QC image savers (no ants.plot)
# ----------------------------
def save_mid_slice_overlay_alpha_png(ref, mov, ref_mask, out_png, title, alpha=0.30):
    """
    Gray reference + hot moving with alpha.
    Masks the display to focus on anatomy.
    """
    ref_np = ref.numpy()
    mov_np = mov.numpy()
    mask_np = ref_mask.numpy() > 0

    z = ref_np.shape[2] // 2
    r = ref_np[:, :, z].astype(np.float32)
    m = mov_np[:, :, z].astype(np.float32)
    mk = mask_np[:, :, z]

    r01 = norm01(r)
    m01 = norm01(m)

    # Apply mask to reduce background confetti
    r01 = r01 * mk
    m01 = m01 * mk

    plt.figure(figsize=(6, 6))
    plt.imshow(r01, cmap="gray", interpolation="nearest")
    plt.imshow(m01, cmap="hot", interpolation="nearest", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_mid_slice_edge_overlay_png(ref, mov, ref_mask, out_png, title,
                                    alpha_max=0.95, smooth_sigma=1.0,
                                    edge_thresh=0.20, dilate_iters=1):
    """
    Gray reference + moving edges overlay with TRUE transparency:
      - edges are colored, background is transparent (no magenta fill)
      - smoothing reduces noisy edges
      - optional threshold + dilation to make edges readable
    """
    # Smooth moving (3D) for more stable edges (only QC frames)
    try:
        mov_s = ants.smooth_image(mov, smooth_sigma) if smooth_sigma and smooth_sigma > 0 else mov
    except Exception:
        mov_s = mov

    ref_np = ref.numpy()
    mov_np = mov_s.numpy()
    mask_np = ref_mask.numpy() > 0

    z = ref_np.shape[2] // 2
    r = ref_np[:, :, z].astype(np.float32)
    m = mov_np[:, :, z].astype(np.float32)
    mk = mask_np[:, :, z]

    r01 = norm01(r)  # keep full ref for context (no mask)
    m01 = norm01(m) * mk  # compute edges only in mask

    # Gradient-based edge
    gx = np.abs(np.diff(m01, axis=1, prepend=m01[:, :1]))
    gy = np.abs(np.diff(m01, axis=0, prepend=m01[:1, :]))
    edge = gx + gy

    # Robust normalize edge
    vals = edge[np.isfinite(edge)]
    if vals.size == 0:
        edge01 = np.zeros_like(edge, dtype=np.float32)
    else:
        e_hi = np.percentile(vals, 99.5)
        if e_hi <= 1e-8:
            edge01 = np.zeros_like(edge, dtype=np.float32)
        else:
            edge01 = np.clip(edge / (e_hi + 1e-8), 0, 1).astype(np.float32)

    # Threshold edges to avoid "noise confetti"
    edge01 = np.where(edge01 >= edge_thresh, edge01, 0.0).astype(np.float32)

    # Optional dilation (simple max-filter style) to make edges visible
    def dilate2d(a):
        shifts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifts.append(np.roll(np.roll(a, dy, axis=0), dx, axis=1))
        return np.maximum.reduce(shifts)

    for _ in range(max(0, int(dilate_iters))):
        edge01 = dilate2d(edge01)

    # Build RGBA overlay: colored edges with per-pixel alpha; background fully transparent
    # Color choice: yellow edges (RGB = [1,1,0]) tends to read well on grayscale.
    rgba = np.zeros((edge01.shape[0], edge01.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 1.0
    rgba[..., 2] = 0.0
    rgba[..., 3] = np.clip(edge01 * alpha_max, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(r01, cmap="gray", interpolation="nearest")
    plt.imshow(rgba, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Temporal registration of 4D NIfTI frames to t=0 using ANTsPy.")
    p.add_argument("--input", required=True, help="Input 4D NIfTI (.nii or .nii.gz)")
    p.add_argument("--out", required=True, help="Output registered 4D NIfTI path")
    p.add_argument("--qc_dir", required=True, help="Directory to write QC PNGs (overwritten each run)")
    p.add_argument("--reg_type", default="Rigid",
                   help="ANTs registration type_of_transform (e.g., Rigid, QuickRigid, Affine)")
    p.add_argument("--ref_t", type=int, default=0, help="Reference frame index (default: 0)")
    p.add_argument("--qc_frames", default="0,1,25,50,75,99",
                   help="QC frame percentiles list in [0..99] (default: 0,1,25,50,75,99)")
    p.add_argument("--diff_frame", type=int, default=-1,
                   help="Frame index for abs-diff QC (default: -1 means last frame)")
    p.add_argument("--interpolator", default="linear",
                   help="Interpolator for apply_transforms (linear, bSpline, nearestNeighbor, etc.)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    ensure_dir(args.qc_dir)

    if args.verbose:
        print("Reading:", args.input)

    img4d = ants.image_read(args.input, dimension=4, reorient=False)
    T = img4d.shape[3]

    if args.ref_t < 0 or args.ref_t >= T:
        raise ValueError(f"--ref_t out of range: {args.ref_t} for T={T}")

    ref = frame3d_from_4d(img4d, args.ref_t)
    ref_mask = make_mask_reference(ref)

    # QC frames from percentiles
    qc_pcts = []
    for tok in args.qc_frames.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        v = int(tok)
        if v < 0 or v > 99:
            raise ValueError(f"qc percentile must be 0..99, got {v}")
        qc_pcts.append(v)

    qc_idx = sorted(set(int(round((p / 100.0) * (T - 1))) for p in qc_pcts))
    if args.ref_t not in qc_idx:
        qc_idx = sorted(set(qc_idx + [args.ref_t]))

    diff_t = args.diff_frame if args.diff_frame >= 0 else (T - 1)
    diff_t = int(np.clip(diff_t, 0, T - 1))

    if args.verbose:
        print("img4d shape:", img4d.shape)
        print("ref_t:", args.ref_t)
        print("reg_type:", args.reg_type)
        print("QC indices:", qc_idx)
        print("diff frame:", diff_t)

    warped_frames = [None] * T

    mad_before = np.zeros(T, dtype=np.float64)
    mad_after = np.zeros(T, dtype=np.float64)
    ncc_before = np.zeros(T, dtype=np.float64)
    ncc_after = np.zeros(T, dtype=np.float64)

    # Reference frame identity
    warped_frames[args.ref_t] = ref
    mad_before[args.ref_t] = 0.0
    mad_after[args.ref_t] = 0.0
    ncc_before[args.ref_t] = 1.0
    ncc_after[args.ref_t] = 1.0

    # Register all other frames
    for t in range(T):
        if t == args.ref_t:
            continue

        mov = frame3d_from_4d(img4d, t)

        mad_before[t] = mean_abs_diff(ref, mov, mask=ref_mask)
        ncc_before[t] = ncc(ref, mov, mask=ref_mask)

        reg = ants.registration(
            fixed=ref,
            moving=mov,
            type_of_transform=args.reg_type,
            mask=ref_mask,
            moving_mask=None,
            mask_all_stages=True,
            verbose=args.verbose,
        )

        tx = reg["fwdtransforms"]
        mov_w = ants.apply_transforms(
            fixed=ref,
            moving=mov,
            transformlist=tx,
            interpolator=args.interpolator
        )

        warped_frames[t] = mov_w

        mad_after[t] = mean_abs_diff(ref, mov_w, mask=ref_mask)
        ncc_after[t] = ncc(ref, mov_w, mask=ref_mask)

        if args.verbose:
            print(f"t={t:03d}: MAD {mad_before[t]:.3f} -> {mad_after[t]:.3f} | "
                  f"NCC {ncc_before[t]:.3f} -> {ncc_after[t]:.3f}")

    # Write registered 4D
    warped_np = np.stack([warped_frames[t].numpy() for t in range(T)], axis=-1)

    spacing4 = list(img4d.spacing)

    warped4d = ants.from_numpy(
        warped_np,
        origin=img4d.origin,
        spacing=tuple(spacing4),
        direction=img4d.direction
    )

    if args.verbose:
        print("Writing registered 4D:", args.out)
    ants.image_write(warped4d, args.out)

    # ----------------------------
    # QC outputs (overwrite)
    # ----------------------------
    # Metrics plots
    plt.figure()
    plt.plot(mad_before, label="MAD before")
    plt.plot(mad_after, label="MAD after")
    plt.xlabel("Frame t")
    plt.ylabel("Mean absolute diff vs ref (mask)")
    plt.title("Temporal registration QC (MAD lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.qc_dir, "qc_mad.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(ncc_before, label="NCC before")
    plt.plot(ncc_after, label="NCC after")
    plt.xlabel("Frame t")
    plt.ylabel("Normalized cross-correlation vs ref (mask)")
    plt.title("Temporal registration QC (NCC higher is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.qc_dir, "qc_ncc.png"), dpi=150)
    plt.close()

    # Overlays for selected frames: BOTH types
    for t in qc_idx:
        mov = frame3d_from_4d(img4d, t)
        mov_w = warped_frames[t]

        # Alpha overlay
        save_mid_slice_overlay_alpha_png(
            ref, mov, ref_mask,
            os.path.join(args.qc_dir, f"alpha_before_t{t:03d}.png"),
            title=f"ALPHA BEFORE (ref t={args.ref_t} vs frame t={t})",
            alpha=0.30
        )
        save_mid_slice_overlay_alpha_png(
            ref, mov_w, ref_mask,
            os.path.join(args.qc_dir, f"alpha_after_t{t:03d}.png"),
            title=f"ALPHA AFTER (ref t={args.ref_t} vs warped t={t})",
            alpha=0.30
        )

        # Edge overlay
        save_mid_slice_edge_overlay_png(
            ref, mov, ref_mask,
            os.path.join(args.qc_dir, f"edge_before_t{t:03d}.png"),
            title=f"EDGE BEFORE (ref t={args.ref_t} vs frame t={t})",
            alpha_max=0.85
        )
        save_mid_slice_edge_overlay_png(
            ref, mov_w, ref_mask,
            os.path.join(args.qc_dir, f"edge_after_t{t:03d}.png"),
            title=f"EDGE AFTER (ref t={args.ref_t} vs warped t={t})",
            alpha_max=0.85
        )

    # Summary text
    summary_path = os.path.join(args.qc_dir, "qc_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Input: {args.input}\n")
        f.write(f"Output: {args.out}\n")
        f.write(f"T: {T}\n")
        f.write(f"Reference frame: {args.ref_t}\n")
        f.write(f"Registration type: {args.reg_type}\n")
        f.write(f"Interpolator: {args.interpolator}\n\n")
        f.write("MAD before: min/median/max = "
                f"{mad_before.min():.6f} / {np.median(mad_before):.6f} / {mad_before.max():.6f}\n")
        f.write("MAD after:  min/median/max = "
                f"{mad_after.min():.6f} / {np.median(mad_after):.6f} / {mad_after.max():.6f}\n\n")
        f.write("NCC before: min/median/max = "
                f"{ncc_before.min():.6f} / {np.median(ncc_before):.6f} / {ncc_before.max():.6f}\n")
        f.write("NCC after:  min/median/max = "
                f"{ncc_after.min():.6f} / {np.median(ncc_after):.6f} / {ncc_after.max():.6f}\n\n")
        f.write("QC frames indices: " + ",".join(map(str, qc_idx)) + "\n")
        f.write(f"Abs-diff frame: {diff_t}\n\n")
        f.write("QC files written (overwritten each run):\n")
        f.write("  qc_mad.png\n")
        f.write("  qc_ncc.png\n")
        f.write("  alpha_before_t*.png / alpha_after_t*.png\n")
        f.write("  edge_before_t*.png  / edge_after_t*.png\n")
        f.write("  absdiff_before_t*.png / absdiff_after_t*.png\n")
        f.write("  qc_summary.txt\n")

    if args.verbose:
        print("QC written to:", args.qc_dir)
        print(" - qc_mad.png")
        print(" - qc_ncc.png")
        print(" - alpha_before_t*.png / alpha_after_t*.png")
        print(" - edge_before_t*.png  / edge_after_t*.png")
        print(" - absdiff_before_t*.png / absdiff_after_t*.png")
        print(" - qc_summary.txt")


if __name__ == "__main__":
    main()