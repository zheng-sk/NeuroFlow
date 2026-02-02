#!/usr/bin/env python3
"""
registration_core.py

Core logic for temporal registration of 4D NIfTI frames using ANTsPy.
Refactored from temporal_register_to_t0.py for reusability.
"""

import os
import numpy as np
import ants
import matplotlib

# Set non-interactive backend for plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    arr = img4d.numpy()[..., t]
    return ants.from_numpy(
        arr,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )

def make_mask_reference(ref3d: "ants.ANTsImage") -> "ants.ANTsImage":
    m = ants.get_mask(ref3d, cleanup=2)
    m = ants.iMath(m, "FillHoles")
    m = ants.iMath(m, "GetLargestComponent")
    return m

def mean_abs_diff(a_img, b_img, mask=None) -> float:
    a = a_img.numpy()
    b = b_img.numpy()
    if mask is None:
        return float(np.mean(np.abs(a - b)))
    m = mask.numpy() > 0
    if not np.any(m):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[m] - b[m])))

def ncc(a_img, b_img, mask) -> float:
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
# Plotting Helpers
# ----------------------------
def _save_alpha_png(ref, mov, ref_mask, out_png, title, alpha=0.30):
    ref_np = ref.numpy()
    mov_np = mov.numpy()
    mask_np = ref_mask.numpy() > 0
    
    # Mid-slice Z
    z = ref_np.shape[2] // 2
    r = ref_np[:, :, z].astype(np.float32)
    m = mov_np[:, :, z].astype(np.float32)
    mk = mask_np[:, :, z]

    r01 = norm01(r) * mk
    m01 = norm01(m) * mk

    plt.figure(figsize=(6, 6))
    plt.imshow(r01, cmap="gray", interpolation="nearest")
    plt.imshow(m01, cmap="hot", interpolation="nearest", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _save_edge_png(ref, mov, ref_mask, out_png, title, alpha_max=0.95):
    # Simplified version of the original edge plotter for brevity in core lib
    ref_np = ref.numpy()
    mov_np = mov.numpy()
    mask_np = ref_mask.numpy() > 0

    z = ref_np.shape[2] // 2
    r = ref_np[:, :, z].astype(np.float32)
    m = mov_np[:, :, z].astype(np.float32)
    mk = mask_np[:, :, z]

    r01 = norm01(r) 
    m01 = norm01(m) * mk

    gx = np.abs(np.diff(m01, axis=1, prepend=m01[:, :1]))
    gy = np.abs(np.diff(m01, axis=0, prepend=m01[:1, :]))
    edge = gx + gy
    
    # Threshold basic
    edge = np.where(edge > 0.2, edge, 0)
    
    # Create RGBA
    rgba = np.zeros((edge.shape[0], edge.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = 1.0; rgba[..., 1] = 1.0; rgba[..., 2] = 0.0 # Yellow
    rgba[..., 3] = np.clip(edge * alpha_max, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(r01, cmap="gray", interpolation="nearest")
    plt.imshow(rgba, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ----------------------------
# Main Processing Logic
# ----------------------------
def register_4d_nifti(input_path, output_path, qc_dir, ref_t=0, reg_type="Rigid", interpolator="linear", qc_frames_str="0,1,50,99"):
    """
    Registers a single 4D NIfTI file.
    Returns True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        ensure_dir(qc_dir)

        # 1. Load Data
        img4d = ants.image_read(input_path, dimension=4, reorient=False)
        T = img4d.shape[3]

        if ref_t < 0 or ref_t >= T:
             # Fallback to T=0 if ref is invalid
             ref_t = 0
        
        ref = frame3d_from_4d(img4d, ref_t)
        ref_mask = make_mask_reference(ref)

        # 2. Registration Loop
        warped_frames = [None] * T
        warped_frames[ref_t] = ref # Identity
        
        # Metrics containers
        mad_vals = [] 
        ncc_vals = []

        for t in range(T):
            if t == ref_t:
                mad_vals.append(0)
                ncc_vals.append(1)
                continue
            
            mov = frame3d_from_4d(img4d, t)
            
            # Register
            reg = ants.registration(
                fixed=ref,
                moving=mov,
                type_of_transform=reg_type,
                mask=ref_mask,
                verbose=False
            )
            
            # Apply transform
            mov_w = ants.apply_transforms(
                fixed=ref,
                moving=mov,
                transformlist=reg["fwdtransforms"],
                interpolator=interpolator
            )
            
            warped_frames[t] = mov_w
            
            # Compute basic metrics for QC
            mad_vals.append(mean_abs_diff(ref, mov_w, mask=ref_mask))
            ncc_vals.append(ncc(ref, mov_w, mask=ref_mask))

        # 3. Reassemble 4D
        warped_np = np.stack([warped_frames[t].numpy() for t in range(T)], axis=-1)
        spacing4 = list(img4d.spacing)
        
        warped4d = ants.from_numpy(
            warped_np,
            origin=img4d.origin,
            spacing=tuple(spacing4),
            direction=img4d.direction
        )
        
        ants.image_write(warped4d, output_path)

        # 4. Generate Basic QC (Plots)
        # Parse QC frames indices
        qc_pcts = [int(x) for x in qc_frames_str.split(",") if x.strip().isdigit()]
        qc_idx = sorted(set(int(round((p / 100.0) * (T - 1))) for p in qc_pcts))

        for t_idx in qc_idx:
            mov_w = warped_frames[t_idx] if warped_frames[t_idx] else ref
            fname = f"qc_overlap_t{t_idx:03d}.png"
            _save_alpha_png(ref, mov_w, ref_mask, os.path.join(qc_dir, fname), title=f"QC Reg t={t_idx}")

        # Metrics Plot
        plt.figure()
        plt.plot(ncc_vals, label="NCC after reg")
        plt.title(f"Registration Stability (Mean NCC: {np.mean(ncc_vals):.4f})")
        plt.xlabel("Frame")
        plt.savefig(os.path.join(qc_dir, "qc_metrics.png"))
        plt.close()

        return True
    
    except Exception as e:
        print(f"Error registering {input_path}: {str(e)}")
        return False