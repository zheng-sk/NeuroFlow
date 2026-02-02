#!/usr/bin/env python3
"""
registration_core.py

Core logic for temporal registration of 4D NIfTI frames using ANTsPy.
It focuses on registering all time frames t -> t=0 (reference).
"""

import os
import numpy as np
import ants
import matplotlib

# Set non-interactive backend for plots to avoid crashes on servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    # Extract 3D frame respecting spatial metadata
    arr = img4d.numpy()[..., t]
    return ants.from_numpy(
        arr,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )

def make_mask_reference(ref3d: "ants.ANTsImage") -> "ants.ANTsImage":
    # Simple mask generation for the reference frame
    m = ants.get_mask(ref3d, cleanup=2)
    m = ants.iMath(m, "FillHoles")
    m = ants.iMath(m, "GetLargestComponent")
    return m

def register_4d_nifti(input_path, output_path, qc_dir, ref_t=0, reg_type="Rigid"):
    """
    Registers a 4D NIfTI file frame-by-frame to a reference time point.
    
    Args:
        input_path (str): Path to input 4D NIfTI.
        output_path (str): Path to save registered 4D NIfTI.
        qc_dir (str): Folder to save QC images.
        ref_t (int): Index of the reference frame (default 0).
        reg_type (str): ANTs registration type (Rigid, Affine, SyN).
    
    Returns:
        bool: True if successful.
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        ensure_dir(qc_dir)

        # 1. Load Image
        img4d = ants.image_read(input_path, dimension=4, reorient=False)
        T = img4d.shape[3]
        
        if T <= 1:
            print(f"Skipping {os.path.basename(input_path)}: Not a 4D image (T={T})")
            return False

        if ref_t < 0 or ref_t >= T:
             ref_t = 0
        
        # 2. Setup Reference
        ref = frame3d_from_4d(img4d, ref_t)
        ref_mask = make_mask_reference(ref)

        # 3. Registration Loop
        warped_frames = [None] * T
        warped_frames[ref_t] = ref # Identity for reference
        
        # For simplicity in this step, we are not saving the transforms to disk,
        # just applying them to create the motion-corrected magnitude image.
        
        for t in range(T):
            if t == ref_t:
                continue
            
            mov = frame3d_from_4d(img4d, t)
            
            # Register t -> ref
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
                interpolator="linear"
            )
            
            warped_frames[t] = mov_w

        # 4. Reassemble 4D Volume
        warped_np = np.stack([warped_frames[t].numpy() for t in range(T)], axis=-1)
        spacing4 = list(img4d.spacing)
        
        warped4d = ants.from_numpy(
            warped_np,
            origin=img4d.origin,
            spacing=tuple(spacing4),
            direction=img4d.direction
        )
        
        # 5. Save Output
        ants.image_write(warped4d, output_path)

        # 6. Basic QC Plot (Middle slices comparison for a sample frame)
        # Check the middle frame (or last) vs reference
        qc_t = T // 2
        if qc_t == ref_t: qc_t = T - 1
        
        ref_slice = ref.numpy()[:, :, ref.shape[2]//2]
        mov_slice = warped_frames[qc_t].numpy()[:, :, ref.shape[2]//2]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(ref_slice, cmap='gray')
        plt.title(f"Reference (t={ref_t})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mov_slice, cmap='gray')
        plt.title(f"Registered Frame (t={qc_t})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(qc_dir, f"qc_registration_t{qc_t}.png"))
        plt.close()

        return True
    
    except Exception as e:
        print(f"\nError processing {input_path}: {str(e)}")
        return False