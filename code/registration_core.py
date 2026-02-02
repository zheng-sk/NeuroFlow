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


        warped_frames = [None] * T
        warped_frames[ref_t] = ref 
        
        # Store transform lists for each time point
        transforms_map = [None] * T
        transforms_map[ref_t] = [] # Identity
        
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
            
            # Save transforms 
            # Note: reg['fwdtransforms'] are temporary files.
            # To be safe for subsequent files, we should ideally rely on them existing,
            # or copy them. ANTsPy usually keeps them until process exit.
            transforms_map[t] = reg["fwdtransforms"]
            
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

        # 6. Basic QC
        qc_t = T // 2
        if qc_t == ref_t: qc_t = T - 1
        
        _save_qc_plot(ref, warped_frames[qc_t], qc_dir, ref_t, qc_t)

        return transforms_map
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None

def apply_transforms_to_file(input_path, output_path, transforms_map, ref_img_path=None):
    """
    Applies a list of temporal transforms to another 4D file (e.g. Velocity).
    transforms_map: list where index t contains transform files for frame t.
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        
        img4d = ants.image_read(input_path, dimension=4, reorient=False)
        T = img4d.shape[3]
        
        if T != len(transforms_map):
            print(f"Warning: Frame count mismatch for {os.path.basename(input_path)}. Img={T}, Transforms={len(transforms_map)}")
            return False

        # Reference geometry from frame 0 of THIS image (to keep valid header info)
        ref_geom = frame3d_from_4d(img4d, 0)
        
        warped_frames = []
        
        for t in range(T):
            mov = frame3d_from_4d(img4d, t)
            tx_list = transforms_map[t]
            
            if not tx_list: # Identity
                warped_frames.append(mov)
            else:
                mov_w = ants.apply_transforms(
                    fixed=ref_geom,
                    moving=mov,
                    transformlist=tx_list,
                    interpolator="linear"
                )
                warped_frames.append(mov_w)

        # Save
        warped_np = np.stack([w.numpy() for w in warped_frames], axis=-1)
        out_img = ants.from_numpy(
            warped_np,
            origin=img4d.origin,
            spacing=img4d.spacing,
            direction=img4d.direction
        )
        ants.image_write(out_img, output_path)
        return True

    except Exception as e:
        print(f"Error applying transforms to {input_path}: {e}")
        return False

def _save_qc_plot(ref, mov, qc_dir, t_ref, t_mov):
    ref_slice = ref.numpy()[:, :, ref.shape[2]//2]
    mov_slice = mov.numpy()[:, :, ref.shape[2]//2]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(ref_slice, cmap='gray'); plt.title(f"Reference (t={t_ref})"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(mov_slice, cmap='gray'); plt.title(f"Registered (t={t_mov})"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, f"qc_registration_t{t_mov}.png"))
    plt.close()
