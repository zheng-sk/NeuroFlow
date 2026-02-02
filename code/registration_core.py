#!/usr/bin/env python3
"""
registration_core.py

Core logic for temporal registration of 4D NIfTI frames using ANTsPy.
It focuses on registering all time frames t -> t=0 (reference).
"""

import os
import shutil
import subprocess
import tempfile
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
        list or None: A list of transforms per frame (transforms_map), or None if failed.
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        ensure_dir(qc_dir)

        # 1. Load Image
        img4d = ants.image_read(input_path, dimension=4, reorient=False)
        T = img4d.shape[3]
        
        if T <= 1:
            print(f"Skipping {os.path.basename(input_path)}: Not a 4D image (T={T})")
            return None

        if ref_t < 0 or ref_t >= T:
             ref_t = 0
        
        # 2. Setup Reference
        ref = frame3d_from_4d(img4d, ref_t)
        ref_mask = make_mask_reference(ref)

        # 3. Registration Loop
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
        # spacing needs to handle 4D (x,y,z,t) vs 3D (x,y,z)
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
        import traceback
        traceback.print_exc()
        return None

def apply_transforms_to_file(input_path, output_path, transforms_map, ref_img_path=None):
    """
    Applies a list of temporal transforms to another 4D file (e.g. Phase scalars, CD).
    Scalar mode.
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        
        img4d = ants.image_read(input_path, dimension=4, reorient=False)
        T = img4d.shape[3]
        
        if T != len(transforms_map):
            print(f"Warning: Frame count mismatch for {os.path.basename(input_path)}. Img={T}, Transforms={len(transforms_map)}")
            return False

        # Properly resolve fixed reference geometry.
        # Ideally, we must use the SAME geometry used during registration (magnitude t=0)
        # to ensure all outputs are resampled to that exact grid.
        if ref_img_path and os.path.exists(ref_img_path):
             mag4d = ants.image_read(ref_img_path, dimension=4, reorient=False)
             # assuming ref_t=0 for registration always
             ref_geom = frame3d_from_4d(mag4d, 0)
        else:
             # Fallback (less safe if headers differ)
             ref_geom = frame3d_from_4d(img4d, 0)
        
        warped_frames = []
        
        for t in range(T):
            mov = frame3d_from_4d(img4d, t)
            tx_list = transforms_map[t]
            
            if not tx_list: 
                # Identity -> Just resample to match the target grid perfectly
                mov_w = ants.resample_image_to_target(mov, ref_geom, interp_type=1)
                warped_frames.append(mov_w)
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
        # Use ref_geom parameters for output to maintain consistency
        out_img = ants.from_numpy(
            warped_np,
            origin=ref_geom.origin,
            spacing=(*ref_geom.spacing, img4d.spacing[3]), # Combine spatial 3D + temporal
            direction=img4d.direction # Direction is usually 4x4, let's keep original 4D direction
            # Note: ANTsPy handling of 4D direction vs 3D can be tricky. 
            # Ideally we want the spatial part to match ref_geom.
        )
        # Safer reconstruction simply using original 4D header but copying data, 
        # IF we assume no rotation of the grid happened (which is true for temporal reg to t=0).
        # But if we want to be strict on grid:
        
        ants.image_write(out_img, output_path)
        return True

    except Exception as e:
        print(f"Error applying transforms to {input_path}: {e}")
        return False

def apply_transforms_to_velocity_triplet(vx_path, vy_path, vz_path,
                                        out_vx, out_vy, out_vz,
                                        transforms_map, fixed_ref_mag_path,
                                        ref_t=0, interpolator="linear",
                                        ants_apply="antsApplyTransforms"):
    """
    Aplica transforms temporales (t -> ref_t) a un campo vectorial (Vx,Vy,Vz).
    Usa imagetype=1 en ANTsPy para reorientar vectores correctamente.
    """
    try:
        ensure_dir(os.path.dirname(out_vx))

        vx4d = ants.image_read(vx_path, dimension=4, reorient=False)
        vy4d = ants.image_read(vy_path, dimension=4, reorient=False)
        vz4d = ants.image_read(vz_path, dimension=4, reorient=False)

        T = vx4d.shape[3]
        if not (vy4d.shape[3] == T and vz4d.shape[3] == T):
             print("Error: Velocity components have different frame counts.")
             return False
             
        if len(transforms_map) != T:
             print(f"Error: Transforms map size ({len(transforms_map)}) != Frames ({T})")
             return False

        # fixed geometry: usar MAG(ref_t) para que TODO quede exactamente en el mismo grid
        mag4d = ants.image_read(fixed_ref_mag_path, dimension=4, reorient=False)
        fixed_ref = frame3d_from_4d(mag4d, ref_t)

        out_vx_frames, out_vy_frames, out_vz_frames = [], [], []

        for t in range(T):
            vx = frame3d_from_4d(vx4d, t)
            vy = frame3d_from_4d(vy4d, t)
            vz = frame3d_from_4d(vz4d, t)

            tx_list = transforms_map[t]

            if not tx_list:
                # identidad: solo resample a fixed grid por consistencia
                out_vx_frames.append(ants.resample_image_to_target(vx, fixed_ref, interp_type=1))
                out_vy_frames.append(ants.resample_image_to_target(vy, fixed_ref, interp_type=1))
                out_vz_frames.append(ants.resample_image_to_target(vz, fixed_ref, interp_type=1))
                continue

            # merge to vector image (3 components)
            vvec = ants.merge_channels([vx, vy, vz])

            # Apply transforms with imagetype=1 (Vector)
            # This handles reorientation correctly for vectors without needing CLI
            vvec_w = ants.apply_transforms(
                fixed=fixed_ref,
                moving=vvec,
                transformlist=tx_list,
                interpolator=interpolator,
                imagetype=1, # Vector
                verbose=False
            )

            # Split channels
            comps = ants.split_channels(vvec_w)
            if len(comps) != 3:
                print(f"Error: Expected 3 components after warp, got {len(comps)}")
                return False
            
            vx_w, vy_w, vz_w = comps

            out_vx_frames.append(vx_w)
            out_vy_frames.append(vy_w)
            out_vz_frames.append(vz_w)

        # reassemble 4D outputs (keep original temporal spacing/header from inputs)
        def stack4d(frames, template4d):
            arr = np.stack([f.numpy() for f in frames], axis=-1)
            # Use template spacing for 4D consistency
            return ants.from_numpy(arr, origin=template4d.origin, spacing=template4d.spacing, direction=template4d.direction)

        ants.image_write(stack4d(out_vx_frames, vx4d), out_vx)
        ants.image_write(stack4d(out_vy_frames, vy4d), out_vy)
        ants.image_write(stack4d(out_vz_frames, vz4d), out_vz)

        return True

    except Exception as e:
        print(f"Error applying vector transforms: {str(e)}")
        import traceback
        traceback.print_exc()
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
