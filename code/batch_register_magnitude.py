#!/usr/bin/env python3
"""
batch_register_magnitude.py

Workflow:
1. Find all leaf directories (patient folders).
2. Inside each folder, identify the MAGNITUDE image.
3. Perform Temporal Registration on Magnitude -> Get Transforms.
4. Identify VELOCITY triplet (Vx, Vy, Vz) and apply transforms as VECTOR Field (reorientation).
5. Identify other scalars (Phase, etc.) and apply transforms as Scalar.
"""

import os
import argparse
import glob
from registration_core import register_4d_nifti, apply_transforms_to_file, apply_transforms_to_velocity_triplet

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def identify_images(folder):
    """
    Returns:
      mag_file: str
      velocity_triplet: tuple (vx, vy, vz) or None
      other_files: list of str
    """
    # Grab all .nii or .nii.gz
    files = glob.glob(os.path.join(folder, "*.nii.gz"))
    if not files:
        files = glob.glob(os.path.join(folder, "*.nii"))
    
    mag_file = None
    vx, vy, vz = None, None, None
    other_files = []
    
    for f in files:
        fname = os.path.basename(f).lower()
        
        # 1. Check for Magnitude
        # Exclude phase/velocity keywords
        is_mag = False
        terms_forbidden = ["phase", "vx", "vy", "vz", "velocity", "flow_x", "flow_y", "flow_z"]
        if ("magnitude" in fname or "mag" in fname) and not any(x in fname for x in terms_forbidden):
             is_mag = True
             
        if is_mag:
            if mag_file is None:
                mag_file = f
            else:
                # If multiple mags, keep first, put others in 'others' incase needed
                other_files.append(f)
            continue
        
        # 2. Check for Velocity Components
        # Common namings: "*_Vx.nii" or "*flow_x*"
        if "vx" in fname or "flow_x" in fname:
            vx = f
        elif "vy" in fname or "flow_y" in fname:
            vy = f
        elif "vz" in fname or "flow_z" in fname:
            vz = f
        else:
            other_files.append(f)
            
    # Form triplet if complete
    velocity_triplet = None
    if vx and vy and vz:
        velocity_triplet = (vx, vy, vz)
    else:
        # If incomplete triplet, treat them as independent scalars (better than crashing/skipping)
        if vx: other_files.append(vx)
        if vy: other_files.append(vy)
        if vz: other_files.append(vz)
    
    return mag_file, velocity_triplet, other_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reg-type", default="Rigid")
    args = parser.parse_args()

    # 1. Walk directories to find patient folders
    patient_dirs = []
    print("Scanning directories...")
    for root, dirs, files in os.walk(args.input_dir):
        if any(f.endswith('.nii') or f.endswith('.nii.gz') for f in files):
            patient_dirs.append(root)
    
    patient_dirs.sort()
    
    if not patient_dirs:
        print("No folders with NIfTI files found.")
        return

    print(f"Found {len(patient_dirs)} folders containing NIfTI files.")
    
    # 2. Main Progess Loop
    pbar = tqdm(patient_dirs, desc="Processing Patients")
    
    for p_dir in pbar:
        mag_path, velocity_triplet, other_paths = identify_images(p_dir)
        
        rel_dir = os.path.relpath(p_dir, args.input_dir)
        target_dir = os.path.join(args.output_dir, rel_dir)
        
        short_name = os.path.basename(p_dir)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(folder=short_name[:15])

        if not mag_path:
            continue

        # Paths
        mag_out = os.path.join(target_dir, os.path.basename(mag_path))
        qc_dir = os.path.join(target_dir, "QC_Registration")

        # Log
        if type(pbar).__name__ != 'tqdm':
             print(f"\nProcessing {short_name}...")

        # Step A: Register Magnitude (Source of Transforms)
        # ----------------------------
        transforms = register_4d_nifti(
            input_path=mag_path,
            output_path=mag_out,
            qc_dir=qc_dir,
            ref_t=0,
            reg_type=args.reg_type
        )
        
        if transforms is None:
            continue

        # Step B: Apply to Velocities (Vector Mode -> Reorientation)
        # ----------------------------
        if velocity_triplet:
            vx_in, vy_in, vz_in = velocity_triplet
            vx_out = os.path.join(target_dir, os.path.basename(vx_in))
            vy_out = os.path.join(target_dir, os.path.basename(vy_in))
            vz_out = os.path.join(target_dir, os.path.basename(vz_in))
            
            ok = apply_transforms_to_velocity_triplet(
                vx_path=vx_in, vy_path=vy_in, vz_path=vz_in,
                out_vx=vx_out, out_vy=vy_out, out_vz=vz_out,
                transforms_map=transforms,
                fixed_ref_mag_path=mag_path, # Use original mag as geometry ref
                ref_t=0
            )
            if not ok:
                print(f"Warning: Velocity vector transform failed for {short_name}")

        # Step C: Apply to Others (Phase, CD, etc.) as Scalars
        # ----------------------------
        for f_path in other_paths:
            f_out = os.path.join(target_dir, os.path.basename(f_path))
            apply_transforms_to_file(
                input_path=f_path,
                output_path=f_out,
                transforms_map=transforms,
                ref_img_path=mag_path 
            )

    print("\nBatch Processing Complete.")

if __name__ == "__main__":
    main()