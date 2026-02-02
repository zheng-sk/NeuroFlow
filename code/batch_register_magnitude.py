#!/usr/bin/env python3
"""
batch_register_magnitude.py

Workflow:
1. Find all leaf directories (patient folders).
2. Inside each folder, identify the MAGNITUDE image.
3. Perform Temporal Registration on Magnitude -> Get Transforms.
4. Identify VELOCITY/PHASE images in the same folder.
5. Apply the SAME Transforms to Velocity images.
"""

import os
import argparse
import glob
from registration_core import register_4d_nifti, apply_transforms_to_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def identify_images(folder):
    """
    Returns a dict: {'mag': path, 'others': [path, path, ...]}
    Rules: 
     - Mag: contains 'mag' or 'magnitude' (case insensitive), excludes 'phase','v'.
     - Others: Everything else ending in .nii.gz
    """
    files = glob.glob(os.path.join(folder, "*.nii.gz"))
    if not files:
        files = glob.glob(os.path.join(folder, "*.nii"))
    
    mag_file = None
    velocity_files = []
    
    for f in files:
        fname = os.path.basename(f).lower()
        
        # Check for Magnitude
        is_mag = False
        if ("magnitude" in fname or "mag" in fname) and \
           not any(x in fname for x in ["phase", "vx", "vy", "vz", "velocity", "flow_x", "flow_y", "flow_z"]):
             is_mag = True
             
        if is_mag:
            # If we have multiple mags, pick the first one 
            if mag_file is None:
                mag_file = f
            # else: print(f"Warning: Multiple Mag files in {folder}")
        else:
            velocity_files.append(f)
            
    return mag_file, velocity_files

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
        # Heuristic: if dir has NIfTIs, it's a patient dir
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
        mag_path, other_paths = identify_images(p_dir)
        
        rel_dir = os.path.relpath(p_dir, args.input_dir)
        target_dir = os.path.join(args.output_dir, rel_dir)
        
        # Update progress bar
        short_name = os.path.basename(p_dir)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(folder=short_name[:15])

        if not mag_path:
            # Skip if no magnitude found to drive registration
            # or just print a warning and skip
            continue

        # Paths
        mag_out = os.path.join(target_dir, os.path.basename(mag_path))
        qc_dir = os.path.join(target_dir, "QC_Registration")

        # Step A: Register Magnitude
        # ----------------------------
        # Only print inside if not using tqdm, or use pbar.write
        if type(pbar).__name__ != 'tqdm':
             print(f"\nProcessing {short_name}...")

        transforms = register_4d_nifti(
            input_path=mag_path,
            output_path=mag_out,
            qc_dir=qc_dir,
            ref_t=0,
            reg_type=args.reg_type
        )
        
        if transforms is None:
            continue

        # Step B: Apply to Others (Velocity/Phase)
        # ----------------------------
        for vel_path in other_paths:
            vel_out = os.path.join(target_dir, os.path.basename(vel_path))
            # Optional: ignore files if they are just processed inputs or derivatives
            
            ok = apply_transforms_to_file(
                input_path=vel_path,
                output_path=vel_out,
                transforms_map=transforms,
                ref_img_path=mag_path 
            )

    print("\nBatch Processing Complete.")

if __name__ == "__main__":
    main()