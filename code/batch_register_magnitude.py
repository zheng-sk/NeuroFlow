#!/usr/bin/env python3
"""
batch_register_magnitude.py

Batch processes a folder of NIfTI images, extracting ONLY generic Magnitude images
and performing temporal registration (motion correction) on them.

Filters:
  - FileName MUST contain: "Magnitude" or "Mag" (case insensitive)
  - FileName MUST NOT contain: "Phase", "Vx", "Vy", "Vz", "Flow"
"""

import os
import argparse
import glob
from registration_core import register_4d_nifti

# Try to import tqdm for progress bar, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit=""):
        print(f"--- {desc} ---")
        return iterable

def is_magnitude_image(filepath):
    """
    Determines if a file is a Magnitude image based on naming conventions.
    Exclude Phase or velocity component files.
    """
    fname = os.path.basename(filepath).lower()
    
    # Must have magnitude indicator
    if "magnitude" not in fname and "mag" not in fname:
        return False
        
    # Must NOT have phase/velocity indicators
    # Common 4D Flow patterns found in EDA: "Phase_Vx", "Vx.nii", "fl3d1_v..."
    forbidden = ["phase", "vx", "vy", "vz", "velocity", "flow_x", "flow_y", "flow_z"]
    
    for term in forbidden:
        if term in fname:
            return False
            
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch Temporal Registration (Magnitude Only).")
    parser.add_argument("--input-dir", required=True, help="Root directory containing input NIfTIs")
    parser.add_argument("--output-dir", required=True, help="Root directory to save registered NIfTIs")
    parser.add_argument("--reg-type", default="Rigid", help="Registration type (Rigid, Affine). Default: Rigid")
    parser.add_argument("--recursive", action="store_true", help="Search recursively in subdirectories")
    
    args = parser.parse_args()

    # 1. Find all NIfTI files
    pattern = "**/*.nii.gz" if args.recursive else "*.nii.gz"
    search_path = os.path.join(args.input_dir, pattern)
    all_files = glob.glob(search_path, recursive=args.recursive)
    
    # 2. Filter for Magnitude only
    mag_files = [f for f in all_files if is_magnitude_image(f)]
    mag_files.sort()

    if not mag_files:
        print(f"No Magnitude files found in {args.input_dir} matching criteria.")
        return

    print("="*50)
    print(f"MAGNITUDE TEMPORAL REGISTRATION")
    print("="*50)
    print(f"Input Directory : {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"files found     : {len(all_files)}")
    print(f"Magnitude files : {len(mag_files)} (to be processed)")
    print(f"Ref Frame       : t=0")
    print("="*50)

    success_count = 0
    fail_count = 0

    # 3. Process Loop with Progress Bar
    # 'unit' param is purely cosmetic for tqdm
    pbar = tqdm(mag_files, desc="Registering", unit="vol")
    
    for input_path in pbar:
        # Build output path (maintain folder structure)
        rel_path = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        # QC directory per file
        qc_subdir = f"QC_{os.path.basename(input_path).replace('.nii.gz', '')}"
        qc_path = os.path.join(os.path.dirname(output_path), qc_subdir)
        
        # Update progress bar description with current file name (shortened)
        short_name = os.path.basename(input_path)
        if len(short_name) > 30: short_name = short_name[:27] + "..."
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(file=short_name)
        
        # Run registration
        ok = register_4d_nifti(
            input_path=input_path,
            output_path=output_path,
            qc_dir=qc_path,
            ref_t=0,
            reg_type=args.reg_type
        )
        
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*50)
    print(f"DONE. Success: {success_count} | Failed: {fail_count}")
    print("="*50)

if __name__ == "__main__":
    main()