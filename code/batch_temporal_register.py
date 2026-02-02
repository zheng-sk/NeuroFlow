#!/usr/bin/env python3
"""
batch_temporal_register.py

Batch processes a folder of 4D NIfTI images, performing temporal registration
on each one relative to its first frame (t=0).

Usage:
    python batch_temporal_register.py \
        --input-dir ../data/nifti_patients \
        --output-dir ../data/registered_patients \
        --pattern "*.nii.gz" \
        --recursive
"""

import os
import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor # If parallelization is desired later
try:
    from tqdm import tqdm
except ImportError:
    # A simple fallback if tqdm is not installed
    def tqdm(iterable, desc=""):
        print(f"--- {desc} ---")
        return iterable

from registration_core import register_4d_nifti

def main():
    parser = argparse.ArgumentParser(description="Batch Temporal Registration for 4D NIfTI data.")
    parser.add_argument("--input-dir", required=True, help="Root directory containing input NIfTIs")
    parser.add_argument("--output-dir", required=True, help="Root directory to save registered NIfTIs")
    parser.add_argument("--pattern", default="*.nii.gz", help="File pattern to match (default: *.nii.gz)")
    parser.add_argument("--reg-type", default="Rigid", help="ANTs registration type (Rigid, Affine, SyN)")
    parser.add_argument("--recursive", action="store_true", help="Search recursively in subdirectories")
    parser.add_argument("--ref-t", type=int, default=0, help="Reference time frame index (default: 0)")
    
    args = parser.parse_args()

    # 1. Find files
    search_path = os.path.join(args.input_dir, "**", args.pattern) if args.recursive else os.path.join(args.input_dir, args.pattern)
    files = glob.glob(search_path, recursive=args.recursive)
    files.sort()

    if not files:
        print(f"No files found matching '{args.pattern}' in {args.input_dir}")
        return

    print(f"Found {len(files)} files to process.")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.reg_type} | Reference Frame: t={args.ref_t}")

    # 2. Process loop
    success_count = 0
    fail_count = 0
    
    # Using tqdm for progress tracking
    pbar = tqdm(files, desc="Registering 4D Patients", unit="file")
    
    for input_path in pbar:
        # Construct output path preserving relative structure
        rel_path = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        # Directory for QC images specific to this file
        # Example: registered_patients/Patient001/QC_FileName/
        qc_subdir_name = f"QC_{os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')}"
        qc_dir_path = os.path.join(os.path.dirname(output_path), qc_subdir_name)

        # Update progress bar description
        pbar.set_postfix({"current": os.path.basename(input_path)[:20]})
        
        # Execute registration
        # Note: We pass reg_type and other parameters
        ok = register_4d_nifti(
            input_path=input_path,
            output_path=output_path,
            qc_dir=qc_dir_path,
            ref_t=args.ref_t,
            reg_type=args.reg_type,
            interpolator="linear",
            qc_frames_str="0,50,99" # Sample default frames for QC
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total Files: {len(files)}")
    print(f"Successful : {success_count}")
    print(f"Failed     : {fail_count}")

if __name__ == "__main__":
    main()