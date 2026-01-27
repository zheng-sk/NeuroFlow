import nibabel as nib
import numpy as np
import os
import shutil
import argparse
import csv
import sys
import time

def process_patient(u_path, v_path, w_path, mag_path, output_dir, patient_id):
    print(f"\n--- Processing Case: {patient_id} ---")
    start_time = time.time()
    
    # Check if files exist
    files = [u_path, v_path, w_path, mag_path]
    names = ['Vx', 'Vy', 'Vz', 'Mag']
    missing = []
    for f, n in zip(files, names):
        if not os.path.exists(f):
            missing.append(f"{n}: {f}")
    
    if missing:
        print("X Error: Missing input files:")
        for m in missing:
            print(f"  - {m}")
        return False

    # 1. Load Data
    print("   -> Loading NIfTI files...")
    try:
        img_u = nib.load(u_path)
        img_v = nib.load(v_path)
        img_w = nib.load(w_path)
        img_mag = nib.load(mag_path)
    except Exception as e:
        print(f"X Error loading files: {e}")
        return False

    # Get data as float32
    u = img_u.get_fdata().astype(np.float32)
    v = img_v.get_fdata().astype(np.float32)
    w = img_w.get_fdata().astype(np.float32)
    mag = img_mag.get_fdata().astype(np.float32)
    
    affine = img_u.affine 
    header = img_u.header

    # Check dimensions
    if u.shape != v.shape or u.shape != w.shape or u.shape != mag.shape:
        print(f"X Error: Mismatch in dimensions.")
        print(f"  U:{u.shape}, V:{v.shape}, W:{w.shape}, Mag:{mag.shape}")
        return False

    # 2. Calculate Speed and PC-MRA
    print("   -> Calculating Speed (sqrt(u^2+v^2+w^2))...")
    speed = np.sqrt(u**2 + v**2 + w**2)
    
    print("   -> Calculating PC-MRA (Mag * Speed)...")
    pcmra = mag * speed

    # 3. Save
    if not os.path.exists(output_dir):
        print(f"   -> Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    def save_nii(data, filename):
        out_path = os.path.join(output_dir, filename)
        new_img = nib.Nifti1Image(data, affine, header)
        nib.save(new_img, out_path)
        return out_path
    
    print("   -> Saving output files...")
    shutil.copy2(u_path, os.path.join(output_dir, "Vx.nii.gz"))
    print(f"      Copied: {os.path.join(output_dir, 'Vx.nii.gz')}")
    shutil.copy2(v_path, os.path.join(output_dir, "Vy.nii.gz"))
    print(f"      Copied: {os.path.join(output_dir, 'Vy.nii.gz')}")
    shutil.copy2(w_path, os.path.join(output_dir, "Vz.nii.gz"))
    print(f"      Copied: {os.path.join(output_dir, 'Vz.nii.gz')}")
    
    # Save Calculated/Anatomical components
    save_nii(mag, "input_mag_raw.nii.gz")
    save_nii(speed, "input_speed_raw.nii.gz")
    save_nii(pcmra, "input_pcmra_raw.nii.gz")
    
    elapsed = time.time() - start_time
    print(f"   [OK] Completed in {elapsed:.2f} seconds.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Calculates Speed Calculate Speed and PC-MRA from dataset.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset.csv")
    parser.add_argument("--data-root", type=str, default="../data", help="Root folder to prepend to relative paths in CSV (default: ../data)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return

    print(f"Reading Dataset: {args.csv}")
    print(f"Data Root: {args.data_root}")
    
    # Map CSV column names to code variables
    # CSV: Case_ID, Path_Mag, Path_Vx, Path_Vy, Path_Vz
    col_map = {
        'id': 'Case_ID',
        'mag': 'Path_Mag',
        'u': 'Path_Vx',
        'v': 'Path_Vy',
        'w': 'Path_Vz'
    }

    success_count = 0
    fail_count = 0
    
    with open(args.csv, mode='r') as f:
        reader = csv.DictReader(f)
        
        # Strip whitespace from headers
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        # Verify columns
        missing = [c for k, c in col_map.items() if c not in reader.fieldnames]
        if missing:
            print(f"Error: CSV missing columns: {missing}")
            return

        rows = list(reader)
        total_patients = len(rows)
        print(f"Found {total_patients} patients to process.\n")
        
        for i, row in enumerate(rows):
            print(f"Progress: [{i+1}/{total_patients}]")
            
            # Construct absolute paths
            # Assuming paths in CSV are relative to data folder (e.g. nifti_patients/...)
            def get_path(col_name):
                rel_path = row[col_map[col_name]].strip()
                # If path starts with /, assume absolute, else join with root
                if os.path.isabs(rel_path):
                    return rel_path
                return os.path.join(args.data_root, rel_path)

            case_id = row[col_map['id']].strip()
            
            # Create a specific output folder per patient inside 'sorted_patients' or similar
            # Or assume we save it back to the source folder? 
            # Request implies just generating inputs. Let's create a 'processed_inputs' folder parallel to nifti_patients
            # or just use the same folder structure.
            
            # Let's define output dir based on Case_ID to keep it organized
            # Ex: ../data/processed_inputs/001_20240313_3T/
            output_dir = os.path.join(args.data_root, "processed_inputs", case_id)

            ok = process_patient(
                get_path('u'), 
                get_path('v'), 
                get_path('w'), 
                get_path('mag'), 
                output_dir,
                case_id
            )
            
            if ok:
                success_count += 1
            else:
                fail_count += 1
                
    print(f"\n--- Summary ---")
    print(f"Total: {total_patients}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
