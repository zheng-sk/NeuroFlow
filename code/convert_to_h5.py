import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch convert processed NIfTIs to HDF5 for prediction.")
    parser.add_argument("--input-root", type=str, default="../data/processed_inputs", help="Folder with processed patient subfolders")
    parser.add_argument("--output-dir", type=str, default="../data/h5_inputs", help="Folder to save .h5 files")
    parser.add_argument("--script-path", type=str, default="../src/prepare_data/prepare_nifti_data.py", help="Path to prepare_nifti_data.py")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_root):
        print(f"Error: Input directory not found: {args.input_root}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Get list of patient folders
    patient_ids = [d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))]
    patient_ids.sort()

    print(f"Found {len(patient_ids)} patients in {args.input_root}")

    for pid in patient_ids:
        print(f"\n--- Converting: {pid} ---")
        
        p_dir = os.path.join(args.input_root, pid)
        output_h5 = f"{pid}.h5"
        
        # Define paths to expected files (from our previous step)
        u_path = os.path.join(p_dir, "Vx.nii.gz")
        v_path = os.path.join(p_dir, "Vy.nii.gz")
        w_path = os.path.join(p_dir, "Vz.nii.gz")
        mag_path = os.path.join(p_dir, "input_mag_raw.nii.gz")
        
        # Check existence
        if not all(os.path.exists(f) for f in [u_path, v_path, w_path, mag_path]):
            print(f"Skipping {pid}: Missing required NIfTI files in {p_dir}")
            continue
        
        cmd = [
            sys.executable, args.script_path,
            "--u", u_path,
            "--v", v_path,
            "--w", w_path,
            "--mag", mag_path,
            "--venc", "0.90",
            "--output-dir", args.output_dir,
            "--output-filename", output_h5
        ]
        
        print("Running preparation script...")
        try:
            subprocess.run(cmd, check=True)
            print(f"--> Created: {os.path.join(args.output_dir, output_h5)}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pid}: {e}")

    print("\nBatch conversion completed.")

if __name__ == "__main__":
    main()
