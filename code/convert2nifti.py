import os
import glob
import numpy as np
import nibabel as nib
import pydicom
import subprocess
import shutil

def lectura_directa_dicom_4d(carpeta_dicom, nifti_referencia_path, salida_path):
    """
    Read all DICOM files in the folder, skip dcm2niix, and build the 4D volume
    via direct pixel reads. Uses 'nifti_referencia_path' only to copy spatial
    orientation (affine).
    """
    files = glob.glob(os.path.join(carpeta_dicom, "*"))
    files = [f for f in files if f.upper().endswith(('.IMA', '.DCM'))]
    
    if len(files) == 0: return False
    
    print(f"      -> Reading {len(files)} DICOM files directly...")
    
    # 1. Read key metadata for sorting
    # Structure: list of tuples (Z_position, Time_info, Pixel_Data)
    data_list = []
    
    try:
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=False) # Read all dicom
            
            # Z position (Slice Location)
            # ImagePositionPatient is [x, y, z]
            pos = ds.get("ImagePositionPatient", [0,0,0])
            z = float(pos[2])
            
            # Time (TriggerTime or InstanceNumber as fallback)
            t = float(ds.get("TriggerTime", 0))
            instancia = int(ds.get("InstanceNumber", 0))
            
            # Store pixel data (transpose because DICOM is (Y,X) and NIfTI is (X,Y))
            pixel_data = ds.pixel_array.T 
            
            data_list.append({
                'z': z,
                't': t,
                'i': instancia,
                'data': pixel_data
            })
            
        # 2. Identify dimensions
        # Unique Z values sorted
        z_unicos = sorted(list(set([d['z'] for d in data_list])))
        num_z = len(z_unicos)
        
        # Compute T based on total
        if len(files) % num_z != 0:
            print("      [!] Warning: total images do not perfectly match the slices.")
            # Continue with best effort
            
        num_t = len(files) // num_z
        print(f"      -> Detected geometry: {num_z} slices x {num_t} timepoints")
        
        # 3. Build 4D matrix
        # Spatial dimensions taken from the first frame
        dim_x, dim_y = data_list[0]['data'].shape
        
        # Create empty volume (X, Y, Z, T)
        if num_t > 1:
            vol_4d = np.zeros((dim_x, dim_y, num_z, num_t))
        else:
            vol_4d = np.zeros((dim_x, dim_y, num_z)) # Static 3D case
            
        # Organize data
        # First group by Z
        from collections import defaultdict
        dicoms_by_z = defaultdict(list)
        for item in data_list:
            dicoms_by_z[item['z']].append(item)
            
        # Fill matrix
        for i, z_val in enumerate(z_unicos):
            # Get all timepoints for this Z slice
            grupo_t = dicoms_by_z[z_val]
            
            # Sort by time (using TriggerTime 't', or InstanceNumber 'i' when T is equal)
            grupo_t.sort(key=lambda x: (x['t'], x['i']))
            
            for j, item in enumerate(grupo_t):
                if j >= num_t: break
                
                if num_t > 1:
                    vol_4d[:, :, i, j] = item['data']
                else:
                    vol_4d[:, :, i] = item['data']
                    
        # 4. Save NIfTI
        # Use the affine from the reference NIfTI (the one dcm2niix built but with good orientation)
        if os.path.exists(nifti_referencia_path):
            ref = nib.load(nifti_referencia_path)
            affine = ref.affine
            header = ref.header
        else:
            # Simple fallback (identity) if there is no reference; may be rotated
            print("      [!] No NIfTI reference found, using identity affine (may be rotated).")
            affine = np.eye(4)
            header = None
            
        new_img = nib.Nifti1Image(vol_4d, affine, header)
        nib.save(new_img, salida_path)
        print(f"      -> Success! Saved volume {vol_4d.shape}")
        return True
        
    except Exception as e:
        print(f"      [Direct Read Error] {e}")
        return False

def procesar_v11_final(input_root, output_root, dcm2niix_path="dcm2niix"):
    print("--- Processing V11: Bypass dcm2niix for 3T data ---")
    
    try:
        subprocess.run([dcm2niix_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        print("Error: dcm2niix not found.")
        return

    pacientes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for i, paciente in enumerate(pacientes):
        print(f"[{i+1}/{len(pacientes)}] {paciente}...")
        path_dicom_paciente = os.path.join(input_root, paciente)
        path_nifti_paciente = os.path.join(output_root, paciente)
        
        if not os.path.exists(path_nifti_paciente): os.makedirs(path_nifti_paciente)
        
        series = [s for s in os.listdir(path_dicom_paciente) if os.path.isdir(os.path.join(path_dicom_paciente, s)) and "Unknown" not in s]
        
        for serie in series:
            dicom_serie_path = os.path.join(path_dicom_paciente, serie)
            nifti_final_path = os.path.join(path_nifti_paciente, f"{serie}.nii.gz")
            
            # Count actual DICOM files
            num_dicoms = len([f for f in os.listdir(dicom_serie_path) if f.endswith('.IMA') or f.endswith('.DCM')])
            
            # STEP 1: Try standard conversion with dcm2niix (to get the correct affine)
            # Use a temp folder
            temp_dir = os.path.join(path_nifti_paciente, "temp_" + serie)
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            
            # Run plain dcm2niix
            cmd = [dcm2niix_path, "-z", "y", "-f", "ref", "-o", temp_dir, dicom_serie_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            generated = glob.glob(os.path.join(temp_dir, "*.nii.gz"))
            
            # STEP 2: Check if dcm2niix failed (significant data loss)
            usar_lectura_directa = False
            
            if len(generated) > 0:
                # Load to inspect dimensions
                try:
                    img_ref = nib.load(generated[0])
                    # If there are many DICOMs (e.g., >100) but the NIfTI has small Z depth (e.g., <50) and T=1
                    # it means data was lost.
                    shape = img_ref.shape
                    slices_nifti = shape[2] if len(shape) >= 3 else 1
                    time_nifti = shape[3] if len(shape) >= 4 else 1
                    
                    total_volumen = slices_nifti * time_nifti
                    
                    # Criterion: If the NIfTI has less than half the "logical voxels" than DICOM files
                    if num_dicoms > 50 and total_volumen < (num_dicoms * 0.9): 
                        print(f"   [dcm2niix failure detected] DICOMs={num_dicoms} vs NIfTI={shape}. Switching to manual mode.")
                        usar_lectura_directa = True
                    else:
                        # 7T case or simple anatomy: looks correct
                        shutil.move(generated[0], nifti_final_path)
                        print(f"   [OK] {serie}")
                except:
                    usar_lectura_directa = True
            else:
                usar_lectura_directa = True
                
            # STEP 3: Execute direct read if needed
            if usar_lectura_directa:
                ref_path = generated[0] if len(generated) > 0 else "None"
                success = lectura_directa_dicom_4d(dicom_serie_path, ref_path, nifti_final_path)
                if not success:
                    print(f"   [!] Critical failure in {serie}")

            # Cleanup
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)


# --- CONFIGURATION ---
ruta_dicom_ordenada = r"../Data/Disease Patients/sorted_patients"
ruta_nifti_destino = r"../Data/Disease Patients/nifti_patients"
ruta_ejecutable = "dcm2niix" # O la ruta completa si no está en PATH

if __name__ == "__main__":
    procesar_v11_final(ruta_dicom_ordenada, ruta_nifti_destino, ruta_ejecutable)