# 4DFlowNet `code/` Instructions

## Project layout by purpose

```text
code/
├── conversion/            # DICOM/NIfTI/H5 conversion workflows
├── preprocessing/         # Derived feature generation
├── registration/          # Temporal + inter-scan registration and QC
├── inference/             # Model prediction workflows
└── *.py                   # Backward-compatible wrappers (legacy entrypoints)
```

## Recommended workflow

1. **DICOM -> NIfTI**
   - Main script: `code/conversion/dicom_to_nifti.py`
   - Legacy wrapper: `code/convert2nifti2.py`
2. **Derived features (speed / PC-MRA)**
   - Script: `code/preprocessing/calculate_mag.py`
3. **Temporal registration (motion correction)**
   - Per folder/patient: `code/registration/batch_register_magnitude.py`
   - Per 4D file: `code/registration/temporal_register_to_t0.py`
4. **Inter-scan 7T -> 3T registration**
   - Per case: `code/registration/register_7T_to_3T_with_qc.py`
   - Batch over a folder: `code/registration/batch_register_7T_to_3T.py`
5. **Full registration pipeline (temporal + 7T -> 3T)**
   - One command over a folder: `code/registration/batch_full_register_7T_to_3T.py`
   - Saves final registered images and brain masks (`BrainMasks/`), and can auto-clean intermediates/QC
   - Optional `--final-only` validates that all final 4D registered outputs exist per case before cleanup
6. **(Optional) NIfTI -> H5**
   - Script: `code/conversion/nifti_to_h5.py`
7. **(Optional) Batch inference**
   - Script: `code/inference/batch_predict.py`

## Legacy compatible entrypoints

You can still use the old script names in `code/`:
- `convert2nifti2.py`, `calculate_mag.py`, `batch_register_magnitude.py`,
  `batch_temporal_register.py`, `temporal_register_to_t0.py`,
  `convert_to_h5.py`, `batch_predict.py`, `register_7T_to_3T_with_qc.py`,
  `batch_register_7T_to_3T.py`, `batch_full_register_7T_to_3T.py`.

These wrappers redirect to the reorganized modules.

## Minimal examples

```bash
# 1) DICOM -> NIfTI
python code/conversion/dicom_to_nifti.py \
  --input-root data/sorted_patients \
  --output-root data/nifti_patients

# 2) Speed + PC-MRA
python code/preprocessing/calculate_mag.py \
  --csv data/dataset.csv \
  --data-root data

# 3a) Temporal registration per folder
python code/registration/batch_register_magnitude.py \
  --input-dir data/processed_inputs \
  --output-dir data/registered_patients \
  --reg-type Rigid

# 3b) Temporal registration per file
python code/registration/temporal_register_to_t0.py \
  --input data/processed_inputs/CASE/input_mag_raw.nii.gz \
  --out data/registered_patients/CASE/input_mag_raw.nii.gz \
  --qc_dir data/registered_patients/CASE/QC

# 4a) Inter-scan registration (single case, 7T -> 3T)
python code/registration/register_7T_to_3T_with_qc.py \
  --fixed_mag_3t data/temporal_registered/001_20240313_3T/input_mag_raw.nii.gz \
  --moving_mag_7t data/temporal_registered/001_20240313_7T/input_mag_raw.nii.gz \
  --moving_phase_x data/temporal_registered/001_20240313_7T/input_phase_x_raw.nii.gz \
  --moving_phase_y data/temporal_registered/001_20240313_7T/input_phase_y_raw.nii.gz \
  --moving_phase_z data/temporal_registered/001_20240313_7T/input_phase_z_raw.nii.gz \
  --out_dir data/registered_7T_in_3T/001_20240313 \
  --qc_dir data/registered_7T_in_3T/001_20240313/QC

# 4b) Inter-scan registration (batch)
python code/registration/batch_register_7T_to_3T.py \
  --input-dir data/temporal_registered \
  --output-dir data/registered_7T_in_3T \
  --fixed-suffix _3T \
  --moving-suffix _7T

# 5) Full pipeline: temporal first, then 7T -> 3T batch
python code/registration/batch_full_register_7T_to_3T.py \
  --input-dir data/processed_inputs \
  --output-dir data/registered_7T_in_3T \
  --final-only \
  --fixed-suffix _3T \
  --moving-suffix _7T
```
