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
   - Progress/timing options: `--show-frame-progress`, `--verbose`, `--timing-report <csv_path>`
4. **Inter-scan 7T -> 3T registration**
   - Per case: `code/registration/register_7T_to_3T_with_qc.py`
   - Batch over a folder: `code/registration/batch_register_7T_to_3T.py`
   - Batch defaults expect temporally-registered velocity names: `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`
5. **Full registration pipeline (temporal + 7T -> 3T)**
   - One command over a folder: `code/registration/batch_full_register_7T_to_3T.py`
   - Saves final registered images and brain masks (`BrainMasks/`), and can auto-clean intermediates/QC
   - If `temporal-dir` already has the required files, temporal stage is skipped automatically
   - Use `--force-temporal` to force recomputation of temporal registration
   - Optional `--final-only` validates that all final 4D registered outputs exist per case before cleanup
   - Optional temporal telemetry: `--show-temporal-frame-progress` and `--temporal-timing-report <csv_path>`
6. **(Optional) Export paired LR/HR NIfTI dataset**
   - Script: `code/registration/export_paired_lr_hr_dataset.py`
   - Creates two folders: `lr_3t/` (temporally registered 3T) and `hr_7t_in_3t/` (7T registered into 3T space)
   - Writes paired CSV compatible with `data/nifti_cases_template.csv`
7. **(Optional) NIfTI -> H5**
   - Script: `code/conversion/nifti_to_h5.py`
8. **(Optional) Phase range quality check**
   - Script: `code/registration/check_phase_ranges.py`
   - Validates whether phase files look like raw phase (`0..4096`) or velocity (m/s)
   - Prints per-file stats (`min/max/percentiles`) and summary, with optional CSV report
9. **(Optional) Batch inference**
   - Script: `code/inference/batch_predict.py`
   - Supports `--input-format h5|nifti` and `--output-format h5|nifti`
   - Direct NIfTI mode bypasses H5 conversion and writes `u_SR.nii.gz`, `v_SR.nii.gz`, `w_SR.nii.gz`
   - Defaults are connected to **temporally registered 3T (LowRes)** inputs: `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`, `input_mag_raw.nii.gz`
   - By default only `*_3T` folders are inferred (`--case-suffix _3T`)
   - For evaluation on high-res aligned data, override with `--case-suffix _7T` and custom filenames
   - Optional raw-phase preprocessing in direct NIfTI mode: `--auto-convert-raw-phase` with `--venc` (or per-axis VENC)
   - Progress/timing options: `--show-patch-progress`, `--verbose`, `--timing-report <csv_path>`
   - Device controls: `--device auto|cpu|gpu` and `--gpu-memory-limit-mb <int>`
   - Apple Metal safety: CPU fallback on GPU errors is enabled by default (disable with `--no-cpu-fallback-on-gpu-error`)

## Raw phase conversion behavior

- **H5 workflow (`nifti_to_h5`)**: raw phase to m/s conversion is applied during NIfTI -> H5 conversion in `src/prepare_data/prepare_nifti_data.py`.
- **Direct NIfTI inference workflow**: raw phase to m/s conversion is applied at prediction time only when `--auto-convert-raw-phase` is enabled.
- Both workflows support **unsigned raw phase** (`~0..4096`) and **signed raw phase** (`~-4096..4096` / `~-2048..2048`), using VENC.

## Legacy compatible entrypoints

You can still use the old script names in `code/`:
- `convert2nifti2.py`, `calculate_mag.py`, `batch_register_magnitude.py`,
  `batch_temporal_register.py`, `temporal_register_to_t0.py`,
  `convert_to_h5.py`, `batch_predict.py`, `register_7T_to_3T_with_qc.py`,
  `batch_register_7T_to_3T.py`, `batch_full_register_7T_to_3T.py`,
  `export_paired_lr_hr_dataset.py`.

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
  --reg-type Rigid \
  --show-frame-progress \
  --timing-report data/reports/temporal_timing.csv

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
  --phase-x-name Vx.nii.gz \
  --phase-y-name Vy.nii.gz \
  --phase-z-name Vz.nii.gz \
  --fixed-suffix _3T \
  --moving-suffix _7T

# 5) Full pipeline: temporal first, then 7T -> 3T batch
python code/registration/batch_full_register_7T_to_3T.py \
  --input-dir data/processed_inputs \
  --output-dir data/registered_7T_in_3T \
  --show-temporal-frame-progress \
  --temporal-timing-report data/reports/temporal_timing.csv \
  --phase-x-name Vx.nii.gz \
  --phase-y-name Vy.nii.gz \
  --phase-z-name Vz.nii.gz \
  --final-only \
  --fixed-suffix _3T \
  --moving-suffix _7T

# 6) Export paired LR/HR folders + CSV for training/prediction
python code/registration/export_paired_lr_hr_dataset.py \
  --temporal-dir data/registered_7T_in_3T/_temporal_registered \
  --registered-dir data/registered_7T_in_3T \
  --output-root data/paired_dataset \
  --mode copy \
  --venc 0.90 \
  --csv-path data/paired_dataset/paired_nifti_cases.csv

# 7) Validate phase ranges before conversion/prediction
python code/registration/check_phase_ranges.py \
  --input-root data/registered_7T_in_3T/_temporal_registered \
  --recursive \
  --expected-mode velocity \
  --venc 0.90 \
  --csv-report data/reports/phase_range_check.csv \
  --verbose

# 7b) Validate signed raw phase ranges (e.g., approximately -4096..4096)
python code/registration/check_phase_ranges.py \
  --input-root data/paired_dataset/hr_7t_in_3t \
  --recursive \
  --expected-mode raw \
  --raw-min -4096 \
  --raw-max 4096 \
  --csv-report data/reports/phase_range_check_raw_signed.csv

# 8) Prepare H5 inputs from paired LR NIfTI (for H5 prediction workflow)
python code/conversion/nifti_to_h5.py \
  --input-root data/paired_dataset/lr_3t \
  --output-dir data/paired_dataset/h5_lr \
  --venc 0.90

# 9) H5 prediction (batch, CUDA GPU)
python code/inference/batch_predict.py \
  --input-format h5 \
  --output-format h5 \
  --input-dir data/paired_dataset/h5_lr \
  --h5-pattern "*_3T.h5" \
  --output-dir data/predictions_h5_3t_gpu \
  --batch-size 8 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_h5_3t_gpu.csv \
  --device gpu

# 10) H5 prediction (batch, CPU fallback-safe)
python code/inference/batch_predict.py \
  --input-format h5 \
  --output-format h5 \
  --input-dir data/paired_dataset/h5_lr \
  --h5-pattern "*_3T.h5" \
  --output-dir data/predictions_h5_3t_cpu \
  --batch-size 2 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_h5_3t_cpu.csv \
  --device cpu

# 11) Direct NIfTI prediction (batch, 3T only, CUDA GPU, SR x2)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_sr2 \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_3t_gpu_sr2.csv \
  --device gpu

# 12) Direct NIfTI prediction (batch, 3T only, same-resolution mode SR x1)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_sr1 \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 1 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_3t_gpu_sr1.csv \
  --device gpu

# 13) Direct NIfTI prediction with raw-phase conversion enabled (use only for raw phase inputs)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_raw \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 2 \
  --show-patch-progress \
  --timing-report data/reports/predict_nifti_3t_gpu_raw.csv \
  --device gpu \
  --auto-convert-raw-phase \
  --venc 0.90

# 14) Single-case direct NIfTI prediction (quick debug)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/registered_7T_in_3T/_temporal_registered/001_20240313_3T \
  --output-dir data/predictions_nifti_single \
  --batch-size 1 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_single.csv \
  --device gpu
```
