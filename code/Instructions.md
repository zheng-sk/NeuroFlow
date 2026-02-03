# 4DFlowNet `code/` Instructions

## Recommended execution order

1. **Convert DICOM to NIfTI** using `convert2nifti2.py`  
   (`convert2nifti.py` is deprecated and only kept as a compatibility wrapper).
2. **Compute magnitude-derived inputs** using `calculate_mag.py`.
3. **Run temporal registration to the magnitude reference** using either:
   - `batch_register_magnitude.py` (recommended for full patient folders: magnitude + velocity + other scalars), or
   - `temporal_register_to_t0.py` (single 4D file workflow).
4. *(Optional)* Convert prepared NIfTI inputs to HDF5 with `convert_to_h5.py`.

---

## What each script does

- `convert2nifti2.py`  
  Main DICOM-to-NIfTI converter. Tries `dcm2niix` first, then falls back to direct DICOM reading when needed. Standardizes orientation and keeps phase-component handling safe.

- `convert2nifti.py`  
  Deprecated wrapper. It runs `convert2nifti2.py` logic for backward compatibility.

- `calculate_mag.py`  
  Reads Vx/Vy/Vz/Magnitude NIfTIs from a CSV manifest, computes:
  - `input_speed_raw.nii.gz` = `sqrt(Vx^2 + Vy^2 + Vz^2)`
  - `input_pcmra_raw.nii.gz` = `Magnitude * Speed`
  and writes standardized per-case outputs.

- `registration_core.py`  
  Shared registration utilities:
  - estimate temporal transforms from a magnitude 4D sequence,
  - apply those transforms to scalar images,
  - apply those transforms to vector velocity triplets with proper vector handling.

- `temporal_register_to_t0.py`  
  Registers a single 4D NIfTI to reference frame `t=0` (or custom `--ref_t`) and writes QC plots.

- `batch_temporal_register.py`  
  Batch wrapper for running temporal registration over many 4D NIfTI files.

- `batch_register_magnitude.py`  
  Patient-folder workflow that uses the **magnitude file as transform source** and propagates transforms to velocity/scalar companions.

- `convert_to_h5.py`  
  Converts prepared NIfTI inputs into `.h5` files for model-ready pipelines.

- `batch_predict.py`  
  Batch inference helper for model predictions.

---

## Minimal command examples

```bash
# 1) DICOM -> NIfTI
python code/convert2nifti2.py \
  --input-root ../data/sorted_patients \
  --output-root ../data/nifti_patients

# 2) Derived magnitude inputs
python code/calculate_mag.py \
  --csv ../data/dataset.csv \
  --data-root ../data

# 3a) Registration by patient folder (recommended)
python code/batch_register_magnitude.py \
  --input-dir ../data/processed_inputs \
  --output-dir ../data/registered_patients \
  --reg-type Rigid

# 3b) Registration for generic 4D files
python code/batch_temporal_register.py \
  --input-dir ../data/processed_inputs \
  --output-dir ../data/registered_patients \
  --recursive
```
