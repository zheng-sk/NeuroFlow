# 4DFlowNet `code/` Instructions

## Nueva organización por propósito

```text
code/
├── conversion/            # DICOM/NIfTI/H5 conversion workflows
├── preprocessing/         # Derived feature generation
├── registration/          # Temporal + inter-scan registration and QC
├── inference/             # Model prediction workflows
└── *.py                   # Backward-compatible wrappers (old entrypoints)
```

## Flujo recomendado

1. **DICOM -> NIfTI**
   - Script principal: `code/conversion/dicom_to_nifti.py`
   - Wrapper legado: `code/convert2nifti2.py`
2. **Features derivados (speed / PC-MRA)**
   - Script: `code/preprocessing/calculate_mag.py`
3. **Registro temporal (motion correction)**
   - Por carpeta/paciente: `code/registration/batch_register_magnitude.py`
   - Por archivo 4D: `code/registration/temporal_register_to_t0.py`
4. **(Opcional) NIfTI -> H5**
   - Script: `code/conversion/nifti_to_h5.py`
5. **(Opcional) Predicción batch**
   - Script: `code/inference/batch_predict.py`

## Entrypoints legados (compatibles)

Puedes seguir usando los nombres antiguos en `code/`:
- `convert2nifti2.py`, `calculate_mag.py`, `batch_register_magnitude.py`,
  `batch_temporal_register.py`, `temporal_register_to_t0.py`,
  `convert_to_h5.py`, `batch_predict.py`, `register_7T_to_3T_with_qc.py`.

Estos wrappers redirigen a los módulos nuevos.

## Ejemplos mínimos

```bash
# 1) DICOM -> NIfTI
python code/conversion/dicom_to_nifti.py \
  --input-root data/sorted_patients \
  --output-root data/nifti_patients

# 2) Speed + PC-MRA
python code/preprocessing/calculate_mag.py \
  --csv data/dataset.csv \
  --data-root data

# 3a) Registro temporal por carpeta
python code/registration/batch_register_magnitude.py \
  --input-dir data/processed_inputs \
  --output-dir data/registered_patients \
  --reg-type Rigid

# 3b) Registro temporal por archivo
python code/registration/temporal_register_to_t0.py \
  --input data/processed_inputs/CASE/input_mag_raw.nii.gz \
  --out data/registered_patients/CASE/input_mag_raw.nii.gz \
  --qc_dir data/registered_patients/CASE/QC
```
