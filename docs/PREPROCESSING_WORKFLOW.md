# Preprocessing Workflow (Step-by-Step)

This is the current pipeline you requested:

`DICOM->NIfTI -> calculate_mag -> temporal registration -> CoW ROI detection/crop -> inter-scan registration -> CoW segmentation`

If you already have NIfTI, start at Step 2.

## 0) Expected Folder Layout

Recommended pair naming (used by batch scripts):

- `<CASE>_3T` for fixed / LR
- `<CASE>_7T` for moving / HR source

Example:

```text
data/
  sorted_patients/
    001_20240313_3T/
    001_20240313_7T/
```

## 1) DICOM -> NIfTI

Script:

- `code/conversion/dicom_to_nifti.py`

Core behavior:

1. Tries `dcm2niix` first.
2. If needed, falls back to direct DICOM 4D read.
3. Preserves raw phase intensity values (no LUT/rescale conversion).
4. Applies LPS->RAS component sign convention for phase components:

```text
Vx -> sign inversion
Vy -> sign inversion
Vz -> unchanged
```

For unsigned RAW phase, inversion is done around the raw center:

```text
out = 2*c - in, c in {2048, 4096}
```

Command:

```bash
python code/conversion/dicom_to_nifti.py \
  --input-root data/sorted_patients \
  --output-root data/nifti_patients \
  --canonicalize
```

## 2) Prepare Inputs (`calculate_mag`)

Script:

- `code/preprocessing/calculate_mag.py`

Purpose:

- Standardize per-case files used downstream.
- By default, this step mostly copies/re-saves required inputs:
  - `Vx.nii.gz`
  - `Vy.nii.gz`
  - `Vz.nii.gz`
  - `input_mag_raw.nii.gz`

Optional derived maps:

- speed:

```text
speed = sqrt(Vx^2 + Vy^2 + Vz^2)
```

- pcmra:

```text
pcmra = magnitude * speed
```

CSV required columns:

- `Case_ID`, `Path_Mag`, `Path_Vx`, `Path_Vy`, `Path_Vz`

Command:

```bash
python code/preprocessing/calculate_mag.py \
  --csv data/dataset.csv \
  --data-root data \
  --output-subdir processed_inputs
```

## 3) Temporal Registration (within each scan)

Recommended batch script:

- `code/registration/batch_register_magnitude.py`

Method:

1. Register magnitude 4D frame-by-frame to `t0`.
2. Reuse transforms on velocity triplet with vector mode (`imagetype=1`) to preserve vector geometry.

Per-frame QC metrics:

```text
MAD = mean(abs(I_ref - I_mov))
NCC = <a-a_mean, b-b_mean> / (||a-a_mean|| * ||b-b_mean||)
```

Command:

```bash
python code/registration/batch_register_magnitude.py \
  --input-dir data/processed_inputs \
  --output-dir data/temporal_registered \
  --reg-type Rigid \
  --show-frame-progress \
  --timing-report data/reports/temporal_timing.csv
```

## 4) CoW ROI Detection + Common Crop (3T/7T pair)

Script:

- `code/preprocessing/yolo_crop_patient_pairs.py`

Method:

1. Run YOLO on magnitude.
2. If magnitude is 4D, use temporal MIP before detection:

```text
MIP(x,y,z) = max_t M(x,y,z,t)
```

3. Build one common bbox enclosing detections from 3T and 7T.
4. Crop all NIfTI files from both folders using that same bbox.

Note: current implementation keeps full Z (X/Y crop only).

Command:

```bash
python code/preprocessing/yolo_crop_patient_pairs.py \
  --input-dir data/temporal_registered \
  --output-dir data/temporal_registered_cow_crop \
  --yolo-model models/yolo-cow-detection.pt \
  --fixed-suffix _3T \
  --moving-suffix _7T \
  --magnitude-name input_mag_raw.nii.gz
```

## 5) Inter-Scan Registration (7T -> 3T)

Batch script:

- `code/registration/batch_register_7T_to_3T.py`

Per-case engine:

- `code/registration/register_7T_to_3T_with_qc.py`

Method (estimation):

1. Build robust temporal references:

```text
fixed_ref  = median_t(3T_mag_4D)
moving_ref = median_t(7T_mag_4D)
```

2. Optional masks (`ants` or `hdbet`).
3. N4 bias correction.
4. Rician denoising.
5. Winsorization:

```text
I_w = clip(I, q01, q99)
```

6. Optional histogram matching.
7. Estimate transform on magnitude, apply to full 4D magnitude + phase triplet.

Phase warp modes:

- `direct`:

```text
P'(x) = P(T^-1(x))
```

- `complex`:

```text
R = M*cos(P)
I = M*sin(P)
R' = warp(R), I' = warp(I)
P' = atan2(I', R')
```

Command (recommended for cropped CoW inputs):

```bash
python code/registration/batch_register_7T_to_3T.py \
  --input-dir data/temporal_registered_cow_crop \
  --output-dir data/registered_7T_in_3T_cow_crop \
  --mask-method none \
  --phase-warp-mode direct \
  --interpolator-phase nearestNeighbor
```

## 6) Optional CoW Segmentation (after crop + inter-scan registration)

Single-case scripts:

- `code/segmentation/segment_cow_crops.py`
- `code/segmentation/segment_cow_patient_pipeline.py`

For batch case-by-case with angiography construction (MAG + Vx/Vy/Vz):

- `code/segmentation/batch_segment_cow_magnitude.py`

Command:

```bash
python code/segmentation/batch_segment_cow_magnitude.py \
  --input-root data/registered_7T_in_3T_cow_crop \
  --recursive \
  --mag-pattern "mag_7T_in_3T.nii.gz" \
  --mag-name "mag_7T_in_3T.nii.gz" \
  --vx-name "phaseX_7T_in_3T.nii.gz" \
  --vy-name "phaseY_7T_in_3T.nii.gz" \
  --vz-name "phaseZ_7T_in_3T.nii.gz" \
  --output-dir data/cow_segmentation_patient_batch \
  --ensemble-mode union
```

If you explicitly want AI-only and no postprocessing:

```bash
python code/segmentation/batch_segment_cow_magnitude.py \
  --input-root data/registered_7T_in_3T_cow_crop \
  --recursive \
  --mag-pattern "mag_7T_in_3T.nii.gz" \
  --mag-name "mag_7T_in_3T.nii.gz" \
  --vx-name "phaseX_7T_in_3T.nii.gz" \
  --vy-name "phaseY_7T_in_3T.nii.gz" \
  --vz-name "phaseZ_7T_in_3T.nii.gz" \
  --output-dir data/cow_segmentation_patient_batch \
  --ai-only \
  --no-postprocess
```

## 7) Final Pair Export Before Training

Script:

- `code/registration/export_paired_lr_hr_dataset.py`

This creates:

- `lr_3t/<case>/...`
- `hr_7t_in_3t/<case>/...`
- paired CSV for training (with `mask` empty by default)
- includes `hr_mag` column for optional 4-output training (`u,v,w,mag`)

Command:

```bash
python code/registration/export_paired_lr_hr_dataset.py \
  --temporal-dir data/temporal_registered \
  --registered-dir data/registered_7T_in_3T_cow_crop \
  --output-root data/paired_dataset \
  --csv-path data/paired_dataset/paired_nifti_cases.csv \
  --venc 0.9
```

## 8) Attach CoW Masks to Training CSV

Script:

- `code/registration/attach_cow_masks_to_csv.py`

Purpose:

- read paired CSV rows
- infer case path from `hr_u`
- locate `cow_seg_final.nii.gz` under segmentation output root
- write an updated CSV with `mask` paths filled

Command:

```bash
python code/registration/attach_cow_masks_to_csv.py \
  --csv-in data/paired_dataset/paired_nifti_cases.csv \
  --csv-out data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --masks-root data/cow_segmentation_patient_batch \
  --mask-name cow_seg_final.nii.gz \
  --hr-col hr_u \
  --hr-root-name hr_7t_in_3t \
  --path-mode relative-to-cwd \
  --strict
```

## Sign Policy Checklist

Use this to avoid double sign inversions:

1. DICOM->NIfTI applies LPS->RAS phase-component sign convention.
2. Training/inference defaults do not add extra U/V sign inversion.
3. Use legacy U/V inversion flags only for old datasets that did not bake signs upstream.
