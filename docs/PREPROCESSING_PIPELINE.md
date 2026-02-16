# Preprocessing Pipeline (DICOM to Training-Ready Data)

This document describes the recommended end-to-end preprocessing and training preparation flow in this repository, with emphasis on:

- sign conventions
- where sign inversions happen
- normalization equations
- which stage changes geometry vs intensity

The sequence below follows this order:

1. DICOM -> NIfTI
2. Temporal registration
3. CoW crop
4. Inter-scan registration (7T -> 3T)
5. Training data assembly and training normalization


## 1) DICOM -> NIfTI

Primary script:

- `code/conversion/dicom_to_nifti.py`
- wrapper: `code/convert2nifti2.py`

Purpose:

- read 4D-flow DICOM series
- write NIfTI
- standardize orientation to RAS+
- apply LPS->RAS vector-component sign convention for phase components

Key sign rule (component domain):

- `Vx` flips sign
- `Vy` flips sign
- `Vz` keeps sign

Implementation detail:

- Function: `apply_lps2ras_component_sign(...)` in `code/conversion/dicom_to_nifti.py`
- For unsigned raw phase (`~0..4096`), sign flip is performed around the raw center to preserve encoding range.

Formally:

```text
If component in {Vx, Vy}:
  if unsigned raw:
      out = 2*c - in      (c = 2048, or 4096 guard for larger range)
  else:
      out = -in
Else (Vz):
  out = in
```

Important:

- This stage usually performs the required component sign convention already.
- Downstream stages should not invert `U/V` again unless using a legacy dataset that did not bake this convention.


## 2) Temporal Registration (within each scan)

Primary scripts:

- `code/registration/batch_temporal_register.py`
- `code/registration/temporal_register_to_t0.py`
- full temporal propagation workflow: `code/registration/batch_register_magnitude.py`

Purpose:

- reduce intra-scan motion across time frames
- estimate transforms from magnitude
- apply the same transforms to velocity/phase frames

Reference choice:

```text
fixed frame = frame t0 (default t=0), or equivalent configured reference
```

Metrics used for QC:

```text
MAD = mean(|I_ref - I_mov|)
NCC = <(a-a_mean), (b-b_mean)> / (||a-a_mean|| * ||b-b_mean||)
```

Vector handling:

- For velocity triplets, transforms are applied in vector mode (`imagetype=1`) in ANTs when using `apply_transforms_to_velocity_triplet(...)`.
- This is not a manual global `*-1`; vector orientation is handled by the transform machinery.

Sign behavior at this stage:

- no explicit global sign inversion policy is applied
- sign may change locally only as part of geometric vector reorientation from the transform


## 3) CoW Crop (paired 3T/7T)

Primary script:

- `code/preprocessing/yolo_crop_patient_pairs.py`
- wrapper: `code/yolo_crop_patient_pairs.py`

Purpose:

- detect CoW ROI from magnitude
- compute one common crop box for the 3T/7T pair
- crop all NIfTI files in both folders with the same box

Detection rule:

- If magnitude is 4D, detection runs on temporal MIP:

```text
MIP(x,y,z) = max_t M(x,y,z,t)
```

Crop merge:

- build one box that encloses both detections (3T and 7T)
- keeps full Z by default (X/Y constrained ROI)

Geometry update:

- cropped affine translation is shifted by crop offset:

```text
t_new = t_old + A_old(0:3,0:3) * offset_xyz
```

Sign behavior at this stage:

- none (spatial crop only)


## 4) Inter-Scan Registration (7T -> 3T)

Primary scripts:

- `code/registration/batch_register_7T_to_3T.py`
- `code/registration/register_7T_to_3T_with_qc.py`
- combined temporal + inter-scan pipeline: `code/registration/batch_full_register_7T_to_3T.py`

Purpose:

- estimate one transform from 7T magnitude to 3T magnitude
- apply it to 7T magnitude and phase/velocity so outputs live in 3T space

Robust references:

```text
fixed_ref  = median_t(3T_mag_4D)
moving_ref = median_t(7T_mag_4D)
```

Estimation preprocessing (magnitude only, for registration robustness):

- optional mask (`ants` or `hdbet`)
- N4 bias correction
- denoising
- winsorization
- optional histogram match

Winsorization:

```text
I_w = clip(I, q01, q99)
```

Phase warp modes:

1. `direct` (default):

```text
P'(x) = P(T^{-1}(x))
```

2. `complex`:

```text
R = M * cos(P)
I = M * sin(P)
R' = warp(R), I' = warp(I)
P' = atan2(I', R')
```

Sign behavior at this stage:

- no additional repository-level `U/V` sign policy is applied here
- this stage is geometric alignment; component sign convention should already come from DICOM->NIfTI


## 5) Training Preparation and Normalization

Typical scripts:

- pair export CSV/data: `code/registration/export_paired_lr_hr_dataset.py`
- training entrypoint: `src/trainer_nifti.py`
- dataset transforms: `src/Network/NiftiPatchDataset.py`

Expected training CSV fields include:

```text
lr_u, lr_v, lr_w, lr_mag_u, lr_mag_v, lr_mag_w, hr_u, hr_v, hr_w, mask, venc
```

### 5.1 Raw-to-velocity conversion (if enabled)

`trainer_nifti.py` defaults to `--raw-phase-input`.

In `NiftiPatchDataset` and `code/predict_nifti.py`, conversion is auto-detected:

```text
if unsigned RAW:
    v = (raw - raw_center) / raw_scale * venc_component
elif signed RAW:
    v = raw / scale * venc_component
    (scale = 2048 or 4096 based on dynamic range)
else:
    fallback to unsigned-style formula (legacy-compatible behavior)
```

Default:

- `raw_center = 2048`
- `raw_scale = 2048`

RAW detection heuristic:

```text
unsigned RAW: min >= 0 and max in (1000, 8192]
signed RAW:   min < -500 and max > 500 and centered near 0
```

Sign policy for training conversion:

- default: no extra `U/V` inversion (to avoid double inversion if DICOM->NIfTI already corrected signs)
- optional legacy behavior: `--legacy-invert-uv-sign-on-raw`

### 5.2 VENC normalization

After conversion:

```text
venc_global = max(venc_u, venc_v, venc_w)
u_norm = u / venc_global
v_norm = v / venc_global
w_norm = w / venc_global
```

If VENC is missing/non-positive, fallback is estimated from max absolute LR velocity.

### 5.3 Magnitude normalization

```text
mag_norm = mag / mag_scale
```

Default:

- `mag_scale = 4095`

### 5.4 Mask binarization

```text
mask_bin = 1(mask >= mask_threshold)
```

Default threshold:

- `mask_threshold = 0.5`

### 5.5 Patch-space augmentation (vector aware)

Rotation is not a simple global sign flip.

- For vector fields, a rotation requires:
  - axis permutation (component swaps)
  - component sign changes implied by rotation matrix

Implemented in:

- `src/Network/PatchHandler3D.py` (`rotate90`, `rotate180_3d`)
- called by `src/Network/NiftiPatchDataset.py` during training augmentation

Example (180 deg in XY plane):

```text
u stays u, v -> -v, w -> -w
```


## Sign-Inversion Checklist (to avoid double inversion)

Recommended modern path:

1. Convert DICOM->NIfTI with `code/conversion/dicom_to_nifti.py` (sign convention baked for `Vx/Vy`).
2. Keep training/prediction with default no extra `U/V` inversion.
3. Use `--legacy-invert-uv-sign-on-raw` only if data provenance confirms missing sign correction upstream.

Training flag:

- `src/trainer_nifti.py`: `--legacy-invert-uv-sign-on-raw`

Prediction flag:

- `code/predict_nifti.py`: `--legacy-invert-uv-sign-on-raw`


## Practical Command Skeleton

Adjust paths and options to your dataset.

1. DICOM -> NIfTI

```bash
python code/convert2nifti2.py --help
```

2. Temporal registration

```bash
python code/batch_temporal_register.py --help
```

3. CoW crop (paired)

```bash
python code/yolo_crop_patient_pairs.py --help
```

4. Inter-scan registration (7T -> 3T)

```bash
python code/batch_register_7T_to_3T.py --help
```

5. Export paired LR/HR and train

```bash
python code/registration/export_paired_lr_hr_dataset.py --help
python src/trainer_nifti.py --help
```
