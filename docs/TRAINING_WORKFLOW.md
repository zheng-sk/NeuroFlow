# Training Workflow (NIfTI, PyTorch + MONAI)

## 1) Training Inputs

Training script:

- `src/trainer_nifti.py`

Dataset loader and transforms:

- `src/Network/NiftiPatchDataset.py`

Required CSV columns:

```text
lr_u,lr_v,lr_w,lr_mag_u,lr_mag_v,lr_mag_w,hr_u,hr_v,hr_w,mask,venc
```

If you train with magnitude output (`--predict-mag`), add:

```text
hr_mag
```

Notes:

- `mask` is optional (empty allowed).
- if `mask` is empty, full-volume mask is used.
- CSV supports absolute or relative paths.

## 2) Mask Format (Segmentation Mask for CoW)

Expected training mask format:

1. NIfTI (`.nii` or `.nii.gz`)
2. Same spatial grid as HR targets (`hr_u/hr_v/hr_w`)
3. Values binarized in loader as:

```text
mask_bin = 1(mask >= mask_threshold)
```

Default:

- `mask_threshold = 0.5`

Practical recommendation:

- store masks as `uint8` with values `{0,1}`.

Populate `mask` column from CoW segmentations:

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

## 3) Normalization and Conversions

### 3.1 RAW phase -> velocity

Default behavior assumes RAW-like phase and converts:

- unsigned RAW:

```text
v = (raw - 2048) / 2048 * venc
```

- signed RAW:

```text
v = raw / scale * venc
scale in {2048, 4096}
```

### 3.2 VENC normalization

```text
venc_global = max(venc_u, venc_v, venc_w)
U_norm = U / venc_global
V_norm = V / venc_global
W_norm = W / venc_global
```

### 3.3 Magnitude normalization

```text
MAG_norm = MAG / mag_scale
```

Default:

- `mag_scale = 4095`

## 4) 4D Handling

When volumes are 4D:

- loader picks one time frame per sample
- train mode can sample random time frame
- val mode uses deterministic index/range logic

## 5) Patch Sampling

- LR patch size: `patch_size`
- HR patch size: `patch_size * res_increase`

Optional minimum mask coverage:

- `--legacy-minimum-coverage`
- `--legacy-max-sampling-attempts`
- optional strict failure: `--legacy-disallow-empty-fallback`

## 6) Data Augmentation

Vector-aware random 90 deg rotations are applied in training mode.

Important: velocity rotation is not scalar rotation; it includes:

- axis permutation
- sign changes induced by rotation

Mask and magnitude are rotated as scalar fields.

## 7) Loss and Metric

Trainer:

- `src/Network/TrainerController.py`

Per-voxel speed-MSE term:

```text
mse = (u_pred-u)^2 + (v_pred-v)^2 + (w_pred-w)^2
```

Batch loss combines fluid and non-fluid regions:

```text
fluid_mse     = sum(mse * mask) / (sum(mask) + eps)
non_fluid_mse = sum(mse * (1-mask)) / (sum(1-mask) + eps)
loss = fluid_mse + non_fluid_mse
```

When `--predict-mag` is enabled:

```text
loss = vel_mse + mag_loss_weight * mag_mse
```

Relative error metric is computed over masked regions.

## 8) Training Command

```bash
cd src
python trainer_nifti.py \
  --train-csv ../data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --val-csv ../data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --patch-size 16 \
  --res-increase 2 \
  --batch-size 4 \
  --epochs 60 \
  --mask-threshold 0.5
```

If your velocity is already in m/s:

```bash
--already-velocity-input
```

Use legacy U/V inversion only for old datasets:

```bash
--legacy-invert-uv-sign-on-raw
```

Train with 4 outputs (`u,v,w,mag`):

```bash
cd src
python trainer_nifti.py \
  --train-csv ../data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --val-csv ../data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --patch-size 16 \
  --res-increase 2 \
  --batch-size 4 \
  --epochs 60 \
  --mask-threshold 0.5 \
  --predict-mag \
  --mag-loss-weight 1.0
```

Performance-oriented options:

- cache loaded NIfTI cases in memory:

```bash
--cache-dataset --cache-eager
```

- disable rotation/time-frame augmentation in training:

```bash
--no-augmentation
```

- keep random patches but without rotation augmentation:

```bash
--rotation-prob 0.0
```

- force deterministic center patch during training:

```bash
--deterministic-train-patches
```
