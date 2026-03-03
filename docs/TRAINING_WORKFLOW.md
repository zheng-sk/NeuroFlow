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

Default (recommended):

```text
MAG_norm = ScaleIntensity(MAG, minv=0, maxv=1)   # MONAI min-max per frame
```

Legacy optional mode:

```text
MAG_norm = MAG / mag_scale
```

Legacy default value:

- `mag_scale = 4095`

CLI:

- `--mag-norm-mode monai_minmax` (default)
- `--mag-norm-mode divisor --mag-scale 4095`

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

## 8) Build x2 Experiments via Downsampling (0.5)

Goal:

- create synthetic LR inputs at half spatial resolution (`scale=0.5`)
- then train SR with `--res-increase 2`

Tools:

- `code/preprocessing/downsample_nifti_tree.py`
- `code/preprocessing/remap_case_csv_for_x2.py`

### 8.1 Option A (recommended): downsample 7T to create LR

Use this when you want a clean synthetic-LR setup where target HR stays in 7T space.

1) Downsample 7T tree:

```bash
python code/preprocessing/downsample_nifti_tree.py \
  --input-root data/paired_dataset/hr_7t_in_3t \
  --output-root data/paired_dataset/lr_from_7t_x05 \
  --scale 0.5 \
  --time-axis -1
```

2) Build a new paired CSV where `lr_*` points to the downsampled 7T tree:

```bash
python code/preprocessing/remap_case_csv_for_x2.py \
  --in-csv data/paired_dataset/train_random_1_src.csv \
  --out-csv data/paired_dataset/train_random_1_x2_from7t.csv \
  --mode lr_from_hr \
  --source-root data/paired_dataset/hr_7t_in_3t \
  --new-lr-root data/paired_dataset/lr_from_7t_x05
```

### 8.2 Option B: downsample 3T to create even lower LR

Use this when you want to test robustness from a more degraded clinical-like input.

1) Downsample 3T tree:

```bash
python code/preprocessing/downsample_nifti_tree.py \
  --input-root data/paired_dataset/lr_3t \
  --output-root data/paired_dataset/lr_3t_x05 \
  --scale 0.5 \
  --time-axis -1
```

2) Build a new paired CSV where `lr_*` points to the downsampled 3T tree:

```bash
python code/preprocessing/remap_case_csv_for_x2.py \
  --in-csv data/paired_dataset/train_random_1_src.csv \
  --out-csv data/paired_dataset/train_random_1_x2_from3t.csv \
  --mode lr_from_lr \
  --source-root data/paired_dataset/lr_3t \
  --new-lr-root data/paired_dataset/lr_3t_x05
```

Important:

- Keep `hr_*` and `mask` in HR grid (do not downsample them for this x2 SR setup).
- Use `--res-increase 2` in training.
- Mask/seg files are automatically resampled with nearest-neighbor when filename contains:
  - `mask`, `seg`, or `label`

## 9) Training Command

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

Train with fitted + exaggerated noise augmentation (plateau-like profile):

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
  --noise-aug-prob 1.0 \
  --noise-aug-fit-summary-csv ../output/noise_pdf/nonvascular_noise_fit_summary.csv \
  --noise-aug-exaggerated-side-expand 1.8 \
  --noise-aug-exaggerated-edge-boost 0.30 \
  --noise-aug-exaggerated-edge-power 1.5 \
  --noise-aug-phase-scale 0.06 \
  --noise-aug-mag-scale 0.04 \
  --noise-aug-range-mult 1.5 \
  --noise-aug-level-min 1.0 \
  --noise-aug-level-max 2.2 \
  --noise-aug-apply-mag \
  --no-noise-aug-clip-mag
```

Notes:
- This command uses best-fit families from the fit-summary CSV (for example, Magnitude `SkewNormal`, Phase `GeneralizedNormal`) and then exaggerates them.
- Update `--noise-aug-fit-summary-csv` to the exact CSV you want to use for the current experiment.

Performance-oriented options:

- cache loaded NIfTI cases in memory (MONAI `CacheDataset`):

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

## 10) Training Stability and Reproducibility

Recommended options for stable runs:

```bash
--seed 42 \
--deterministic \
--accuracy-include-mag \
--accuracy-mag-weight 1.0 \
--lr-scheduler reduce_on_plateau \
--lr-reduce-factor 0.5 \
--lr-reduce-patience 8 \
--lr-min 1e-6 \
--early-stopping-patience 20 \
--early-stopping-min-delta 0.0 \
--overfit-patience 8 \
--overfit-min-delta 0.0
```

Notes:

- `val_accuracy` includes magnitude only when `--predict-mag` and `--accuracy-include-mag` are enabled.
- Early stop can trigger by:
  - no validation improvement (`--early-stopping-*`)
  - sustained overfitting pattern (train improves while val worsens, `--overfit-*`).

Complete example command (x2 SR + 4-channel output + reproducibility + scheduler + early stop):

```bash
python src/trainer_nifti.py \
  --train-csv data/paired_dataset/train_random_1_x2_from7t.csv \
  --val-csv data/paired_dataset/val_random_1_x2_from7t.csv \
  --network-name 4DFlowNet_x2_uq \
  --patch-size 16 \
  --res-increase 2 \
  --batch-size 4 \
  --epochs 120 \
  --predict-mag \
  --mag-loss-weight 1.0 \
  --raw-phase-input \
  --mask-threshold 0.5 \
  --seed 42 \
  --deterministic \
  --accuracy-include-mag \
  --accuracy-mag-weight 1.0 \
  --lr-scheduler reduce_on_plateau \
  --lr-reduce-factor 0.5 \
  --lr-reduce-patience 8 \
  --lr-min 1e-6 \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 0.0 \
  --overfit-patience 8 \
  --overfit-min-delta 0.0 \
  --tb-image-every-epochs 10
```
