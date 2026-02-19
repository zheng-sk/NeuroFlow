# Inference Workflow (Direct NIfTI)

Prediction script:

- `code/predict_nifti.py`

## 1) Inputs

Required:

- LR velocity: `u`, `v`, `w`
- LR magnitude: either
  - one file per component (`--mag-u`, `--mag-v`, `--mag-w`), or
  - one shared file (`--mag`)
- trained checkpoint: `--model-path`

## 2) Internal Processing

If `--raw-phase-input` (default), velocity is converted from RAW-like values using:

- unsigned RAW:

```text
v = (raw - raw_center) / raw_scale * venc
```

- signed RAW:

```text
v = raw / scale * venc, scale in {2048, 4096}
```

Then normalization:

```text
U_norm = U / venc
V_norm = V / venc
W_norm = W / venc
MAG_norm = ScaleIntensity(MAG, minv=0, maxv=1)   # default: --mag-norm-mode monai_minmax
```

Legacy optional mode:

```text
MAG_norm = MAG / mag_scale   # --mag-norm-mode divisor
```

Model predicts normalized HR outputs:

- velocity channels (`u,v,w`) are rescaled by `venc`
- magnitude channel (`mag`, when enabled):
  - stays in `[0,1]` for `--mag-norm-mode monai_minmax`
  - is rescaled by `mag_scale` only for `--mag-norm-mode divisor`

## 3) Sliding Window Modes

Default:

- MONAI `sliding_window_inference`

Legacy reconstruction option:

- `--legacy-overlap-inference`
- uses overlap/trim patch stitching equivalent to the old patch generator logic.

## 4) Command

```bash
python code/predict_nifti.py \
  --u /path/lr_u.nii.gz \
  --v /path/lr_v.nii.gz \
  --w /path/lr_w.nii.gz \
  --mag /path/lr_mag.nii.gz \
  --model-path /path/model-best.pt \
  --output-prefix /path/output/pred \
  --patch-size 16 \
  --res-increase 2
```

For checkpoints trained with 4 outputs (`u,v,w,mag`), set:

```bash
--predict-mag
```

## 5) Outputs

- `<output-prefix>_u.nii.gz`
- `<output-prefix>_v.nii.gz`
- `<output-prefix>_w.nii.gz`
- `<output-prefix>_uvw.nii.gz`
- `<output-prefix>_mag.nii.gz` (only when model outputs magnitude)

`_uvw` stacks the 3 components in one 4D NIfTI.

## 6) Sign-Consistency Note

For modern datasets processed with repository DICOM->NIfTI conversion:

- keep default (no extra U/V inversion)

Only for old datasets that need legacy correction:

```bash
--legacy-invert-uv-sign-on-raw
```

## 7) Prediction-Only Metrics (from predicted NIfTI)

If you only have predicted NIfTI outputs (without baseline/reference), use:

```bash
python code/inference/generate_pred_only_uq_report.py \
  --u-path /path/pred_u.nii.gz \
  --v-path /path/pred_v.nii.gz \
  --w-path /path/pred_w.nii.gz \
  --mag-path /path/pred_mag.nii.gz \
  --mask-path /path/mask.nii.gz \
  --out-dir /path/pred_only_metrics
```

This exports absolute metrics (flow, vorticity/velocity stats, WSS, voxel distributions, temporal geometry)
plus an HTML report at:

- `/path/pred_only_metrics/pred_only_report.html`

## 8) Comparative Metrics from Existing NIfTI (pred + LR + HR)

If you already have:
- prediction NIfTI (`pred_u/pred_v/pred_w`),
- baseline LR NIfTI (`lr_u/lr_v/lr_w`),
- reference HR NIfTI (`hr_u/hr_v/hr_w`),

you can build `analysis_payload.npz` and run the same comparative report as the default SR/UQ flow:

```bash
python code/inference/build_uq_payload_from_nifti.py \
  --pred-u /path/pred_u.nii.gz \
  --pred-v /path/pred_v.nii.gz \
  --pred-w /path/pred_w.nii.gz \
  --lr-u /path/lr_u.nii.gz \
  --lr-v /path/lr_v.nii.gz \
  --lr-w /path/lr_w.nii.gz \
  --hr-u /path/hr_u.nii.gz \
  --hr-v /path/hr_v.nii.gz \
  --hr-w /path/hr_w.nii.gz \
  --mask /path/mask.nii.gz \
  --out-dir /path/uq_from_nifti \
  --run-report
```

Notes:
- `--pred-mag`, `--lr-mag`, and `--hr-mag` are optional.
- If any magnitude is missing, it is derived from speed (`sqrt(u^2+v^2+w^2)`).
