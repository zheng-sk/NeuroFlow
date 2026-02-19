# SR UQ Metrics Guide

This document describes the metrics computed by:

- `code/inference/run_sr_inference_case.py`
- `code/inference/generate_sr_uq_report.py`
- `code/inference/run_sr_uq_pipeline.py`

It is intended as a practical reference for analysis and reporting in 4D flow super-resolution experiments.

## 1) Data conventions

- Input velocity channels: `u, v, w`
- Optional magnitude channel: `mag`
- Model output in this workflow: 4 channels (`u, v, w, mag`)
- Mask: binary vessel mask (`mask > 0.5`)
- Most metrics are aggregated over all processed frames.

## 2) Intraluminal distribution metrics (Table-2-like)

For each valid slice (along the selected `flow_axis`) and using only voxels inside the mask, the report computes:

- Mean velocity magnitude [m/s]
- SD velocity magnitude [m/s]
- Skewness velocity magnitude
- Kurtosis velocity magnitude
- Mean vorticity magnitude [1/s]
- SD vorticity magnitude [1/s]
- Skewness vorticity magnitude
- Kurtosis vorticity magnitude

The same set is computed for:

- Reference (`ref`)
- Baseline (`3T`)
- Super-resolved (`3T SR`)

Relative error (RE) is reported as:

- `RE(method) = |method - ref| / |ref|`

A Wilcoxon signed-rank p-value compares paired RE values between baseline and SR.

## 3) Flow-rate metrics

Flow is integrated per slice and frame:

- `Q(t, s) = sum(v_axis(t, s, in-mask voxels)) * voxel_area`

where `v_axis` is the velocity component aligned with the selected `flow_axis`.

From `Q`, the report provides:

- Mean flow profile across frames: `mean_Q(s)`
- Temporal SD profile: `SD_Q(s)`
- `MAD_Q` against a scalar reference (`q_ref`):
  - `MAD_Q = mean_s |mean_Q(s) - q_ref|`
- `mean_SD_Q = mean_s SD_Q(s)`
- Percent versions normalized by `|q_ref|`

A Wilcoxon p-value compares absolute flow errors of baseline vs SR.

## 4) WSS metrics (Table-3-like)

Wall shear stress (WSS) is estimated on boundary points using inward normal finite differences and dynamic viscosity `mu`.

Summary statistics are reported for each method:

- Maximum
- Mean
- SD
- Quantile 97.5%
- Median
- Quantile 2.5%
- IQR (75% - 25%)

RE is also computed against reference for each statistic.

A Wilcoxon p-value compares pointwise absolute WSS errors (baseline vs SR).

## 5) Geometry temporal uncertainty metrics

If multiple temporal masks are available, per-frame surface distances are computed against frame 0:

- Mean surface distance [mm]
- SD surface distance [mm]
- Symmetric mean surface distance [mm]
- Hausdorff distance [mm]

The report includes an aggregated geometry summary.

If all temporal masks are identical (for example, frame-0 mask propagated to all registered frames),
geometry temporal metrics are marked as **N/A** (NaN) and labeled accordingly in the summary/CSV.

## 6) Voxel-value distribution inside mask

The report also includes histogram-based comparisons (inside mask, all processed frames) for:

- `u`
- `v`
- `w`
- `mag`

and exports descriptive statistics:

- count
- mean
- std
- median
- p05
- p95
- min
- max

File: `metrics/voxel_distribution_stats.csv`

## 7) Flow axis (`flow_axis`) explanation

`flow_axis` is the anatomical axis used to:

- define orthogonal cross-sections
- integrate flow per slice

Allowed values: `0`, `1`, `2`, or `auto`.

When `auto`, the script scores each axis using reference-flow consistency:

- lower temporal relative SD is better
- smoother mean flow profile is better
- larger valid-flow slice coverage is better

Current score:

- `score = rel_sd + 0.20 * smoothness - 0.15 * coverage`
- lower score is selected

The chosen axis and candidate scores are saved in `summary_metrics.json`.

## 8) `q_ref` explanation

`q_ref` is the scalar reference flow used for MAD/percentage flow metrics.

- If `--q-ref` is provided, that value is used directly.
- If not provided, the script uses the median of the reference mean-flow profile.

Interpretation:

- `MAD_Q` and `mean_SD_Q` are easier to compare across runs when `q_ref` is fixed and externally defined.
- If `q_ref` is auto-derived, metrics are still useful for within-case comparisons, but less standardized across cohorts.

## 9) Practical interpretation notes

- Very small p-values in Wilcoxon tests indicate consistent paired improvement/degradation, not effect size magnitude.
- RE can become unstable when reference values are near zero; inspect raw values jointly.
- WSS estimates depend on mask quality, spacing, and boundary sampling density.
- For publication-level comparisons, keep `flow_axis`, `q_ref`, `mu`, and frame selection consistent.
- If LR baseline is downsampled (`LR shape != HR shape`), the report upsamples LR velocity to HR grid for metric computation.
- ROI analysis is supported via `--roi-bbox` or `--roi-json`; mask-based metrics are restricted to that bbox.

## 10) Commands (Inference + Metrics)

All examples use relative paths from repository root.

### 10.1 Run SR inference for a case (all frames)

```bash
python code/inference/run_sr_inference_case.py \
  --case-csv data/paired_dataset/train_random_1_src.csv \
  --case-index 0 \
  --model-path models/4DFlowNet_20260216-1714/4DFlowNet-best.pt \
  --out-dir output/uq_case0 \
  --predict-mag \
  --raw-phase-input \
  --patch-size 48 \
  --sw-batch-size 2 \
  --overlap 0.25 \
  --res-increase 1
```

Outputs:

- `output/uq_case0/analysis_payload.npz`
- `output/uq_case0/inference_metadata.json`
- predicted NIfTI files in `output/uq_case0/nifti/`

### 10.2 Run SR inference for specific frame(s)

Single frame:

```bash
python code/inference/run_sr_inference_case.py \
  --case-csv data/paired_dataset/train_random_1_src.csv \
  --case-index 0 \
  --model-path models/4DFlowNet_20260216-1714/4DFlowNet-best.pt \
  --out-dir output/uq_case0_frame3 \
  --predict-mag \
  --raw-phase-input \
  --frame-index 3 \
  --patch-size 48 \
  --sw-batch-size 2 \
  --overlap 0.25 \
  --res-increase 1
```

Multiple frames:

```bash
--frame-index 0 3 7
```

### 10.3 Generate UQ report + metrics from payload

```bash
python code/inference/generate_sr_uq_report.py \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --flow-axis auto \
  --selected-frame 0 \
  --max-display-slices 8 \
  --panel-cols 4 \
  --hist-bins 120 \
  --lr-mag-channel 0 \
  --mask-min-slice-voxels 25 \
  --mu-pa-s 0.0035 \
  --max-wall-points 30000 \
  --report-title "4D Flow SR Uncertainty Quantification Report"
```

ROI-restricted report (using bbox in HR voxel coordinates):

```bash
python code/inference/generate_sr_uq_report.py \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --roi-bbox 30 90 40 120 12 44
```

ROI from JSON (recommended with interactive selector):

```bash
python code/inference/generate_sr_uq_report.py \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --roi-json output/uq_case0/roi_bbox.json
```

Interactive selector (exports `bbox_hr_xyz` + mapped `bbox_lr_xyz`):

```bash
python code/inference/select_metric_roi_bbox.py \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union
```

Selector + export directo del reporte ROI:

```bash
python code/inference/select_metric_roi_bbox.py \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union \
  --run-report
```

Main artifacts:

- `output/uq_case0/report.html`
- `output/uq_case0/metrics/summary_metrics.json`
- `output/uq_case0/metrics/table2_like_all_slices.csv`
- `output/uq_case0/metrics/table2_like_per_frame_all_slices.csv`
- `output/uq_case0/metrics/flow_metrics.csv`
- `output/uq_case0/metrics/flow_metrics_per_frame.csv`
- `output/uq_case0/metrics/flow_rate_curves_per_frame.csv`
- `output/uq_case0/metrics/table3_like_wss.csv`
- `output/uq_case0/metrics/table3_like_wss_per_frame.csv`
- `output/uq_case0/metrics/geometry_temporal_surface_metrics.csv`
- `output/uq_case0/metrics/voxel_distribution_stats.csv`

### 10.4 One-command pipeline (inference + report)

```bash
python code/inference/run_sr_uq_pipeline.py \
  --case-csv data/paired_dataset/train_random_1_src.csv \
  --case-index 0 \
  --model-path models/4DFlowNet_20260216-1714/4DFlowNet-best.pt \
  --out-dir output/uq_case0 \
  --predict-mag \
  --raw-phase-input \
  --flow-axis auto \
  --res-increase 1 \
  --patch-size 48 \
  --sw-batch-size 2 \
  --overlap 0.25 \
  --max-display-slices 8 \
  --panel-cols 4 \
  --hist-bins 120
```

Optional:

- set scalar reference flow: `--q-ref 11.72`
- set custom CCA range: `--cca-range 8:20`

### 10.5 Prediction-only metrics from NIfTI (no baseline/reference)

Use this mode when you only have predicted NIfTI files (`u,v,w`, optional `mag`, optional `mask`).
It computes absolute metrics (no relative error vs reference).

```bash
python code/inference/generate_pred_only_uq_report.py \
  --u-path output/uq_case0/nifti/pred_u.nii.gz \
  --v-path output/uq_case0/nifti/pred_v.nii.gz \
  --w-path output/uq_case0/nifti/pred_w.nii.gz \
  --mag-path output/uq_case0/nifti/pred_mag.nii.gz \
  --mask-path data/masks/case0_mask.nii.gz \
  --out-dir output/uq_case0_pred_only \
  --flow-axis auto \
  --hist-bins 120
```

ROI-restricted prediction-only metrics:

```bash
python code/inference/generate_pred_only_uq_report.py \
  --u-path output/uq_case0/nifti/pred_u.nii.gz \
  --v-path output/uq_case0/nifti/pred_v.nii.gz \
  --w-path output/uq_case0/nifti/pred_w.nii.gz \
  --mask-path data/masks/case0_mask.nii.gz \
  --out-dir output/uq_case0_pred_only \
  --roi-json output/uq_case0/roi_bbox.json
```

Prediction-only artifacts:

- `output/uq_case0_pred_only/pred_only_report.html`
- `output/uq_case0_pred_only/metrics/pred_only_summary_metrics.json`
- `output/uq_case0_pred_only/metrics/pred_only_table2_all_slices.csv`
- `output/uq_case0_pred_only/metrics/pred_only_table2_per_frame_all_slices.csv`
- `output/uq_case0_pred_only/metrics/pred_only_table2_compact.csv`
- `output/uq_case0_pred_only/metrics/pred_only_flow_metrics.csv`
- `output/uq_case0_pred_only/metrics/pred_only_flow_metrics_per_frame.csv`
- `output/uq_case0_pred_only/metrics/pred_only_flow_rate_curves_per_frame.csv`
- `output/uq_case0_pred_only/metrics/pred_only_table3_wss.csv`
- `output/uq_case0_pred_only/metrics/pred_only_table3_wss_per_frame.csv`
- `output/uq_case0_pred_only/metrics/pred_only_geometry_temporal_surface_metrics.csv`
- `output/uq_case0_pred_only/metrics/pred_only_voxel_distribution_stats.csv`

### 10.6 Build comparable payload from NIfTI (pred + 3T LR + 7T HR)

Use this when you already have NIfTI predictions and also want the **same comparative metrics**
as the standard SR/UQ report (`baseline=3T`, `reference=7T`, `sr=prediction`).

One command (build payload + run full report):

```bash
python code/inference/build_uq_payload_from_nifti.py \
  --pred-u output/uq_case0/nifti/pred_u.nii.gz \
  --pred-v output/uq_case0/nifti/pred_v.nii.gz \
  --pred-w output/uq_case0/nifti/pred_w.nii.gz \
  --pred-mag output/uq_case0/nifti/pred_mag.nii.gz \
  --lr-u data/case0/lr_u.nii.gz \
  --lr-v data/case0/lr_v.nii.gz \
  --lr-w data/case0/lr_w.nii.gz \
  --lr-mag data/case0/lr_mag.nii.gz \
  --hr-u data/case0/hr_u_7t.nii.gz \
  --hr-v data/case0/hr_v_7t.nii.gz \
  --hr-w data/case0/hr_w_7t.nii.gz \
  --hr-mag data/case0/hr_mag_7t.nii.gz \
  --mask data/case0/mask.nii.gz \
  --out-dir output/uq_case0_from_nifti \
  --baseline-label \"3T\" \
  --ref-label \"7T\" \
  --sr-label \"3T SR\" \
  --run-report
```

If prediction does not include magnitude, just omit `--pred-mag` (script derives it from speed).
Likewise, if `--hr-mag` or `--lr-mag` are missing, they are derived automatically from velocity magnitude.
