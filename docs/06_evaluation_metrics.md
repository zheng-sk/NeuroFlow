# SR UQ Metrics Guide

This document describes the metrics computed by:

- `neuroflow/inference/run_sr_inference_case.py`
- `neuroflow/evaluation/generate_sr_uq_report.py`
- `neuroflow/inference/run_sr_uq_pipeline.py`

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

Flow can be integrated in two modes:

- `axis` (legacy): per slice and frame
- `centerline`: per orthogonal plane sampled along a centerline

### Axis mode

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

### Centerline mode

Centerline mode builds a 3D vessel centerline from the mask (cleanup + skeleton + graph path),
then samples perpendicular planes and computes:

- `Q(t, p) = sum(v(t) · n_p in slab_p) * voxel_volume / slab_thickness`

where `n_p` is the local centerline tangent (plane normal), and `slab_p` is a thin slab around plane `p`.

Temporal flow per frame is then aggregated across valid planes (median or mean).

Centerline QC artifacts exported by the report:

- `figures/centerline_overlay.png` (mask + centerline + valid/invalid planes)
- `figures/centerline_3d.png` (visualización 3D de máscara + centerline + planos)
- `figures/centerline_plane_sections.png` (proximal/mid/distal section shape QC)
- `figures/centerline_flow_along_vessel_peak.png` (`Q(s)` sanity check at peak frame)
- `metrics/centerline_section_qc.csv` (offset, compactness, elongation, wall-distance ratios per plane)
- `metrics/centerline_sign_qc.csv` (temporal sign agreement, Pearson r y p-value vs reference)

Additional centerline p-values (saved in `summary_metrics.json` under `statistics.centerline`):

- `qc_peak_plane_abs_err_wilcoxon_p_baseline_vs_sr`
- `qc_all_planes_abs_err_wilcoxon_p_baseline_vs_sr`

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
python -m neuroflow.inference.run_sr_inference_case \
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
python -m neuroflow.inference.run_sr_inference_case \
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
python -m neuroflow.evaluation.generate_sr_uq_report \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --flow-axis auto \
  --flow-method axis \
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

Centerline-based flow report:

```bash
python -m neuroflow.evaluation.generate_sr_uq_report \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --flow-method centerline \
  --centerline-mask-mode union \
  --centerline-closing-iters 1 \
  --centerline-n-planes 7 \
  --centerline-slab-thickness-mm 1.5 \
  --centerline-min-plane-voxels 10 \
  --centerline-min-valid-support 10 \
  --centerline-aggregate median
```

ROI-restricted report (using bbox in HR voxel coordinates):

```bash
python -m neuroflow.evaluation.generate_sr_uq_report \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --roi-bbox 30 90 40 120 12 44
```

ROI from JSON (recommended with interactive selector):

```bash
python -m neuroflow.evaluation.generate_sr_uq_report \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --metadata-json output/uq_case0/inference_metadata.json \
  --out-dir output/uq_case0 \
  --roi-json output/uq_case0/roi_bbox.json
```

Interactive 3D selector (PyVista box widget; exports `bbox_hr_xyz` + mapped `bbox_lr_xyz`):

```bash
python -m neuroflow.evaluation.select_metric_roi_bbox \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union
```

Controls:

- drag box handles to adjust ROI
- live bbox info is shown on-screen (voxel bounds, voxel size, physical size in mm)
- `Enter` or `Space` to accept
- `Esc` to cancel

Interactive selector + automatic ROI report:

```bash
python -m neuroflow.evaluation.select_metric_roi_bbox \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union \
  --run-report
```

When `--run-report` is used and `--report-out-dir` is not provided, the report is saved to a new folder
whose name includes bbox limits, e.g.:

- `output/uq_case0_bbox_x30-90_y40-120_z12-44`

Requirement: `pyvista` must be installed in your environment.
If you get `render_window is None` / `IsCurrent` errors, run from a GUI desktop session and ensure off-screen mode is disabled:

```bash
unset PYVISTA_OFF_SCREEN
python -m neuroflow.evaluation.select_metric_roi_bbox \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union
```

Selector backend options:

- `--selector-mode auto` (default): PyVista -> Napari -> manual fallback
- `--selector-mode 3d`: force PyVista 3D box widget
- `--selector-mode napari`: use Napari 3D viewer with 2 corner points
- `--selector-mode manual`: terminal prompt only

If your environment still cannot open VTK windows, install Napari and use:

```bash
pip install "napari[all]"
python -m neuroflow.evaluation.select_metric_roi_bbox \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union \
  --selector-mode napari
```

If no GUI backend is available, use terminal fallback:

```bash
python -m neuroflow.evaluation.select_metric_roi_bbox \
  --payload-npz output/uq_case0/analysis_payload.npz \
  --out-json output/uq_case0/roi_bbox.json \
  --temporal-mode union \
  --selector-mode manual \
  --run-report
```

Note: in manual mode, pressing Enter does not accept the default bbox immediately; it asks for explicit confirmation.

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
python -m neuroflow.inference.run_sr_uq_pipeline \
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
python -m neuroflow.evaluation.generate_pred_only_uq_report \
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
python -m neuroflow.evaluation.generate_pred_only_uq_report \
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
python -m neuroflow.evaluation.build_uq_payload_from_nifti \
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


---

# Appendix: Aneurysm geometry metrics


This document describes how to select an aneurysm region, define the neck, create VTK meshes, calculate morphology metrics, and generate quality-control figures.

The recommended workflow is:

1. Select a rough aneurysm ROI from the CoW segmentation.
2. Define the aneurysm neck manually or accept the automatic candidate.
3. Save an aneurysm-only mask after removing the connected artery portion.
4. Calculate geometry metrics and export VTK meshes.
5. Generate visual overlays for review.

## Inputs

Example case used below:

```text
subject_001
```

Main inputs:

```text
3D/4D magnitude image:
data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz

CoW segmentation:
data/paired_dataset/cow_segmentation_ens301_current/subject_001/cow_seg_final.nii.gz
```

The segmentation should be binary or thresholdable:

```text
0 = background
1 = vessel / aneurysm region
```

## Professional Precrops

Professional aneurysm-focused precrops are available under:

```text
output/nii-cropped/
```

Folder structure:

```text
output/nii-cropped/001/
  3T/
    input_mag_raw.nii.gz
    mask.nii.gz
    Vx.nii.gz, Vy.nii.gz, Vz.nii.gz
  3T_downsampled/
    input_mag_raw.nii.gz
    mask.nii.gz
    Vx.nii.gz, Vy.nii.gz, Vz.nii.gz
  7T/
    input_mag_raw.nii.gz
    Vx.nii.gz, Vy.nii.gz, Vz.nii.gz
  7T_masked/
    input_mag_raw.nii.gz
    Vx.nii.gz, Vy.nii.gz, Vz.nii.gz
  results_denoised/
    cow_seg_final.nii.gz
    pred_mag.nii.gz
    pred_u.nii.gz, pred_v.nii.gz, pred_w.nii.gz
  results_super-resolved/
    cow_seg_final.nii.gz
    pred_mag.nii.gz
    pred_u.nii.gz, pred_v.nii.gz, pred_w.nii.gz
```

The same structure exists for patients:

```text
001, 002, 003, 004, 005, 006, 007
```

For geometry measurements on professional crops, use:

```text
ROI/mask input:
output/nii-cropped/<ID>/3T/mask.nii.gz

background image:
output/nii-cropped/<ID>/7T/input_mag_raw.nii.gz
```

For predicted-mask comparisons, use:

```text
output/nii-cropped/<ID>/results_denoised/cow_seg_final.nii.gz
output/nii-cropped/<ID>/results_super-resolved/cow_seg_final.nii.gz
```

Because the professional crops already focus on the aneurysm region, the first rough ROI box selection step can be skipped.

## Precrop Workflow: 3D Plane Selection

For the professional crop of patient 001:

```bash
python -m neuroflow.visualization.select_aneurysm_neck_plane \
  --roi output/nii-cropped/001/3T/mask.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --out output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz
```

Then calculate metrics:

```bash
python -m neuroflow.evaluation.calculate_aneurysm_shape_metrics \
  --seg output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz \
  --neck-json output/aneurysm_plane_precrops/001_aneurysm_plane_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_precrop_plane
```

Generate QC overlay:

```bash
python -m neuroflow.visualization.visualize_aneurysm_measurements \
  --roi output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_precrop_plane/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_precrop_plane/measurement_overlay.png
```

## Precrop Workflow: Automatic 3D Test

The automatic mode can be used as a screening test:

```bash
python -m neuroflow.visualization.select_aneurysm_neck \
  --mode auto \
  --roi output/nii-cropped/001/3T/mask.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --out output/aneurysm_precrop_auto/001_aneurysm_auto.nii.gz
```

Then calculate metrics:

```bash
python -m neuroflow.evaluation.calculate_aneurysm_shape_metrics \
  --seg output/aneurysm_precrop_auto/001_aneurysm_auto.nii.gz \
  --neck-json output/aneurysm_precrop_auto/001_aneurysm_auto_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_precrop_auto
```

Important: automatic detection must be visually checked. If it keeps almost the full crop or cuts the wrong side, use the 3D plane selector instead.

## Step 1: Select Rough Aneurysm ROI

Use the 3D ROI selector to crop the aneurysm region from the full CoW segmentation.

For automatic or manual neck detection, include:

```text
aneurysm bulge + small part of the connecting artery
```

Do not include large surrounding arteries, because they can bias the final measurements if they are not removed later.

Command:

```bash
python -m neuroflow.visualization.select_aneurysm_roi \
  --seg data/paired_dataset/cow_segmentation_ens301_current/subject_001/cow_seg_final.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --out output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --box-size 10
```

Output:

```text
output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz
```

## Step 2A: Manual Neck Selection

Use this if you want to define the aneurysm neck yourself.

Command:

```bash
python -m neuroflow.visualization.select_aneurysm_neck \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --out output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz
```

In the window:

```text
1. Choose the 2D view where the neck is clearest.
2. Click neck endpoint 1.
3. Click neck endpoint 2 in the same panel.
4. Click one point inside the aneurysm bulge side.
5. Press Save or s.
```

Outputs:

```text
output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz
output/aneurysm_rois/001_aneurysm_sac_manual_neck.json
```

The JSON file stores:

```text
manual neck endpoints
neck width
neck plane
aneurysm-side point
height from the neck plane
```

## Step 2B: Semi-Automatic Neck Selection

Use this if you want the code to propose a neck candidate.

The automatic candidate is shown in cyan. It is estimated as the narrowest cross-section perpendicular to the longest 3D axis of the selected region.

Command:

```bash
python -m neuroflow.visualization.select_aneurysm_neck \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --out output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz
```

In the window:

```text
cyan points = automatic neck candidate
press a or Auto Save = accept automatic neck
click 3 manual points = override automatic neck
```

Outputs:

```text
output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz
output/aneurysm_rois/001_aneurysm_sac_auto_neck.json
```

## Step 2C: 3D Plane-Based Neck Selection

Use this if you want to define the neck as a 3D cutting plane instead of clicking points in 2D.

The idea is:

```text
move/rotate a plane to the aneurysm opening
keep the aneurysm side
remove the artery side
calculate neck area from the plane intersection
```

Command:

```bash
python -m neuroflow.visualization.select_aneurysm_neck_plane \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --out output/aneurysm_rois/001_aneurysm_sac_plane.nii.gz
```

Controls:

```text
drag/rotate cyan plane = position the neck cut
f = flip the side that will be kept
s = save aneurysm-only mask and neck JSON
r = reset camera
h = toggle help text
q or Esc = quit
```

Outputs:

```text
output/aneurysm_rois/001_aneurysm_sac_plane.nii.gz
output/aneurysm_rois/001_aneurysm_sac_plane_neck.json
```

The neck width is calculated from the plane intersection area:

```text
neck equivalent diameter = 2 * sqrt(neck area / pi)
```

## Step 3: Calculate Metrics and Export VTK Meshes

Use the aneurysm-only mask and the neck JSON.

Manual-neck example:

```bash
python -m neuroflow.evaluation.calculate_aneurysm_shape_metrics \
  --seg output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --neck-json output/aneurysm_rois/001_aneurysm_sac_manual_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_sac_manual
```

Automatic-neck example:

```bash
python -m neuroflow.evaluation.calculate_aneurysm_shape_metrics \
  --seg output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --neck-json output/aneurysm_rois/001_aneurysm_sac_auto_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_sac_auto
```

3D plane-neck example:

```bash
python -m neuroflow.evaluation.calculate_aneurysm_shape_metrics \
  --seg output/aneurysm_rois/001_aneurysm_sac_plane.nii.gz \
  --neck-json output/aneurysm_rois/001_aneurysm_sac_plane_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_sac_plane
```

Main outputs:

```text
shape_metrics.json
shape_metrics.csv
surface.vtk
volume_mesh.vtk
```

## Step 4: Generate Measurement Overlay

Manual-neck example:

```bash
python -m neuroflow.visualization.visualize_aneurysm_measurements \
  --roi output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_manual/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_manual/measurement_overlay.png
```

Automatic-neck example:

```bash
python -m neuroflow.visualization.visualize_aneurysm_measurements \
  --roi output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_auto/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_auto/measurement_overlay.png
```

The overlay shows:

```text
2D image slices
aneurysm contour
PCA extent lines
3D surface view
metrics table
automatic neck voxels, if available
```

## Step 5: Generate Paper-Style Summary Figure

Manual-neck example:

```bash
python -m neuroflow.evaluation.plot_aneurysm_paper_summary \
  --roi output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_manual/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_manual/paper_style_summary.png
```

Automatic-neck example:

```bash
python -m neuroflow.evaluation.plot_aneurysm_paper_summary \
  --roi output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/subject_001/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_auto/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_auto/paper_style_summary.png
```

This figure shows:

```text
2D measurement view
3D surface view
shape class cue
convex hull view
neck and height lines
```

## What the Neck Means

The neck is the opening where the aneurysm connects to the parent artery.

Manual neck:

```text
distance between the two clicked neck endpoints
```

Automatic neck:

```text
narrowest cross-sectional area near the selected aneurysm/artery connection
```

The automatic neck is converted to an equivalent circular diameter:

```text
neck_width_mm = 2 * sqrt(area / pi)
```

Manual selection is more anatomically controlled. Automatic selection is useful for consistency, but it should be checked visually.

## Mesh Generation Details

The metrics script writes two VTK meshes.

Surface mesh:

```text
file: surface.vtk
format: VTK legacy ASCII POLYDATA
method: marching cubes
iso-level: 0.5
elements: triangular faces
coordinates: physical world coordinates in mm
```

Volume mesh:

```text
file: volume_mesh.vtk
format: VTK legacy ASCII UNSTRUCTURED_GRID
cell type: VTK_VOXEL = 11
elements: one hexahedral voxel cell per foreground voxel
coordinates: physical world coordinates in mm
```

Inspect mesh counts:

```bash
rg -n "^POINTS|^POLYGONS|^CELLS|^CELL_TYPES" \
  output/aneurysm_shape_metrics/001_sac_manual/surface.vtk \
  output/aneurysm_shape_metrics/001_sac_manual/volume_mesh.vtk
```

Example interpretation:

```text
POINTS in surface.vtk   = surface vertices
POLYGONS in surface.vtk = number of surface triangles
CELLS in volume_mesh.vtk = number of voxel/hexahedral cells
```

## Metrics Calculated

Primary size metrics:

```text
volume_mm3
  3D volume occupied by the aneurysm-only mask.

surface_area_mm2
  External surface area from the marching-cubes surface.

manual_neck_width_mm
  Width of the aneurysm opening from the selected or accepted neck.

height_from_neck_plane_mm
  Maximum distance from the neck plane to the aneurysm dome.

equivalent_sphere_diam_mm
  Diameter of a sphere with the same volume.
```

Shape metrics:

```text
sphericity
  How close the aneurysm is to a sphere. Closer to 1 means more spherical.

non_sphericity_index
  Deviation from a sphere. Higher means less spherical.

undulation_index
  Irregularity relative to the convex hull. Higher means more lobulated or uneven.
```

Axis metrics:

```text
axis_major_mm
  Longest 3D extent from PCA.

axis_intermediate_mm
  Middle 3D extent from PCA.

axis_minor_mm
  Shortest 3D extent from PCA.

elongation
  axis_major_mm / axis_minor_mm.

flatness
  axis_intermediate_mm / axis_minor_mm.
```

Ratio metrics:

```text
aspect_ratio_manual
  height_from_neck_plane_mm / manual_neck_width_mm.

bottleneck_factor_manual
  maximum cross-section diameter / manual_neck_width_mm.
```

QC metrics:

```text
bbox_width_mm, bbox_height_mm, bbox_depth_mm
  Bounding-box dimensions. Useful for checking the selected mask, not primary morphology metrics.

num_connected_components
  Number of disconnected regions before keeping the largest one.

centroid_x_mm, centroid_y_mm, centroid_z_mm
  Physical center location of the aneurysm-only mask.
```

## Recommended Metrics to Report

Prioritize:

```text
volume_mm3
surface_area_mm2
manual_neck_width_mm
height_from_neck_plane_mm
aspect_ratio_manual
bottleneck_factor_manual
sphericity
non_sphericity_index
undulation_index
equivalent_sphere_diam_mm
axis_major_mm
axis_intermediate_mm
axis_minor_mm
```

Use bounding-box values mainly for quality control.

## Notes

The final geometry metrics should be computed from the aneurysm-only mask after neck cutting, not from the full CoW segmentation.

If using automatic neck detection, the rough ROI should include the aneurysm and a small connection to the parent artery. If the rough ROI contains only the aneurysm bulge with no connection, the automatic neck becomes less anatomical.

If the automatic cyan neck candidate is not at the artery connection, manually select the neck instead.
