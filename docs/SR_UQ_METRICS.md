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

## 10) References

1. Alzate et al. *Uncertainty Quantification in Hemodynamic Metrics from 4D Flow MRI with Super-resolution in a Carotid Bifurcation Model.* Journal of Digital Imaging (paper provided in project materials).
2. Wilcoxon F. *Individual Comparisons by Ranking Methods.* Biometrics Bulletin, 1945.
3. MONAI documentation (sliding-window inference concept used in validation/inference pipelines).
