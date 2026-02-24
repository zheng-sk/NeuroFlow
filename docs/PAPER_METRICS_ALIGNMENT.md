# Paper Metrics Alignment

This note tracks how current report outputs align with the two target papers:

- `Uncertainty Quantification in Hemodynamic Metrics from 4D Flow MRI with Super-resolution in a Carotid Bifurcation Model`
- `Cerebrovascular super-resolution 4D Flow MRI - using deep learning to non-invasively quantify velocity, flow, and relative pressure`

## Implemented in current pipeline

### Carotid UQ paper style

- Surface and Hausdorff geometry metrics (temporal summary).
- Intraluminal velocity/vorticity statistics (mean, SD, skewness, kurtosis) with relative error.
- Wilcoxon comparison for intraluminal relative errors.
- Flow analysis with temporal profile and per-frame error tables.
- WSS metrics and WSS distributions available behind `--include-wss`.
- Bland-Altman and correlation diagnostics (baseline vs reference, SR vs reference).

### Cerebrovascular SR paper style (feasible subset)

- Correlation and Bland-Altman for velocity magnitude and temporal flow.
- Peak-flow component-wise correlation and Bland-Altman (`u`, `v`, `w`, `|v|`).
- Peak velocity metrics in `core` and `wall` regions:
  - MAE
  - RMSE
  - Relative error (%)
  - Cosine similarity
- Flow peak and temporal error summaries:
  - peak absolute/relative error
  - RMSE over time
  - mean relative error over time

## Pending (not yet implemented)

- Relative pressure metrics from vWERP workflow:
  - pressure traces
  - pressure RMSE / relative error
  - pressure correlation and Bland-Altman

Reason: this repository currently has no vWERP pressure solver pipeline integrated in `code/inference/`.

## Output artifacts for current alignment

- `metrics/correlation_metrics.csv`
- `metrics/bland_altman_stats.csv`
- `metrics/peak_velocity_metrics.csv`
- `metrics/flow_peak_metrics.csv`
- `metrics/summary_metrics.json`

## Recommended next milestone

Integrate a deterministic pressure module (vWERP-equivalent), then add:

1. Pressure trace extraction for matched inlet/outlet section pairs.
2. Pressure regression/Bland-Altman vs reference (in-silico).
3. Pressure comparison between resolution sets (in-vivo).
