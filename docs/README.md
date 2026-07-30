# NeuroFlow documentation

Stage-by-stage detail behind the pipeline table in the
[main README](../README.md). Numbering follows the README's stages.

| Doc | Covers | README stages |
| --- | --- | --- |
| [`01_preprocessing.md`](01_preprocessing.md) | Study organisation, DICOM → NIfTI, magnitude/speed/PC-MRA, temporal motion correction, YOLO CoW ROI crop, inter-scan registration. Includes the flowchart, per-phase risks, and QC checklist. | 1–6 |
| [`02_cow_segmentation.md`](02_cow_segmentation.md) | Circle-of-Willis segmentation: when to use `segment_cow_crops` vs `segment_cow_patient_pipeline`, nnU-Net ensembling, classical vesselness, and mask post-processing. | 7 |
| [`03_dataset_and_splits.md`](03_dataset_and_splits.md) | Paired LR/HR export, trigger-time frame pairing, the manifest schema, ×2 low-resolution generation, and patient-level LOOCV splits. | 8–10 |
| [`04_training.md`](04_training.md) | Building training CSVs, normalisation equations, mask behaviour, patch sampling, noise augmentation, and loss definitions. | 11 |
| [`05_inference.md`](05_inference.md) | Direct NIfTI prediction, sliding-window vs legacy overlap reconstruction, and output interpretation. | 12 |
| [`06_evaluation_metrics.md`](06_evaluation_metrics.md) | Intraluminal velocity statistics, flow/Qref, WSS, background suppression, geometry distances, statistical tests, and the aneurysm geometry appendix. | 13 |
| [`07_reproducing_the_paper.md`](07_reproducing_the_paper.md) | Configuration IDs A–G mapped to commands and manuscript table rows. | — |

## Conventions

- All commands assume the package is installed (`pip install -e ".[viz,segmentation]"`).
  Nothing requires `cd` or a `PYTHONPATH`.
- Modules run as `python -m neuroflow.<subpackage>.<module>`; the four most
  common stages also have console entry points (`neuroflow-dicom2nifti`,
  `neuroflow-train`, `neuroflow-predict`, `neuroflow-segment-cow`).
- Example paths use placeholder subject IDs such as `subject_001`. Real cohort
  identifiers are not published; see Data availability in the main README.
