# NeuroFlow Documentation Index

This folder is split by task so each workflow is easier to run and maintain.

## Documents

- `docs/PREPROCESSING_WORKFLOW.md`
  - End-to-end preprocessing from DICOM/NIfTI to training-ready paired data.
  - Includes: DICOM->NIfTI, `calculate_mag`, temporal registration, CoW ROI crop, inter-scan registration, optional CoW segmentation.

- `docs/TRAINING_WORKFLOW.md`
  - How to build training CSVs and train `4DFlowNet` from NIfTI.
  - Includes normalization equations, mask behavior, patch sampling, and loss definitions.

- `docs/SEGMENTATION_COW_WORKFLOW.md`
  - CoW segmentation pipelines and when to use each script.
  - Includes exact differences between `segment_cow_crops.py` and `segment_cow_patient_pipeline.py`.

- `docs/INFERENCE_WORKFLOW.md`
  - Direct NIfTI prediction with `code/predict_nifti.py` and output interpretation.

- `docs/SR_UQ_METRICS.md`
  - Definitions and interpretation for SR uncertainty-quantification metrics.
  - Includes: Table-2-like intraluminal stats, flow/Qref metrics, WSS metrics, geometry distances, flow-axis selection, and statistical tests.

## Legacy docs (kept for compatibility)

- `docs/PREPROCESSING_PIPELINE.md`
- `docs/COW_SEGMENTATION.md`
