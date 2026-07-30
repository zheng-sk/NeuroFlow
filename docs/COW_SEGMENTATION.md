# CoW Semantic Segmentation (Cropped + Registered Inputs)

This document describes the segmentation workflow for Circle of Willis (CoW) when inputs are already:

1. CoW-cropped (YOLO crop step already done), and
2. Inter-scan registered (7T aligned to 3T space).

The main entrypoint for this scenario is:

- `code/segmentation/segment_cow_crops.py`


## 1) Installation

From repository root:

```bash
pip install -r requirements.txt
```

This installs:

- base project dependencies (PyTorch + MONAI),
- YOLO dependencies (`ultralytics`, `opencv-python-headless`),
- morphology/classic branch deps (`scikit-image`, `SimpleITK`, `batchgenerators`),
- local nnU-Net v2 package vendored in this repo (`topcow-2024-nnunet`).

Note:

- segmentation scripts also support fallback import directly from `topcow-2024-nnunet/` if the folder exists in repo root.
- editable install is still recommended for reproducibility.


## 2) Required Models

Expected default model locations:

- YOLO detector: `models/yolo-cow-detection.pt`
- nnU-Net model folder: `models/topcow-claim-models/`

Expected nnU-Net folder layout:

```text
models/topcow-claim-models/
├── dataset.json
├── plans.json
├── fold_0/checkpoint_best.pth
├── fold_0/checkpoint_final.pth
...
└── fold_4/checkpoint_best.pth
```


## 3) Main Workflow for Cropped + Registered Inputs

Use this command when each input `.nii.gz` is already a CoW crop in final registration space:

```bash
python code/segmentation/segment_cow_crops.py \
  --input data/registered_7T_in_3T_cow_crop/subject_001 \
  --model-dir models/topcow-claim-models \
  --output-dir data/cow_segmentation
```

You can pass:

- a single file (`--input /path/case.nii.gz`), or
- a folder with multiple `.nii.gz` volumes.
- if your directory contains nested case folders, add `--recursive`.

Outputs are saved as:

- `<input_name>_seg.nii.gz` in `--output-dir`


## 4) Segmentation Logic

`segment_cow_crops.py` runs:

1. Input harmonization:
- accepts 3D or 4D NIfTI.
- if 4D, projects to 3D (`max`, `percentile`, or `topk_mean`) before nnU-Net inference.

2. AI branch:
- predicts with `checkpoint_best.pth` and `checkpoint_final.pth`.
- ensembles both predictions.
- collapses labels to a binary CoW mask (`label=1`).

Velocity/sign note:

- the patient-folder pipeline builds angiography from speed (`sqrt(vx^2 + vy^2 + vz^2)`), so no extra CoW-stage sign inversion is required.
- keep the repository sign convention at DICOM->NIfTI stage and avoid adding extra U/V flips here.

3. Optional classical branch:
- computes vesselness (Frangi + Sato).
- thresholds by percentile and applies morphology + connected-component filtering.

4. AI/classical fusion:
- `union` (default), `intersection`, `ai`, or `classic`.

5. Optional postprocessing:
- closing/opening,
- hole filling,
- small-component removal.


## 5) Useful Flags

- `--projection-method max|percentile|topk_mean`
- `--no-classic-cow` (AI-only segmentation)
- `--ensemble-mode union|intersection|ai|classic`
- `--no-postprocess`
- `--post-close-radius`, `--post-open-radius`, `--post-min-component-size`
- `--temp-root` and `--keep-temp` for nnU-Net staging/debug


## 6) Patient-Folder Pipeline (Optional)

If you start from a patient folder containing `input_mag_raw.nii.gz`, `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`, use:

- `code/segmentation/segment_cow_patient_pipeline.py`

This script computes a 3D angiography proxy from MAG and velocity-derived speed, then runs the same AI/classic/ensemble logic.

Batch variant (case-by-case using patient folders):

- `code/segmentation/batch_segment_cow_magnitude.py`

Example (registered 7T-in-3T filenames):

```bash
python code/segmentation/batch_segment_cow_magnitude.py \
  --input-root data/registered_7T_in_3T_cow_crop \
  --recursive \
  --mag-pattern "mag_7T_in_3T.nii.gz" \
  --mag-name "mag_7T_in_3T.nii.gz" \
  --vx-name "phaseX_7T_in_3T.nii.gz" \
  --vy-name "phaseY_7T_in_3T.nii.gz" \
  --vz-name "phaseZ_7T_in_3T.nii.gz" \
  --output-dir data/cow_segmentation_patient_batch \
  --ensemble-mode union
```


## 7) YOLO BBox Utility (Optional)

To only compute CoW bounding boxes (without nnU-Net segmentation), use:

```bash
python code/segmentation/compute_cow_bbox.py \
  --input data/temporal_registered_cow_crop \
  --recursive \
  --output-json data/reports/cow_bboxes.json
```


## 8) Legacy End-to-End Script

`code/segmentation/legacy_segment_circle_of_willis.py` is kept for backward compatibility with an older flow (YOLO crop + nnU-Net in one script).  
For new work with already cropped + registered inputs, prefer `segment_cow_crops.py`.
