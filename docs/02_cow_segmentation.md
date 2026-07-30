# CoW Segmentation Workflows

This guide explains exactly what each segmentation script does and when to use it.

## 1) `segment_cow_crops.py`

Path:

- `neuroflow/segmentation/segment_cow_crops.py`

Use when:

- you already have cropped CoW volumes in final space (typically after inter-scan registration).

Pipeline:

1. Read each `*.nii.gz`.
2. If input is 4D, convert to 3D using temporal projection.
3. nnU-Net inference with both checkpoints:
   - `checkpoint_best.pth`
   - `checkpoint_final.pth`
4. Ensemble both predictions.
5. Collapse to binary CoW mask:

```text
mask = 1 if label > 0 else 0
```

6. Optional classical vesselness branch (Frangi + Sato).
7. Optional fusion mode (`union`, `intersection`, `ai`, `classic`).
8. Optional postprocess (closing/opening/holes/components).

Output:

- `<input_name>_seg.nii.gz`

## 2) `segment_cow_patient_pipeline.py`

Path:

- `neuroflow/segmentation/segment_cow_patient_pipeline.py`

Use when:

- you start from a patient folder with raw inputs:
  - `input_mag_raw.nii.gz`
  - `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`

Extra preprocessing done by this script:

1. Convert RAW phase to velocity (if RAW-like).
2. Compute speed:

```text
speed = sqrt(vx^2 + vy^2 + vz^2)
```

3. Build angio proxy:

```text
mag_med   = median_t(mag)
speed_pX  = percentile_t(speed, X)
angio_3d  = norm(mag_med) * norm(speed_pX)
```

4. Then run the same segmentation core (AI/classic/ensemble/postprocess).

Output:

- `cow_seg_final.nii.gz` (plus optional intermediates)

## 3) Do They Do the Same CoW Segmentation?

Short answer:

- Yes for the segmentation core (nnU-Net inference + optional classic branch + fusion + postprocess).
- No in input preparation:
  - `segment_cow_crops.py`: assumes input volume is already the angio-like crop to segment.
  - `segment_cow_patient_pipeline.py`: computes an angio volume from MAG+velocity before segmenting.

## 4) 4D -> 3D Projection in `segment_cow_crops.py`

If input is 4D, one 3D volume is generated before nnU-Net using:

- `max`:

```text
proj(x,y,z) = max_t V(x,y,z,t)
```

- `percentile`:

```text
proj(x,y,z) = percentile_t(V, p)
```

- `topk_mean`:

```text
proj(x,y,z) = mean(top_k_t(V))
```

Interpretation:

- `max` is most sensitive, can over-include bright structures.
- `percentile` is usually more robust to spikes/noise.
- `topk_mean` smooths temporal outliers.

## 5) Prevent Mask Expansion (No Dilation-Like Postprocess)

To minimize over-segmentation:

- disable classical branch
- use AI-only ensemble
- disable postprocess

Single case command:

```bash
neuroflow-segment-cow \
  --input /path/to/mag_7T_in_3T.nii.gz \
  --model-dir models/topcow-claim-models \
  --output-dir data/cow_segmentation \
  --projection-method percentile \
  --projection-percentile 95 \
  --no-classic-cow \
  --ensemble-mode ai \
  --no-postprocess
```

Single case command with angiography construction (MAG + Vx/Vy/Vz):

```bash
python -m neuroflow.segmentation.segment_cow_patient_pipeline \
  --patient-dir /path/to/case_folder \
  --mag-name "mag_7T_in_3T.nii.gz" \
  --vx-name "phaseX_7T_in_3T.nii.gz" \
  --vy-name "phaseY_7T_in_3T.nii.gz" \
  --vz-name "phaseZ_7T_in_3T.nii.gz" \
  --model-dir models/topcow-claim-models \
  --output-dir data/cow_segmentation_patient_batch \
  --no-classic-cow \
  --no-postprocess
```

Batch patient-folder command (builds angiography from MAG + velocity):

```bash
python -m neuroflow.segmentation.batch_segment_cow_magnitude \
  --input-root data/registered_7T_in_3T_cow_crop \
  --recursive \
  --mag-pattern "mag_7T_in_3T.nii.gz" \
  --mag-name "mag_7T_in_3T.nii.gz" \
  --vx-name "phaseX_7T_in_3T.nii.gz" \
  --vy-name "phaseY_7T_in_3T.nii.gz" \
  --vz-name "phaseZ_7T_in_3T.nii.gz" \
  --output-dir data/cow_segmentation_patient_batch \
  --ai-only \
  --no-postprocess
```

## 6) Output as Training Mask

For training CSV `mask` column:

- point to generated segmentation NIfTI
- recommended values `{0,1}` in `uint8`
- same HR space/shape as target HR velocity

Fill the `mask` column automatically:

```bash
python -m neuroflow.registration.attach_cow_masks_to_csv \
  --csv-in data/paired_dataset/paired_nifti_cases.csv \
  --csv-out data/paired_dataset/paired_nifti_cases_with_cow_mask.csv \
  --masks-root data/cow_segmentation_patient_batch \
  --mask-name cow_seg_final.nii.gz \
  --hr-col hr_u \
  --hr-root-name hr_7t_in_3t \
  --path-mode relative-to-cwd \
  --strict
```


---

# Appendix: Circle-of-Willis segmentation reference


This document describes the segmentation workflow for Circle of Willis (CoW) when inputs are already:

1. CoW-cropped (YOLO crop step already done), and
2. Inter-scan registered (7T aligned to 3T space).

The main entrypoint for this scenario is:

- `neuroflow/segmentation/segment_cow_crops.py`


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
neuroflow-segment-cow \
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

- `neuroflow/segmentation/segment_cow_patient_pipeline.py`

This script computes a 3D angiography proxy from MAG and velocity-derived speed, then runs the same AI/classic/ensemble logic.

Batch variant (case-by-case using patient folders):

- `neuroflow/segmentation/batch_segment_cow_magnitude.py`

Example (registered 7T-in-3T filenames):

```bash
python -m neuroflow.segmentation.batch_segment_cow_magnitude \
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
python -m neuroflow.segmentation.compute_cow_bbox \
  --input data/temporal_registered_cow_crop \
  --recursive \
  --output-json data/reports/cow_bboxes.json
```


## 8) Legacy End-to-End Script

`neuroflow/segmentation/legacy_segment_circle_of_willis.py` is kept for backward compatibility with an older flow (YOLO crop + nnU-Net in one script).  
For new work with already cropped + registered inputs, prefer `segment_cow_crops.py`.
