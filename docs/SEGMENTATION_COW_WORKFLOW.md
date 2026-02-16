# CoW Segmentation Workflows

This guide explains exactly what each segmentation script does and when to use it.

## 1) `segment_cow_crops.py`

Path:

- `code/segmentation/segment_cow_crops.py`

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

- `code/segmentation/segment_cow_patient_pipeline.py`

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
python code/segmentation/segment_cow_crops.py \
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
python code/segmentation/segment_cow_patient_pipeline.py \
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
python code/segmentation/batch_segment_cow_magnitude.py \
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
