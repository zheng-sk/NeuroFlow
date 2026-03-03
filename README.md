# 4DFlowNet
Super Resolution 4D Flow MRI using Residual Neural Network

This is an implementation of the paper [4DFlowNet: Super-Resolution 4D Flow MRI](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full).

## Framework migration status

- Legacy branch implementation: TensorFlow/Keras.
- Current migration branch implementation: PyTorch + MONAI (`requirements.txt`).

Install dependencies for this branch:

```bash
pip install -r requirements.txt
```

If your environment is offline and editable install of `topcow-2024-nnunet` fails, segmentation scripts can still import `nnunetv2` directly from the vendored `topcow-2024-nnunet/` folder in repo root.

Previous requirements files were moved to `requirements/legacy/`.

## Updates 4DFlowNet v2.0
- Loss function has been updated (MSE fluid + MSE non fluid)
- L2 regularization added
- Divergence loss turned off
- Evaluation metrics updated
- Final activation layer switch to linear to allow phase aliasing

These changes are implemented for [Cerebrovascular super-resolution 4D Flow MRI](https://www.biorxiv.org/content/10.1101/2021.08.25.457611v1.full)

## Manuscript version

Original network implementation from the manuscript can be found under the following branches:
- release/manuscript_version (for TF2.0)
- tf1.8 (for Tensorflow 1.8.0)

The pre-trained networks weights can be found here:

- [Original 4DFlowNet pre-trained weights](https://auckland.figshare.com/articles/Super_Resolution_4DFlow_MRI/12253424)

- [Cerebrovascular 4DFlowNet weights](https://auckland.figshare.com/articles/software/Cerebrovascular_4DFlowNet_-_Super_Resolution_4D_Flow_MRI/19158122)

Training dataset is available for download from:
- [Aortic CFD dataset](https://auckland.figshare.com/articles/dataset/4DFlowNet_-_high_resolution_aortic_CFD_dataset/24424888)

# Example results

Below are the example prediction results from an actual 4D Flow MRI of a bifurcation phantom dataset. 

LowRes input (voxel size 4mm)
<p align="left">
    <img src="https://i.imgur.com/O48FbAh.gif" width="330">
</p>

High Res Ground Truth vs noise-free Super Resolution (2mm)
<p align="left">
    <img src="https://i.imgur.com/67CRdGn.gif" width="350">
</p>

High Res Ground Truth vs noise-free Super Resolution (1mm)
<p align="left">
    <img src="https://i.imgur.com/DMQa2Lr.gif" width="350">
</p>


# Recommended workflow (PyTorch + MONAI, no HDF5)

Use direct NIfTI training/inference in this migration branch.  
The old HDF5 patch-index workflow is still available as a legacy option, but it is no longer the recommended default.

Documentation index (recommended entrypoint):

- [`docs/README.md`](docs/README.md)

Task-specific guides:

- Preprocessing: [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md)
- Training: [`docs/TRAINING_WORKFLOW.md`](docs/TRAINING_WORKFLOW.md)
- CoW segmentation: [`docs/SEGMENTATION_COW_WORKFLOW.md`](docs/SEGMENTATION_COW_WORKFLOW.md)
- Inference: [`docs/INFERENCE_WORKFLOW.md`](docs/INFERENCE_WORKFLOW.md)

Quick command (cropped + interscan-registered inputs):

```bash
python code/segmentation/segment_cow_crops.py \
  --input data/registered_7T_in_3T_cow_crop \
  --recursive \
  --model-dir models/topcow-claim-models \
  --output-dir data/cow_segmentation
```

## Direct NIfTI training and prediction (no HDF5)

The migration branch supports an end-to-end MONAI NIfTI workflow:

1. Train directly from NIfTI pairs using `src/trainer_nifti.py`
2. Predict directly from NIfTI and save NIfTI outputs using `code/predict_nifti.py`

### NIfTI training CSV format

Create CSV files (train/val) with one case per row and the following headers:

`lr_u,lr_v,lr_w,lr_mag_u,lr_mag_v,lr_mag_w,hr_u,hr_v,hr_w,mask,venc`

Notes:
- Paths can be absolute or relative to the CSV location.
- `mask` is optional (empty value allowed); when missing, full-volume mask is used.
- `venc` is optional; if omitted or `0`, it is estimated from LR velocity max absolute value.

You can generate these case CSVs from existing legacy patch CSVs (`train.csv`, `validate.csv`) with:

```bash
python src/prepare_data/generate_nifti_case_csv.py \
  --legacy-train-csv data/train.csv \
  --legacy-val-csv data/validate.csv \
  --output-train-csv data/train_nifti.csv \
  --output-val-csv data/validate_nifti.csv \
  --lr-root /path/to/lr_nifti_dir \
  --hr-root /path/to/hr_nifti_dir \
  --strict-exists
```

Default filename patterns:
- LR: `{lr_stem}_u.nii.gz`, `{lr_stem}_v.nii.gz`, `{lr_stem}_w.nii.gz`
- LR magnitude: `{lr_stem}_mag_u.nii.gz`, `{lr_stem}_mag_v.nii.gz`, `{lr_stem}_mag_w.nii.gz`
- HR: `{hr_stem}_u.nii.gz`, `{hr_stem}_v.nii.gz`, `{hr_stem}_w.nii.gz`
- Mask: `{hr_stem}_mask.nii.gz`

Use `--*-pattern` flags in the generator if your naming differs.

### Train from NIfTI

```bash
cd src
python trainer_nifti.py \
  --train-csv /path/to/train_nifti.csv \
  --val-csv /path/to/val_nifti.csv \
  --patch-size 16 \
  --res-increase 2 \
  --batch-size 4 \
  --epochs 60
```

By default, `trainer_nifti.py` assumes velocity inputs are raw phase-like values and converts to physical velocity.
The conversion now auto-detects unsigned/signed RAW ranges:

- unsigned RAW (centered near 2048): `vel = (raw - 2048) / 2048 * venc`
- signed RAW (centered near 0): `vel = raw / scale * venc`, with `scale=2048` or `4096`
- No additional U/V sign inversion (to avoid double inversion when DICOM->NIfTI already applied LPS->RAS sign correction)

If your NIfTI velocity is already physical (m/s), disable this with:

```bash
--already-velocity-input
```

Only for older datasets that still need the extra legacy sign convention, enable:

```bash
--legacy-invert-uv-sign-on-raw
```

Legacy-compatible patch sampling (coverage-aware) can be enabled with:

```bash
cd src
python trainer_nifti.py \
  --train-csv /path/to/train_nifti.csv \
  --val-csv /path/to/val_nifti.csv \
  --patch-size 16 \
  --res-increase 2 \
  --legacy-minimum-coverage 0.2 \
  --legacy-max-sampling-attempts 100
```

Optional strict behavior:
- Add `--legacy-disallow-empty-fallback` to fail if no patch reaches the minimum coverage in the allowed attempts.
- Without this flag, the sampler falls back to the best-coverage patch (legacy-friendly behavior to keep training progressing).

### Predict from NIfTI

```bash
python code/predict_nifti.py \
  --u /path/lr_u.nii.gz --v /path/lr_v.nii.gz --w /path/lr_w.nii.gz \
  --mag-u /path/lr_mag_u.nii.gz --mag-v /path/lr_mag_v.nii.gz --mag-w /path/lr_mag_w.nii.gz \
  --model-path /path/model-best.pt \
  --output-prefix /path/output/pred \
  --patch-size 16 --res-increase 2
```

Output files:
- `<output-prefix>_u.nii.gz`
- `<output-prefix>_v.nii.gz`
- `<output-prefix>_w.nii.gz`
- `<output-prefix>_uvw.nii.gz`

For legacy-style patch reconstruction during prediction (similar to original `PatchGenerator` overlap/trim rules), add:

```bash
--legacy-overlap-inference
```

This mode uses:
- patch size = `patch_size`
- effective stride = `patch_size - 4`
- border trim = 2 LR voxels per side (scaled in HR)
- end padding to enforce exact tiling
- overlap crop-and-stitch back to full HR output

### Detailed transform order (NIfTI training pipeline)

The NIfTI training/validation dataloader in `src/Network/NiftiPatchDataset.py` applies transforms in this order:

1. `LoadImaged`  
   - Loads `lr_u, lr_v, lr_w, lr_mag_u, lr_mag_v, lr_mag_w, hr_u, hr_v, hr_w`, and optional `mask`.

2. `EnsureChannelFirstd`  
   - Converts arrays to channel-first shape.

3. `_StackNormalizeFieldsd` (custom transform)  
   - Stacks LR velocity into `lr_vel` (3 channels).  
   - Stacks HR velocity into `hr_vel` (3 channels).  
   - Stacks LR magnitude into `lr_mag` (3 channels).  
   - By default, converts raw phase-like velocity to physical velocity (auto-detecting unsigned/signed RAW):
     - unsigned RAW: `vel = (raw - 2048) / 2048 * venc`
     - signed RAW: `vel = raw / scale * venc`, with `scale=2048` or `4096`
     - no extra U/V inversion (LPS->RAS sign correction is expected to be baked during DICOM->NIfTI conversion)
     - optional legacy mode: `--legacy-invert-uv-sign-on-raw`
   - Applies normalization:
     - velocity: divide by `venc` (CSV value, or estimated from LR velocity max abs if `venc <= 0`)
     - magnitude: divide by `mag_scale` (default 4095)
   - Builds binary mask (`mask >= mask_threshold`), or full-volume mask if no mask provided.

4. `_RandomVectorRotate90d` (train only)  
   - Random 90/180/270-degree rotation in random plane.
   - Velocity channels use vector-aware rotation logic (axis swap + sign changes).
   - Magnitude and mask are scalar rotations.

5. `_PairedRandomPatchd`  
   - Training:
     - random LR patch of size `patch_size`
     - matching HR patch at scaled coordinates (`patch_size * res_increase`)
     - optional legacy coverage filter (`legacy-minimum-coverage`)
   - Validation:
     - deterministic center patch (no random sampling)

6. Tensor conversion  
   - Returns tensors in the same tuple order expected by `TrainerController`:
     - `u, v, w, u_mag, v_mag, w_mag, u_hr, v_hr, w_hr, venc, mask`

### Synthetic noise augmentation theory (NIfTI training)

Synthetic noise is added in `_AddNonvascularNoiseAugd` (`src/Network/NiftiPatchDataset.py`) after normalization and patch extraction.

1. Sample from a probability family  
   - Draw a base random variable from the selected PDF:
     - `z ~ p(z)` where `p` can be `normal`, `student_t`, `skew_normal`, `generalized_normal`, `hat`, etc.
   - If `--noise-aug-fit-summary-csv` is provided, the best-fit family per channel is loaded once from CSV and reused during training.

2. Apply shape exaggeration (optional)  
   - `z1 = z * side_expand`
   - `z2 = z1 + sign(z1) * edge_boost * |z1|^(edge_power)`
   - `z_tilde = clip(z2, -35, 35)`
   - This widens sides and/or boosts tails before final amplitude scaling.

3. Apply amplitude scaling  
   - A random level is sampled each call: `level ~ Uniform(level_min, level_max)`
   - Effective scale:
     - `scale_eff = scale * range_mult * level`
   - Final additive noise:
     - `n = z_tilde * scale_eff`

4. Apply spatial weighting with mask  
   - `x_noisy = x + w * n`
   - `w = 1` in non-vascular voxels.
   - In vascular voxels, `w = noise_aug_masked_fraction` (default `0.0`, i.e., untouched).

Interpretation:
- The PDF controls the noise *shape* (Gaussian-like, heavy tails, plateau-like, skewed, etc.).
- `scale`, `range_mult`, and `level` control the noise *amplitude*.
- Increasing amplitude terms increases the amount of injected noise.

### Legacy-mode knobs (what they replicate)

- `--legacy-minimum-coverage`  
  Replicates the old `minimum_coverage` concept from patch CSV generation.

- `--legacy-max-sampling-attempts`  
  Replicates the old retry loop behavior when searching for sufficiently covered patches.

- `--legacy-disallow-empty-fallback`  
  If enabled, no fallback patch is accepted when threshold is not met.

- `--legacy-overlap-inference`  
  Uses old patch overlap/trim reconstruction logic instead of MONAI sliding-window blending.

### Important notes when using one magnitude image for U/V/W

If you pass one file via `--mag`, that file is duplicated to `mag_u`, `mag_v`, `mag_w`.
Inside the network, magnitude is computed as:

`mag = sqrt(u_mag^2 + v_mag^2 + w_mag^2)`

So if all three are identical `M`, then `mag = sqrt(3) * M`.

If you want `mag` to behave like a single-channel magnitude value `M`, pre-scale input magnitude by `1/sqrt(3)` before using it for all three channels.


### Key parameters (current NIfTI workflow)

Training (`trainer_nifti.py`):

| Param | Description | Default |
|------|-------------|--------:|
| `patch-size` | LR training patch size (HR uses `patch_size * res_increase`) | 16 |
| `res-increase` | Upsampling ratio | 2 |
| `batch-size` | Training batch size | 4 |
| `epochs` | Number of training epochs | 60 |
| `initial-learning-rate` | Adam learning rate | 2e-4 |
| `low-resblock` | LR residual blocks | 8 |
| `hi-resblock` | HR residual blocks | 4 |
| `train-samples-per-volume` | Random patch samples per train volume per epoch | 64 |
| `val-samples-per-volume` | Patch samples per validation volume per epoch | 16 |
| `legacy-minimum-coverage` | Optional minimum mask coverage per sampled train patch | 0.0 |
| `legacy-max-sampling-attempts` | Max retries to find a patch matching coverage | 100 |
| `noise-aug-masked-fraction` | Fraction of synthetic noise amplitude also applied inside vascular mask (`0..1`) | 0.0 |
| `legacy-invert-uv-sign-on-raw` | Legacy-only extra U/V sign inversion after RAW->m/s conversion | off |

Prediction (`code/predict_nifti.py`):

| Param | Description | Default |
|------|-------------|--------:|
| `patch-size` | LR patch size for inference | 16 |
| `res-increase` | Upsampling ratio | 2 |
| `sw-batch-size` | Inference mini-batch size | 2 |
| `overlap` | Sliding-window overlap ratio | 0.25 |
| `legacy-overlap-inference` | Use legacy overlap/trim patch reconstruction | off |
| `round-small-values` | Zero values below `venc/2048` | off |
| `already-velocity-input` | Disable default raw phase-to-velocity conversion | off |
| `legacy-invert-uv-sign-on-raw` | Legacy-only extra U/V sign inversion after RAW->m/s conversion | off |

## Legacy HDF5 workflow (optional, not recommended)

The following scripts are kept for backward compatibility with previous experiments:
- `src/trainer.py`
- `src/predictor.py`
- `code/inference/batch_predict.py`
- `src/prepare_data/prepare_mri_data.py`
- `src/prepare_data/prepare_nifti_data.py`

Use this legacy path only if you need exact compatibility with older HDF5-based pipelines/checkpoints.
For all new work, prefer direct NIfTI training/prediction documented above.

## Contact Information

If you encounter any problems in using the code, please open an issue in this repository or feel free to contact me by email.

Author: Edward Ferdian (edwardferdian03@gmail.com).
