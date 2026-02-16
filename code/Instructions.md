# 4DFlowNet `code/` Instructions

## Project layout by purpose

```text
code/
├── conversion/            # DICOM/NIfTI/H5 conversion workflows
├── preprocessing/         # Derived feature generation
├── registration/          # Temporal + inter-scan registration and QC
├── inference/             # Model prediction workflows
├── visualization/         # PyVista QC/flow visualization scripts
└── *.py                   # Backward-compatible wrappers (legacy entrypoints)
```

## Recommended workflow

1. **DICOM -> NIfTI**
   - Main script: `code/conversion/dicom_to_nifti.py`
   - Legacy wrapper: `code/convert2nifti2.py`
2. **Prepare input volumes (raw velocity + magnitude)**
   - Script: `code/preprocessing/calculate_mag.py`
   - Default output per case: `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`, `input_mag_raw.nii.gz`
   - Optional outputs: `input_speed_raw.nii.gz` (`--compute-speed`) and `input_pcmra_raw.nii.gz` (`--compute-pcmra`)
   - RAW phase to m/s conversion is **disabled by default** (`--auto-convert-raw-phase` to enable)
3. **Temporal registration (motion correction)**
   - Per folder/patient: `code/registration/batch_register_magnitude.py`
   - Per 4D file: `code/registration/temporal_register_to_t0.py`
   - Progress/timing options: `--show-frame-progress`, `--verbose`, `--timing-report <csv_path>`
4. **(Optional) CoW ROI crop after temporal registration**
   - Script: `code/preprocessing/yolo_crop_patient_pairs.py`
   - Typical input root: temporally registered folders (`.../<CASE>_3T`, `.../<CASE>_7T`)
   - Builds one common ROI per pair (3T+7T) and applies it to all NIfTI files in both folders
   - Current behavior keeps the full **Z axis** and crops only X/Y
5. **Inter-scan 7T -> 3T registration**
   - Per case: `code/registration/register_7T_to_3T_with_qc.py`
   - Batch over a folder: `code/registration/batch_register_7T_to_3T.py`
   - Batch defaults expect temporally-registered velocity names: `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`
   - For CoW-cropped inputs, prefer `--mask-method none` (skip brain-mask estimation)
6. **Full registration pipeline (temporal + 7T -> 3T)**
   - One command over a folder: `code/registration/batch_full_register_7T_to_3T.py`
   - Saves final registered images and brain masks (`BrainMasks/`), and can auto-clean intermediates/QC
   - This full pipeline does **not** include the YOLO crop stage
   - If `temporal-dir` already has the required files, temporal stage is skipped automatically
   - Use `--force-temporal` to force recomputation of temporal registration
   - Optional `--final-only` validates that all final 4D registered outputs exist per case before cleanup
   - Use `--keep-qc` if you want to preserve inter-scan QC folders
   - Optional temporal telemetry: `--show-temporal-frame-progress` and `--temporal-timing-report <csv_path>`
7. **(Optional) Export paired LR/HR NIfTI dataset**
   - Script: `code/registration/export_paired_lr_hr_dataset.py`
   - Creates two folders: `lr_3t/` (temporally registered 3T) and `hr_7t_in_3t/` (7T registered into 3T space)
   - Writes paired CSV compatible with `data/nifti_cases_template.csv`
8. **(Optional) NIfTI -> H5**
   - Script: `code/conversion/nifti_to_h5.py`
9. **(Optional) Phase range quality check**
   - Script: `code/registration/check_phase_ranges.py`
   - Validates whether phase files look like raw phase (`0..4096`) or velocity (m/s)
   - Prints per-file stats (`min/max/percentiles`) and summary, with optional CSV report
10. **(Optional) Batch inference**
   - Script: `code/inference/batch_predict.py`
   - Supports `--input-format h5|nifti` and `--output-format h5|nifti`
   - Direct NIfTI mode bypasses H5 conversion and writes `u_SR.nii.gz`, `v_SR.nii.gz`, `w_SR.nii.gz`
   - Defaults are connected to **temporally registered 3T (LowRes)** inputs: `Vx.nii.gz`, `Vy.nii.gz`, `Vz.nii.gz`, `input_mag_raw.nii.gz`
   - By default only `*_3T` folders are inferred (`--case-suffix _3T`)
   - For evaluation on high-res aligned data, override with `--case-suffix _7T` and custom filenames
   - Optional raw-phase preprocessing in direct NIfTI mode: `--auto-convert-raw-phase` with `--venc` (or per-axis VENC)
   - Progress/timing options: `--show-patch-progress`, `--verbose`, `--timing-report <csv_path>`
   - Device controls: `--device auto|cpu|gpu` and `--gpu-memory-limit-mb <int>`
   - Apple Metal safety: CPU fallback on GPU errors is enabled by default (disable with `--no-cpu-fallback-on-gpu-error`)
11. **(Optional) Visualization / QC with PyVista**
   - Streamlines/panel (interactive): `code/visualization/viz_streamlines.py`
   - 3D streamlines GIF in segmented ROI: `code/visualization/viz_flow_gif.py`
   - Direction QA (forward vs backward): `code/visualization/viz_direction_qa.py`
   - Affine + direction ranking QA (sign/permutation search): `code/visualization/viz_affine_direction_qc.py`
   - Physiological flux QA by planes (`Q(t)=∫v·n dA`): `code/visualization/viz_flux_planes_qc.py`
   - Component QA (Vx/Vy/Vz per CoW point): `code/visualization/viz_component_qc.py`
   - ROI sign QA (3D crop + synchronized 2D slices): `code/visualization/viz_roi_sign_qc.py`
   - Particle advection GIF (flow over time): `code/visualization/viz_particle_gif.py`
   - Slice GIF over time: `code/visualization/viz_slice_gif.py`
   - Vessel surface render (single or dual view): `code/visualization/viz_surface.py`
   - Legacy wrapper still exists: `code/visualization/flow_qc_pyvista.py` (delegates to modular scripts)

## Raw phase conversion behavior

- **`calculate_mag.py` preprocessing**: by default it preserves raw phase values in `Vx/Vy/Vz`; conversion to m/s is only applied when `--auto-convert-raw-phase` is enabled.
- **H5 workflow (`nifti_to_h5`)**: raw phase to m/s conversion is applied during NIfTI -> H5 conversion in `src/prepare_data/prepare_nifti_data.py`.
- **Direct NIfTI inference workflow**: raw phase to m/s conversion is applied at prediction time only when `--auto-convert-raw-phase` is enabled.
- **Sign convention**: DICOM -> NIfTI conversion (`code/conversion/dicom_to_nifti.py`) now bakes LPS->RAS sign correction for phase components (`Vx/Vy` sign inverted, `Vz` unchanged). Downstream raw->m/s conversion should not invert signs again.
- **Orientation convention**: DICOM -> NIfTI conversion canonicalizes magnitude and phase to RAS+ by default. Use `--phase-flips-only` only if you explicitly want to avoid axis permutations in phase components.
- Both workflows support **unsigned raw phase** (`~0..4096`) and **signed raw phase** (`~-4096..4096` / `~-2048..2048`), using VENC.

## Legacy compatible entrypoints

You can still use the old script names in `code/`:
- `convert2nifti2.py`, `calculate_mag.py`, `batch_register_magnitude.py`,
  `batch_temporal_register.py`, `temporal_register_to_t0.py`,
  `convert_to_h5.py`, `batch_predict.py`, `register_7T_to_3T_with_qc.py`,
  `batch_register_7T_to_3T.py`, `batch_full_register_7T_to_3T.py`,
  `export_paired_lr_hr_dataset.py`, `yolo_crop_patient_pairs.py`.

These wrappers redirect to the reorganized modules.

## Visualization scripts (modular)

Use these scripts depending on the figure you want:

- **Interactive flow exploration (slice / stream / panel):**
  - `python code/visualization/viz_streamlines.py ...`
- **3D GIF of segmented flow streamlines:**
  - `python code/visualization/viz_flow_gif.py ...`
- **Direction QA (forward/backward + optional signed components):**
  - `python code/visualization/viz_direction_qa.py ...`
- **Affine + direction ranking QA (best sign/permutation candidate):**
  - `python code/visualization/viz_affine_direction_qc.py ...`
- **Physiological flux QA with anatomical planes:**
  - `python code/visualization/viz_flux_planes_qc.py ...`
- **Component QA (signed Vx/Vy/Vz + per-point CSV export):**
  - `python code/visualization/viz_component_qc.py ...`
- **ROI sign QA (interactive 3D crop + synchronized X/Y/Z 2D slices):**
  - `python code/visualization/viz_roi_sign_qc.py ...`
- **Particle advection GIF (blood flow over time):**
  - `python code/visualization/viz_particle_gif.py ...`
- **Simple slice GIF through time:**
  - `python code/visualization/viz_slice_gif.py ...`
- **Surface visualization (closer to CoW render style):**
  - `python code/visualization/viz_surface.py ...`

### Example: `data/test_pyvista` (streamlines panel)

```bash
python code/visualization/viz_streamlines.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --mode panel \
  --frame 0 \
  --auto-convert-raw \
  --venc 1.0 \
  --seed-mode speed \
  --speed-seed-percentile 98.0 \
  --n-seed-points 220 \
  --zero-velocity-below 0.05 \
  --stream-color-by-speed \
  --background black
```

### Example: interactive ROI clip + component view (`Vx`/`Vy`/`Vz`)

```bash
python code/visualization/viz_streamlines.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --mode stream \
  --scalar vx \
  --stream-color-by-speed \
  --interactive-clip-box \
  --background black
```

Notes:
- Set `--scalar` to `vx`, `vy`, `vz`, or `speed`
- Use `--clip-box-bounds x0,x1,y0,y1,z0,z1` for a fixed ROI crop
- Add `--clip-invert` to keep outside instead of inside

### Example: software-like interactive stream editor (no relaunch)

```bash
python code/visualization/viz_streamlines.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --mode stream \
  --scalar speed \
  --interactive-controls \
  --interactive-clip-box \
  --roi-only-interaction \
  --clip-widget-no-translation \
  --background black
```

In-window controls:
- Sliders: seeds, seed percentile, step, max length, tube radius, arrow stride
- Keys: `1/2/3/4` scalar (`speed/vx/vy/vz`), `n/m` seeds, `[/]` percentile, `,/.` tube, `;/'` step, `k/l` max length, `i` integration direction, `f` arrows, `c` color, `b` invert clip, `h` HUD, `u` reset camera

### Example: interactive ROI + synchronized 2D sign check (recommended for Vx/Vy/Vz validation)

```bash
python code/visualization/viz_roi_sign_qc.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --frame 0 \
  --scalar vx \
  --interactive-clip-box \
  --background white
```

In-window keys for `viz_roi_sign_qc.py`:
- `1/2/3/4`: switch scalar (`speed/vx/vy/vz`)
- `c`: toggle ROI clip on/off
- `b`: show/hide bounding box widget
- `h`: show/hide HUD text
- `r`: reset 3D camera

### Example: `data/test_pyvista` (3D flow GIF in segmented region)

```bash
python code/visualization/viz_flow_gif.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --gif-path data/test_pyvista/flow3d_segmented.gif \
  --auto-convert-raw \
  --venc 1.0 \
  --seed-mode speed \
  --speed-seed-percentile 98.0 \
  --n-seed-points 220 \
  --zero-velocity-below 0.05 \
  --integration-direction forward \
  --show-direction-arrows \
  --arrow-stride 35 \
  --arrow-factor 0.8 \
  --max-length 120 \
  --step 0.2 \
  --tube-radius 0.22 \
  --stream-color-by-speed \
  --gif-fps 12 \
  --orbit-deg-per-frame 2.0 \
  --camera-distance-scale 1.5 \
  --background black
```

### Example: direction QA (forward vs backward)

```bash
python code/visualization/viz_direction_qa.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --seed-mode speed \
  --speed-seed-percentile 97.0 \
  --n-seed-points 400 \
  --max-length 140 \
  --step 0.15 \
  --tube-radius 0.20 \
  --signed-component vx \
  --show-direction-arrows \
  --background black
```

### Example: affine + direction ranking QA

```bash
python code/visualization/viz_affine_direction_qc.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --frame-step 4 \
  --max-frames 6 \
  --top-k 10 \
  --affine-lambda 0.8 \
  --export-ranking-csv data/test_pyvista/affine_direction_ranking.csv
```

### Example: physiological flux QA with inlet/outlet planes

```bash
python code/visualization/viz_flux_planes_qc.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --all-frames \
  --dt-seconds 0.04 \
  --plane "ICA_L:-18.0,-9.0,7.0:1.0,0.0,0.0:2.5:70:in" \
  --plane "ICA_R: 18.0,-9.0,7.0:-1.0,0.0,0.0:2.5:70:in" \
  --plane "BASILAR:0.0,-20.0,3.0:0.0,1.0,0.0:2.5:70:in" \
  --plane "MCA_L:-25.0,5.0,10.0:-1.0,0.0,0.0:2.0:70:out" \
  --plane "MCA_R: 25.0,5.0,10.0: 1.0,0.0,0.0:2.0:70:out" \
  --plane "ACA:0.0,18.0,13.0:0.0,1.0,0.0:2.0:70:out" \
  --auto-flip-negative-role \
  --csv-out data/test_pyvista/flux_planes_qc.csv \
  --summary-out data/test_pyvista/flux_planes_qc_summary.txt \
  --plot-png data/test_pyvista/flux_planes_qc.png
```

`--plane` format:
- `name:cx,cy,cz:nx,ny,nz[:radius_mm[:resolution[:role]]]`
- `role` can be `in`, `out`, or `none`

### Example: component QA (Vx/Vy/Vz at each CoW point)

```bash
python code/visualization/viz_component_qc.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --aggregate frame \
  --frame 0 \
  --report-all-frames \
  --export-csv data/test_pyvista/cow_component_values_frame0.csv \
  --show-glyphs \
  --glyph-stride 35 \
  --glyph-factor 1.8 \
  --background black
```

### Example: particle flow GIF (time-evolving blood flow)

```bash
python code/visualization/viz_particle_gif.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --gif-path data/test_pyvista/particle_flow.gif \
  --auto-convert-raw \
  --venc 1.0 \
  --n-particles 900 \
  --spawn-per-frame 120 \
  --trail-length 8 \
  --speed-seed-percentile 95.0 \
  --zero-velocity-below 0.05 \
  --dt-scale 0.6 \
  --gif-fps 12 \
  --background black
```

### Example: surface render style (single/dual view)

```bash
python code/visualization/viz_surface.py \
  --vx-path data/test_pyvista/4dflow_v100_inplane_rl_7.nii.gz \
  --vy-path data/test_pyvista/4dflow_v100_inplane_ap_6.nii.gz \
  --vz-path data/test_pyvista/4dflow_v100_through_8.nii.gz \
  --mag-path data/test_pyvista/4dflow_mag.nii.gz \
  --mask data/test_pyvista/4dflow_mag_mask.nii.gz \
  --auto-convert-raw \
  --venc 1.0 \
  --scalar speed \
  --aggregate mean \
  --two-views \
  --off-screen \
  --screenshot data/test_pyvista/cow_surface_speed_dual.png
```

## Minimal examples

```bash
# 1) DICOM -> NIfTI
python code/conversion/dicom_to_nifti.py \
  --input-root data/sorted_patients \
  --output-root data/nifti_patients

# 2) Prepare processed inputs (RAW velocity + magnitude only, default)
python code/preprocessing/calculate_mag.py \
  --csv data/dataset.csv \
  --data-root data

# 2b) Optional: also compute speed + PC-MRA (and optionally convert RAW -> m/s)
python code/preprocessing/calculate_mag.py \
  --csv data/dataset.csv \
  --data-root data \
  --compute-speed \
  --compute-pcmra \
  --auto-convert-raw-phase \
  --venc 0.90

# 3a) Temporal registration per folder
python code/registration/batch_register_magnitude.py \
  --input-dir data/processed_inputs \
  --output-dir data/temporal_registered \
  --reg-type Rigid \
  --show-frame-progress \
  --timing-report data/reports/temporal_timing.csv

# 3b) Temporal registration per file
python code/registration/temporal_register_to_t0.py \
  --input data/processed_inputs/CASE/input_mag_raw.nii.gz \
  --out data/temporal_registered/CASE/input_mag_raw.nii.gz \
  --qc_dir data/temporal_registered/CASE/QC

# 4a) Inter-scan registration (single case, 7T -> 3T)
python code/registration/register_7T_to_3T_with_qc.py \
  --fixed_mag_3t data/temporal_registered/001_20240313_3T/input_mag_raw.nii.gz \
  --moving_mag_7t data/temporal_registered/001_20240313_7T/input_mag_raw.nii.gz \
  --moving_phase_x data/temporal_registered/001_20240313_7T/Vx.nii.gz \
  --moving_phase_y data/temporal_registered/001_20240313_7T/Vy.nii.gz \
  --moving_phase_z data/temporal_registered/001_20240313_7T/Vz.nii.gz \
  --out_dir data/registered_7T_in_3T/001_20240313 \
  --qc_dir data/registered_7T_in_3T/001_20240313/QC

# 4b) Inter-scan registration (batch)
python code/registration/batch_register_7T_to_3T.py \
  --input-dir data/temporal_registered \
  --output-dir data/registered_7T_in_3T \
  --phase-x-name Vx.nii.gz \
  --phase-y-name Vy.nii.gz \
  --phase-z-name Vz.nii.gz \
  --fixed-suffix _3T \
  --moving-suffix _7T

# 4c) CoW ROI crop after temporal registration (common crop for 3T+7T pair)
python code/preprocessing/yolo_crop_patient_pairs.py \
  --input-dir data/temporal_registered \
  --output-dir data/temporal_registered_cow_crop \
  --yolo-model models/yolo-cow-detection.pt \
  --yolo-device mps \
  --margin-xy 20 \
  --shape-mismatch skip \
  --write-folder-plans

# 4d) Inter-scan registration over cropped temporal inputs (no brain masks)
python code/registration/batch_register_7T_to_3T.py \
  --input-dir data/temporal_registered_cow_crop \
  --output-dir data/registered_7T_in_3T_cow_crop \
  --phase-x-name Vx.nii.gz \
  --phase-y-name Vy.nii.gz \
  --phase-z-name Vz.nii.gz \
  --fixed-suffix _3T \
  --moving-suffix _7T \
  --mask-method none

# 5) Full pipeline: temporal first, then 7T -> 3T batch
python code/registration/batch_full_register_7T_to_3T.py \
  --input-dir data/processed_inputs \
  --output-dir data/registered_7T_in_3T \
  --show-temporal-frame-progress \
  --temporal-timing-report data/reports/temporal_timing.csv \
  --phase-x-name Vx.nii.gz \
  --phase-y-name Vy.nii.gz \
  --phase-z-name Vz.nii.gz \
  --final-only \
  --keep-qc \
  --fixed-suffix _3T \
  --moving-suffix _7T

# 6) Export paired LR/HR folders + CSV for training/prediction
python code/registration/export_paired_lr_hr_dataset.py \
  --temporal-dir data/registered_7T_in_3T/_temporal_registered \
  --registered-dir data/registered_7T_in_3T \
  --output-root data/paired_dataset \
  --mode copy \
  --venc 0.90 \
  --csv-path data/paired_dataset/paired_nifti_cases.csv

# 7) Validate phase ranges before conversion/prediction
python code/registration/check_phase_ranges.py \
  --input-root data/registered_7T_in_3T/_temporal_registered \
  --recursive \
  --expected-mode velocity \
  --venc 0.90 \
  --csv-report data/reports/phase_range_check.csv \
  --verbose

# 7b) Validate signed raw phase ranges (e.g., approximately -4096..4096)
python code/registration/check_phase_ranges.py \
  --input-root data/paired_dataset/hr_7t_in_3t \
  --recursive \
  --expected-mode raw \
  --raw-min -4096 \
  --raw-max 4096 \
  --csv-report data/reports/phase_range_check_raw_signed.csv

# 8) Prepare H5 inputs from paired LR NIfTI (for H5 prediction workflow)
python code/conversion/nifti_to_h5.py \
  --input-root data/paired_dataset/lr_3t \
  --output-dir data/paired_dataset/h5_lr \
  --venc 0.90

# 9) H5 prediction (batch, CUDA GPU)
python code/inference/batch_predict.py \
  --input-format h5 \
  --output-format h5 \
  --input-dir data/paired_dataset/h5_lr \
  --h5-pattern "*_3T.h5" \
  --output-dir data/predictions_h5_3t_gpu \
  --batch-size 8 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_h5_3t_gpu.csv \
  --device gpu

# 10) H5 prediction (batch, CPU fallback-safe)
python code/inference/batch_predict.py \
  --input-format h5 \
  --output-format h5 \
  --input-dir data/paired_dataset/h5_lr \
  --h5-pattern "*_3T.h5" \
  --output-dir data/predictions_h5_3t_cpu \
  --batch-size 2 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_h5_3t_cpu.csv \
  --device cpu

# 11) Direct NIfTI prediction (batch, 3T only, CUDA GPU, SR x2)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_sr2 \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_3t_gpu_sr2.csv \
  --device gpu

# 12) Direct NIfTI prediction (batch, 3T only, same-resolution mode SR x1)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_sr1 \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 1 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_3t_gpu_sr1.csv \
  --device gpu

# 13) Direct NIfTI prediction with raw-phase conversion enabled (use only for raw phase inputs)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/paired_dataset/lr_3t \
  --output-dir data/predictions_nifti_3t_gpu_raw \
  --recursive \
  --case-suffix '' \
  --u-name Vx.nii.gz \
  --v-name Vy.nii.gz \
  --w-name Vz.nii.gz \
  --mag-name input_mag_raw.nii.gz \
  --batch-size 4 \
  --res-increase 2 \
  --show-patch-progress \
  --timing-report data/reports/predict_nifti_3t_gpu_raw.csv \
  --device gpu \
  --auto-convert-raw-phase \
  --venc 0.90

# 14) Single-case direct NIfTI prediction (quick debug)
python code/inference/batch_predict.py \
  --input-format nifti \
  --output-format nifti \
  --input-dir data/registered_7T_in_3T/_temporal_registered/001_20240313_3T \
  --output-dir data/predictions_nifti_single \
  --batch-size 1 \
  --res-increase 2 \
  --show-patch-progress \
  --verbose \
  --timing-report data/reports/predict_nifti_single.csv \
  --device gpu
```
