# Aneurysm Geometry Metrics Workflow

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
001_20240313
```

Main inputs:

```text
3D/4D magnitude image:
data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz

CoW segmentation:
data/paired_dataset/cow_segmentation_ens301_current/001_20240313/cow_seg_final.nii.gz
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
.venv_neuroflow/bin/python code/inference/select_aneurysm_neck_plane.py \
  --roi output/nii-cropped/001/3T/mask.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --out output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz
```

Then calculate metrics:

```bash
.venv_neuroflow/bin/python code/inference/calculate_aneurysm_shape_metrics.py \
  --seg output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz \
  --neck-json output/aneurysm_plane_precrops/001_aneurysm_plane_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_precrop_plane
```

Generate QC overlay:

```bash
.venv_neuroflow/bin/python code/inference/visualize_aneurysm_measurements.py \
  --roi output/aneurysm_plane_precrops/001_aneurysm_plane.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_precrop_plane/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_precrop_plane/measurement_overlay.png
```

## Precrop Workflow: Automatic 3D Test

The automatic mode can be used as a screening test:

```bash
.venv_neuroflow/bin/python code/inference/select_aneurysm_neck.py \
  --mode auto \
  --roi output/nii-cropped/001/3T/mask.nii.gz \
  --bg output/nii-cropped/001/7T/input_mag_raw.nii.gz \
  --out output/aneurysm_precrop_auto/001_aneurysm_auto.nii.gz
```

Then calculate metrics:

```bash
.venv_neuroflow/bin/python code/inference/calculate_aneurysm_shape_metrics.py \
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
.venv_neuroflow/bin/python code/inference/select_aneurysm_roi.py \
  --seg data/paired_dataset/cow_segmentation_ens301_current/001_20240313/cow_seg_final.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
.venv_neuroflow/bin/python code/inference/select_aneurysm_neck.py \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
.venv_neuroflow/bin/python code/inference/select_aneurysm_neck.py \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
.venv_neuroflow/bin/python code/inference/select_aneurysm_neck_plane.py \
  --roi output/aneurysm_rois/001_aneurysm_roi_with_connection.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
.venv_neuroflow/bin/python code/inference/calculate_aneurysm_shape_metrics.py \
  --seg output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --neck-json output/aneurysm_rois/001_aneurysm_sac_manual_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_sac_manual
```

Automatic-neck example:

```bash
.venv_neuroflow/bin/python code/inference/calculate_aneurysm_shape_metrics.py \
  --seg output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --neck-json output/aneurysm_rois/001_aneurysm_sac_auto_neck.json \
  --out-dir output/aneurysm_shape_metrics/001_sac_auto
```

3D plane-neck example:

```bash
.venv_neuroflow/bin/python code/inference/calculate_aneurysm_shape_metrics.py \
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
.venv_neuroflow/bin/python code/inference/visualize_aneurysm_measurements.py \
  --roi output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_manual/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_manual/measurement_overlay.png
```

Automatic-neck example:

```bash
.venv_neuroflow/bin/python code/inference/visualize_aneurysm_measurements.py \
  --roi output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
.venv_neuroflow/bin/python code/inference/plot_aneurysm_paper_summary.py \
  --roi output/aneurysm_rois/001_aneurysm_sac_manual.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
  --metrics-json output/aneurysm_shape_metrics/001_sac_manual/shape_metrics.json \
  --out output/aneurysm_shape_metrics/001_sac_manual/paper_style_summary.png
```

Automatic-neck example:

```bash
.venv_neuroflow/bin/python code/inference/plot_aneurysm_paper_summary.py \
  --roi output/aneurysm_rois/001_aneurysm_sac_auto.nii.gz \
  --bg data/paired_dataset/hr_7t_in_3t/001_20240313/input_mag_raw.nii.gz \
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
