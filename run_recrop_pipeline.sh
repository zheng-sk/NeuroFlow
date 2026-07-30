#!/usr/bin/env bash
# Recompute pipeline after fixing the L-R ROI extent.
#
# Steps:
#   1. Re-crop with symmetric X margin (50 px each side, was 20 px left-only)
#   2. Re-register 7T -> 3T on the new crops
#   3. Re-segment CoW to rebuild masks
#   4. Re-export LR (3T) and unmasked HR (7T-in-3T)
#   5. Apply masks to create hr_7t_in_3t_masked

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ROOT
GPU=${GPU:-0}

# ---------------------------------------------------------------------------
# Step 1: Re-crop with symmetric X margin
# ---------------------------------------------------------------------------
echo
echo "=== Step 1: YOLO crop (margin-x 50, both sides) ==="
python "${ROOT}/code/preprocessing/yolo_crop_patient_pairs.py" \
  --input-dir  "${ROOT}/data/temporal_registered" \
  --output-dir "${ROOT}/data/temporal_registered_cow_crop" \
  --yolo-model "${ROOT}/models/yolo-cow-detection.pt" \
  --fixed-suffix _3T \
  --moving-suffix _7T \
  --margin-x 50 \
  --margin-x-side both \
  --margin-y 30 \
  --margin-y-side both \
  --write-folder-plans \
  --output-json "${ROOT}/data/temporal_registered_cow_crop/plans.json"

# ---------------------------------------------------------------------------
# Step 2: Re-register 7T -> 3T on the new crops
# ---------------------------------------------------------------------------
echo
echo "=== Step 2: Inter-scan registration 7T -> 3T ==="
CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/code/registration/batch_register_7T_to_3T.py" \
  --input-dir  "${ROOT}/data/temporal_registered_cow_crop" \
  --output-dir "${ROOT}/data/registered_7T_in_3T_cow_crop" \
  --mask-method none \
  --phase-warp-mode direct \
  --interpolator-phase nearestNeighbor \
  --overwrite

# ---------------------------------------------------------------------------
# Step 3: Re-segment CoW on the new registered data
# ---------------------------------------------------------------------------
echo
echo "=== Step 3: CoW segmentation ==="
CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/code/segmentation/batch_segment_cow_magnitude.py" \
  --input-root  "${ROOT}/data/registered_7T_in_3T_cow_crop" \
  --recursive \
  --mag-pattern "mag_7T_in_3T.nii.gz" \
  --mag-name    "mag_7T_in_3T.nii.gz" \
  --vx-name     "phaseX_7T_in_3T.nii.gz" \
  --vy-name     "phaseY_7T_in_3T.nii.gz" \
  --vz-name     "phaseZ_7T_in_3T.nii.gz" \
  --output-dir  "${ROOT}/data/paired_dataset/cow_segmentation_ens301_current" \
  --ensemble-mode union

# ---------------------------------------------------------------------------
# Step 4: Re-export LR (3T crop) and unmasked HR (registered 7T)
# ---------------------------------------------------------------------------
echo
echo "=== Step 4: Export paired LR / HR dataset ==="
python "${ROOT}/code/registration/export_paired_lr_hr_dataset.py" \
  --temporal-dir   "${ROOT}/data/temporal_registered_cow_crop" \
  --registered-dir "${ROOT}/data/registered_7T_in_3T_cow_crop" \
  --output-root    "${ROOT}/data/paired_dataset" \
  --mode copy \
  --overwrite

# ---------------------------------------------------------------------------
# Step 5: Apply masks to create hr_7t_in_3t_masked
# ---------------------------------------------------------------------------
echo
echo "=== Step 5: Apply CoW masks to HR data ==="
python - <<'EOF'
import os, sys
from pathlib import Path
import nibabel as nib
import numpy as np

repo_root = Path(os.environ['ROOT'])

hr_src   = repo_root / "data" / "paired_dataset" / "hr_7t_in_3t"
hr_dst   = repo_root / "data" / "paired_dataset" / "hr_7t_in_3t_masked"
mask_root = repo_root / "data" / "paired_dataset" / "cow_segmentation_ens301_current"

errors = []
for case_src in sorted(hr_src.iterdir()):
    if not case_src.is_dir():
        continue
    case_id = case_src.name
    mask_path = mask_root / case_id / "cow_seg_final.nii.gz"

    if not mask_path.exists():
        print(f"[SKIP] {case_id}: no mask at {mask_path}", flush=True)
        errors.append(case_id)
        continue

    case_dst = hr_dst / case_id
    case_dst.mkdir(parents=True, exist_ok=True)

    mask_img = nib.load(str(mask_path))
    mask3d = (np.asarray(mask_img.dataobj, dtype=np.float32) > 0.5).astype(np.float32)

    for src_file in sorted(case_src.glob("*.nii.gz")):
        dst_file = case_dst / src_file.name
        hr_img  = nib.load(str(src_file))
        hr_data = np.asarray(hr_img.dataobj, dtype=np.float32)

        if hr_data.ndim == 4:
            mask_bc = mask3d[..., np.newaxis]
        else:
            mask_bc = mask3d

        masked = (hr_data * mask_bc).astype(np.float32)
        header = hr_img.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)
        nib.save(nib.Nifti1Image(masked, hr_img.affine, header), str(dst_file))

    print(f"[OK] {case_id}", flush=True)

if errors:
    print(f"\nWarning: {len(errors)} case(s) had missing masks: {errors}", file=sys.stderr)
    sys.exit(1)
EOF

echo
echo "Pipeline complete."
echo "  Crops:      data/temporal_registered_cow_crop"
echo "  Registered: data/registered_7T_in_3T_cow_crop"
echo "  Masks:      data/paired_dataset/cow_segmentation_ens301_current"
echo "  LR data:    data/paired_dataset/lr_3t"
echo "  HR masked:  data/paired_dataset/hr_7t_in_3t_masked"
