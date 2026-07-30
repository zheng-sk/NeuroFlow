#!/usr/bin/env bash
# Re-run Dataset301_CoW3TMagProj segmentation for the 4 corrected crop cases.
# Uses temporal max-projection of 3T magnitude as nnUNet input.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash run_resegment_corrected_cases.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU=${CUDA_VISIBLE_DEVICES:-0}

WORKSPACE="${ROOT}/segmentation_nnunet_workspace"
VENV="${WORKSPACE}/.venv_nnunet"
CROP_DIR="${ROOT}/data/temporal_registered_cow_crop_corrected"
SEG_OUT="${ROOT}/data/paired_dataset/cow_segmentation_ens301_current"

ENSEMBLE_DIR="${WORKSPACE}/nnUNet_results/Dataset301_CoW3TMagProj/ensembles/ensemble___nnUNetTrainerSkeletonRecall__nnUNetResEncUNetMPlans__2d___nnUNetTrainerSkeletonRecall__nnUNetResEncUNetMPlans__3d_fullres___0_1_2_3_4"

TMP_INPUT="${ROOT}/data/tmp/seg301_input"
TMP_PRED_2D="${ROOT}/data/tmp/seg301_pred_2d"
TMP_PRED_3D="${ROOT}/data/tmp/seg301_pred_3d"
TMP_ENSEMBLE="${ROOT}/data/tmp/seg301_ensemble"
TMP_POSTPROC="${ROOT}/data/tmp/seg301_postproc"

CASES=("002_20240326" "005_20241211" "006_20241218" "007_20250826")

# Activate workspace venv + set nnUNet env vars once, used by all steps.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WORKSPACE}/env_nnunet.sh"

# ---------------------------------------------------------------------------
# Step 1: Build nnUNet input folder (temporal max projection of 3T magnitude)
# ---------------------------------------------------------------------------
echo "=== Step 1: Prepare nnUNet input (temporal max projection) ==="
mkdir -p "${TMP_INPUT}"

python - \
  "${CROP_DIR}" \
  "${TMP_INPUT}" \
  "${CASES[@]}" \
<<'PYEOF'
import sys
import nibabel as nib
import numpy as np
from pathlib import Path

crop_dir = Path(sys.argv[1])
tmp_input = Path(sys.argv[2])
case_list = sys.argv[3:]

for case_id in case_list:
    mag_path = crop_dir / f"{case_id}_3T" / "input_mag_raw.nii.gz"
    out_path = tmp_input / f"{case_id}_0000.nii.gz"

    img = nib.load(str(mag_path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    proj = np.max(data, axis=3) if data.ndim == 4 else data

    header = img.header.copy()
    header.set_data_shape(proj.shape)
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(proj, img.affine, header), str(out_path))
    print(f"  [OK] {case_id}: {data.shape} -> projection {proj.shape}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: nnUNet inference
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: nnUNet inference ==="
mkdir -p "${TMP_PRED_2D}" "${TMP_PRED_3D}" "${TMP_ENSEMBLE}" "${TMP_POSTPROC}"

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_predict \
  -d Dataset301_CoW3TMagProj \
  -i "${TMP_INPUT}" -o "${TMP_PRED_2D}" \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainerSkeletonRecall -c 2d -p nnUNetResEncUNetMPlans \
  --save_probabilities

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_predict \
  -d Dataset301_CoW3TMagProj \
  -i "${TMP_INPUT}" -o "${TMP_PRED_3D}" \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainerSkeletonRecall -c 3d_fullres -p nnUNetResEncUNetMPlans \
  --save_probabilities

# ---------------------------------------------------------------------------
# Step 3: Ensemble predictions
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Ensemble ==="
nnUNetv2_ensemble \
  -i "${TMP_PRED_2D}" "${TMP_PRED_3D}" \
  -o "${TMP_ENSEMBLE}" \
  -np 8

# ---------------------------------------------------------------------------
# Step 4: Apply postprocessing
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Postprocessing ==="
nnUNetv2_apply_postprocessing \
  -i "${TMP_ENSEMBLE}" \
  -o "${TMP_POSTPROC}" \
  -pp_pkl_file "${ENSEMBLE_DIR}/postprocessing.pkl" \
  -np 8 \
  -plans_json "${ENSEMBLE_DIR}/plans.json"

# ---------------------------------------------------------------------------
# Step 5: Copy final masks to paired_dataset
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Copy masks to cow_segmentation_ens301_current ==="

for case_id in "${CASES[@]}"; do
  src="${TMP_POSTPROC}/${case_id}.nii.gz"
  dst_dir="${SEG_OUT}/${case_id}"
  dst="${dst_dir}/cow_seg_final.nii.gz"
  mkdir -p "${dst_dir}"
  cp "${src}" "${dst}"
  echo "  [OK] ${case_id} -> ${dst}"
done

echo ""
echo "Segmentation complete for: ${CASES[*]}"
echo "Masks saved to: ${SEG_OUT}"

# ---------------------------------------------------------------------------
# Step 6: Update lr_3t from corrected 3T crops
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 6: Update lr_3t paired dataset ==="
LR_OUT="${ROOT}/data/paired_dataset/lr_3t"

for case_id in "${CASES[@]}"; do
  src_dir="${CROP_DIR}/${case_id}_3T"
  dst_dir="${LR_OUT}/${case_id}"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/input_mag_raw.nii.gz" "${dst_dir}/input_mag_raw.nii.gz"
  cp "${src_dir}/Vx.nii.gz"            "${dst_dir}/Vx.nii.gz"
  cp "${src_dir}/Vy.nii.gz"            "${dst_dir}/Vy.nii.gz"
  cp "${src_dir}/Vz.nii.gz"            "${dst_dir}/Vz.nii.gz"
  echo "  [OK] lr_3t/${case_id}"
done

# ---------------------------------------------------------------------------
# Step 6b: Regenerate lr_3tx05 (0.5x downsample of lr_3t, used by x2 SR model)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 6b: Regenerate lr_3tx05 ==="
LR_X05_OUT="${ROOT}/data/paired_dataset/lr_3tx05"

for case_id in "${CASES[@]}"; do
  python "${ROOT}/code/preprocessing/downsample_nifti_tree.py" \
    --input-root  "${LR_OUT}/${case_id}" \
    --output-root "${LR_X05_OUT}/${case_id}" \
    --scale 0.5
  echo "  [OK] lr_3tx05/${case_id}"
done

# ---------------------------------------------------------------------------
# Step 7: Update hr_7t_in_3t from corrected inter-scan registrations
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 7: Update hr_7t_in_3t paired dataset ==="
HR_OUT="${ROOT}/data/paired_dataset/hr_7t_in_3t"
REG_DIR="${ROOT}/data/registered_7T_in_3T_cow_crop"

for case_id in "${CASES[@]}"; do
  src_dir="${REG_DIR}/${case_id}"
  dst_dir="${HR_OUT}/${case_id}"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/mag_7T_in_3T.nii.gz"    "${dst_dir}/input_mag_raw.nii.gz"
  cp "${src_dir}/phaseX_7T_in_3T.nii.gz" "${dst_dir}/Vx.nii.gz"
  cp "${src_dir}/phaseY_7T_in_3T.nii.gz" "${dst_dir}/Vy.nii.gz"
  cp "${src_dir}/phaseZ_7T_in_3T.nii.gz" "${dst_dir}/Vz.nii.gz"
  echo "  [OK] hr_7t_in_3t/${case_id}"
done

# ---------------------------------------------------------------------------
# Step 8: Re-apply CoW masks to create hr_7t_in_3t_masked
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 8: Apply masks to hr_7t_in_3t_masked ==="

python - \
  "${HR_OUT}" \
  "${ROOT}/data/paired_dataset/hr_7t_in_3t_masked" \
  "${SEG_OUT}" \
  "${CASES[@]}" \
<<'PYEOF'
import sys
import nibabel as nib
import numpy as np
from pathlib import Path

hr_src    = Path(sys.argv[1])
hr_dst    = Path(sys.argv[2])
mask_root = Path(sys.argv[3])
case_list = sys.argv[4:]

errors = []
for case_id in case_list:
    mask_path = mask_root / case_id / "cow_seg_final.nii.gz"
    if not mask_path.exists():
        print(f"[SKIP] {case_id}: no mask at {mask_path}", flush=True)
        errors.append(case_id)
        continue

    case_dst = hr_dst / case_id
    case_dst.mkdir(parents=True, exist_ok=True)

    mask_img = nib.load(str(mask_path))
    mask3d = (np.asarray(mask_img.dataobj, dtype=np.float32) > 0.5).astype(np.float32)

    for src_file in sorted((hr_src / case_id).glob("*.nii.gz")):
        dst_file = case_dst / src_file.name
        hr_img  = nib.load(str(src_file))
        hr_data = np.asarray(hr_img.dataobj, dtype=np.float32)

        mask_bc = mask3d[..., np.newaxis] if hr_data.ndim == 4 else mask3d
        masked = (hr_data * mask_bc).astype(np.float32)

        header = hr_img.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)
        nib.save(nib.Nifti1Image(masked, hr_img.affine, header), str(dst_file))

    print(f"[OK] {case_id}", flush=True)

if errors:
    import sys as _sys
    print(f"\nWarning: {len(errors)} case(s) had missing masks: {errors}", file=_sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "=== All steps complete ==="
echo "  LR data:      data/paired_dataset/lr_3t           (updated: ${CASES[*]})"
echo "  HR unmasked:  data/paired_dataset/hr_7t_in_3t     (updated: ${CASES[*]})"
echo "  HR masked:    data/paired_dataset/hr_7t_in_3t_masked (updated: ${CASES[*]})"
echo "  Masks:        data/paired_dataset/cow_segmentation_ens301_current"
