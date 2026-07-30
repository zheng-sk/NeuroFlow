#!/usr/bin/env bash
# Run geometry (segmentation + metrics) for missing LOOCV folds only.
# Sequential — one fold at a time to avoid GPU OOM.
#
# Usage:
#   bash run_geometry_only.sh                          # CBAM defaults
#   XVAL_EXPERIMENT_NAME=xval_original_unmasked \
#     EXP_PREFIX="" OUT_ROOT=output/inference_only_original \
#     bash run_geometry_only.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${ROOT}/.venv_neuroflow/bin/python
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python

XVAL_EXPERIMENT_NAME=${XVAL_EXPERIMENT_NAME:-xval_cbam_preups_unmasked}
EXP_PREFIX=${EXP_PREFIX-nf_preups_cbam_}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}
SEG_MODEL_DIR=${SEG_MODEL_DIR:-${ROOT}/models/topcow-claim-models}
SEG_GPU=${SEG_GPU:-0}

LOG_DIR=${ROOT}/output/xval_eval_logs
mkdir -p "${LOG_DIR}"

echo "[CONFIG] XVAL_EXPERIMENT_NAME = ${XVAL_EXPERIMENT_NAME}"
echo "[CONFIG] EXP_PREFIX           = '${EXP_PREFIX}'"
echo "[CONFIG] OUT_ROOT             = ${OUT_ROOT}"
echo "[CONFIG] SEG_GPU              = ${SEG_GPU}"

FOLDS=(
  fold_000_001_20240313
  fold_001_002_20240326
  fold_002_003_20240422
  fold_003_004_20240619
  fold_004_005_20241211
  fold_005_006_20241218
  fold_006_007_20250826
)

run_fold_geometry() {
  local task=$1
  local fold=$2
  local exp_name="${EXP_PREFIX}${task}_${fold}"
  local case_tag="case000"
  local exp_dir="${OUT_ROOT}/${exp_name}_${case_tag}"
  local meta_json="${exp_dir}/inference_metadata.json"
  local seg_tmp="${exp_dir}/_segmentation_tmp"
  local final_mask="${exp_dir}/cow_seg_final.nii.gz"

  if [[ ! -f "${meta_json}" ]]; then
    echo "[ERROR] Missing inference_metadata.json: ${meta_json}" >&2
    return 1
  fi

  echo "[INFO] Segmenting → ${exp_name} | GPU=${SEG_GPU}"
  rm -rf "${seg_tmp}"

  CUDA_VISIBLE_DEVICES="${SEG_GPU}" "${PYTHON_BIN}" \
    "${ROOT}/code/segmentation/segment_cow_patient_pipeline.py" \
    --patient-dir   "${exp_dir}" \
    --output-dir    "${seg_tmp}" \
    --model-dir     "${SEG_MODEL_DIR}" \
    --mag-name      "nifti/pred_mag.nii.gz" \
    --vx-name       "nifti/pred_u.nii.gz" \
    --vy-name       "nifti/pred_v.nii.gz" \
    --vz-name       "nifti/pred_w.nii.gz" \
    --venc          0.90 \
    --time-axis     -1 \
    --angio-mode    mag_only \
    --mag-projection-method     percentile \
    --mag-projection-percentile 100 \
    --ensemble-mode union \
    --classic-percentile        97.5 \
    --post-min-component-size   30 \
    --classic-sigmas            0.8,1.2,1.6,2.0

  local seg_out="${seg_tmp}/${exp_name}_${case_tag}/cow_seg_final.nii.gz"
  if [[ ! -f "${seg_out}" ]]; then
    echo "[ERROR] Segmentation output missing: ${seg_out}" >&2
    return 1
  fi
  cp "${seg_out}" "${final_mask}"

  echo "[INFO] Geometry metrics → ${exp_name}"
  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_geometry_metrics.py" \
    --pred-mask     "${final_mask}" \
    --metadata-json "${meta_json}" \
    --out-dir       "${exp_dir}/metrics/geometry"
}

FAILED=0
for task in r1_noise x2_noise; do
  for fold in "${FOLDS[@]}"; do
    exp_name="${EXP_PREFIX}${task}_${fold}"
    geo_csv="${OUT_ROOT}/${exp_name}_case000/metrics/geometry/geometry_metrics_compact.csv"
    if [[ -f "${geo_csv}" ]]; then
      echo "[SKIP] ${task} ${fold}"
      continue
    fi
    run_fold_geometry "${task}" "${fold}" \
      > "${LOG_DIR}/geometry_${task}_${fold}.log" 2>&1 \
      && echo "[OK]   ${task} ${fold}" \
      || { echo "[FAIL] ${task} ${fold}"; FAILED=$((FAILED+1)); }
  done
done

echo ""
echo "Done. Failed: ${FAILED}"
