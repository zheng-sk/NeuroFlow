#!/usr/bin/env bash
# Run 7T-Den and 7T-SR predictions for all subjects using their LOO fold model.
#
# Usage:
#   GPU=0 bash run_predict_7t.sh
#
# Outputs:
#   output/7t_den/{subject}/   -- denoised 7T
#   output/7t_sr/{subject}/    -- super-resolved 7T

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-.venv_neuroflow/bin/python}
GPU=${GPU:-0}
MODEL_VARIANT=${MODEL_VARIANT:-pre_upsample_attention}

DEN_CSV=${ROOT}/data/paired_dataset/predict_7t_den/hr_7t_in_3t_masked.csv
SR_CSV=${ROOT}/data/paired_dataset/predict_7t_sr/hr_7t_in_3t_masked.csv
DEN_OUT=${ROOT}/output/7t_den
SR_OUT=${ROOT}/output/7t_sr

# subject | case-index | fold
CASES=(
  "001_20240313 0 fold_000_001_20240313"
  "002_20240326 1 fold_001_002_20240326"
  "003_20240422 2 fold_002_003_20240422"
  "004_20240619 3 fold_003_004_20240619"
  "005_20241211 4 fold_004_005_20241211"
  "006_20241218 5 fold_005_006_20241218"
  "007_20250826 6 fold_006_007_20250826"
)

run_inference() {
  local task=$1   # r1_noise | x2_noise
  local subject=$2
  local idx=$3
  local fold=$4
  local csv=$5
  local out_dir=$6
  local patch=$7
  local res=$8

  local exp_name="${task}_${fold}"
  local model_dir
  model_dir=$(ls -td "${ROOT}/models/loo_masked_input/${task}/${fold}/${exp_name}_"* 2>/dev/null | head -n1)

  if [[ -z "${model_dir}" ]]; then
    echo "[ERROR] Model not found for ${exp_name}" >&2
    return 1
  fi

  local model_path="${model_dir}/${exp_name}-best.pt"
  if [[ ! -f "${model_path}" ]]; then
    echo "[ERROR] Checkpoint not found: ${model_path}" >&2
    return 1
  fi

  echo "[INFO] ${task} | ${subject} | model=$(basename ${model_dir})"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${ROOT}/code/inference/run_sr_inference_case.py" \
    --case-csv       "${csv}" \
    --case-index     "${idx}" \
    --model-path     "${model_path}" \
    --model-variant  "${MODEL_VARIANT}" \
    --out-dir        "${out_dir}/${subject}" \
    --patch-size     "${patch}" \
    --sw-batch-size  2 \
    --res-increase   "${res}" \
    --predict-mag \
    --seg-threshold  0.5 \
    --apply-mask-to-lr-inputs \
    --apply-mask-to-lr-magnitude \
    --use-csv-frame-selection
}

mkdir -p "${DEN_OUT}" "${SR_OUT}"

for entry in "${CASES[@]}"; do
  read -r subject idx fold <<< "${entry}"

  echo "=========================================="
  echo " Subject: ${subject}  (case-index=${idx})"
  echo "=========================================="

  run_inference r1_noise "${subject}" "${idx}" "${fold}" "${DEN_CSV}" "${DEN_OUT}" 48 1
  run_inference x2_noise "${subject}" "${idx}" "${fold}" "${SR_CSV}"  "${SR_OUT}" 24 2

  echo ""
done

echo "=== Done ==="
echo "7T-Den → ${DEN_OUT}"
echo "7T-SR  → ${SR_OUT}"
