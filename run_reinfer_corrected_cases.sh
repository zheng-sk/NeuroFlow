#!/usr/bin/env bash
# Re-run LOOCV inference for the 4 corrected cases using loo_masked_input models.
# Only runs inference + metrics for the affected folds (002, 005, 006, 007).
#
# Usage:
#   XVAL_EXPERIMENT_NAME=loo_masked_input bash run_reinfer_corrected_cases.sh
#
# Optional overrides:
#   GPU_0=0 GPU_1=1 GPU_2=2     (GPUs assigned round-robin, default 0,1,2)
#   OUT_ROOT=output/inference_only_masked

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

XVAL_EXPERIMENT_NAME=${XVAL_EXPERIMENT_NAME:-loo_masked_input}
EXP_PREFIX=${EXP_PREFIX:-""}
MODEL_VARIANT=${MODEL_VARIANT:-pre_upsample_attention}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only_masked}
APPLY_MASK_TO_LR=${APPLY_MASK_TO_LR:-1}

GPU_0=${GPU_0:-0}
GPU_1=${GPU_1:-1}
GPU_2=${GPU_2:-2}

LOG_DIR="${ROOT}/models/${XVAL_EXPERIMENT_NAME}/logs_reinfer"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# Only the 4 corrected folds
AFFECTED_FOLDS=(
  fold_001_002_20240326
  fold_004_005_20241211
  fold_005_006_20241218
  fold_006_007_20250826
)

GPUS=("${GPU_0}" "${GPU_1}" "${GPU_2}")

echo "[CONFIG] XVAL_EXPERIMENT_NAME = ${XVAL_EXPERIMENT_NAME}"
echo "[CONFIG] OUT_ROOT             = ${OUT_ROOT}"
echo "[CONFIG] APPLY_MASK_TO_LR    = ${APPLY_MASK_TO_LR}"
echo ""

run_fold_inference() {
  local task=$1    # r1_noise | x2_noise
  local fold=$2
  local gpu=$3
  local patch=$4
  local res=$5
  local sw_batch=${6:-2}

  local exp_name="${EXP_PREFIX}${task}_${fold}"
  local model_dir
  model_dir=$(ls -td "${ROOT}/models/${XVAL_EXPERIMENT_NAME}/${task}/${fold}/${exp_name}_"* 2>/dev/null | head -n1)

  if [[ -z "${model_dir}" ]]; then
    echo "[ERROR] Model dir not found for ${exp_name}" >&2
    return 1
  fi

  local model_path="${model_dir}/${exp_name}-best.pt"
  if [[ ! -f "${model_path}" ]]; then
    echo "[ERROR] Checkpoint not found: ${model_path}" >&2
    return 1
  fi

  local csv_dir
  if [[ "${task}" == "r1_noise" ]]; then
    csv_dir="${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1/${fold}"
  else
    csv_dir="${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2/${fold}"
  fi
  local case_csv="${csv_dir}/val.csv"

  if [[ ! -f "${case_csv}" ]]; then
    echo "[ERROR] Case CSV not found: ${case_csv}" >&2
    return 1
  fi

  local out_dir="${OUT_ROOT}/${exp_name}_case000"
  echo "[INFO] Inference → ${exp_name} | GPU=${gpu} | out=${out_dir}"

  local mask_flags=()
  if [[ "${APPLY_MASK_TO_LR}" == "1" ]]; then
    mask_flags=(--apply-mask-to-lr-inputs --apply-mask-to-lr-magnitude)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${ROOT}/code/inference/run_sr_inference_case.py" \
    --case-csv        "${case_csv}" \
    --case-index      0 \
    --model-path      "${model_path}" \
    --out-dir         "${out_dir}" \
    --patch-size      "${patch}" \
    --sw-batch-size   "${sw_batch}" \
    --res-increase    "${res}" \
    --predict-mag \
    --model-variant   "${MODEL_VARIANT}" \
    --seg-threshold   0.5 \
    --use-csv-frame-selection \
    "${mask_flags[@]}"
}

# ──────────────────────────────────────────────
# Phase 1: Inference (all 4 affected folds in parallel)
# ──────────────────────────────────────────────
echo "=========================================="
echo " Phase 1: Inference for affected folds"
echo "=========================================="

PIDS=()
IDX=0

for fold in "${AFFECTED_FOLDS[@]}"; do
  gpu=${GPUS[$((IDX % 3))]}
  run_fold_inference r1_noise "${fold}" "${gpu}" 48 1 2 \
    > "${LOG_DIR}/infer_r1_${fold}.log" 2>&1 &
  PIDS+=($!)
  echo "[LAUNCHED] r1_noise ${fold} on GPU ${gpu} (PID $!)"
  IDX=$((IDX+1))
done

for fold in "${AFFECTED_FOLDS[@]}"; do
  gpu=${GPUS[$((IDX % 3))]}
  run_fold_inference x2_noise "${fold}" "${gpu}" 24 2 2 \
    > "${LOG_DIR}/infer_x2_${fold}.log" 2>&1 &
  PIDS+=($!)
  echo "[LAUNCHED] x2_noise ${fold} on GPU ${gpu} (PID $!)"
  IDX=$((IDX+1))
done

echo ""
echo "Waiting for all inference jobs (${#PIDS[@]} total)..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[WARN] PID ${pid} failed" >&2
    FAILED=$((FAILED+1))
  fi
done

if [[ "${FAILED}" -gt 0 ]]; then
  echo "[ERROR] ${FAILED} inference job(s) failed. Check logs in ${LOG_DIR}." >&2
  exit 1
fi
echo "Inference complete."

echo ""
echo "=== Done ==="
echo "Predictions written to: ${OUT_ROOT}"
echo "To compute metrics, run: XVAL_EXPERIMENT_NAME=${XVAL_EXPERIMENT_NAME} OUT_ROOT=${OUT_ROOT} APPLY_MASK_TO_LR=${APPLY_MASK_TO_LR} bash run_xval_eval.sh"
