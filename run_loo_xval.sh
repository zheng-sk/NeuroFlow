#!/usr/bin/env bash

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
XVAL_EXPERIMENT_NAME=${XVAL_EXPERIMENT_NAME:-}
MODE=${MODE:-both}                      # both | denoise | sr
GPU_DENOISE=${GPU_DENOISE:-0}
GPU_SR=${GPU_SR:-${GPU_DENOISE}}
MODEL_VARIANT=${MODEL_VARIANT:-pre_upsample_attention}
NOISE_CSV=${NOISE_CSV:-${ROOT}/output/noise_pdf/nonvascular_noise_fit_summary.csv}

DENOISE_FOLDS_DIR=${DENOISE_FOLDS_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1}
SR_FOLDS_DIR=${SR_FOLDS_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2}

DENOISE_PATCH=${DENOISE_PATCH:-48}
DENOISE_RES=${DENOISE_RES:-1}
DENOISE_BATCH=${DENOISE_BATCH:-8}
DENOISE_VAL_SW_BATCH=${DENOISE_VAL_SW_BATCH:-2}

SR_PATCH=${SR_PATCH:-24}
SR_RES=${SR_RES:-2}
SR_BATCH=${SR_BATCH:-8}
SR_VAL_SW_BATCH=${SR_VAL_SW_BATCH:-2}

EPOCHS=${EPOCHS:-500}
SEED=${SEED:-42}
SKIP_EXISTING=${SKIP_EXISTING:-0}

usage() {
  cat <<EOF
Usage:
  XVAL_EXPERIMENT_NAME=<name> bash run_loo_xval.sh

Optional environment variables:
  MODE=both|denoise|sr
  GPU_DENOISE=0
  GPU_SR=1
  DENOISE_FOLDS_DIR=${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1
  SR_FOLDS_DIR=${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2
  NOISE_CSV=${ROOT}/output/noise_pdf/nonvascular_noise_fit_summary.csv
  SKIP_EXISTING=1
EOF
}

if [[ -z "${XVAL_EXPERIMENT_NAME}" ]]; then
  echo "Error: XVAL_EXPERIMENT_NAME is required." >&2
  usage >&2
  exit 1
fi

run_one_fold() {
  local gpu=$1
  local task_tag=$2
  local csv_dir=$3
  local patch=$4
  local res=$5
  local batch=$6
  local val_sw_batch=$7

  local fold_name
  fold_name=$(basename "${csv_dir}")

  if [[ ! -f "${csv_dir}/train.csv" || ! -f "${csv_dir}/val.csv" ]]; then
    echo "Skipping ${task_tag}/${fold_name}: train.csv or val.csv missing" >&2
    return 0
  fi

  local model_root_dir="${ROOT}/models/${XVAL_EXPERIMENT_NAME}/${task_tag}/${fold_name}"
  local run_name="nf_preups_cbam_${task_tag}_${fold_name}"
  local log_dir="${ROOT}/models/${XVAL_EXPERIMENT_NAME}/logs"
  local log_path="${log_dir}/${task_tag}_${fold_name}.log"
  local train_args=(
    --train-csv "${csv_dir}/train.csv"
    --val-csv "${csv_dir}/val.csv"
    --network-name "${run_name}"
    --model-variant "${MODEL_VARIANT}"
    --patch-size "${patch}"
    --res-increase "${res}"
    --batch-size "${batch}"
    --epochs "${EPOCHS}"
    --predict-mag
    --seed "${SEED}"
    --deterministic
    --overfit-patience 100
    --early-stopping-patience 80
    --cache-dataset
    --val-full-volume
    --val-sw-batch-size "${val_sw_batch}"
    --non-fluid-loss-weight 0.3
    --noise-aug-prob 0.8
    --noise-aug-fit-summary-csv "${NOISE_CSV}"
    --noise-aug-exaggerated-side-expand 1.8
    --noise-aug-exaggerated-edge-boost 0.30
    --noise-aug-exaggerated-edge-power 1.5
    --noise-aug-phase-scale 0.06
    --noise-aug-mag-scale 0.04
    --noise-aug-range-mult 1.5
    --noise-aug-level-min 1.0
    --noise-aug-level-max 5.0
    --noise-aug-masked-fraction 0.1
    --model-root-dir "${model_root_dir}"
  )

  mkdir -p "${model_root_dir}" "${log_dir}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && find "${model_root_dir}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "Skipping ${task_tag}/${fold_name}: existing run found under ${model_root_dir}"
    return 0
  fi

  echo
  echo "===================================================================="
  echo "Training ${task_tag} | fold=${fold_name} | gpu=${gpu}"
  echo "Output root: ${model_root_dir}"
  echo "Log: ${log_path}"
  echo "===================================================================="

  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${ROOT}/src/trainer_nifti.py" \
    "${train_args[@]}" \
    2>&1 | tee "${log_path}"
}

run_fold_series() {
  local gpu=$1
  local task_tag=$2
  local folds_dir=$3
  local patch=$4
  local res=$5
  local batch=$6
  local val_sw_batch=$7

  if [[ ! -d "${folds_dir}" ]]; then
    echo "Error: folds dir not found: ${folds_dir}" >&2
    exit 1
  fi

  local found=0
  local csv_dir
  for csv_dir in "${folds_dir}"/fold_*; do
    [[ -d "${csv_dir}" ]] || continue
    found=1
    run_one_fold "${gpu}" "${task_tag}" "${csv_dir}" "${patch}" "${res}" "${batch}" "${val_sw_batch}"
  done

  if [[ "${found}" == "0" ]]; then
    echo "Error: no fold_* directories found in ${folds_dir}" >&2
    exit 1
  fi
}

mkdir -p "${ROOT}/models/${XVAL_EXPERIMENT_NAME}"

case "${MODE}" in
  both)
    run_fold_series "${GPU_DENOISE}" "r1_noise" "${DENOISE_FOLDS_DIR}" "${DENOISE_PATCH}" "${DENOISE_RES}" "${DENOISE_BATCH}" "${DENOISE_VAL_SW_BATCH}"
    run_fold_series "${GPU_SR}" "x2_noise" "${SR_FOLDS_DIR}" "${SR_PATCH}" "${SR_RES}" "${SR_BATCH}" "${SR_VAL_SW_BATCH}"
    ;;
  denoise)
    run_fold_series "${GPU_DENOISE}" "r1_noise" "${DENOISE_FOLDS_DIR}" "${DENOISE_PATCH}" "${DENOISE_RES}" "${DENOISE_BATCH}" "${DENOISE_VAL_SW_BATCH}"
    ;;
  sr)
    run_fold_series "${GPU_SR}" "x2_noise" "${SR_FOLDS_DIR}" "${SR_PATCH}" "${SR_RES}" "${SR_BATCH}" "${SR_VAL_SW_BATCH}"
    ;;
  *)
    echo "Error: unsupported MODE=${MODE}. Use both, denoise, or sr." >&2
    exit 1
    ;;
esac
