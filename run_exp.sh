#!/usr/bin/env bash

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
GPU=${GPU:-0}
CASE_INDEX=${CASE_INDEX:-0}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}
NOISE_CSV=${NOISE_CSV:-${ROOT}/output/noise_pdf/nonvascular_noise_fit_summary.csv}

# Update these to the train/val split you actually want to use.
# Important: masked-input experiments require train.csv and val.csv with a
# non-empty `mask` column.
DNS_CSV_DIR=${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1/fold_000_001_20240313
SR_CSV_DIR=${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2/fold_000_001_20240313

latest_model_dir() {
  ls -td "${ROOT}/models/$1"_* 2>/dev/null | head -n1
}


run_exp_masked () {
  local GPU=$1
  local NAME=$2
  local VARIANT=$3
  local CSV_DIR=$4
  local PATCH=$5
  local RES=$6
  local BATCH=$7
  local VAL_SW_BATCH=$8

  CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT}/src/trainer_nifti.py \
    --train-csv ${CSV_DIR}/train.csv \
    --val-csv ${CSV_DIR}/val.csv \
    --network-name ${NAME} \
    --model-variant ${VARIANT} \
    --patch-size ${PATCH} \
    --res-increase ${RES} \
    --batch-size ${BATCH} \
    --epochs 500 \
    --predict-mag \
    --seed 42 \
    --deterministic \
    --overfit-patience 100 \
    --early-stopping-patience 80 \
    --cache-dataset \
    --val-full-volume \
    --val-sw-batch-size ${VAL_SW_BATCH} \
    --non-fluid-loss-weight 0.3 \
    --apply-mask-to-lr-inputs \
    --apply-mask-to-lr-magnitude
}


run_exp_unmasked_noise () {
  local GPU=$1
  local NAME=$2
  local VARIANT=$3
  local CSV_DIR=$4
  local PATCH=$5
  local RES=$6
  local BATCH=$7
  local VAL_SW_BATCH=$8

  CUDA_VISIBLE_DEVICES=${GPU} python ${ROOT}/src/trainer_nifti.py \
    --train-csv ${CSV_DIR}/train.csv \
    --val-csv ${CSV_DIR}/val.csv \
    --network-name ${NAME} \
    --model-variant ${VARIANT} \
    --patch-size ${PATCH} \
    --res-increase ${RES} \
    --batch-size ${BATCH} \
    --epochs 500 \
    --predict-mag \
    --seed 42 \
    --deterministic \
    --overfit-patience 100 \
    --early-stopping-patience 80 \
    --cache-dataset \
    --val-full-volume \
    --val-sw-batch-size ${VAL_SW_BATCH} \
    --non-fluid-loss-weight 0.3 \
    --noise-aug-prob 0.8 \
    --noise-aug-fit-summary-csv ${NOISE_CSV} \
    --noise-aug-exaggerated-side-expand 1.8 \
    --noise-aug-exaggerated-edge-boost 0.30 \
    --noise-aug-exaggerated-edge-power 1.5 \
    --noise-aug-phase-scale 0.06 \
    --noise-aug-mag-scale 0.04 \
    --noise-aug-range-mult 1.5 \
    --noise-aug-level-min 1.0 \
    --noise-aug-level-max 5.0 \
    --noise-aug-masked-fraction 0.1
}

run_inference_case () {
  local EXP_NAME=$1
  local CASE_CSV=$2
  local PATCH=$3
  local RES=$4
  local SW_BATCH=$5
  local MASKED=${6:-0}
  local MODEL_VARIANT=${7:-}

  local MODEL_DIR
  MODEL_DIR=$(latest_model_dir "${EXP_NAME}")

  if [[ -z "${MODEL_DIR}" ]]; then
    echo "Model directory not found for ${EXP_NAME}" >&2
    return 1
  fi

  local MODEL_PATH=${MODEL_DIR}/${EXP_NAME}-best.pt
  local OUT_DIR=${OUT_ROOT}/${EXP_NAME}_case$(printf "%03d" "${CASE_INDEX}")

  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Checkpoint not found: ${MODEL_PATH}" >&2
    return 1
  fi
  if [[ ! -f "${CASE_CSV}" ]]; then
    echo "Case CSV not found: ${CASE_CSV}" >&2
    return 1
  fi

  local EXTRA_ARGS=(--predict-mag)
  if [[ "${MASKED}" == "1" ]]; then
    EXTRA_ARGS+=(--apply-mask-to-lr-inputs)
  fi
  if [[ -n "${MODEL_VARIANT}" ]]; then
    EXTRA_ARGS+=(--model-variant "${MODEL_VARIANT}")
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/code/inference/run_sr_inference_case.py" \
    --case-csv "${CASE_CSV}" \
    --case-index "${CASE_INDEX}" \
    --model-path "${MODEL_PATH}" \
    --out-dir "${OUT_DIR}" \
    --patch-size "${PATCH}" \
    --sw-batch-size "${SW_BATCH}" \
    --res-increase "${RES}" \
    "${EXTRA_ARGS[@]}"
}


# ---------------------------------------------------------------------------
# Masked-input experiments without noise augmentation
# ---------------------------------------------------------------------------

# Denoising (res_increase=1)
# run_exp_masked 1 nf_orig_3t7t_r1_masked original ${DNS_CSV_DIR} 48 1 8 2
# run_exp_masked 2 nf_preups_cbam_3t7t_r1_masked pre_upsample_attention ${DNS_CSV_DIR} 48 1 8 2

# Super-resolution (res_increase=2)
# run_exp_masked 0 nf_orig_053t7t_x2_masked original ${SR_CSV_DIR} 24 2 8 2
# run_exp_masked 1 nf_preups_cbam_053t7t_x2_masked pre_upsample_attention ${SR_CSV_DIR} 24 2 8 2


# ---------------------------------------------------------------------------
# Unmasked-input experiments with noise augmentation
# ---------------------------------------------------------------------------

# Denoising (res_increase=1)
# run_exp_unmasked_noise 0 nf_orig_3t7t_r1_noise original ${DNS_CSV_DIR} 48 1 8 2
# run_exp_unmasked_noise 1 nf_preups_cbam_3t7t_r1_noise pre_upsample_attention ${DNS_CSV_DIR} 48 1 8 2

# Super-resolution (res_increase=2)
# run_exp_unmasked_noise 0 nf_orig_053t7t_x2_noise original ${SR_CSV_DIR} 24 2 8 2
# run_exp_unmasked_noise 1 nf_preups_cbam_053t7t_x2_noise pre_upsample_attention ${SR_CSV_DIR} 24 2 8 2


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

# Single-stage denoising:
# GPU=0 CASE_INDEX=0 run_inference_case nf_orig_3t7t_r1_masked "${DNS_CSV_DIR}/val.csv" 48 1 2 1 original
#
# Single-stage SR:
# GPU=1 CASE_INDEX=0 run_inference_case nf_preups_cbam_053t7t_x2_masked "${SR_CSV_DIR}/val.csv" 24 2 2 1 pre_upsample_attention
