#!/usr/bin/env bash

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODE=${MODE:-train}
GPU=${GPU:-0}
FOLD=${FOLD:-fold_000_001_20240313}

CSV_DIR=${CSV_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2/${FOLD}}
CASE_CSV=${CASE_CSV:-${CSV_DIR}/val.csv}
TRAIN_CSV=${TRAIN_CSV:-${CSV_DIR}/train.csv}
VAL_CSV=${VAL_CSV:-${CSV_DIR}/val.csv}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}

STAGE1_DIR=${STAGE1_DIR:-${ROOT}/models/nf_cbampreup_053t7t_x2_ps24_20260319-0855}
STAGE2_DIR=${STAGE2_DIR:-${ROOT}/models/nf_orig_3t7t_r1_masked_20260320-1002}
NOISE_CSV=${NOISE_CSV:-${ROOT}/output/noise_pdf/nonvascular_noise_fit_summary.csv}

STAGE1_CKPT=${STAGE1_CKPT-${STAGE1_DIR}/nf_cbampreup_053t7t_x2_ps24-best.pt}
STAGE2_CKPT=${STAGE2_CKPT-${STAGE2_DIR}/nf_orig_3t7t_r1_masked-best.pt}

NAME=${NAME:-nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24}
PATCH=${PATCH:-24}
RES=${RES:-2}
LOW_RESBLOCK=${LOW_RESBLOCK:-8}
HI_RESBLOCK=${HI_RESBLOCK:-4}

BATCH=${BATCH:-8}
VAL_BATCH=${VAL_BATCH:-1}
EPOCHS=${EPOCHS:-300}
LR=${LR:-1e-5}
NOISE_AUG_PROB=${NOISE_AUG_PROB:-0.0}
FREEZE_STAGE1=${FREEZE_STAGE1:-0}
FREEZE_STAGE2=${FREEZE_STAGE2:-0}
VAL_FULL_VOLUME=${VAL_FULL_VOLUME:-0}

CASCADE_USE_SEG_HEAD=${CASCADE_USE_SEG_HEAD:-0}
CASCADE_USE_SEG_MASK_AT_INFERENCE=${CASCADE_USE_SEG_MASK_AT_INFERENCE:-0}
SEG_LOSS_WEIGHT=${SEG_LOSS_WEIGHT:-0.0}
SEG_LOSS_DICE_WEIGHT=${SEG_LOSS_DICE_WEIGHT:-1.0}
SEG_LOSS_BCE_WEIGHT=${SEG_LOSS_BCE_WEIGHT:-1.0}
SEG_VESSELNESS_LOSS_WEIGHT=${SEG_VESSELNESS_LOSS_WEIGHT:-0.0}
SEG_VESSELNESS_LOSS_TYPE=${SEG_VESSELNESS_LOSS_TYPE:-mse}
ATTN_LOSS_WEIGHT=${ATTN_LOSS_WEIGHT:-0.0}
ATTN_LOSS_TYPE=${ATTN_LOSS_TYPE:-bce}
ATTN_SUPERVISION_TARGET=${ATTN_SUPERVISION_TARGET:-mask}
SEG_HEAD_BIAS_INIT=${SEG_HEAD_BIAS_INIT:--4.0}

MODEL_DIR=${MODEL_DIR:-}
MODEL_PATH=${MODEL_PATH:-}
RESTORE=${RESTORE:-0}
RESTORE_DIR=${RESTORE_DIR:-}
RESTORE_FILE=${RESTORE_FILE:-}
CASE_INDEX=${CASE_INDEX:-0}
SW_BATCH=${SW_BATCH:-${VAL_BATCH}}
OVERLAP=${OVERLAP:-0.25}
FRAME_ARGS=${FRAME_ARGS:-}
USE_CSV_FRAME_SELECTION=${USE_CSV_FRAME_SELECTION:-0}
APPLY_MASK_TO_LR_INPUTS=${APPLY_MASK_TO_LR_INPUTS:-0}
APPLY_MASK_TO_LR_MAGNITUDE=${APPLY_MASK_TO_LR_MAGNITUDE:-1}
USE_REFERENCE_MASK_FOR_CASCADE=${USE_REFERENCE_MASK_FOR_CASCADE:-1}
SAVE_SEGMENTATION=${SAVE_SEGMENTATION:-1}
SEG_THRESHOLD=${SEG_THRESHOLD:-0.5}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-pred}

latest_model_dir() {
  ls -td "${ROOT}/models/$1"_* 2>/dev/null | head -n1
}

bool_flag() {
  local value=$1
  local flag=$2
  if [[ "${value}" == "1" ]]; then
    printf '%s\n' "${flag}"
  fi
}

ensure_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

train_mode() {
  ensure_file "${TRAIN_CSV}" "train CSV"
  ensure_file "${VAL_CSV}" "val CSV"

  local val_args=()
  if [[ "${VAL_FULL_VOLUME}" == "1" ]]; then
    val_args+=(--val-full-volume --val-sw-batch-size "${VAL_BATCH}")
  else
    val_args+=(--val-samples-per-volume 8)
  fi

  local freeze_args=()
  if [[ "${FREEZE_STAGE1}" == "1" ]]; then
    freeze_args+=(--cascade-freeze-stage1)
  fi
  if [[ "${FREEZE_STAGE2}" == "1" ]]; then
    freeze_args+=(--cascade-freeze-stage2)
  fi

  local ckpt_args=()
  if [[ -n "${STAGE1_CKPT}" ]]; then
    ensure_file "${STAGE1_CKPT}" "stage-1 checkpoint"
    ckpt_args+=(--cascade-stage1-checkpoint "${STAGE1_CKPT}")
  fi
  if [[ -n "${STAGE2_CKPT}" ]]; then
    ensure_file "${STAGE2_CKPT}" "stage-2 checkpoint"
    ckpt_args+=(--cascade-stage2-checkpoint "${STAGE2_CKPT}")
  fi

  local restore_args=()
  if [[ "${RESTORE}" == "1" ]]; then
    if [[ -z "${RESTORE_DIR}" || -z "${RESTORE_FILE}" ]]; then
      echo "RESTORE=1 requires RESTORE_DIR and RESTORE_FILE" >&2
      exit 1
    fi
    ensure_file "${RESTORE_DIR}/${RESTORE_FILE}" "restore checkpoint"
    restore_args+=(--restore --restore-dir "${RESTORE_DIR}" --restore-file "${RESTORE_FILE}")
  fi

  local seg_args=()
  if [[ "${CASCADE_USE_SEG_HEAD}" == "1" ]]; then
    seg_args+=(--cascade-use-seg-head)
  fi
  if [[ "${CASCADE_USE_SEG_MASK_AT_INFERENCE}" == "1" ]]; then
    seg_args+=(--cascade-use-seg-mask-at-inference)
  fi
  if [[ "${SEG_LOSS_WEIGHT}" != "0" && "${SEG_LOSS_WEIGHT}" != "0.0" ]]; then
    seg_args+=(
      --seg-loss-weight "${SEG_LOSS_WEIGHT}"
      --seg-loss-dice-weight "${SEG_LOSS_DICE_WEIGHT}"
      --seg-loss-bce-weight "${SEG_LOSS_BCE_WEIGHT}"
    )
  fi
  if [[ "${SEG_VESSELNESS_LOSS_WEIGHT}" != "0" && "${SEG_VESSELNESS_LOSS_WEIGHT}" != "0.0" ]]; then
    seg_args+=(
      --seg-vesselness-loss-weight "${SEG_VESSELNESS_LOSS_WEIGHT}"
      --seg-vesselness-loss-type "${SEG_VESSELNESS_LOSS_TYPE}"
    )
  fi
  if [[ "${ATTN_LOSS_WEIGHT}" != "0" && "${ATTN_LOSS_WEIGHT}" != "0.0" ]]; then
    seg_args+=(
      --attn-loss-weight "${ATTN_LOSS_WEIGHT}"
      --attn-loss-type "${ATTN_LOSS_TYPE}"
      --attn-supervision-target "${ATTN_SUPERVISION_TARGET}"
    )
  fi

  echo "============================================================"
  echo "Mode:       train"
  echo "Experiment: ${NAME}"
  echo "GPU:        ${GPU}"
  echo "Train CSV:  ${TRAIN_CSV}"
  echo "Val CSV:    ${VAL_CSV}"
  echo "Stage1 ckpt:${STAGE1_CKPT}"
  echo "Stage2 ckpt:${STAGE2_CKPT}"
  if [[ "${RESTORE}" == "1" ]]; then
    echo "Restore:    ${RESTORE_DIR}/${RESTORE_FILE}"
  fi
  echo "Seg head:   ${CASCADE_USE_SEG_HEAD}"
  echo "Seg mask:   ${CASCADE_USE_SEG_MASK_AT_INFERENCE}"
  echo "Seg loss:   ${SEG_LOSS_WEIGHT}"
  echo "Seg dice:   ${SEG_LOSS_DICE_WEIGHT}"
  echo "Seg bce:    ${SEG_LOSS_BCE_WEIGHT}"
  echo "Seg vess:   ${SEG_VESSELNESS_LOSS_WEIGHT} (${SEG_VESSELNESS_LOSS_TYPE})"
  echo "Attn loss:  ${ATTN_LOSS_WEIGHT} (${ATTN_LOSS_TYPE}, target=${ATTN_SUPERVISION_TARGET})"
  echo "Seg bias:   ${SEG_HEAD_BIAS_INIT}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/src/trainer_nifti.py" \
    --train-csv "${TRAIN_CSV}" \
    --val-csv "${VAL_CSV}" \
    --network-name "${NAME}" \
    --model-variant cascade_sr_dn_masked \
    --patch-size "${PATCH}" \
    --res-increase "${RES}" \
    --batch-size "${BATCH}" \
    --epochs "${EPOCHS}" \
    --predict-mag \
    --seed 42 \
    --deterministic \
    --initial-learning-rate "${LR}" \
    --overfit-patience 100 \
    --early-stopping-patience 60 \
    --cache-dataset \
    --non-fluid-loss-weight 0.3 \
    --noise-aug-prob "${NOISE_AUG_PROB}" \
    --noise-aug-fit-summary-csv "${NOISE_CSV}" \
    --noise-aug-exaggerated-side-expand 1.8 \
    --noise-aug-exaggerated-edge-boost 0.30 \
    --noise-aug-exaggerated-edge-power 1.5 \
    --noise-aug-phase-scale 0.06 \
    --noise-aug-mag-scale 0.04 \
    --noise-aug-range-mult 1.5 \
    --noise-aug-level-min 1.0 \
    --noise-aug-level-max 5.0 \
    --noise-aug-masked-fraction 0.1 \
    --cascade-seg-head-bias-init "${SEG_HEAD_BIAS_INIT}" \
    "${restore_args[@]}" \
    "${ckpt_args[@]}" \
    "${freeze_args[@]}" \
    "${seg_args[@]}" \
    "${val_args[@]}"
}

infer_mode() {
  ensure_file "${CASE_CSV}" "case CSV"

  if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -z "${MODEL_DIR}" ]]; then
      MODEL_DIR=$(latest_model_dir "${NAME}")
    fi
    if [[ -z "${MODEL_DIR}" ]]; then
      echo "Model directory not found for ${NAME}" >&2
      exit 1
    fi
    MODEL_PATH="${MODEL_DIR}/${NAME}-best.pt"
  fi
  ensure_file "${MODEL_PATH}" "model checkpoint"

  local out_dir="${OUT_ROOT}/${NAME}_case$(printf "%03d" "${CASE_INDEX}")"
  local infer_args=(
    --case-csv "${CASE_CSV}"
    --case-index "${CASE_INDEX}"
    --model-path "${MODEL_PATH}"
    --model-variant cascade_sr_dn_masked
    --out-dir "${out_dir}"
    --output-prefix "${OUTPUT_PREFIX}"
    --patch-size "${PATCH}"
    --sw-batch-size "${SW_BATCH}"
    --overlap "${OVERLAP}"
    --res-increase "${RES}"
    --low-resblock "${LOW_RESBLOCK}"
    --hi-resblock "${HI_RESBLOCK}"
    --predict-mag
    --seg-threshold "${SEG_THRESHOLD}"
  )

  if [[ "${CASCADE_USE_SEG_HEAD}" == "1" ]]; then
    infer_args+=(--cascade-use-seg-head)
  fi
  if [[ "${CASCADE_USE_SEG_MASK_AT_INFERENCE}" == "1" ]]; then
    infer_args+=(--cascade-use-seg-mask-at-inference)
  fi
  if [[ "${APPLY_MASK_TO_LR_INPUTS}" == "1" ]]; then
    infer_args+=(--apply-mask-to-lr-inputs)
  fi
  if [[ "${APPLY_MASK_TO_LR_MAGNITUDE}" == "0" ]]; then
    infer_args+=(--no-apply-mask-to-lr-magnitude)
  fi
  if [[ "${USE_REFERENCE_MASK_FOR_CASCADE}" == "0" ]]; then
    infer_args+=(--no-use-reference-mask-for-cascade)
  fi
  if [[ "${SAVE_SEGMENTATION}" == "0" ]]; then
    infer_args+=(--no-save-segmentation)
  fi
  if [[ "${USE_CSV_FRAME_SELECTION}" == "1" ]]; then
    infer_args+=(--use-csv-frame-selection)
  fi
  if [[ -n "${FRAME_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local frame_tokens=( ${FRAME_ARGS} )
    infer_args+=(--frame-index "${frame_tokens[@]}")
  fi

  echo "============================================================"
  echo "Mode:          infer"
  echo "Experiment:    ${NAME}"
  echo "GPU:           ${GPU}"
  echo "Case CSV:      ${CASE_CSV}"
  echo "Case index:    ${CASE_INDEX}"
  echo "Model:         ${MODEL_PATH}"
  echo "Output:        ${out_dir}"
  echo "Ref. mask:     ${USE_REFERENCE_MASK_FOR_CASCADE}"
  echo "Seg head:      ${CASCADE_USE_SEG_HEAD}"
  echo "Seg mask:      ${CASCADE_USE_SEG_MASK_AT_INFERENCE}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/code/inference/run_sr_inference_case.py" "${infer_args[@]}"
}

infer_stage1mask_mode() {
  ensure_file "${CASE_CSV}" "case CSV"

  if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -z "${MODEL_DIR}" ]]; then
      MODEL_DIR=$(latest_model_dir "${NAME}")
    fi
    if [[ -z "${MODEL_DIR}" ]]; then
      echo "Model directory not found for ${NAME}" >&2
      exit 1
    fi
    MODEL_PATH="${MODEL_DIR}/${NAME}-best.pt"
  fi
  ensure_file "${MODEL_PATH}" "model checkpoint"

  local out_dir="${OUT_ROOT}/${NAME}_case$(printf "%03d" "${CASE_INDEX}")"
  local infer_args=(
    --case-csv "${CASE_CSV}"
    --case-index "${CASE_INDEX}"
    --model-path "${MODEL_PATH}"
    --out-dir "${out_dir}"
    --seg-model-dir "${SEG_MODEL_DIR}"
    --stage1-patch-size "${PATCH}"
    --stage1-sw-batch-size "${SW_BATCH}"
    --stage2-patch-size "$((PATCH * RES))"
    --stage2-sw-batch-size "${SW_BATCH}"
    --overlap "${OVERLAP}"
    --res-increase "${RES}"
    --low-resblock "${LOW_RESBLOCK}"
    --hi-resblock "${HI_RESBLOCK}"
    --predict-mag
    --gpu "${GPU}"
  )

  if [[ "${USE_CSV_FRAME_SELECTION}" == "1" ]]; then
    infer_args+=(--use-csv-frame-selection)
  fi
  if [[ -n "${FRAME_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local frame_tokens=( ${FRAME_ARGS} )
    infer_args+=(--frame-index "${frame_tokens[@]}")
  fi

  echo "============================================================"
  echo "Mode:          infer_stage1mask"
  echo "Experiment:    ${NAME}"
  echo "GPU:           ${GPU}"
  echo "Case CSV:      ${CASE_CSV}"
  echo "Case index:    ${CASE_INDEX}"
  echo "Model:         ${MODEL_PATH}"
  echo "Output:        ${out_dir}"
  echo "Stage1 patch:  ${PATCH}"
  echo "Stage2 patch:  $((PATCH * RES))"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python "${ROOT}/code/inference/run_cascade_stage1mask_inference_case.py" "${infer_args[@]}"
}

case "${MODE}" in
  train)
    train_mode
    ;;
  infer)
    infer_mode
    ;;
  infer_stage1mask)
    infer_stage1mask_mode
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use MODE=train, MODE=infer, or MODE=infer_stage1mask." >&2
    exit 1
    ;;
esac

# Examples:
# MODE=train GPU=0 CASCADE_USE_SEG_HEAD=1 CASCADE_USE_SEG_MASK_AT_INFERENCE=1 SEG_LOSS_WEIGHT=0.2 ./run_cascade_e2e.sh
# MODE=infer GPU=0 NAME=nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead CASE_INDEX=0 CASCADE_USE_SEG_HEAD=1 CASCADE_USE_SEG_MASK_AT_INFERENCE=1 USE_REFERENCE_MASK_FOR_CASCADE=0 ./run_cascade_e2e.sh
# MODE=infer_stage1mask GPU=0 NAME=nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead CASE_INDEX=0 ./run_cascade_e2e.sh
