#!/usr/bin/env bash

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
GPU=${GPU:-2}
CASE_INDEX=${CASE_INDEX:-0}
# Fold names embed subject identifiers, so there is no published default.
# Set FOLD explicitly, or leave it empty to use the first discovered fold.
FOLD=${FOLD:-}
if [[ -z "${FOLD}" ]]; then
  _folds_dir=${R1_FOLDS_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1}
  for _d in "${_folds_dir}"/fold_*; do
    [[ -d "${_d}" ]] && { FOLD=$(basename "${_d}"); break; }
  done
fi
if [[ -z "${FOLD}" ]]; then
  echo "Error: set FOLD=<fold_name> (no fold_* directory found to default to)." >&2
  return 1 2>/dev/null || exit 1
fi

CASE_CSV_3T=${CASE_CSV_3T:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1/${FOLD}/val.csv}
CASE_CSV_053T=${CASE_CSV_053T:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2/${FOLD}/val.csv}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}

MODEL_VARIANT_OVERRIDE=${MODEL_VARIANT_OVERRIDE:-}
USE_SEG_HEAD=${USE_SEG_HEAD:-${CASCADE_USE_SEG_HEAD:-}}
SAVE_SEGMENTATION=${SAVE_SEGMENTATION:-1}
SEG_THRESHOLD=${SEG_THRESHOLD:-0.5}
FRAME_ARGS=${FRAME_ARGS:-}
USE_CSV_FRAME_SELECTION=${USE_CSV_FRAME_SELECTION:-0}

latest_model_dir() {
  ls -td "${ROOT}/models/$1"_* 2>/dev/null | head -n1
}

run_inference_case() {
  local exp_name=$1
  local case_csv=$2
  local patch=$3
  local res=$4
  local sw_batch=$5
  local masked=${6:-0}
  local model_variant=${7:-}

  local model_dir
  model_dir=$(latest_model_dir "${exp_name}")

  if [[ -z "${model_dir}" ]]; then
    echo "Model directory not found for ${exp_name}" >&2
    return 1
  fi

  local model_path=${model_dir}/${exp_name}-best.pt
  local out_dir=${OUT_ROOT}/${exp_name}_case$(printf "%03d" "${CASE_INDEX}")

  if [[ ! -f "${model_path}" ]]; then
    echo "Checkpoint not found: ${model_path}" >&2
    return 1
  fi

  if [[ ! -f "${case_csv}" ]]; then
    echo "Case CSV not found: ${case_csv}" >&2
    return 1
  fi

  local extra_args=(
    --case-csv "${case_csv}"
    --case-index "${CASE_INDEX}"
    --model-path "${model_path}"
    --out-dir "${out_dir}"
    --patch-size "${patch}"
    --sw-batch-size "${sw_batch}"
    --res-increase "${res}"
    --predict-mag
    --seg-threshold "${SEG_THRESHOLD}"
  )

  if [[ "${masked}" == "1" ]]; then
    extra_args+=(--apply-mask-to-lr-inputs)
  fi
  if [[ -n "${model_variant}" ]]; then
    extra_args+=(--model-variant "${model_variant}")
  elif [[ -n "${MODEL_VARIANT_OVERRIDE}" ]]; then
    extra_args+=(--model-variant "${MODEL_VARIANT_OVERRIDE}")
  fi
  if [[ -n "${USE_SEG_HEAD}" ]]; then
    if [[ "${USE_SEG_HEAD}" == "1" ]]; then
      extra_args+=(--use-seg-head)
    else
      extra_args+=(--no-use-seg-head)
    fi
  fi
  if [[ "${SAVE_SEGMENTATION}" == "0" ]]; then
    extra_args+=(--no-save-segmentation)
  fi
  if [[ "${USE_CSV_FRAME_SELECTION}" == "1" ]]; then
    extra_args+=(--use-csv-frame-selection)
  fi
  if [[ -n "${FRAME_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local frame_tokens=( ${FRAME_ARGS} )
    extra_args+=(--frame-index "${frame_tokens[@]}")
  fi

  echo "============================================================"
  echo "Experiment: ${exp_name}"
  echo "GPU:        ${GPU}"
  echo "Case CSV:   ${case_csv}"
  echo "Case index: ${CASE_INDEX}"
  echo "Masked LR:  ${masked}"
  echo "Model:      ${model_path}"
  echo "Output:     ${out_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${GPU}" python -m neuroflow.inference.run_sr_inference_case "${extra_args[@]}"
}

# Example usage:
# source scripts/run_sr_uq_reports.sh
#
# Single-stage 3T -> 7T inference:
# GPU=0 CASE_INDEX=0 run_inference_case nf_orig_3t7t_r1_masked "${CASE_CSV_3T}" 48 1 2 1 original
#
# Single-stage 0.53T -> 7T inference:
# GPU=1 CASE_INDEX=0 run_inference_case nf_orig_053t7t_x2_masked "${CASE_CSV_053T}" 24 2 2 1 original
#
# Cascade inference with learned seg mask (no reference inter-stage mask):
# GPU=0 CASE_INDEX=0 USE_SEG_HEAD=1 \
#   run_inference_case nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead "${CASE_CSV_053T}" 24 2 1 0 cascade_sr_dn_masked
