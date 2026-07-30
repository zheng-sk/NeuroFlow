#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-${SCRIPT_DIR}}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}
SEG_MODEL_DIR=${SEG_MODEL_DIR:-${ROOT}/models/topcow-claim-models}
SEG_ANGIO_MODE=${SEG_ANGIO_MODE:-mag_only}
SEG_MAG_PROJECTION_METHOD=${SEG_MAG_PROJECTION_METHOD:-percentile}
SEG_MAG_PROJECTION_PERCENTILE=${SEG_MAG_PROJECTION_PERCENTILE:-100}
SEG_ENSEMBLE_MODE=${SEG_ENSEMBLE_MODE:-union}
SEG_CLASSIC_PERCENTILE=${SEG_CLASSIC_PERCENTILE:-97.5}
SEG_POST_MIN_COMPONENT_SIZE=${SEG_POST_MIN_COMPONENT_SIZE:-30}
SEG_CLASSIC_SIGMAS=${SEG_CLASSIC_SIGMAS:-0.8,1.2,1.6,2.0}
SEG_SAVE_INTERMEDIATES=${SEG_SAVE_INTERMEDIATES:-1}
SEG_GPU=${SEG_GPU:-}
NON_FLUID_WEIGHT=${NON_FLUID_WEIGHT:-0.3}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv_neuroflow/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

calc_velocity_metrics() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local PAYLOAD=${EXP_DIR}/analysis_payload.npz
  local METRICS_DIR=${EXP_DIR}/metrics/velocity

  if [[ ! -f "${PAYLOAD}" ]]; then
    echo "Missing payload: ${PAYLOAD}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_velocity_metrics.py" \
    --payload-npz "${PAYLOAD}" \
    --out-dir "${METRICS_DIR}" \
    --non-fluid-loss-weight "${NON_FLUID_WEIGHT}"
}


calc_background_metrics() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local PAYLOAD=${EXP_DIR}/analysis_payload.npz
  local METRICS_DIR=${EXP_DIR}/metrics/background

  if [[ ! -f "${PAYLOAD}" ]]; then
    echo "Missing payload: ${PAYLOAD}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_background_metrics.py" \
    --payload-npz "${PAYLOAD}" \
    --out-dir "${METRICS_DIR}"
}


segment_prediction_cow() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local META_JSON=${EXP_DIR}/inference_metadata.json

  if [[ ! -f "${META_JSON}" ]]; then
    echo "Missing inference metadata: ${META_JSON}" >&2
    return 1
  fi

  local SEG_TMP_ROOT=${EXP_DIR}/_segmentation_tmp
  local CASE_OUT_DIR=${SEG_TMP_ROOT}/${EXP_NAME}_${CASE_TAG}
  local FINAL_MASK=${EXP_DIR}/cow_seg_final.nii.gz

  rm -rf "${SEG_TMP_ROOT}"

  local CMD_PREFIX=()
  if [[ -n "${SEG_GPU}" ]]; then
    CMD_PREFIX=(env CUDA_VISIBLE_DEVICES="${SEG_GPU}")
  fi

  "${CMD_PREFIX[@]}" "${PYTHON_BIN}" "${ROOT}/code/segmentation/segment_cow_patient_pipeline.py" \
    --patient-dir "${EXP_DIR}" \
    --output-dir "${SEG_TMP_ROOT}" \
    --model-dir "${SEG_MODEL_DIR}" \
    --mag-name "nifti/pred_mag.nii.gz" \
    --vx-name "nifti/pred_u.nii.gz" \
    --vy-name "nifti/pred_v.nii.gz" \
    --vz-name "nifti/pred_w.nii.gz" \
    --venc 0.90 \
    --time-axis -1 \
    --angio-mode "${SEG_ANGIO_MODE}" \
    --mag-projection-method "${SEG_MAG_PROJECTION_METHOD}" \
    --mag-projection-percentile "${SEG_MAG_PROJECTION_PERCENTILE}" \
    --ensemble-mode "${SEG_ENSEMBLE_MODE}" \
    --classic-percentile "${SEG_CLASSIC_PERCENTILE}" \
    --post-min-component-size "${SEG_POST_MIN_COMPONENT_SIZE}" \
    --classic-sigmas "${SEG_CLASSIC_SIGMAS}" \
    $([[ "${SEG_SAVE_INTERMEDIATES}" == "1" ]] && printf '%s' "--save-intermediates")

  if [[ ! -f "${CASE_OUT_DIR}/cow_seg_final.nii.gz" ]]; then
    echo "Missing segmented mask after pipeline run: ${CASE_OUT_DIR}/cow_seg_final.nii.gz" >&2
    return 1
  fi

  cp "${CASE_OUT_DIR}/cow_seg_final.nii.gz" "${FINAL_MASK}"
  echo "Saved predicted CoW mask: ${FINAL_MASK}"
}


calc_geometry_metrics() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local META_JSON=${EXP_DIR}/inference_metadata.json
  local PRED_MASK=${EXP_DIR}/cow_seg_final.nii.gz
  local METRICS_DIR=${EXP_DIR}/metrics/geometry

  if [[ ! -f "${META_JSON}" ]]; then
    echo "Missing inference metadata: ${META_JSON}" >&2
    return 1
  fi

  if [[ ! -f "${PRED_MASK}" ]]; then
    echo "Missing predicted CoW mask: ${PRED_MASK}" >&2
    echo "Run segment_prediction_cow ${EXP_NAME} ${CASE_INDEX} first." >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_geometry_metrics.py" \
    --pred-mask "${PRED_MASK}" \
    --metadata-json "${META_JSON}" \
    --out-dir "${METRICS_DIR}"
}


calc_geometry_metrics_seghead() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local META_JSON=${EXP_DIR}/inference_metadata.json
  local PRED_MASK=${EXP_DIR}/nifti/pred_seg_bin.nii.gz
  local METRICS_DIR=${EXP_DIR}/metrics/geometry_seghead

  if [[ ! -f "${META_JSON}" ]]; then
    echo "Missing inference metadata: ${META_JSON}" >&2
    return 1
  fi

  if [[ ! -f "${PRED_MASK}" ]]; then
    echo "Missing seghead mask: ${PRED_MASK}" >&2
    echo "Re-run inference with SAVE_SEGMENTATION=1 and a model/checkpoint that emits seg_map." >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_geometry_metrics.py" \
    --pred-mask "${PRED_MASK}" \
    --metadata-json "${META_JSON}" \
    --out-dir "${METRICS_DIR}"
}


calc_geometry_metrics_stage1mask() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  local CASE_TAG
  CASE_TAG=$(printf "case%03d" "${CASE_INDEX}")
  local EXP_DIR=${OUT_ROOT}/${EXP_NAME}_${CASE_TAG}
  local META_JSON=${EXP_DIR}/inference_metadata.json
  local PRED_MASK=${EXP_DIR}/stage1_superresolution/cow_seg_final.nii.gz
  local METRICS_DIR=${EXP_DIR}/metrics/geometry_stage1mask

  if [[ ! -f "${META_JSON}" ]]; then
    echo "Missing inference metadata: ${META_JSON}" >&2
    return 1
  fi

  if [[ ! -f "${PRED_MASK}" ]]; then
    echo "Missing stage1 external mask: ${PRED_MASK}" >&2
    echo "Run stage1-mask cascade inference first." >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/code/inference/calculate_geometry_metrics.py" \
    --pred-mask "${PRED_MASK}" \
    --metadata-json "${META_JSON}" \
    --out-dir "${METRICS_DIR}"
}


calc_geometry_metrics_all() {
  local EXP_NAME=$1
  local CASE_INDEX=${2:-0}

  echo "[geometry] external segmentation pipeline"
  calc_geometry_metrics "${EXP_NAME}" "${CASE_INDEX}"

  echo "[geometry] seghead prediction"
  calc_geometry_metrics_seghead "${EXP_NAME}" "${CASE_INDEX}"
}


# ---------------------------------------------------------------------------
# Velocity metrics examples
# ---------------------------------------------------------------------------

# Denoising
# source /home/ceib/jaalzate/NeuroFlow/calculate_metrics.sh
# calc_velocity_metrics nf_orig_3t7t_r1_ps48 0
# calc_velocity_metrics nf_cbampreup_3t7t_r1_ps48 0

# Super-resolution
# source /home/ceib/jaalzate/NeuroFlow/calculate_metrics.sh
# calc_velocity_metrics nf_orig_053t7t_x2_ps24 0
# calc_velocity_metrics nf_cbampreup_053t7t_x2_ps24 0


# ---------------------------------------------------------------------------
# Background metrics examples
# ---------------------------------------------------------------------------

# Denoising
# source /Users/alejo/Documents/Internship/NeuroFlow/calculate_metrics.sh
# calc_background_metrics nf_orig_3t7t_r1_ps48 0
# calc_background_metrics nf_cbampreup_3t7t_r1_ps48 0

# Super-resolution
# source /Users/alejo/Documents/Internship/NeuroFlow/calculate_metrics.sh
# calc_background_metrics nf_orig_053t7t_x2_ps24 0
# calc_background_metrics nf_cbampreup_053t7t_x2_ps24 0


# ---------------------------------------------------------------------------
# Geometry metrics examples
# ---------------------------------------------------------------------------

# Re-segment one prediction using the same patient-pipeline settings as the
# original 7T-mask generation command, adapted only to predicted filenames.
# source /Users/alejo/Documents/Internship/NeuroFlow/calculate_metrics.sh
# segment_prediction_cow nf_orig_3t7t_r1_ps48 0
# calc_geometry_metrics nf_orig_3t7t_r1_ps48 0

# Denoising
# source /Users/alejo/Documents/Internship/NeuroFlow/calculate_metrics.sh
# segment_prediction_cow nf_orig_3t7t_r1_ps48 0
# calc_geometry_metrics nf_orig_3t7t_r1_ps48 0
# segment_prediction_cow nf_cbampreup_3t7t_r1_ps48 0
# calc_geometry_metrics nf_cbampreup_3t7t_r1_ps48 0

# Super-resolution
# source /Users/alejo/Documents/Internship/NeuroFlow/calculate_metrics.sh
# segment_prediction_cow nf_orig_053t7t_x2_ps24 0
# calc_geometry_metrics nf_orig_053t7t_x2_ps24 0
# segment_prediction_cow nf_cbampreup_053t7t_x2_ps24 0
# calc_geometry_metrics nf_cbampreup_053t7t_x2_ps24 0
#
# Seghead geometry directly from nifti/pred_seg_bin.nii.gz
# calc_geometry_metrics_seghead nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead 0
#
# Stage1 external-mask geometry directly from stage1_superresolution/cow_seg_final.nii.gz
# calc_geometry_metrics_stage1mask nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead_stage1mask 0
#
# Run both external segmentation geometry and seghead geometry
# calc_geometry_metrics_all nf_cascadee2e_cbamsr_origdn_053t7t_x2_ps24_seghead 0
