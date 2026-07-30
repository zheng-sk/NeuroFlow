#!/usr/bin/env bash
# Full LOOCV evaluation: inference + velocity/background/geometry metrics
# for both r1_noise (denoising) and x2_noise (2x SR) folds.
# Distributes across 3 GPUs in parallel.
#
# Usage:
#   bash run_xval_eval.sh                              # defaults (CBAM pre-upsample)
#   XVAL_EXPERIMENT_NAME=xval_original_unmasked \
#     EXP_PREFIX="" MODEL_VARIANT=original \
#     OUT_ROOT=output/inference_only_original \
#     bash run_xval_eval.sh
#
# Geometry phase uses the external ToP-CoW segmentation pipeline.
# Set SEG_GPU to pin it to a specific GPU (default: empty = no CUDA_VISIBLE_DEVICES override).

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${ROOT}/.venv_neuroflow/bin/python
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python

# ── Configurable parameters ──────────────────────────────────────────────────
XVAL_EXPERIMENT_NAME=${XVAL_EXPERIMENT_NAME:-xval_cbam_preups_unmasked}
# Prefix prepended to "{task}_{fold}" to form the checkpoint name.
# CBAM: "nf_preups_cbam_"   Original: ""
EXP_PREFIX=${EXP_PREFIX-nf_preups_cbam_}
MODEL_VARIANT=${MODEL_VARIANT:-pre_upsample_attention}
OUT_ROOT=${OUT_ROOT:-${ROOT}/output/inference_only}
SEG_MODEL_DIR=${SEG_MODEL_DIR:-${ROOT}/models/topcow-claim-models}
SEG_GPU=${SEG_GPU:-0}
# Set to 1 to pass --apply-mask-to-lr-inputs and --apply-mask-to-lr-magnitude at inference
APPLY_MASK_TO_LR=${APPLY_MASK_TO_LR:-0}
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR=${ROOT}/output/xval_eval_logs
mkdir -p "${LOG_DIR}"

echo "[CONFIG] XVAL_EXPERIMENT_NAME = ${XVAL_EXPERIMENT_NAME}"
echo "[CONFIG] EXP_PREFIX           = '${EXP_PREFIX}'"
echo "[CONFIG] MODEL_VARIANT        = ${MODEL_VARIANT}"
echo "[CONFIG] OUT_ROOT             = ${OUT_ROOT}"
echo "[CONFIG] SEG_MODEL_DIR        = ${SEG_MODEL_DIR}"
echo "[CONFIG] SEG_GPU              = '${SEG_GPU}'"
echo "[CONFIG] APPLY_MASK_TO_LR    = ${APPLY_MASK_TO_LR}"

# ──────────────────────────────────────────────
# Helper: run inference for one fold
# ──────────────────────────────────────────────
run_fold_inference() {
  local task=$1        # r1_noise | x2_noise
  local fold=$2        # e.g. fold_000_<case-id>
  local gpu=$3
  local patch=$4       # 48 for r1, 24 for x2
  local res=$5         # 1 for r1, 2 for x2
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

  # Choose correct val CSV
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
    -m neuroflow.inference.run_sr_inference_case \
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
# Helper: segment prediction + geometry metrics
# ──────────────────────────────────────────────
run_fold_geometry() {
  local task=$1
  local fold=$2
  local gpu=$3
  local exp_name="${EXP_PREFIX}${task}_${fold}"
  local case_tag="case000"
  local exp_dir="${OUT_ROOT}/${exp_name}_${case_tag}"
  local meta_json="${exp_dir}/inference_metadata.json"
  local seg_tmp="${exp_dir}/_segmentation_tmp"
  local final_mask="${exp_dir}/cow_seg_final.nii.gz"

  if [[ ! -f "${meta_json}" ]]; then
    echo "[ERROR] Missing inference_metadata.json for ${exp_name}" >&2
    return 1
  fi

  echo "[INFO] Segmenting prediction → ${exp_name} | GPU=${gpu}"
  rm -rf "${seg_tmp}"

  local cmd_prefix=()
  [[ -n "${gpu}" ]] && cmd_prefix=(env CUDA_VISIBLE_DEVICES="${gpu}")

  "${cmd_prefix[@]}" "${PYTHON_BIN}" \
    -m neuroflow.segmentation.segment_cow_patient_pipeline \
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
    --mag-projection-method    percentile \
    --mag-projection-percentile 100 \
    --ensemble-mode union \
    --classic-percentile 97.5 \
    --post-min-component-size 30 \
    --classic-sigmas 0.8,1.2,1.6,2.0

  local seg_out="${seg_tmp}/${exp_name}_${case_tag}/cow_seg_final.nii.gz"
  if [[ ! -f "${seg_out}" ]]; then
    echo "[ERROR] Segmentation output missing: ${seg_out}" >&2
    return 1
  fi
  cp "${seg_out}" "${final_mask}"

  echo "[INFO] Geometry metrics → ${exp_name}"
  "${PYTHON_BIN}" -m neuroflow.evaluation.calculate_geometry_metrics \
    --pred-mask     "${final_mask}" \
    --metadata-json "${meta_json}" \
    --out-dir       "${exp_dir}/metrics/geometry"
}

# ──────────────────────────────────────────────
# Helper: compute metrics for one fold
# ──────────────────────────────────────────────
run_fold_metrics() {
  local task=$1
  local fold=$2
  local exp_name="${EXP_PREFIX}${task}_${fold}"
  local case_tag="case000"
  local exp_dir="${OUT_ROOT}/${exp_name}_${case_tag}"
  local payload="${exp_dir}/analysis_payload.npz"

  if [[ ! -f "${payload}" ]]; then
    echo "[ERROR] Missing payload for ${exp_name}: ${payload}" >&2
    return 1
  fi

  echo "[INFO] Velocity metrics → ${exp_name}"
  "${PYTHON_BIN}" -m neuroflow.evaluation.calculate_velocity_metrics \
    --payload-npz "${payload}" \
    --out-dir     "${exp_dir}/metrics/velocity" \
    --non-fluid-loss-weight 0.3

  echo "[INFO] Background metrics → ${exp_name}"
  "${PYTHON_BIN}" -m neuroflow.evaluation.calculate_background_metrics \
    --payload-npz "${payload}" \
    --out-dir     "${exp_dir}/metrics/background"
}

# ──────────────────────────────────────────────
# Fold discovery
#
# Fold names embed subject identifiers, so they are discovered from the LOOCV
# split directories at runtime instead of being hardcoded here. This keeps the
# cohort out of the published repository. Point *_FOLDS_DIR at your own splits
# (produced by neuroflow.registration.generate_loo_cv_csv).
# ──────────────────────────────────────────────
R1_FOLDS_DIR=${R1_FOLDS_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_r1}
X2_FOLDS_DIR=${X2_FOLDS_DIR:-${ROOT}/data/paired_dataset/loo_trainval_hrmasked_x2}

discover_folds() {
  local dir=$1
  if [[ ! -d "${dir}" ]]; then
    echo "Error: folds dir not found: ${dir}" >&2
    exit 1
  fi
  local found=()
  local d
  for d in "${dir}"/fold_*; do
    [[ -d "${d}" ]] && found+=("$(basename "${d}")")
  done
  if [[ ${#found[@]} -eq 0 ]]; then
    echo "Error: no fold_* directories found in ${dir}" >&2
    exit 1
  fi
  printf '%s\n' "${found[@]}"
}

mapfile -t R1_FOLDS < <(discover_folds "${R1_FOLDS_DIR}")
mapfile -t X2_FOLDS < <(discover_folds "${X2_FOLDS_DIR}")

# GPU assignment (round-robin across 0,1,2)
GPUS=(0 1 2)

# ──────────────────────────────────────────────
# Phase 1: Inference (all folds in parallel)
# ──────────────────────────────────────────────
echo "=========================================="
echo " Phase 1: Running inference for all folds"
echo "=========================================="

PIDS=()
IDX=0

for fold in "${R1_FOLDS[@]}"; do
  gpu=${GPUS[$((IDX % 3))]}
  run_fold_inference r1_noise "${fold}" "${gpu}" 48 1 2 \
    > "${LOG_DIR}/infer_r1_${fold}.log" 2>&1 &
  PIDS+=($!)
  echo "[LAUNCHED] r1_noise ${fold} on GPU ${gpu} (PID $!)"
  IDX=$((IDX+1))
done

for fold in "${X2_FOLDS[@]}"; do
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
echo "Inference complete. Failed jobs: ${FAILED}"

# ──────────────────────────────────────────────
# Phase 2: Metrics (all folds sequentially — fast)
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Phase 2: Computing metrics for all folds"
echo "=========================================="

for fold in "${R1_FOLDS[@]}"; do
  run_fold_metrics r1_noise "${fold}" \
    > "${LOG_DIR}/metrics_r1_${fold}.log" 2>&1 \
    && echo "[OK] r1_noise ${fold}" \
    || echo "[FAIL] r1_noise ${fold}"
done

for fold in "${X2_FOLDS[@]}"; do
  run_fold_metrics x2_noise "${fold}" \
    > "${LOG_DIR}/metrics_x2_${fold}.log" 2>&1 \
    && echo "[OK] x2_noise ${fold}" \
    || echo "[FAIL] x2_noise ${fold}"
done

# ──────────────────────────────────────────────
# Phase 3: Geometry (segment + metrics, sequential — segmentation model is GPU-heavy)
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Phase 3: Segmentation + geometry metrics"
echo "=========================================="

GEO_FAILED=0

for fold in "${R1_FOLDS[@]}"; do
  exp_name="${EXP_PREFIX}r1_noise_${fold}"
  geo_csv="${OUT_ROOT}/${exp_name}_case000/metrics/geometry/geometry_metrics_compact.csv"
  if [[ -f "${geo_csv}" ]]; then
    echo "[SKIP] geometry r1_noise ${fold} (already done)"
    continue
  fi
  run_fold_geometry r1_noise "${fold}" "${SEG_GPU}" \
    > "${LOG_DIR}/geometry_r1_${fold}.log" 2>&1 \
    && echo "[OK] geometry r1_noise ${fold}" \
    || { echo "[FAIL] geometry r1_noise ${fold}"; GEO_FAILED=$((GEO_FAILED+1)); }
done

for fold in "${X2_FOLDS[@]}"; do
  exp_name="${EXP_PREFIX}x2_noise_${fold}"
  geo_csv="${OUT_ROOT}/${exp_name}_case000/metrics/geometry/geometry_metrics_compact.csv"
  if [[ -f "${geo_csv}" ]]; then
    echo "[SKIP] geometry x2_noise ${fold} (already done)"
    continue
  fi
  run_fold_geometry x2_noise "${fold}" "${SEG_GPU}" \
    > "${LOG_DIR}/geometry_x2_${fold}.log" 2>&1 \
    && echo "[OK] geometry x2_noise ${fold}" \
    || { echo "[FAIL] geometry x2_noise ${fold}"; GEO_FAILED=$((GEO_FAILED+1)); }
done

echo "Geometry complete. Failed jobs: ${GEO_FAILED}"

echo ""
echo "=========================================="
echo " All done. Results in: ${OUT_ROOT}"
echo " Logs in:              ${LOG_DIR}"
echo "=========================================="
