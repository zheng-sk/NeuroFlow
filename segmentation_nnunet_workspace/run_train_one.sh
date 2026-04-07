#!/usr/bin/env bash

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "${WORKSPACE_ROOT}/env_nnunet.sh"

DATASET_ID="${DATASET_ID:-301}"
FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"

CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON_BIN}" -m nnunetv2.run.run_training \
  "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
  -tr "${TRAINER}" \
  --npz | tee "${WORKSPACE_ROOT}/logs/train_${DATASET_ID}_fold${FOLD}.log"
