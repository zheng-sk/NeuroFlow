#!/usr/bin/env bash

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"

if [[ -z "${NNUNET_REPO_DIR:-}" ]]; then
  if [[ -d "${WORKSPACE_ROOT}/nnUNet" ]]; then
    NNUNET_REPO_DIR="${WORKSPACE_ROOT}/nnUNet"
  else
    NNUNET_REPO_DIR="${WORKSPACE_ROOT}/topcow-2024-nnunet"
  fi
fi

export NNUNET_REPO_DIR
export nnUNet_raw="${nnUNet_raw:-${WORKSPACE_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${WORKSPACE_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${WORKSPACE_ROOT}/nnUNet_results}"
export PYTHONPATH="${NNUNET_REPO_DIR}:${REPO_ROOT}:${PYTHONPATH}"

echo "NNUNET_REPO_DIR=${NNUNET_REPO_DIR}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
