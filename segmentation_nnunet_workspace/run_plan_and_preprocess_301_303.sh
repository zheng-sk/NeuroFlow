#!/usr/bin/env bash

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "${WORKSPACE_ROOT}/env_nnunet.sh"

for dataset_id in 301 302 303; do
  "${PYTHON_BIN}" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
    -d "${dataset_id}" \
    --verify_dataset_integrity | tee "${WORKSPACE_ROOT}/logs/plan_and_preprocess_${dataset_id}.log"
done
