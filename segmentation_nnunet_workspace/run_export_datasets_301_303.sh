#!/usr/bin/env bash

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"

MANIFEST_CSV="${MANIFEST_CSV:-${REPO_ROOT}/data/segmentation_nnunet/manifest_master_3t_gt7t.csv}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-${WORKSPACE_ROOT}/nnUNet_raw}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/segmentation/export_nnunet_cow_dataset.py" \
  --manifest-csv "${MANIFEST_CSV}" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --experiment mag_proj --dataset-id 301 --dataset-name CoW3TMagProj \
  --experiment mag_vel_proj --dataset-id 302 --dataset-name CoW3TMagVelProj \
  --experiment angio_mag_speed --dataset-id 303 --dataset-name CoW3TAngioMagSpeed \
  --mag-projection-method max \
  --vel-projection-method abs_max \
  --speed-percentile 90 \
  --verbose
