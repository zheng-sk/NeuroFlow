#!/usr/bin/env bash
# Fix all artifacts for the 4 corrected crop cases (002, 005, 006, 007):
#   1. Regenerate cow_segmentation_x05 from cow_segmentation_ens301_current
#   2. Re-segment cow_seg_final in output/inference_only_masked (all 8 dirs)
#   3. Re-compute geometry metrics for those 8 dirs
#
# Usage:
#   SEG_GPU=0 bash run_fix_corrected_cases.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${ROOT}/.venv_neuroflow/bin/python"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python

export OUT_ROOT="${ROOT}/output/inference_only_masked"
export SEG_MODEL_DIR="${ROOT}/models/topcow-claim-models"
export SEG_GPU=${SEG_GPU:-0}
export PYTHON_BIN

CASES=("002_20240326" "005_20241211" "006_20241218" "007_20250826")
FOLDS=(
  "fold_001_002_20240326"
  "fold_004_005_20241211"
  "fold_005_006_20241218"
  "fold_006_007_20250826"
)

# ---------------------------------------------------------------------------
# Step 1: Regenerate cow_segmentation_x05 (0.5x of ens301_current)
# ---------------------------------------------------------------------------
echo "========================================"
echo " Step 1: Regenerate cow_segmentation_x05"
echo "========================================"

SEG_IN="${ROOT}/data/paired_dataset/cow_segmentation_ens301_current"
SEG_OUT="${ROOT}/data/paired_dataset/cow_segmentation_x05"

for case_id in "${CASES[@]}"; do
  echo "[INFO] Downsampling ${case_id} ..."
  "${PYTHON_BIN}" "${ROOT}/code/preprocessing/downsample_nifti_tree.py" \
    --input-root  "${SEG_IN}/${case_id}" \
    --output-root "${SEG_OUT}/${case_id}" \
    --scale 0.5
  echo "[OK]   cow_segmentation_x05/${case_id}"
done

# Verify
echo ""
echo "--- x05 shape verification ---"
"${PYTHON_BIN}" - "${CASES[@]}" <<'PYEOF'
import sys, nibabel as nib
from pathlib import Path
base_ens = Path("/home/ceib/jaalzate/NeuroFlow/data/paired_dataset/cow_segmentation_ens301_current")
base_x05 = Path("/home/ceib/jaalzate/NeuroFlow/data/paired_dataset/cow_segmentation_x05")
for c in sys.argv[1:]:
    ens = nib.load(str(base_ens / c / "cow_seg_final.nii.gz"))
    x05 = nib.load(str(base_x05 / c / "cow_seg_final.nii.gz"))
    print(f"  {c}: ens301={ens.shape} -> x05={x05.shape}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2 & 3: Re-segment + geometry metrics for all 8 inference dirs
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo " Step 2+3: Segment + geometry metrics"
echo "========================================"

# Source the metrics helper functions
# shellcheck disable=SC1091
source "${ROOT}/calculate_metrics.sh"

FAILED=0
for task in r1_noise x2_noise; do
  for fold in "${FOLDS[@]}"; do
    exp_name="${task}_${fold}"
    exp_dir="${OUT_ROOT}/${exp_name}_case000"

    echo ""
    echo "[INFO] Processing ${exp_name} ..."

    # Remove stale geometry metrics so calc_geometry_metrics runs fresh
    rm -rf "${exp_dir}/metrics/geometry"

    CUDA_VISIBLE_DEVICES="${SEG_GPU}" \
      segment_prediction_cow "${exp_name}" 0 \
      && echo "[OK]   segment_prediction_cow ${exp_name}" \
      || { echo "[FAIL] segment_prediction_cow ${exp_name}" >&2; FAILED=$((FAILED+1)); continue; }

    calc_geometry_metrics "${exp_name}" 0 \
      && echo "[OK]   calc_geometry_metrics ${exp_name}" \
      || { echo "[FAIL] calc_geometry_metrics ${exp_name}" >&2; FAILED=$((FAILED+1)); }
  done
done

# ---------------------------------------------------------------------------
# Final shape check: cow_seg_final vs pred_mag for all 8 dirs
# ---------------------------------------------------------------------------
echo ""
echo "--- Shape match verification ---"
"${PYTHON_BIN}" - "${OUT_ROOT}" "${FOLDS[@]}" <<'PYEOF'
import sys, nibabel as nib
from pathlib import Path
out_root = Path(sys.argv[1])
folds = sys.argv[2:]
all_ok = True
for task in ["r1_noise", "x2_noise"]:
    for fold in folds:
        d = out_root / f"{task}_{fold}_case000"
        seg = nib.load(str(d / "cow_seg_final.nii.gz"))
        mag = nib.load(str(d / "nifti/pred_mag.nii.gz"))
        match = seg.shape[:3] == mag.shape[:3]
        status = "OK  " if match else "FAIL"
        print(f"  [{status}] {task}_{fold}: seg={seg.shape[:3]}, mag={mag.shape[:3]}")
        if not match:
            all_ok = False
if not all_ok:
    sys.exit(1)
PYEOF

echo ""
if [[ "${FAILED}" -gt 0 ]]; then
  echo "[ERROR] ${FAILED} step(s) failed. See output above." >&2
  exit 1
fi
echo "=== All steps complete ==="
echo "  x05 masks:  data/paired_dataset/cow_segmentation_x05  (updated: ${CASES[*]})"
echo "  Seg+metrics: output/inference_only_masked              (updated: ${FOLDS[*]})"
