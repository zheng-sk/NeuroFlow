# 07 — Reproducing the manuscript

Maps the configuration IDs used in the manuscript tables to concrete commands.

## Prerequisites

1. A paired dataset exported and split as in
   [`03_dataset_and_splits.md`](03_dataset_and_splits.md), producing
   `loo_trainval_hrmasked_r1` (×1) and `loo_trainval_hrmasked_x2` (×2).
2. The nonvascular-noise fit summary used by the noise augmentation:
   `output/noise_pdf/nonvascular_noise_fit_summary.csv`. Override the location
   with `NOISE_CSV=`.
3. `models/topcow-claim-models/` in place for the geometry metrics, which
   re-segment each prediction.

## Configurations

Both input strategies share the same architecture and schedule; they differ only
in whether the Circle-of-Willis mask is applied to the network input.

| ID | Task | Architecture | Input | Command |
| --- | --- | --- | --- | --- |
| A | ×1 | Original | Synthetic noise | `XVAL_EXPERIMENT_NAME=A_orig_x1 MODE=denoise MODEL_VARIANT=original bash scripts/run_loo_xval.sh` |
| B | ×1 | CBAM-DNS | Synthetic noise | `XVAL_EXPERIMENT_NAME=B_cbam_x1 MODE=denoise bash scripts/run_loo_xval.sh` |
| E | ×1 | CBAM-DNS | Masked | `XVAL_EXPERIMENT_NAME=E_cbam_x1_masked MODE=denoise APPLY_MASK_TO_LR_INPUTS=1 APPLY_MASK_TO_LR_MAGNITUDE=1 bash scripts/run_loo_xval.sh` |
| C | ×2 | Original | Synthetic noise | `XVAL_EXPERIMENT_NAME=C_orig_x2 MODE=sr MODEL_VARIANT=original bash scripts/run_loo_xval.sh` |
| D | ×2 | CBAM-SupRes | Synthetic noise | `XVAL_EXPERIMENT_NAME=D_cbam_x2 MODE=sr bash scripts/run_loo_xval.sh` |
| F | ×2 | CBAM-SupRes | Masked | `XVAL_EXPERIMENT_NAME=F_cbam_x2_masked MODE=sr APPLY_MASK_TO_LR_INPUTS=1 APPLY_MASK_TO_LR_MAGNITUDE=1 bash scripts/run_loo_xval.sh` |
| G | ×2 | Stock 4DFlowNet | Original | No retraining. Use the published 4DFlowNet weights and run inference only. |

`MODEL_VARIANT` defaults to `pre_upsample_attention`, which is the CBAM model
published as CBAM-DNS at ×1 and CBAM-SupRes at ×2.

## Shared hyperparameters

From [`../scripts/run_loo_xval.sh`](../scripts/run_loo_xval.sh):

| Setting | ×1 | ×2 |
| --- | --- | --- |
| LR patch size | 48 | 24 |
| `res-increase` | 1 | 2 |
| Batch size | 8 | 8 |
| Epochs | 500 | 500 |
| Seed | 42, `--deterministic` | 42, `--deterministic` |
| `--non-fluid-loss-weight` | 0.3 | 0.3 |
| `--noise-aug-prob` | 0.8 | 0.8 |
| `--noise-aug-masked-fraction` | 0.1 | 0.1 |
| `--noise-aug-phase-scale` / `--noise-aug-mag-scale` | 0.06 / 0.04 | 0.06 / 0.04 |
| `--noise-aug-level-min` / `-max` | 1.0 / 5.0 | 1.0 / 5.0 |
| Early stopping / overfit patience | 80 / 100 | 80 / 100 |
| Validation | full volume, sliding window | full volume, sliding window |

## Evaluation

```bash
XVAL_EXPERIMENT_NAME=D_cbam_x2 EXP_PREFIX=nf_preups_cbam_ \
  bash scripts/run_xval_eval.sh
```

`EXP_PREFIX` must match the checkpoint naming of the run: `nf_preups_cbam_` for
CBAM runs, empty for the Original baseline. Set `APPLY_MASK_TO_LR=1` when
evaluating a masked-input configuration (E, F) so inference sees the same input
as training.

This produces, per fold and task:

- `metrics/velocity/` — intraluminal MSE, total and per component (Table: velocity)
- `metrics/background/` — background suppression (Table: background)
- `metrics/geometry/` — Dice and surface distances after re-segmenting the
  prediction (Table: geometry)

`scripts/run_geometry_only.sh` re-runs just the geometry stage for folds whose
metrics are missing, which is useful after a segmentation-model change.

## Reported statistics

All manuscript numbers are LOOCV means ± standard deviation across the seven
subjects. Velocity MSE is reported in units of 10⁻³, computed inside the
Circle-of-Willis mask, with VENC = 0.9 m/s for both field strengths.

## Ablations not on this branch

The cascade, phase-attention, and JiT-SR architectures cited as supplementary
material were removed from `main` to keep the published artifact scoped to the
reported results. They are preserved in full on `neuroflow_dev`, along with
`run_cascade_e2e.sh` and `run_exp.sh`:

```bash
git checkout neuroflow_dev
```

On that branch `available_model_variants()` additionally returns
`cascade_sr_dn_masked`, `phase1_attention`, `phase2_attention`,
`phase3_transformer_cross_attention`, and `jit_sr_3d`.
