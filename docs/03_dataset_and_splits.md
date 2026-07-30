# 03 — Paired dataset and cross-validation splits

Covers README stages 8–10: turning registered, segmented volume pairs into the
case manifest that training consumes, generating the ×2 low-resolution variant,
and building patient-level leave-one-out splits.

## 8. Export the paired LR/HR dataset

```bash
python -m neuroflow.registration.export_paired_lr_hr_dataset \
  --temporal-dir   data/temporal_registered_cow_crop \
  --registered-dir data/registered_7T_in_3T_cow_crop \
  --output-root    data/paired_dataset
```

Produces:

```
data/paired_dataset/
  lr_3t/<case>/{Vx,Vy,Vz,input_mag_raw}.nii.gz     # 3T, the network input
  hr_7t_in_3t/<case>/{Vx,Vy,Vz,input_mag_raw}.nii.gz  # 7T resampled into 3T space
  cases.csv                                        # the manifest
```

`--csv-only` regenerates the manifest from an existing export without
recopying volumes, which is what you want after fixing a mask or a crop.

### Frame pairing

3T and 7T series rarely have the same number of cardiac frames (9–14 vs 10–15 in
this cohort). Frames are paired by **nearest normalised trigger time**:

1. Each series' trigger times are normalised onto `[0, 1]` across its own length,
   so the pairing is on cardiac phase rather than absolute milliseconds.
2. Every LR frame is matched to the HR frame with the nearest normalised phase.

The result is recorded per case in `hr_time_index_map`, with `pairing_method`
set to `trigger_time_nearest`.

> Because normalisation is relative to each sequence's own length, a mapping
> computed on truncated trigger-time lists is **not** a prefix of the mapping
> computed on the full lists. `tests/test_trigger_time_pairing.py` documents this.

To (re)generate just the pairing columns on an existing CSV:

```bash
python -m neuroflow.registration.generate_trigger_time_frame_pairs \
  --in-csv  data/paired_dataset/cases.csv \
  --out-csv data/paired_dataset/cases_paired.csv
```

### Attaching Circle-of-Willis masks

```bash
python -m neuroflow.registration.attach_cow_masks_to_csv \
  --in-csv   data/paired_dataset/cases.csv \
  --out-csv  data/paired_dataset/cases_with_masks.csv \
  --mask-root data/cow_segmentation
```

The `mask` column defines the intraluminal region for the loss. Rows may leave
it empty, in which case the full volume is used and the non-fluid loss term has
no effect for that case.

### Manifest schema

23 columns; see [`../examples/manifest_template.csv`](../examples/manifest_template.csv).

| Group | Columns | Notes |
| --- | --- | --- |
| LR velocity | `lr_u`, `lr_v`, `lr_w` | RAW phase unless `--already-velocity-input` |
| LR magnitude | `lr_mag_u`, `lr_mag_v`, `lr_mag_w` | often the same file three times |
| HR velocity | `hr_u`, `hr_v`, `hr_w` | 7T resampled into 3T space |
| HR magnitude | `hr_mag` | target for `--predict-mag` |
| Loss region | `mask` | optional |
| VENC | `venc`, `venc_u`, `venc_v`, `venc_w` | per-component overrides win; `venc <= 0` is estimated |
| Frame range | `time_start`, `time_end`, `time_index` | `time_end` is exclusive |
| Pairing | `lr_time_index`, `hr_time_index`, `lr_trigger_time_ms`, `hr_trigger_time_ms`, `pairing_method` | `;`-separated |

**QC:** `time_end` must equal the number of usable LR frames, and
`hr_time_index_map` must contain exactly that many entries. A mismatch means a
frame was dropped during registration.

## 9. ×2 low-resolution generation

For the super-resolution task the network input is a further 2× downsampled
volume, produced by trilinear resampling at scale 0.5.

**Option A — downsample the 7T tree** (target stays in 7T space, clean synthetic LR):

```bash
python -m neuroflow.preprocessing.downsample_nifti_tree \
  --input-root  data/paired_dataset/hr_7t_in_3t \
  --output-root data/paired_dataset/lr_from_7t_x05 \
  --scale 0.5 --time-axis -1

python -m neuroflow.preprocessing.remap_case_csv_for_x2 \
  --in-csv  data/paired_dataset/cases.csv \
  --out-csv data/paired_dataset/cases_x2_from7t.csv \
  --mode lr_from_hr \
  --source-root data/paired_dataset/hr_7t_in_3t \
  --new-lr-root data/paired_dataset/lr_from_7t_x05
```

**Option B — downsample the 3T tree** (a more degraded, clinically realistic input):

```bash
python -m neuroflow.preprocessing.downsample_nifti_tree \
  --input-root  data/paired_dataset/lr_3t \
  --output-root data/paired_dataset/lr_3t_x05 \
  --scale 0.5 --time-axis -1

python -m neuroflow.preprocessing.remap_case_csv_for_x2 \
  --in-csv  data/paired_dataset/cases.csv \
  --out-csv data/paired_dataset/cases_x2_from3t.csv \
  --mode lr_from_lr \
  --source-root data/paired_dataset/lr_3t \
  --new-lr-root data/paired_dataset/lr_3t_x05
```

The manuscript's ×2 results use Option B. Masks must be downsampled to match;
`neuroflow.segmentation.resample_cow_masks_to_reference` handles that.

**QC:** LR dimensions must be exactly half the HR dimensions in x, y, z, with the
time axis untouched. The affine voxel size should double.

## 10. Patient-level LOOCV splits

With n = 7 subjects, evaluation is leave-one-out. Splits are held out **by
patient**, never by frame, so no subject contributes to both train and
validation.

```bash
# x1 (denoising) folds
python -m neuroflow.registration.generate_loo_cv_csv \
  --csv-in  data/paired_dataset/cases_with_masks.csv \
  --out-dir data/paired_dataset/loo_trainval_hrmasked_r1 \
  --train-val-only

# x2 (super-resolution) folds
python -m neuroflow.registration.generate_loo_cv_csv \
  --csv-in  data/paired_dataset/cases_x2_from3t.csv \
  --out-dir data/paired_dataset/loo_trainval_hrmasked_x2 \
  --train-val-only
```

Each fold directory holds `train.csv` and `val.csv`:

```
data/paired_dataset/loo_trainval_hrmasked_r1/
  fold_000_<case>/{train,val}.csv
  fold_001_<case>/{train,val}.csv
  ...
```

Key options:

| Flag | Effect |
| --- | --- |
| `--train-val-only` | Leave-one-out validation, no separate test split. Used for the manuscript. |
| `--val-num-patients N` | Validation patients per fold; `0` disables the validation split. |
| `--hr-col`, `--hr-root-name` | How the patient/case ID is recovered from the HR path. |
| `--val-strategy cyclic\|random`, `--seed` | Validation-patient selection. |
| `--strict` | Fail rather than warn on missing files. |

Fold directory names embed the case ID. The published scripts discover them by
globbing rather than hardcoding, so no identifier enters the repository — see
`discover_folds` in [`../scripts/run_xval_eval.sh`](../scripts/run_xval_eval.sh).

**QC:** with n = 7 you should get 7 folds; each `val.csv` should contain exactly
one patient's rows, and no case ID should appear in both files of a fold.

Next: [`04_training.md`](04_training.md).
