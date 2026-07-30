"""
Generate inference CSVs for predictions over 7T images.

Produces 4 CSVs under data/paired_dataset/:

  predict_7t_den/                  -- r1_noise applied TO 7T (7T-Den)
    hr_7t_in_3t.csv                  input = unmasked 7T
    hr_7t_in_3t_masked.csv           input = masked 7T

  predict_7t_sr/                   -- x2_noise applied TO 7T (7T-SR)
    hr_7t_in_3t.csv                  input = unmasked 7T
    hr_7t_in_3t_masked.csv           input = masked 7T

In both cases lr_* = 7T images (the csv variant determines masked/unmasked).
hr_* is set to the same 7T images (no separate ground truth exists).
Timing metadata is taken from the 7T columns of each subject's LOOCV val.csv
(hr_trigger_time_ms_map → lr_trigger_time_ms_map, hr_time_index_map kept).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paired_dataset"

SUBJECT_TO_R1_FOLD = {
    "001_20240313": DATA / "loo_trainval_hrmasked_r1" / "fold_000_001_20240313",
    "002_20240326": DATA / "loo_trainval_hrmasked_r1" / "fold_001_002_20240326",
    "003_20240422": DATA / "loo_trainval_hrmasked_r1" / "fold_002_003_20240422",
    "004_20240619": DATA / "loo_trainval_hrmasked_r1" / "fold_003_004_20240619",
    "005_20241211": DATA / "loo_trainval_hrmasked_r1" / "fold_004_005_20241211",
    "006_20241218": DATA / "loo_trainval_hrmasked_r1" / "fold_005_006_20241218",
    "007_20250826": DATA / "loo_trainval_hrmasked_r1" / "fold_006_007_20250826",
}

SUBJECT_TO_X2_FOLD = {
    "001_20240313": DATA / "loo_trainval_hrmasked_x2" / "fold_000_001_20240313",
    "002_20240326": DATA / "loo_trainval_hrmasked_x2" / "fold_001_002_20240326",
    "003_20240422": DATA / "loo_trainval_hrmasked_x2" / "fold_002_003_20240422",
    "004_20240619": DATA / "loo_trainval_hrmasked_x2" / "fold_003_004_20240619",
    "005_20241211": DATA / "loo_trainval_hrmasked_x2" / "fold_004_005_20241211",
    "006_20241218": DATA / "loo_trainval_hrmasked_x2" / "fold_005_006_20241218",
    "007_20250826": DATA / "loo_trainval_hrmasked_x2" / "fold_006_007_20250826",
}

SUBJECTS = list(SUBJECT_TO_R1_FOLD.keys())

FIELDNAMES = [
    "lr_u", "lr_v", "lr_w",
    "lr_mag_u", "lr_mag_v", "lr_mag_w",
    "hr_u", "hr_v", "hr_w", "hr_mag",
    "mask",
    "venc", "venc_u", "venc_v", "venc_w",
    "time_start", "time_end", "time_index",
    "hr_time_index_map",
    "lr_trigger_time_ms_map", "hr_trigger_time_ms_map",
    "pairing_method",
]


def read_val_row(val_csv: Path, subject: str) -> dict:
    with open(val_csv, newline="") as f:
        for row in csv.DictReader(f):
            if subject in row["lr_u"]:
                return row
    raise ValueError(f"Subject {subject} not found in {val_csv}")


def make_7t_input_row(subject: str, input_dir: str, fold_map: dict) -> dict:
    """
    Build a row where the 7T image is the LR input.

    lr_* / hr_* both point to `input_dir` (same image, no separate GT).
    Timing comes from the 7T columns of the original val.csv:
      - lr_trigger_time_ms_map ← hr_trigger_time_ms_map  (7T trigger times)
      - hr_time_index_map kept as-is                     (7T frame indices)
    time_end is recomputed from the number of selected 7T frames.
    """
    fold_dir = fold_map[subject]
    meta = read_val_row(fold_dir / "val.csv", subject)

    root_7t = f"data/paired_dataset/{input_dir}/{subject}"
    mask_path = (
        f"data/paired_dataset/cow_segmentation"
        f"/{subject}/cow_seg_final.nii.gz"
    )

    # 7T frame indices (previously hr side)
    hr_time_index_map = meta["hr_time_index_map"]
    n_frames = len(hr_time_index_map.split(";")) if hr_time_index_map else 0

    return {
        "lr_u":   f"{root_7t}/Vx.nii.gz",
        "lr_v":   f"{root_7t}/Vy.nii.gz",
        "lr_w":   f"{root_7t}/Vz.nii.gz",
        "lr_mag_u": f"{root_7t}/input_mag_raw.nii.gz",
        "lr_mag_v": f"{root_7t}/input_mag_raw.nii.gz",
        "lr_mag_w": f"{root_7t}/input_mag_raw.nii.gz",
        "hr_u":   f"{root_7t}/Vx.nii.gz",
        "hr_v":   f"{root_7t}/Vy.nii.gz",
        "hr_w":   f"{root_7t}/Vz.nii.gz",
        "hr_mag": f"{root_7t}/input_mag_raw.nii.gz",
        "mask":   mask_path,
        "venc":   meta["venc"],
        "venc_u": meta["venc_u"],
        "venc_v": meta["venc_v"],
        "venc_w": meta["venc_w"],
        "time_start": meta["time_start"],
        "time_end":   n_frames,
        "time_index": "",
        "hr_time_index_map":      hr_time_index_map,
        # 7T trigger times become the LR trigger times
        "lr_trigger_time_ms_map": meta["hr_trigger_time_ms_map"],
        "hr_trigger_time_ms_map": meta["hr_trigger_time_ms_map"],
        "pairing_method": meta["pairing_method"],
    }


def write_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} rows → {out_path.relative_to(ROOT)}")


configs = [
    # (input_dir,              fold_map,           out_subdir)
    ("hr_7t_in_3t",        SUBJECT_TO_R1_FOLD, "predict_7t_den"),
    ("hr_7t_in_3t_masked", SUBJECT_TO_R1_FOLD, "predict_7t_den"),
    ("hr_7t_in_3t",        SUBJECT_TO_X2_FOLD, "predict_7t_sr"),
    ("hr_7t_in_3t_masked", SUBJECT_TO_X2_FOLD, "predict_7t_sr"),
]

for input_dir, fold_map, out_subdir in configs:
    rows = [make_7t_input_row(s, input_dir, fold_map) for s in SUBJECTS]
    out_path = DATA / out_subdir / f"{input_dir}.csv"
    write_csv(out_path, rows)
