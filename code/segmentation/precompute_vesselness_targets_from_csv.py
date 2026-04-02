import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from segment_cow_patient_pipeline import (
    compute_classical_vesselness_map,
    convert_raw_to_velocity_repo_style,
    ensure_time_last,
    load_mask_aligned,
    parse_sigmas,
    robust_norm_in_mask,
    save_3d_with_ref,
    temporal_projection,
)


def _resolve_path(path_value: str, base_dir: Path) -> Path | None:
    value = str(path_value or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path

    base_abs = base_dir.resolve()
    search_roots = [base_abs, Path.cwd().resolve()]

    cursor = base_abs
    while True:
        parent = cursor.parent
        if parent == cursor:
            break
        search_roots.append(parent)
        cursor = parent

    seen = set()
    ordered_roots = []
    for root in search_roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        ordered_roots.append(root)

    for root in ordered_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    return (base_abs / path).resolve()


def _infer_hr_mag_path(row: dict, base_dir: Path) -> Path:
    hr_mag = _resolve_path(row.get("hr_mag", ""), base_dir)
    if hr_mag is not None:
        return hr_mag
    hr_u = _resolve_path(row["hr_u"], base_dir)
    if hr_u is None:
        raise FileNotFoundError("Missing hr_u in CSV row while trying to infer hr_mag.")
    inferred = hr_u.parent / "input_mag_raw.nii.gz"
    if inferred.exists():
        return inferred
    raise FileNotFoundError(
        f"Could not infer hr_mag for row with hr_u={row['hr_u']!r}. "
        "Add `hr_mag` to the CSV or ensure input_mag_raw.nii.gz exists beside hr_u."
    )


def _case_id(row: dict, idx: int) -> str:
    for key in ("case_id", "case", "subject_id", "patient_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    for key in ("mask", "hr_u"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return Path(value).resolve().parent.name
    return f"case_{idx:04d}"


def _compute_angio_from_row(row: dict, base_dir: Path, args):
    hr_mag_path = _infer_hr_mag_path(row, base_dir)
    hr_u_path = _resolve_path(row["hr_u"], base_dir)
    hr_v_path = _resolve_path(row["hr_v"], base_dir)
    hr_w_path = _resolve_path(row["hr_w"], base_dir)
    if hr_u_path is None or hr_v_path is None or hr_w_path is None:
        raise FileNotFoundError("CSV row must contain hr_u, hr_v, and hr_w paths.")
    mask_path = _resolve_path(row.get("mask", ""), base_dir) if str(row.get("mask", "") or "").strip() else None

    mag_img = nib.load(str(hr_mag_path))
    mag4d = ensure_time_last(np.asarray(mag_img.dataobj, dtype=np.float32), args.time_axis)

    if mask_path is not None:
        analysis_mask = load_mask_aligned(mask_path, mag_img, args.mask_threshold)
    else:
        analysis_mask = np.ones(mag4d.shape[:3], dtype=np.float32)
    analysis_mask_bool = analysis_mask > 0

    if args.angio_mode == "mag_only":
        mag_proj = temporal_projection(
            mag4d,
            method=args.mag_projection_method,
            percentile=args.mag_projection_percentile,
            topk=args.mag_projection_topk,
        ).astype(np.float32)
        angio_3d = robust_norm_in_mask(mag_proj, analysis_mask).astype(np.float32)
        return mag_img, analysis_mask_bool, angio_3d

    vx_img = nib.load(str(hr_u_path))
    vy_img = nib.load(str(hr_v_path))
    vz_img = nib.load(str(hr_w_path))
    vx4d_raw = ensure_time_last(np.asarray(vx_img.dataobj, dtype=np.float32), args.time_axis)
    vy4d_raw = ensure_time_last(np.asarray(vy_img.dataobj, dtype=np.float32), args.time_axis)
    vz4d_raw = ensure_time_last(np.asarray(vz_img.dataobj, dtype=np.float32), args.time_axis)

    if not (mag4d.shape == vx4d_raw.shape == vy4d_raw.shape == vz4d_raw.shape):
        raise ValueError(
            "Shape mismatch:\n"
            f"MAG {mag4d.shape}\n"
            f"Vx  {vx4d_raw.shape}\n"
            f"Vy  {vy4d_raw.shape}\n"
            f"Vz  {vz4d_raw.shape}"
        )

    vx4d = convert_raw_to_velocity_repo_style(vx4d_raw, args.venc, invert_sign=args.invert_uv_sign_on_raw)
    vy4d = convert_raw_to_velocity_repo_style(vy4d_raw, args.venc, invert_sign=args.invert_uv_sign_on_raw)
    vz4d = convert_raw_to_velocity_repo_style(vz4d_raw, args.venc, invert_sign=False)
    speed4d = np.sqrt(vx4d * vx4d + vy4d * vy4d + vz4d * vz4d).astype(np.float32)

    mag_med = np.median(mag4d, axis=3).astype(np.float32)
    speed_proj = np.percentile(speed4d, args.speed_percentile, axis=3).astype(np.float32)
    angio_3d = (
        robust_norm_in_mask(mag_med, analysis_mask) * robust_norm_in_mask(speed_proj, analysis_mask)
    ).astype(np.float32)
    return mag_img, analysis_mask_bool, angio_3d


def main():
    p = argparse.ArgumentParser(description="Precompute classical vesselness targets from a paired train/val CSV.")
    p.add_argument("--csv", required=True, help="Input CSV with hr_u/hr_v/hr_w and optional hr_mag/mask columns.")
    p.add_argument("--out-dir", required=True, help="Directory where vesselness NIfTI targets will be written.")
    p.add_argument("--out-csv", required=True, help="Output CSV with added `vesselness_target` column.")
    p.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D NIfTI inputs.")
    p.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold used when loading the reference mask.")
    p.add_argument("--venc", type=float, default=0.90, help="VENC used for raw->velocity conversion in mag_speed mode.")
    p.add_argument(
        "--angio-mode",
        choices=["mag_only", "mag_speed"],
        default="mag_only",
        help="How to build the angio-like 3D input before Frangi/Sato.",
    )
    p.add_argument("--mag-projection-method", default="percentile", choices=["median", "max", "percentile", "topk_mean"])
    p.add_argument("--mag-projection-percentile", type=float, default=100.0)
    p.add_argument("--mag-projection-topk", type=int, default=3)
    p.add_argument("--speed-percentile", type=float, default=95.0)
    p.add_argument("--classic-sigmas", default="0.8,1.2,1.6,2.0")
    p.add_argument("--invert-uv-sign-on-raw", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Skip recomputation when the vesselness file exists.")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sigmas = parse_sigmas(args.classic_sigmas)
    base_dir = csv_path.parent

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "vesselness_target" not in fieldnames:
        fieldnames.append("vesselness_target")

    updated_rows = []
    for idx, row in enumerate(rows):
        case_id = _case_id(row, idx)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / "vesselness_target.nii.gz"

        if not (args.skip_existing and out_path.exists()):
            ref_img, analysis_mask_bool, angio_3d = _compute_angio_from_row(row, base_dir, args)
            vesselness = compute_classical_vesselness_map(
                angio_3d=angio_3d,
                analysis_mask_bool=analysis_mask_bool,
                sigmas=sigmas,
            )
            save_3d_with_ref(vesselness, ref_img, out_path, dtype=np.float32)
            print(f"[{idx + 1}/{len(rows)}] Saved vesselness target: {out_path}")
        else:
            print(f"[{idx + 1}/{len(rows)}] Reusing existing vesselness target: {out_path}")

        updated = dict(row)
        updated["vesselness_target"] = str(out_path)
        updated_rows.append(updated)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Wrote CSV with vesselness_target column: {out_csv}")


if __name__ == "__main__":
    main()
