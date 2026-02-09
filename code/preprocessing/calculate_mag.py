#!/usr/bin/env python3
"""Prepare 4D flow inputs (raw velocity + magnitude) with optional derived volumes."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

CSV_COLUMNS = {
    "case_id": "Case_ID",
    "magnitude": "Path_Mag",
    "vx": "Path_Vx",
    "vy": "Path_Vy",
    "vz": "Path_Vz",
}


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else root / path


def save_nifti_like(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), ref_img.affine, header)
    qform, qcode = ref_img.get_qform(coded=True)
    sform, scode = ref_img.get_sform(coded=True)
    if qform is not None and qcode is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None and scode is not None:
        image.set_sform(sform, int(scode))
    nib.save(image, str(out_path))


def ensure_time_last(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        return data
    raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape}")


def looks_like_raw_phase(data: np.ndarray) -> tuple[bool, bool]:
    mn = float(np.min(data))
    mx = float(np.max(data))
    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500.0) and (mx > 500.0) and (max_abs <= 8192.0) and (centered_ratio < 0.25)
    return is_unsigned_raw, is_signed_raw


def convert_raw_to_velocity_if_needed(
    data: np.ndarray,
    venc: float | None,
    invert_sign_if_raw: bool,
    label: str,
) -> tuple[np.ndarray, bool]:
    is_unsigned_raw, is_signed_raw = looks_like_raw_phase(data)
    looks_raw = is_unsigned_raw or is_signed_raw
    if not looks_raw:
        return data.astype(np.float32, copy=False), False

    if venc is None:
        raise ValueError(f"{label} appears RAW but no VENC was provided.")

    mn = float(np.min(data))
    mx = float(np.max(data))
    print(f"  [{label}] RAW-like range detected ({mn:.1f}..{mx:.1f}), converting with VENC={venc:.4f} m/s")

    out = data.astype(np.float32, copy=False)
    if is_unsigned_raw:
        out = (out - 2048.0) / 2048.0 * float(venc)
    else:
        max_abs = max(abs(float(np.min(out))), abs(float(np.max(out))))
        scale = 4096.0 if max_abs > 3000.0 else 2048.0
        out = out / scale * float(venc)

    # Match repository convention used in H5 pipeline.
    if invert_sign_if_raw:
        out = -out
    return out, True


def apply_n4_bias_to_magnitude(mag: np.ndarray, shrink_factor: int = 4) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "N4 requested but SimpleITK is not installed. Install it with: pip install SimpleITK"
        ) from exc

    if mag.ndim not in (3, 4):
        raise ValueError(f"N4 expects 3D/4D magnitude, got shape {mag.shape}")

    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    ref_3d = np.mean(mag, axis=-1) if mag.ndim == 4 else mag
    ref_3d = np.clip(ref_3d, a_min=0.0, a_max=None)

    ref_img = sitk.GetImageFromArray(ref_3d)
    mask_img = sitk.OtsuThreshold(ref_img, 0, 1, 200)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    # Some SimpleITK versions do not expose SetShrinkFactor on the filter object.
    if hasattr(n4, "SetShrinkFactor"):
        n4.SetShrinkFactor(int(shrink_factor))
    elif int(shrink_factor) != 1:
        print("  N4 note: this SimpleITK build does not support SetShrinkFactor; running without shrink.")
    _ = n4.Execute(ref_img, mask_img)
    log_bias_img = n4.GetLogBiasFieldAsImage(ref_img)
    bias_field = np.exp(sitk.GetArrayFromImage(log_bias_img)).astype(np.float32)
    bias_field = np.clip(bias_field, a_min=1e-6, a_max=None)

    if mag.ndim == 4:
        corrected = mag / bias_field[..., np.newaxis]
    else:
        corrected = mag / bias_field
    return corrected.astype(np.float32, copy=False)


def align_magnitude_to_velocity_shape(mag: np.ndarray, vel_shape: tuple[int, ...]) -> np.ndarray:
    if mag.shape == vel_shape:
        return mag

    if len(vel_shape) == 4 and mag.ndim == 3 and mag.shape == vel_shape[:3]:
        return np.repeat(mag[..., np.newaxis], vel_shape[-1], axis=-1)

    raise ValueError(
        "Shape mismatch between magnitude and velocity. "
        f"Magnitude={mag.shape}, velocity={vel_shape}"
    )


def process_case(
    vx_path: Path,
    vy_path: Path,
    vz_path: Path,
    mag_path: Path,
    output_dir: Path,
    case_id: str,
    venc_u: float | None,
    venc_v: float | None,
    venc_w: float | None,
    auto_convert_raw_phase: bool,
    compute_speed: bool,
    compute_pcmra: bool,
    apply_n4: bool,
    n4_shrink_factor: int,
) -> bool:
    print(f"\n--- Processing case: {case_id} ---")
    start = time.time()

    missing = []
    for name, path in (("Vx", vx_path), ("Vy", vy_path), ("Vz", vz_path), ("Magnitude", mag_path)):
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        print("Missing input files:")
        for item in missing:
            print(f"- {item}")
        return False

    try:
        vx_img = nib.load(str(vx_path))
        vy_img = nib.load(str(vy_path))
        vz_img = nib.load(str(vz_path))
        mag_img = nib.load(str(mag_path))
    except Exception as exc:  # pragma: no cover - I/O failure path
        print(f"Failed to read NIfTI files: {exc}")
        return False

    vx = vx_img.get_fdata(dtype=np.float32)
    vy = vy_img.get_fdata(dtype=np.float32)
    vz = vz_img.get_fdata(dtype=np.float32)
    mag = mag_img.get_fdata(dtype=np.float32)

    try:
        vx = ensure_time_last(vx)
        vy = ensure_time_last(vy)
        vz = ensure_time_last(vz)
        mag = ensure_time_last(mag)
    except ValueError as exc:
        print(f"Invalid dimension: {exc}")
        return False

    if not (vx.shape == vy.shape == vz.shape):
        print("Velocity shape mismatch detected:")
        print(f"- Vx: {vx.shape}")
        print(f"- Vy: {vy.shape}")
        print(f"- Vz: {vz.shape}")
        return False

    try:
        mag = align_magnitude_to_velocity_shape(mag, vx.shape)
    except ValueError as exc:
        print(str(exc))
        return False

    if auto_convert_raw_phase:
        try:
            vx, vx_converted = convert_raw_to_velocity_if_needed(vx, venc_u, invert_sign_if_raw=True, label="Vx")
            vy, vy_converted = convert_raw_to_velocity_if_needed(vy, venc_v, invert_sign_if_raw=True, label="Vy")
            vz, vz_converted = convert_raw_to_velocity_if_needed(vz, venc_w, invert_sign_if_raw=False, label="Vz")
        except ValueError as exc:
            print(f"RAW conversion error: {exc}")
            return False
        if not any((vx_converted, vy_converted, vz_converted)):
            print("  RAW conversion: not needed (inputs look like velocity already).")

    mag_raw_for_save = mag.astype(np.float32, copy=True)
    mag_for_pcmra = mag_raw_for_save
    if apply_n4:
        print(f"  Applying N4 bias correction to magnitude (shrink_factor={n4_shrink_factor})...")
        try:
            mag_for_pcmra = apply_n4_bias_to_magnitude(mag_raw_for_save, shrink_factor=n4_shrink_factor)
        except RuntimeError as exc:
            print(f"N4 error: {exc}")
            return False

    need_speed = compute_speed or compute_pcmra
    speed = None
    if need_speed:
        speed = np.sqrt(vx * vx + vy * vy + vz * vz).astype(np.float32, copy=False)

    pcmra = None
    if compute_pcmra:
        assert speed is not None  # guaranteed by need_speed
        pcmra = (mag_for_pcmra * speed).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save velocity components as loaded (RAW phase unless conversion is explicitly requested).
    save_nifti_like(vx, vx_img, output_dir / "Vx.nii.gz")
    save_nifti_like(vy, vy_img, output_dir / "Vy.nii.gz")
    save_nifti_like(vz, vz_img, output_dir / "Vz.nii.gz")
    save_nifti_like(mag_raw_for_save, mag_img, output_dir / "input_mag_raw.nii.gz")
    if apply_n4:
        save_nifti_like(mag_for_pcmra, mag_img, output_dir / "input_mag_n4.nii.gz")
    if compute_speed and speed is not None:
        save_nifti_like(speed, vx_img, output_dir / "input_speed_raw.nii.gz")
    if compute_pcmra and pcmra is not None:
        save_nifti_like(pcmra, vx_img, output_dir / "input_pcmra_raw.nii.gz")

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare magnitude + velocity volumes from dataset.csv, with optional speed/PC-MRA"
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to dataset CSV")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        type=Path,
        help="Root folder used to resolve relative paths inside the CSV",
    )
    parser.add_argument(
        "--output-subdir",
        default="processed_inputs",
        help="Output subdirectory inside --data-root",
    )
    parser.add_argument("--venc", type=float, default=0.90, help="Default VENC in m/s")
    parser.add_argument("--venc-u", type=float, default=None, help="VENC for Vx/U in m/s")
    parser.add_argument("--venc-v", type=float, default=None, help="VENC for Vy/V in m/s")
    parser.add_argument("--venc-w", type=float, default=None, help="VENC for Vz/W in m/s")
    parser.add_argument(
        "--compute-speed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute and save input_speed_raw.nii.gz",
    )
    parser.add_argument(
        "--compute-pcmra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute and save input_pcmra_raw.nii.gz",
    )
    parser.add_argument(
        "--auto-convert-raw-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-detect RAW phase and convert to velocity before optional speed/PC-MRA computation",
    )
    parser.add_argument(
        "--apply-n4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply N4 bias-field correction to magnitude before PC-MRA",
    )
    parser.add_argument("--n4-shrink-factor", type=int, default=4, help="N4 shrink factor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        return

    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        required = [CSV_COLUMNS[key] for key in CSV_COLUMNS]
        missing_columns = [name for name in required if name not in (reader.fieldnames or [])]
        if missing_columns:
            print(f"Missing CSV columns: {missing_columns}")
            return

        rows = list(reader)

    total = len(rows)
    success = 0
    failed = 0
    print(f"Found {total} cases")

    output_root = args.data_root / args.output_subdir
    venc_u = args.venc_u if args.venc_u is not None else args.venc
    venc_v = args.venc_v if args.venc_v is not None else args.venc
    venc_w = args.venc_w if args.venc_w is not None else args.venc

    for index, row in enumerate(rows, start=1):
        print(f"Progress [{index}/{total}]")
        case_id = row[CSV_COLUMNS["case_id"]].strip()

        out_dir = output_root / case_id
        ok = process_case(
            vx_path=resolve_path(args.data_root, row[CSV_COLUMNS["vx"]]),
            vy_path=resolve_path(args.data_root, row[CSV_COLUMNS["vy"]]),
            vz_path=resolve_path(args.data_root, row[CSV_COLUMNS["vz"]]),
            mag_path=resolve_path(args.data_root, row[CSV_COLUMNS["magnitude"]]),
            output_dir=out_dir,
            case_id=case_id,
            venc_u=venc_u,
            venc_v=venc_v,
            venc_w=venc_w,
            auto_convert_raw_phase=args.auto_convert_raw_phase,
            compute_speed=args.compute_speed,
            compute_pcmra=args.compute_pcmra,
            apply_n4=args.apply_n4,
            n4_shrink_factor=args.n4_shrink_factor,
        )

        if ok:
            success += 1
        else:
            failed += 1

    print("\n--- Summary ---")
    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
