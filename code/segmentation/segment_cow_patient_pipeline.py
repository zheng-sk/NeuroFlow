#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, generate_binary_structure, iterate_structure, label

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NNUNET_RAW = REPO_ROOT / "nnUNet_raw"
DEFAULT_NNUNET_PREPROCESSED = REPO_ROOT / "nnUNet_preprocessed"
DEFAULT_NNUNET_RESULTS = REPO_ROOT / "models"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "topcow-claim-models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cow_segmentation_patient"
LOCAL_NNUNET_REPO = REPO_ROOT / "topcow-2024-nnunet"

# Configure nnU-Net paths before importing nnunetv2.
os.environ.setdefault("nnUNet_raw", str(DEFAULT_NNUNET_RAW))
os.environ.setdefault("nnUNet_preprocessed", str(DEFAULT_NNUNET_PREPROCESSED))
os.environ.setdefault("nnUNet_results", str(DEFAULT_NNUNET_RESULTS))

try:
    from nnunetv2.ensembling.ensemble import ensemble_folders
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ModuleNotFoundError:
    if (LOCAL_NNUNET_REPO / "nnunetv2").exists():
        sys.path.insert(0, str(LOCAL_NNUNET_REPO))
        from nnunetv2.ensembling.ensemble import ensemble_folders
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    else:
        raise ModuleNotFoundError(
            "nnunetv2 not found. Install with: pip install -e ./topcow-2024-nnunet "
            "or keep topcow-2024-nnunet/ in the repository root."
        )

OUTPUT_COW_LABEL = 1


def get_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def ensure_time_last(data: np.ndarray, time_axis: int = -1) -> np.ndarray:
    if data.ndim == 3:
        return data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D/4D, got {data.ndim}D with shape={data.shape}")

    resolved = time_axis if time_axis >= 0 else data.ndim + time_axis
    if resolved < 0 or resolved >= data.ndim:
        raise ValueError(f"Invalid time_axis={time_axis} for shape={data.shape}")
    if resolved != 3:
        data = np.moveaxis(data, resolved, 3)
    return data


def convert_raw_to_velocity_repo_style(data: np.ndarray, venc: float, invert_sign: bool = False) -> np.ndarray:
    arr = data.astype(np.float32, copy=False)
    mn, mx = float(np.min(arr)), float(np.max(arr))

    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500.0) and (mx > 500.0) and (max_abs <= 8192.0) and (centered_ratio < 0.25)

    converted = arr
    if (is_unsigned_raw or is_signed_raw) and (venc is not None):
        if is_unsigned_raw:
            converted = (arr - 2048.0) / 2048.0 * float(venc)
        else:
            scale = 4096.0 if max_abs > 3000.0 else 2048.0
            converted = arr / scale * float(venc)
        if invert_sign:
            converted = -converted

    return converted.astype(np.float32, copy=False)


def robust_norm_in_mask(vol: np.ndarray, mask3d: np.ndarray, p_lo: float = 1, p_hi: float = 99, eps: float = 1e-8) -> np.ndarray:
    vals = vol[mask3d > 0]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip(vol, lo, hi)
    out = (clipped - lo) / (hi - lo + eps)
    out = out * mask3d
    return out.astype(np.float32)


def save_3d_with_ref(vol: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path, dtype=np.float32) -> None:
    header = ref_img.header.copy()
    header.set_data_shape(vol.shape)
    header.set_zooms(ref_img.header.get_zooms()[:3])
    out_img = nib.Nifti1Image(vol.astype(dtype), ref_img.affine, header)
    nib.save(out_img, str(out_path))


def load_mask_aligned(mask_path: Path, ref_img: nib.Nifti1Image, threshold: float) -> np.ndarray:
    mask_img = nib.load(str(mask_path))
    if len(mask_img.shape) == 4:
        mask_img = nib.Nifti1Image(
            np.asarray(mask_img.dataobj[..., 0], dtype=np.float32), mask_img.affine, mask_img.header
        )

    ref_space = (ref_img.shape[:3], ref_img.affine)
    same_shape = mask_img.shape[:3] == ref_img.shape[:3]
    same_affine = np.allclose(mask_img.affine, ref_img.affine, atol=1e-4)
    if not (same_shape and same_affine):
        mask_img = nibproc.resample_from_to(mask_img, ref_space, order=0)

    return (np.asarray(mask_img.dataobj, dtype=np.float32) > float(threshold)).astype(np.float32)


def parse_sigmas(sigmas_str: str):
    values = [float(x.strip()) for x in sigmas_str.split(",") if x.strip()]
    if len(values) == 0:
        raise ValueError("--classic-sigmas must contain at least one value, e.g. 1,2,3,4")
    return values


def temporal_projection(vol4d: np.ndarray, method: str = "max", percentile: float = 95.0, topk: int = 3) -> np.ndarray:
    if vol4d.ndim != 4:
        raise ValueError(f"Expected 4D volume for temporal projection, got shape={vol4d.shape}")
    if method == "median":
        return np.median(vol4d, axis=3)
    if method == "max":
        return np.max(vol4d, axis=3)
    if method == "percentile":
        return np.percentile(vol4d, percentile, axis=3)
    if method == "topk_mean":
        k = int(max(1, min(topk, vol4d.shape[3])))
        part = np.partition(vol4d, vol4d.shape[3] - k, axis=3)
        topk_vals = part[..., -k:]
        return np.mean(topk_vals, axis=3)
    raise ValueError(f"Unsupported temporal projection method: {method}")


def remove_small_components(mask: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool)

    structure = generate_binary_structure(3, connectivity)
    labeled, num = label(mask.astype(bool), structure=structure)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    keep_ids = np.where(counts >= int(min_size))[0]
    keep_ids = keep_ids[keep_ids != 0]
    if len(keep_ids) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_ids)


def filter_components_touching_roi(mask: np.ndarray, roi_mask: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
    structure = generate_binary_structure(3, connectivity)
    labeled, num = label(mask.astype(bool), structure=structure)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    keep_ids = []
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        if int(np.sum(comp)) < int(min_size):
            continue
        if np.any(comp & roi_mask):
            keep_ids.append(comp_id)

    if len(keep_ids) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_ids)


def compute_angiography_from_patient(
    patient_dir: Path,
    mag_name: str,
    vx_name: str,
    vy_name: str,
    vz_name: str,
    venc: float,
    time_axis: int,
    mask_path: Path | None,
    mask_threshold: float,
    speed_percentile: float,
    angio_mode: str,
    mag_projection_method: str,
    mag_projection_percentile: float,
    mag_projection_topk: int,
    invert_uv_sign_on_raw: bool,
):
    mag_path = patient_dir / mag_name
    if not mag_path.exists():
        raise FileNotFoundError(f"Missing required file: {mag_path}")

    mag_img = nib.load(str(mag_path))
    mag4d = ensure_time_last(np.asarray(mag_img.dataobj, dtype=np.float32), time_axis)

    if mask_path is not None:
        analysis_mask = load_mask_aligned(mask_path, mag_img, mask_threshold)
    else:
        analysis_mask = np.ones(mag4d.shape[:3], dtype=np.float32)
    analysis_mask_bool = analysis_mask > 0

    if angio_mode == "mag_only":
        mag_proj = temporal_projection(
            mag4d,
            method=mag_projection_method,
            percentile=mag_projection_percentile,
            topk=mag_projection_topk,
        ).astype(np.float32)
        angio_3d = robust_norm_in_mask(mag_proj, analysis_mask).astype(np.float32)
    elif angio_mode == "mag_speed":
        vx_path = patient_dir / vx_name
        vy_path = patient_dir / vy_name
        vz_path = patient_dir / vz_name
        for p in [vx_path, vy_path, vz_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file for mag_speed mode: {p}")

        vx_img = nib.load(str(vx_path))
        vy_img = nib.load(str(vy_path))
        vz_img = nib.load(str(vz_path))
        vx4d_raw = ensure_time_last(np.asarray(vx_img.dataobj, dtype=np.float32), time_axis)
        vy4d_raw = ensure_time_last(np.asarray(vy_img.dataobj, dtype=np.float32), time_axis)
        vz4d_raw = ensure_time_last(np.asarray(vz_img.dataobj, dtype=np.float32), time_axis)

        if not (mag4d.shape == vx4d_raw.shape == vy4d_raw.shape == vz4d_raw.shape):
            raise ValueError(
                "Shape mismatch:\n"
                f"MAG {mag4d.shape}\n"
                f"Vx  {vx4d_raw.shape}\n"
                f"Vy  {vy4d_raw.shape}\n"
                f"Vz  {vz4d_raw.shape}"
            )

        # Keep repository sign convention from upstream DICOM->NIfTI; no extra CoW-stage U/V flip.
        vx4d = convert_raw_to_velocity_repo_style(vx4d_raw, venc, invert_sign=invert_uv_sign_on_raw)
        vy4d = convert_raw_to_velocity_repo_style(vy4d_raw, venc, invert_sign=invert_uv_sign_on_raw)
        vz4d = convert_raw_to_velocity_repo_style(vz4d_raw, venc, invert_sign=False)
        speed4d = np.sqrt(vx4d * vx4d + vy4d * vy4d + vz4d * vz4d).astype(np.float32)

        mag_med = np.median(mag4d, axis=3).astype(np.float32)
        speed_proj = np.percentile(speed4d, speed_percentile, axis=3).astype(np.float32)
        angio_3d = (
            robust_norm_in_mask(mag_med, analysis_mask) * robust_norm_in_mask(speed_proj, analysis_mask)
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported --angio-mode: {angio_mode}")

    return {
        "mag_img": mag_img,
        "mag_shape_4d": mag4d.shape,
        "analysis_mask": analysis_mask,
        "analysis_mask_bool": analysis_mask_bool,
        "angio_3d": angio_3d,
    }


def classical_vesselness_cow_segmentation(
    angio_3d: np.ndarray,
    analysis_mask_bool: np.ndarray,
    sigmas,
    percentile_threshold: float,
    morph_radius: int,
    use_morph_open: bool,
    use_morph_close: bool,
    min_component_size: int,
    z_min_frac: float,
    z_max_frac: float,
):
    try:
        from skimage.filters import frangi, sato
    except ImportError as exc:
        raise ImportError(
            "Classic vesselness segmentation needs scikit-image (frangi/sato). Install with: pip install scikit-image"
        ) from exc

    input_vol = angio_3d.astype(np.float32, copy=False)
    mask = analysis_mask_bool.astype(bool)

    input_n = robust_norm_in_mask(input_vol, mask.astype(np.float32))
    frangi_resp = frangi(input_n, sigmas=sigmas, black_ridges=False)
    sato_resp = sato(input_n, sigmas=sigmas, black_ridges=False)

    frangi_n = robust_norm_in_mask(frangi_resp.astype(np.float32), mask.astype(np.float32))
    sato_n = robust_norm_in_mask(sato_resp.astype(np.float32), mask.astype(np.float32))
    vesselness = ((frangi_n + sato_n) * 0.5).astype(np.float32)

    valid = mask & np.isfinite(vesselness)
    if not np.any(valid):
        return np.zeros_like(input_vol, dtype=np.uint8), {"threshold": None, "voxels": 0}

    threshold = float(np.percentile(vesselness[valid], percentile_threshold))
    bin_mask = (vesselness >= threshold) & mask

    structure = generate_binary_structure(3, 1)
    if morph_radius > 0:
        morph_structure = iterate_structure(structure, morph_radius)
        if use_morph_open:
            bin_mask = binary_opening(bin_mask, structure=morph_structure)
        if use_morph_close:
            bin_mask = binary_closing(bin_mask, structure=morph_structure)

    cleaned = remove_small_components(bin_mask, min_size=min_component_size, connectivity=1)

    shape = cleaned.shape
    z0 = int(shape[2] * np.clip(z_min_frac, 0.0, 1.0))
    z1 = int(shape[2] * np.clip(z_max_frac, 0.0, 1.0))
    if z1 <= z0:
        z0, z1 = 0, shape[2]

    roi_mask = np.zeros(shape, dtype=bool)
    roi_mask[:, :, z0:z1] = True
    roi_mask &= mask
    kept = filter_components_touching_roi(cleaned, roi_mask, min_size=min_component_size, connectivity=1)

    out = kept if np.any(kept) else cleaned
    return out.astype(np.uint8), {"threshold": threshold, "voxels": int(np.sum(out))}


def ensemble_binary_predictions(ai_mask: np.ndarray, classic_mask: np.ndarray, mode: str) -> np.ndarray:
    ai = ai_mask > 0
    classic = classic_mask > 0

    if mode == "union":
        return (ai | classic).astype(np.uint8)
    if mode == "intersection":
        return (ai & classic).astype(np.uint8)
    if mode == "ai":
        return ai.astype(np.uint8)
    if mode == "classic":
        return classic.astype(np.uint8)
    raise ValueError(f"Unsupported ensemble mode: {mode}")


def postprocess_binary_cow_mask(
    segmentation: np.ndarray,
    cow_label: int = OUTPUT_COW_LABEL,
    closing_radius: int = 1,
    opening_radius: int = 0,
    fill_holes: bool = True,
    min_component_size: int = 0,
) -> np.ndarray:
    mask = segmentation > 0
    structure = generate_binary_structure(3, 1)

    if closing_radius > 0:
        close_structure = iterate_structure(structure, closing_radius)
        mask = binary_closing(mask, structure=close_structure)
    if opening_radius > 0:
        open_structure = iterate_structure(structure, opening_radius)
        mask = binary_opening(mask, structure=open_structure)
    if fill_holes:
        mask = binary_fill_holes(mask)
    if min_component_size > 0:
        mask = remove_small_components(mask, min_size=min_component_size, connectivity=1)

    out = np.zeros_like(segmentation, dtype=np.uint8)
    out[mask] = np.uint8(cow_label)
    return out


def run_prediction(model_dir, input_folder, output_folder, checkpoint_name, device):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name=checkpoint_name,
    )
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,
    )


def predict_ai_mask_from_angio(angio_3d: np.ndarray, ref_img: nib.Nifti1Image, case_id: str, model_dir: str, tmp_root: Path):
    tmp_input = tmp_root / "nnunet_input"
    tmp_best = tmp_root / "pred_best"
    tmp_final = tmp_root / "pred_final"
    tmp_ens = tmp_root / "pred_ens"
    for d in [tmp_input, tmp_best, tmp_final, tmp_ens]:
        d.mkdir(parents=True, exist_ok=True)

    in_file = tmp_input / f"{case_id}_0000.nii.gz"
    save_3d_with_ref(angio_3d, ref_img, in_file, dtype=np.float32)

    device = get_torch_device()
    print(f"Using device: {device}")

    run_prediction(model_dir, str(tmp_input), str(tmp_best), "checkpoint_best.pth", device)
    run_prediction(model_dir, str(tmp_input), str(tmp_final), "checkpoint_final.pth", device)
    ensemble_folders([str(tmp_best), str(tmp_final)], str(tmp_ens), num_processes=1)

    pred_file = tmp_ens / f"{case_id}.nii.gz"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")

    pred_data = np.round(nib.load(str(pred_file)).get_fdata()).astype(np.uint8)
    ai_mask = np.zeros_like(pred_data, dtype=np.uint8)
    ai_mask[pred_data > 0] = np.uint8(OUTPUT_COW_LABEL)
    return ai_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 3D angiography from patient folder, then run AI + classical CoW segmentation, ensemble, and postprocessing."
    )
    parser.add_argument("--patient-dir", type=Path, required=True, help="Folder containing all required patient images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root folder.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Path to nnU-Net trained model folder.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary folder for debugging.")
    parser.add_argument("--save-intermediates", action="store_true", help="Save intermediate masks/maps.")

    parser.add_argument("--mag-name", default="input_mag_raw.nii.gz", help="MAG filename inside patient folder.")
    parser.add_argument("--vx-name", default="Vx.nii.gz", help="Vx filename inside patient folder.")
    parser.add_argument("--vy-name", default="Vy.nii.gz", help="Vy filename inside patient folder.")
    parser.add_argument("--vz-name", default="Vz.nii.gz", help="Vz filename inside patient folder.")
    parser.add_argument("--venc", type=float, default=0.90, help="VENC in m/s.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis in input NIfTI.")
    parser.add_argument("--mask-path", type=Path, default=None, help="Optional external mask path.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument(
        "--angio-mode",
        choices=["mag_speed", "mag_only"],
        default="mag_speed",
        help="How to build the angio-like 3D volume: MAG*speed proxy (default) or MAG-only temporal projection.",
    )
    parser.add_argument("--speed-percentile", type=float, default=90.0, help="Percentile over speed(t) for angiography.")
    parser.add_argument(
        "--mag-projection-method",
        choices=["median", "max", "percentile", "topk_mean"],
        default="percentile",
        help="Temporal projection method for MAG when --angio-mode mag_only.",
    )
    parser.add_argument(
        "--mag-projection-percentile",
        type=float,
        default=95.0,
        help="Percentile used when --mag-projection-method percentile.",
    )
    parser.add_argument(
        "--mag-projection-topk",
        type=int,
        default=3,
        help="Top-k used when --mag-projection-method topk_mean.",
    )
    parser.add_argument(
        "--legacy-invert-uv-sign-on-raw",
        action="store_true",
        help=(
            "Legacy mode: invert U/V signs during RAW->velocity conversion "
            "(matches older CoW patient pipeline behavior)."
        ),
    )

    parser.add_argument("--no-classic-cow", action="store_true", help="Disable classic vesselness branch.")
    parser.add_argument(
        "--classic-only",
        action="store_true",
        help="Skip AI inference and use only the classical vesselness branch.",
    )
    parser.add_argument("--classic-sigmas", default="1,2,3,4", help="Comma-separated sigmas for frangi/sato.")
    parser.add_argument("--classic-percentile", type=float, default=95.0, help="Percentile threshold over vesselness.")
    parser.add_argument("--classic-morph-radius", type=int, default=1, help="Morphology radius for classic branch.")
    parser.add_argument("--classic-use-morph-open", action="store_true", help="Enable opening in classic branch.")
    parser.add_argument("--classic-no-morph-close", action="store_true", help="Disable closing in classic branch.")
    parser.add_argument("--classic-min-component-size", type=int, default=80, help="Min component size in classic branch.")
    parser.add_argument("--classic-z-min-frac", type=float, default=0.15, help="Lower z fraction for classic ROI filtering.")
    parser.add_argument("--classic-z-max-frac", type=float, default=0.95, help="Upper z fraction for classic ROI filtering.")

    parser.add_argument("--ensemble-mode", choices=["union", "intersection", "ai", "classic"], default="union")

    parser.add_argument("--no-postprocess", action="store_true", help="Disable final postprocessing.")
    parser.add_argument("--post-close-radius", type=int, default=1, help="Final closing radius.")
    parser.add_argument("--post-open-radius", type=int, default=0, help="Final opening radius.")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable fill holes in final mask.")
    parser.add_argument("--post-min-component-size", type=int, default=30, help="Remove tiny final components.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.classic_only and args.no_classic_cow:
        raise ValueError("--classic-only and --no-classic-cow are mutually exclusive.")
    patient_dir = args.patient_dir
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    case_id = patient_dir.name
    case_out = args.output_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    classic_sigmas = parse_sigmas(args.classic_sigmas)

    prep = compute_angiography_from_patient(
        patient_dir=patient_dir,
        mag_name=args.mag_name,
        vx_name=args.vx_name,
        vy_name=args.vy_name,
        vz_name=args.vz_name,
        venc=args.venc,
        time_axis=args.time_axis,
        mask_path=args.mask_path,
        mask_threshold=args.mask_threshold,
        speed_percentile=args.speed_percentile,
        angio_mode=args.angio_mode,
        mag_projection_method=args.mag_projection_method,
        mag_projection_percentile=args.mag_projection_percentile,
        mag_projection_topk=args.mag_projection_topk,
        invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
    )

    ref_img = prep["mag_img"]
    angio_3d = prep["angio_3d"]
    analysis_mask_bool = prep["analysis_mask_bool"]

    tmp_root = case_out / "_tmp_nnunet"
    if args.classic_only:
        ai_mask = np.zeros(angio_3d.shape, dtype=np.uint8)
        print("Skipping AI inference (--classic-only).")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        ai_mask = predict_ai_mask_from_angio(angio_3d, ref_img, case_id, args.model_dir, tmp_root)

    if args.no_classic_cow:
        classic_mask = np.zeros_like(ai_mask, dtype=np.uint8)
        classic_info = {"threshold": None, "voxels": 0}
        effective_mode = "ai"
    else:
        classic_mask, classic_info = classical_vesselness_cow_segmentation(
            angio_3d=angio_3d,
            analysis_mask_bool=analysis_mask_bool,
            sigmas=classic_sigmas,
            percentile_threshold=args.classic_percentile,
            morph_radius=max(0, args.classic_morph_radius),
            use_morph_open=args.classic_use_morph_open,
            use_morph_close=not args.classic_no_morph_close,
            min_component_size=max(0, args.classic_min_component_size),
            z_min_frac=args.classic_z_min_frac,
            z_max_frac=args.classic_z_max_frac,
        )
        effective_mode = "classic" if args.classic_only else args.ensemble_mode

    ensemble_mask = ensemble_binary_predictions(ai_mask, classic_mask, effective_mode)
    final_mask = ensemble_mask
    if not args.no_postprocess:
        final_mask = postprocess_binary_cow_mask(
            final_mask,
            cow_label=OUTPUT_COW_LABEL,
            closing_radius=max(0, args.post_close_radius),
            opening_radius=max(0, args.post_open_radius),
            fill_holes=not args.no_fill_holes,
            min_component_size=max(0, args.post_min_component_size),
        )

    save_3d_with_ref(final_mask, ref_img, case_out / "cow_seg_final.nii.gz", dtype=np.uint8)

    if args.save_intermediates:
        save_3d_with_ref(angio_3d, ref_img, case_out / "angio_3d.nii.gz", dtype=np.float32)
        save_3d_with_ref(prep["analysis_mask"], ref_img, case_out / "analysis_mask.nii.gz", dtype=np.float32)
        save_3d_with_ref(ai_mask, ref_img, case_out / "cow_seg_ai.nii.gz", dtype=np.uint8)
        save_3d_with_ref(classic_mask, ref_img, case_out / "cow_seg_classic.nii.gz", dtype=np.uint8)
        save_3d_with_ref(ensemble_mask, ref_img, case_out / "cow_seg_ensemble.nii.gz", dtype=np.uint8)

    if (not args.classic_only) and (not args.keep_temp):
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("Case:", case_id)
    print("Input:", patient_dir.resolve())
    print("Output:", case_out.resolve())
    print("Shape (x,y,z,t):", prep["mag_shape_4d"])
    print("Mask voxels:", int(np.sum(analysis_mask_bool)))
    print("Angio mode:", args.angio_mode)
    print("Legacy invert U/V on RAW:", bool(args.legacy_invert_uv_sign_on_raw))
    print("Classic threshold:", classic_info["threshold"])
    print("AI voxels:", int(np.sum(ai_mask > 0)))
    print("Classic voxels:", int(classic_info["voxels"]))
    print(f"Ensemble mode: {effective_mode}")
    print("Final voxels:", int(np.sum(final_mask > 0)))


if __name__ == "__main__":
    main()
