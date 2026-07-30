import argparse
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, generate_binary_structure, iterate_structure, label

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NNUNET_RAW = REPO_ROOT / "nnUNet_raw"
DEFAULT_NNUNET_PREPROCESSED = REPO_ROOT / "nnUNet_preprocessed"
DEFAULT_NNUNET_RESULTS = REPO_ROOT / "models"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "topcow-claim-models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cow_segmentation"
DEFAULT_TEMP_ROOT = REPO_ROOT / "data" / "tmp" / "crop_seg"

# Configure nnU-Net paths before importing nnunetv2.
os.environ.setdefault("nnUNet_raw", str(DEFAULT_NNUNET_RAW))
os.environ.setdefault("nnUNet_preprocessed", str(DEFAULT_NNUNET_PREPROCESSED))
os.environ.setdefault("nnUNet_results", str(DEFAULT_NNUNET_RESULTS))

try:
    from nnunetv2.ensembling.ensemble import ensemble_folders
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ModuleNotFoundError as exc:  # pragma: no cover - install-time guard
    raise ModuleNotFoundError(
        "nnunetv2 is required for Circle-of-Willis segmentation. Install the "
        'pinned version with:  pip install -e ".[segmentation]"  (or '
        "pip install nnunetv2==2.5.1). See requirements.txt."
    ) from exc

OUTPUT_BINARY_COW = True
OUTPUT_COW_LABEL = 1


def get_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data, img


def save_nifti(data, output_path, reference_img):
    header = reference_img.header.copy()
    if tuple(header.get_data_shape()) != tuple(data.shape):
        header.set_data_shape(data.shape)
    out = nib.Nifti1Image(data, reference_img.affine, header)
    nib.save(out, output_path)


def parse_sigmas(sigmas_str: str):
    values = [float(x.strip()) for x in sigmas_str.split(",") if x.strip()]
    if len(values) == 0:
        raise ValueError("--classic-sigmas must contain at least one value, e.g. 1,2,3")
    return values


def temporal_projection(vol4d: np.ndarray, method: str = "max", percentile: float = 95.0, topk: int = 3) -> np.ndarray:
    if method == "max":
        return np.max(vol4d, axis=3)
    if method == "percentile":
        return np.percentile(vol4d, percentile, axis=3)
    if method == "topk_mean":
        k = int(np.clip(topk, 1, vol4d.shape[3]))
        sorted_vol = np.sort(vol4d, axis=3)
        return np.mean(sorted_vol[..., -k:], axis=3)
    raise ValueError(f"Unsupported projection method: {method}")


def collect_nifti_files(path_str, recursive=False):
    p = Path(path_str)
    if p.is_file() and str(p).endswith(".nii.gz"):
        return [str(p)]
    if p.is_dir():
        pattern = "**/*.nii.gz" if recursive else "*.nii.gz"
        return sorted([str(x) for x in p.glob(pattern)])
    raise FileNotFoundError(f"Input path not found: {path_str}")


def ensure_3d_input(input_file, temp_dir, projection_method="max", projection_percentile=95.0, projection_topk=3):
    data, img = load_nifti(input_file)
    if data.ndim == 3:
        return input_file
    if data.ndim == 4:
        proj = temporal_projection(
            data,
            method=projection_method,
            percentile=projection_percentile,
            topk=projection_topk,
        )
        base = os.path.basename(input_file).replace(".nii.gz", "")
        out = os.path.join(temp_dir, f"{base}_proj3d.nii.gz")
        save_nifti(proj, out, img)
        return out
    raise ValueError(f"Unsupported shape for {input_file}: {data.shape}")


def build_nnunet_case_id(path_str):
    base = os.path.basename(path_str).replace(".nii.gz", "")
    return base.replace(".", "_")


def collapse_to_single_cow_class(segmentation: np.ndarray, cow_label: int = 1) -> np.ndarray:
    out = np.zeros_like(segmentation, dtype=np.uint8)
    out[segmentation > 0] = np.uint8(cow_label)
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


def robust_norm_in_mask(vol, mask3d, p_lo=1, p_hi=99, eps=1e-8):
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


def classical_vesselness_cow_segmentation(
    angio_3d: np.ndarray,
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
            "Classic vesselness segmentation needs scikit-image (frangi/sato). "
            "Install with: pip install scikit-image"
        ) from exc

    input_vol = angio_3d.astype(np.float32, copy=False)
    mask = np.ones_like(input_vol, dtype=bool)

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
    kept = filter_components_touching_roi(cleaned, roi_mask, min_size=min_component_size, connectivity=1)

    if np.any(kept):
        out = kept
    else:
        print("Warning: classic vesselness found no ROI-connected component; using cleaned mask fallback.")
        out = cleaned

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
    cow_label: int = 1,
    closing_radius: int = 1,
    opening_radius: int = 0,
    fill_holes: bool = True,
    min_component_size: int = 0,
) -> np.ndarray:
    mask = segmentation > 0
    structure = generate_binary_structure(3, 1)  # 6-connectivity

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


def main():
    parser = argparse.ArgumentParser(description="Segment CoW from precomputed crop volumes.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a crop .nii.gz or folder with crop .nii.gz files.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to nnU-Net trained model folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for final crop segmentations.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary folders for debugging.",
    )
    parser.add_argument(
        "--temp-root",
        default=str(DEFAULT_TEMP_ROOT),
        help="Temporary root folder used for nnU-Net staging files.",
    )
    parser.add_argument(
        "--projection-method",
        choices=["max", "percentile", "topk_mean"],
        default="max",
        help="How to convert 4D angiography input to 3D.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .nii.gz files when --input is a directory.",
    )
    parser.add_argument(
        "--projection-percentile",
        type=float,
        default=95.0,
        help="Percentile used when --projection-method percentile.",
    )
    parser.add_argument(
        "--projection-topk",
        type=int,
        default=3,
        help="Top-k used when --projection-method topk_mean.",
    )
    parser.add_argument(
        "--no-classic-cow",
        action="store_true",
        help="Disable classical vesselness CoW branch and use AI-only mask.",
    )
    parser.add_argument(
        "--classic-sigmas",
        default="1,2,3,4",
        help="Comma-separated sigma values for frangi/sato (e.g. 1,2,3,4).",
    )
    parser.add_argument(
        "--classic-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold over vesselness map.",
    )
    parser.add_argument(
        "--classic-morph-radius",
        type=int,
        default=1,
        help="Radius for classic morphology operations.",
    )
    parser.add_argument(
        "--classic-use-morph-open",
        action="store_true",
        help="Enable opening in classic vesselness branch.",
    )
    parser.add_argument(
        "--classic-no-morph-close",
        action="store_true",
        help="Disable closing in classic vesselness branch.",
    )
    parser.add_argument(
        "--classic-min-component-size",
        type=int,
        default=80,
        help="Minimum component size for classic vesselness branch.",
    )
    parser.add_argument(
        "--classic-z-min-frac",
        type=float,
        default=0.15,
        help="Lower z fraction for ROI-connected component filtering.",
    )
    parser.add_argument(
        "--classic-z-max-frac",
        type=float,
        default=0.95,
        help="Upper z fraction for ROI-connected component filtering.",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=["union", "intersection", "ai", "classic"],
        default="union",
        help="How to combine AI and classical masks.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable final binary CoW mask postprocessing.",
    )
    parser.add_argument(
        "--post-close-radius",
        type=int,
        default=1,
        help="Final 3D closing radius (joins short gaps).",
    )
    parser.add_argument(
        "--post-open-radius",
        type=int,
        default=0,
        help="Final 3D opening radius (removes thin noisy protrusions).",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Disable final 3D hole filling.",
    )
    parser.add_argument(
        "--post-min-component-size",
        type=int,
        default=30,
        help="Final minimum connected component size.",
    )
    args = parser.parse_args()

    classic_sigmas = parse_sigmas(args.classic_sigmas)

    os.makedirs(args.output_dir, exist_ok=True)

    tmp_root = str(args.temp_root)
    tmp_input = os.path.join(tmp_root, "nnunet_input")
    tmp_best = os.path.join(tmp_root, "pred_best")
    tmp_final = os.path.join(tmp_root, "pred_final")
    tmp_ens = os.path.join(tmp_root, "pred_ens")
    tmp_proj = os.path.join(tmp_root, "proj")

    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_input, exist_ok=True)
    os.makedirs(tmp_best, exist_ok=True)
    os.makedirs(tmp_final, exist_ok=True)
    os.makedirs(tmp_ens, exist_ok=True)
    os.makedirs(tmp_proj, exist_ok=True)

    files = collect_nifti_files(args.input, recursive=args.recursive)
    if len(files) == 0:
        hint = " Try --recursive if your cases are in nested folders." if not args.recursive else ""
        raise RuntimeError(f"No .nii.gz files found in input.{hint}")

    # Map nnUNet case ids back to original filenames and projected inputs
    case_to_original = {}
    case_to_prepared = {}
    for f in files:
        prepared = ensure_3d_input(
            f,
            tmp_proj,
            projection_method=args.projection_method,
            projection_percentile=args.projection_percentile,
            projection_topk=args.projection_topk,
        )
        case_id = build_nnunet_case_id(f)
        case_to_original[case_id] = f
        case_to_prepared[case_id] = prepared
        dst = os.path.join(tmp_input, f"{case_id}_0000.nii.gz")
        shutil.copy2(prepared, dst)

    device = get_torch_device()
    print(f"Using device: {device}")

    run_prediction(args.model_dir, tmp_input, tmp_best, "checkpoint_best.pth", device)
    run_prediction(args.model_dir, tmp_input, tmp_final, "checkpoint_final.pth", device)
    ensemble_folders([tmp_best, tmp_final], tmp_ens, num_processes=1)

    # Export final segmentation with original crop names
    for case_id, original_path in case_to_original.items():
        pred_file = os.path.join(tmp_ens, f"{case_id}.nii.gz")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Missing prediction file: {pred_file}")

        pred_img = nib.load(pred_file)
        pred_data = np.round(pred_img.get_fdata()).astype(np.uint8)

        if OUTPUT_BINARY_COW:
            ai_mask = collapse_to_single_cow_class(pred_data, OUTPUT_COW_LABEL)

            if args.no_classic_cow:
                classic_mask = np.zeros_like(ai_mask, dtype=np.uint8)
                classic_info = {"threshold": None, "voxels": 0}
                effective_mode = "ai"
            else:
                angio_3d, _ = load_nifti(case_to_prepared[case_id])
                if angio_3d.shape != ai_mask.shape:
                    raise ValueError(
                        f"Shape mismatch for case {case_id}: classic input {angio_3d.shape} vs AI mask {ai_mask.shape}"
                    )

                classic_mask, classic_info = classical_vesselness_cow_segmentation(
                    angio_3d,
                    sigmas=classic_sigmas,
                    percentile_threshold=args.classic_percentile,
                    morph_radius=max(0, args.classic_morph_radius),
                    use_morph_open=args.classic_use_morph_open,
                    use_morph_close=not args.classic_no_morph_close,
                    min_component_size=max(0, args.classic_min_component_size),
                    z_min_frac=args.classic_z_min_frac,
                    z_max_frac=args.classic_z_max_frac,
                )
                effective_mode = args.ensemble_mode

            pred_data = ensemble_binary_predictions(ai_mask, classic_mask, effective_mode)
            print(
                f"{case_id} voxels -> AI: {int(np.sum(ai_mask > 0))}, "
                f"classic: {classic_info['voxels']}, "
                f"ensemble({effective_mode}): {int(np.sum(pred_data > 0))}"
            )

            if not args.no_postprocess:
                pred_data = postprocess_binary_cow_mask(
                    pred_data,
                    cow_label=OUTPUT_COW_LABEL,
                    closing_radius=max(0, args.post_close_radius),
                    opening_radius=max(0, args.post_open_radius),
                    fill_holes=not args.no_fill_holes,
                    min_component_size=max(0, args.post_min_component_size),
                )

        else:
            pred_data[pred_data == 13] = 15

        out_name = os.path.basename(original_path).replace(".nii.gz", "_seg.nii.gz")
        out_path = os.path.join(args.output_dir, out_name)
        nib.save(nib.Nifti1Image(pred_data, pred_img.affine, pred_img.header), out_path)
        print(f"Saved: {out_path}")

    if not args.keep_temp:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
