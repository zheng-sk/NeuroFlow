import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensity

# External-library compatibility warnings (PyTorch/MONAI/CUDA bindings).
warnings.filterwarnings(
    "ignore",
    message=r"The cuda\.cudart module is deprecated and will be removed in a future release",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Using a non-tuple sequence for multidimensional indexing is deprecated",
    category=UserWarning,
    module=r"monai\.inferers\.utils",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_src_to_path() -> None:
    src_dir = _repo_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def load_case_table(csv_path: str, include_hr_mag: bool = True) -> List[Dict[str, Any]]:
    _add_src_to_path()
    from Network.NiftiPatchDataset import load_nifti_case_table

    return load_nifti_case_table(csv_path, include_hr_mag=include_hr_mag)


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def ensure_time_first(data: np.ndarray, time_axis: int) -> np.ndarray:
    if data.ndim == 3:
        return data[np.newaxis, ...]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape}")
    if time_axis < 0:
        time_axis = data.ndim + time_axis
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)
    return data


def adjust_affine_for_upsample(affine: np.ndarray, res_increase: int) -> np.ndarray:
    new_affine = affine.copy()
    new_affine[:3, :3] = new_affine[:3, :3] / float(res_increase)
    return new_affine


def detect_raw_mode(arr: np.ndarray) -> str:
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500.0) and (mx > 500.0) and (max_abs <= 8192.0) and (centered_ratio < 0.25)
    if is_unsigned_raw:
        return "unsigned"
    if is_signed_raw:
        return "signed"
    return "not_raw"


def raw_to_velocity(data: np.ndarray, venc: float, raw_center: float, raw_scale: float, invert_sign: bool) -> np.ndarray:
    out = data.astype(np.float32)
    mode = detect_raw_mode(out)
    if mode == "unsigned":
        out = (out - float(raw_center)) / float(raw_scale) * float(venc)
    elif mode == "signed":
        max_abs = max(abs(float(np.min(out))), abs(float(np.max(out))))
        scale = 4096.0 if max_abs > 3000.0 else 2048.0
        out = out / scale * float(venc)
    else:
        out = (out - float(raw_center)) / float(raw_scale) * float(venc)
    if invert_sign:
        out = -out
    return out


def resolve_component_venc(case: Dict[str, Any], key: str) -> float:
    v_comp = float(case.get(key, 0.0))
    if v_comp > 0:
        return v_comp
    return float(case.get("venc", 0.0))


def normalize_magnitude_volume(arr: np.ndarray, mode: str, mag_scale: float) -> np.ndarray:
    mode = str(mode).strip().lower()
    vol = arr.astype(np.float32)
    if mode == "divisor":
        return vol / float(mag_scale)
    if mode == "monai_minmax":
        scaler = ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=False)
        out = scaler(vol)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported mag_norm_mode={mode!r}. Use 'monai_minmax' or 'divisor'.")


def choose_frame_indices(case: Dict[str, Any], t_count: int, explicit_indices: Optional[List[int]] = None) -> List[int]:
    if explicit_indices:
        out: List[int] = []
        for idx in explicit_indices:
            if idx < 0 or idx >= t_count:
                raise ValueError(f"Frame index out of range: {idx}, t_count={t_count}")
            out.append(int(idx))
        return out

    time_start = int(case.get("time_start", -1))
    time_end = int(case.get("time_end", -1))
    time_index = int(case.get("time_index", -1))

    if time_start < 0:
        time_start = 0
    if time_end <= 0 or time_end > t_count:
        time_end = t_count
    if time_start >= time_end:
        raise ValueError(f"Invalid time range [{time_start}, {time_end}) for t_count={t_count}")

    if time_index >= 0:
        if time_index < time_start or time_index >= time_end:
            raise ValueError(f"time_index={time_index} outside [{time_start}, {time_end})")
        return [int(time_index)]

    return list(range(int(time_start), int(time_end)))


def load_sr_model(
    model_path: str,
    res_increase: int,
    low_resblock: int,
    hi_resblock: int,
    device: torch.device,
    predict_mag: Optional[bool] = None,
    model_variant: Optional[str] = None,
):
    _add_src_to_path()
    from Network.model_factory import build_sr_model, normalize_model_variant

    checkpoint = torch.load(model_path, map_location=device)
    if predict_mag is None and isinstance(checkpoint, dict) and "predict_mag" in checkpoint:
        predict_mag = bool(checkpoint["predict_mag"])
    if predict_mag is None:
        predict_mag = False

    checkpoint_variant = "original"
    if isinstance(checkpoint, dict) and "model_variant" in checkpoint:
        checkpoint_variant = str(checkpoint["model_variant"])
    resolved_variant = normalize_model_variant(model_variant or checkpoint_variant)

    model = build_sr_model(
        model_variant=resolved_variant,
        res_increase=res_increase,
        low_resblock=low_resblock,
        hi_resblock=hi_resblock,
        predict_mag=predict_mag,
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    setattr(model, "model_variant", resolved_variant)
    return model, bool(predict_mag)


def _predict_with_sliding_window(
    model,
    lr_input_norm: np.ndarray,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    def predictor_fn(x):
        u_t, v_t, w_t, um_t, vm_t, wm_t = torch.chunk(x, 6, dim=1)
        return model(u_t, v_t, w_t, um_t, vm_t, wm_t)

    lr_tensor = torch.from_numpy(lr_input_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using a non-tuple sequence for multidimensional indexing is deprecated",
                category=UserWarning,
                module=r"monai\.inferers\.utils",
            )
            pred = sliding_window_inference(
                inputs=lr_tensor,
                roi_size=roi_size,
                sw_batch_size=int(sw_batch_size),
                predictor=predictor_fn,
                overlap=float(overlap),
                mode="gaussian",
            )
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _read_case_volumes(case: Dict[str, Any], time_axis: int) -> Dict[str, Any]:
    lr_u_4d, lr_u_img = load_nifti(case["lr_u"])
    lr_v_4d, _ = load_nifti(case["lr_v"])
    lr_w_4d, _ = load_nifti(case["lr_w"])
    lr_mu_4d, _ = load_nifti(case["lr_mag_u"])
    lr_mv_4d, _ = load_nifti(case["lr_mag_v"])
    lr_mw_4d, _ = load_nifti(case["lr_mag_w"])

    hr_u_4d, hr_u_img = load_nifti(case["hr_u"])
    hr_v_4d, _ = load_nifti(case["hr_v"])
    hr_w_4d, _ = load_nifti(case["hr_w"])

    if not case.get("hr_mag"):
        raise ValueError("Case does not include hr_mag but 4-channel workflow requires hr_mag.")
    hr_mag_4d, _ = load_nifti(case["hr_mag"])

    mask_data = None
    mask_time_resolved = False
    if case.get("mask"):
        mask_raw, _ = load_nifti(case["mask"])
        if mask_raw.ndim == 4:
            mask_data = ensure_time_first(mask_raw, time_axis)
            mask_time_resolved = True
        elif mask_raw.ndim == 3:
            mask_data = mask_raw.astype(np.float32)
            mask_time_resolved = False
        else:
            raise ValueError(f"Mask must be 3D/4D, got shape {mask_raw.shape}")

    out = {
        "lr_u": ensure_time_first(lr_u_4d, time_axis),
        "lr_v": ensure_time_first(lr_v_4d, time_axis),
        "lr_w": ensure_time_first(lr_w_4d, time_axis),
        "lr_mu": ensure_time_first(lr_mu_4d, time_axis),
        "lr_mv": ensure_time_first(lr_mv_4d, time_axis),
        "lr_mw": ensure_time_first(lr_mw_4d, time_axis),
        "hr_u": ensure_time_first(hr_u_4d, time_axis),
        "hr_v": ensure_time_first(hr_v_4d, time_axis),
        "hr_w": ensure_time_first(hr_w_4d, time_axis),
        "hr_mag": ensure_time_first(hr_mag_4d, time_axis),
        "mask": mask_data,
        "mask_time_resolved": bool(mask_time_resolved),
        "lr_img": lr_u_img,
        "hr_img": hr_u_img,
    }

    t_count = min(
        out["lr_u"].shape[0],
        out["lr_v"].shape[0],
        out["lr_w"].shape[0],
        out["lr_mu"].shape[0],
        out["lr_mv"].shape[0],
        out["lr_mw"].shape[0],
        out["hr_u"].shape[0],
        out["hr_v"].shape[0],
        out["hr_w"].shape[0],
        out["hr_mag"].shape[0],
    )
    if out["mask"] is not None and out["mask_time_resolved"]:
        t_count = min(t_count, out["mask"].shape[0])
    out["t_count"] = int(t_count)
    return out


def _prepare_frame(
    case: Dict[str, Any],
    volumes: Dict[str, Any],
    frame_idx: int,
    raw_phase_input: bool,
    legacy_invert_uv_sign_on_raw: bool,
    raw_center: float,
    raw_scale: float,
    mag_scale: float,
    mag_norm_mode: str,
    mask_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lr_u = volumes["lr_u"][frame_idx].astype(np.float32)
    lr_v = volumes["lr_v"][frame_idx].astype(np.float32)
    lr_w = volumes["lr_w"][frame_idx].astype(np.float32)

    hr_u = volumes["hr_u"][frame_idx].astype(np.float32)
    hr_v = volumes["hr_v"][frame_idx].astype(np.float32)
    hr_w = volumes["hr_w"][frame_idx].astype(np.float32)
    hr_mag = volumes["hr_mag"][frame_idx].astype(np.float32)

    lr_mu = volumes["lr_mu"][frame_idx].astype(np.float32)
    lr_mv = volumes["lr_mv"][frame_idx].astype(np.float32)
    lr_mw = volumes["lr_mw"][frame_idx].astype(np.float32)

    if volumes["mask"] is None:
        mask_arr = np.ones_like(hr_u, dtype=np.float32)
    elif volumes.get("mask_time_resolved", False):
        mask_arr = volumes["mask"][frame_idx].astype(np.float32)
    else:
        mask_arr = volumes["mask"].astype(np.float32)
    mask_bin = (mask_arr >= float(mask_threshold)).astype(np.float32)

    venc_u = resolve_component_venc(case, "venc_u")
    venc_v = resolve_component_venc(case, "venc_v")
    venc_w = resolve_component_venc(case, "venc_w")

    if raw_phase_input:
        if venc_u <= 0:
            venc_u = float(np.max(np.abs(lr_u)))
        if venc_v <= 0:
            venc_v = float(np.max(np.abs(lr_v)))
        if venc_w <= 0:
            venc_w = float(np.max(np.abs(lr_w)))

        lr_u = raw_to_velocity(lr_u, venc_u, raw_center, raw_scale, legacy_invert_uv_sign_on_raw)
        lr_v = raw_to_velocity(lr_v, venc_v, raw_center, raw_scale, legacy_invert_uv_sign_on_raw)
        lr_w = raw_to_velocity(lr_w, venc_w, raw_center, raw_scale, False)

        hr_u = raw_to_velocity(hr_u, venc_u, raw_center, raw_scale, legacy_invert_uv_sign_on_raw)
        hr_v = raw_to_velocity(hr_v, venc_v, raw_center, raw_scale, legacy_invert_uv_sign_on_raw)
        hr_w = raw_to_velocity(hr_w, venc_w, raw_center, raw_scale, False)
    else:
        if venc_u <= 0:
            venc_u = float(np.max(np.abs(lr_u)))
        if venc_v <= 0:
            venc_v = float(np.max(np.abs(lr_v)))
        if venc_w <= 0:
            venc_w = float(np.max(np.abs(lr_w)))

    venc_scalar = float(max(venc_u, venc_v, venc_w))
    if venc_scalar <= 0:
        venc_scalar = float(np.max(np.abs(np.stack([lr_u, lr_v, lr_w], axis=0))))
    if venc_scalar <= 0:
        venc_scalar = 1.0

    lr_vel_norm = np.stack([lr_u, lr_v, lr_w], axis=0).astype(np.float32) / venc_scalar
    hr_vel_norm = np.stack([hr_u, hr_v, hr_w], axis=0).astype(np.float32) / venc_scalar
    lr_mag_norm = np.stack(
        [
            normalize_magnitude_volume(lr_mu, mag_norm_mode, mag_scale),
            normalize_magnitude_volume(lr_mv, mag_norm_mode, mag_scale),
            normalize_magnitude_volume(lr_mw, mag_norm_mode, mag_scale),
        ],
        axis=0,
    ).astype(np.float32)
    hr_mag_norm = normalize_magnitude_volume(hr_mag, mag_norm_mode, mag_scale)[None, ...]

    lr_input_norm = np.concatenate([lr_vel_norm, lr_mag_norm], axis=0).astype(np.float32)
    gt_4ch_norm = np.concatenate([hr_vel_norm, hr_mag_norm], axis=0).astype(np.float32)
    return lr_input_norm, gt_4ch_norm, mask_bin, venc_scalar


def run_case_inference(
    case: Dict[str, Any],
    model,
    frame_indices: List[int],
    patch_size: int,
    sw_batch_size: int,
    overlap: float,
    device: torch.device,
    raw_phase_input: bool,
    legacy_invert_uv_sign_on_raw: bool,
    raw_center: float,
    raw_scale: float,
    mag_scale: float,
    mag_norm_mode: str,
    mask_threshold: float,
    time_axis: int,
) -> Dict[str, Any]:
    volumes = _read_case_volumes(case=case, time_axis=time_axis)

    lr_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    mask_all: List[np.ndarray] = []
    venc_all: List[float] = []

    for i, frame_idx in enumerate(frame_indices):
        lr_input_norm, gt_4ch_norm, mask_bin, venc_scalar = _prepare_frame(
            case=case,
            volumes=volumes,
            frame_idx=frame_idx,
            raw_phase_input=raw_phase_input,
            legacy_invert_uv_sign_on_raw=legacy_invert_uv_sign_on_raw,
            raw_center=raw_center,
            raw_scale=raw_scale,
            mag_scale=mag_scale,
            mag_norm_mode=mag_norm_mode,
            mask_threshold=mask_threshold,
        )

        pred_norm = _predict_with_sliding_window(
            model=model,
            lr_input_norm=lr_input_norm,
            roi_size=(int(patch_size), int(patch_size), int(patch_size)),
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            device=device,
        )
        if pred_norm.shape[0] != 4:
            raise ValueError(f"Expected 4 output channels (u,v,w,mag), got shape {pred_norm.shape}")

        lr_all.append(lr_input_norm)
        gt_all.append(gt_4ch_norm)
        pred_all.append(pred_norm)
        mask_all.append(mask_bin)
        venc_all.append(float(venc_scalar))

        print(f"Processed frame {i + 1}/{len(frame_indices)} (t={frame_idx})")

    return {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "lr_norm": np.stack(lr_all, axis=0).astype(np.float32),
        "gt_norm": np.stack(gt_all, axis=0).astype(np.float32),
        "pred_norm": np.stack(pred_all, axis=0).astype(np.float32),
        "mask": np.stack(mask_all, axis=0).astype(np.float32),
        "venc": np.asarray(venc_all, dtype=np.float32),
        "lr_affine": np.asarray(volumes["lr_img"].affine, dtype=np.float32),
        "hr_affine": np.asarray(volumes["hr_img"].affine, dtype=np.float32),
        "lr_spacing": np.asarray(volumes["lr_img"].header.get_zooms()[:3], dtype=np.float32),
        "hr_spacing": np.asarray(volumes["hr_img"].header.get_zooms()[:3], dtype=np.float32),
        "t_count": int(volumes["t_count"]),
    }


def save_predicted_nifti(
    pred_norm: np.ndarray,
    venc_per_frame: np.ndarray,
    mag_scale: float,
    mag_norm_mode: str,
    out_prefix: str,
    lr_affine: np.ndarray,
    res_increase: int,
    save_uvw: bool = True,
) -> Dict[str, str]:
    # pred_norm: [T, C, X, Y, Z], C=4
    pred_phys = pred_norm.astype(np.float32).copy()
    for t in range(pred_phys.shape[0]):
        pred_phys[t, :3] *= float(venc_per_frame[t])
    if str(mag_norm_mode).strip().lower() == "divisor":
        pred_phys[:, 3] *= float(mag_scale)

    out_affine = adjust_affine_for_upsample(lr_affine, res_increase=res_increase)

    out_paths: Dict[str, str] = {}
    stems = ["u", "v", "w", "mag"]
    for ch, stem in enumerate(stems):
        arr = pred_phys[:, ch]  # [T, X, Y, Z]
        if arr.shape[0] == 1:
            arr_out = arr[0]
        else:
            arr_out = np.moveaxis(arr, 0, -1)  # [X, Y, Z, T]
        p = f"{out_prefix}_{stem}.nii.gz"
        nib.save(nib.Nifti1Image(arr_out.astype(np.float32), out_affine), p)
        out_paths[stem] = p

    if save_uvw:
        uvw = np.moveaxis(pred_phys[:, :3], 1, -1)  # [T, X, Y, Z, 3]
        uvw = np.moveaxis(uvw, 0, 3)  # [X, Y, Z, T, 3]
        if uvw.shape[3] == 1:
            uvw_out = uvw[:, :, :, 0, :]
        else:
            uvw_out = uvw
        p = f"{out_prefix}_uvw.nii.gz"
        nib.save(nib.Nifti1Image(uvw_out.astype(np.float32), out_affine), p)
        out_paths["uvw"] = p

    return out_paths


def save_payload_npz(path: str, payload: Dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        frame_indices=payload["frame_indices"],
        lr_norm=payload["lr_norm"],
        gt_norm=payload["gt_norm"],
        pred_norm=payload["pred_norm"],
        mask=payload["mask"],
        venc=payload["venc"],
        lr_affine=payload["lr_affine"],
        hr_affine=payload["hr_affine"],
        lr_spacing=payload["lr_spacing"],
        hr_spacing=payload["hr_spacing"],
    )


def save_metadata_json(path: str, metadata: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
