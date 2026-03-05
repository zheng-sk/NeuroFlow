import csv
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset as MonaiDataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, RandomizableTransform, ScaleIntensity
from torch.utils.data import Dataset

from .PatchHandler3D import rotate180_3d, rotate90


def _resolve_path(path_value: str, base_dir: str) -> str:
    if path_value is None:
        return ""
    path_value = path_value.strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value

    base_abs = os.path.abspath(base_dir)
    search_roots = [base_abs, os.path.abspath(os.getcwd())]

    # Also try every CSV ancestor directory so paths like `data/...` work when
    # CSV lives in `data/paired_dataset` and trainer is launched from `src/`.
    cursor = base_abs
    while True:
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        search_roots.append(parent)
        cursor = parent

    seen = set()
    ordered_roots = []
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        ordered_roots.append(root)

    for root in ordered_roots:
        candidate = os.path.abspath(os.path.join(root, path_value))
        if os.path.exists(candidate):
            return candidate

    # Keep deterministic behavior for downstream error messages.
    return os.path.abspath(os.path.join(base_abs, path_value))


def load_nifti_case_table(csv_path: str, include_hr_mag: bool = False) -> List[Dict]:
    """
    Load case descriptors from CSV.

    Required columns:
        lr_u, lr_v, lr_w, lr_mag_u, lr_mag_v, lr_mag_w, hr_u, hr_v, hr_w
    Optional columns:
        hr_mag, mask, venc, venc_u, venc_v, venc_w, time_start, time_end, time_index
    """
    required = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w"]
    optional = ["hr_mag", "mask", "venc", "venc_u", "venc_v", "venc_w", "time_start", "time_end", "time_index"]

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    cases = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            missing = [k for k in required if not row.get(k, "").strip()]
            if missing:
                raise ValueError(f"Row {idx} in {csv_path} missing required columns/values: {missing}")

            case = {}
            for key in required + optional:
                value = row.get(key, "")
                if key in ("venc", "venc_u", "venc_v", "venc_w"):
                    case[key] = float(value) if str(value).strip() else 0.0
                elif key in ("time_start", "time_end", "time_index"):
                    case[key] = int(value) if str(value).strip() else -1
                elif key == "hr_mag":
                    if str(value).strip():
                        case[key] = _resolve_path(value, base_dir)
                    elif include_hr_mag:
                        # Backward-compatible fallback for older CSVs without hr_mag.
                        hr_u_path = _resolve_path(row.get("hr_u", ""), base_dir)
                        inferred = os.path.join(os.path.dirname(hr_u_path), "input_mag_raw.nii.gz")
                        case[key] = inferred if os.path.exists(inferred) else ""
                    else:
                        case[key] = ""
                elif key == "mask":
                    case[key] = _resolve_path(value, base_dir) if str(value).strip() else None
                else:
                    case[key] = _resolve_path(value, base_dir)
            if include_hr_mag and not case.get("hr_mag"):
                raise ValueError(
                    f"Row {idx} in {csv_path} requires hr_mag for --predict-mag. "
                    "Add `hr_mag` column or ensure `<hr_u_dir>/input_mag_raw.nii.gz` exists."
                )
            cases.append(case)
    return cases


def _build_monai_case_dataset(cases: List[Dict], load_transform, use_cache: bool, cache_eager: bool):
    """
    Build the MONAI dataset used for case-level loading.
    - without cache: monai.data.Dataset
    - with cache:    monai.data.CacheDataset
    """
    prepared_cases = [dict(c) for c in cases]
    if not use_cache:
        return MonaiDataset(data=prepared_cases, transform=load_transform)

    def _create_cache_dataset(extra_kwargs=None):
        kwargs = dict(data=prepared_cases, transform=load_transform, cache_rate=1.0, num_workers=0)
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        try:
            return CacheDataset(**kwargs)
        except (TypeError, ValueError):
            # Backward compatibility for older MONAI versions.
            kwargs.pop("num_workers", None)
            return CacheDataset(**kwargs)

    if cache_eager:
        return _create_cache_dataset()

    # Lazy runtime cache if supported by this MONAI version.
    for runtime_cache in (True, "threads", "processes"):
        try:
            return _create_cache_dataset({"runtime_cache": runtime_cache})
        except (TypeError, ValueError):
            continue

    print("Warning: runtime lazy cache is not supported in this MONAI version; using eager cache instead.")
    return _create_cache_dataset()


class _StackNormalizeFieldsd(RandomizableTransform):
    def __init__(
        self,
        mag_scale: float,
        mag_norm_mode: str = "monai_minmax",
        mask_threshold: float = 0.5,
        include_hr_mag: bool = False,
        raw_phase_input: bool = True,
        invert_uv_sign_on_raw: bool = False,
        raw_center: float = 2048.0,
        raw_scale: float = 2048.0,
        random_time_frame: bool = False,
        time_axis: int = -1,
        fixed_time_index: int = 0,
    ):
        super().__init__()
        self.mag_scale = float(mag_scale)
        self.mag_norm_mode = str(mag_norm_mode).strip().lower()
        if self.mag_norm_mode not in {"monai_minmax", "divisor"}:
            raise ValueError(f"Unsupported mag_norm_mode={mag_norm_mode!r}. Use 'monai_minmax' or 'divisor'.")
        self.mask_threshold = float(mask_threshold)
        self.include_hr_mag = bool(include_hr_mag)
        self.raw_phase_input = bool(raw_phase_input)
        self.invert_uv_sign_on_raw = bool(invert_uv_sign_on_raw)
        self.raw_center = float(raw_center)
        self.raw_scale = float(raw_scale)
        self.random_time_frame = bool(random_time_frame)
        self.time_axis = int(time_axis)
        self.fixed_time_index = int(fixed_time_index)
        self._mag_scaler = ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=False)

    def _normalize_time_axis(self, ndim: int) -> int:
        axis = self.time_axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Invalid time axis {self.time_axis} for ndim={ndim}")
        return axis

    def _select_time_frame(self, arr: np.ndarray, t_index: int) -> np.ndarray:
        if arr.ndim == 3:
            return arr
        if arr.ndim != 4:
            raise ValueError(f"Expected 3D/4D volume after channel squeeze, got shape {arr.shape}")
        time_axis = self._normalize_time_axis(arr.ndim)
        return np.take(arr, indices=t_index, axis=time_axis)

    def _resolve_component_venc(self, d, key):
        # component venc > shared venc > 0 (auto later)
        v_comp = float(d.get(key, 0.0))
        if v_comp > 0:
            return v_comp
        v_shared = float(d.get("venc", 0.0))
        return v_shared

    @staticmethod
    def _detect_raw_mode(arr: np.ndarray) -> str:
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

    def _raw_to_velocity(self, arr, venc, invert_sign=False):
        mode = self._detect_raw_mode(arr)
        vel = arr.astype(np.float32)
        if mode == "unsigned":
            vel = (vel - self.raw_center) / self.raw_scale * float(venc)
        elif mode == "signed":
            max_abs = max(abs(float(np.min(vel))), abs(float(np.max(vel))))
            scale = 4096.0 if max_abs > 3000.0 else 2048.0
            vel = vel / scale * float(venc)
        else:
            # Legacy fallback: preserve previous behavior when values do not look RAW.
            vel = (vel - self.raw_center) / self.raw_scale * float(venc)
        if invert_sign:
            vel = -vel
        return vel

    def _normalize_magnitude_monai(self, arr: np.ndarray) -> np.ndarray:
        scaled = self._mag_scaler(arr.astype(np.float32))
        if isinstance(scaled, torch.Tensor):
            scaled = scaled.detach().cpu().numpy()
        scaled = np.asarray(scaled, dtype=np.float32)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    def __call__(self, data):
        d = dict(data)

        # Keep channel-0 only since EnsureChannelFirstd is used on scalar NIfTI volumes.
        lr_u_raw = d["lr_u"][0].astype(np.float32)
        lr_v_raw = d["lr_v"][0].astype(np.float32)
        lr_w_raw = d["lr_w"][0].astype(np.float32)
        hr_u_raw = d["hr_u"][0].astype(np.float32)
        hr_v_raw = d["hr_v"][0].astype(np.float32)
        hr_w_raw = d["hr_w"][0].astype(np.float32)
        mag_u_raw = d["lr_mag_u"][0].astype(np.float32)
        mag_v_raw = d["lr_mag_v"][0].astype(np.float32)
        mag_w_raw = d["lr_mag_w"][0].astype(np.float32)
        hr_mag_raw = d["hr_mag"][0].astype(np.float32) if self.include_hr_mag else None
        mask_raw = d["mask"][0].astype(np.float32) if "mask" in d and d["mask"] is not None else None

        # 4D NIfTI support: sample/select one time frame and continue with 3D patching.
        if lr_u_raw.ndim == 4:
            time_lengths = []
            for arr in [lr_u_raw, lr_v_raw, lr_w_raw, hr_u_raw, hr_v_raw, hr_w_raw, mag_u_raw, mag_v_raw, mag_w_raw]:
                axis = self._normalize_time_axis(arr.ndim)
                time_lengths.append(arr.shape[axis])
            if hr_mag_raw is not None and hr_mag_raw.ndim == 4:
                axis = self._normalize_time_axis(hr_mag_raw.ndim)
                time_lengths.append(hr_mag_raw.shape[axis])
            if mask_raw is not None and mask_raw.ndim == 4:
                axis = self._normalize_time_axis(mask_raw.ndim)
                time_lengths.append(mask_raw.shape[axis])

            t_count = int(min(time_lengths))
            if t_count <= 0:
                raise ValueError("No valid time frames found in 4D NIfTI inputs.")

            time_start = int(d.get("time_start", -1))
            time_end = int(d.get("time_end", -1))
            time_index = int(d.get("time_index", -1))

            if time_start < 0:
                time_start = 0
            if time_end <= 0 or time_end > t_count:
                time_end = t_count
            if time_start >= time_end:
                raise ValueError(f"Invalid time range [{time_start}, {time_end}) for {t_count} frames.")

            if time_index >= 0:
                if time_index < time_start or time_index >= time_end:
                    raise ValueError(
                        f"time_index={time_index} is outside selected range [{time_start}, {time_end})."
                    )
                t_index = time_index
            elif self.random_time_frame:
                t_index = int(self.R.randint(time_start, time_end))
            else:
                range_len = time_end - time_start
                offset = int(np.clip(self.fixed_time_index, 0, range_len - 1))
                t_index = time_start + offset
        else:
            t_index = 0

        lr_u = self._select_time_frame(lr_u_raw, t_index)
        lr_v = self._select_time_frame(lr_v_raw, t_index)
        lr_w = self._select_time_frame(lr_w_raw, t_index)
        hr_u = self._select_time_frame(hr_u_raw, t_index)
        hr_v = self._select_time_frame(hr_v_raw, t_index)
        hr_w = self._select_time_frame(hr_w_raw, t_index)
        mag_u = self._select_time_frame(mag_u_raw, t_index)
        mag_v = self._select_time_frame(mag_v_raw, t_index)
        mag_w = self._select_time_frame(mag_w_raw, t_index)
        hr_mag = self._select_time_frame(hr_mag_raw, t_index) if hr_mag_raw is not None else None
        mask = self._select_time_frame(mask_raw, t_index) if mask_raw is not None else None

        venc_u = self._resolve_component_venc(d, "venc_u")
        venc_v = self._resolve_component_venc(d, "venc_v")
        venc_w = self._resolve_component_venc(d, "venc_w")

        # Legacy-compatible default: assume velocity NIfTI is raw phase-like [0..4095].
        if self.raw_phase_input:
            if venc_u <= 0:
                venc_u = float(np.max(np.abs(lr_u)))
            if venc_v <= 0:
                venc_v = float(np.max(np.abs(lr_v)))
            if venc_w <= 0:
                venc_w = float(np.max(np.abs(lr_w)))

            # Default behavior avoids double inversion when DICOM->NIfTI already
            # applied LPS->RAS sign correction to Vx/Vy. Enable legacy inversion
            # with `invert_uv_sign_on_raw=True` only for old datasets.
            lr_u = self._raw_to_velocity(lr_u, venc=venc_u, invert_sign=self.invert_uv_sign_on_raw)
            lr_v = self._raw_to_velocity(lr_v, venc=venc_v, invert_sign=self.invert_uv_sign_on_raw)
            lr_w = self._raw_to_velocity(lr_w, venc=venc_w, invert_sign=False)
            hr_u = self._raw_to_velocity(hr_u, venc=venc_u, invert_sign=self.invert_uv_sign_on_raw)
            hr_v = self._raw_to_velocity(hr_v, venc=venc_v, invert_sign=self.invert_uv_sign_on_raw)
            hr_w = self._raw_to_velocity(hr_w, venc=venc_w, invert_sign=False)
        else:
            if venc_u <= 0:
                venc_u = float(np.max(np.abs(lr_u)))
            if venc_v <= 0:
                venc_v = float(np.max(np.abs(lr_v)))
            if venc_w <= 0:
                venc_w = float(np.max(np.abs(lr_w)))

        lr_vel = np.stack([lr_u, lr_v, lr_w], axis=0).astype(np.float32)
        hr_vel = np.stack([hr_u, hr_v, hr_w], axis=0).astype(np.float32)
        venc = float(max(venc_u, venc_v, venc_w))
        if venc <= 0:
            venc = float(np.max(np.abs(lr_vel)))
        if venc <= 0:
            venc = 1.0

        lr_vel = lr_vel / venc
        hr_vel = hr_vel / venc
        if self.mag_norm_mode == "monai_minmax":
            lr_mag = np.stack(
                [
                    self._normalize_magnitude_monai(mag_u),
                    self._normalize_magnitude_monai(mag_v),
                    self._normalize_magnitude_monai(mag_w),
                ],
                axis=0,
            ).astype(np.float32)
            if hr_mag is not None:
                hr_mag = self._normalize_magnitude_monai(hr_mag)
        else:
            lr_mag = np.stack([mag_u, mag_v, mag_w], axis=0).astype(np.float32) / self.mag_scale
            if hr_mag is not None:
                hr_mag = hr_mag.astype(np.float32) / self.mag_scale

        if mask is not None:
            mask = (mask >= self.mask_threshold).astype(np.float32)
        else:
            mask = np.ones(hr_vel.shape[1:], dtype=np.float32)

        d["lr_vel"] = lr_vel
        d["hr_vel"] = hr_vel
        d["lr_mag"] = lr_mag
        if hr_mag is not None:
            d["hr_mag"] = hr_mag
        d["mask"] = mask
        d["venc"] = np.float32(venc)
        return d


class _RandomVectorRotate90d(RandomizableTransform):
    def __init__(self, prob: float):
        super().__init__(prob=float(prob))
        self.rotation_idx = 0
        self.plane_nr = 0

    def randomize(self, data=None):
        super().randomize(None)
        self.rotation_idx = 0
        self.plane_nr = 0
        if not self._do_transform:
            return
        self.rotation_idx = int(self.R.choice([1, 2, 3]))
        self.plane_nr = int(self.R.choice([1, 2, 3]))

    @staticmethod
    def _rotate_scalar(img, rotation_idx: int, plane_nr: int):
        if plane_nr == 1:
            ax = (0, 1)
        elif plane_nr == 2:
            ax = (0, 2)
        else:
            ax = (1, 2)
        return np.rot90(img, k=rotation_idx, axes=ax)

    def __call__(self, data):
        d = dict(data)
        # MONAI versions differ in RandomizableTransform.randomize signature.
        # Passing data keeps compatibility when `data` is required.
        self.randomize(d)
        if not self._do_transform:
            return d

        rotation_idx = self.rotation_idx
        plane_nr = self.plane_nr

        # Velocity vectors require axis swaps/sign updates.
        u, v, w = d["lr_vel"][0], d["lr_vel"][1], d["lr_vel"][2]
        if rotation_idx == 2:
            u, v, w = rotate180_3d(u, v, w, plane_nr, is_phase_img=True)
        else:
            u, v, w = rotate90(u, v, w, plane_nr, rotation_idx, is_phase_img=True)
        d["lr_vel"] = np.stack([u, v, w], axis=0).astype(np.float32)

        u, v, w = d["hr_vel"][0], d["hr_vel"][1], d["hr_vel"][2]
        if rotation_idx == 2:
            u, v, w = rotate180_3d(u, v, w, plane_nr, is_phase_img=True)
        else:
            u, v, w = rotate90(u, v, w, plane_nr, rotation_idx, is_phase_img=True)
        d["hr_vel"] = np.stack([u, v, w], axis=0).astype(np.float32)

        # Magnitude and mask are scalar fields.
        d["lr_mag"] = np.stack(
            [
                self._rotate_scalar(d["lr_mag"][0], rotation_idx, plane_nr),
                self._rotate_scalar(d["lr_mag"][1], rotation_idx, plane_nr),
                self._rotate_scalar(d["lr_mag"][2], rotation_idx, plane_nr),
            ],
            axis=0,
        ).astype(np.float32)
        if "hr_mag" in d:
            d["hr_mag"] = self._rotate_scalar(d["hr_mag"], rotation_idx, plane_nr).astype(np.float32)
        d["mask"] = self._rotate_scalar(d["mask"], rotation_idx, plane_nr).astype(np.float32)
        return d


class _AddNonvascularNoiseAugd(RandomizableTransform):
    """Add random noise on LR inputs outside the vascular mask.

    The sampling distribution can be:
    - direct synthetic family (normal/student_t/skew_normal/etc), or
    - loaded once from a precomputed fit summary CSV (best model per channel).

    During training we only sample from the precomputed function; no per-batch
    refitting is performed.
    """

    _SUPPORTED_DISTS = {
        "normal",
        "gaussian",
        "student_t",
        "laplace",
        "uniform",
        "skew_normal",
        "generalized_normal",
        "logistic",
        "cauchy",
        "hat",
    }
    _PRETTY_TO_INTERNAL = {
        "Normal": "normal",
        "Laplace": "laplace",
        "StudentT": "student_t",
        "Cauchy": "cauchy",
        "Logistic": "logistic",
        "GeneralizedNormal": "generalized_normal",
        "SkewNormal": "skew_normal",
    }

    def __init__(
        self,
        prob: float = 0.0,
        phase_dist: str = "student_t",
        phase_scale: float = 0.04,
        phase_shape: float = 3.0,
        mag_dist: str = "skew_normal",
        mag_scale: float = 0.02,
        mag_shape: float = 4.0,
        range_multiplier: float = 2.0,
        hat_mix: float = 0.7,
        apply_to_magnitude: bool = True,
        clip_magnitude: bool = False,
        fit_summary_csv: str = "",
        exaggerated_side_expand: float = 1.0,
        exaggerated_edge_boost: float = 0.0,
        exaggerated_edge_power: float = 1.8,
        level_min: float = 0.8,
        level_max: float = 1.4,
        masked_fraction: float = 0.0,
        keep_original_prob: float = 0.0,
        zero_outside_prob: float = 0.0,
    ):
        super().__init__(prob=float(prob))
        self.phase_dist = self._normalize_dist_name(phase_dist)
        self.phase_scale = float(phase_scale)
        self.phase_shape = float(phase_shape)
        self.mag_dist = self._normalize_dist_name(mag_dist)
        self.mag_scale = float(mag_scale)
        self.mag_shape = float(mag_shape)
        self.range_multiplier = float(range_multiplier)
        self.hat_mix = float(np.clip(hat_mix, 0.0, 1.0))
        self.apply_to_magnitude = bool(apply_to_magnitude)
        self.clip_magnitude = bool(clip_magnitude)
        self.fit_summary_csv = str(fit_summary_csv or "").strip()
        self.exaggerated_side_expand = max(float(exaggerated_side_expand), 1.0)
        self.exaggerated_edge_boost = max(float(exaggerated_edge_boost), 0.0)
        self.exaggerated_edge_power = max(float(exaggerated_edge_power), 1e-6)
        self.level_min = float(level_min)
        self.level_max = float(level_max)
        self.masked_fraction = float(np.clip(masked_fraction, 0.0, 1.0))
        self.keep_original_prob = float(np.clip(keep_original_prob, 0.0, 1.0))
        self.zero_outside_prob = float(np.clip(zero_outside_prob, 0.0, 1.0))
        if (self.keep_original_prob + self.zero_outside_prob) > 1.0:
            raise ValueError(
                "keep_original_prob + zero_outside_prob must be <= 1.0 "
                f"(got {self.keep_original_prob + self.zero_outside_prob:.4f})."
            )
        self._channel_fit_config = self._load_channel_fit_config(self.fit_summary_csv)

    @classmethod
    def _normalize_dist_name(cls, name: str) -> str:
        dist = str(name).strip().lower().replace("-", "_")
        if dist == "gaussian":
            dist = "normal"
        if dist not in cls._SUPPORTED_DISTS:
            raise ValueError(f"Unsupported noise distribution {name!r}.")
        return dist

    def _load_channel_fit_config(self, fit_summary_csv: str) -> dict[str, tuple[str, float]]:
        """Load per-channel best-fit model from fit summary CSV (if available)."""
        if not fit_summary_csv:
            return {}
        candidates = []
        if os.path.isabs(fit_summary_csv):
            candidates.append(fit_summary_csv)
        else:
            cwd = os.path.abspath(os.getcwd())
            candidates.extend(
                [
                    os.path.abspath(fit_summary_csv),
                    os.path.abspath(os.path.join(cwd, fit_summary_csv)),
                    os.path.abspath(os.path.join(cwd, "..", fit_summary_csv)),
                ]
            )

        csv_path = ""
        for cand in candidates:
            if os.path.exists(cand):
                csv_path = cand
                break
        if not csv_path and candidates:
            csv_path = candidates[0]
        if not os.path.exists(csv_path):
            print(f"[WARN] noise fit summary CSV not found: {csv_path}. Falling back to configured distributions.")
            return {}

        wanted = {"Magnitude", "Phase Vx", "Phase Vy", "Phase Vz"}
        loaded: dict[str, tuple[str, float]] = {}
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    channel = str(row.get("channel", "")).strip()
                    if channel not in wanted or channel in loaded:
                        continue

                    pretty_model = str(row.get("model", "")).strip()
                    internal = self._PRETTY_TO_INTERNAL.get(pretty_model, "")
                    if not internal:
                        continue
                    internal = self._normalize_dist_name(internal)

                    params_text = str(row.get("params", "")).strip()
                    params = []
                    if params_text:
                        for tok in params_text.split(";"):
                            tok = tok.strip()
                            if not tok:
                                continue
                            params.append(float(tok))

                    shape = self._infer_shape_param(internal, params)
                    loaded[channel] = (internal, shape)
        except Exception as exc:
            print(f"[WARN] Failed to parse noise fit summary CSV {csv_path}: {exc}")
            return {}

        if loaded:
            text = ", ".join(f"{k}:{v[0]}(shape={v[1]:.4g})" for k, v in sorted(loaded.items()))
            print(f"[INFO] Noise augmentation using precomputed best-fit families from CSV: {text}")
        else:
            print(f"[WARN] No usable best-fit rows found in {csv_path}. Falling back to configured distributions.")
        return loaded

    @staticmethod
    def _infer_shape_param(dist: str, params: list[float]) -> float:
        if not params:
            return 0.0
        if dist in {"student_t", "generalized_normal", "skew_normal"}:
            return float(params[0])
        return 0.0

    def _sample_skew_normal(self, size, alpha: float):
        alpha = float(alpha)
        delta = alpha / math.sqrt(1.0 + alpha * alpha)
        u0 = self.R.normal(0.0, 1.0, size=size)
        v = self.R.normal(0.0, 1.0, size=size)
        z = delta * np.abs(u0) + math.sqrt(max(1.0 - delta * delta, 1e-8)) * v

        # Standardize to near zero-mean, unit-variance.
        mean = delta * math.sqrt(2.0 / math.pi)
        var = max(1.0 - (2.0 * delta * delta / math.pi), 1e-8)
        return (z - mean) / math.sqrt(var)

    def _sample_generalized_normal(self, size, beta: float):
        # Exponential power / generalized normal (version 1), standardized.
        beta = max(float(beta), 0.2)
        gamma = self.R.gamma(shape=1.0 / beta, scale=1.0, size=size)
        signs = self.R.choice(np.array([-1.0, 1.0], dtype=np.float32), size=size)
        z = signs * np.power(gamma, 1.0 / beta)
        var = max(math.gamma(3.0 / beta) / math.gamma(1.0 / beta), 1e-8)
        return z / math.sqrt(var)

    def _sample_standard(self, size, dist: str, shape_param: float):
        dist = self._normalize_dist_name(dist)

        if dist == "normal":
            return self.R.normal(0.0, 1.0, size=size)
        if dist == "student_t":
            df = max(float(shape_param), 1.1)
            return self.R.standard_t(df=df, size=size)
        if dist == "laplace":
            # Laplace with variance=1
            return self.R.laplace(loc=0.0, scale=1.0 / math.sqrt(2.0), size=size)
        if dist == "uniform":
            # Uniform with variance=1
            s = math.sqrt(3.0)
            return self.R.uniform(low=-s, high=s, size=size)
        if dist == "logistic":
            # Logistic with variance=1.
            z = self.R.logistic(loc=0.0, scale=1.0, size=size)
            return z / (math.pi / math.sqrt(3.0))
        if dist == "cauchy":
            # Heavy-tail fallback with robust clipping and standardization.
            z = self.R.standard_cauchy(size=size)
            z = np.clip(z, -25.0, 25.0)
            z_std = float(np.std(z))
            return z / max(z_std, 1e-6)
        if dist == "skew_normal":
            return self._sample_skew_normal(size=size, alpha=shape_param)
        if dist == "generalized_normal":
            return self._sample_generalized_normal(size=size, beta=shape_param)

        # "Hat-like" flattened profile: uniform body + heavy-tail component.
        df = max(float(shape_param), 1.1)
        u = self.R.uniform(low=-1.0, high=1.0, size=size)
        t = np.tanh(self.R.standard_t(df=df, size=size) / 2.5)
        selector = self.R.random_sample(size=size) < self.hat_mix
        z = np.where(selector, u, t)
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))
        return (z - z_mean) / max(z_std, 1e-6)

    def _draw_level(self) -> float:
        lo = float(min(self.level_min, self.level_max))
        hi = float(max(self.level_min, self.level_max))
        if abs(hi - lo) < 1e-12:
            return lo
        return float(self.R.uniform(lo, hi))

    def _apply_exaggeration_profile(self, z: np.ndarray) -> np.ndarray:
        zz = np.asarray(z, dtype=np.float64)
        zz = zz * self.exaggerated_side_expand
        if self.exaggerated_edge_boost > 0.0:
            mag = np.abs(zz)
            zz = zz + np.sign(zz) * self.exaggerated_edge_boost * np.power(mag, self.exaggerated_edge_power)
        zz = np.clip(zz, -35.0, 35.0)
        return zz.astype(np.float32)

    def _resolve_dist_and_shape(self, channel_name: str, fallback_dist: str, fallback_shape: float) -> tuple[str, float]:
        cfg = self._channel_fit_config.get(channel_name)
        if cfg is not None:
            return cfg
        return self._normalize_dist_name(fallback_dist), float(fallback_shape)

    def _sample_noise(self, shape, dist: str, scale: float, shape_param: float, channel_name: str):
        dist_name, resolved_shape = self._resolve_dist_and_shape(
            channel_name=channel_name,
            fallback_dist=dist,
            fallback_shape=shape_param,
        )
        z = self._sample_standard(shape, dist=dist_name, shape_param=resolved_shape)
        z = self._apply_exaggeration_profile(z)
        level = self._draw_level()
        eff_scale = float(scale) * max(float(self.range_multiplier), 0.0) * max(level, 0.0)
        return (z * eff_scale).astype(np.float32)

    @staticmethod
    def _nearest_resize_mask(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        sx, sy, sz = mask.shape
        tx, ty, tz = target_shape
        ix = np.clip(np.round(np.linspace(0, sx - 1, tx)).astype(np.int64), 0, sx - 1)
        iy = np.clip(np.round(np.linspace(0, sy - 1, ty)).astype(np.int64), 0, sy - 1)
        iz = np.clip(np.round(np.linspace(0, sz - 1, tz)).astype(np.int64), 0, sz - 1)
        return mask[np.ix_(ix, iy, iz)]

    @classmethod
    def _align_mask_to_target(cls, mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        if tuple(mask.shape) == tuple(target_shape):
            return mask.astype(np.float32)

        sx, sy, sz = mask.shape
        tx, ty, tz = target_shape
        fx = sx // tx if tx > 0 else 0
        fy = sy // ty if ty > 0 else 0
        fz = sz // tz if tz > 0 else 0

        # Common LR/HR case: exact integer factors (e.g., 32->16 with factor=2).
        # Use max-pooling on vessel mask to conservatively avoid adding noise near vessels.
        if (
            fx > 0
            and fy > 0
            and fz > 0
            and sx == tx * fx
            and sy == ty * fy
            and sz == tz * fz
        ):
            pooled = mask.reshape(tx, fx, ty, fy, tz, fz).max(axis=(1, 3, 5))
            return pooled.astype(np.float32)

        # Fallback for non-integer mismatches.
        return cls._nearest_resize_mask(mask, target_shape).astype(np.float32)

    def __call__(self, data):
        d = dict(data)
        # Compatibility across MONAI versions where randomize may require `data`.
        self.randomize(d)
        if not self._do_transform:
            return d

        if "mask" not in d or d["mask"] is None:
            return d
        mask = np.asarray(d["mask"], dtype=np.float32)
        target_shape = None
        if "lr_vel" in d and d["lr_vel"] is not None:
            target_shape = tuple(np.asarray(d["lr_vel"]).shape[1:4])
        elif "lr_mag" in d and d["lr_mag"] is not None:
            target_shape = tuple(np.asarray(d["lr_mag"]).shape[1:4])
        if target_shape is None:
            return d

        mask_lr = self._align_mask_to_target(mask, target_shape)
        nonvascular = (mask_lr < 0.5).astype(np.float32)
        vascular = 1.0 - nonvascular
        noise_weight = nonvascular + self.masked_fraction * vascular
        if float(noise_weight.sum()) <= 0.0:
            return d

        mode_draw = float(self.R.uniform())
        if mode_draw < self.keep_original_prob:
            return d
        if mode_draw < (self.keep_original_prob + self.zero_outside_prob):
            outside = (mask_lr < 0.5).astype(np.float32)
            keep = 1.0 - outside
            if "lr_vel" in d and d["lr_vel"] is not None:
                vel = np.asarray(d["lr_vel"], dtype=np.float32)
                d["lr_vel"] = (vel * keep[None, ...]).astype(np.float32)
            if self.apply_to_magnitude and "lr_mag" in d and d["lr_mag"] is not None:
                mag = np.asarray(d["lr_mag"], dtype=np.float32)
                d["lr_mag"] = (mag * keep[None, ...]).astype(np.float32)
            return d

        if "lr_vel" in d and d["lr_vel"] is not None:
            vel = np.asarray(d["lr_vel"], dtype=np.float32)
            vel_noise = np.zeros_like(vel, dtype=np.float32)
            for c_idx, c_name in enumerate(["Phase Vx", "Phase Vy", "Phase Vz"]):
                vel_noise[c_idx] = self._sample_noise(
                    shape=vel[c_idx].shape,
                    dist=self.phase_dist,
                    scale=self.phase_scale,
                    shape_param=self.phase_shape,
                    channel_name=c_name,
                )
            vel = vel + vel_noise * noise_weight[None, ...]
            d["lr_vel"] = vel.astype(np.float32)

        if self.apply_to_magnitude and "lr_mag" in d and d["lr_mag"] is not None:
            mag = np.asarray(d["lr_mag"], dtype=np.float32)
            mag_noise = np.zeros_like(mag, dtype=np.float32)
            for c_idx in range(int(mag.shape[0])):
                mag_noise[c_idx] = self._sample_noise(
                    shape=mag[c_idx].shape,
                    dist=self.mag_dist,
                    scale=self.mag_scale,
                    shape_param=self.mag_shape,
                    channel_name="Magnitude",
                )
            mag = mag + mag_noise * noise_weight[None, ...]
            if self.clip_magnitude:
                mag = np.clip(mag, 0.0, 1.0)
            d["lr_mag"] = mag.astype(np.float32)
        return d


class _PairedRandomPatchd(RandomizableTransform):
    def __init__(
        self,
        patch_size: int,
        res_increase: int,
        random_center: bool,
        include_hr_mag: bool = False,
        minimum_coverage: float = 0.0,
        max_sampling_attempts: int = 100,
        allow_empty_fallback: bool = True,
    ):
        super().__init__(prob=1.0)
        self.patch_size = int(patch_size)
        self.res_increase = int(res_increase)
        self.random_center = bool(random_center)
        self.include_hr_mag = bool(include_hr_mag)
        self.minimum_coverage = float(minimum_coverage)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.allow_empty_fallback = bool(allow_empty_fallback)

    def _randint(self, low: int, high: int) -> int:
        return int(self.R.randint(low, high))

    def _crop_pair(self, d, x0, y0, z0):
        p = self.patch_size
        hp = p * self.res_increase
        hx0 = x0 * self.res_increase
        hy0 = y0 * self.res_increase
        hz0 = z0 * self.res_increase

        lr_vel = np.ascontiguousarray(d["lr_vel"][:, x0 : x0 + p, y0 : y0 + p, z0 : z0 + p])
        lr_mag = np.ascontiguousarray(d["lr_mag"][:, x0 : x0 + p, y0 : y0 + p, z0 : z0 + p])
        hr_vel = np.ascontiguousarray(d["hr_vel"][:, hx0 : hx0 + hp, hy0 : hy0 + hp, hz0 : hz0 + hp])
        hr_mag = (
            np.ascontiguousarray(d["hr_mag"][hx0 : hx0 + hp, hy0 : hy0 + hp, hz0 : hz0 + hp])
            if self.include_hr_mag
            else None
        )
        mask = np.ascontiguousarray(d["mask"][hx0 : hx0 + hp, hy0 : hy0 + hp, hz0 : hz0 + hp])
        return lr_vel, lr_mag, hr_vel, hr_mag, mask

    def _compute_valid_lr_starts(self, d):
        _, lx, ly, lz = d["lr_vel"].shape
        _, hx, hy, hz = d["hr_vel"].shape
        mx, my, mz = d["mask"].shape

        p = self.patch_size
        hp = p * self.res_increase

        # LR start must fit LR patch size.
        max_x_lr = lx - p
        max_y_lr = ly - p
        max_z_lr = lz - p

        # LR start is projected to HR space via (start * res_increase), so ensure
        # HR/mask patches are always complete even when HR!=LR*res_increase by 1 voxel.
        max_x_hr = (hx - hp) // self.res_increase
        max_y_hr = (hy - hp) // self.res_increase
        max_z_hr = (hz - hp) // self.res_increase

        max_x_mask = (mx - hp) // self.res_increase
        max_y_mask = (my - hp) // self.res_increase
        max_z_mask = (mz - hp) // self.res_increase

        max_x = min(max_x_lr, max_x_hr, max_x_mask)
        max_y = min(max_y_lr, max_y_hr, max_y_mask)
        max_z = min(max_z_lr, max_z_hr, max_z_mask)

        if max_x < 0 or max_y < 0 or max_z < 0:
            raise ValueError(
                "Patch size is incompatible with paired LR/HR/mask shapes. "
                f"patch_size={p}, hr_patch={hp}, lr_shape={(lx, ly, lz)}, "
                f"hr_shape={(hx, hy, hz)}, mask_shape={(mx, my, mz)}, res_increase={self.res_increase}"
            )
        return int(max_x), int(max_y), int(max_z)

    def __call__(self, data):
        d = dict(data)

        max_x, max_y, max_z = self._compute_valid_lr_starts(d)

        # Validation mode: deterministic center patch.
        if not self.random_center:
            x0 = max_x // 2
            y0 = max_y // 2
            z0 = max_z // 2
            lr_vel, lr_mag, hr_vel, hr_mag, mask = self._crop_pair(d, x0, y0, z0)
            d["lr_vel"], d["lr_mag"], d["hr_vel"], d["mask"] = lr_vel, lr_mag, hr_vel, mask
            if self.include_hr_mag and hr_mag is not None:
                d["hr_mag"] = hr_mag
            return d

        # Training mode: random patch with optional legacy minimum coverage constraint.
        best = None
        best_cov = -1.0

        attempts = max(self.max_sampling_attempts, 1)
        for _ in range(attempts):
            x0 = self._randint(0, max_x + 1)
            y0 = self._randint(0, max_y + 1)
            z0 = self._randint(0, max_z + 1)

            lr_vel, lr_mag, hr_vel, hr_mag, mask = self._crop_pair(d, x0, y0, z0)
            coverage = float(mask.mean())
            if coverage > best_cov:
                best_cov = coverage
                best = (lr_vel, lr_mag, hr_vel, hr_mag, mask)

            if coverage >= self.minimum_coverage:
                d["lr_vel"], d["lr_mag"], d["hr_vel"], d["mask"] = lr_vel, lr_mag, hr_vel, mask
                if self.include_hr_mag and hr_mag is not None:
                    d["hr_mag"] = hr_mag
                return d

        if self.allow_empty_fallback and best is not None:
            d["lr_vel"], d["lr_mag"], d["hr_vel"], hr_mag, d["mask"] = best
            if self.include_hr_mag and hr_mag is not None:
                d["hr_mag"] = hr_mag
            d["patch_coverage"] = np.float32(best_cov)
            return d

        raise RuntimeError(
            f"Unable to find patch meeting minimum coverage {self.minimum_coverage} "
            f"after {attempts} attempts."
        )
        return d


class NiftiPatchDataset(Dataset):
    def __init__(
        self,
        cases: List[Dict],
        patch_size: int,
        res_increase: int,
        samples_per_volume: int,
        augment: bool,
        include_hr_mag: bool = False,
        cache_dataset: bool = False,
        cache_eager: bool = True,
        random_time_frame: bool | None = None,
        random_patch_sampling: bool | None = None,
        rotation_prob: float | None = None,
        mag_scale: float = 4095.0,
        mag_norm_mode: str = "monai_minmax",
        mask_threshold: float = 0.5,
        raw_phase_input: bool = True,
        invert_uv_sign_on_raw: bool = False,
        raw_center: float = 2048.0,
        raw_scale: float = 2048.0,
        time_axis: int = -1,
        minimum_coverage: float = 0.0,
        max_sampling_attempts: int = 100,
        allow_empty_fallback: bool = True,
        noise_aug_prob: float = 0.0,
        noise_aug_phase_dist: str = "student_t",
        noise_aug_phase_scale: float = 0.04,
        noise_aug_phase_shape: float = 3.0,
        noise_aug_mag_dist: str = "skew_normal",
        noise_aug_mag_scale: float = 0.02,
        noise_aug_mag_shape: float = 4.0,
        noise_aug_range_mult: float = 2.0,
        noise_aug_hat_mix: float = 0.7,
        noise_aug_apply_mag: bool = True,
        noise_aug_clip_mag: bool = False,
        noise_aug_fit_summary_csv: str = "",
        noise_aug_exaggerated_side_expand: float = 1.0,
        noise_aug_exaggerated_edge_boost: float = 0.0,
        noise_aug_exaggerated_edge_power: float = 1.8,
        noise_aug_level_min: float = 0.8,
        noise_aug_level_max: float = 1.4,
        noise_aug_masked_fraction: float = 0.0,
        noise_aug_keep_original_prob: float = 0.0,
        noise_aug_zero_outside_prob: float = 0.0,
    ):
        self.cases = list(cases)
        self.samples_per_volume = int(samples_per_volume)
        self.samples_per_volume = max(self.samples_per_volume, 1)
        self.include_hr_mag = bool(include_hr_mag)
        self.cache_dataset = bool(cache_dataset)
        self.cache_eager = bool(cache_eager)

        if random_time_frame is None:
            random_time_frame = augment
        if random_patch_sampling is None:
            random_patch_sampling = augment
        if rotation_prob is None:
            rotation_prob = 0.5 if augment else 0.0

        load_keys = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w", "mask"]
        if self.include_hr_mag:
            load_keys.append("hr_mag")
        self.load_transforms = Compose(
            [
                LoadImaged(keys=load_keys, image_only=True, allow_missing_keys=True),
                EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
            ]
        )
        post_transforms = [
            # Converts loaded arrays to normalized tensors and selects time frame.
            _StackNormalizeFieldsd(
                mag_scale=mag_scale,
                mag_norm_mode=mag_norm_mode,
                mask_threshold=mask_threshold,
                include_hr_mag=self.include_hr_mag,
                raw_phase_input=raw_phase_input,
                invert_uv_sign_on_raw=invert_uv_sign_on_raw,
                raw_center=raw_center,
                raw_scale=raw_scale,
                random_time_frame=bool(random_time_frame),
                time_axis=time_axis,
            ),
        ]
        if float(rotation_prob) > 0:
            post_transforms.append(_RandomVectorRotate90d(prob=float(rotation_prob)))
        post_transforms.append(
            _PairedRandomPatchd(
                patch_size=patch_size,
                res_increase=res_increase,
                random_center=bool(random_patch_sampling),
                include_hr_mag=self.include_hr_mag,
                minimum_coverage=minimum_coverage,
                max_sampling_attempts=max_sampling_attempts,
                allow_empty_fallback=allow_empty_fallback,
            )
        )
        if float(noise_aug_prob) > 0.0:
            post_transforms.append(
                _AddNonvascularNoiseAugd(
                    prob=float(noise_aug_prob),
                    phase_dist=noise_aug_phase_dist,
                    phase_scale=float(noise_aug_phase_scale),
                    phase_shape=float(noise_aug_phase_shape),
                    mag_dist=noise_aug_mag_dist,
                    mag_scale=float(noise_aug_mag_scale),
                    mag_shape=float(noise_aug_mag_shape),
                    range_multiplier=float(noise_aug_range_mult),
                    hat_mix=float(noise_aug_hat_mix),
                    apply_to_magnitude=bool(noise_aug_apply_mag),
                    clip_magnitude=bool(noise_aug_clip_mag),
                    fit_summary_csv=noise_aug_fit_summary_csv,
                    exaggerated_side_expand=float(noise_aug_exaggerated_side_expand),
                    exaggerated_edge_boost=float(noise_aug_exaggerated_edge_boost),
                    exaggerated_edge_power=float(noise_aug_exaggerated_edge_power),
                    level_min=float(noise_aug_level_min),
                    level_max=float(noise_aug_level_max),
                    masked_fraction=float(noise_aug_masked_fraction),
                    keep_original_prob=float(noise_aug_keep_original_prob),
                    zero_outside_prob=float(noise_aug_zero_outside_prob),
                )
            )
        self.post_transforms = Compose(post_transforms)

        case_inputs = [self._prepare_case_dict(case_idx) for case_idx in range(len(self.cases))]
        if self.cache_dataset:
            print(f"Using MONAI CacheDataset for {len(case_inputs)} case(s)...")
            if self.cache_eager:
                print("Cache mode: eager")
            else:
                print("Cache mode: runtime-lazy (if supported by current MONAI)")
        self.case_dataset = _build_monai_case_dataset(
            cases=case_inputs,
            load_transform=self.load_transforms,
            use_cache=self.cache_dataset,
            cache_eager=self.cache_eager,
        )
        if self.cache_dataset:
            print("MONAI CacheDataset ready.")

    def __len__(self):
        return len(self.cases) * self.samples_per_volume

    def __getitem__(self, idx):
        case_idx = idx % len(self.cases)
        sample = self.post_transforms(dict(self.case_dataset[case_idx]))

        lr_vel = torch.from_numpy(sample["lr_vel"]).float()
        lr_mag = torch.from_numpy(sample["lr_mag"]).float()
        hr_vel = torch.from_numpy(sample["hr_vel"]).float()
        hr_mag = torch.from_numpy(sample["hr_mag"]).float() if self.include_hr_mag else None
        mask = torch.from_numpy(sample["mask"]).float()
        venc = torch.tensor(sample["venc"], dtype=torch.float32)

        out = (
            lr_vel[0:1],
            lr_vel[1:2],
            lr_vel[2:3],
            lr_mag[0:1],
            lr_mag[1:2],
            lr_mag[2:3],
            hr_vel[0:1],
            hr_vel[1:2],
            hr_vel[2:3],
        )
        if self.include_hr_mag and hr_mag is not None:
            out = out + (hr_mag.unsqueeze(0),)
        out = out + (venc, mask)
        return out

    def _prepare_case_dict(self, case_idx: int) -> dict:
        case = dict(self.cases[case_idx])
        if not case.get("mask"):
            case.pop("mask", None)
        if not case.get("hr_mag"):
            case.pop("hr_mag", None)
        return case


class NiftiFullVolumeDataset(Dataset):
    """Case-level dataset for full-volume validation (no patch cropping)."""

    def __init__(
        self,
        cases: List[Dict],
        include_hr_mag: bool = False,
        cache_dataset: bool = False,
        cache_eager: bool = True,
        random_time_frame: bool = False,
        mag_scale: float = 4095.0,
        mag_norm_mode: str = "monai_minmax",
        mask_threshold: float = 0.5,
        raw_phase_input: bool = True,
        invert_uv_sign_on_raw: bool = False,
        raw_center: float = 2048.0,
        raw_scale: float = 2048.0,
        time_axis: int = -1,
    ):
        self.cases = list(cases)
        self.include_hr_mag = bool(include_hr_mag)
        self.cache_dataset = bool(cache_dataset)
        self.cache_eager = bool(cache_eager)

        load_keys = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w", "mask"]
        if self.include_hr_mag:
            load_keys.append("hr_mag")
        self.load_transforms = Compose(
            [
                LoadImaged(keys=load_keys, image_only=True, allow_missing_keys=True),
                EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
            ]
        )
        self.post_transforms = Compose(
            [
                _StackNormalizeFieldsd(
                    mag_scale=mag_scale,
                    mag_norm_mode=mag_norm_mode,
                    mask_threshold=mask_threshold,
                    include_hr_mag=self.include_hr_mag,
                    raw_phase_input=raw_phase_input,
                    invert_uv_sign_on_raw=invert_uv_sign_on_raw,
                    raw_center=raw_center,
                    raw_scale=raw_scale,
                    random_time_frame=bool(random_time_frame),
                    time_axis=time_axis,
                )
            ]
        )

        case_inputs = [self._prepare_case_dict(case_idx) for case_idx in range(len(self.cases))]
        if self.cache_dataset:
            print(f"Using MONAI CacheDataset (full-volume) for {len(case_inputs)} case(s)...")
            if self.cache_eager:
                print("Cache mode: eager")
            else:
                print("Cache mode: runtime-lazy (if supported by current MONAI)")
        self.case_dataset = _build_monai_case_dataset(
            cases=case_inputs,
            load_transform=self.load_transforms,
            use_cache=self.cache_dataset,
            cache_eager=self.cache_eager,
        )
        if self.cache_dataset:
            print("MONAI CacheDataset (full-volume) ready.")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        sample = self.post_transforms(dict(self.case_dataset[idx]))

        lr_vel = torch.from_numpy(sample["lr_vel"]).float()
        lr_mag = torch.from_numpy(sample["lr_mag"]).float()
        hr_vel = torch.from_numpy(sample["hr_vel"]).float()
        hr_mag = torch.from_numpy(sample["hr_mag"]).float() if self.include_hr_mag else None
        mask = torch.from_numpy(sample["mask"]).float()
        venc = torch.tensor(sample["venc"], dtype=torch.float32)

        out = (
            lr_vel[0:1],
            lr_vel[1:2],
            lr_vel[2:3],
            lr_mag[0:1],
            lr_mag[1:2],
            lr_mag[2:3],
            hr_vel[0:1],
            hr_vel[1:2],
            hr_vel[2:3],
        )
        if self.include_hr_mag and hr_mag is not None:
            out = out + (hr_mag.unsqueeze(0),)
        out = out + (venc, mask)
        return out

    def _prepare_case_dict(self, case_idx: int) -> dict:
        case = dict(self.cases[case_idx])
        if not case.get("mask"):
            case.pop("mask", None)
        if not case.get("hr_mag"):
            case.pop("hr_mag", None)
        return case


def create_nifti_patch_dataloader(
    csv_path: str,
    patch_size: int,
    res_increase: int,
    batch_size: int,
    samples_per_volume: int,
    shuffle: bool,
    augment: bool,
    include_hr_mag: bool = False,
    cache_dataset: bool = False,
    cache_eager: bool = True,
    random_time_frame: bool | None = None,
    random_patch_sampling: bool | None = None,
    rotation_prob: float | None = None,
    num_workers: int = 0,
    mag_scale: float = 4095.0,
    mag_norm_mode: str = "monai_minmax",
    mask_threshold: float = 0.5,
    raw_phase_input: bool = True,
    invert_uv_sign_on_raw: bool = False,
    raw_center: float = 2048.0,
    raw_scale: float = 2048.0,
    time_axis: int = -1,
    minimum_coverage: float = 0.0,
    max_sampling_attempts: int = 100,
    allow_empty_fallback: bool = True,
    noise_aug_prob: float = 0.0,
    noise_aug_phase_dist: str = "student_t",
    noise_aug_phase_scale: float = 0.04,
    noise_aug_phase_shape: float = 3.0,
    noise_aug_mag_dist: str = "skew_normal",
    noise_aug_mag_scale: float = 0.02,
    noise_aug_mag_shape: float = 4.0,
    noise_aug_range_mult: float = 2.0,
    noise_aug_hat_mix: float = 0.7,
    noise_aug_apply_mag: bool = True,
    noise_aug_clip_mag: bool = False,
    noise_aug_fit_summary_csv: str = "",
    noise_aug_exaggerated_side_expand: float = 1.0,
    noise_aug_exaggerated_edge_boost: float = 0.0,
    noise_aug_exaggerated_edge_power: float = 1.8,
    noise_aug_level_min: float = 0.8,
    noise_aug_level_max: float = 1.4,
    noise_aug_masked_fraction: float = 0.0,
    noise_aug_keep_original_prob: float = 0.0,
    noise_aug_zero_outside_prob: float = 0.0,
    seed: Optional[int] = None,
):
    cases = load_nifti_case_table(csv_path, include_hr_mag=include_hr_mag)
    dataset = NiftiPatchDataset(
        cases=cases,
        patch_size=patch_size,
        res_increase=res_increase,
        samples_per_volume=samples_per_volume,
        augment=augment,
        include_hr_mag=include_hr_mag,
        cache_dataset=cache_dataset,
        cache_eager=cache_eager,
        random_time_frame=random_time_frame,
        random_patch_sampling=random_patch_sampling,
        rotation_prob=rotation_prob,
        mag_scale=mag_scale,
        mag_norm_mode=mag_norm_mode,
        mask_threshold=mask_threshold,
        raw_phase_input=raw_phase_input,
        invert_uv_sign_on_raw=invert_uv_sign_on_raw,
        raw_center=raw_center,
        raw_scale=raw_scale,
        time_axis=time_axis,
        minimum_coverage=minimum_coverage,
        max_sampling_attempts=max_sampling_attempts,
        allow_empty_fallback=allow_empty_fallback,
        noise_aug_prob=noise_aug_prob,
        noise_aug_phase_dist=noise_aug_phase_dist,
        noise_aug_phase_scale=noise_aug_phase_scale,
        noise_aug_phase_shape=noise_aug_phase_shape,
        noise_aug_mag_dist=noise_aug_mag_dist,
        noise_aug_mag_scale=noise_aug_mag_scale,
        noise_aug_mag_shape=noise_aug_mag_shape,
        noise_aug_range_mult=noise_aug_range_mult,
        noise_aug_hat_mix=noise_aug_hat_mix,
        noise_aug_apply_mag=noise_aug_apply_mag,
        noise_aug_clip_mag=noise_aug_clip_mag,
        noise_aug_fit_summary_csv=noise_aug_fit_summary_csv,
        noise_aug_exaggerated_side_expand=noise_aug_exaggerated_side_expand,
        noise_aug_exaggerated_edge_boost=noise_aug_exaggerated_edge_boost,
        noise_aug_exaggerated_edge_power=noise_aug_exaggerated_edge_power,
        noise_aug_level_min=noise_aug_level_min,
        noise_aug_level_max=noise_aug_level_max,
        noise_aug_masked_fraction=noise_aug_masked_fraction,
        noise_aug_keep_original_prob=noise_aug_keep_original_prob,
        noise_aug_zero_outside_prob=noise_aug_zero_outside_prob,
    )
    print(f"NIfTI dataset {csv_path}: {len(cases)} volume(s), {len(dataset)} patch samples")
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def _seed_worker(worker_id):
            worker_seed = int(seed) + int(worker_id)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return loader


def create_nifti_full_volume_dataloader(
    csv_path: str,
    batch_size: int,
    shuffle: bool,
    include_hr_mag: bool = False,
    cache_dataset: bool = False,
    cache_eager: bool = True,
    random_time_frame: bool = False,
    num_workers: int = 0,
    mag_scale: float = 4095.0,
    mag_norm_mode: str = "monai_minmax",
    mask_threshold: float = 0.5,
    raw_phase_input: bool = True,
    invert_uv_sign_on_raw: bool = False,
    raw_center: float = 2048.0,
    raw_scale: float = 2048.0,
    time_axis: int = -1,
    seed: Optional[int] = None,
):
    cases = load_nifti_case_table(csv_path, include_hr_mag=include_hr_mag)
    dataset = NiftiFullVolumeDataset(
        cases=cases,
        include_hr_mag=include_hr_mag,
        cache_dataset=cache_dataset,
        cache_eager=cache_eager,
        random_time_frame=random_time_frame,
        mag_scale=mag_scale,
        mag_norm_mode=mag_norm_mode,
        mask_threshold=mask_threshold,
        raw_phase_input=raw_phase_input,
        invert_uv_sign_on_raw=invert_uv_sign_on_raw,
        raw_center=raw_center,
        raw_scale=raw_scale,
        time_axis=time_axis,
    )
    print(f"NIfTI full-volume dataset {csv_path}: {len(cases)} volume(s)")
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def _seed_worker(worker_id):
            worker_seed = int(seed) + int(worker_id)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return loader
