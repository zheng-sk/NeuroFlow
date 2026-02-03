import csv
import os
from typing import Dict, List

import numpy as np
import torch
from monai.data import DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Transform
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
    # Primary behavior: resolve relative to CSV directory.
    candidate_csv_dir = os.path.abspath(os.path.join(base_dir, path_value))
    if os.path.exists(candidate_csv_dir):
        return candidate_csv_dir

    # Fallback 1: resolve from current working directory.
    candidate_cwd = os.path.abspath(os.path.join(os.getcwd(), path_value))
    if os.path.exists(candidate_cwd):
        return candidate_cwd

    # Fallback 2: resolve from parent of CSV directory (helps when CSV lives in data/
    # and paths are given as data/... project-relative).
    candidate_csv_parent = os.path.abspath(os.path.join(os.path.dirname(base_dir), path_value))
    if os.path.exists(candidate_csv_parent):
        return candidate_csv_parent

    # Keep deterministic behavior for downstream error messages.
    return candidate_csv_dir


def load_nifti_case_table(csv_path: str) -> List[Dict]:
    """
    Load case descriptors from CSV.

    Required columns:
        lr_u, lr_v, lr_w, lr_mag_u, lr_mag_v, lr_mag_w, hr_u, hr_v, hr_w
    Optional columns:
        mask, venc, venc_u, venc_v, venc_w, time_start, time_end, time_index
    """
    required = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w"]
    optional = ["mask", "venc", "venc_u", "venc_v", "venc_w", "time_start", "time_end", "time_index"]

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
                elif key == "mask":
                    case[key] = _resolve_path(value, base_dir) if str(value).strip() else None
                else:
                    case[key] = _resolve_path(value, base_dir)
            cases.append(case)
    return cases


class _StackNormalizeFieldsd(Transform):
    def __init__(
        self,
        mag_scale: float,
        mask_threshold: float,
        raw_phase_input: bool = True,
        raw_center: float = 2048.0,
        raw_scale: float = 2048.0,
        random_time_frame: bool = False,
        time_axis: int = -1,
        fixed_time_index: int = 0,
    ):
        self.mag_scale = float(mag_scale)
        self.mask_threshold = float(mask_threshold)
        self.raw_phase_input = bool(raw_phase_input)
        self.raw_center = float(raw_center)
        self.raw_scale = float(raw_scale)
        self.random_time_frame = bool(random_time_frame)
        self.time_axis = int(time_axis)
        self.fixed_time_index = int(fixed_time_index)

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

    def _raw_to_velocity(self, arr, venc, invert_sign=False):
        vel = (arr.astype(np.float32) - self.raw_center) / self.raw_scale * float(venc)
        if invert_sign:
            vel = -vel
        return vel

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
        mask_raw = d["mask"][0].astype(np.float32) if "mask" in d and d["mask"] is not None else None

        # 4D NIfTI support: sample/select one time frame and continue with 3D patching.
        if lr_u_raw.ndim == 4:
            time_lengths = []
            for arr in [lr_u_raw, lr_v_raw, lr_w_raw, hr_u_raw, hr_v_raw, hr_w_raw, mag_u_raw, mag_v_raw, mag_w_raw]:
                axis = self._normalize_time_axis(arr.ndim)
                time_lengths.append(arr.shape[axis])
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
                t_index = int(np.random.randint(time_start, time_end))
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

            # Same semantic behavior as legacy prepare_nifti_data.py:
            # U/V sign inversion, W unchanged.
            lr_u = self._raw_to_velocity(lr_u, venc=venc_u, invert_sign=True)
            lr_v = self._raw_to_velocity(lr_v, venc=venc_v, invert_sign=True)
            lr_w = self._raw_to_velocity(lr_w, venc=venc_w, invert_sign=False)
            hr_u = self._raw_to_velocity(hr_u, venc=venc_u, invert_sign=True)
            hr_v = self._raw_to_velocity(hr_v, venc=venc_v, invert_sign=True)
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
        lr_mag = np.stack([mag_u, mag_v, mag_w], axis=0).astype(np.float32)

        venc = float(max(venc_u, venc_v, venc_w))
        if venc <= 0:
            venc = float(np.max(np.abs(lr_vel)))
        if venc <= 0:
            venc = 1.0

        lr_vel = lr_vel / venc
        hr_vel = hr_vel / venc
        lr_mag = lr_mag / self.mag_scale

        if mask is not None:
            mask = (mask >= self.mask_threshold).astype(np.float32)
        else:
            mask = np.ones(hr_vel.shape[1:], dtype=np.float32)

        d["lr_vel"] = lr_vel
        d["hr_vel"] = hr_vel
        d["lr_mag"] = lr_mag
        d["mask"] = mask
        d["venc"] = np.float32(venc)
        return d


class _RandomVectorRotate90d(Transform):
    def __init__(self, prob: float):
        self.prob = float(prob)

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
        if np.random.rand() >= self.prob:
            return d

        rotation_idx = int(np.random.choice([1, 2, 3]))
        plane_nr = int(np.random.choice([1, 2, 3]))

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
        d["mask"] = self._rotate_scalar(d["mask"], rotation_idx, plane_nr).astype(np.float32)
        return d


class _PairedRandomPatchd(Transform):
    def __init__(
        self,
        patch_size: int,
        res_increase: int,
        random_center: bool,
        minimum_coverage: float = 0.0,
        max_sampling_attempts: int = 100,
        allow_empty_fallback: bool = True,
    ):
        self.patch_size = int(patch_size)
        self.res_increase = int(res_increase)
        self.random_center = bool(random_center)
        self.minimum_coverage = float(minimum_coverage)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.allow_empty_fallback = bool(allow_empty_fallback)

    def _crop_pair(self, d, x0, y0, z0):
        p = self.patch_size
        hp = p * self.res_increase
        hx0 = x0 * self.res_increase
        hy0 = y0 * self.res_increase
        hz0 = z0 * self.res_increase

        lr_vel = np.ascontiguousarray(d["lr_vel"][:, x0 : x0 + p, y0 : y0 + p, z0 : z0 + p])
        lr_mag = np.ascontiguousarray(d["lr_mag"][:, x0 : x0 + p, y0 : y0 + p, z0 : z0 + p])
        hr_vel = np.ascontiguousarray(d["hr_vel"][:, hx0 : hx0 + hp, hy0 : hy0 + hp, hz0 : hz0 + hp])
        mask = np.ascontiguousarray(d["mask"][hx0 : hx0 + hp, hy0 : hy0 + hp, hz0 : hz0 + hp])
        return lr_vel, lr_mag, hr_vel, mask

    def __call__(self, data):
        d = dict(data)

        _, lx, ly, lz = d["lr_vel"].shape
        p = self.patch_size
        hp = p * self.res_increase

        if lx < p or ly < p or lz < p:
            raise ValueError(f"Patch size {p} is larger than LR image size {(lx, ly, lz)}")

        # Validation mode: deterministic center patch.
        if not self.random_center:
            x0 = (lx - p) // 2
            y0 = (ly - p) // 2
            z0 = (lz - p) // 2
            lr_vel, lr_mag, hr_vel, mask = self._crop_pair(d, x0, y0, z0)
            d["lr_vel"], d["lr_mag"], d["hr_vel"], d["mask"] = lr_vel, lr_mag, hr_vel, mask
            return d

        # Training mode: random patch with optional legacy minimum coverage constraint.
        best = None
        best_cov = -1.0

        attempts = max(self.max_sampling_attempts, 1)
        for _ in range(attempts):
            x0 = np.random.randint(0, lx - p + 1)
            y0 = np.random.randint(0, ly - p + 1)
            z0 = np.random.randint(0, lz - p + 1)

            lr_vel, lr_mag, hr_vel, mask = self._crop_pair(d, x0, y0, z0)
            coverage = float(mask.mean())
            if coverage > best_cov:
                best_cov = coverage
                best = (lr_vel, lr_mag, hr_vel, mask)

            if coverage >= self.minimum_coverage:
                d["lr_vel"], d["lr_mag"], d["hr_vel"], d["mask"] = lr_vel, lr_mag, hr_vel, mask
                return d

        if self.allow_empty_fallback and best is not None:
            d["lr_vel"], d["lr_mag"], d["hr_vel"], d["mask"] = best
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
        mag_scale: float = 4095.0,
        mask_threshold: float = 0.5,
        raw_phase_input: bool = True,
        raw_center: float = 2048.0,
        raw_scale: float = 2048.0,
        time_axis: int = -1,
        minimum_coverage: float = 0.0,
        max_sampling_attempts: int = 100,
        allow_empty_fallback: bool = True,
    ):
        self.cases = list(cases)
        self.samples_per_volume = int(samples_per_volume)
        self.samples_per_volume = max(self.samples_per_volume, 1)

        load_keys = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w", "mask"]
        transforms = [
            LoadImaged(keys=load_keys, image_only=True, allow_missing_keys=True),
            EnsureChannelFirstd(keys=load_keys, channel_dim="no_channel", allow_missing_keys=True),
            _StackNormalizeFieldsd(
                mag_scale=mag_scale,
                mask_threshold=mask_threshold,
                raw_phase_input=raw_phase_input,
                raw_center=raw_center,
                raw_scale=raw_scale,
                random_time_frame=augment,
                time_axis=time_axis,
            ),
        ]
        if augment:
            transforms.append(_RandomVectorRotate90d(prob=0.5))
        transforms.append(
            _PairedRandomPatchd(
                patch_size=patch_size,
                res_increase=res_increase,
                random_center=augment,
                minimum_coverage=minimum_coverage,
                max_sampling_attempts=max_sampling_attempts,
                allow_empty_fallback=allow_empty_fallback,
            )
        )
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.cases) * self.samples_per_volume

    def __getitem__(self, idx):
        case_idx = idx % len(self.cases)
        case = dict(self.cases[case_idx])
        if not case.get("mask"):
            case.pop("mask", None)
        sample = self.transforms(case)

        lr_vel = torch.from_numpy(sample["lr_vel"]).float()
        lr_mag = torch.from_numpy(sample["lr_mag"]).float()
        hr_vel = torch.from_numpy(sample["hr_vel"]).float()
        mask = torch.from_numpy(sample["mask"]).float()
        venc = torch.tensor(sample["venc"], dtype=torch.float32)

        return (
            lr_vel[0:1],
            lr_vel[1:2],
            lr_vel[2:3],
            lr_mag[0:1],
            lr_mag[1:2],
            lr_mag[2:3],
            hr_vel[0:1],
            hr_vel[1:2],
            hr_vel[2:3],
            venc,
            mask,
        )


def create_nifti_patch_dataloader(
    csv_path: str,
    patch_size: int,
    res_increase: int,
    batch_size: int,
    samples_per_volume: int,
    shuffle: bool,
    augment: bool,
    num_workers: int = 0,
    mag_scale: float = 4095.0,
    mask_threshold: float = 0.5,
    raw_phase_input: bool = True,
    raw_center: float = 2048.0,
    raw_scale: float = 2048.0,
    time_axis: int = -1,
    minimum_coverage: float = 0.0,
    max_sampling_attempts: int = 100,
    allow_empty_fallback: bool = True,
):
    cases = load_nifti_case_table(csv_path)
    dataset = NiftiPatchDataset(
        cases=cases,
        patch_size=patch_size,
        res_increase=res_increase,
        samples_per_volume=samples_per_volume,
        augment=augment,
        mag_scale=mag_scale,
        mask_threshold=mask_threshold,
        raw_phase_input=raw_phase_input,
        raw_center=raw_center,
        raw_scale=raw_scale,
        time_axis=time_axis,
        minimum_coverage=minimum_coverage,
        max_sampling_attempts=max_sampling_attempts,
        allow_empty_fallback=allow_empty_fallback,
    )
    print(f"NIfTI dataset {csv_path}: {len(cases)} volume(s), {len(dataset)} patch samples")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return loader
