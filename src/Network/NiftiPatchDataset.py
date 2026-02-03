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
    return os.path.abspath(os.path.join(base_dir, path_value))


def load_nifti_case_table(csv_path: str) -> List[Dict]:
    """
    Load case descriptors from CSV.

    Required columns:
        lr_u, lr_v, lr_w, lr_mag_u, lr_mag_v, lr_mag_w, hr_u, hr_v, hr_w
    Optional columns:
        mask, venc
    """
    required = ["lr_u", "lr_v", "lr_w", "lr_mag_u", "lr_mag_v", "lr_mag_w", "hr_u", "hr_v", "hr_w"]
    optional = ["mask", "venc"]

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
                if key == "venc":
                    case[key] = float(value) if str(value).strip() else 0.0
                else:
                    case[key] = _resolve_path(value, base_dir)
            cases.append(case)
    return cases


class _StackNormalizeFieldsd(Transform):
    def __init__(self, mag_scale: float, mask_threshold: float):
        self.mag_scale = float(mag_scale)
        self.mask_threshold = float(mask_threshold)

    def __call__(self, data):
        d = dict(data)

        lr_vel = np.concatenate([d["lr_u"], d["lr_v"], d["lr_w"]], axis=0).astype(np.float32)
        hr_vel = np.concatenate([d["hr_u"], d["hr_v"], d["hr_w"]], axis=0).astype(np.float32)
        lr_mag = np.concatenate([d["lr_mag_u"], d["lr_mag_v"], d["lr_mag_w"]], axis=0).astype(np.float32)

        venc = float(d.get("venc", 0.0))
        if venc <= 0:
            venc = float(np.max(np.abs(lr_vel)))
        if venc <= 0:
            venc = 1.0

        lr_vel = lr_vel / venc
        hr_vel = hr_vel / venc
        lr_mag = lr_mag / self.mag_scale

        if "mask" in d and d["mask"] is not None:
            mask = d["mask"]
            if mask.ndim == 4:
                mask = mask[0]
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
            _StackNormalizeFieldsd(mag_scale=mag_scale, mask_threshold=mask_threshold),
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
        sample = self.transforms(dict(self.cases[case_idx]))

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
