"""Shared helpers for ANTs-based registration workflows."""

from __future__ import annotations

from pathlib import Path

import ants
import numpy as np


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    """Extract one 3D frame from a 4D ANTs image, preserving 3D geometry metadata."""
    array = img4d.numpy()[..., t]
    return ants.from_numpy(
        array,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )


def stack_4d(frames: list["ants.ANTsImage"], template_4d: "ants.ANTsImage") -> "ants.ANTsImage":
    """Stack 3D frames back into one 4D image using the template 4D metadata."""
    array = np.stack([frame.numpy() for frame in frames], axis=-1)
    return ants.from_numpy(
        array,
        origin=template_4d.origin,
        spacing=tuple(template_4d.spacing),
        direction=template_4d.direction,
    )


def make_mask_reference(ref3d: "ants.ANTsImage") -> "ants.ANTsImage":
    """Build a robust foreground mask from a reference image."""
    mask = ants.get_mask(ref3d, cleanup=2)
    mask = ants.iMath(mask, "FillHoles")
    mask = ants.iMath(mask, "GetLargestComponent")
    return mask


def parse_qc_frames(qc_frames_str: str, total_frames: int, ref_t: int) -> list[int]:
    """Parse percentile string (e.g. '0,50,99') into concrete frame indices."""
    percentiles: list[int] = []
    for token in qc_frames_str.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 99:
            raise ValueError(f"QC percentile must be in [0, 99], got: {value}")
        percentiles.append(value)

    indices = sorted(set(int(round((p / 100.0) * (total_frames - 1))) for p in percentiles))
    if ref_t not in indices:
        indices.append(ref_t)
    return sorted(set(indices))


def norm01(array: np.ndarray, p_lo: float = 1, p_hi: float = 99) -> np.ndarray:
    """Robust [0, 1] normalization using finite-value percentiles."""
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    lo, hi = np.percentile(values, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)

    clipped = np.clip(array, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def mean_abs_diff(a_img: "ants.ANTsImage", b_img: "ants.ANTsImage", mask: "ants.ANTsImage" | None = None) -> float:
    """Compute mean absolute difference, optionally restricted to mask > 0."""
    a = a_img.numpy()
    b = b_img.numpy()

    if mask is None:
        return float(np.mean(np.abs(a - b)))

    m = mask.numpy() > 0
    if not np.any(m):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[m] - b[m])))


def normalized_cross_correlation(
    a_img: "ants.ANTsImage",
    b_img: "ants.ANTsImage",
    mask: "ants.ANTsImage" | None = None,
) -> float:
    """Compute normalized cross-correlation, optionally inside mask > 0."""
    a = a_img.numpy().astype(np.float64)
    b = b_img.numpy().astype(np.float64)

    if mask is not None:
        m = mask.numpy() > 0
        if np.any(m):
            a = a[m]
            b = b[m]
        else:
            a = a.ravel()
            b = b.ravel()
    else:
        a = a.ravel()
        b = b.ravel()

    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)
