"""QC plotting utilities shared across registration workflows."""

from __future__ import annotations

import os
from typing import Iterable

import ants
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .common import frame3d_from_4d, norm01
except ImportError:  # pragma: no cover - allows direct script execution from this folder
    from common import frame3d_from_4d, norm01


def save_alpha_overlay(
    ref3d: "ants.ANTsImage",
    mov3d: "ants.ANTsImage",
    mask3d: "ants.ANTsImage" | None,
    out_png: str,
    title: str,
    alpha: float = 0.30,
) -> None:
    ref = ref3d.numpy()
    mov = mov3d.numpy()
    mask = (mask3d.numpy() > 0) if mask3d is not None else np.ones_like(ref, dtype=bool)

    z = ref.shape[2] // 2
    ref2d = norm01(ref[:, :, z].astype(np.float32)) * mask[:, :, z]
    mov2d = norm01(mov[:, :, z].astype(np.float32)) * mask[:, :, z]

    plt.figure(figsize=(6, 6))
    plt.imshow(ref2d, cmap="gray", interpolation="nearest")
    plt.imshow(mov2d, cmap="hot", interpolation="nearest", alpha=alpha)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_edge_overlay(
    ref3d: "ants.ANTsImage",
    mov3d: "ants.ANTsImage",
    mask3d: "ants.ANTsImage" | None,
    out_png: str,
    title: str,
    alpha_max: float = 0.95,
    smooth_sigma: float = 1.0,
    edge_thresh: float = 0.25,
    dilate_iters: int = 1,
) -> None:
    try:
        moving = ants.smooth_image(mov3d, smooth_sigma) if smooth_sigma and smooth_sigma > 0 else mov3d
    except Exception:
        moving = mov3d

    ref = ref3d.numpy()
    mov = moving.numpy()
    mask = (mask3d.numpy() > 0) if mask3d is not None else np.ones_like(ref, dtype=bool)

    z = ref.shape[2] // 2
    ref2d = norm01(ref[:, :, z].astype(np.float32))
    mov2d = norm01(mov[:, :, z].astype(np.float32)) * mask[:, :, z]

    gx = np.abs(np.diff(mov2d, axis=1, prepend=mov2d[:, :1]))
    gy = np.abs(np.diff(mov2d, axis=0, prepend=mov2d[:1, :]))
    edge = gx + gy

    vals = edge[np.isfinite(edge)]
    if vals.size == 0:
        edge01 = np.zeros_like(edge, dtype=np.float32)
    else:
        edge_high = np.percentile(vals, 99.5)
        edge01 = np.zeros_like(edge, dtype=np.float32) if edge_high <= 1e-8 else np.clip(edge / (edge_high + 1e-8), 0, 1).astype(np.float32)

    edge01 = np.where(edge01 >= edge_thresh, edge01, 0.0).astype(np.float32)

    def dilate2d(array: np.ndarray) -> np.ndarray:
        shifts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifts.append(np.roll(np.roll(array, dy, axis=0), dx, axis=1))
        return np.maximum.reduce(shifts)

    for _ in range(max(0, int(dilate_iters))):
        edge01 = dilate2d(edge01)

    rgba = np.zeros((edge01.shape[0], edge01.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 1.0
    rgba[..., 2] = 0.0
    rgba[..., 3] = np.clip(edge01 * alpha_max, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(ref2d, cmap="gray", interpolation="nearest")
    plt.imshow(rgba, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_absdiff_png(ref3d: "ants.ANTsImage", mov3d: "ants.ANTsImage", out_png: str, title: str) -> None:
    ref = ref3d.numpy()
    mov = mov3d.numpy()
    z = ref.shape[2] // 2
    diff = np.abs(ref[:, :, z].astype(np.float32) - mov[:, :, z].astype(np.float32))
    diff01 = norm01(diff)

    plt.figure(figsize=(6, 6))
    plt.imshow(diff01, cmap="magma", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_metric_plot(values_before: np.ndarray, values_after: np.ndarray, out_png: str, title: str, y_label: str) -> None:
    plt.figure()
    plt.plot(values_before, label="Before")
    plt.plot(values_after, label="After")
    plt.xlabel("Frame t")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_triptych_qc_examples(
    reference: "ants.ANTsImage",
    original_4d: "ants.ANTsImage",
    warped_frames: list["ants.ANTsImage"],
    qc_indices: Iterable[int],
    ref_t: int,
    qc_dir: str,
) -> None:
    reference_slice = reference.numpy()[:, :, reference.shape[2] // 2]

    for t in qc_indices:
        moving_slice = frame3d_from_4d(original_4d, t).numpy()[:, :, reference.shape[2] // 2]
        warped_slice = warped_frames[t].numpy()[:, :, reference.shape[2] // 2]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(reference_slice, cmap="gray")
        plt.title(f"Reference (t={ref_t})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(moving_slice, cmap="gray")
        plt.title(f"Before (t={t})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(warped_slice, cmap="gray")
        plt.title(f"After (t={t})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(qc_dir, f"qc_t{t:03d}.png"), dpi=150)
        plt.close()
