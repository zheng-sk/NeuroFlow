#!/usr/bin/env python3
"""Batch inference runner for 4DFlowNet (H5 or direct NIfTI inputs)."""

from __future__ import annotations

import argparse
import glob
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from Network.PatchGenerator import PatchGenerator
from Network.SR4DFlowNet import SR4DFlowNet
from utils import prediction_utils
from utils.ImageDataset import ImageDataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "h5_lr"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "predictions"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "4DFlowNet_cerebro" / "4DFlowNet_cerebro.h5"


@dataclass(frozen=True)
class NiftiCase:
    case_name: str
    rel_path: str
    case_dir: Path
    u_path: Path
    v_path: Path
    w_path: Path
    mag_u_path: Path
    mag_v_path: Path
    mag_w_path: Path


def prepare_network(patch_size: int, res_increase: int, low_resblock: int, hi_resblock: int) -> tf.keras.Model:
    input_shape = (patch_size, patch_size, patch_size, 1)
    u = tf.keras.layers.Input(shape=input_shape, name="u")
    v = tf.keras.layers.Input(shape=input_shape, name="v")
    w = tf.keras.layers.Input(shape=input_shape, name="w")

    u_mag = tf.keras.layers.Input(shape=input_shape, name="u_mag")
    v_mag = tf.keras.layers.Input(shape=input_shape, name="v_mag")
    w_mag = tf.keras.layers.Input(shape=input_shape, name="w_mag")

    net = SR4DFlowNet(res_increase)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
    return tf.keras.Model([u, v, w, u_mag, v_mag, w_mag], prediction)


def resolve_model_path(model_path: Path) -> Path | None:
    if model_path.exists():
        return model_path

    print(f"Error: model not found at {model_path}")
    candidates = sorted(glob.glob(str(REPO_ROOT / "models" / "**" / "4DFlowNet-best.h5"), recursive=True))
    if not candidates:
        return None

    fallback = Path(candidates[0])
    print(f"Found alternative model: {fallback}")
    return fallback


def is_metal_gpu_error(message: str) -> bool:
    markers = (
        "kIOGPUCommandBufferCallbackError",
        "Metal Performance Shaders operations encoded on it may not have completed",
        "GPU Address Fault Error",
        "command buffer exited with error status",
    )
    return any(marker in message for marker in markers)


class ModelExecutor:
    """Wraps model prediction with optional GPU->CPU fallback on Apple Metal faults."""

    def __init__(self, primary_model: tf.keras.Model, cpu_fallback_model: tf.keras.Model | None = None):
        self._model = primary_model
        self._cpu_fallback_model = cpu_fallback_model
        self.using_cpu_fallback = False

    def predict(self, inputs) -> np.ndarray:
        try:
            outputs = self._model(inputs, training=False)
            return outputs.numpy() if hasattr(outputs, "numpy") else np.asarray(outputs)
        except Exception as exc:
            message = str(exc)
            if self._cpu_fallback_model is not None and is_metal_gpu_error(message):
                print("Warning: Metal GPU error detected. Switching to CPU fallback model and continuing.")
                self._model = self._cpu_fallback_model
                self._cpu_fallback_model = None
                self.using_cpu_fallback = True
                outputs = self._model(inputs, training=False)
                return outputs.numpy() if hasattr(outputs, "numpy") else np.asarray(outputs)
            raise


def infer_frame_components(
    model_executor: ModelExecutor,
    patch_generator: PatchGenerator,
    dataset,
    batch_size: int,
    round_values: bool,
    show_patch_progress: bool,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocities, magnitudes = patch_generator.patchify(dataset)
    num_patches = len(velocities[0])

    prediction_chunks = []
    batch_starts = range(0, num_patches, batch_size)
    iterator = tqdm(
        batch_starts,
        desc=progress_label,
        unit="batch",
        leave=False,
    ) if show_patch_progress else batch_starts

    for start in iterator:
        patch_slice = np.index_exp[start : start + batch_size]
        prediction_chunks.append(
            model_executor.predict(
                [
                    velocities[0][patch_slice],
                    velocities[1][patch_slice],
                    velocities[2][patch_slice],
                    magnitudes[0][patch_slice],
                    magnitudes[1][patch_slice],
                    magnitudes[2][patch_slice],
                ]
            )
        )

    results = np.concatenate(prediction_chunks, axis=0)

    components = []
    for component_idx in range(3):
        component = patch_generator._patchup_with_overlap(
            results[:, :, :, :, component_idx],
            patch_generator.nr_x,
            patch_generator.nr_y,
            patch_generator.nr_z,
        )
        component = component * dataset.venc

        if round_values and hasattr(dataset, "velocity_per_px"):
            component[np.abs(component) < dataset.velocity_per_px] = 0
        components.append(component.astype(np.float32))

    return components[0], components[1], components[2]


def predict_patient_h5(
    model_executor: ModelExecutor,
    input_filepath: str,
    output_filepath: str,
    patch_size: int,
    res_increase: int,
    batch_size: int = 8,
    round_values: bool = True,
    show_patch_progress: bool = False,
    verbose: bool = False,
) -> tuple[int, float]:
    print(f"Loading data from: {input_filepath}")

    patch_generator = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    num_frames = dataset.get_dataset_len(input_filepath)
    if num_frames == 0:
        print(f"Skipping {input_filepath}: no data found")
        return 0, 0.0

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    case_start = time.perf_counter()

    for frame_idx in range(num_frames):
        frame_start = time.perf_counter()
        print(f"   Frame {frame_idx + 1}/{num_frames} ...")
        dataset.load_vectorfield(input_filepath, frame_idx)
        comp_u, comp_v, comp_w = infer_frame_components(
            model_executor=model_executor,
            patch_generator=patch_generator,
            dataset=dataset,
            batch_size=batch_size,
            round_values=round_values,
            show_patch_progress=show_patch_progress,
            progress_label=f"{Path(input_filepath).name} f{frame_idx + 1}/{num_frames}",
        )

        prediction_utils.save_to_h5(
            output_filepath,
            dataset.velocity_colnames[0],
            np.expand_dims(comp_u, axis=0),
            compression="gzip",
        )
        prediction_utils.save_to_h5(
            output_filepath,
            dataset.velocity_colnames[1],
            np.expand_dims(comp_v, axis=0),
            compression="gzip",
        )
        prediction_utils.save_to_h5(
            output_filepath,
            dataset.velocity_colnames[2],
            np.expand_dims(comp_w, axis=0),
            compression="gzip",
        )

        if dataset.dx is not None:
            new_spacing = np.expand_dims(dataset.dx / res_increase, axis=0)
            prediction_utils.save_to_h5(output_filepath, dataset.dx_colname, new_spacing, compression="gzip")

        if verbose:
            frame_seconds = time.perf_counter() - frame_start
            print(f"      done in {frame_seconds:.2f}s")

    return num_frames, time.perf_counter() - case_start


class NiftiInferenceDataset:
    velocity_colnames = ["u", "v", "w"]
    dx_colname = "dx"

    def __init__(
        self,
        case: NiftiCase,
        time_axis: int,
        mag_scale: float,
        auto_convert_raw_phase: bool,
        venc: float | None,
        venc_u: float | None,
        venc_v: float | None,
        venc_w: float | None,
    ) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required for direct NIfTI inference. Install with `pip install nibabel`.") from exc

        self.nib = nib
        self.case = case
        self.time_axis_arg = time_axis
        self.mag_scale = float(mag_scale) if mag_scale else 4095.0
        self.auto_convert_raw_phase = auto_convert_raw_phase

        self.u_img = nib.load(str(case.u_path))
        self.v_img = nib.load(str(case.v_path))
        self.w_img = nib.load(str(case.w_path))
        self.mag_u_img = nib.load(str(case.mag_u_path))
        self.mag_v_img = nib.load(str(case.mag_v_path))
        self.mag_w_img = nib.load(str(case.mag_w_path))

        self.template_affine = self.u_img.affine.copy()
        self.template_header = self.u_img.header.copy()
        self.template_zooms = tuple(self.u_img.header.get_zooms())

        self.u_time_axis = self._resolve_time_axis(self.u_img.ndim, time_axis)
        self._validate_shapes()

        self.num_frames = self._resolve_num_frames(self.u_img.shape, self.u_time_axis)
        self.v_frames = self._resolve_num_frames(self.v_img.shape, self._resolve_time_axis(self.v_img.ndim, time_axis))
        self.w_frames = self._resolve_num_frames(self.w_img.shape, self._resolve_time_axis(self.w_img.ndim, time_axis))
        if self.v_frames != self.num_frames or self.w_frames != self.num_frames:
            raise ValueError("U/V/W NIfTI files must have the same number of frames.")

        self.mag_u_frames = self._resolve_num_frames(
            self.mag_u_img.shape, self._resolve_time_axis(self.mag_u_img.ndim, time_axis)
        )
        self.mag_v_frames = self._resolve_num_frames(
            self.mag_v_img.shape, self._resolve_time_axis(self.mag_v_img.ndim, time_axis)
        )
        self.mag_w_frames = self._resolve_num_frames(
            self.mag_w_img.shape, self._resolve_time_axis(self.mag_w_img.ndim, time_axis)
        )
        for name, frames in (("mag_u", self.mag_u_frames), ("mag_v", self.mag_v_frames), ("mag_w", self.mag_w_frames)):
            if frames not in (1, self.num_frames):
                raise ValueError(f"{name} must have 1 frame or match velocity frame count ({self.num_frames}).")

        self.venc_u = float(venc_u) if venc_u is not None else float(venc) if venc is not None else None
        self.venc_v = float(venc_v) if venc_v is not None else float(venc) if venc is not None else None
        self.venc_w = float(venc_w) if venc_w is not None else float(venc) if venc is not None else None

        if self.venc_u is None:
            self.venc_u = self._estimate_venc(self.u_img, self.u_time_axis, "u")
        if self.venc_v is None:
            self.venc_v = self._estimate_venc(self.v_img, self._resolve_time_axis(self.v_img.ndim, time_axis), "v")
        if self.venc_w is None:
            self.venc_w = self._estimate_venc(self.w_img, self._resolve_time_axis(self.w_img.ndim, time_axis), "w")

        self.venc = np.float32(max(self.venc_u, self.venc_v, self.venc_w))
        self.velocity_per_px = self.venc / np.float32(2048.0)
        self.dx = np.asarray(self.template_zooms[:3], dtype=np.float32)

    @staticmethod
    def _resolve_time_axis(ndim: int, axis: int) -> int | None:
        if ndim == 3:
            return None
        if ndim != 4:
            raise ValueError(f"Expected 3D or 4D NIfTI, got {ndim}D.")
        resolved = axis if axis >= 0 else ndim + axis
        if resolved < 0 or resolved >= ndim:
            raise ValueError(f"Invalid time axis {axis} for NIfTI ndim={ndim}.")
        return resolved

    @staticmethod
    def _resolve_num_frames(shape: tuple[int, ...], time_axis: int | None) -> int:
        if time_axis is None:
            return 1
        return int(shape[time_axis])

    def _spatial_shape(self, shape: tuple[int, ...], time_axis: int | None) -> tuple[int, int, int]:
        if time_axis is None:
            return tuple(shape)  # type: ignore[return-value]
        return tuple(dim for idx, dim in enumerate(shape) if idx != time_axis)  # type: ignore[return-value]

    def _validate_shapes(self) -> None:
        u_spatial = self._spatial_shape(self.u_img.shape, self._resolve_time_axis(self.u_img.ndim, self.time_axis_arg))
        v_spatial = self._spatial_shape(self.v_img.shape, self._resolve_time_axis(self.v_img.ndim, self.time_axis_arg))
        w_spatial = self._spatial_shape(self.w_img.shape, self._resolve_time_axis(self.w_img.ndim, self.time_axis_arg))
        if u_spatial != v_spatial or u_spatial != w_spatial:
            raise ValueError("U/V/W NIfTI files must have matching spatial shape.")

        for name, img in (("mag_u", self.mag_u_img), ("mag_v", self.mag_v_img), ("mag_w", self.mag_w_img)):
            mag_spatial = self._spatial_shape(img.shape, self._resolve_time_axis(img.ndim, self.time_axis_arg))
            if mag_spatial != u_spatial:
                raise ValueError(f"{name} spatial shape does not match velocity spatial shape.")

    def _extract_frame(self, img, time_axis: int | None, frame_idx: int) -> np.ndarray:
        if time_axis is None:
            return np.asarray(img.dataobj, dtype=np.float32)
        if self._resolve_num_frames(img.shape, time_axis) == 1:
            frame_idx = 0

        slices = [slice(None)] * img.ndim
        slices[time_axis] = frame_idx
        return np.asarray(img.dataobj[tuple(slices)], dtype=np.float32)

    @staticmethod
    def _looks_like_raw_phase(values: np.ndarray) -> bool:
        mn = float(np.min(values))
        mx = float(np.max(values))
        return mn >= 0.0 and mx > 1000.0 and mx <= 8192.0

    def _convert_raw_if_needed(self, values: np.ndarray, venc_component: float, invert_sign: bool, label: str) -> np.ndarray:
        if not self.auto_convert_raw_phase:
            return values
        if not self._looks_like_raw_phase(values):
            return values
        if venc_component is None or venc_component <= 0:
            raise ValueError(f"Cannot convert raw phase for {label}: invalid VENC ({venc_component}).")
        converted = (values.astype(np.float32) - 2048.0) / 2048.0 * float(venc_component)
        if invert_sign:
            converted = -converted
        return converted

    def _estimate_venc(self, img, time_axis: int | None, label: str) -> float:
        maxima = 0.0
        frames = self._resolve_num_frames(img.shape, time_axis)
        for frame_idx in range(frames):
            frame = self._extract_frame(img, time_axis, frame_idx)
            maxima = max(maxima, float(np.max(np.abs(frame))))
        if maxima <= 0.0:
            raise ValueError(f"Could not estimate VENC for {label}; all values are zero.")
        return maxima

    def get_num_frames(self) -> int:
        return self.num_frames

    def load_vectorfield(self, frame_idx: int) -> None:
        u_frame = self._extract_frame(self.u_img, self.u_time_axis, frame_idx)
        v_frame = self._extract_frame(self.v_img, self._resolve_time_axis(self.v_img.ndim, self.time_axis_arg), frame_idx)
        w_frame = self._extract_frame(self.w_img, self._resolve_time_axis(self.w_img.ndim, self.time_axis_arg), frame_idx)

        u_frame = self._convert_raw_if_needed(u_frame, self.venc_u, invert_sign=True, label="u")
        v_frame = self._convert_raw_if_needed(v_frame, self.venc_v, invert_sign=True, label="v")
        w_frame = self._convert_raw_if_needed(w_frame, self.venc_w, invert_sign=False, label="w")

        mag_u_frame = self._extract_frame(
            self.mag_u_img, self._resolve_time_axis(self.mag_u_img.ndim, self.time_axis_arg), frame_idx
        )
        mag_v_frame = self._extract_frame(
            self.mag_v_img, self._resolve_time_axis(self.mag_v_img.ndim, self.time_axis_arg), frame_idx
        )
        mag_w_frame = self._extract_frame(
            self.mag_w_img, self._resolve_time_axis(self.mag_w_img.ndim, self.time_axis_arg), frame_idx
        )

        self.u = (u_frame / self.venc).astype(np.float32)
        self.v = (v_frame / self.venc).astype(np.float32)
        self.w = (w_frame / self.venc).astype(np.float32)

        mag_scale = self.mag_scale if self.mag_scale > 0 else 4095.0
        self.mag_u = (mag_u_frame / mag_scale).astype(np.float32)
        self.mag_v = (mag_v_frame / mag_scale).astype(np.float32)
        self.mag_w = (mag_w_frame / mag_scale).astype(np.float32)

    def _scaled_affine(self, res_increase: int) -> np.ndarray:
        affine = self.template_affine.copy()
        affine[:3, :3] = affine[:3, :3] / float(res_increase)
        return affine

    def _to_output_layout(self, time_first_data: np.ndarray) -> np.ndarray:
        if self.u_time_axis is None:
            return time_first_data[0]
        return np.moveaxis(time_first_data, 0, self.u_time_axis)

    def _output_zooms(self, output_ndim: int, res_increase: int) -> tuple[float, ...]:
        if len(self.template_zooms) < 3:
            raise ValueError("Template NIfTI does not contain at least 3 spatial zooms.")
        spatial = [self.template_zooms[0] / res_increase, self.template_zooms[1] / res_increase, self.template_zooms[2] / res_increase]
        if output_ndim == 3:
            return tuple(spatial)

        extra = list(self.template_zooms[3:output_ndim])
        while len(extra) < (output_ndim - 3):
            extra.append(1.0)
        return tuple(spatial + extra)

    def save_predictions_as_nifti(
        self,
        comp_u: np.ndarray,
        comp_v: np.ndarray,
        comp_w: np.ndarray,
        output_dir: Path,
        res_increase: int,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        component_map = {
            "u_SR.nii.gz": comp_u,
            "v_SR.nii.gz": comp_v,
            "w_SR.nii.gz": comp_w,
        }

        saved_paths: list[Path] = []
        affine = self._scaled_affine(res_increase)
        for filename, data_time_first in component_map.items():
            data_out = self._to_output_layout(np.asarray(data_time_first, dtype=np.float32))
            header = self.template_header.copy()
            header.set_data_dtype(np.float32)
            header.set_zooms(self._output_zooms(data_out.ndim, res_increase))
            image = self.nib.Nifti1Image(data_out, affine=affine, header=header)
            out_path = output_dir / filename
            self.nib.save(image, str(out_path))
            saved_paths.append(out_path)

        return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prediction for 4DFlowNet (H5 or NIfTI inputs).")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input root directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for results")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model weights (.h5)")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="TensorFlow execution device preference (auto uses GPU when available)",
    )
    parser.add_argument(
        "--gpu-memory-limit-mb",
        type=int,
        default=None,
        help="Optional GPU logical memory limit in MB (useful on Apple Metal if GPU resets)",
    )
    parser.add_argument(
        "--no-cpu-fallback-on-gpu-error",
        action="store_false",
        dest="cpu_fallback_on_gpu_error",
        help="Disable automatic CPU fallback if Apple Metal GPU fails during prediction",
    )
    parser.set_defaults(cpu_fallback_on_gpu_error=True)

    parser.add_argument("--input-format", choices=["h5", "nifti"], default="h5")
    parser.add_argument(
        "--output-format",
        choices=["h5", "nifti"],
        default=None,
        help="Output format (default matches --input-format)",
    )

    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--res-increase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-round-values",
        action="store_false",
        dest="round_values",
        help="Disable zeroing values below venc/2048",
    )
    parser.set_defaults(round_values=True)

    parser.add_argument("--h5-pattern", default="*.h5", help="Glob for H5 inputs when --input-format=h5")

    parser.add_argument("--recursive", action="store_true", help="Recursively discover NIfTI cases")
    parser.add_argument(
        "--u-name",
        default="Vx.nii.gz",
        help="Velocity U filename inside each NIfTI case folder",
    )
    parser.add_argument(
        "--v-name",
        default="Vy.nii.gz",
        help="Velocity V filename inside each NIfTI case folder",
    )
    parser.add_argument(
        "--w-name",
        default="Vz.nii.gz",
        help="Velocity W filename inside each NIfTI case folder",
    )
    parser.add_argument(
        "--mag-name",
        default="input_mag_raw.nii.gz",
        help="Shared magnitude filename for all components",
    )
    parser.add_argument(
        "--case-suffix",
        default="_3T",
        help="Only process NIfTI case folders ending with this suffix (empty string = no filter)",
    )
    parser.add_argument("--mag-u-name", default=None, help="Per-component magnitude U filename (overrides --mag-name)")
    parser.add_argument("--mag-v-name", default=None, help="Per-component magnitude V filename (overrides --mag-name)")
    parser.add_argument("--mag-w-name", default=None, help="Per-component magnitude W filename (overrides --mag-name)")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis index for 4D NIfTI (default: last axis)")
    parser.add_argument("--mag-scale", type=float, default=4095.0, help="Magnitude normalization scale")
    parser.add_argument(
        "--auto-convert-raw-phase",
        action="store_true",
        help="Convert raw phase-like velocity inputs (0..4096) to m/s using VENC",
    )
    parser.add_argument("--venc", type=float, default=None, help="Shared VENC (m/s) for NIfTI inputs")
    parser.add_argument("--venc-u", type=float, default=None, help="VENC for U component (m/s)")
    parser.add_argument("--venc-v", type=float, default=None, help="VENC for V component (m/s)")
    parser.add_argument("--venc-w", type=float, default=None, help="VENC for W component (m/s)")
    parser.add_argument("--show-patch-progress", action="store_true", help="Show patch-batch progress per frame")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame and per-case timing details")
    parser.add_argument(
        "--timing-report",
        default=None,
        help="Optional CSV path for per-case timing and frame counts",
    )
    return parser.parse_args()


def configure_tensorflow(device_mode: str, gpu_memory_limit_mb: int | None) -> None:
    """Configure TensorFlow device usage before model creation."""
    gpus = tf.config.list_physical_devices("GPU")

    if device_mode == "cpu":
        if gpus:
            tf.config.set_visible_devices([], "GPU")
        print("TensorFlow device mode: CPU")
        return

    if device_mode == "gpu":
        if not gpus:
            raise SystemExit("Requested --device gpu but no GPU was detected by TensorFlow.")
        print(f"TensorFlow device mode: GPU ({len(gpus)} detected)")
    else:
        if gpus:
            print(f"TensorFlow device mode: AUTO -> GPU ({len(gpus)} detected)")
        else:
            print("TensorFlow device mode: AUTO -> CPU (no GPU detected)")

    if gpus and gpu_memory_limit_mb is not None:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit_mb)],
            )
            print(f"Configured GPU memory limit: {gpu_memory_limit_mb} MB")
        except Exception as exc:  # pragma: no cover - depends on backend support
            print(f"Warning: could not set GPU memory limit ({exc})")


def build_model_with_weights(
    model_path: Path,
    patch_size: int,
    res_increase: int,
    force_cpu: bool = False,
) -> tf.keras.Model:
    if force_cpu:
        with tf.device("/CPU:0"):
            model = prepare_network(patch_size, res_increase, 8, 4)
            model.load_weights(str(model_path))
            return model

    model = prepare_network(patch_size, res_increase, 8, 4)
    model.load_weights(str(model_path))
    return model


def create_model_executor(args: argparse.Namespace, model_path: Path) -> ModelExecutor:
    print(f"--- Loading Model: {model_path.name} ---")
    primary_model = build_model_with_weights(
        model_path=model_path,
        patch_size=args.patch_size,
        res_increase=args.res_increase,
        force_cpu=(args.device == "cpu"),
    )

    cpu_fallback_model = None
    wants_fallback = args.cpu_fallback_on_gpu_error and args.device != "cpu"
    if wants_fallback:
        print("Preparing CPU fallback model for GPU fault tolerance...")
        cpu_fallback_model = build_model_with_weights(
            model_path=model_path,
            patch_size=args.patch_size,
            res_increase=args.res_increase,
            force_cpu=True,
        )

    print("Model loaded successfully.")
    return ModelExecutor(primary_model=primary_model, cpu_fallback_model=cpu_fallback_model)


def resolve_mag_paths(case_dir: Path, args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    mag_u_path = case_dir / args.mag_u_name if args.mag_u_name else None
    mag_v_path = case_dir / args.mag_v_name if args.mag_v_name else None
    mag_w_path = case_dir / args.mag_w_name if args.mag_w_name else None

    if mag_u_path and mag_v_path and mag_w_path:
        if mag_u_path.exists() and mag_v_path.exists() and mag_w_path.exists():
            return mag_u_path, mag_v_path, mag_w_path
        return None, None, None

    shared_mag = case_dir / args.mag_name
    if shared_mag.exists():
        return shared_mag, shared_mag, shared_mag
    return None, None, None


def discover_nifti_cases(input_dir: Path, args: argparse.Namespace) -> list[NiftiCase]:
    case_dirs: list[Path] = []
    suffix_filter = args.case_suffix or ""

    def allowed_case_dir(path: Path) -> bool:
        return not suffix_filter or path.name.endswith(suffix_filter)

    if args.recursive:
        for root, _dirs, files in os.walk(input_dir):
            names = set(files)
            root_path = Path(root)
            if args.u_name in names and args.v_name in names and args.w_name in names and allowed_case_dir(root_path):
                case_dirs.append(root_path)
    else:
        root_names = {p.name for p in input_dir.iterdir() if p.is_file()}
        if args.u_name in root_names and args.v_name in root_names and args.w_name in root_names and allowed_case_dir(input_dir):
            case_dirs.append(input_dir)
        case_dirs.extend(sorted(path for path in input_dir.iterdir() if path.is_dir() and allowed_case_dir(path)))

    cases: list[NiftiCase] = []
    for case_dir in sorted(case_dirs):
        u_path = case_dir / args.u_name
        v_path = case_dir / args.v_name
        w_path = case_dir / args.w_name
        if not (u_path.exists() and v_path.exists() and w_path.exists()):
            continue

        mag_u_path, mag_v_path, mag_w_path = resolve_mag_paths(case_dir, args)
        if mag_u_path is None or mag_v_path is None or mag_w_path is None:
            continue

        rel_path = os.path.relpath(case_dir, input_dir)
        cases.append(
            NiftiCase(
                case_name=case_dir.name,
                rel_path=rel_path,
                case_dir=case_dir,
                u_path=u_path,
                v_path=v_path,
                w_path=w_path,
                mag_u_path=mag_u_path,
                mag_v_path=mag_v_path,
                mag_w_path=mag_w_path,
            )
        )
    return cases


def predict_case_nifti(
    model_executor: ModelExecutor,
    case: NiftiCase,
    output_dir: Path,
    args: argparse.Namespace,
    output_format: str,
) -> tuple[int, float, str]:
    case_start = time.perf_counter()
    dataset = NiftiInferenceDataset(
        case=case,
        time_axis=args.time_axis,
        mag_scale=args.mag_scale,
        auto_convert_raw_phase=args.auto_convert_raw_phase,
        venc=args.venc,
        venc_u=args.venc_u,
        venc_v=args.venc_v,
        venc_w=args.venc_w,
    )
    patch_generator = PatchGenerator(args.patch_size, args.res_increase)
    num_frames = dataset.get_num_frames()
    print(f"Loading NIfTI case: {case.rel_path} (frames={num_frames})")

    if output_format == "h5":
        output_basename = case.rel_path.replace(os.sep, "_")
        if output_basename in ("", "."):
            output_basename = case.case_name
        output_path = output_dir / f"{output_basename}_SR.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()
            print(f"   Frame {frame_idx + 1}/{num_frames} ...")
            dataset.load_vectorfield(frame_idx)
            comp_u, comp_v, comp_w = infer_frame_components(
                model_executor=model_executor,
                patch_generator=patch_generator,
                dataset=dataset,
                batch_size=args.batch_size,
                round_values=args.round_values,
                show_patch_progress=args.show_patch_progress,
                progress_label=f"{case.case_name} f{frame_idx + 1}/{num_frames}",
            )

            prediction_utils.save_to_h5(str(output_path), "u", np.expand_dims(comp_u, axis=0), compression="gzip")
            prediction_utils.save_to_h5(str(output_path), "v", np.expand_dims(comp_v, axis=0), compression="gzip")
            prediction_utils.save_to_h5(str(output_path), "w", np.expand_dims(comp_w, axis=0), compression="gzip")
            prediction_utils.save_to_h5(
                str(output_path),
                dataset.dx_colname,
                np.expand_dims(dataset.dx / args.res_increase, axis=0),
                compression="gzip",
            )
            if args.verbose:
                print(f"      done in {time.perf_counter() - frame_start:.2f}s")
        print(f"Saved to {output_path}")
        return num_frames, time.perf_counter() - case_start, str(output_path)

    with tempfile.TemporaryDirectory(prefix="4dflownet_nifti_pred_") as tmpdir:
        component_maps = None
        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()
            print(f"   Frame {frame_idx + 1}/{num_frames} ...")
            dataset.load_vectorfield(frame_idx)
            comp_u, comp_v, comp_w = infer_frame_components(
                model_executor=model_executor,
                patch_generator=patch_generator,
                dataset=dataset,
                batch_size=args.batch_size,
                round_values=args.round_values,
                show_patch_progress=args.show_patch_progress,
                progress_label=f"{case.case_name} f{frame_idx + 1}/{num_frames}",
            )

            if component_maps is None:
                frame_shape = comp_u.shape
                full_shape = (num_frames,) + frame_shape
                component_maps = [
                    np.memmap(Path(tmpdir) / "u_pred.dat", mode="w+", dtype=np.float32, shape=full_shape),
                    np.memmap(Path(tmpdir) / "v_pred.dat", mode="w+", dtype=np.float32, shape=full_shape),
                    np.memmap(Path(tmpdir) / "w_pred.dat", mode="w+", dtype=np.float32, shape=full_shape),
                ]

            component_maps[0][frame_idx] = comp_u
            component_maps[1][frame_idx] = comp_v
            component_maps[2][frame_idx] = comp_w
            if args.verbose:
                print(f"      done in {time.perf_counter() - frame_start:.2f}s")

        if component_maps is None:
            print(f"Skipping {case.rel_path}: no frames found.")
            return 0, time.perf_counter() - case_start, str(output_dir / case.rel_path)

        for mmap_arr in component_maps:
            mmap_arr.flush()

        if case.rel_path == ".":
            case_output_dir = output_dir
        else:
            case_output_dir = output_dir / case.rel_path
        saved = dataset.save_predictions_as_nifti(
            comp_u=component_maps[0],
            comp_v=component_maps[1],
            comp_w=component_maps[2],
            output_dir=case_output_dir,
            res_increase=args.res_increase,
        )
        print("Saved:")
        for path in saved:
            print(f" - {path}")
        return num_frames, time.perf_counter() - case_start, str(case_output_dir)


def write_timing_report(report_path: str, rows: list[str]) -> None:
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("input_format,case,frames,seconds_total,seconds_per_frame,input_path,output_path,status\n")
        for row in rows:
            handle.write(row + "\n")
    print(f"Timing report written to: {report_path}")


def main() -> None:
    args = parse_args()
    output_format = args.output_format or args.input_format
    timing_rows: list[str] = []

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}")
        return

    if args.device == "gpu" and platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        print("Note: Apple Metal GPU mode can be unstable in some TensorFlow builds.")
        if args.cpu_fallback_on_gpu_error:
            print("CPU fallback on GPU faults is enabled.")

    configure_tensorflow(args.device, args.gpu_memory_limit_mb)

    model_path = resolve_model_path(args.model_path)
    if model_path is None:
        return

    model_executor = create_model_executor(args, model_path)

    if args.input_format == "h5":
        if output_format != "h5":
            raise SystemExit("H5 input only supports H5 output (spatial metadata is not available for NIfTI export).")

        input_files = sorted(glob.glob(str(args.input_dir / args.h5_pattern)))
        if not input_files:
            print(f"No H5 files found in {args.input_dir} matching '{args.h5_pattern}'")
            return

        print(f"Found {len(input_files)} H5 patients to process.")
        for input_file in tqdm(input_files, desc="Predicting H5 cases", unit="case"):
            input_path = Path(input_file)
            output_name = input_path.name.replace(".h5", "_SR.h5")
            output_path = args.output_dir / output_name

            print(f"\nProcessing: {input_path.name}")
            start = time.time()
            try:
                frames, seconds = predict_patient_h5(
                    model_executor=model_executor,
                    input_filepath=str(input_path),
                    output_filepath=str(output_path),
                    patch_size=args.patch_size,
                    res_increase=args.res_increase,
                    batch_size=args.batch_size,
                    round_values=args.round_values,
                    show_patch_progress=args.show_patch_progress,
                    verbose=args.verbose,
                )
                print(f"[Done] Saved to {output_name} (Took {time.time() - start:.1f}s)")
                status = "ok"
            except Exception as exc:  # pragma: no cover - defensive batch loop
                frames = 0
                seconds = time.time() - start
                status = f"failed:{type(exc).__name__}"
                print(f"[FAIL] {input_path.name}: {exc}")
                if is_metal_gpu_error(str(exc)):
                    print(
                        "Hint: Apple Metal GPU reset detected. Retry with "
                        "`--device cpu` or keep GPU fallback enabled."
                    )

            timing_rows.append(
                ",".join(
                    [
                        "h5",
                        input_path.name,
                        str(frames),
                        f"{seconds:.4f}",
                        f"{(seconds / max(1, frames)):.4f}",
                        str(input_path),
                        str(output_path),
                        status,
                    ]
                )
            )
        if args.timing_report:
            write_timing_report(args.timing_report, timing_rows)
        return

    cases = discover_nifti_cases(args.input_dir, args)
    if not cases:
        print(
            "No NIfTI cases found. Expected per-case files: "
            f"{args.u_name}, {args.v_name}, {args.w_name}, and magnitude file(s). "
            f"Current case filter: suffix='{args.case_suffix}'."
        )
        return
    if args.auto_convert_raw_phase and all(value is None for value in (args.venc, args.venc_u, args.venc_v, args.venc_w)):
        raise SystemExit(
            "Direct NIfTI raw-phase conversion requires VENC. "
            "Provide --venc (or --venc-u/--venc-v/--venc-w)."
        )

    print(f"Found {len(cases)} NIfTI cases to process.")
    for case in tqdm(cases, desc="Predicting NIfTI cases", unit="case"):
        print(f"\nProcessing case: {case.rel_path}")
        start = time.time()
        out_location = str(args.output_dir / case.rel_path)
        try:
            frames, seconds, out_location = predict_case_nifti(
                model_executor=model_executor,
                case=case,
                output_dir=args.output_dir,
                args=args,
                output_format=output_format,
            )
            print(f"[Done] {case.rel_path} (Took {time.time() - start:.1f}s)")
            status = "ok"
        except Exception as exc:  # pragma: no cover - defensive batch loop
            frames = 0
            seconds = time.time() - start
            status = f"failed:{type(exc).__name__}"
            print(f"[FAIL] {case.rel_path}: {exc}")
            if is_metal_gpu_error(str(exc)):
                print(
                    "Hint: Apple Metal GPU reset detected. Retry with "
                    "`--device cpu` or keep GPU fallback enabled."
                )
        timing_rows.append(
            ",".join(
                [
                    "nifti",
                    case.rel_path.replace(",", "_"),
                    str(frames),
                    f"{seconds:.4f}",
                    f"{(seconds / max(1, frames)):.4f}",
                    str(case.case_dir),
                    out_location,
                    status,
                ]
            )
        )

    if args.timing_report:
        write_timing_report(args.timing_report, timing_rows)


if __name__ == "__main__":
    main()
