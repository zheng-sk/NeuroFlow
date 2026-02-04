#!/usr/bin/env python3
"""Batch inference runner for 4DFlowNet H5 inputs."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
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

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "h5_lr"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "predictions"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "4DFlowNet_cerebro" / "4DFlowNet_cerebro.h5"


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


def predict_patient(
    model: tf.keras.Model,
    input_filepath: str,
    output_filepath: str,
    patch_size: int,
    res_increase: int,
    batch_size: int = 8,
    round_values: bool = True,
) -> None:
    print(f"Loading data from: {input_filepath}")

    patch_generator = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    num_frames = dataset.get_dataset_len(input_filepath)
    if num_frames == 0:
        print(f"Skipping {input_filepath}: no data found")
        return

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    for frame_idx in range(num_frames):
        print(f"   Frame {frame_idx + 1}/{num_frames} ...")
        dataset.load_vectorfield(input_filepath, frame_idx)

        velocities, magnitudes = patch_generator.patchify(dataset)
        num_patches = len(velocities[0])

        prediction_chunks = []
        for start in range(0, num_patches, batch_size):
            patch_slice = np.index_exp[start : start + batch_size]
            prediction_chunks.append(
                model.predict(
                    [
                        velocities[0][patch_slice],
                        velocities[1][patch_slice],
                        velocities[2][patch_slice],
                        magnitudes[0][patch_slice],
                        magnitudes[1][patch_slice],
                        magnitudes[2][patch_slice],
                    ],
                    verbose=0,
                )
            )

        results = np.concatenate(prediction_chunks, axis=0)

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

            prediction_utils.save_to_h5(
                output_filepath,
                dataset.velocity_colnames[component_idx],
                np.expand_dims(component, axis=0),
                compression="gzip",
            )

        if dataset.dx is not None:
            new_spacing = np.expand_dims(dataset.dx / res_increase, axis=0)
            prediction_utils.save_to_h5(output_filepath, dataset.dx_colname, new_spacing, compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Prediction for 4DFlowNet")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with .h5 input files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for results")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model weights (.h5)")

    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--res-increase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    if model_path is None:
        return

    print(f"--- Loading Model: {model_path.name} ---")
    model = prepare_network(args.patch_size, args.res_increase, 8, 4)
    model.load_weights(str(model_path))
    print("Model loaded successfully.")

    input_files = sorted(glob.glob(str(args.input_dir / "*.h5")))
    if not input_files:
        print(f"No .h5 files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} patients to process.")

    for input_file in input_files:
        input_path = Path(input_file)
        output_name = input_path.name.replace(".h5", "_SR.h5")
        output_path = args.output_dir / output_name

        print(f"\nProcessing: {input_path.name}")
        start = time.time()
        predict_patient(
            model,
            str(input_path),
            str(output_path),
            args.patch_size,
            args.res_increase,
            args.batch_size,
        )
        print(f"[Done] Saved to {output_name} (Took {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
