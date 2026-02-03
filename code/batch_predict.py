import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

# Add src to python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from Network.PatchGenerator import PatchGenerator
from Network.SR4DFlowNet import SR4DFlowNet
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


def prepare_network(res_increase, low_resblock, hi_resblock, device):
    model = SR4DFlowNet(res_increase, low_resblock=low_resblock, hi_resblock=hi_resblock).to(device)
    return model


def load_model_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()


def predict_patient(model, input_filepath, output_filepath, patch_size, res_increase, batch_size=8, round_values=True):
    print(f"Loading data from: {input_filepath}")

    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    nr_rows = dataset.get_dataset_len(input_filepath)
    if nr_rows == 0:
        print(f"Skipping {input_filepath}: No data found")
        return

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    device = next(model.parameters()).device

    for nrow in range(nr_rows):
        print(f"   Frame {nrow + 1}/{nr_rows} ...")
        dataset.load_vectorfield(input_filepath, nrow)
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])

        result_batches = []

        for current_idx in range(0, data_size, batch_size):
            patch_index = np.index_exp[current_idx : current_idx + batch_size]
            with torch.no_grad():
                u = torch.from_numpy(velocities[0][patch_index]).to(device)
                v = torch.from_numpy(velocities[1][patch_index]).to(device)
                w = torch.from_numpy(velocities[2][patch_index]).to(device)
                u_mag = torch.from_numpy(magnitudes[0][patch_index]).to(device)
                v_mag = torch.from_numpy(magnitudes[1][patch_index]).to(device)
                w_mag = torch.from_numpy(magnitudes[2][patch_index]).to(device)
                sr_batch = model(u, v, w, u_mag, v_mag, w_mag).cpu().numpy()
                sr_batch = np.moveaxis(sr_batch, 1, -1)
                result_batches.append(sr_batch)

        results = np.concatenate(result_batches, axis=0) if result_batches else np.zeros((0, 0, 0, 0, 3))

        for i in range(3):
            v_comp = pgen._patchup_with_overlap(results[:, :, :, :, i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            v_comp = v_comp * dataset.venc

            if round_values and hasattr(dataset, "velocity_per_px"):
                v_comp[np.abs(v_comp) < dataset.velocity_per_px] = 0

            v_comp = np.expand_dims(v_comp, axis=0)
            prediction_utils.save_to_h5(output_filepath, dataset.velocity_colnames[i], v_comp, compression="gzip")

        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0)
            prediction_utils.save_to_h5(output_filepath, dataset.dx_colname, new_spacing, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Batch Prediction for 4DFlowNet (PyTorch)")
    parser.add_argument("--input-dir", default="../data/h5_lr", help="Directory with .h5 input files")
    parser.add_argument("--output-dir", default="../data/predictions", help="Directory for results")
    parser.add_argument("--model-path", default="../models/4DFlowNet_cerebro/4DFlowNet-best.pt", help="Path to model checkpoint (.pt)")

    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--res-increase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        candidates = glob.glob("../models/**/*best.pt", recursive=True)
        if candidates:
            print(f"Found alternative model: {candidates[0]}")
            args.model_path = candidates[0]
        else:
            return

    print(f"--- Loading Model: {os.path.basename(args.model_path)} ---")
    model = prepare_network(args.res_increase, 8, 4, device)
    load_model_weights(model, args.model_path, device)
    print("Model loaded successfully.")

    input_files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    input_files.sort()

    if not input_files:
        print(f"No .h5 files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} patients to process.")

    for fpath in input_files:
        fname = os.path.basename(fpath)
        out_name = fname.replace(".h5", "_SR.h5")
        out_path = os.path.join(args.output_dir, out_name)

        print(f"\nProcessing: {fname}")
        start_t = time.time()

        predict_patient(model, fpath, out_path, args.patch_size, args.res_increase, args.batch_size)
        print(f"[Done] Saved to {out_name} (Took {time.time() - start_t:.1f}s)")


if __name__ == "__main__":
    main()
