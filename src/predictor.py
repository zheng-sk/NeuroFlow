import os
import time

import numpy as np
import torch

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


if __name__ == "__main__":
    data_dir = "../data"
    filename = "example_data.h5"

    output_dir = "../result"
    output_filename = "example_result.h5"

    model_path = "../models/4DFlowNet/4DFlowNet-best.pt"

    # Params
    patch_size = 24
    res_increase = 2
    batch_size = 8
    round_small_values = True

    # Network
    low_resblock = 8
    hi_resblock = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting up
    input_filepath = f"{data_dir}/{filename}"
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    network = prepare_network(res_increase, low_resblock, hi_resblock, device)
    load_model_weights(network, model_path, device)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # loop through all the rows in the input file
    for nrow in range(0, nr_rows):
        print("\n--------------------------")
        print(f"\nProcessed ({nrow + 1}/{nr_rows}) - {time.ctime()}")
        dataset.load_vectorfield(input_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")

        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        result_batches = []
        start_time = time.time()

        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end="\r")
            patch_index = np.index_exp[current_idx : current_idx + batch_size]

            with torch.no_grad():
                u = torch.from_numpy(velocities[0][patch_index]).to(device)
                v = torch.from_numpy(velocities[1][patch_index]).to(device)
                w = torch.from_numpy(velocities[2][patch_index]).to(device)
                u_mag = torch.from_numpy(magnitudes[0][patch_index]).to(device)
                v_mag = torch.from_numpy(magnitudes[1][patch_index]).to(device)
                w_mag = torch.from_numpy(magnitudes[2][patch_index]).to(device)

                sr_images = network(u, v, w, u_mag, v_mag, w_mag).cpu().numpy()
                sr_images = np.moveaxis(sr_images, 1, -1)
                result_batches.append(sr_images)

        results = np.concatenate(result_batches, axis=0) if result_batches else np.zeros((0, 0, 0, 0, 3))
        time_taken = time.time() - start_time
        print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

        for i in range(0, 3):
            v_comp = pgen._patchup_with_overlap(results[:, :, :, :, i], pgen.nr_x, pgen.nr_y, pgen.nr_z)

            # Denormalized
            v_comp = v_comp * dataset.venc
            if round_small_values:
                v_comp[np.abs(v_comp) < dataset.velocity_per_px] = 0

            v_comp = np.expand_dims(v_comp, axis=0)
            prediction_utils.save_to_h5(f"{output_dir}/{output_filename}", dataset.velocity_colnames[i], v_comp, compression="gzip")

        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0)
            prediction_utils.save_to_h5(f"{output_dir}/{output_filename}", dataset.dx_colname, new_spacing, compression="gzip")

    print("Done!")
