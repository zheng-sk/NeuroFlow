import sys
import os
import argparse
import time
import glob
import numpy as np
import tensorflow as tf

# Add src to python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset

def prepare_network(patch_size, res_increase, low_resblock, hi_resblock):
    # Prepare input
    input_shape = (patch_size,patch_size,patch_size,1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u,v,w,u_mag, v_mag, w_mag]

    # network & output
    net = SR4DFlowNet(res_increase)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model

def predict_patient(model, input_filepath, output_filepath, patch_size, res_increase, batch_size=8, round_values=True):
    print(f"Loading data from: {input_filepath}")
    
    # Initialize helpers
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()
    
    # Check dimensions
    nr_rows = dataset.get_dataset_len(input_filepath)
    # If file doesn't exist or is empty
    if nr_rows == 0:
        print(f"Skipping {input_filepath}: No data found")
        return

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Loop time frames
    for nrow in range(nr_rows):
        print(f"   Frame {nrow+1}/{nr_rows} ...")
        
        # Load data
        dataset.load_vectorfield(input_filepath, nrow)
        
        # Patchify
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        
        # Buffer for results
        results = np.zeros((0, patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        
        # Predict in batches
        for current_idx in range(0, data_size, batch_size):
            patch_index = np.index_exp[current_idx:current_idx+batch_size]
            sr_batch = model.predict([
                velocities[0][patch_index],
                velocities[1][patch_index],
                velocities[2][patch_index],
                magnitudes[0][patch_index],
                magnitudes[1][patch_index],
                magnitudes[2][patch_index]
            ], verbose=0)
            results = np.append(results, sr_batch, axis=0)
            
        # Reconstruct (Unpatchify) & Save
        for i in range(3): # u, v, w
            # Stitch patches together
            v_comp = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            
            # Denormalize (multiply by VENC)
            v_comp = v_comp * dataset.venc
            
            # Optional: threshold small noise
            if round_values and hasattr(dataset, 'velocity_per_px'):
                v_comp[np.abs(v_comp) < dataset.velocity_per_px] = 0
            
            # Add batch dimension and save
            v_comp = np.expand_dims(v_comp, axis=0)
            
            # Save using utils
            # We append to h5 file, so for first frame we might need to handle 'w' vs 'a' mode?
            # prediction_utils.save_to_h5 generally appends or creates datasets
            prediction_utils.save_to_h5(output_filepath, dataset.velocity_colnames[i], v_comp, compression='gzip')

        # Save spacing if available (only need once really, but per frame is safe for 4D struct)
        if dataset.dx is not None:
             new_spacing = dataset.dx / res_increase
             new_spacing = np.expand_dims(new_spacing, axis=0) 
             prediction_utils.save_to_h5(output_filepath, dataset.dx_colname, new_spacing, compression='gzip')

def main():
    parser = argparse.ArgumentParser(description="Batch Prediction for 4DFlowNet")
    parser.add_argument("--input-dir", default="../data/h5_lr", help="Directory with .h5 input files")
    parser.add_argument("--output-dir", default="../data/predictions", help="Directory for results")
    parser.add_argument("--model-path", default="../models/4DFlowNet_cerebro/4DFlowNet_cerebro.h5", help="Path to .h5 weights")
    
    # Model config (Must match training!)
    parser.add_argument("--patch-size", type=int, default=24) # 24 for standard 4DFlowNet? Check config
    parser.add_argument("--res-increase", type=int, default=2) 
    parser.add_argument("--batch-size", type=int, default=8)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        # Try to find best model in workspace automatically if default fails
        candidates = glob.glob("../models/**/4DFlowNet-best.h5", recursive=True)
        if candidates:
            print(f"Found alternative model: {candidates[0]}")
            args.model_path = candidates[0]
        else:
            return

    # 1. Load Model
    print(f"--- Loading Model: {os.path.basename(args.model_path)} ---")
    model = prepare_network(args.patch_size, args.res_increase, 8, 4)
    model.load_weights(args.model_path)
    print("Model loaded successfully.")

    # 2. Find Inputs
    input_files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    input_files.sort()
    
    if not input_files:
        print(f"No .h5 files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} patients to process.")

    # 3. Predict loop
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
