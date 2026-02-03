import numpy as np
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    data_dir = '../data'
    
    # ---- Patch index files ----
    training_file = '{}/train_test.csv'.format(data_dir)
    validate_file = '{}/validate.csv'.format(data_dir)

    QUICKSAVE = False
    benchmark_file = '{}/benchmark.csv'.format(data_dir)
    
    restore = False
    if restore:
        model_dir = "../models/4DFlowNet"
        model_file = "4DFlowNet-best.pt"

    # Hyperparameters optimisation variables
    initial_learning_rate = 2e-4
    epochs =  60
    batch_size = 20
    mask_threshold = 0.6

    # Network setting
    network_name = '4DFlowNet'
    patch_size = 16
    res_increase = 1
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    low_resblock = 8
    hi_resblock = 4

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- MONAI/PyTorch DataLoader -------------------
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None)

    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=True, n_parallel=None)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this bechmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        ph = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(benchmark_set, shuffle=False) 

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, low_resblock, hi_resblock)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.param_groups[0]["lr"])

    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
