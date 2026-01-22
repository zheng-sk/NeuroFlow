import argparse
import h5py
import numpy as np
import PatchData as pd


def load_mask(input_filepath, mask_threshold):
    with h5py.File(input_filepath, mode='r') as hdf5:
        mask = np.asarray(hdf5['mask'][0])
    binary_mask = (mask >= mask_threshold) * 1
    return binary_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../../data', help='Base directory for HDF5 files and CSV output')
    parser.add_argument('--lr-file', type=str, required=True, help='Low-res HDF5 filename')
    parser.add_argument('--hr-file', type=str, required=True, help='High-res HDF5 filename')
    parser.add_argument('--output-csv', type=str, default='train.csv', help='Output CSV filename')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size (isotropic)')
    parser.add_argument('--n-patch', type=int, default=10, help='Number of patches per time frame')
    parser.add_argument('--mask-threshold', type=float, default=0.4, help='Threshold for mask')
    parser.add_argument('--minimum-coverage', type=float, default=0.2, help='Minimum fluid coverage for patches')
    parser.add_argument('--all-rotation', action='store_true', help='Use all 90/180/270 rotations')
    parser.add_argument('--empty-patch-allowed', type=int, default=0, help='Max number of empty patches per frame')
    args = parser.parse_args()

    base_path = args.data_dir
    lr_file = args.lr_file
    hr_file = args.hr_file
    output_csv = args.output_csv

    input_filepath = f'{base_path}/{lr_file}'
    binary_mask = load_mask(input_filepath, args.mask_threshold)

    pd.write_header(f'{base_path}/{output_csv}')

    with h5py.File(input_filepath, mode='r') as hdf5:
        data_nr = len(hdf5['u'])

    indexes = np.arange(data_nr)
    for index in indexes:
        print('Generating patches for row', index)
        pd.generate_random_patches(
            lr_file,
            hr_file,
            f'{base_path}/{output_csv}',
            index,
            args.n_patch,
            binary_mask,
            args.patch_size,
            args.minimum_coverage,
            args.empty_patch_allowed,
            args.all_rotation,
        )

    print(f'Done. File saved in {base_path}/{output_csv}')


if __name__ == '__main__':
    main()
