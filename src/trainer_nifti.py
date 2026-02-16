import argparse

from Network.NiftiPatchDataset import create_nifti_patch_dataloader
from Network.TrainerController import TrainerController


def main():
    parser = argparse.ArgumentParser(description="Train 4DFlowNet directly from NIfTI files (no HDF5).")
    parser.add_argument("--train-csv", type=str, required=True, help="CSV with NIfTI training cases.")
    parser.add_argument("--val-csv", type=str, required=True, help="CSV with NIfTI validation cases.")
    parser.add_argument("--patch-size", type=int, default=16, help="Low-resolution patch size.")
    parser.add_argument("--res-increase", type=int, default=2, help="Upsampling ratio.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs.")
    parser.add_argument("--initial-learning-rate", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--network-name", type=str, default="4DFlowNet_nifti", help="Run/model name.")
    parser.add_argument("--low-resblock", type=int, default=8, help="Number of low-res residual blocks.")
    parser.add_argument("--hi-resblock", type=int, default=4, help="Number of high-res residual blocks.")
    parser.add_argument("--train-samples-per-volume", type=int, default=64, help="Random patch samples per train volume per epoch.")
    parser.add_argument("--val-samples-per-volume", type=int, default=16, help="Patch samples per validation volume per epoch.")
    parser.add_argument("--mag-scale", type=float, default=4095.0, help="Magnitude normalization divisor.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument(
        "--raw-phase-input",
        dest="raw_phase_input",
        action="store_true",
        default=True,
        help="Assume velocity NIfTI values are raw phase-like and convert to velocity before normalization.",
    )
    parser.add_argument(
        "--already-velocity-input",
        dest="raw_phase_input",
        action="store_false",
        help="Disable raw-phase conversion (input velocity is already physical).",
    )
    parser.add_argument(
        "--legacy-invert-uv-sign-on-raw",
        action="store_true",
        help="Legacy mode: invert U/V signs after RAW->velocity conversion. Keep disabled for DICOM->NIfTI outputs that already applied LPS->RAS sign correction.",
    )
    parser.add_argument("--raw-center", type=float, default=2048.0, help="Raw phase center value.")
    parser.add_argument("--raw-scale", type=float, default=2048.0, help="Raw phase scaling denominator.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D NIfTI (default last axis).")
    parser.add_argument(
        "--legacy-minimum-coverage",
        type=float,
        default=0.0,
        help="Legacy-compatible minimum mask coverage for sampled training patches (0..1).",
    )
    parser.add_argument(
        "--legacy-max-sampling-attempts",
        type=int,
        default=100,
        help="Max attempts to find a patch meeting minimum coverage.",
    )
    parser.add_argument(
        "--legacy-disallow-empty-fallback",
        action="store_true",
        help="If set, fail when no patch meets minimum coverage instead of falling back to best available patch.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--restore", action="store_true", help="Restore training from an existing checkpoint.")
    parser.add_argument("--restore-dir", type=str, default="", help="Checkpoint directory to restore from.")
    parser.add_argument("--restore-file", type=str, default="", help="Checkpoint filename (.pt).")
    args = parser.parse_args()

    train_loader = create_nifti_patch_dataloader(
        csv_path=args.train_csv,
        patch_size=args.patch_size,
        res_increase=args.res_increase,
        batch_size=args.batch_size,
        samples_per_volume=args.train_samples_per_volume,
        shuffle=True,
        augment=True,
        num_workers=args.num_workers,
        mag_scale=args.mag_scale,
        mask_threshold=args.mask_threshold,
        raw_phase_input=args.raw_phase_input,
        invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
        raw_center=args.raw_center,
        raw_scale=args.raw_scale,
        time_axis=args.time_axis,
        minimum_coverage=args.legacy_minimum_coverage,
        max_sampling_attempts=args.legacy_max_sampling_attempts,
        allow_empty_fallback=not args.legacy_disallow_empty_fallback,
    )
    val_loader = create_nifti_patch_dataloader(
        csv_path=args.val_csv,
        patch_size=args.patch_size,
        res_increase=args.res_increase,
        batch_size=args.batch_size,
        samples_per_volume=args.val_samples_per_volume,
        shuffle=False,
        augment=False,
        num_workers=args.num_workers,
        mag_scale=args.mag_scale,
        mask_threshold=args.mask_threshold,
        raw_phase_input=args.raw_phase_input,
        invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
        raw_center=args.raw_center,
        raw_scale=args.raw_scale,
        time_axis=args.time_axis,
        minimum_coverage=0.0,
        max_sampling_attempts=args.legacy_max_sampling_attempts,
        allow_empty_fallback=True,
    )

    print(f"4DFlowNet NIfTI patch {args.patch_size}, lr {args.initial_learning_rate}, batch {args.batch_size}")
    network = TrainerController(
        patch_size=args.patch_size,
        res_increase=args.res_increase,
        initial_learning_rate=args.initial_learning_rate,
        quicksave_enable=False,
        network_name=args.network_name,
        low_resblock=args.low_resblock,
        hi_resblock=args.hi_resblock,
    )
    network.init_model_dir()

    if args.restore:
        if not args.restore_dir or not args.restore_file:
            raise ValueError("--restore requires --restore-dir and --restore-file")
        print(f"Restoring model {args.restore_file}...")
        network.restore_model(args.restore_dir, args.restore_file)
        print("Learning rate", network.optimizer.param_groups[0]["lr"])

    network.train_network(train_loader, val_loader, n_epoch=args.epochs, testset=None)


if __name__ == "__main__":
    main()
