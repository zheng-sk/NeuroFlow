import argparse
import random

import numpy as np
import torch

from Network.NiftiPatchDataset import create_nifti_full_volume_dataloader, create_nifti_patch_dataloader
from Network.TrainerController import TrainerController
from Network.model_factory import available_model_variants, normalize_model_variant


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
    parser.add_argument(
        "--model-variant",
        type=str,
        default="original",
        help=(
            "Model architecture variant. "
            f"Available: {', '.join(available_model_variants())}. "
            "Aliases: phase1/fase1, phase2/fase2."
        ),
    )
    parser.add_argument("--low-resblock", type=int, default=8, help="Number of low-res residual blocks.")
    parser.add_argument("--hi-resblock", type=int, default=4, help="Number of high-res residual blocks.")
    parser.add_argument("--train-samples-per-volume", type=int, default=64, help="Random patch samples per train volume per epoch.")
    parser.add_argument("--val-samples-per-volume", type=int, default=16, help="Patch samples per validation volume per epoch.")
    parser.add_argument(
        "--mag-scale",
        type=float,
        default=4095.0,
        help="Magnitude normalization divisor (used only with --mag-norm-mode divisor).",
    )
    parser.add_argument(
        "--mag-norm-mode",
        type=str,
        default="monai_minmax",
        choices=["monai_minmax", "divisor"],
        help="Magnitude normalization mode. monai_minmax applies MONAI ScaleIntensity to [0,1] per frame.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold.")
    parser.add_argument(
        "--predict-mag",
        action="store_true",
        help="Enable 4th output channel to predict HR magnitude (requires `hr_mag` in CSV).",
    )
    parser.add_argument(
        "--mag-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to magnitude MSE term when --predict-mag is enabled.",
    )
    parser.add_argument(
        "--non-fluid-loss-weight",
        type=float,
        default=0.1,
        help="Weight for loss outside vascular mask (lower values prioritize intraluminal ROI).",
    )
    parser.add_argument(
        "--outside-tv-weight",
        type=float,
        default=1e-5,
        help="Weight of anti-texture TV penalty outside vascular mask.",
    )
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
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Use MONAI CacheDataset to cache loaded NIfTI cases and reduce per-batch I/O.",
    )
    parser.add_argument(
        "--cache-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When caching is enabled, preload all cases at startup (default: true).",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable rotation augmentation and random time-frame sampling in training.",
    )
    parser.add_argument(
        "--deterministic-train-patches",
        action="store_true",
        help="Use center patch in training instead of random patch sampling.",
    )
    parser.add_argument(
        "--rotation-prob",
        type=float,
        default=0.5,
        help="Rotation augmentation probability in training (ignored with --no-augmentation).",
    )
    parser.add_argument(
        "--noise-aug-prob",
        type=float,
        default=0.0,
        help="Probability of adding synthetic non-vascular noise to LR inputs during training.",
    )
    parser.add_argument(
        "--noise-aug-phase-dist",
        type=str,
        default="student_t",
        choices=["normal", "student_t", "laplace", "uniform", "skew_normal", "generalized_normal", "hat"],
        help="Noise distribution for LR velocity channels.",
    )
    parser.add_argument(
        "--noise-aug-phase-scale",
        type=float,
        default=0.04,
        help="Base std/scale for LR velocity noise (normalized velocity domain).",
    )
    parser.add_argument(
        "--noise-aug-phase-shape",
        type=float,
        default=3.0,
        help="Shape parameter for phase distribution (df/beta/alpha depending on distribution).",
    )
    parser.add_argument(
        "--noise-aug-mag-dist",
        type=str,
        default="skew_normal",
        choices=["normal", "student_t", "laplace", "uniform", "skew_normal", "generalized_normal", "hat"],
        help="Noise distribution for LR magnitude channels.",
    )
    parser.add_argument(
        "--noise-aug-mag-scale",
        type=float,
        default=0.02,
        help="Base std/scale for LR magnitude noise (normalized magnitude domain).",
    )
    parser.add_argument(
        "--noise-aug-mag-shape",
        type=float,
        default=4.0,
        help="Shape parameter for magnitude distribution (df/beta/alpha depending on distribution).",
    )
    parser.add_argument(
        "--noise-aug-range-mult",
        type=float,
        default=2.0,
        help="Range exaggeration multiplier applied to synthetic noise scale.",
    )
    parser.add_argument(
        "--noise-aug-hat-mix",
        type=float,
        default=0.7,
        help="For `hat` distribution: fraction of samples from uniform component [0,1].",
    )
    parser.add_argument(
        "--noise-aug-apply-mag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable synthetic noise on LR magnitude channels.",
    )
    parser.add_argument(
        "--noise-aug-clip-mag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip noisy LR magnitude back to [0,1]. Default keeps full exaggerated range.",
    )
    parser.add_argument(
        "--noise-aug-fit-summary-csv",
        type=str,
        default="",
        help=(
            "Optional fit summary CSV from plot_nonvascular_noise_pdf_panel.py. "
            "If provided, use best-fit family per channel (loaded once) instead of generic per-modality families."
        ),
    )
    parser.add_argument(
        "--noise-aug-exaggerated-side-expand",
        type=float,
        default=1.0,
        help="Side expansion factor for synthetic noise profile (>=1).",
    )
    parser.add_argument(
        "--noise-aug-exaggerated-edge-boost",
        type=float,
        default=0.0,
        help="Tail boost factor for synthetic noise profile (>=0).",
    )
    parser.add_argument(
        "--noise-aug-exaggerated-edge-power",
        type=float,
        default=1.8,
        help="Tail boost shape power (>0).",
    )
    parser.add_argument(
        "--noise-aug-level-min",
        type=float,
        default=0.8,
        help="Minimum random augmentation level multiplier.",
    )
    parser.add_argument(
        "--noise-aug-level-max",
        type=float,
        default=1.4,
        help="Maximum random augmentation level multiplier.",
    )
    parser.add_argument(
        "--noise-aug-masked-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of generated noise amplitude also applied inside vascular mask [0,1]. "
            "0 keeps masked region untouched."
        ),
    )
    parser.add_argument(
        "--noise-aug-keep-original-prob",
        type=float,
        default=0.2,
        help=(
            "Within active noise augmentation, probability of leaving LR input unchanged "
            "(no synthetic background corruption)."
        ),
    )
    parser.add_argument(
        "--noise-aug-zero-outside-prob",
        type=float,
        default=0.2,
        help=(
            "Within active noise augmentation, probability of setting LR values outside mask to zero. "
            "Additive-noise mode uses the remaining probability."
        ),
    )
    parser.add_argument(
        "--tb-image-every-epochs",
        type=int,
        default=10,
        help="Log validation reconstruction images to TensorBoard every N epochs (0 disables).",
    )
    parser.add_argument(
        "--tb-image-axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice axis for TensorBoard validation reconstruction images.",
    )
    parser.add_argument(
        "--tb-image-batch-index",
        type=int,
        default=0,
        help="Batch index used for TensorBoard validation reconstruction images.",
    )
    parser.add_argument(
        "--val-full-volume",
        action="store_true",
        help="Run validation on full volumes using sliding-window inference instead of center patches.",
    )
    parser.add_argument(
        "--val-sw-patch-size",
        type=int,
        default=None,
        help="Sliding-window ROI size for full-volume validation (default: --patch-size).",
    )
    parser.add_argument(
        "--val-sw-batch-size",
        type=int,
        default=2,
        help="Sliding-window batch size for full-volume validation.",
    )
    parser.add_argument(
        "--val-sw-overlap",
        type=float,
        default=0.25,
        help="Sliding-window overlap [0,1) for full-volume validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic backend behavior (may reduce performance).",
    )
    parser.add_argument(
        "--accuracy-include-mag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include magnitude relative error in train/val accuracy when --predict-mag is enabled.",
    )
    parser.add_argument(
        "--accuracy-mag-weight",
        type=float,
        default=1.0,
        help="Weight of magnitude relative error in combined accuracy metric.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["none", "reduce_on_plateau"],
        help="Learning-rate scheduler strategy.",
    )
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5, help="LR reduction factor for plateau scheduler.")
    parser.add_argument("--lr-reduce-patience", type=int, default=8, help="Epoch patience before LR reduction.")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop if val_loss does not improve for N epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum val_loss improvement to reset early-stopping counter.",
    )
    parser.add_argument(
        "--overfit-patience",
        type=int,
        default=8,
        help="Stop if overfitting pattern persists for N epochs (0 disables).",
    )
    parser.add_argument(
        "--overfit-min-delta",
        type=float,
        default=0.0,
        help="Minimum val_loss increase used to detect overfitting pattern.",
    )
    parser.add_argument("--restore", action="store_true", help="Restore training from an existing checkpoint.")
    parser.add_argument("--restore-dir", type=str, default="", help="Checkpoint directory to restore from.")
    parser.add_argument("--restore-file", type=str, default="", help="Checkpoint filename (.pt).")
    args = parser.parse_args()
    args.model_variant = normalize_model_variant(args.model_variant)
    if args.model_variant not in set(available_model_variants()):
        valid = ", ".join(available_model_variants())
        raise ValueError(f"Invalid --model-variant={args.model_variant!r}. Valid options: {valid}")

    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            from monai.utils import set_determinism

            set_determinism(seed=seed)
        except Exception:
            pass

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if args.cache_dataset and args.num_workers > 0:
        print(
            "Warning: --cache-dataset with num_workers>0 may duplicate cache per worker. "
            "For max memory efficiency use --num-workers 0."
        )

    train_rotation_prob = 0.0 if args.no_augmentation else max(float(args.rotation_prob), 0.0)
    train_random_time_frame = not args.no_augmentation
    train_random_patch = not args.deterministic_train_patches

    train_loader = create_nifti_patch_dataloader(
        csv_path=args.train_csv,
        patch_size=args.patch_size,
        res_increase=args.res_increase,
        batch_size=args.batch_size,
        samples_per_volume=args.train_samples_per_volume,
        shuffle=True,
        augment=True,
        include_hr_mag=args.predict_mag,
        cache_dataset=args.cache_dataset,
        cache_eager=args.cache_eager,
        random_time_frame=train_random_time_frame,
        random_patch_sampling=train_random_patch,
        rotation_prob=train_rotation_prob,
        num_workers=args.num_workers,
        mag_scale=args.mag_scale,
        mag_norm_mode=args.mag_norm_mode,
        mask_threshold=args.mask_threshold,
        raw_phase_input=args.raw_phase_input,
        invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
        raw_center=args.raw_center,
        raw_scale=args.raw_scale,
        time_axis=args.time_axis,
        minimum_coverage=args.legacy_minimum_coverage,
        max_sampling_attempts=args.legacy_max_sampling_attempts,
        allow_empty_fallback=not args.legacy_disallow_empty_fallback,
        noise_aug_prob=args.noise_aug_prob,
        noise_aug_phase_dist=args.noise_aug_phase_dist,
        noise_aug_phase_scale=args.noise_aug_phase_scale,
        noise_aug_phase_shape=args.noise_aug_phase_shape,
        noise_aug_mag_dist=args.noise_aug_mag_dist,
        noise_aug_mag_scale=args.noise_aug_mag_scale,
        noise_aug_mag_shape=args.noise_aug_mag_shape,
        noise_aug_range_mult=args.noise_aug_range_mult,
        noise_aug_hat_mix=args.noise_aug_hat_mix,
        noise_aug_apply_mag=args.noise_aug_apply_mag,
        noise_aug_clip_mag=args.noise_aug_clip_mag,
        noise_aug_fit_summary_csv=args.noise_aug_fit_summary_csv,
        noise_aug_exaggerated_side_expand=args.noise_aug_exaggerated_side_expand,
        noise_aug_exaggerated_edge_boost=args.noise_aug_exaggerated_edge_boost,
        noise_aug_exaggerated_edge_power=args.noise_aug_exaggerated_edge_power,
        noise_aug_level_min=args.noise_aug_level_min,
        noise_aug_level_max=args.noise_aug_level_max,
        noise_aug_masked_fraction=args.noise_aug_masked_fraction,
        noise_aug_keep_original_prob=args.noise_aug_keep_original_prob,
        noise_aug_zero_outside_prob=args.noise_aug_zero_outside_prob,
        seed=args.seed,
    )
    if args.val_full_volume:
        if args.batch_size != 1:
            print("Validation full-volume mode forces val batch size to 1.")
        val_loader = create_nifti_full_volume_dataloader(
            csv_path=args.val_csv,
            batch_size=1,
            shuffle=False,
            include_hr_mag=args.predict_mag,
            cache_dataset=args.cache_dataset,
            cache_eager=args.cache_eager,
            random_time_frame=False,
            num_workers=args.num_workers,
            mag_scale=args.mag_scale,
            mag_norm_mode=args.mag_norm_mode,
            mask_threshold=args.mask_threshold,
            raw_phase_input=args.raw_phase_input,
            invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
            raw_center=args.raw_center,
            raw_scale=args.raw_scale,
            time_axis=args.time_axis,
            seed=args.seed,
        )
    else:
        val_loader = create_nifti_patch_dataloader(
            csv_path=args.val_csv,
            patch_size=args.patch_size,
            res_increase=args.res_increase,
            batch_size=args.batch_size,
            samples_per_volume=args.val_samples_per_volume,
            shuffle=False,
            augment=False,
            include_hr_mag=args.predict_mag,
            cache_dataset=args.cache_dataset,
            cache_eager=args.cache_eager,
            random_time_frame=False,
            random_patch_sampling=False,
            rotation_prob=0.0,
            num_workers=args.num_workers,
            mag_scale=args.mag_scale,
            mag_norm_mode=args.mag_norm_mode,
            mask_threshold=args.mask_threshold,
            raw_phase_input=args.raw_phase_input,
            invert_uv_sign_on_raw=args.legacy_invert_uv_sign_on_raw,
            raw_center=args.raw_center,
            raw_scale=args.raw_scale,
            time_axis=args.time_axis,
            minimum_coverage=0.0,
            max_sampling_attempts=args.legacy_max_sampling_attempts,
            allow_empty_fallback=True,
            noise_aug_prob=0.0,
            seed=args.seed,
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
        model_variant=args.model_variant,
        predict_mag=args.predict_mag,
        mag_loss_weight=args.mag_loss_weight,
        non_fluid_loss_weight=args.non_fluid_loss_weight,
        outside_tv_weight=args.outside_tv_weight,
        tb_image_every_n_epochs=args.tb_image_every_epochs,
        tb_image_axis=args.tb_image_axis,
        tb_image_batch_index=args.tb_image_batch_index,
        accuracy_include_mag=args.accuracy_include_mag,
        accuracy_mag_weight=args.accuracy_mag_weight,
        lr_scheduler=args.lr_scheduler,
        lr_reduce_factor=args.lr_reduce_factor,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_min=args.lr_min,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        overfit_patience=args.overfit_patience,
        overfit_min_delta=args.overfit_min_delta,
        val_full_volume=args.val_full_volume,
        val_sw_patch_size=args.val_sw_patch_size if args.val_sw_patch_size is not None else args.patch_size,
        val_sw_batch_size=args.val_sw_batch_size,
        val_sw_overlap=args.val_sw_overlap,
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
