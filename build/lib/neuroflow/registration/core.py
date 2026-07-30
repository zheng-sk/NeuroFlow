#!/usr/bin/env python3
"""Core temporal-registration utilities for 4D NIfTI files using ANTsPy."""

from __future__ import annotations

import os
import time
import traceback

import ants

try:
    from .common import ensure_dir, frame3d_from_4d, make_mask_reference, parse_qc_frames, stack_4d
    from .qc import save_triptych_qc_examples
except ImportError:  # pragma: no cover - allows direct script execution from this folder
    from common import ensure_dir, frame3d_from_4d, make_mask_reference, parse_qc_frames, stack_4d
    from qc import save_triptych_qc_examples


def register_4d_nifti(
    input_path: str,
    output_path: str,
    qc_dir: str,
    ref_t: int = 0,
    reg_type: str = "Rigid",
    interpolator: str = "linear",
    qc_frames_str: str = "0,50,99",
    verbose: bool = False,
    show_frame_progress: bool = False,
    progress_label: str | None = None,
):
    """Register each frame of a 4D NIfTI to one reference frame and return transforms."""
    try:
        ensure_dir(os.path.dirname(output_path))
        ensure_dir(qc_dir)

        image_4d = ants.image_read(input_path, dimension=4, reorient=False)
        total_frames = image_4d.shape[3]
        if total_frames <= 1:
            print(f"Skipping {os.path.basename(input_path)}: expected 4D data, got T={total_frames}")
            return None

        if ref_t < 0 or ref_t >= total_frames:
            ref_t = 0

        reference = frame3d_from_4d(image_4d, ref_t)
        reference_mask = make_mask_reference(reference)

        warped_frames = [None] * total_frames
        warped_frames[ref_t] = reference
        registered_count = 1

        transforms_map = [None] * total_frames
        transforms_map[ref_t] = []

        for t in range(total_frames):
            if t == ref_t:
                continue

            frame_start = time.perf_counter()

            moving = frame3d_from_4d(image_4d, t)
            registration = ants.registration(
                fixed=reference,
                moving=moving,
                type_of_transform=reg_type,
                mask=reference_mask,
                moving_mask=None,
                mask_all_stages=True,
                verbose=verbose,
            )

            transforms_map[t] = registration["fwdtransforms"]
            warped_frames[t] = ants.apply_transforms(
                fixed=reference,
                moving=moving,
                transformlist=registration["fwdtransforms"],
                interpolator=interpolator,
            )
            registered_count += 1

            if show_frame_progress:
                elapsed = time.perf_counter() - frame_start
                prefix = progress_label or os.path.basename(input_path)
                print(
                    f"[TEMPORAL] {prefix}: frame {t + 1}/{total_frames} "
                    f"(completed {registered_count}/{total_frames}) in {elapsed:.2f}s"
                )

        ants.image_write(stack_4d(warped_frames, image_4d), output_path)

        qc_indices = parse_qc_frames(qc_frames_str, total_frames, ref_t)
        save_triptych_qc_examples(reference, image_4d, warped_frames, qc_indices, ref_t, qc_dir)
        return transforms_map

    except Exception as exc:  # pragma: no cover - defensive logging for CLI workflow
        print(f"Error registering {input_path}: {exc}")
        traceback.print_exc()
        return None


def apply_transforms_to_file(
    input_path: str,
    output_path: str,
    transforms_map,
    ref_img_path: str | None = None,
    ref_t: int = 0,
    interpolator: str = "linear",
) -> bool:
    """Apply an existing transform map to a scalar 4D image."""
    try:
        ensure_dir(os.path.dirname(output_path))

        image_4d = ants.image_read(input_path, dimension=4, reorient=False)
        total_frames = image_4d.shape[3]
        if total_frames != len(transforms_map):
            print(
                f"Warning: frame mismatch for {os.path.basename(input_path)} "
                f"(image={total_frames}, transforms={len(transforms_map)})"
            )
            return False

        if ref_img_path and os.path.exists(ref_img_path):
            ref_source_4d = ants.image_read(ref_img_path, dimension=4, reorient=False)
            fixed_reference = frame3d_from_4d(ref_source_4d, ref_t)
        else:
            fixed_reference = frame3d_from_4d(image_4d, ref_t)

        warped_frames = []
        for t in range(total_frames):
            moving = frame3d_from_4d(image_4d, t)
            transform_list = transforms_map[t]

            if not transform_list:
                warped = ants.resample_image_to_target(moving, fixed_reference, interp_type=1)
            else:
                warped = ants.apply_transforms(
                    fixed=fixed_reference,
                    moving=moving,
                    transformlist=transform_list,
                    interpolator=interpolator,
                )
            warped_frames.append(warped)

        ants.image_write(stack_4d(warped_frames, image_4d), output_path)
        return True

    except Exception as exc:  # pragma: no cover - defensive logging for CLI workflow
        print(f"Error applying transforms to {input_path}: {exc}")
        return False


def apply_transforms_to_velocity_triplet(
    vx_path: str,
    vy_path: str,
    vz_path: str,
    out_vx: str,
    out_vy: str,
    out_vz: str,
    transforms_map,
    fixed_ref_mag_path: str,
    ref_t: int = 0,
    interpolator: str = "linear",
) -> bool:
    """Apply existing transforms to velocity components (vector mode for proper reorientation)."""
    try:
        ensure_dir(os.path.dirname(out_vx))

        vx_4d = ants.image_read(vx_path, dimension=4, reorient=False)
        vy_4d = ants.image_read(vy_path, dimension=4, reorient=False)
        vz_4d = ants.image_read(vz_path, dimension=4, reorient=False)

        total_frames = vx_4d.shape[3]
        if vy_4d.shape[3] != total_frames or vz_4d.shape[3] != total_frames:
            print("Error: velocity components have different frame counts.")
            return False

        if len(transforms_map) != total_frames:
            print(f"Error: transforms map size ({len(transforms_map)}) != frame count ({total_frames})")
            return False

        magnitude_4d = ants.image_read(fixed_ref_mag_path, dimension=4, reorient=False)
        fixed_reference = frame3d_from_4d(magnitude_4d, ref_t)

        warped_vx = []
        warped_vy = []
        warped_vz = []

        for t in range(total_frames):
            vx = frame3d_from_4d(vx_4d, t)
            vy = frame3d_from_4d(vy_4d, t)
            vz = frame3d_from_4d(vz_4d, t)
            transform_list = transforms_map[t]

            if not transform_list:
                warped_vx.append(ants.resample_image_to_target(vx, fixed_reference, interp_type=1))
                warped_vy.append(ants.resample_image_to_target(vy, fixed_reference, interp_type=1))
                warped_vz.append(ants.resample_image_to_target(vz, fixed_reference, interp_type=1))
                continue

            vector_image = ants.merge_channels([vx, vy, vz])
            warped_vector = ants.apply_transforms(
                fixed=fixed_reference,
                moving=vector_image,
                transformlist=transform_list,
                interpolator=interpolator,
                imagetype=1,
                verbose=False,
            )

            channels = ants.split_channels(warped_vector)
            if len(channels) != 3:
                print(f"Error: expected 3 vector components after warp, got {len(channels)}")
                return False

            warped_vx.append(channels[0])
            warped_vy.append(channels[1])
            warped_vz.append(channels[2])

        ants.image_write(stack_4d(warped_vx, vx_4d), out_vx)
        ants.image_write(stack_4d(warped_vy, vy_4d), out_vy)
        ants.image_write(stack_4d(warped_vz, vz_4d), out_vz)
        return True

    except Exception as exc:  # pragma: no cover - defensive logging for CLI workflow
        print(f"Error applying vector transforms: {exc}")
        traceback.print_exc()
        return False
