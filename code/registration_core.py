#!/usr/bin/env python3
"""
Core temporal-registration utilities for 4D NIfTI files using ANTsPy.

Main flow:
1) Estimate transforms from a 4D magnitude sequence (frame t -> reference frame).
2) Reuse transforms for scalar files or velocity triplets.
"""

import os
import traceback

import ants
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def frame3d_from_4d(img4d: "ants.ANTsImage", t: int) -> "ants.ANTsImage":
    """Extract a 3D frame from a 4D ANTs image while preserving geometry metadata."""
    array = img4d.numpy()[..., t]
    return ants.from_numpy(
        array,
        spacing=img4d.spacing[:3],
        origin=img4d.origin[:3],
        direction=img4d.direction[:3, :3],
    )


def make_mask_reference(ref3d: "ants.ANTsImage") -> "ants.ANTsImage":
    """Build a robust foreground mask from the reference frame."""
    mask = ants.get_mask(ref3d, cleanup=2)
    mask = ants.iMath(mask, "FillHoles")
    mask = ants.iMath(mask, "GetLargestComponent")
    return mask


def _parse_qc_frames(qc_frames_str: str, total_frames: int, ref_t: int) -> list[int]:
    percentiles = []
    for token in qc_frames_str.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 99:
            raise ValueError(f"QC percentile must be in [0, 99], got: {value}")
        percentiles.append(value)

    indices = sorted(set(int(round((p / 100.0) * (total_frames - 1))) for p in percentiles))
    if ref_t not in indices:
        indices.append(ref_t)
    return sorted(set(indices))


def register_4d_nifti(
    input_path: str,
    output_path: str,
    qc_dir: str,
    ref_t: int = 0,
    reg_type: str = "Rigid",
    interpolator: str = "linear",
    qc_frames_str: str = "0,50,99",
    verbose: bool = False,
):
    """
    Register each time frame of a 4D NIfTI to a reference frame.

    Returns a per-frame transform map, or None on failure.
    """
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

        transforms_map = [None] * total_frames
        transforms_map[ref_t] = []

        for t in range(total_frames):
            if t == ref_t:
                continue

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
            warped = ants.apply_transforms(
                fixed=reference,
                moving=moving,
                transformlist=registration["fwdtransforms"],
                interpolator=interpolator,
            )
            warped_frames[t] = warped

        warped_np = np.stack([warped_frames[t].numpy() for t in range(total_frames)], axis=-1)
        warped_4d = ants.from_numpy(
            warped_np,
            origin=image_4d.origin,
            spacing=tuple(image_4d.spacing),
            direction=image_4d.direction,
        )
        ants.image_write(warped_4d, output_path)

        qc_indices = _parse_qc_frames(qc_frames_str, total_frames, ref_t)
        _save_qc_examples(reference, image_4d, warped_frames, qc_indices, ref_t, qc_dir)

        return transforms_map

    except Exception as exc:
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
    """Apply temporal transforms to a scalar 4D image."""
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

        warped_np = np.stack([frame.numpy() for frame in warped_frames], axis=-1)
        out_image = ants.from_numpy(
            warped_np,
            origin=image_4d.origin,
            spacing=tuple(image_4d.spacing),
            direction=image_4d.direction,
        )
        ants.image_write(out_image, output_path)
        return True

    except Exception as exc:
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
    """
    Apply temporal transforms to a velocity vector field triplet (Vx, Vy, Vz).

    Uses vector mode (imagetype=1) so ANTs reorients vectors correctly.
    """
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

        def stack_4d(frames, template_4d):
            array = np.stack([frame.numpy() for frame in frames], axis=-1)
            return ants.from_numpy(
                array,
                origin=template_4d.origin,
                spacing=template_4d.spacing,
                direction=template_4d.direction,
            )

        ants.image_write(stack_4d(warped_vx, vx_4d), out_vx)
        ants.image_write(stack_4d(warped_vy, vy_4d), out_vy)
        ants.image_write(stack_4d(warped_vz, vz_4d), out_vz)
        return True

    except Exception as exc:
        print(f"Error applying vector transforms: {exc}")
        traceback.print_exc()
        return False


def _save_qc_examples(
    reference: "ants.ANTsImage",
    original_4d: "ants.ANTsImage",
    warped_frames,
    qc_indices: list[int],
    ref_t: int,
    qc_dir: str,
) -> None:
    reference_slice = reference.numpy()[:, :, reference.shape[2] // 2]

    for t in qc_indices:
        moving_slice = frame3d_from_4d(original_4d, t).numpy()[:, :, reference.shape[2] // 2]
        warped_slice = warped_frames[t].numpy()[:, :, reference.shape[2] // 2]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(reference_slice, cmap="gray")
        plt.title(f"Reference (t={ref_t})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(moving_slice, cmap="gray")
        plt.title(f"Before (t={t})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(warped_slice, cmap="gray")
        plt.title(f"After (t={t})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(qc_dir, f"qc_t{t:03d}.png"), dpi=150)
        plt.close()
