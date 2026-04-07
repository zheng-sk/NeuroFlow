#!/usr/bin/env python3
"""Export multiple 3T CoW segmentation experiment datasets to nnU-Net raw format."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


EXPERIMENT_SPECS = {
    "mag_proj": {
        "description": "3D temporal projection of 3T magnitude only.",
        "channels": ["mag_proj"],
    },
    "mag_vel_proj": {
        "description": "3D temporal projection of 3T magnitude plus projected velocity components.",
        "channels": ["mag_proj", "vx_proj", "vy_proj", "vz_proj"],
    },
    "angio_mag_speed": {
        "description": "3D angio proxy built from 3T magnitude and speed summaries.",
        "channels": ["angio"],
    },
    "mag_frame": {
        "description": "Each 4D magnitude frame exported as an independent 3D training case.",
        "channels": ["mag_frame"],
    },
    "mag_vel_frame": {
        "description": "Each 4D frame exported with magnitude and 3 velocity components as channels.",
        "channels": ["mag_frame", "vx_frame", "vy_frame", "vz_frame"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one or more 3T CoW segmentation experiments from the master manifest "
            "to nnU-Net raw dataset format."
        )
    )
    parser.add_argument("--manifest-csv", required=True, help="Master manifest CSV.")
    parser.add_argument("--nnunet-raw-dir", default="nnUNet_raw", help="Root nnUNet_raw folder.")
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(EXPERIMENT_SPECS),
        required=True,
        help="Experiment to export. Repeat to export several datasets in one run.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        type=int,
        required=True,
        help="Dataset id matching each --experiment in order.",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        default=None,
        help="Optional dataset name matching each --experiment in order. Defaults are generated automatically.",
    )
    parser.add_argument("--case-id-col", default="case_id")
    parser.add_argument("--mag-col", default="lr_mag")
    parser.add_argument("--u-col", default="lr_u")
    parser.add_argument("--v-col", default="lr_v")
    parser.add_argument("--w-col", default="lr_w")
    parser.add_argument("--mask-col", default="mask_gt")
    parser.add_argument("--ready-col", default="ready_for_nnunet")
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--mag-projection-method",
        choices=["median", "max", "percentile", "topk_mean"],
        default="max",
        help=(
            "Temporal projection for magnitude. Default is max to match current "
            "segment_cow_crops.py behavior."
        ),
    )
    parser.add_argument("--mag-projection-percentile", type=float, default=95.0)
    parser.add_argument("--mag-projection-topk", type=int, default=3)

    parser.add_argument(
        "--vel-projection-method",
        choices=["abs_max", "abs_percentile", "abs_topk_mean", "rms"],
        default="abs_max",
        help="Temporal summary used for projected velocity-component channels.",
    )
    parser.add_argument("--vel-projection-percentile", type=float, default=95.0)
    parser.add_argument("--vel-projection-topk", type=int, default=3)

    parser.add_argument(
        "--speed-percentile",
        type=float,
        default=90.0,
        help="Temporal percentile for speed in angio_mag_speed.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Temporal stride for frame-based experiments.",
    )
    parser.add_argument(
        "--max-frames-per-case",
        type=int,
        default=0,
        help="Optional cap on exported frames per case for frame-based experiments (0 disables).",
    )
    parser.add_argument("--mask-time-index", type=int, default=0, help="Frame used when GT mask is 4D.")
    return parser.parse_args()


def resolve_path(value: str, csv_parent: Path) -> Path:
    raw = (value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd().resolve() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (csv_parent / path).resolve()


def sanitize_identifier(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip())
    return cleaned.strip("_") or "case"


def maybe_remove_dir(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)


def select_mask_3d(mask_data: np.ndarray, time_index: int) -> np.ndarray:
    if mask_data.ndim == 3:
        return mask_data.astype(np.uint8, copy=False)
    if mask_data.ndim != 4:
        raise ValueError(f"Expected 3D/4D mask, got {mask_data.shape}")
    if not (0 <= time_index < mask_data.shape[3]):
        raise ValueError(f"Invalid mask_time_index={time_index} for mask shape {mask_data.shape}")
    return mask_data[..., time_index].astype(np.uint8, copy=False)


def project_volume(vol: np.ndarray, method: str, percentile: float, topk: int) -> np.ndarray:
    if vol.ndim == 3:
        return vol.astype(np.float32, copy=False)
    if vol.ndim != 4:
        raise ValueError(f"Expected 3D/4D volume, got shape={vol.shape}")
    if method == "median":
        return np.median(vol, axis=3).astype(np.float32)
    if method == "max":
        return np.max(vol, axis=3).astype(np.float32)
    if method == "percentile":
        return np.percentile(vol, percentile, axis=3).astype(np.float32)
    if method == "topk_mean":
        k = int(np.clip(topk, 1, vol.shape[3]))
        sorted_vol = np.sort(vol, axis=3)
        return np.mean(sorted_vol[..., -k:], axis=3).astype(np.float32)
    raise ValueError(f"Unsupported projection method: {method}")


def project_velocity_component(vol: np.ndarray, method: str, percentile: float, topk: int) -> np.ndarray:
    if vol.ndim == 3:
        base = np.abs(vol).astype(np.float32)
        return base if method != "rms" else np.sqrt(base * base, dtype=np.float32)
    if vol.ndim != 4:
        raise ValueError(f"Expected 3D/4D velocity volume, got shape={vol.shape}")
    if method == "abs_max":
        return np.max(np.abs(vol), axis=3).astype(np.float32)
    if method == "abs_percentile":
        return np.percentile(np.abs(vol), percentile, axis=3).astype(np.float32)
    if method == "abs_topk_mean":
        abs_vol = np.abs(vol)
        k = int(np.clip(topk, 1, abs_vol.shape[3]))
        sorted_vol = np.sort(abs_vol, axis=3)
        return np.mean(sorted_vol[..., -k:], axis=3).astype(np.float32)
    if method == "rms":
        return np.sqrt(np.mean(np.square(vol, dtype=np.float32), axis=3), dtype=np.float32).astype(np.float32)
    raise ValueError(f"Unsupported velocity projection method: {method}")


def robust_norm(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    finite = np.asarray(vol[np.isfinite(vol)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip(vol, lo, hi)
    return ((clipped - lo) / (hi - lo + eps)).astype(np.float32)


def compute_angio_mag_speed(mag: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, speed_percentile: float) -> np.ndarray:
    mag_med = project_volume(mag, method="median", percentile=50.0, topk=1)
    speed4d = np.sqrt(np.square(vx, dtype=np.float32) + np.square(vy, dtype=np.float32) + np.square(vz, dtype=np.float32))
    speed_proj = np.percentile(speed4d, speed_percentile, axis=3).astype(np.float32) if speed4d.ndim == 4 else speed4d.astype(np.float32)
    return (robust_norm(mag_med) * robust_norm(speed_proj)).astype(np.float32)


def iter_frame_indices(num_frames: int, frame_step: int, max_frames_per_case: int) -> list[int]:
    indices = list(range(0, num_frames, max(1, frame_step)))
    if max_frames_per_case > 0:
        indices = indices[: max_frames_per_case]
    return indices


def write_dataset_json(dataset_dir: Path, dataset_name: str, num_cases: int, channel_names: list[str], description: str) -> None:
    payload = {
        "channel_names": {str(i): name for i, name in enumerate(channel_names)},
        "labels": {"background": 0, "cow": 1},
        "numTraining": int(num_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": description,
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_channels(images_tr: Path, case_token: str, channels: list[np.ndarray], affine, header) -> None:
    for idx, arr in enumerate(channels):
        out_path = images_tr / f"{case_token}_{idx:04d}.nii.gz"
        out_header = header.copy()
        out_header.set_data_shape(arr.shape)
        out_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine, out_header), str(out_path))


def save_label(labels_tr: Path, case_token: str, mask_bin: np.ndarray, affine, header) -> None:
    out_path = labels_tr / f"{case_token}.nii.gz"
    out_header = header.copy()
    out_header.set_data_shape(mask_bin.shape)
    out_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask_bin.astype(np.uint8), affine, out_header), str(out_path))


def default_dataset_name(exp: str) -> str:
    mapping = {
        "mag_proj": "CoW3TMagProj",
        "mag_vel_proj": "CoW3TMagVelProj",
        "angio_mag_speed": "CoW3TAngioMagSpeed",
        "mag_frame": "CoW3TMagFrame",
        "mag_vel_frame": "CoW3TMagVelFrame",
    }
    return mapping[exp]


def export_experiment(args: argparse.Namespace, exp: str, dataset_id: int, dataset_name: str, rows: list[dict[str, str]], manifest_parent: Path, nnunet_raw_dir: Path) -> None:
    spec = EXPERIMENT_SPECS[exp]
    dataset_folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_dir = nnunet_raw_dir / dataset_folder_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} already exists. Use --overwrite to replace it.")
        maybe_remove_dir(dataset_dir)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    for row in rows:
        case_id = (row.get(args.case_id_col) or "").strip()
        if not case_id:
            skipped += 1
            continue

        ready = (row.get(args.ready_col) or "").strip()
        if not args.allow_not_ready and ready not in {"1", "true", "True"}:
            skipped += 1
            if args.verbose:
                print(f"[SKIP:{exp}] {case_id}: ready_for_nnunet={ready!r}")
            continue

        mag_path = resolve_path(row.get(args.mag_col, ""), manifest_parent)
        u_path = resolve_path(row.get(args.u_col, ""), manifest_parent)
        v_path = resolve_path(row.get(args.v_col, ""), manifest_parent)
        w_path = resolve_path(row.get(args.w_col, ""), manifest_parent)
        mask_path = resolve_path(row.get(args.mask_col, ""), manifest_parent)

        required = [mag_path, mask_path]
        if exp in {"mag_vel_proj", "angio_mag_speed", "mag_vel_frame"}:
            required.extend([u_path, v_path, w_path])
        if any(not p.is_file() for p in required):
            raise FileNotFoundError(f"{case_id}: missing required files for experiment {exp}.")

        mag_img = nib.load(str(mag_path))
        mag_data = np.asarray(mag_img.dataobj, dtype=np.float32)
        mask_img = nib.load(str(mask_path))
        mask_data = np.asarray(mask_img.dataobj)
        mask_3d = select_mask_3d(mask_data, time_index=args.mask_time_index)
        mask_bin = (mask_3d > 0).astype(np.uint8)

        if exp in {"mag_vel_proj", "angio_mag_speed", "mag_vel_frame"}:
            u_img = nib.load(str(u_path))
            v_img = nib.load(str(v_path))
            w_img = nib.load(str(w_path))
            u_data = np.asarray(u_img.dataobj, dtype=np.float32)
            v_data = np.asarray(v_img.dataobj, dtype=np.float32)
            w_data = np.asarray(w_img.dataobj, dtype=np.float32)

        case_token_base = sanitize_identifier(case_id)

        if exp == "mag_proj":
            channels = [
                project_volume(
                    mag_data,
                    method=args.mag_projection_method,
                    percentile=args.mag_projection_percentile,
                    topk=args.mag_projection_topk,
                )
            ]
            if tuple(channels[0].shape) != tuple(mask_bin.shape):
                raise ValueError(f"{case_id}: mag_proj shape {channels[0].shape} != mask shape {mask_bin.shape}")
            save_channels(images_tr, case_token_base, channels, mag_img.affine, mag_img.header)
            save_label(labels_tr, case_token_base, mask_bin, mask_img.affine, mask_img.header)
            exported += 1

        elif exp == "mag_vel_proj":
            channels = [
                project_volume(mag_data, args.mag_projection_method, args.mag_projection_percentile, args.mag_projection_topk),
                project_velocity_component(u_data, args.vel_projection_method, args.vel_projection_percentile, args.vel_projection_topk),
                project_velocity_component(v_data, args.vel_projection_method, args.vel_projection_percentile, args.vel_projection_topk),
                project_velocity_component(w_data, args.vel_projection_method, args.vel_projection_percentile, args.vel_projection_topk),
            ]
            if any(tuple(ch.shape) != tuple(mask_bin.shape) for ch in channels):
                raise ValueError(f"{case_id}: projected channels do not match mask shape {mask_bin.shape}")
            save_channels(images_tr, case_token_base, channels, mag_img.affine, mag_img.header)
            save_label(labels_tr, case_token_base, mask_bin, mask_img.affine, mask_img.header)
            exported += 1

        elif exp == "angio_mag_speed":
            if mag_data.ndim != 4 or u_data.ndim != 4 or v_data.ndim != 4 or w_data.ndim != 4:
                raise ValueError(f"{case_id}: angio_mag_speed requires 4D mag and velocity volumes.")
            angio = compute_angio_mag_speed(mag_data, u_data, v_data, w_data, speed_percentile=args.speed_percentile)
            if tuple(angio.shape) != tuple(mask_bin.shape):
                raise ValueError(f"{case_id}: angio shape {angio.shape} != mask shape {mask_bin.shape}")
            save_channels(images_tr, case_token_base, [angio], mag_img.affine, mag_img.header)
            save_label(labels_tr, case_token_base, mask_bin, mask_img.affine, mask_img.header)
            exported += 1

        elif exp == "mag_frame":
            num_frames = mag_data.shape[3] if mag_data.ndim == 4 else 1
            for t in iter_frame_indices(num_frames, args.frame_step, args.max_frames_per_case):
                frame = mag_data[..., t] if mag_data.ndim == 4 else mag_data
                case_token = f"{case_token_base}_t{t:03d}"
                if tuple(frame.shape) != tuple(mask_bin.shape):
                    raise ValueError(f"{case_id} frame {t}: shape {frame.shape} != mask shape {mask_bin.shape}")
                save_channels(images_tr, case_token, [frame.astype(np.float32)], mag_img.affine, mag_img.header)
                save_label(labels_tr, case_token, mask_bin, mask_img.affine, mask_img.header)
                exported += 1

        elif exp == "mag_vel_frame":
            if mag_data.ndim != 4 or u_data.ndim != 4 or v_data.ndim != 4 or w_data.ndim != 4:
                raise ValueError(f"{case_id}: mag_vel_frame requires 4D mag and velocity volumes.")
            num_frames = mag_data.shape[3]
            for t in iter_frame_indices(num_frames, args.frame_step, args.max_frames_per_case):
                channels = [
                    mag_data[..., t].astype(np.float32),
                    u_data[..., t].astype(np.float32),
                    v_data[..., t].astype(np.float32),
                    w_data[..., t].astype(np.float32),
                ]
                if any(tuple(ch.shape) != tuple(mask_bin.shape) for ch in channels):
                    raise ValueError(f"{case_id} frame {t}: frame channels do not match mask shape {mask_bin.shape}")
                case_token = f"{case_token_base}_t{t:03d}"
                save_channels(images_tr, case_token, channels, mag_img.affine, mag_img.header)
                save_label(labels_tr, case_token, mask_bin, mask_img.affine, mask_img.header)
                exported += 1

        else:
            raise ValueError(f"Unsupported experiment: {exp}")

        if args.verbose:
            print(f"[OK:{exp}] {case_id}")

    description = spec["description"]
    if exp == "mag_proj":
        description += f" MAG projection method={args.mag_projection_method}."
    if exp == "mag_vel_proj":
        description += (
            f" MAG projection={args.mag_projection_method}; velocity projection={args.vel_projection_method}."
        )
    if exp == "angio_mag_speed":
        description += f" speed percentile={args.speed_percentile}."
    if exp in {"mag_frame", "mag_vel_frame"}:
        description += f" frame_step={args.frame_step}, max_frames_per_case={args.max_frames_per_case}."

    write_dataset_json(dataset_dir, dataset_name, exported, spec["channels"], description)

    print(f"nnU-Net export complete: {exp}")
    print(f"Dataset folder:    {dataset_dir}")
    print(f"Exported cases:    {exported}")
    print(f"Skipped cases:     {skipped}")


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    manifest_parent = manifest_csv.resolve().parent
    nnunet_raw_dir = Path(args.nnunet_raw_dir).resolve()

    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    if len(args.experiment) != len(args.dataset_id):
        raise ValueError("You must provide one --dataset-id for each --experiment.")
    if args.dataset_name is not None and len(args.dataset_name) not in {0, len(args.experiment)}:
        raise ValueError("If provided, --dataset-name must be repeated once per --experiment.")

    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Manifest CSV has no header: {manifest_csv}")
        rows = list(reader)

    provided_names = args.dataset_name or []
    for idx, exp in enumerate(args.experiment):
        dataset_name = provided_names[idx] if idx < len(provided_names) else default_dataset_name(exp)
        export_experiment(
            args=args,
            exp=exp,
            dataset_id=args.dataset_id[idx],
            dataset_name=dataset_name,
            rows=rows,
            manifest_parent=manifest_parent,
            nnunet_raw_dir=nnunet_raw_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
