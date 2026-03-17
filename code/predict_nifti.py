import argparse
import os
import sys
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensity

# Add src to python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from Network.model_factory import build_sr_model, normalize_model_variant


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def ensure_time_first(data: np.ndarray, time_axis: int):
    if data.ndim == 3:
        return data[np.newaxis, ...]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape}")
    if time_axis < 0:
        time_axis = data.ndim + time_axis
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)
    return data


def load_model(
    model_path,
    res_increase,
    low_resblock,
    hi_resblock,
    device,
    predict_mag: Optional[bool] = None,
    model_variant: Optional[str] = None,
    channel_nr: int = 64,
):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if predict_mag is None and "predict_mag" in checkpoint:
            predict_mag = bool(checkpoint["predict_mag"])
        if model_variant in (None, ""):
            model_variant = checkpoint.get("model_variant", "original")
        res_increase = int(checkpoint.get("res_increase", res_increase))
        low_resblock = int(checkpoint.get("low_resblock", low_resblock))
        hi_resblock = int(checkpoint.get("hi_resblock", hi_resblock))
        channel_nr = int(checkpoint.get("channel_nr", channel_nr))

    if predict_mag is None:
        predict_mag = False
    model_variant = normalize_model_variant(model_variant or "original")

    model = build_sr_model(
        model_variant=model_variant,
        res_increase=res_increase,
        low_resblock=low_resblock,
        hi_resblock=hi_resblock,
        channel_nr=channel_nr,
        predict_mag=predict_mag,
    ).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, bool(predict_mag), model_variant, res_increase


def adjust_affine_for_upsample(affine: np.ndarray, res_increase: int):
    new_affine = affine.copy()
    new_affine[:3, :3] = new_affine[:3, :3] / float(res_increase)
    return new_affine


def raw_to_velocity(data: np.ndarray, venc: float, raw_center: float, raw_scale: float, invert_sign: bool):
    out = data.astype(np.float32)
    mn = float(np.min(out))
    mx = float(np.max(out))
    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500.0) and (mx > 500.0) and (max_abs <= 8192.0) and (centered_ratio < 0.25)

    if is_unsigned_raw:
        out = (out - float(raw_center)) / float(raw_scale) * float(venc)
    elif is_signed_raw:
        scale = 4096.0 if max_abs > 3000.0 else 2048.0
        out = out / scale * float(venc)
    else:
        # Legacy fallback for non-raw-like ranges when --raw-phase-input is set.
        out = (out - float(raw_center)) / float(raw_scale) * float(venc)

    if invert_sign:
        out = -out
    return out


def normalize_magnitude(data: np.ndarray, mode: str, mag_scale: float) -> np.ndarray:
    mode = str(mode).strip().lower()
    arr = data.astype(np.float32)
    if mode == "divisor":
        return arr / float(mag_scale)
    if mode == "monai_minmax":
        scaler = ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=False)
        out = scaler(arr)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported mag_norm_mode={mode!r}. Use 'monai_minmax' or 'divisor'.")


def _legacy_compute_padding(shape_xyz, patch_size: int):
    effective_patch_size = patch_size - 4
    if effective_patch_size <= 0:
        raise ValueError("Legacy overlap mode requires patch_size > 4.")

    side_pad = (patch_size - effective_patch_size) // 2
    padded = [shape_xyz[0] + 2 * side_pad, shape_xyz[1] + 2 * side_pad, shape_xyz[2] + 2 * side_pad]

    end_pads = []
    for dim in padded:
        res = dim % effective_patch_size
        if res > (2 * side_pad):
            pad = patch_size - res
        else:
            pad = (2 * side_pad) - res
        end_pads.append(pad)
    return side_pad, effective_patch_size, tuple(end_pads)


def _legacy_pad_channel(img: np.ndarray, side_pad: int, end_pads):
    img = np.pad(img, ((side_pad, side_pad), (side_pad, side_pad), (side_pad, side_pad)), mode="constant")
    img = np.pad(img, ((0, end_pads[0]), (0, end_pads[1]), (0, end_pads[2])), mode="constant")
    return img


def _legacy_extract_patches(img: np.ndarray, patch_size: int, effective_patch_size: int):
    all_pads = patch_size - effective_patch_size
    nr_x = (img.shape[0] - all_pads) // effective_patch_size
    nr_y = (img.shape[1] - all_pads) // effective_patch_size
    nr_z = (img.shape[2] - all_pads) // effective_patch_size

    patches = []
    for i in range(nr_x):
        x_start = i * effective_patch_size
        for j in range(nr_y):
            y_start = j * effective_patch_size
            for k in range(nr_z):
                z_start = k * effective_patch_size
                patches.append(img[x_start : x_start + patch_size, y_start : y_start + patch_size, z_start : z_start + patch_size])
    return np.asarray(patches, dtype=np.float32), nr_x, nr_y, nr_z


def _legacy_patchup_with_overlap(patches: np.ndarray, nr_x: int, nr_y: int, nr_z: int, patch_size: int, res_increase: int, end_pads):
    effective_patch_size = patch_size - 4
    side_pad = (patch_size - effective_patch_size) // 2
    side_pad_hr = side_pad * res_increase

    hr_patch_size = patches.shape[1]
    n = hr_patch_size - side_pad_hr
    patches = patches[:, side_pad_hr:n, side_pad_hr:n, side_pad_hr:n]

    z_stacks = []
    for k in range(len(patches) // nr_z):
        z_start = k * nr_z
        z_stacks.append(np.concatenate(patches[z_start : z_start + nr_z], axis=2))

    y_stacks = []
    for j in range(len(z_stacks) // nr_y):
        y_start = j * nr_y
        y_stacks.append(np.concatenate(z_stacks[y_start : y_start + nr_y], axis=1))

    results = np.concatenate(y_stacks, axis=0)

    padding_hr = (end_pads[0] * res_increase, end_pads[1] * res_increase, end_pads[2] * res_increase)
    if padding_hr[0] > 0:
        results = results[:-padding_hr[0], :, :]
    if padding_hr[1] > 0:
        results = results[:, :-padding_hr[1], :]
    if padding_hr[2] > 0:
        results = results[:, :, :-padding_hr[2]]
    return results


def _legacy_overlap_predict(model, lr_input: np.ndarray, patch_size: int, res_increase: int, batch_size: int, device):
    # lr_input shape: [6, X, Y, Z]
    side_pad, effective_patch_size, end_pads = _legacy_compute_padding(lr_input.shape[1:], patch_size)

    channel_patches = []
    nr_x = nr_y = nr_z = None
    for c in range(lr_input.shape[0]):
        padded = _legacy_pad_channel(lr_input[c], side_pad=side_pad, end_pads=end_pads)
        patches, x, y, z = _legacy_extract_patches(padded, patch_size=patch_size, effective_patch_size=effective_patch_size)
        channel_patches.append(patches)
        if nr_x is None:
            nr_x, nr_y, nr_z = x, y, z

    total_patches = channel_patches[0].shape[0]
    pred_batches = []
    for start in range(0, total_patches, batch_size):
        end = min(start + batch_size, total_patches)
        with torch.no_grad():
            u = torch.from_numpy(channel_patches[0][start:end]).unsqueeze(1).to(device)
            v = torch.from_numpy(channel_patches[1][start:end]).unsqueeze(1).to(device)
            w = torch.from_numpy(channel_patches[2][start:end]).unsqueeze(1).to(device)
            um = torch.from_numpy(channel_patches[3][start:end]).unsqueeze(1).to(device)
            vm = torch.from_numpy(channel_patches[4][start:end]).unsqueeze(1).to(device)
            wm = torch.from_numpy(channel_patches[5][start:end]).unsqueeze(1).to(device)
            pred = model(u, v, w, um, vm, wm).cpu().numpy()
        pred_batches.append(pred)

    pred_all = np.concatenate(pred_batches, axis=0)  # [N, C, hp, hp, hp], C in {3,4}
    pred_u = _legacy_patchup_with_overlap(
        pred_all[:, 0], nr_x=nr_x, nr_y=nr_y, nr_z=nr_z, patch_size=patch_size, res_increase=res_increase, end_pads=end_pads
    )
    pred_v = _legacy_patchup_with_overlap(
        pred_all[:, 1], nr_x=nr_x, nr_y=nr_y, nr_z=nr_z, patch_size=patch_size, res_increase=res_increase, end_pads=end_pads
    )
    pred_w = _legacy_patchup_with_overlap(
        pred_all[:, 2], nr_x=nr_x, nr_y=nr_y, nr_z=nr_z, patch_size=patch_size, res_increase=res_increase, end_pads=end_pads
    )
    outputs = [pred_u, pred_v, pred_w]
    if pred_all.shape[1] == 4:
        pred_mag = _legacy_patchup_with_overlap(
            pred_all[:, 3],
            nr_x=nr_x,
            nr_y=nr_y,
            nr_z=nr_z,
            patch_size=patch_size,
            res_increase=res_increase,
            end_pads=end_pads,
        )
        outputs.append(pred_mag)
    return np.stack(outputs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Predict 4DFlowNet output directly from NIfTI input.")
    parser.add_argument("--u", type=str, required=True, help="Input LR velocity U NIfTI (3D/4D).")
    parser.add_argument("--v", type=str, required=True, help="Input LR velocity V NIfTI (3D/4D).")
    parser.add_argument("--w", type=str, required=True, help="Input LR velocity W NIfTI (3D/4D).")
    parser.add_argument("--mag-u", type=str, default="", help="Input LR magnitude NIfTI. If used alone, it is shared across components.")
    parser.add_argument("--mag-v", type=str, default="", help="Optional extra LR magnitude NIfTI. Defaults to the shared magnitude.")
    parser.add_argument("--mag-w", type=str, default="", help="Optional extra LR magnitude NIfTI. Defaults to the shared magnitude.")
    parser.add_argument("--mag", type=str, default="", help="Single LR magnitude NIfTI shared across components (recommended).")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint path (.pt).")
    parser.add_argument("--output-prefix", type=str, required=True, help="Output prefix (without _u/_v/_w suffix).")
    parser.add_argument("--patch-size", type=int, default=16, help="LR inference patch size.")
    parser.add_argument("--res-increase", type=int, default=2, help="Upsampling ratio.")
    parser.add_argument("--sw-batch-size", type=int, default=2, help="Sliding window batch size.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding-window overlap [0,1).")
    parser.add_argument(
        "--legacy-overlap-inference",
        action="store_true",
        help="Use legacy PatchGenerator-style overlap/trim reconstruction instead of MONAI sliding window.",
    )
    parser.add_argument("--venc", type=float, default=0.0, help="Optional venc override for normalization/denormalization.")
    parser.add_argument(
        "--raw-phase-input",
        dest="raw_phase_input",
        action="store_true",
        default=True,
        help="Assume velocity NIfTI values are raw phase-like and convert before normalization.",
    )
    parser.add_argument(
        "--already-velocity-input",
        dest="raw_phase_input",
        action="store_false",
        help="Disable raw-phase conversion (input velocity already physical).",
    )
    parser.add_argument(
        "--legacy-invert-uv-sign-on-raw",
        action="store_true",
        help="Legacy mode: invert U/V signs after RAW->velocity conversion. Keep disabled for DICOM->NIfTI outputs that already applied LPS->RAS sign correction.",
    )
    parser.add_argument("--raw-center", type=float, default=2048.0, help="Raw phase center value.")
    parser.add_argument("--raw-scale", type=float, default=2048.0, help="Raw phase scaling denominator.")
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
    parser.add_argument("--round-small-values", action="store_true", help="Zero values under venc/2048.")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D NIfTI (default last axis).")
    parser.add_argument("--low-resblock", type=int, default=8, help="Number of low-res residual blocks.")
    parser.add_argument("--hi-resblock", type=int, default=4, help="Number of high-res residual blocks.")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="",
        help="Optional model variant override. If empty, uses checkpoint metadata when available.",
    )
    parser.add_argument(
        "--predict-mag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable 4-channel output (u,v,w,mag). "
            "If not provided, tries checkpoint metadata (`predict_mag`)."
        ),
    )
    args = parser.parse_args()

    u, u_img = load_nifti(args.u)
    v, _ = load_nifti(args.v)
    w, _ = load_nifti(args.w)

    u = ensure_time_first(u, args.time_axis)
    v = ensure_time_first(v, args.time_axis)
    w = ensure_time_first(w, args.time_axis)
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("U/V/W shape mismatch")

    mag_u = mag_v = mag_w = None
    if args.mag_u:
        mag_u, _ = load_nifti(args.mag_u)
        mag_u = ensure_time_first(mag_u, args.time_axis)
    if args.mag_v:
        mag_v, _ = load_nifti(args.mag_v)
        mag_v = ensure_time_first(mag_v, args.time_axis)
    if args.mag_w:
        mag_w, _ = load_nifti(args.mag_w)
        mag_w = ensure_time_first(mag_w, args.time_axis)

    if args.mag and (mag_u is None or mag_v is None or mag_w is None):
        mag_all, _ = load_nifti(args.mag)
        mag_all = ensure_time_first(mag_all, args.time_axis)
        if mag_u is None:
            mag_u = mag_all
        if mag_v is None:
            mag_v = mag_all
        if mag_w is None:
            mag_w = mag_all

    shared_mag = mag_u if mag_u is not None else mag_v if mag_v is not None else mag_w
    if shared_mag is not None:
        if mag_u is None:
            mag_u = shared_mag
        if mag_v is None:
            mag_v = shared_mag
        if mag_w is None:
            mag_w = shared_mag

    if mag_u is None or mag_v is None or mag_w is None:
        raise ValueError("Magnitude is required. Provide --mag or one of --mag-u/--mag-v/--mag-w")
    if mag_u.shape != u.shape or mag_v.shape != u.shape or mag_w.shape != u.shape:
        raise ValueError("Magnitude and velocity shapes must match")

    venc = float(args.venc)
    if venc <= 0:
        venc = float(np.max(np.abs(np.stack([u, v, w], axis=0))))
        if venc <= 0:
            venc = 1.0

    if args.raw_phase_input:
        # Default behavior avoids double inversion when DICOM->NIfTI already
        # applied LPS->RAS sign correction to Vx/Vy.
        u = raw_to_velocity(
            u,
            venc=venc,
            raw_center=args.raw_center,
            raw_scale=args.raw_scale,
            invert_sign=args.legacy_invert_uv_sign_on_raw,
        )
        v = raw_to_velocity(
            v,
            venc=venc,
            raw_center=args.raw_center,
            raw_scale=args.raw_scale,
            invert_sign=args.legacy_invert_uv_sign_on_raw,
        )
        w = raw_to_velocity(w, venc=venc, raw_center=args.raw_center, raw_scale=args.raw_scale, invert_sign=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, predict_mag, model_variant, checkpoint_res_increase = load_model(
        args.model_path,
        args.res_increase,
        args.low_resblock,
        args.hi_resblock,
        device,
        predict_mag=args.predict_mag,
        model_variant=args.model_variant,
    )
    args.res_increase = int(checkpoint_res_increase)

    def predictor_fn(x):
        u_t, v_t, w_t, um_t, vm_t, wm_t = torch.chunk(x, 6, dim=1)
        return model(u_t, v_t, w_t, um_t, vm_t, wm_t)

    pred_frames = []
    roi_size = (args.patch_size, args.patch_size, args.patch_size)

    for t in range(u.shape[0]):
        mag_u_norm = normalize_magnitude(mag_u[t], mode=args.mag_norm_mode, mag_scale=args.mag_scale)
        mag_v_norm = normalize_magnitude(mag_v[t], mode=args.mag_norm_mode, mag_scale=args.mag_scale)
        mag_w_norm = normalize_magnitude(mag_w[t], mode=args.mag_norm_mode, mag_scale=args.mag_scale)
        lr_input = np.stack(
            [
                u[t] / venc,
                v[t] / venc,
                w[t] / venc,
                mag_u_norm,
                mag_v_norm,
                mag_w_norm,
            ],
            axis=0,
        ).astype(np.float32)
        if args.legacy_overlap_inference:
            pred_np = _legacy_overlap_predict(
                model=model,
                lr_input=lr_input,
                patch_size=args.patch_size,
                res_increase=args.res_increase,
                batch_size=args.sw_batch_size,
                device=device,
            )
        else:
            lr_tensor = torch.from_numpy(lr_input).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = sliding_window_inference(
                    inputs=lr_tensor,
                    roi_size=roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=predictor_fn,
                    overlap=args.overlap,
                    mode="gaussian",
                )
            pred_np = pred.squeeze(0).cpu().numpy()

        if pred_np.shape[0] not in (3, 4):
            raise ValueError(f"Unexpected model output channels: {pred_np.shape[0]} (expected 3 or 4)")

        pred_np[:3] = pred_np[:3] * venc
        if pred_np.shape[0] == 4 and str(args.mag_norm_mode).strip().lower() == "divisor":
            pred_np[3] = pred_np[3] * float(args.mag_scale)

        if args.round_small_values:
            threshold = venc / 2048.0
            pred_np[:3][np.abs(pred_np[:3]) < threshold] = 0
        pred_frames.append(pred_np)
        print(f"Processed frame {t + 1}/{u.shape[0]}")

    pred_stack = np.stack(pred_frames, axis=0)  # [T, C, X, Y, Z]
    u_out = np.moveaxis(pred_stack[:, 0], 0, -1)  # [X, Y, Z, T]
    v_out = np.moveaxis(pred_stack[:, 1], 0, -1)
    w_out = np.moveaxis(pred_stack[:, 2], 0, -1)
    mag_out = np.moveaxis(pred_stack[:, 3], 0, -1) if pred_stack.shape[1] == 4 else None

    out_affine = adjust_affine_for_upsample(u_img.affine, args.res_increase)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)

    u_path = f"{args.output_prefix}_u.nii.gz"
    v_path = f"{args.output_prefix}_v.nii.gz"
    w_path = f"{args.output_prefix}_w.nii.gz"
    uvw_path = f"{args.output_prefix}_uvw.nii.gz"
    mag_path = f"{args.output_prefix}_mag.nii.gz"

    nib.save(nib.Nifti1Image(u_out.astype(np.float32), out_affine), u_path)
    nib.save(nib.Nifti1Image(v_out.astype(np.float32), out_affine), v_path)
    nib.save(nib.Nifti1Image(w_out.astype(np.float32), out_affine), w_path)
    if mag_out is not None:
        nib.save(nib.Nifti1Image(mag_out.astype(np.float32), out_affine), mag_path)

    # Combined vector-field NIfTI. Shape: [X, Y, Z, T, 3]
    uvw_out = np.moveaxis(pred_stack[:, :3], 1, -1).astype(np.float32)
    uvw_out = np.moveaxis(uvw_out, 0, 3)
    nib.save(nib.Nifti1Image(uvw_out, out_affine), uvw_path)

    print("Prediction saved:")
    print(" ", u_path)
    print(" ", v_path)
    print(" ", w_path)
    if mag_out is not None:
        print(" ", mag_path)
    print(" ", uvw_path)
    print(f"Model variant={model_variant}, predict_mag={predict_mag}, output_channels={pred_stack.shape[1]}")


if __name__ == "__main__":
    main()
