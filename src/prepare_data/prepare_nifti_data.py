import argparse
import os
import numpy as np
import nibabel as nib
from NiftiData import NiftiData


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    spacing = np.asarray(img.header.get_zooms()[:3])
    return data, spacing


def ensure_time_first(data, time_axis):
    if data.ndim == 3:
        return data[np.newaxis, ...]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D NIfTI, got {data.ndim}D")
    if time_axis < 0:
        time_axis = data.ndim + time_axis
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)
    return data


def convert_raw_to_velocity_if_needed(data, venc, invert_sign=False, name="Component"):
    """
    Checks if data is raw DICOM intensity (approx 0..4096, centered at 2048).
    If so, converts to physical velocity (m/s).
    Optional sign inversion is for legacy inputs only.
    """
    # Heuristic:
    # - Unsigned raw phase: ~0..4096, centered near 2048
    # - Signed raw phase:   ~-4096..4096 (or ~-2048..2048), centered near 0
    # Velocity data is usually already around (-VENC..+VENC).
    mn, mx = np.min(data), np.max(data)

    is_unsigned_raw = (mn >= 0) and (mx > 1000) and (mx <= 8192)
    max_abs = max(abs(float(mn)), abs(float(mx)))
    centered_ratio = abs(float(mx + mn)) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500) and (mx > 500) and (max_abs <= 8192) and (centered_ratio < 0.25)

    if (is_unsigned_raw or is_signed_raw) and venc is not None:
        print(f"   [{name}] Raw intensity detected ({mn:.0f}..{mx:.0f}). Converting to m/s with VENC={venc}...")
        if is_unsigned_raw:
            # Formula for unsigned raw phase: (val - 2048) / 2048 * VENC
            data = (data.astype(np.float32) - 2048.0) / 2048.0 * venc
        else:
            # Signed raw phase can be either ~[-4096,4096] or ~[-2048,2048].
            scale = 4096.0 if max_abs > 3000 else 2048.0
            data = data.astype(np.float32) / scale * venc

        if invert_sign:
            print(f"   [{name}] applying legacy LPS->RAS sign inversion (-1).")
            data = -data

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--u', type=str, required=True, help='NIfTI for velocity U (3D or 4D)')
    parser.add_argument('--v', type=str, required=True, help='NIfTI for velocity V (3D or 4D)')
    parser.add_argument('--w', type=str, required=True, help='NIfTI for velocity W (3D or 4D)')
    parser.add_argument('--mag-u', type=str, default='', help='NIfTI for magnitude U (3D or 4D)')
    parser.add_argument('--mag-v', type=str, default='', help='NIfTI for magnitude V (3D or 4D)')
    parser.add_argument('--mag-w', type=str, default='', help='NIfTI for magnitude W (3D or 4D)')
    parser.add_argument('--mag', type=str, default='', help='Single magnitude NIfTI used for all components if mag-u/v/w not set')
    parser.add_argument('--mask', type=str, default='', help='Optional mask NIfTI (3D or 4D)')
    parser.add_argument('--mask-threshold', type=float, default=0.1, help='Threshold for creating mask from magnitude when mask is missing')
    parser.add_argument('--output-dir', type=str, default='../../data', help='Output directory where the dataset will be saved')
    parser.add_argument('--output-filename', type=str, default='mri_data.h5', help='Output filename (HDF5)')
    parser.add_argument('--venc', type=float, default=None, help='VENC value (m/s) applied to all components')
    parser.add_argument('--venc-u', type=float, default=None, help='VENC value (m/s) for U')
    parser.add_argument('--venc-v', type=float, default=None, help='VENC value (m/s) for V')
    parser.add_argument('--venc-w', type=float, default=None, help='VENC value (m/s) for W')
    parser.add_argument('--time-axis', type=int, default=-1, help='Time axis index for 4D NIfTI (default: last axis)')
    args = parser.parse_args()

    output_filepath = os.path.join(args.output_dir, args.output_filename)

    u_data, spacing = load_nifti(args.u)
    v_data, _ = load_nifti(args.v)
    w_data, _ = load_nifti(args.w)

    u_data = ensure_time_first(u_data, args.time_axis)
    v_data = ensure_time_first(v_data, args.time_axis)
    w_data = ensure_time_first(w_data, args.time_axis)

    if u_data.shape != v_data.shape or u_data.shape != w_data.shape:
        raise ValueError("U/V/W shapes do not match")

    mag_u_data = None
    mag_v_data = None
    mag_w_data = None

    if args.mag_u:
        mag_u_data, _ = load_nifti(args.mag_u)
        mag_u_data = ensure_time_first(mag_u_data, args.time_axis)
    if args.mag_v:
        mag_v_data, _ = load_nifti(args.mag_v)
        mag_v_data = ensure_time_first(mag_v_data, args.time_axis)
    if args.mag_w:
        mag_w_data, _ = load_nifti(args.mag_w)
        mag_w_data = ensure_time_first(mag_w_data, args.time_axis)

    if args.mag and (mag_u_data is None or mag_v_data is None or mag_w_data is None):
        mag_data, _ = load_nifti(args.mag)
        mag_data = ensure_time_first(mag_data, args.time_axis)
        if mag_u_data is None:
            mag_u_data = mag_data
        if mag_v_data is None:
            mag_v_data = mag_data
        if mag_w_data is None:
            mag_w_data = mag_data

    shared_mag_data = mag_u_data if mag_u_data is not None else mag_v_data if mag_v_data is not None else mag_w_data
    if shared_mag_data is not None:
        if mag_u_data is None:
            mag_u_data = shared_mag_data
        if mag_v_data is None:
            mag_v_data = shared_mag_data
        if mag_w_data is None:
            mag_w_data = shared_mag_data

    if mag_u_data is None or mag_v_data is None or mag_w_data is None:
        raise ValueError("Magnitude data missing. Provide --mag or one of --mag-u/--mag-v/--mag-w")

    if mag_u_data.shape != u_data.shape or mag_v_data.shape != u_data.shape or mag_w_data.shape != u_data.shape:
        raise ValueError("Magnitude shapes do not match velocity shapes")

    mask_data = None
    if args.mask:
        mask_data, _ = load_nifti(args.mask)
        mask_data = ensure_time_first(mask_data, args.time_axis)

    venc_u = args.venc_u if args.venc_u is not None else args.venc
    venc_v = args.venc_v if args.venc_v is not None else args.venc
    venc_w = args.venc_w if args.venc_w is not None else args.venc

    if venc_u is None:
        venc_u = float(np.max(np.abs(u_data)))
    if venc_v is None:
        venc_v = float(np.max(np.abs(v_data)))
    if venc_w is None:
        venc_w = float(np.max(np.abs(w_data)))

    # Convert RAW phase to m/s only. Sign correction should already be baked into NIfTI.
    u_data = convert_raw_to_velocity_if_needed(u_data, venc_u, invert_sign=False, name="U")
    v_data = convert_raw_to_velocity_if_needed(v_data, venc_v, invert_sign=False, name="V")
    w_data = convert_raw_to_velocity_if_needed(w_data, venc_w, invert_sign=False, name="W")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    time_frames = u_data.shape[0]
    for t in range(time_frames):
        print(f"\rProcessing {t+1}/{time_frames} (frame {t})          ", end="\r")

        if mask_data is not None:
            frame_mask = mask_data[t] if mask_data.shape[0] == time_frames else mask_data[0]
        else:
            mag_ref = mag_u_data[t]
            frame_mask = (mag_ref >= args.mask_threshold).astype(np.float32)

        nifti_data = NiftiData()
        nifti_data.set_velocity_components(
            u=u_data[t],
            v=v_data[t],
            w=w_data[t],
            mag_u=mag_u_data[t],
            mag_v=mag_v_data[t],
            mag_w=mag_w_data[t],
            venc_u=venc_u,
            venc_v=venc_v,
            venc_w=venc_w,
            spacing=spacing,
            mask=frame_mask
        )
        save_mask = (t == 0)
        nifti_data.save_dataset(output_filepath, t, save_mask=save_mask)

    print(f"\nDone! saved at {output_filepath}")


if __name__ == '__main__':
    main()
