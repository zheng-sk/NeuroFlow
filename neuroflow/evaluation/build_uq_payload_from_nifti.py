import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch


def _load_nifti_time_first(path: str, time_axis: int) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim == 4:
        if time_axis < 0:
            time_axis = data.ndim + time_axis
        if time_axis < 0 or time_axis >= data.ndim:
            raise ValueError(f"Invalid time_axis={time_axis} for data shape {data.shape} in {path}")
        if time_axis != 0:
            data = np.moveaxis(data, time_axis, 0)
    else:
        raise ValueError(f"Expected 3D/4D NIfTI, got shape {data.shape} for {path}")

    return data.astype(np.float32), img


def _align_time(arr_txyz: np.ndarray, target_t: int, name: str) -> np.ndarray:
    if arr_txyz.shape[0] == target_t:
        return arr_txyz
    if arr_txyz.shape[0] == 1:
        return np.repeat(arr_txyz, target_t, axis=0)
    raise ValueError(f"{name} has T={arr_txyz.shape[0]} but target T={target_t}")


def _resample_txyz(arr_txyz: np.ndarray, out_shape_xyz: Sequence[int], mode: str = "trilinear") -> np.ndarray:
    x = torch.from_numpy(arr_txyz[:, None, ...].astype(np.float32))
    if mode == "nearest":
        y = torch.nn.functional.interpolate(x, size=tuple(int(v) for v in out_shape_xyz), mode=mode)
    else:
        y = torch.nn.functional.interpolate(x, size=tuple(int(v) for v in out_shape_xyz), mode=mode, align_corners=False)
    return y[:, 0].numpy().astype(np.float32)


def _ensure_shape(arr_txyz: np.ndarray, shape_xyz: Sequence[int], name: str) -> None:
    if tuple(arr_txyz.shape[1:]) != tuple(int(v) for v in shape_xyz):
        raise ValueError(f"{name} has shape {arr_txyz.shape[1:]}, expected {tuple(shape_xyz)}")


def _derive_mag(u_txyz: np.ndarray, v_txyz: np.ndarray, w_txyz: np.ndarray) -> np.ndarray:
    return np.sqrt(u_txyz**2 + v_txyz**2 + w_txyz**2).astype(np.float32)


def _normalize_mag_txyz(mag_txyz: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mag_txyz, dtype=np.float32)
    for t in range(mag_txyz.shape[0]):
        a = mag_txyz[t].astype(np.float32)
        mn = float(np.nanmin(a))
        mx = float(np.nanmax(a))
        if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-8:
            out[t] = np.zeros_like(a, dtype=np.float32)
        else:
            out[t] = np.clip((a - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)
    return out


def _load_optional_mag(path: str, time_axis: int, target_t: int, expected_shape_xyz: Sequence[int], name: str) -> Optional[np.ndarray]:
    if not path:
        return None
    arr, _ = _load_nifti_time_first(path, time_axis)
    arr = _align_time(arr, target_t, name)
    _ensure_shape(arr, expected_shape_xyz, name)
    return arr.astype(np.float32)


def _load_or_derive_mask(mask_path: str, time_axis: int, target_t: int, shape_xyz: Sequence[int], threshold: float) -> np.ndarray:
    if not mask_path:
        return np.ones((target_t, *shape_xyz), dtype=np.float32)

    mask, _ = _load_nifti_time_first(mask_path, time_axis)
    mask = _align_time(mask, target_t, "mask")
    _ensure_shape(mask, shape_xyz, "mask")
    return (mask >= float(threshold)).astype(np.float32)


def _compute_venc_per_frame(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    pred_w: np.ndarray,
    lr_u: np.ndarray,
    lr_v: np.ndarray,
    lr_w: np.ndarray,
    hr_u: np.ndarray,
    hr_v: np.ndarray,
    hr_w: np.ndarray,
    venc_override: float,
) -> np.ndarray:
    t_count = pred_u.shape[0]
    if np.isfinite(float(venc_override)) and float(venc_override) > 0.0:
        return np.full((t_count,), float(venc_override), dtype=np.float32)

    venc = np.zeros((t_count,), dtype=np.float32)
    for t in range(t_count):
        vmax = 0.0
        for arr in (pred_u[t], pred_v[t], pred_w[t], lr_u[t], lr_v[t], lr_w[t], hr_u[t], hr_v[t], hr_w[t]):
            a = float(np.max(np.abs(arr)))
            if np.isfinite(a):
                vmax = max(vmax, a)
        if vmax <= 1e-8:
            vmax = 1.0
        venc[t] = float(vmax)
    return venc


def _write_payload_npz(path: Path, payload: Dict[str, Any]) -> None:
    np.savez_compressed(
        str(path),
        frame_indices=payload["frame_indices"],
        lr_norm=payload["lr_norm"],
        gt_norm=payload["gt_norm"],
        pred_norm=payload["pred_norm"],
        mask=payload["mask"],
        venc=payload["venc"],
        lr_affine=payload["lr_affine"],
        hr_affine=payload["hr_affine"],
        lr_spacing=payload["lr_spacing"],
        hr_spacing=payload["hr_spacing"],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build analysis_payload.npz from explicit NIfTI files (prediction + 3T LR + 7T/reference HR), "
            "so you can run the same generate_sr_uq_report.py metrics pipeline without rerunning inference."
        )
    )

    # Prediction (required)
    p.add_argument("--pred-u", required=True)
    p.add_argument("--pred-v", required=True)
    p.add_argument("--pred-w", required=True)
    p.add_argument("--pred-mag", default="", help="Optional predicted magnitude NIfTI.")

    # Baseline LR (required)
    p.add_argument("--lr-u", required=True)
    p.add_argument("--lr-v", required=True)
    p.add_argument("--lr-w", required=True)
    p.add_argument("--lr-mag", default="", help="Optional single LR magnitude NIfTI (shared across channels).")
    p.add_argument("--lr-mag-u", default="", help="Optional LR magnitude U NIfTI.")
    p.add_argument("--lr-mag-v", default="", help="Optional LR magnitude V NIfTI.")
    p.add_argument("--lr-mag-w", default="", help="Optional LR magnitude W NIfTI.")

    # Reference HR (required)
    p.add_argument("--hr-u", required=True)
    p.add_argument("--hr-v", required=True)
    p.add_argument("--hr-w", required=True)
    p.add_argument("--hr-mag", default="", help="Optional HR/reference magnitude NIfTI.")

    p.add_argument("--mask", default="", help="Optional binary mask NIfTI (3D/4D in HR space).")

    p.add_argument("--time-axis", type=int, default=-1)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--venc", type=float, default=float("nan"), help="Optional fixed venc for normalization.")
    p.add_argument("--resample-pred-to-hr", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-report", action="store_true", help="Run generate_sr_uq_report.py after payload creation.")

    # Optional passthrough to report
    p.add_argument("--flow-axis", type=str, default="auto", choices=["auto", "0", "1", "2"])
    p.add_argument("--selected-frame", type=int, default=0)
    p.add_argument("--max-display-slices", type=int, default=8)
    p.add_argument("--panel-cols", type=int, default=4)
    p.add_argument("--hist-bins", type=int, default=120)
    p.add_argument("--lr-mag-channel", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--mask-min-slice-voxels", type=int, default=25)
    p.add_argument("--q-ref", type=float, default=float("nan"))
    p.add_argument("--cca-range", type=str, default="")
    p.add_argument("--mu-pa-s", type=float, default=0.0035)
    p.add_argument("--max-wall-points", type=int, default=30000)
    p.add_argument("--roi-bbox", type=int, nargs=6, default=None)
    p.add_argument("--roi-json", default="")
    p.add_argument("--baseline-label", default="3T")
    p.add_argument("--ref-label", default="7T")
    p.add_argument("--sr-label", default="3T SR")
    p.add_argument("--report-title", default="4D Flow SR Uncertainty Quantification Report")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HR/reference first (defines target grid)
    hr_u, hr_u_img = _load_nifti_time_first(args.hr_u, int(args.time_axis))
    hr_v, hr_v_img = _load_nifti_time_first(args.hr_v, int(args.time_axis))
    hr_w, hr_w_img = _load_nifti_time_first(args.hr_w, int(args.time_axis))

    if not (hr_u.shape == hr_v.shape == hr_w.shape):
        raise ValueError(f"HR u/v/w shapes mismatch: {hr_u.shape}, {hr_v.shape}, {hr_w.shape}")
    if not (np.allclose(hr_u_img.affine, hr_v_img.affine) and np.allclose(hr_u_img.affine, hr_w_img.affine)):
        raise ValueError("HR u/v/w affines mismatch.")

    t_count = int(hr_u.shape[0])
    hr_shape_xyz = tuple(int(v) for v in hr_u.shape[1:])

    # Load prediction
    pred_u, pred_u_img = _load_nifti_time_first(args.pred_u, int(args.time_axis))
    pred_v, pred_v_img = _load_nifti_time_first(args.pred_v, int(args.time_axis))
    pred_w, pred_w_img = _load_nifti_time_first(args.pred_w, int(args.time_axis))

    pred_u = _align_time(pred_u, t_count, "pred_u")
    pred_v = _align_time(pred_v, t_count, "pred_v")
    pred_w = _align_time(pred_w, t_count, "pred_w")

    if not (pred_u.shape == pred_v.shape == pred_w.shape):
        raise ValueError(f"Prediction u/v/w shapes mismatch: {pred_u.shape}, {pred_v.shape}, {pred_w.shape}")

    if tuple(pred_u.shape[1:]) != hr_shape_xyz:
        if bool(args.resample_pred_to_hr):
            pred_u = _resample_txyz(pred_u, hr_shape_xyz, mode="trilinear")
            pred_v = _resample_txyz(pred_v, hr_shape_xyz, mode="trilinear")
            pred_w = _resample_txyz(pred_w, hr_shape_xyz, mode="trilinear")
        else:
            raise ValueError(
                f"Prediction shape {pred_u.shape[1:]} != HR shape {hr_shape_xyz}. "
                f"Use --resample-pred-to-hr."
            )

    # Load baseline LR
    lr_u, lr_u_img = _load_nifti_time_first(args.lr_u, int(args.time_axis))
    lr_v, lr_v_img = _load_nifti_time_first(args.lr_v, int(args.time_axis))
    lr_w, lr_w_img = _load_nifti_time_first(args.lr_w, int(args.time_axis))

    lr_u = _align_time(lr_u, t_count, "lr_u")
    lr_v = _align_time(lr_v, t_count, "lr_v")
    lr_w = _align_time(lr_w, t_count, "lr_w")

    if not (lr_u.shape == lr_v.shape == lr_w.shape):
        raise ValueError(f"LR u/v/w shapes mismatch: {lr_u.shape}, {lr_v.shape}, {lr_w.shape}")

    lr_shape_xyz = tuple(int(v) for v in lr_u.shape[1:])

    # Optional mask in HR space
    mask_txyz = _load_or_derive_mask(
        mask_path=str(args.mask),
        time_axis=int(args.time_axis),
        target_t=t_count,
        shape_xyz=hr_shape_xyz,
        threshold=float(args.mask_threshold),
    )

    # Magnitudes (optional; derive if missing)
    pred_mag = _load_optional_mag(str(args.pred_mag), int(args.time_axis), t_count, hr_shape_xyz, "pred_mag")
    if pred_mag is None:
        pred_mag = _derive_mag(pred_u, pred_v, pred_w)

    hr_mag = _load_optional_mag(str(args.hr_mag), int(args.time_axis), t_count, hr_shape_xyz, "hr_mag")
    if hr_mag is None:
        hr_mag = _derive_mag(hr_u, hr_v, hr_w)

    # LR mag channels
    lr_mag_u = _load_optional_mag(str(args.lr_mag_u), int(args.time_axis), t_count, lr_shape_xyz, "lr_mag_u")
    lr_mag_v = _load_optional_mag(str(args.lr_mag_v), int(args.time_axis), t_count, lr_shape_xyz, "lr_mag_v")
    lr_mag_w = _load_optional_mag(str(args.lr_mag_w), int(args.time_axis), t_count, lr_shape_xyz, "lr_mag_w")

    if any(m is not None for m in (lr_mag_u, lr_mag_v, lr_mag_w)):
        if not all(m is not None for m in (lr_mag_u, lr_mag_v, lr_mag_w)):
            raise ValueError("If using per-channel LR mag, provide all: --lr-mag-u --lr-mag-v --lr-mag-w")
        lr_mag_ch = np.stack([lr_mag_u, lr_mag_v, lr_mag_w], axis=1).astype(np.float32)
    else:
        lr_mag_single = _load_optional_mag(str(args.lr_mag), int(args.time_axis), t_count, lr_shape_xyz, "lr_mag")
        if lr_mag_single is None:
            lr_mag_single = _derive_mag(lr_u, lr_v, lr_w)
        lr_mag_ch = np.stack([lr_mag_single, lr_mag_single, lr_mag_single], axis=1).astype(np.float32)

    # Normalize
    venc = _compute_venc_per_frame(
        pred_u=pred_u,
        pred_v=pred_v,
        pred_w=pred_w,
        lr_u=lr_u,
        lr_v=lr_v,
        lr_w=lr_w,
        hr_u=hr_u,
        hr_v=hr_v,
        hr_w=hr_w,
        venc_override=float(args.venc),
    )

    pred_norm = np.zeros((t_count, 4, *hr_shape_xyz), dtype=np.float32)
    gt_norm = np.zeros((t_count, 4, *hr_shape_xyz), dtype=np.float32)
    lr_norm = np.zeros((t_count, 6, *lr_shape_xyz), dtype=np.float32)

    pred_mag_norm = _normalize_mag_txyz(pred_mag)
    hr_mag_norm = _normalize_mag_txyz(hr_mag)

    lr_mag_norm_ch = np.zeros_like(lr_mag_ch, dtype=np.float32)
    for c in range(3):
        lr_mag_norm_ch[:, c] = _normalize_mag_txyz(lr_mag_ch[:, c])

    for t in range(t_count):
        v = float(venc[t]) if float(venc[t]) > 1e-8 else 1.0

        pred_norm[t, 0] = pred_u[t] / v
        pred_norm[t, 1] = pred_v[t] / v
        pred_norm[t, 2] = pred_w[t] / v
        pred_norm[t, 3] = pred_mag_norm[t]

        gt_norm[t, 0] = hr_u[t] / v
        gt_norm[t, 1] = hr_v[t] / v
        gt_norm[t, 2] = hr_w[t] / v
        gt_norm[t, 3] = hr_mag_norm[t]

        lr_norm[t, 0] = lr_u[t] / v
        lr_norm[t, 1] = lr_v[t] / v
        lr_norm[t, 2] = lr_w[t] / v
        lr_norm[t, 3] = lr_mag_norm_ch[t, 0]
        lr_norm[t, 4] = lr_mag_norm_ch[t, 1]
        lr_norm[t, 5] = lr_mag_norm_ch[t, 2]

    payload = {
        "frame_indices": np.arange(t_count, dtype=np.int32),
        "lr_norm": lr_norm,
        "gt_norm": gt_norm,
        "pred_norm": pred_norm,
        "mask": mask_txyz.astype(np.float32),
        "venc": venc.astype(np.float32),
        "lr_affine": np.asarray(lr_u_img.affine, dtype=np.float32),
        "hr_affine": np.asarray(hr_u_img.affine, dtype=np.float32),
        "lr_spacing": np.asarray(lr_u_img.header.get_zooms()[:3], dtype=np.float32),
        "hr_spacing": np.asarray(hr_u_img.header.get_zooms()[:3], dtype=np.float32),
    }

    payload_path = out_dir / "analysis_payload.npz"
    _write_payload_npz(payload_path, payload)

    metadata = {
        "source": "build_uq_payload_from_nifti",
        "frame_indices": [int(i) for i in range(t_count)],
        "frame_selection_mode": "all",
        "predict_mag": bool(args.pred_mag),
        "derived_pred_mag": bool(not args.pred_mag),
        "derived_hr_mag": bool(not args.hr_mag),
        "derived_lr_mag": bool(not any([args.lr_mag, args.lr_mag_u, args.lr_mag_v, args.lr_mag_w])),
        "resample_pred_to_hr": bool(args.resample_pred_to_hr),
        "venc_mode": "fixed" if (np.isfinite(float(args.venc)) and float(args.venc) > 0.0) else "auto_per_frame",
        "case_paths": {
            "pred_u": str(Path(args.pred_u).resolve()),
            "pred_v": str(Path(args.pred_v).resolve()),
            "pred_w": str(Path(args.pred_w).resolve()),
            "pred_mag": str(Path(args.pred_mag).resolve()) if args.pred_mag else "",
            "lr_u": str(Path(args.lr_u).resolve()),
            "lr_v": str(Path(args.lr_v).resolve()),
            "lr_w": str(Path(args.lr_w).resolve()),
            "lr_mag": str(Path(args.lr_mag).resolve()) if args.lr_mag else "",
            "lr_mag_u": str(Path(args.lr_mag_u).resolve()) if args.lr_mag_u else "",
            "lr_mag_v": str(Path(args.lr_mag_v).resolve()) if args.lr_mag_v else "",
            "lr_mag_w": str(Path(args.lr_mag_w).resolve()) if args.lr_mag_w else "",
            "hr_u": str(Path(args.hr_u).resolve()),
            "hr_v": str(Path(args.hr_v).resolve()),
            "hr_w": str(Path(args.hr_w).resolve()),
            "hr_mag": str(Path(args.hr_mag).resolve()) if args.hr_mag else "",
            "mask": str(Path(args.mask).resolve()) if args.mask else "",
        },
        "payload_path": str(payload_path.resolve()),
    }

    meta_path = out_dir / "inference_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Payload built:")
    print(f"- Payload:  {payload_path}")
    print(f"- Metadata: {meta_path}")

    if args.run_report:
        report_script = (Path(__file__).resolve().parent / "generate_sr_uq_report.py").resolve()
        cmd = [
            sys.executable,
            str(report_script),
            "--payload-npz",
            str(payload_path),
            "--metadata-json",
            str(meta_path),
            "--out-dir",
            str(out_dir),
            "--flow-axis",
            str(args.flow_axis),
            "--selected-frame",
            str(args.selected_frame),
            "--max-display-slices",
            str(args.max_display_slices),
            "--panel-cols",
            str(args.panel_cols),
            "--hist-bins",
            str(args.hist_bins),
            "--lr-mag-channel",
            str(args.lr_mag_channel),
            "--mask-min-slice-voxels",
            str(args.mask_min_slice_voxels),
            "--mu-pa-s",
            str(args.mu_pa_s),
            "--max-wall-points",
            str(args.max_wall_points),
            "--baseline-label",
            str(args.baseline_label),
            "--ref-label",
            str(args.ref_label),
            "--sr-label",
            str(args.sr_label),
            "--report-title",
            str(args.report_title),
        ]
        if args.cca_range:
            cmd.extend(["--cca-range", str(args.cca_range)])
        if np.isfinite(float(args.q_ref)):
            cmd.extend(["--q-ref", str(float(args.q_ref))])
        if args.roi_bbox and len(args.roi_bbox) == 6:
            cmd.append("--roi-bbox")
            cmd.extend([str(v) for v in args.roi_bbox])
        if args.roi_json:
            cmd.extend(["--roi-json", str(args.roi_json)])

        print("\nRunning report:")
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
