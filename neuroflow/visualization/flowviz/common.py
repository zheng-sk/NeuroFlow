#!/usr/bin/env python3
"""Common data loading and geometry helpers for flow visualizations."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv


@dataclass
class FlowData:
    mag: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    affine: np.ndarray
    mask: np.ndarray | None
    paths: dict[str, Path]


_STREAMLINE_PARAMS = set(inspect.signature(pv.DataSetFilters.streamlines_from_source).parameters)


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-dir", type=Path, default=None, help="Case directory with Vx/Vy/Vz/mag files.")
    parser.add_argument("--vx-path", type=Path, default=None, help="Explicit path to Vx/U NIfTI.")
    parser.add_argument("--vy-path", type=Path, default=None, help="Explicit path to Vy/V NIfTI.")
    parser.add_argument("--vz-path", type=Path, default=None, help="Explicit path to Vz/W NIfTI.")
    parser.add_argument("--mag-path", type=Path, default=None, help="Explicit path to magnitude NIfTI.")
    parser.add_argument("--mask", type=Path, default=None, help="Optional vessel mask NIfTI (3D/4D).")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold applied to mask values.")
    parser.add_argument("--auto-convert-raw", action="store_true", help="Auto-detect RAW phase and convert to m/s.")
    parser.add_argument("--venc", type=float, default=None, help="Common VENC in m/s.")
    parser.add_argument("--venc-u", type=float, default=None, help="VENC for Vx/U in m/s.")
    parser.add_argument("--venc-v", type=float, default=None, help="VENC for Vy/V in m/s.")
    parser.add_argument("--venc-w", type=float, default=None, help="VENC for Vz/W in m/s.")


def add_common_preproc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame", type=int, default=0, help="Temporal frame index.")
    parser.add_argument(
        "--zero-velocity-below",
        type=float,
        default=0.0,
        help="Set velocity to zero where speed is below this threshold (m/s).",
    )


def add_streamline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed-mode", choices=["sphere", "speed", "mask"], default="speed")
    parser.add_argument("--seed-radius", type=float, default=10.0, help="Seed sphere radius in mm.")
    parser.add_argument("--n-seed-points", type=int, default=300, help="Approximate number of streamline seeds.")
    parser.add_argument(
        "--speed-seed-percentile",
        type=float,
        default=97.0,
        help="Percentile of speed used to pick speed-based seeds.",
    )
    parser.add_argument(
        "--seed-min-speed",
        type=float,
        default=0.0,
        help="Ignore points below this speed (m/s) when selecting seeds.",
    )
    parser.add_argument("--step", type=float, default=0.5, help="Initial integration step.")
    parser.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Maximum streamline integration length (compat replacement for max_time).",
    )
    parser.add_argument("--tube-radius", type=float, default=0.4, help="Streamline tube radius.")
    parser.add_argument(
        "--integration-direction",
        choices=["both", "forward", "backward"],
        default="both",
    )
    parser.add_argument("--stream-color-by-speed", action="store_true", help="Color streamlines by speed.")
    parser.add_argument("--speed-vmin", type=float, default=None, help="Min speed color scale.")
    parser.add_argument("--speed-vmax", type=float, default=None, help="Max speed color scale.")


def load_ras_canonical(path: Path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=dtype), img.affine.astype(dtype)


def ensure_time_last(data: np.ndarray) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim == 3:
        return data[..., np.newaxis]
    if data.ndim == 4:
        return data
    raise ValueError(f"Expected 3D/4D NIfTI after squeezing singleton dimensions, got {data.shape}")


def align_mask_time(mask_4d: np.ndarray, target_shape: tuple[int, int, int, int]) -> np.ndarray:
    if mask_4d.shape[:3] != target_shape[:3]:
        raise ValueError(
            "Mask spatial shape mismatch. "
            f"mask={mask_4d.shape}, target={target_shape}"
        )
    if mask_4d.shape[-1] == target_shape[-1]:
        return mask_4d
    if mask_4d.shape[-1] == 1:
        return np.repeat(mask_4d, target_shape[-1], axis=-1)
    raise ValueError(
        "Mask temporal size mismatch. "
        f"mask_t={mask_4d.shape[-1]}, data_t={target_shape[-1]}"
    )


def looks_like_raw_phase(data: np.ndarray) -> tuple[bool, bool]:
    mn = float(np.nanmin(data))
    mx = float(np.nanmax(data))
    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    max_abs = max(abs(mn), abs(mx))
    centered_ratio = abs(mx + mn) / (max_abs + 1e-6)
    is_signed_raw = (mn < -500.0) and (mx > 500.0) and (max_abs <= 8192.0) and (centered_ratio < 0.25)
    return is_unsigned_raw, is_signed_raw


def convert_raw_to_velocity_if_needed(
    values: np.ndarray,
    venc: float | None,
    invert_sign_if_raw: bool,
    label: str,
) -> tuple[np.ndarray, bool]:
    is_unsigned_raw, is_signed_raw = looks_like_raw_phase(values)
    is_raw = is_unsigned_raw or is_signed_raw
    out = values.astype(np.float32, copy=False)
    if not is_raw:
        return out, False

    if venc is None:
        raise ValueError(f"{label} looks like RAW phase, but VENC is missing.")

    if is_unsigned_raw:
        out = (out - 2048.0) / 2048.0 * float(venc)
    else:
        max_abs = max(abs(float(np.nanmin(out))), abs(float(np.nanmax(out))))
        scale = 4096.0 if max_abs > 3000.0 else 2048.0
        out = out / scale * float(venc)

    if invert_sign_if_raw:
        out = -out

    print(f"[{label}] RAW detected and converted to m/s (VENC={venc}).")
    return out.astype(np.float32, copy=False), True


def find_existing(case_dir: Path, candidates: list[str], role: str) -> Path:
    for name in candidates:
        path = case_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {role} in {case_dir}. Tried: {', '.join(candidates)}")


def resolve_case_paths(
    case_dir: Path,
    vx_path: Path | None = None,
    vy_path: Path | None = None,
    vz_path: Path | None = None,
    mag_path: Path | None = None,
) -> dict[str, Path]:
    if vx_path is not None and vy_path is not None and vz_path is not None and mag_path is not None:
        return {"vx": vx_path, "vy": vy_path, "vz": vz_path, "mag": mag_path}

    return {
        "vx": find_existing(case_dir, ["Vx.nii.gz", "phaseX_7T_in_3T.nii.gz"], "Vx/phaseX"),
        "vy": find_existing(case_dir, ["Vy.nii.gz", "phaseY_7T_in_3T.nii.gz"], "Vy/phaseY"),
        "vz": find_existing(case_dir, ["Vz.nii.gz", "phaseZ_7T_in_3T.nii.gz"], "Vz/phaseZ"),
        "mag": find_existing(case_dir, ["input_mag_raw.nii.gz", "mag_7T_in_3T.nii.gz"], "magnitude"),
    }


def load_flow_data_from_args(args: argparse.Namespace) -> FlowData:
    explicit = [args.vx_path, args.vy_path, args.vz_path, args.mag_path]
    if any(path is not None for path in explicit):
        if not all(path is not None for path in explicit):
            raise ValueError("If using explicit paths, provide all: --vx-path --vy-path --vz-path --mag-path")
        paths = resolve_case_paths(
            case_dir=Path("."),
            vx_path=args.vx_path,
            vy_path=args.vy_path,
            vz_path=args.vz_path,
            mag_path=args.mag_path,
        )
    else:
        if args.case_dir is None:
            raise ValueError("Provide either --case-dir or all explicit paths.")
        paths = resolve_case_paths(args.case_dir)

    print("Resolved inputs:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    mag, aff_mag = load_ras_canonical(paths["mag"])
    vx, aff_vx = load_ras_canonical(paths["vx"])
    vy, aff_vy = load_ras_canonical(paths["vy"])
    vz, aff_vz = load_ras_canonical(paths["vz"])

    mag = ensure_time_last(mag)
    vx = ensure_time_last(vx)
    vy = ensure_time_last(vy)
    vz = ensure_time_last(vz)

    if not (mag.shape == vx.shape == vy.shape == vz.shape):
        raise ValueError(
            "Shapes must match after loading. "
            f"mag={mag.shape}, vx={vx.shape}, vy={vy.shape}, vz={vz.shape}"
        )

    if not (
        np.allclose(aff_mag, aff_vx, atol=1e-3)
        and np.allclose(aff_mag, aff_vy, atol=1e-3)
        and np.allclose(aff_mag, aff_vz, atol=1e-3)
    ):
        print("Warning: affine matrices differ. Using magnitude affine for the grid.")
    affine = aff_mag

    if args.auto_convert_raw:
        venc_u = args.venc_u if args.venc_u is not None else args.venc
        venc_v = args.venc_v if args.venc_v is not None else args.venc
        venc_w = args.venc_w if args.venc_w is not None else args.venc
        vx, _ = convert_raw_to_velocity_if_needed(vx, venc_u, invert_sign_if_raw=False, label="Vx/U")
        vy, _ = convert_raw_to_velocity_if_needed(vy, venc_v, invert_sign_if_raw=False, label="Vy/V")
        vz, _ = convert_raw_to_velocity_if_needed(vz, venc_w, invert_sign_if_raw=False, label="Vz/W")
    else:
        print("RAW auto-conversion disabled. Velocities are used as loaded.")

    mask_4d = None
    if args.mask is not None:
        mask_data, aff_mask = load_ras_canonical(args.mask)
        mask_data = ensure_time_last(mask_data)
        mask_data = align_mask_time(mask_data, mag.shape)
        mask_4d = (mask_data >= float(args.mask_threshold)).astype(np.uint8)
        if not np.allclose(affine, aff_mask, atol=1e-3):
            print("Warning: mask affine differs from velocity/magnitude affine; verify registration.")
        print(f"Mask loaded: {args.mask} (threshold={args.mask_threshold})")

    return FlowData(mag=mag, vx=vx, vy=vy, vz=vz, affine=affine, mask=mask_4d, paths=paths)


def extract_frame(data_4d: np.ndarray, t: int) -> np.ndarray:
    if t < 0 or t >= data_4d.shape[-1]:
        raise IndexError(f"Frame t={t} out of range [0, {data_4d.shape[-1] - 1}]")
    return data_4d[..., t]


def build_structured_grid(
    mag_t: np.ndarray,
    vx_t: np.ndarray,
    vy_t: np.ndarray,
    vz_t: np.ndarray,
    affine: np.ndarray,
    mask_t: np.ndarray | None = None,
) -> pv.StructuredGrid:
    nx, ny, nz = mag_t.shape
    i, j, k = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )

    ones = np.ones_like(i)
    ijk1 = np.stack([i, j, k, ones], axis=-1)
    xyz1 = ijk1 @ affine.T
    x, y, z = xyz1[..., 0], xyz1[..., 1], xyz1[..., 2]

    grid = pv.StructuredGrid(x, y, z)
    vel = np.stack([vx_t, vy_t, vz_t], axis=-1)
    if mask_t is not None:
        vel = vel.copy()
        vel[mask_t <= 0] = 0.0
    speed = np.linalg.norm(vel, axis=-1)

    grid.point_data["mag"] = mag_t.ravel(order="F")
    grid.point_data["vel"] = vel.reshape(-1, 3, order="F")
    grid.point_data["speed"] = speed.ravel(order="F")
    if mask_t is not None:
        grid.point_data["mask"] = mask_t.astype(np.uint8, copy=False).ravel(order="F")
    return grid


def make_frame_grid(flow: FlowData, t: int, zero_velocity_below: float = 0.0) -> tuple[pv.StructuredGrid, np.ndarray | None]:
    mag_t = extract_frame(flow.mag, t)
    vx_t = extract_frame(flow.vx, t).copy()
    vy_t = extract_frame(flow.vy, t).copy()
    vz_t = extract_frame(flow.vz, t).copy()
    mask_t = extract_frame(flow.mask, t) if flow.mask is not None else None

    if zero_velocity_below > 0:
        speed_t = np.sqrt(vx_t * vx_t + vy_t * vy_t + vz_t * vz_t)
        low = speed_t < float(zero_velocity_below)
        vx_t[low] = 0.0
        vy_t[low] = 0.0
        vz_t[low] = 0.0

    if mask_t is not None:
        outside = mask_t <= 0
        vx_t[outside] = 0.0
        vy_t[outside] = 0.0
        vz_t[outside] = 0.0

    return build_structured_grid(mag_t, vx_t, vy_t, vz_t, flow.affine, mask_t=mask_t), mask_t


def make_seeds_from_grid(
    grid: pv.StructuredGrid,
    seed_mode: str,
    seed_radius: float,
    n_seed_points: int,
    speed_seed_percentile: float,
    seed_min_speed: float = 0.0,
) -> pv.PolyData:
    if seed_mode == "sphere":
        base_res = int(max(8, np.sqrt(max(1, int(n_seed_points)))))
        return pv.Sphere(
            center=grid.center,
            radius=seed_radius,
            theta_resolution=base_res,
            phi_resolution=base_res,
        ).triangulate()

    if seed_mode == "speed":
        speed = grid.point_data["speed"]
        in_mask = (grid.point_data["mask"] > 0.5) if "mask" in grid.point_data else np.ones_like(speed, dtype=bool)
        min_speed = float(max(seed_min_speed, 0.0))
        positive = speed[(speed > min_speed) & in_mask]
        if positive.size == 0:
            raise ValueError("No speed values above seed-min-speed inside mask. Cannot create speed-based seeds.")
        threshold = np.percentile(positive, float(speed_seed_percentile))
        threshold = max(float(threshold), min_speed)
        candidates = np.flatnonzero((speed >= threshold) & in_mask)
        if candidates.size == 0:
            candidates = np.flatnonzero((speed > min_speed) & in_mask)
        if candidates.size == 0:
            candidates = np.flatnonzero(in_mask)
        count = min(int(n_seed_points), int(candidates.size))
        rng = np.random.default_rng(0)
        chosen = rng.choice(candidates, size=count, replace=False)
        return pv.PolyData(grid.points[chosen])

    if seed_mode == "mask":
        speed = grid.point_data["speed"]
        in_mask = (grid.point_data["mask"] > 0.5) if "mask" in grid.point_data else np.ones_like(speed, dtype=bool)
        min_speed = float(max(seed_min_speed, 0.0))
        candidates = np.flatnonzero((speed > min_speed) & in_mask)
        if candidates.size == 0:
            candidates = np.flatnonzero(in_mask)
        if candidates.size == 0:
            raise ValueError("Mask-based seeding failed: no candidate points found.")
        count = min(int(n_seed_points), int(candidates.size))
        rng = np.random.default_rng(0)
        chosen = rng.choice(candidates, size=count, replace=False)
        return pv.PolyData(grid.points[chosen])

    raise ValueError("seed_mode must be 'sphere', 'speed' or 'mask'.")


def compute_speed_clim(
    speed_values: np.ndarray,
    provided: tuple[float, float] | None = None,
    q_low: float = 5.0,
    q_high: float = 99.0,
) -> tuple[float, float]:
    if provided is not None:
        return provided
    positive = speed_values[np.isfinite(speed_values) & (speed_values > 0)]
    if positive.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(positive, q_low))
    hi = float(np.percentile(positive, q_high))
    if hi <= lo:
        hi = lo + 1e-6
    return (lo, hi)


def add_context_geometry(
    plotter: pv.Plotter,
    grid: pv.StructuredGrid,
    slice_normal: str,
    show_slice_context: bool,
    slice_opacity: float,
    vessel_opacity: float,
    vessel_color: str,
) -> None:
    if "mask" in grid.point_data:
        try:
            vessel = grid.contour(isosurfaces=[0.5], scalars="mask")
            if vessel.n_points > 0:
                plotter.add_mesh(vessel, color=vessel_color, opacity=vessel_opacity, smooth_shading=True)
        except Exception:
            pass

    if show_slice_context:
        slc = grid.slice(normal=slice_normal)
        plotter.add_mesh(slc, scalars="mag", opacity=slice_opacity, cmap="gray")


def streamlines_from_source_compat(
    grid: pv.StructuredGrid,
    seeds: pv.PolyData,
    vectors: str,
    step: float,
    max_length: float,
    integration_direction: str,
) -> pv.PolyData:
    kwargs = {
        "vectors": vectors,
        "initial_step_length": step,
        "integration_direction": integration_direction,
    }
    if "max_length" in _STREAMLINE_PARAMS:
        kwargs["max_length"] = max_length
    elif "max_time" in _STREAMLINE_PARAMS:
        kwargs["max_time"] = max_length
    return grid.streamlines_from_source(seeds, **kwargs)


def _bounds_diag(bounds: tuple[float, float, float, float, float, float]) -> float:
    return float(
        np.linalg.norm(
            [
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ]
        )
    )


def merge_bounds(bounds_list: list[tuple[float, float, float, float, float, float]]) -> tuple[float, float, float, float, float, float]:
    if not bounds_list:
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    xmins = [b[0] for b in bounds_list]
    xmaxs = [b[1] for b in bounds_list]
    ymins = [b[2] for b in bounds_list]
    ymaxs = [b[3] for b in bounds_list]
    zmins = [b[4] for b in bounds_list]
    zmaxs = [b[5] for b in bounds_list]
    return (min(xmins), max(xmaxs), min(ymins), max(ymaxs), min(zmins), max(zmaxs))


def fit_camera_to_bounds(
    plotter: pv.Plotter,
    bounds: tuple[float, float, float, float, float, float],
    azimuth_deg: float = 35.0,
    elevation_deg: float = 18.0,
    distance_scale: float = 1.6,
    zoom: float = 1.0,
) -> None:
    center = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=np.float64,
    )
    diag = max(_bounds_diag(bounds), 1e-3)
    eye = center + np.array([1.2 * diag * distance_scale, -0.8 * diag * distance_scale, 0.6 * diag * distance_scale])
    plotter.camera_position = [tuple(eye.tolist()), tuple(center.tolist()), (0.0, 0.0, 1.0)]
    plotter.camera.Azimuth(float(azimuth_deg))
    plotter.camera.Elevation(float(elevation_deg))
    plotter.camera.Zoom(float(zoom))


def build_direction_arrows(
    streams: pv.PolyData,
    stride: int = 30,
    factor: float = 0.7,
) -> pv.PolyData | None:
    if streams.n_points == 0 or "vel" not in streams.point_data:
        return None
    ids = np.arange(0, streams.n_points, max(1, int(stride)), dtype=np.int64)
    if ids.size == 0:
        return None
    pts = streams.extract_points(ids, include_cells=False)
    if pts.n_points == 0:
        return None
    arrows = pts.glyph(orient="vel", scale=False, factor=float(factor))
    return arrows if arrows.n_points > 0 else None
