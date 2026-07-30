#!/usr/bin/env python3
"""Physiological direction QA using net flux curves across user-defined vessel planes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuroflow.visualization.flowviz.common import add_common_preproc_args, add_input_args, extract_frame, load_flow_data_from_args


@dataclass
class PlaneSpec:
    name: str
    center: np.ndarray
    normal: np.ndarray
    radius_mm: float
    resolution: int
    role: str  # in | out | none


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame net flux through anatomical planes to validate physiological flow direction "
            "and consistency with expected inlet/outlet behavior."
        )
    )
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument(
        "--plane",
        action="append",
        required=True,
        help=(
            "Plane spec format: "
            "name:cx,cy,cz:nx,ny,nz[:radius_mm[:resolution[:role]]]. "
            "role in {in,out,none}."
        ),
    )
    parser.add_argument("--default-radius-mm", type=float, default=3.0, help="Default plane radius in mm.")
    parser.add_argument("--default-resolution", type=int, default=60, help="Default in-plane sampling resolution.")
    parser.add_argument("--square-plane", action="store_true", help="Use square area instead of circular disk.")

    parser.add_argument("--all-frames", action="store_true", help="Use all time frames.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame stride when --all-frames is not set.")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap number of frames (0 means no cap).")
    parser.add_argument("--dt-seconds", type=float, default=None, help="Frame duration in seconds for time axis.")
    parser.add_argument(
        "--mask-required",
        action="store_true",
        help="Fail if --mask is not provided. Recommended for CoW physiological QA.",
    )
    parser.add_argument(
        "--auto-flip-negative-role",
        action="store_true",
        help="For mass-balance summary, flip sign of planes with role in/out and negative mean flux.",
    )

    parser.add_argument("--csv-out", type=Path, required=True, help="Output CSV path with per-frame flux metrics.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional text summary output path.")
    parser.add_argument("--plot-png", type=Path, default=None, help="Optional flux curve plot (PNG).")
    return parser.parse_args()


def _parse_vec3(text: str, label: str) -> np.ndarray:
    items = [token.strip() for token in text.split(",")]
    if len(items) != 3:
        raise ValueError(f"{label} must have 3 comma-separated values. Got: {text}")
    return np.array([float(items[0]), float(items[1]), float(items[2])], dtype=np.float64)


def _parse_plane_spec(spec: str, default_radius: float, default_res: int) -> PlaneSpec:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --plane '{spec}'. Expected at least: name:cx,cy,cz:nx,ny,nz[:radius[:resolution[:role]]]"
        )

    name = parts[0]
    center = _parse_vec3(parts[1], "center")
    normal = _parse_vec3(parts[2], "normal")
    nn = float(np.linalg.norm(normal))
    if nn < 1e-8:
        raise ValueError(f"Plane '{name}' has near-zero normal.")
    normal = normal / nn

    radius = float(parts[3]) if len(parts) >= 4 and parts[3] else float(default_radius)
    resolution = int(parts[4]) if len(parts) >= 5 and parts[4] else int(default_res)
    role = parts[5].lower() if len(parts) >= 6 and parts[5] else "none"
    if role not in {"in", "out", "none"}:
        raise ValueError(f"Plane '{name}' has invalid role '{role}'. Use in/out/none.")
    if radius <= 0:
        raise ValueError(f"Plane '{name}' radius must be > 0.")
    if resolution < 4:
        raise ValueError(f"Plane '{name}' resolution must be >= 4.")
    return PlaneSpec(name=name, center=center, normal=normal, radius_mm=radius, resolution=resolution, role=role)


def _frame_indices(nt: int, all_frames: bool, frame: int, frame_step: int, max_frames: int) -> list[int]:
    if nt <= 0:
        return []
    if all_frames:
        ids = list(range(nt))
    else:
        start = int(frame)
        if start < 0 or start >= nt:
            raise ValueError(f"--frame {start} out of range [0, {nt - 1}]")
        ids = list(range(start, nt, max(1, int(frame_step))))
    if max_frames > 0:
        ids = ids[: int(max_frames)]
    return ids


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal.astype(np.float64)
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, ref)
    un = np.linalg.norm(u)
    if un < 1e-8:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(normal, ref)
        un = np.linalg.norm(u)
    u /= max(un, 1e-8)
    v = np.cross(normal, u)
    v /= max(np.linalg.norm(v), 1e-8)
    return u, v


def _plane_samples(plane: PlaneSpec, square_plane: bool) -> tuple[np.ndarray, np.ndarray]:
    u, v = _plane_basis(plane.normal)
    res = int(plane.resolution)
    radius = float(plane.radius_mm)
    step = (2.0 * radius) / float(res)
    coords = np.linspace(-radius + 0.5 * step, radius - 0.5 * step, num=res, dtype=np.float64)
    uu, vv = np.meshgrid(coords, coords, indexing="ij")

    if square_plane:
        keep = np.ones_like(uu, dtype=bool)
    else:
        keep = (uu * uu + vv * vv) <= (radius * radius)

    uu_k = uu[keep]
    vv_k = vv[keep]
    points = plane.center[None, :] + uu_k[:, None] * u[None, :] + vv_k[:, None] * v[None, :]
    dA_mm2 = np.full((points.shape[0],), step * step, dtype=np.float64)
    return points.astype(np.float32), dA_mm2.astype(np.float32)


def _world_to_ijk(points_xyz: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    ijk1 = homo @ inv_affine.T
    return ijk1[:, :3]


def _trilinear(volume: np.ndarray, ijk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = volume.shape
    x = ijk[:, 0]
    y = ijk[:, 1]
    z = ijk[:, 2]
    inside = (x >= 0) & (x < nx - 1) & (y >= 0) & (y < ny - 1) & (z >= 0) & (z < nz - 1)

    out = np.zeros((ijk.shape[0],), dtype=np.float32)
    if not np.any(inside):
        return out, inside

    xi = x[inside]
    yi = y[inside]
    zi = z[inside]
    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    z0 = np.floor(zi).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    xd = (xi - x0).astype(np.float32)
    yd = (yi - y0).astype(np.float32)
    zd = (zi - z0).astype(np.float32)

    c000 = volume[x0, y0, z0]
    c001 = volume[x0, y0, z1]
    c010 = volume[x0, y1, z0]
    c011 = volume[x0, y1, z1]
    c100 = volume[x1, y0, z0]
    c101 = volume[x1, y0, z1]
    c110 = volume[x1, y1, z0]
    c111 = volume[x1, y1, z1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd

    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    vals = c0 * (1.0 - zd) + c1 * zd
    out[inside] = vals.astype(np.float32)
    return out, inside


def _nanmean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _nansign_fraction(values: np.ndarray, positive: bool) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    v = values[finite]
    return float(np.mean(v > 0.0)) if positive else float(np.mean(v < 0.0))


def _save_plot_png(out_path: Path, times: np.ndarray, flux_by_plane: dict[str, np.ndarray], role_by_plane: dict[str, str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Plot skipped (matplotlib unavailable): {exc}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    for name, flux in flux_by_plane.items():
        role = role_by_plane[name]
        suffix = f" [{role}]" if role != "none" else ""
        ax.plot(times, flux, linewidth=1.8, label=f"{name}{suffix}")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flux (ml/s)")
    ax.set_title("Per-plane net flux curves")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Flux plot written to: {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    if args.mask_required and flow.mask is None:
        raise ValueError("--mask-required was set, but no --mask was provided.")

    plane_specs = [
        _parse_plane_spec(spec, default_radius=args.default_radius_mm, default_res=args.default_resolution)
        for spec in args.plane
    ]
    if len({p.name for p in plane_specs}) != len(plane_specs):
        raise ValueError("Plane names must be unique.")

    nt = int(flow.vx.shape[-1])
    frames = _frame_indices(
        nt=nt,
        all_frames=bool(args.all_frames),
        frame=int(args.frame),
        frame_step=int(args.frame_step),
        max_frames=int(args.max_frames),
    )
    if not frames:
        raise ValueError("No frames selected.")

    if args.dt_seconds is not None:
        dt = float(args.dt_seconds)
    else:
        # If temporal spacing is unknown, use 1.0 s only for plotting axis.
        dt = 1.0
    times = np.array(frames, dtype=np.float64) * dt

    inv_affine = np.linalg.inv(np.asarray(flow.affine, dtype=np.float64)).astype(np.float32)
    sample_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {
        p.name: _plane_samples(p, square_plane=bool(args.square_plane)) for p in plane_specs
    }

    flux_by_plane: dict[str, np.ndarray] = {}
    area_by_plane: dict[str, np.ndarray] = {}
    valid_by_plane: dict[str, np.ndarray] = {}

    for p in plane_specs:
        flux_values = np.full((len(frames),), np.nan, dtype=np.float64)
        area_values = np.zeros((len(frames),), dtype=np.float64)
        valid_values = np.zeros((len(frames),), dtype=np.int32)
        points_xyz, dA_mm2 = sample_cache[p.name]
        ijk = _world_to_ijk(points_xyz, inv_affine)

        for j, t in enumerate(frames):
            vx_t = extract_frame(flow.vx, t)
            vy_t = extract_frame(flow.vy, t)
            vz_t = extract_frame(flow.vz, t)
            mask_t = extract_frame(flow.mask, t) if flow.mask is not None else None

            sx, in_x = _trilinear(vx_t, ijk)
            sy, in_y = _trilinear(vy_t, ijk)
            sz, in_z = _trilinear(vz_t, ijk)
            valid = in_x & in_y & in_z

            speed = np.sqrt(sx * sx + sy * sy + sz * sz)
            if float(args.zero_velocity_below) > 0:
                valid &= speed >= float(args.zero_velocity_below)

            if mask_t is not None:
                sm, in_m = _trilinear(mask_t.astype(np.float32), ijk)
                valid &= in_m & (sm > 0.5)

            if not np.any(valid):
                continue

            vn = sx * float(p.normal[0]) + sy * float(p.normal[1]) + sz * float(p.normal[2])
            flux_ml_s = float(np.sum(vn[valid] * dA_mm2[valid]))
            area_mm2 = float(np.sum(dA_mm2[valid]))

            flux_values[j] = flux_ml_s
            area_values[j] = area_mm2
            valid_values[j] = int(np.sum(valid))

        flux_by_plane[p.name] = flux_values
        area_by_plane[p.name] = area_values
        valid_by_plane[p.name] = valid_values

    print("\nFlux summary by plane (ml/s):")
    summary_lines = ["Flux summary by plane (ml/s):"]
    role_by_plane = {p.name: p.role for p in plane_specs}
    for p in plane_specs:
        flux = flux_by_plane[p.name]
        finite = np.isfinite(flux)
        if not np.any(finite):
            warn = f"  {p.name} [{p.role}] no valid samples across selected frames (check center/normal/radius and mask overlap)."
            print(warn)
            summary_lines.append(warn)
            continue
        mean_flux = _nanmean(flux)
        pos_frac = _nansign_fraction(flux, positive=True)
        neg_frac = _nansign_fraction(flux, positive=False)
        p95 = float(np.nanpercentile(flux, 95.0))
        p05 = float(np.nanpercentile(flux, 5.0))
        msg = (
            f"  {p.name} [{p.role}] mean={mean_flux:.4f} "
            f"p05={p05:.4f} p95={p95:.4f} pos_frac={pos_frac:.3f} neg_frac={neg_frac:.3f}"
        )
        print(msg)
        summary_lines.append(msg)
        if p.role in {"in", "out"} and np.isfinite(mean_flux) and mean_flux < 0:
            warn = f"    Warning: {p.name} has negative mean flux for role={p.role}. Consider flipping normal."
            print(warn)
            summary_lines.append(warn)

    in_names = [p.name for p in plane_specs if p.role == "in"]
    out_names = [p.name for p in plane_specs if p.role == "out"]
    if in_names and out_names:
        in_total = np.nansum(np.vstack([flux_by_plane[name] for name in in_names]), axis=0)
        out_total = np.nansum(np.vstack([flux_by_plane[name] for name in out_names]), axis=0)

        if args.auto_flip_negative_role:
            for name in in_names + out_names:
                mean_flux = _nanmean(flux_by_plane[name])
                if np.isfinite(mean_flux) and mean_flux < 0:
                    flux_by_plane[name] = -flux_by_plane[name]
            in_total = np.nansum(np.vstack([flux_by_plane[name] for name in in_names]), axis=0)
            out_total = np.nansum(np.vstack([flux_by_plane[name] for name in out_names]), axis=0)

        balance = in_total - out_total
        denom = 0.5 * (np.abs(in_total) + np.abs(out_total)) + 1e-6
        rel_err = np.abs(balance) / denom
        msg1 = f"\nMass balance over selected frames: mean_abs(in-out)={_nanmean(np.abs(balance)):.4f} ml/s"
        msg2 = f"Relative balance error: mean={_nanmean(rel_err):.4f}, p95={float(np.nanpercentile(rel_err, 95.0)):.4f}"
        print(msg1)
        print(msg2)
        summary_lines.append(msg1.strip())
        summary_lines.append(msg2)
        flux_by_plane["IN_TOTAL"] = in_total
        flux_by_plane["OUT_TOTAL"] = out_total
        flux_by_plane["BALANCE_IN_MINUS_OUT"] = balance
        role_by_plane["IN_TOTAL"] = "none"
        role_by_plane["OUT_TOTAL"] = "none"
        role_by_plane["BALANCE_IN_MINUS_OUT"] = "none"

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame", "time_s"]
    for p in plane_specs:
        fieldnames.extend(
            [
                f"{p.name}_flux_ml_s",
                f"{p.name}_area_mm2",
                f"{p.name}_n_valid",
            ]
        )
    for extra in ("IN_TOTAL", "OUT_TOTAL", "BALANCE_IN_MINUS_OUT"):
        if extra in flux_by_plane:
            fieldnames.append(f"{extra}_flux_ml_s")

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j, t in enumerate(frames):
            row = {"frame": int(t), "time_s": float(times[j])}
            for p in plane_specs:
                row[f"{p.name}_flux_ml_s"] = float(flux_by_plane[p.name][j]) if np.isfinite(flux_by_plane[p.name][j]) else ""
                row[f"{p.name}_area_mm2"] = float(area_by_plane[p.name][j])
                row[f"{p.name}_n_valid"] = int(valid_by_plane[p.name][j])
            for extra in ("IN_TOTAL", "OUT_TOTAL", "BALANCE_IN_MINUS_OUT"):
                if extra in flux_by_plane:
                    val = flux_by_plane[extra][j]
                    row[f"{extra}_flux_ml_s"] = float(val) if np.isfinite(val) else ""
            writer.writerow(row)
    print(f"Flux CSV written to: {args.csv_out.resolve()}")

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Summary written to: {args.summary_out.resolve()}")

    if args.plot_png is not None:
        _save_plot_png(args.plot_png, times, flux_by_plane, role_by_plane)


if __name__ == "__main__":
    main()
