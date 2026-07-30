#!/usr/bin/env python3
"""Pathline-style particle animation through 4D flow time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from neuroflow.visualization.flowviz.common import (
    add_common_preproc_args,
    add_context_geometry,
    add_input_args,
    build_structured_grid,
    compute_speed_clim,
    extract_frame,
    fit_camera_to_bounds,
    load_flow_data_from_args,
    merge_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate moving particles to visualize blood flow direction over time.")
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument("--gif-path", type=Path, required=True)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--dt-scale", type=float, default=0.6, help="Motion scale per frame for particle advection.")

    parser.add_argument("--n-particles", type=int, default=900)
    parser.add_argument("--spawn-per-frame", type=int, default=120)
    parser.add_argument("--trail-length", type=int, default=8)
    parser.add_argument("--particle-size", type=float, default=10.0)
    parser.add_argument("--tail-width", type=float, default=2.2)

    parser.add_argument("--speed-seed-percentile", type=float, default=95.0)
    parser.add_argument("--speed-vmin", type=float, default=None)
    parser.add_argument("--speed-vmax", type=float, default=None)

    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--show-slice-context", action="store_true")
    parser.add_argument("--slice-opacity", type=float, default=0.10)
    parser.add_argument("--vessel-opacity", type=float, default=0.20)
    parser.add_argument("--vessel-color", type=str, default="lightgray")

    parser.add_argument("--orbit-deg-per-frame", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.5)
    parser.add_argument("--camera-zoom", type=float, default=1.25)

    parser.add_argument("--background", type=str, default="white")
    return parser.parse_args()


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
    out = np.zeros(x.shape[0], dtype=np.float32)
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

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    out_vals = c0 * (1 - zd) + c1 * zd
    out[inside] = out_vals.astype(np.float32)
    return out, inside


def _sample_velocity(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mask: np.ndarray | None, ijk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, in_x = _trilinear(vx, ijk)
    sy, in_y = _trilinear(vy, ijk)
    sz, in_z = _trilinear(vz, ijk)
    inside = in_x & in_y & in_z

    vel = np.stack([sx, sy, sz], axis=1)
    if mask is not None:
        sm, in_m = _trilinear(mask.astype(np.float32), ijk)
        inside = inside & in_m & (sm > 0.5)
    return vel, inside


def _seed_points_from_frame(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mask: np.ndarray | None, affine: np.ndarray, percentile: float) -> np.ndarray:
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    if mask is not None:
        candidates = speed[mask > 0]
    else:
        candidates = speed.reshape(-1)
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    thr = float(np.percentile(candidates, percentile))

    if mask is not None:
        idx = np.argwhere((speed >= thr) & (mask > 0))
    else:
        idx = np.argwhere(speed >= thr)
    if idx.size == 0:
        if mask is not None:
            idx = np.argwhere((speed > 0) & (mask > 0))
        else:
            idx = np.argwhere(speed > 0)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ijk = idx.astype(np.float32)
    homo = np.concatenate([ijk, np.ones((ijk.shape[0], 1), dtype=np.float32)], axis=1)
    xyz1 = homo @ affine.T
    return xyz1[:, :3].astype(np.float32)


def _build_trail_poly(trails: np.ndarray, speeds: np.ndarray) -> pv.PolyData | None:
    # trails shape: (T, N, 3)
    t_len, n_parts, _ = trails.shape
    segments = []
    seg_speeds = []
    for k in range(t_len - 1):
        p0 = trails[k]
        p1 = trails[k + 1]
        valid = np.isfinite(p0[:, 0]) & np.isfinite(p1[:, 0])
        if np.any(valid):
            a = p0[valid]
            b = p1[valid]
            s = speeds[valid]
            segments.append((a, b))
            seg_speeds.append(s)

    if not segments:
        return None

    points = []
    lines = []
    point_speed = []
    offset = 0
    for (a, b), s in zip(segments, seg_speeds):
        m = a.shape[0]
        pts = np.empty((2 * m, 3), dtype=np.float32)
        pts[0::2] = a
        pts[1::2] = b
        points.append(pts)
        ps = np.empty((2 * m,), dtype=np.float32)
        ps[0::2] = s
        ps[1::2] = s
        point_speed.append(ps)
        for i in range(m):
            lines.extend([2, offset + 2 * i, offset + 2 * i + 1])
        offset += 2 * m

    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.array(lines, dtype=np.int64)
    poly.point_data["speed"] = np.concatenate(point_speed)
    return poly


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    nt = flow.mag.shape[-1]
    inv_affine = np.linalg.inv(flow.affine)

    provided = None
    if args.speed_vmin is not None and args.speed_vmax is not None:
        provided = (float(args.speed_vmin), float(args.speed_vmax))
    global_speed = np.sqrt(flow.vx * flow.vx + flow.vy * flow.vy + flow.vz * flow.vz)
    if flow.mask is not None:
        global_speed = global_speed[flow.mask > 0]
    clim = compute_speed_clim(global_speed.reshape(-1), provided=provided, q_low=5.0, q_high=99.0)

    n = int(args.n_particles)
    positions = np.full((n, 3), np.nan, dtype=np.float32)
    speeds = np.zeros((n,), dtype=np.float32)
    alive = np.zeros((n,), dtype=bool)
    trails = np.full((int(args.trail_length), n, 3), np.nan, dtype=np.float32)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 900))
    plotter.set_background(args.background)
    args.gif_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(args.gif_path), fps=max(1, int(args.gif_fps)))

    camera_initialized = False
    for t in range(nt):
        vx_t = extract_frame(flow.vx, t)
        vy_t = extract_frame(flow.vy, t)
        vz_t = extract_frame(flow.vz, t)
        mask_t = extract_frame(flow.mask, t) if flow.mask is not None else None

        # Spawn new particles.
        seeds_xyz = _seed_points_from_frame(vx_t, vy_t, vz_t, mask_t, flow.affine, args.speed_seed_percentile)
        dead_idx = np.flatnonzero(~alive)
        spawn_n = min(int(args.spawn_per_frame), dead_idx.size, seeds_xyz.shape[0])
        if spawn_n > 0:
            rng = np.random.default_rng(1234 + t)
            pick_seed = rng.choice(seeds_xyz.shape[0], size=spawn_n, replace=False)
            choose_dead = dead_idx[:spawn_n]
            positions[choose_dead] = seeds_xyz[pick_seed]
            alive[choose_dead] = True
            speeds[choose_dead] = 0.0

        # Advect active particles.
        active_idx = np.flatnonzero(alive)
        if active_idx.size > 0:
            ijk = _world_to_ijk(positions[active_idx], inv_affine)
            vel, inside = _sample_velocity(vx_t, vy_t, vz_t, mask_t, ijk)
            speed = np.linalg.norm(vel, axis=1)

            if args.zero_velocity_below > 0:
                inside = inside & (speed >= float(args.zero_velocity_below))

            idx_inside = active_idx[inside]
            idx_outside = active_idx[~inside]

            if idx_inside.size > 0:
                positions[idx_inside] = positions[idx_inside] + (float(args.dt_scale) * vel[inside]).astype(np.float32)
                speeds[idx_inside] = speed[inside].astype(np.float32)

            if idx_outside.size > 0:
                alive[idx_outside] = False
                positions[idx_outside] = np.nan
                speeds[idx_outside] = 0.0

        # Update trails.
        trails[:-1] = trails[1:]
        trails[-1] = np.nan
        trails[-1, alive] = positions[alive]

        # Build frame geometry.
        plotter.clear()
        # Build a lightweight grid just for context (mask + slice).
        mag_t = extract_frame(flow.mag, t)
        context_grid = build_structured_grid(mag_t, vx_t, vy_t, vz_t, flow.affine, mask_t=mask_t)
        add_context_geometry(
            plotter=plotter,
            grid=context_grid,
            slice_normal=args.slice_normal,
            show_slice_context=args.show_slice_context,
            slice_opacity=args.slice_opacity,
            vessel_opacity=args.vessel_opacity,
            vessel_color=args.vessel_color,
        )

        trail_poly = _build_trail_poly(trails, speeds)
        if trail_poly is not None:
            plotter.add_mesh(
                trail_poly,
                scalars="speed",
                cmap="turbo",
                clim=clim,
                line_width=float(args.tail_width),
                scalar_bar_args={"title": "Velocity (m/s)"},
            )

        if np.any(alive):
            pts = pv.PolyData(positions[alive])
            pts.point_data["speed"] = speeds[alive]
            plotter.add_mesh(
                pts,
                scalars="speed",
                cmap="turbo",
                clim=clim,
                point_size=float(args.particle_size),
                render_points_as_spheres=True,
            )

        plotter.add_text(f"Frame {t + 1}/{nt}", font_size=14)
        plotter.add_axes()

        if not camera_initialized:
            b_list = []
            if trail_poly is not None and trail_poly.n_points > 0:
                b_list.append(trail_poly.bounds)
            if "mask" in context_grid.point_data:
                try:
                    vessel = context_grid.contour(isosurfaces=[0.5], scalars="mask")
                    if vessel.n_points > 0:
                        b_list.append(vessel.bounds)
                except Exception:
                    pass
            fit_camera_to_bounds(
                plotter=plotter,
                bounds=merge_bounds(b_list),
                azimuth_deg=args.camera_azimuth,
                elevation_deg=args.camera_elevation,
                distance_scale=args.camera_distance_scale,
                zoom=args.camera_zoom,
            )
            camera_initialized = True
        elif args.orbit_deg_per_frame != 0:
            plotter.camera.Azimuth(float(args.orbit_deg_per_frame))

        plotter.write_frame()

    plotter.close()
    print(f"Particle flow GIF written to: {args.gif_path.resolve()}")


if __name__ == "__main__":
    main()
