#!/usr/bin/env python3
"""Create 3D flow GIFs (streamlines in segmented vessels)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from flowviz.common import (
    add_common_preproc_args,
    add_context_geometry,
    add_input_args,
    add_streamline_args,
    build_direction_arrows,
    compute_speed_clim,
    extract_frame,
    fit_camera_to_bounds,
    load_flow_data_from_args,
    make_frame_grid,
    make_seeds_from_grid,
    merge_bounds,
    streamlines_from_source_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D flow streamlines GIF.")
    add_input_args(parser)
    add_common_preproc_args(parser)
    add_streamline_args(parser)

    parser.add_argument("--gif-path", type=Path, required=True, help="Output GIF path.")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--orbit-deg-per-frame", type=float, default=2.0)
    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--show-slice-context", action="store_true")
    parser.add_argument("--slice-opacity", type=float, default=0.18)
    parser.add_argument("--vessel-opacity", type=float, default=0.25)
    parser.add_argument("--vessel-color", type=str, default="lightgray")
    parser.add_argument("--show-direction-arrows", action="store_true", help="Overlay arrows on streamlines to show direction.")
    parser.add_argument("--arrow-stride", type=int, default=35)
    parser.add_argument("--arrow-factor", type=float, default=0.8)
    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.5)
    parser.add_argument("--camera-zoom", type=float, default=1.2, help="Camera zoom factor (>1 zoom in).")
    parser.add_argument("--background", type=str, default="black")
    return parser.parse_args()


def _global_speed_clim(flow, args: argparse.Namespace) -> tuple[float, float]:
    if args.speed_vmin is not None and args.speed_vmax is not None:
        return (float(args.speed_vmin), float(args.speed_vmax))

    speed_values = []
    nt = flow.mag.shape[-1]
    for t in range(nt):
        speed_t = np.sqrt(
            extract_frame(flow.vx, t) ** 2
            + extract_frame(flow.vy, t) ** 2
            + extract_frame(flow.vz, t) ** 2
        )
        if flow.mask is not None:
            mask_t = extract_frame(flow.mask, t) > 0
            speed_t = speed_t[mask_t]
        speed_values.append(speed_t.reshape(-1))
    all_speed = np.concatenate(speed_values) if speed_values else np.array([])
    return compute_speed_clim(all_speed, provided=None, q_low=5.0, q_high=99.2)


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)
    nt = flow.mag.shape[-1]
    clim = _global_speed_clim(flow, args)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 900))
    plotter.set_background(args.background)
    args.gif_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(args.gif_path), fps=max(1, int(args.gif_fps)))

    camera_initialized = False
    for t in range(nt):
        grid, _ = make_frame_grid(flow, t, zero_velocity_below=args.zero_velocity_below)

        seeds = make_seeds_from_grid(
            grid=grid,
            seed_mode=args.seed_mode,
            seed_radius=args.seed_radius,
            n_seed_points=args.n_seed_points,
            speed_seed_percentile=args.speed_seed_percentile,
        )
        streams = streamlines_from_source_compat(
            grid=grid,
            seeds=seeds,
            vectors="vel",
            step=args.step,
            max_length=args.max_length,
            integration_direction=args.integration_direction,
        )
        if "vel" in streams.point_data:
            streams.point_data["speed"] = np.linalg.norm(streams.point_data["vel"], axis=1)

        plotter.clear()
        add_context_geometry(
            plotter=plotter,
            grid=grid,
            slice_normal=args.slice_normal,
            show_slice_context=args.show_slice_context,
            slice_opacity=args.slice_opacity,
            vessel_opacity=args.vessel_opacity,
            vessel_color=args.vessel_color,
        )

        tube = streams.tube(radius=args.tube_radius)
        if args.stream_color_by_speed and "speed" in streams.point_data:
            plotter.add_mesh(
                tube,
                scalars="speed",
                cmap="turbo",
                clim=clim,
                scalar_bar_args={"title": "Velocity (m/s)"},
            )
        else:
            plotter.add_mesh(tube, color="deepskyblue")

        if args.show_direction_arrows:
            arrows = build_direction_arrows(streams, stride=args.arrow_stride, factor=args.arrow_factor)
            if arrows is not None:
                plotter.add_mesh(arrows, color="white", opacity=0.9)

        plotter.add_text(f"Frame {t + 1}/{nt}", font_size=14)
        plotter.add_axes()

        if not camera_initialized:
            bounds = [tube.bounds]
            if "mask" in grid.point_data:
                try:
                    vessel = grid.contour(isosurfaces=[0.5], scalars="mask")
                    if vessel.n_points > 0:
                        bounds.append(vessel.bounds)
                except Exception:
                    pass
            fit_camera_to_bounds(
                plotter=plotter,
                bounds=merge_bounds(bounds),
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
    print(f"3D flow GIF written to: {args.gif_path.resolve()}")


if __name__ == "__main__":
    main()
