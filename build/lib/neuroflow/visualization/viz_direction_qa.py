#!/usr/bin/env python3
"""Direction QA for streamlines (forward vs backward)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from neuroflow.visualization.flowviz.common import (
    add_common_preproc_args,
    add_context_geometry,
    add_input_args,
    add_streamline_args,
    build_direction_arrows,
    compute_speed_clim,
    fit_camera_to_bounds,
    load_flow_data_from_args,
    make_frame_grid,
    make_seeds_from_grid,
    merge_bounds,
    streamlines_from_source_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA plot to verify streamline direction consistency.")
    add_input_args(parser)
    add_common_preproc_args(parser)
    add_streamline_args(parser)

    parser.add_argument("--signed-component", choices=["none", "vx", "vy", "vz"], default="none")
    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--show-slice-context", action="store_true")
    parser.add_argument("--slice-opacity", type=float, default=0.15)
    parser.add_argument("--vessel-opacity", type=float, default=0.22)
    parser.add_argument("--vessel-color", type=str, default="lightgray")

    parser.add_argument("--show-direction-arrows", action="store_true")
    parser.add_argument("--arrow-stride", type=int, default=40)
    parser.add_argument("--arrow-factor", type=float, default=0.7)

    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.5)
    parser.add_argument("--camera-zoom", type=float, default=1.2)

    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--background", type=str, default="white")
    return parser.parse_args()


def _show_or_save(plotter: pv.Plotter, args: argparse.Namespace) -> None:
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Direction QA image written to: {args.screenshot.resolve()}")
    else:
        plotter.show()


def _signed_name(component: str) -> str:
    return {
        "vx": "Vx",
        "vy": "Vy",
        "vz": "Vz",
    }.get(component, "")


def _build_streams(grid: pv.StructuredGrid, args: argparse.Namespace, direction: str) -> pv.PolyData:
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
        integration_direction=direction,
    )
    if "vel" in streams.point_data:
        vel = streams.point_data["vel"]
        streams.point_data["speed"] = np.linalg.norm(vel, axis=1)
        if args.signed_component != "none":
            idx = {"vx": 0, "vy": 1, "vz": 2}[args.signed_component]
            streams.point_data["signed"] = vel[:, idx]
    return streams


def _add_stream_mesh(plotter: pv.Plotter, streams: pv.PolyData, args: argparse.Namespace) -> pv.PolyData:
    tube = streams.tube(radius=args.tube_radius)

    if args.signed_component == "none":
        speed_vals = streams.point_data["speed"] if "speed" in streams.point_data else np.array([])
        provided = None
        if args.speed_vmin is not None and args.speed_vmax is not None:
            provided = (float(args.speed_vmin), float(args.speed_vmax))
        clim = compute_speed_clim(speed_vals, provided=provided)
        plotter.add_mesh(
            tube,
            scalars="speed",
            cmap="turbo",
            clim=clim,
            scalar_bar_args={"title": "Velocity (m/s)"},
        )
    else:
        signed_vals = streams.point_data["signed"] if "signed" in streams.point_data else np.array([])
        max_abs = float(np.percentile(np.abs(signed_vals), 99.0)) if signed_vals.size > 0 else 1.0
        max_abs = max(max_abs, 1e-6)
        plotter.add_mesh(
            tube,
            scalars="signed",
            cmap="coolwarm",
            clim=(-max_abs, max_abs),
            scalar_bar_args={"title": f"Signed {_signed_name(args.signed_component)} (m/s)"},
        )

    if args.show_direction_arrows:
        arrows = build_direction_arrows(streams, stride=args.arrow_stride, factor=args.arrow_factor)
        if arrows is not None:
            plotter.add_mesh(arrows, color="white", opacity=0.9)

    return tube


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)
    grid, _ = make_frame_grid(flow, args.frame, zero_velocity_below=args.zero_velocity_below)

    p = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1500, 760))
    p.set_background(args.background)

    directions = ["forward", "backward"]
    for col, direction in enumerate(directions):
        p.subplot(0, col)
        p.add_text(f"Streamlines ({direction})", font_size=12)
        add_context_geometry(
            plotter=p,
            grid=grid,
            slice_normal=args.slice_normal,
            show_slice_context=args.show_slice_context,
            slice_opacity=args.slice_opacity,
            vessel_opacity=args.vessel_opacity,
            vessel_color=args.vessel_color,
        )

        streams = _build_streams(grid, args, direction)
        tube = _add_stream_mesh(p, streams, args)

        bounds = [tube.bounds]
        if "mask" in grid.point_data:
            try:
                vessel = grid.contour(isosurfaces=[0.5], scalars="mask")
                if vessel.n_points > 0:
                    bounds.append(vessel.bounds)
            except Exception:
                pass

        fit_camera_to_bounds(
            plotter=p,
            bounds=merge_bounds(bounds),
            azimuth_deg=args.camera_azimuth,
            elevation_deg=args.camera_elevation,
            distance_scale=args.camera_distance_scale,
            zoom=args.camera_zoom,
        )
        p.add_axes()

    _show_or_save(p, args)


if __name__ == "__main__":
    main()
