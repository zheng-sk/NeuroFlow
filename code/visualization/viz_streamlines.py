#!/usr/bin/env python3
"""Interactive streamlines and panel visualization for 4D flow data."""

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
    fit_camera_to_bounds,
    load_flow_data_from_args,
    make_frame_grid,
    make_seeds_from_grid,
    merge_bounds,
    streamlines_from_source_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 4D flow visualization (slice/stream/panel).")
    add_input_args(parser)
    add_common_preproc_args(parser)
    add_streamline_args(parser)

    parser.add_argument("--mode", choices=["slice", "stream", "both", "panel", "paper2"], default="panel")
    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--glyph-stride", type=int, default=25)
    parser.add_argument("--glyph-factor", type=float, default=1.8)
    parser.add_argument("--show-slice-context", action="store_true")
    parser.add_argument("--slice-opacity", type=float, default=0.20)
    parser.add_argument("--vessel-opacity", type=float, default=0.22)
    parser.add_argument("--vessel-color", type=str, default="lightgray")
    parser.add_argument("--show-direction-arrows", action="store_true", help="Overlay arrow glyphs to show flow direction.")
    parser.add_argument("--arrow-stride", type=int, default=30, help="Point stride for direction arrows on streamlines.")
    parser.add_argument("--arrow-factor", type=float, default=0.7, help="Arrow glyph size factor.")
    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.5)
    parser.add_argument("--camera-zoom", type=float, default=1.2, help="Camera zoom factor (>1 zoom in).")
    parser.add_argument("--off-screen", action="store_true", help="Render without opening an interactive window.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Optional PNG output path.")
    parser.add_argument("--background", type=str, default="white")
    return parser.parse_args()


def _speed_clim_from_args(args: argparse.Namespace, values: np.ndarray) -> tuple[float, float]:
    provided = None
    if args.speed_vmin is not None and args.speed_vmax is not None:
        provided = (float(args.speed_vmin), float(args.speed_vmax))
    return compute_speed_clim(values, provided=provided)


def _show_or_save(plotter: pv.Plotter, args: argparse.Namespace) -> None:
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Image written to: {args.screenshot.resolve()}")
    else:
        if not args.off_screen:
            # Reliable fallback controls when touchpad zoom is not captured by VTK.
            # '=' or 'z' zoom in, '-' or 'x' zoom out, 'r' reset camera.
            plotter.add_key_event("=", lambda: (plotter.camera.Zoom(1.15), plotter.render()))
            plotter.add_key_event("z", lambda: (plotter.camera.Zoom(1.15), plotter.render()))
            plotter.add_key_event("-", lambda: (plotter.camera.Zoom(1.0 / 1.15), plotter.render()))
            plotter.add_key_event("x", lambda: (plotter.camera.Zoom(1.0 / 1.15), plotter.render()))
            plotter.add_key_event("r", lambda: (plotter.reset_camera(), plotter.render()))
        plotter.show()


def render_slice_glyphs(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    p = pv.Plotter(off_screen=args.off_screen)
    p.set_background(args.background)

    slc = grid.slice(normal=args.slice_normal)
    speed_vals = slc.point_data["speed"] if "speed" in slc.point_data else np.array([])
    clim = _speed_clim_from_args(args, speed_vals)
    p.add_mesh(slc, scalars="speed", cmap="turbo", clim=clim, opacity=0.85)

    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)

    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    p.add_mesh(glyphs, scalars="speed", cmap="turbo", clim=clim)

    p.add_axes()
    _show_or_save(p, args)


def _build_stream_mesh(grid: pv.StructuredGrid, args: argparse.Namespace) -> tuple[pv.PolyData, tuple[float, float]]:
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
    stream_speed = streams.point_data["speed"] if "speed" in streams.point_data else np.array([])
    return streams, _speed_clim_from_args(args, stream_speed)


def render_stream(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    p = pv.Plotter(off_screen=args.off_screen)
    p.set_background(args.background)

    add_context_geometry(
        plotter=p,
        grid=grid,
        slice_normal=args.slice_normal,
        show_slice_context=args.show_slice_context,
        slice_opacity=args.slice_opacity,
        vessel_opacity=args.vessel_opacity,
        vessel_color=args.vessel_color,
    )

    streams, clim = _build_stream_mesh(grid, args)
    tube = streams.tube(radius=args.tube_radius)
    if args.stream_color_by_speed and "speed" in streams.point_data:
        p.add_mesh(
            tube,
            scalars="speed",
            cmap="turbo",
            clim=clim,
            scalar_bar_args={"title": "Velocity (m/s)"},
        )
    else:
        p.add_mesh(tube, color="deepskyblue")

    if args.show_direction_arrows:
        arrows = build_direction_arrows(streams, stride=args.arrow_stride, factor=args.arrow_factor)
        if arrows is not None:
            p.add_mesh(arrows, color="white", opacity=0.9)

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


def render_panel(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    p = pv.Plotter(shape=(1, 3), off_screen=args.off_screen)
    p.set_background(args.background)
    slc = grid.slice(normal=args.slice_normal)
    speed_vals = slc.point_data["speed"] if "speed" in slc.point_data else np.array([])
    panel_clim = _speed_clim_from_args(args, speed_vals)

    p.subplot(0, 0)
    p.add_text("Velocity Magnitude", font_size=12)
    p.add_mesh(slc, scalars="speed", cmap="turbo", clim=panel_clim)
    p.add_axes()

    p.subplot(0, 1)
    p.add_text("Velocity Map", font_size=12)
    p.add_mesh(slc, scalars="speed", cmap="turbo", opacity=0.30, clim=panel_clim)
    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)
    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    p.add_mesh(glyphs, scalars="speed", cmap="turbo", clim=panel_clim)
    p.add_axes()

    p.subplot(0, 2)
    p.add_text("Streamlines", font_size=12)
    add_context_geometry(
        plotter=p,
        grid=grid,
        slice_normal=args.slice_normal,
        show_slice_context=args.show_slice_context,
        slice_opacity=args.slice_opacity,
        vessel_opacity=args.vessel_opacity,
        vessel_color=args.vessel_color,
    )

    streams, stream_clim = _build_stream_mesh(grid, args)
    tube = streams.tube(radius=args.tube_radius)
    if args.stream_color_by_speed and "speed" in streams.point_data:
        p.add_mesh(
            tube,
            scalars="speed",
            cmap="turbo",
            clim=stream_clim,
            scalar_bar_args={"title": "Velocity (m/s)"},
        )
    else:
        p.add_mesh(tube, color="deepskyblue")
    p.add_axes()

    _show_or_save(p, args)


def render_paper2(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    """Two-panel layout: velocity magnitude + velocity map (glyphs)."""
    p = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1400, 750))
    p.set_background(args.background)
    slc = grid.slice(normal=args.slice_normal)
    speed_vals = slc.point_data["speed"] if "speed" in slc.point_data else np.array([])
    panel_clim = _speed_clim_from_args(args, speed_vals)

    p.subplot(0, 0)
    p.add_text("Velocity magnitude", font_size=16)
    p.add_mesh(slc, scalars="speed", cmap="turbo", clim=panel_clim)
    p.add_axes()

    p.subplot(0, 1)
    p.add_text("Velocity map", font_size=16)
    p.add_mesh(slc, scalars="speed", cmap="turbo", opacity=0.20, clim=panel_clim)
    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)
    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    p.add_mesh(glyphs, scalars="speed", cmap="turbo", clim=panel_clim)
    p.add_axes()

    _show_or_save(p, args)


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)
    grid, _ = make_frame_grid(flow, args.frame, zero_velocity_below=args.zero_velocity_below)

    if args.mode in ("slice", "both"):
        render_slice_glyphs(grid, args)
    if args.mode in ("stream", "both"):
        render_stream(grid, args)
    if args.mode == "panel":
        render_panel(grid, args)
    if args.mode == "paper2":
        render_paper2(grid, args)


if __name__ == "__main__":
    main()
