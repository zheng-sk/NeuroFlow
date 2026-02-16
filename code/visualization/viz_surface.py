#!/usr/bin/env python3
"""Render segmented vessel surface colored by scalar (speed or magnitude proxy)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from flowviz.common import (
    add_common_preproc_args,
    add_input_args,
    build_structured_grid,
    compute_speed_clim,
    extract_frame,
    load_flow_data_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surface visualization for segmented vessels.")
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument("--scalar", choices=["speed", "mag"], default="speed")
    parser.add_argument("--aggregate", choices=["frame", "mean", "max"], default="frame")
    parser.add_argument("--cmap", type=str, default="hot")
    parser.add_argument("--background", type=str, default="white")
    parser.add_argument("--scalar-title", type=str, default="Speed (m/s)")
    parser.add_argument("--clim-min", type=float, default=None)
    parser.add_argument("--clim-max", type=float, default=None)
    parser.add_argument("--smooth-iters", type=int, default=15)

    parser.add_argument("--two-views", action="store_true")
    parser.add_argument("--left-azimuth", type=float, default=20.0)
    parser.add_argument("--right-azimuth", type=float, default=-160.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--zoom", type=float, default=1.2)

    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def _get_scalar_volume(flow, args: argparse.Namespace) -> np.ndarray:
    if args.scalar == "speed":
        volume_4d = np.sqrt(flow.vx * flow.vx + flow.vy * flow.vy + flow.vz * flow.vz)
    else:
        volume_4d = flow.mag

    if args.aggregate == "frame":
        return extract_frame(volume_4d, args.frame)
    if args.aggregate == "mean":
        return np.mean(volume_4d, axis=-1)
    return np.max(volume_4d, axis=-1)


def _camera_setup(plotter: pv.Plotter, azimuth: float, elevation: float, zoom: float) -> None:
    plotter.view_isometric()
    plotter.camera.Azimuth(float(azimuth))
    plotter.camera.Elevation(float(elevation))
    plotter.camera.Zoom(float(zoom))


def _render_subplot(
    plotter: pv.Plotter,
    surface: pv.PolyData,
    scalars_name: str,
    cmap: str,
    clim: tuple[float, float],
    scalar_title: str,
    azimuth: float,
    elevation: float,
    zoom: float,
) -> None:
    plotter.add_mesh(
        surface,
        scalars=scalars_name,
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        scalar_bar_args={"title": scalar_title, "vertical": False},
    )
    _camera_setup(plotter, azimuth=azimuth, elevation=elevation, zoom=zoom)


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    if flow.mask is None:
        raise ValueError("Surface mode requires --mask (segmented vessel volume).")

    mask_t = extract_frame(flow.mask, args.frame)
    mag_t = extract_frame(flow.mag, args.frame)
    vx_t = extract_frame(flow.vx, args.frame)
    vy_t = extract_frame(flow.vy, args.frame)
    vz_t = extract_frame(flow.vz, args.frame)

    grid = build_structured_grid(mag_t, vx_t, vy_t, vz_t, flow.affine, mask_t=mask_t)

    scalar_volume = _get_scalar_volume(flow, args)
    if flow.mask is not None:
        scalar_volume = np.where(mask_t > 0, scalar_volume, 0.0)

    scalar_name = f"surface_{args.scalar}"
    grid.point_data[scalar_name] = scalar_volume.ravel(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="mask")
    if surface.n_points == 0:
        raise ValueError("Mask contour produced an empty surface. Check mask alignment/threshold.")

    if args.smooth_iters > 0:
        surface = surface.smooth(n_iter=int(args.smooth_iters), relaxation_factor=0.1)

    surface = surface.sample(grid)

    provided_clim = None
    if args.clim_min is not None and args.clim_max is not None:
        provided_clim = (float(args.clim_min), float(args.clim_max))
    clim = compute_speed_clim(surface.point_data[scalar_name], provided=provided_clim, q_low=2.0, q_high=98.5)

    shape = (1, 2) if args.two_views else (1, 1)
    p = pv.Plotter(shape=shape, off_screen=args.off_screen, window_size=(1500, 700))
    p.set_background(args.background)

    p.subplot(0, 0)
    _render_subplot(
        plotter=p,
        surface=surface,
        scalars_name=scalar_name,
        cmap=args.cmap,
        clim=clim,
        scalar_title=args.scalar_title,
        azimuth=args.left_azimuth,
        elevation=args.elevation,
        zoom=args.zoom,
    )

    if args.two_views:
        p.subplot(0, 1)
        _render_subplot(
            plotter=p,
            surface=surface,
            scalars_name=scalar_name,
            cmap=args.cmap,
            clim=clim,
            scalar_title=args.scalar_title,
            azimuth=args.right_azimuth,
            elevation=args.elevation,
            zoom=args.zoom,
        )

    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        p.show(auto_close=False)
        p.screenshot(str(args.screenshot))
        p.close()
        print(f"Surface image written to: {args.screenshot.resolve()}")
    else:
        p.show()


if __name__ == "__main__":
    main()
