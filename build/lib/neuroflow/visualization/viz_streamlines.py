#!/usr/bin/env python3
"""Interactive streamlines and panel visualization for 4D flow data."""

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
    parser = argparse.ArgumentParser(description="Interactive 4D flow visualization (slice/stream/panel).")
    add_input_args(parser)
    add_common_preproc_args(parser)
    add_streamline_args(parser)

    parser.add_argument("--mode", choices=["slice", "stream", "both", "panel", "paper2"], default="panel")
    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--glyph-stride", type=int, default=25)
    parser.add_argument("--glyph-factor", type=float, default=1.8)
    parser.add_argument(
        "--stream-style",
        choices=["line", "tube"],
        default="line",
        help="Render streamlines as thin lines (paper-like) or tubes.",
    )
    parser.add_argument("--line-width", type=float, default=1.6, help="Line width when --stream-style line.")
    parser.add_argument("--stream-opacity", type=float, default=1.0, help="Opacity for streamline geometry.")
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Apply a paper-like preset (thin long streamlines with dense vessel seeding).",
    )
    parser.add_argument(
        "--scalar",
        choices=["speed", "vx", "vy", "vz"],
        default="speed",
        help="Scalar used to color slice/streamlines.",
    )
    parser.add_argument("--scalar-vmin", type=float, default=None, help="Optional min for selected scalar color range.")
    parser.add_argument("--scalar-vmax", type=float, default=None, help="Optional max for selected scalar color range.")
    parser.add_argument(
        "--signed-clim-percentile",
        type=float,
        default=99.0,
        help="Percentile for symmetric color limits when scalar is vx/vy/vz.",
    )
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
    parser.add_argument(
        "--clip-box-bounds",
        type=str,
        default=None,
        help="ROI bounds as x0,x1,y0,y1,z0,z1 in world coordinates (mm).",
    )
    parser.add_argument("--clip-invert", action="store_true", help="Keep outside of the ROI box instead of inside.")
    parser.add_argument(
        "--interactive-clip-box",
        action="store_true",
        help="Enable interactive ROI clip widget (stream mode, on-screen only).",
    )
    parser.add_argument(
        "--interactive-controls",
        action="store_true",
        help="Enable in-window controls (keys + sliders) to edit streamline properties without relaunching.",
    )
    parser.add_argument(
        "--roi-only-interaction",
        action="store_true",
        help="Lock scene interaction style to keep volume orientation fixed while editing the ROI clip box.",
    )
    parser.add_argument(
        "--clip-widget-rotation",
        action="store_true",
        help="Allow rotating the clip widget (disabled by default).",
    )
    parser.add_argument(
        "--clip-widget-no-translation",
        action="store_true",
        help="Disable translation of the clip widget outline (resize-only).",
    )
    parser.add_argument("--off-screen", action="store_true", help="Render without opening an interactive window.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Optional PNG output path.")
    parser.add_argument("--background", type=str, default="white")
    return parser.parse_args()


def _speed_clim_from_args(args: argparse.Namespace, values: np.ndarray) -> tuple[float, float]:
    provided = None
    if args.speed_vmin is not None and args.speed_vmax is not None:
        provided = (float(args.speed_vmin), float(args.speed_vmax))
    return compute_speed_clim(values, provided=provided)


def _parse_clip_bounds(text: str | None) -> tuple[float, float, float, float, float, float] | None:
    if text is None:
        return None
    tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
    if len(tokens) != 6:
        raise ValueError("--clip-box-bounds must have 6 comma-separated values: x0,x1,y0,y1,z0,z1")
    values = [float(tok) for tok in tokens]
    x0, x1, y0, y1, z0, z1 = values
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if z0 > z1:
        z0, z1 = z1, z0
    return (x0, x1, y0, y1, z0, z1)


def _clip_value(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _cycle_direction(direction: str) -> str:
    order = ["both", "forward", "backward"]
    if direction not in order:
        return "both"
    return order[(order.index(direction) + 1) % len(order)]


def _expanded_bounds(bounds: tuple[float, float, float, float, float, float], factor: float = 1.08) -> tuple[float, float, float, float, float, float]:
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    hx = 0.5 * max(bounds[1] - bounds[0], 1e-3) * float(factor)
    hy = 0.5 * max(bounds[3] - bounds[2], 1e-3) * float(factor)
    hz = 0.5 * max(bounds[5] - bounds[4], 1e-3) * float(factor)
    return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


def _ui_colors(background: str) -> tuple[str, str]:
    try:
        rgb = np.array(pv.Color(background).float_rgb, dtype=np.float64)
        luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    except Exception:
        luminance = 0.5
    if luminance >= 0.55:
        return ("black", "dimgray")
    return ("white", "gainsboro")


def _stream_scalar_bar_args(title: str, interactive: bool, ui_color: str) -> dict:
    if not interactive:
        return {"title": title}
    # Keep scalar bar away from bottom sliders and left HUD in interactive mode.
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.87,
        "position_y": 0.20,
        "width": 0.08,
        "height": 0.60,
        "title_font_size": 10,
        "label_font_size": 9,
        "color": ui_color,
    }


def _apply_roi_only_style(plotter: pv.Plotter) -> None:
    """Disable camera rotation while preserving 3D widget interaction."""
    try:
        plotter.enable_custom_trackball_style(
            left="pan",
            shift_left="pan",
            control_left="pan",
            middle="pan",
            shift_middle="pan",
            control_middle="pan",
            right="dolly",
            shift_right="dolly",
            control_right="dolly",
        )
    except Exception:
        # Safe fallback if a backend doesn't expose custom style.
        plotter.enable_trackball_style()


def _ensure_scalar_arrays(mesh: pv.DataSet) -> None:
    if "vel" not in mesh.point_data:
        return
    vel = np.asarray(mesh.point_data["vel"])
    if vel.ndim != 2 or vel.shape[1] < 3:
        return
    if "vx" not in mesh.point_data:
        mesh.point_data["vx"] = vel[:, 0]
    if "vy" not in mesh.point_data:
        mesh.point_data["vy"] = vel[:, 1]
    if "vz" not in mesh.point_data:
        mesh.point_data["vz"] = vel[:, 2]
    if "speed" not in mesh.point_data:
        mesh.point_data["speed"] = np.linalg.norm(vel[:, :3], axis=1)


def _scalar_meta(
    args: argparse.Namespace,
    values: np.ndarray,
    scalar_name: str | None = None,
) -> tuple[str, str, tuple[float, float], str]:
    scalar = args.scalar if scalar_name is None else scalar_name
    if scalar == "speed":
        clim = _speed_clim_from_args(args, values)
        if args.scalar_vmin is not None and args.scalar_vmax is not None:
            clim = (float(args.scalar_vmin), float(args.scalar_vmax))
        return ("speed", "turbo", clim, "Velocity (m/s)")

    finite = values[np.isfinite(values)]
    if args.scalar_vmin is not None and args.scalar_vmax is not None:
        clim = (float(args.scalar_vmin), float(args.scalar_vmax))
    elif finite.size > 0:
        vmax = float(np.percentile(np.abs(finite), float(args.signed_clim_percentile)))
        vmax = max(vmax, 1e-6)
        clim = (-vmax, vmax)
    else:
        clim = (-1.0, 1.0)

    title_map = {"vx": "Vx (m/s)", "vy": "Vy (m/s)", "vz": "Vz (m/s)"}
    return (scalar, "coolwarm", clim, title_map[scalar])


def _show_or_save(plotter: pv.Plotter, args: argparse.Namespace) -> None:
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Image written to: {args.screenshot.resolve()}")
    else:
        if not args.off_screen and not args.roi_only_interaction:
            # Reliable fallback controls when touchpad zoom is not captured by VTK.
            # '=' or 'z' zoom in, '-' or 'x' zoom out, 'r' reset camera.
            plotter.add_key_event("=", lambda: (plotter.camera.Zoom(1.15), plotter.render()))
            plotter.add_key_event("z", lambda: (plotter.camera.Zoom(1.15), plotter.render()))
            plotter.add_key_event("-", lambda: (plotter.camera.Zoom(1.0 / 1.15), plotter.render()))
            plotter.add_key_event("x", lambda: (plotter.camera.Zoom(1.0 / 1.15), plotter.render()))
            plotter.add_key_event("r", lambda: (plotter.reset_camera(), plotter.render()))
        plotter.show()


def _apply_paper_preset(args: argparse.Namespace, has_mask: bool) -> None:
    if not bool(args.paper_style):
        return
    args.stream_style = "line"
    args.line_width = min(float(args.line_width), 1.1)
    args.stream_opacity = min(float(args.stream_opacity), 0.95)
    args.integration_direction = "forward"
    args.step = min(float(args.step), 0.18)
    args.max_length = max(float(args.max_length), 700.0)
    args.stream_color_by_speed = True
    args.show_direction_arrows = False
    args.vessel_opacity = min(float(args.vessel_opacity), 0.16)
    if has_mask:
        args.seed_mode = "mask"
        args.n_seed_points = max(int(args.n_seed_points), 3200)
    else:
        args.seed_mode = "speed"
        args.n_seed_points = max(int(args.n_seed_points), 2600)
        args.speed_seed_percentile = min(float(args.speed_seed_percentile), 92.0)
    args.seed_min_speed = max(float(getattr(args, "seed_min_speed", 0.0)), 0.05)
    print(
        "Paper preset applied: "
        f"seed_mode={args.seed_mode} n_seeds={args.n_seed_points} seed_min_speed={args.seed_min_speed:.3f} "
        f"step={args.step:.3f} max_length={args.max_length:.1f} style={args.stream_style} line_width={args.line_width:.2f}"
    )


def _stream_geometry(streams: pv.PolyData, args: argparse.Namespace) -> pv.PolyData:
    if args.stream_style == "tube":
        geom = streams.tube(radius=float(args.tube_radius))
    else:
        geom = streams.copy(deep=True)
    _ensure_scalar_arrays(geom)
    return geom


def render_slice_glyphs(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    p = pv.Plotter(off_screen=args.off_screen)
    p.set_background(args.background)

    slc = grid.slice(normal=args.slice_normal)
    _ensure_scalar_arrays(slc)
    scalar_name = args.scalar
    values = slc.point_data[scalar_name] if scalar_name in slc.point_data else np.array([])
    scalar_name, cmap, clim, scalar_title = _scalar_meta(args, values)
    p.add_mesh(slc, scalars=scalar_name, cmap=cmap, clim=clim, opacity=0.85, scalar_bar_args={"title": scalar_title})

    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)

    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    glyph_scalar = scalar_name if scalar_name in glyphs.point_data else "speed"
    glyph_cmap = cmap if glyph_scalar == scalar_name else "turbo"
    p.add_mesh(glyphs, scalars=glyph_scalar, cmap=glyph_cmap, clim=clim)

    p.add_axes()
    _show_or_save(p, args)


def _build_stream_mesh(grid: pv.StructuredGrid, args: argparse.Namespace) -> pv.PolyData:
    seeds = make_seeds_from_grid(
        grid=grid,
        seed_mode=args.seed_mode,
        seed_radius=args.seed_radius,
        n_seed_points=args.n_seed_points,
        speed_seed_percentile=args.speed_seed_percentile,
        seed_min_speed=float(getattr(args, "seed_min_speed", 0.0)),
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
        _ensure_scalar_arrays(streams)
    return streams


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

    interactive_session = (
        bool(args.interactive_controls)
        and not args.off_screen
        and args.screenshot is None
    )
    interactive_clip = bool(args.interactive_clip_box) and not args.off_screen and args.screenshot is None
    hud_color, arrow_color = _ui_colors(args.background)

    if interactive_session:
        default_clip_bounds = _parse_clip_bounds(args.clip_box_bounds)
        if default_clip_bounds is None:
            if "mask" in grid.point_data:
                try:
                    vessel = grid.contour(isosurfaces=[0.5], scalars="mask")
                    if vessel.n_points > 0:
                        default_clip_bounds = _expanded_bounds(vessel.bounds, factor=1.08)
                except Exception:
                    default_clip_bounds = None
            if default_clip_bounds is None:
                default_clip_bounds = _expanded_bounds(grid.bounds, factor=1.02)

        state = {
            "scalar": args.scalar,
            "seed_mode": args.seed_mode,
            "seed_radius": float(args.seed_radius),
            "n_seed_points": int(args.n_seed_points),
            "speed_seed_percentile": float(args.speed_seed_percentile),
            "seed_min_speed": float(getattr(args, "seed_min_speed", 0.0)),
            "step": float(args.step),
            "max_length": float(args.max_length),
            "stream_style": args.stream_style,
            "line_width": float(args.line_width),
            "stream_opacity": float(args.stream_opacity),
            "tube_radius": float(args.tube_radius),
            "integration_direction": args.integration_direction,
            "show_direction_arrows": bool(args.show_direction_arrows),
            "arrow_stride": int(args.arrow_stride),
            "arrow_factor": float(args.arrow_factor),
            "color_streams": bool(args.stream_color_by_speed or args.interactive_controls or args.scalar != "speed"),
            "clip_bounds": default_clip_bounds,
            "clip_invert": bool(args.clip_invert),
            "hud": True,
            "camera_init": False,
        }
        actors: dict[str, object | None] = {"stream": None, "arrows": None}

        def update_hud(extra: str = "") -> None:
            if not state["hud"]:
                p.add_text("", name="stream_hud", position="upper_left")
                return
            clip_txt = (
                "None"
                if state["clip_bounds"] is None
                else ",".join([f"{v:.1f}" for v in state["clip_bounds"]])
            )
            lines = [
                "Interactive Streamline Controls",
                f"scalar={state['scalar']}  dir={state['integration_direction']}  color={'on' if state['color_streams'] else 'off'}",
                f"seeds={state['n_seed_points']}  pctl={state['speed_seed_percentile']:.1f}  seed_vmin={state['seed_min_speed']:.3f}",
                f"style={state['stream_style']}  line_w={state['line_width']:.2f}  tube={state['tube_radius']:.3f}  step={state['step']:.3f}",
                f"max_len={state['max_length']:.1f}  arrows={'on' if state['show_direction_arrows'] else 'off'}",
                f"clip_invert={state['clip_invert']}  clip_bounds={clip_txt}",
                "Keys: 1/2/3/4 scalar | n/m seeds | [/] pctl | v/g seed_vmin | ,/. tube",
                "      ;/' step | k/l length | i direction | f arrows | c color | b invert clip | h HUD | u reset camera",
            ]
            if extra:
                lines.append(extra)
            p.add_text("\n".join(lines), name="stream_hud", position="upper_left", font_size=10, color=hud_color)

        def rebuild(reset_camera: bool = False, note: str = "") -> None:
            local_args = argparse.Namespace(**vars(args))
            local_args.seed_mode = state["seed_mode"]
            local_args.seed_radius = float(state["seed_radius"])
            local_args.n_seed_points = int(state["n_seed_points"])
            local_args.speed_seed_percentile = float(state["speed_seed_percentile"])
            local_args.seed_min_speed = float(state["seed_min_speed"])
            local_args.step = float(state["step"])
            local_args.max_length = float(state["max_length"])
            local_args.stream_style = state["stream_style"]
            local_args.line_width = float(state["line_width"])
            local_args.stream_opacity = float(state["stream_opacity"])
            local_args.integration_direction = state["integration_direction"]
            local_args.scalar = state["scalar"]

            try:
                streams = _build_stream_mesh(grid, local_args)
            except Exception as exc:
                update_hud(f"Rebuild error: {exc}")
                p.render()
                return

            stream_geom = _stream_geometry(streams, local_args)
            if state["clip_bounds"] is not None:
                stream_geom = stream_geom.clip_box(bounds=state["clip_bounds"], invert=bool(state["clip_invert"]))

            scalar = state["scalar"]
            values = stream_geom.point_data[scalar] if scalar in stream_geom.point_data else np.array([])
            scalar_name, cmap, clim, scalar_title = _scalar_meta(local_args, values, scalar_name=scalar)

            if actors["stream"] is not None:
                p.remove_actor(actors["stream"], reset_camera=False)
                actors["stream"] = None
            if actors["arrows"] is not None:
                p.remove_actor(actors["arrows"], reset_camera=False)
                actors["arrows"] = None

            if stream_geom.n_points > 0:
                mesh_kwargs = {"opacity": float(state["stream_opacity"]), "name": "stream_tube"}
                if state["stream_style"] == "line":
                    mesh_kwargs["line_width"] = float(state["line_width"])
                if state["color_streams"] or scalar != "speed":
                    actors["stream"] = p.add_mesh(
                        stream_geom,
                        scalars=scalar_name,
                        cmap=cmap,
                        clim=clim,
                        scalar_bar_args=_stream_scalar_bar_args(
                            scalar_title, interactive=True, ui_color=hud_color
                        ),
                        **mesh_kwargs,
                    )
                else:
                    actors["stream"] = p.add_mesh(stream_geom, color="deepskyblue", **mesh_kwargs)

            if state["show_direction_arrows"]:
                arrows = build_direction_arrows(streams, stride=int(state["arrow_stride"]), factor=float(state["arrow_factor"]))
                if arrows is not None and arrows.n_points > 0:
                    if state["clip_bounds"] is not None:
                        arrows = arrows.clip_box(bounds=state["clip_bounds"], invert=bool(state["clip_invert"]))
                    if arrows.n_points > 0:
                        actors["arrows"] = p.add_mesh(arrows, color=arrow_color, opacity=0.9, name="stream_arrows")

            if reset_camera or not state["camera_init"]:
                bounds = [stream_geom.bounds] if stream_geom.n_points > 0 else [grid.bounds]
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
                state["camera_init"] = True
                if args.roi_only_interaction:
                    # Keep vessel orientation fixed while preserving box widget interaction.
                    _apply_roi_only_style(p)

            update_hud(note)
            p.render()

        def set_scalar(name: str) -> None:
            state["scalar"] = name
            if name != "speed":
                state["color_streams"] = True
            rebuild(note=f"scalar -> {name}")

        def on_box(poly: pv.PolyData) -> None:
            state["clip_bounds"] = tuple(float(v) for v in poly.bounds)
            rebuild(note="ROI box updated")

        # Sliders
        slider_kwargs = {
            "style": "modern",
            "title_height": 0.015,
            "slider_width": 0.015,
            "tube_width": 0.006,
            "title_color": hud_color,
            "color": "gray",
        }
        p.add_slider_widget(
            lambda value: (state.__setitem__("n_seed_points", int(_clip_value(value, 30, 8000))), rebuild(note="seeds updated")),
            rng=(30, 4000),
            value=float(state["n_seed_points"]),
            title="Seeds",
            pointa=(0.03, 0.12),
            pointb=(0.30, 0.12),
            fmt="%.0f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("speed_seed_percentile", _clip_value(value, 50.0, 99.9)), rebuild(note="percentile updated")),
            rng=(50.0, 99.9),
            value=float(state["speed_seed_percentile"]),
            title="Seed pctl",
            pointa=(0.03, 0.08),
            pointb=(0.30, 0.08),
            fmt="%.1f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("seed_min_speed", _clip_value(value, 0.0, 1.5)), rebuild(note="seed min speed updated")),
            rng=(0.0, 1.5),
            value=float(state["seed_min_speed"]),
            title="Seed vmin",
            pointa=(0.03, 0.04),
            pointb=(0.30, 0.04),
            fmt="%.3f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("step", _clip_value(value, 0.02, 3.0)), rebuild(note="step updated")),
            rng=(0.02, 3.0),
            value=float(state["step"]),
            title="Step",
            pointa=(0.34, 0.12),
            pointb=(0.61, 0.12),
            fmt="%.3f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("max_length", _clip_value(value, 20.0, 3000.0)), rebuild(note="max length updated")),
            rng=(20.0, 3000.0),
            value=float(state["max_length"]),
            title="Length",
            pointa=(0.34, 0.08),
            pointb=(0.61, 0.08),
            fmt="%.1f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("tube_radius", _clip_value(value, 0.02, 3.0)), rebuild(note="tube radius updated")),
            rng=(0.02, 3.0),
            value=float(state["tube_radius"]),
            title="Tube",
            pointa=(0.65, 0.12),
            pointb=(0.92, 0.12),
            fmt="%.3f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("line_width", _clip_value(value, 0.2, 8.0)), rebuild(note="line width updated")),
            rng=(0.2, 8.0),
            value=float(state["line_width"]),
            title="Line w",
            pointa=(0.65, 0.04),
            pointb=(0.92, 0.04),
            fmt="%.2f",
            **slider_kwargs,
        )
        p.add_slider_widget(
            lambda value: (state.__setitem__("arrow_stride", int(_clip_value(value, 5, 120))), rebuild(note="arrow stride updated")),
            rng=(5, 120),
            value=float(state["arrow_stride"]),
            title="Arrow",
            pointa=(0.65, 0.08),
            pointb=(0.92, 0.08),
            fmt="%.0f",
            **slider_kwargs,
        )

        # Keyboard shortcuts
        p.add_key_event("1", lambda: set_scalar("speed"))
        p.add_key_event("2", lambda: set_scalar("vx"))
        p.add_key_event("3", lambda: set_scalar("vy"))
        p.add_key_event("4", lambda: set_scalar("vz"))
        p.add_key_event("t", lambda: (state.__setitem__("stream_style", "tube" if state["stream_style"] == "line" else "line"), rebuild(note=f"style -> {state['stream_style']}")))
        p.add_key_event("n", lambda: (state.__setitem__("n_seed_points", max(30, int(state["n_seed_points"]) - 50)), rebuild(note="seeds -50")))
        p.add_key_event("m", lambda: (state.__setitem__("n_seed_points", min(8000, int(state["n_seed_points"]) + 50)), rebuild(note="seeds +50")))
        p.add_key_event("[", lambda: (state.__setitem__("speed_seed_percentile", _clip_value(state["speed_seed_percentile"] - 1.0, 50.0, 99.9)), rebuild(note="percentile -1")))
        p.add_key_event("]", lambda: (state.__setitem__("speed_seed_percentile", _clip_value(state["speed_seed_percentile"] + 1.0, 50.0, 99.9)), rebuild(note="percentile +1")))
        p.add_key_event("v", lambda: (state.__setitem__("seed_min_speed", _clip_value(state["seed_min_speed"] * 0.9, 0.0, 1.5)), rebuild(note="seed_vmin *0.9")))
        p.add_key_event("g", lambda: (state.__setitem__("seed_min_speed", _clip_value(max(state["seed_min_speed"], 0.001) * 1.1, 0.0, 1.5)), rebuild(note="seed_vmin *1.1")))
        p.add_key_event(",", lambda: (state.__setitem__("tube_radius", _clip_value(state["tube_radius"] * 0.9, 0.02, 3.0)), rebuild(note="tube *0.9")))
        p.add_key_event(".", lambda: (state.__setitem__("tube_radius", _clip_value(state["tube_radius"] * 1.1, 0.02, 3.0)), rebuild(note="tube *1.1")))
        p.add_key_event("o", lambda: (state.__setitem__("line_width", _clip_value(state["line_width"] * 0.9, 0.2, 8.0)), rebuild(note="line width *0.9")))
        p.add_key_event("p", lambda: (state.__setitem__("line_width", _clip_value(state["line_width"] * 1.1, 0.2, 8.0)), rebuild(note="line width *1.1")))
        p.add_key_event(";", lambda: (state.__setitem__("step", _clip_value(state["step"] * 0.9, 0.02, 3.0)), rebuild(note="step *0.9")))
        p.add_key_event("'", lambda: (state.__setitem__("step", _clip_value(state["step"] * 1.1, 0.02, 3.0)), rebuild(note="step *1.1")))
        p.add_key_event("k", lambda: (state.__setitem__("max_length", _clip_value(state["max_length"] * 0.9, 20.0, 3000.0)), rebuild(note="max_length *0.9")))
        p.add_key_event("l", lambda: (state.__setitem__("max_length", _clip_value(state["max_length"] * 1.1, 20.0, 3000.0)), rebuild(note="max_length *1.1")))
        p.add_key_event("i", lambda: (state.__setitem__("integration_direction", _cycle_direction(state["integration_direction"])), rebuild(note=f"direction -> {state['integration_direction']}")))
        p.add_key_event("f", lambda: (state.__setitem__("show_direction_arrows", not state["show_direction_arrows"]), rebuild(note="toggle arrows")))
        p.add_key_event("c", lambda: (state.__setitem__("color_streams", not state["color_streams"]), rebuild(note="toggle color")))
        p.add_key_event("b", lambda: (state.__setitem__("clip_invert", not state["clip_invert"]), rebuild(note="toggle clip invert")))
        p.add_key_event("h", lambda: (state.__setitem__("hud", not state["hud"]), rebuild(note="toggle HUD")))
        p.add_key_event("u", lambda: rebuild(reset_camera=True, note="camera reset"))

        if interactive_clip:
            p.add_box_widget(
                callback=on_box,
                bounds=state["clip_bounds"] if state["clip_bounds"] is not None else grid.bounds,
                rotation_enabled=bool(args.clip_widget_rotation),
                outline_translation=not bool(args.clip_widget_no_translation),
                interaction_event="end",
            )

        p.add_axes()
        rebuild(reset_camera=True)
        _show_or_save(p, args)
        return

    streams = _build_stream_mesh(grid, args)
    stream_geom = _stream_geometry(streams, args)
    clip_bounds = _parse_clip_bounds(args.clip_box_bounds)
    if clip_bounds is not None:
        stream_geom = stream_geom.clip_box(bounds=clip_bounds, invert=bool(args.clip_invert))

    scalar_name = args.scalar
    values = stream_geom.point_data[scalar_name] if scalar_name in stream_geom.point_data else np.array([])
    scalar_name, cmap, clim, scalar_title = _scalar_meta(args, values, scalar_name=scalar_name)
    color_streams = args.scalar != "speed" or args.stream_color_by_speed

    if interactive_clip:
        if color_streams and stream_geom.n_points > 0:
            p.add_mesh_clip_box(
                stream_geom,
                invert=bool(args.clip_invert),
                rotation_enabled=bool(args.clip_widget_rotation),
                outline_translation=not bool(args.clip_widget_no_translation),
                scalars=scalar_name,
                cmap=cmap,
                clim=clim,
                scalar_bar_args={"title": scalar_title},
            )
        elif stream_geom.n_points > 0:
            p.add_mesh_clip_box(
                stream_geom,
                invert=bool(args.clip_invert),
                rotation_enabled=bool(args.clip_widget_rotation),
                outline_translation=not bool(args.clip_widget_no_translation),
                color="deepskyblue",
            )
    else:
        base_kwargs = {"opacity": float(args.stream_opacity)}
        if args.stream_style == "line":
            base_kwargs["line_width"] = float(args.line_width)
        if color_streams and stream_geom.n_points > 0:
            p.add_mesh(
                stream_geom,
                scalars=scalar_name,
                cmap=cmap,
                clim=clim,
                scalar_bar_args=_stream_scalar_bar_args(
                    scalar_title, interactive=False, ui_color=hud_color
                ),
                **base_kwargs,
            )
        elif stream_geom.n_points > 0:
            p.add_mesh(stream_geom, color="deepskyblue", **base_kwargs)

    if args.show_direction_arrows:
        arrows = build_direction_arrows(streams, stride=args.arrow_stride, factor=args.arrow_factor)
        if arrows is not None:
            if clip_bounds is not None and not interactive_clip:
                arrows = arrows.clip_box(bounds=clip_bounds, invert=bool(args.clip_invert))
            p.add_mesh(arrows, color=arrow_color, opacity=0.9)

    bounds = [stream_geom.bounds] if stream_geom.n_points > 0 else [grid.bounds]
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
    if interactive_clip and args.roi_only_interaction:
        _apply_roi_only_style(p)

    p.add_axes()
    _show_or_save(p, args)


def render_panel(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    p = pv.Plotter(shape=(1, 3), off_screen=args.off_screen)
    p.set_background(args.background)
    slc = grid.slice(normal=args.slice_normal)
    _ensure_scalar_arrays(slc)
    scalar_name = args.scalar
    values = slc.point_data[scalar_name] if scalar_name in slc.point_data else np.array([])
    scalar_name, cmap, panel_clim, scalar_title = _scalar_meta(args, values)

    p.subplot(0, 0)
    p.add_text("Scalar Map", font_size=12)
    p.add_mesh(slc, scalars=scalar_name, cmap=cmap, clim=panel_clim, scalar_bar_args={"title": scalar_title})
    p.add_axes()

    p.subplot(0, 1)
    p.add_text("Velocity Map", font_size=12)
    p.add_mesh(slc, scalars=scalar_name, cmap=cmap, opacity=0.30, clim=panel_clim)
    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)
    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    glyph_scalar = scalar_name if scalar_name in glyphs.point_data else "speed"
    glyph_cmap = cmap if glyph_scalar == scalar_name else "turbo"
    p.add_mesh(glyphs, scalars=glyph_scalar, cmap=glyph_cmap, clim=panel_clim)
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

    streams = _build_stream_mesh(grid, args)
    stream_geom = _stream_geometry(streams, args)
    clip_bounds = _parse_clip_bounds(args.clip_box_bounds)
    if clip_bounds is not None:
        stream_geom = stream_geom.clip_box(bounds=clip_bounds, invert=bool(args.clip_invert))
    stream_values = stream_geom.point_data[scalar_name] if scalar_name in stream_geom.point_data else np.array([])
    _, _, stream_clim, _ = _scalar_meta(args, stream_values)
    color_streams = args.scalar != "speed" or args.stream_color_by_speed
    mesh_kwargs = {"opacity": float(args.stream_opacity)}
    if args.stream_style == "line":
        mesh_kwargs["line_width"] = float(args.line_width)
    if color_streams and stream_geom.n_points > 0:
        p.add_mesh(
            stream_geom,
            scalars=scalar_name,
            cmap=cmap,
            clim=stream_clim,
            scalar_bar_args={"title": scalar_title},
            **mesh_kwargs,
        )
    elif stream_geom.n_points > 0:
        p.add_mesh(stream_geom, color="deepskyblue", **mesh_kwargs)
    p.add_axes()

    _show_or_save(p, args)


def render_paper2(grid: pv.StructuredGrid, args: argparse.Namespace) -> None:
    """Two-panel layout: velocity magnitude + velocity map (glyphs)."""
    p = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1400, 750))
    p.set_background(args.background)
    slc = grid.slice(normal=args.slice_normal)
    _ensure_scalar_arrays(slc)
    scalar_name = args.scalar
    values = slc.point_data[scalar_name] if scalar_name in slc.point_data else np.array([])
    scalar_name, cmap, panel_clim, scalar_title = _scalar_meta(args, values)

    p.subplot(0, 0)
    p.add_text("Scalar map", font_size=16)
    p.add_mesh(slc, scalars=scalar_name, cmap=cmap, clim=panel_clim, scalar_bar_args={"title": scalar_title})
    p.add_axes()

    p.subplot(0, 1)
    p.add_text("Velocity map", font_size=16)
    p.add_mesh(slc, scalars=scalar_name, cmap=cmap, opacity=0.20, clim=panel_clim)
    stride = max(int(args.glyph_stride), 1)
    point_ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
    if "mask" in slc.point_data:
        point_ids = point_ids[slc.point_data["mask"][point_ids] > 0.5]
    if point_ids.size == 0:
        point_ids = np.array([0], dtype=np.int64)
    pts = slc.extract_points(point_ids, include_cells=False)
    glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))
    glyph_scalar = scalar_name if scalar_name in glyphs.point_data else "speed"
    glyph_cmap = cmap if glyph_scalar == scalar_name else "turbo"
    p.add_mesh(glyphs, scalars=glyph_scalar, cmap=glyph_cmap, clim=panel_clim)
    p.add_axes()

    _show_or_save(p, args)


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)
    _apply_paper_preset(args, has_mask=flow.mask is not None)
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
