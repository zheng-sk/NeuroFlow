#!/usr/bin/env python3
"""Interactive ROI + orthogonal slice QC for velocity sign and direction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from neuroflow.visualization.flowviz.common import (
    add_common_preproc_args,
    add_input_args,
    compute_speed_clim,
    fit_camera_to_bounds,
    load_flow_data_from_args,
    make_frame_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive QC viewer with one 3D ROI view and three synchronized 2D slices "
            "(X/Y/Z) to validate velocity sign (Vx/Vy/Vz) inside a selected region."
        )
    )
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument("--scalar", choices=["speed", "vx", "vy", "vz"], default="vx")
    parser.add_argument("--signed-percentile", type=float, default=99.0)
    parser.add_argument("--speed-vmin", type=float, default=None)
    parser.add_argument("--speed-vmax", type=float, default=None)
    parser.add_argument("--component-cmap", type=str, default="coolwarm")
    parser.add_argument("--speed-cmap", type=str, default="turbo")

    parser.add_argument("--vessel-opacity", type=float, default=0.95)
    parser.add_argument("--context-opacity", type=float, default=0.12)
    parser.add_argument("--slice-opacity", type=float, default=1.0)
    parser.add_argument("--show-slice-glyphs", action="store_true")
    parser.add_argument("--glyph-stride", type=int, default=35)
    parser.add_argument("--glyph-factor", type=float, default=0.9)

    parser.add_argument(
        "--clip-box-bounds",
        type=str,
        default=None,
        help="ROI bounds as x0,x1,y0,y1,z0,z1 in world coordinates (mm).",
    )
    parser.add_argument("--interactive-clip-box", action="store_true")
    parser.add_argument("--clip-widget-rotation", action="store_true")
    parser.add_argument("--clip-widget-no-translation", action="store_true")
    parser.add_argument("--start-without-clip", action="store_true", help="Start with clip disabled (toggle with key 'c').")

    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.45)
    parser.add_argument("--camera-zoom", type=float, default=1.15)

    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=1100)
    parser.add_argument("--background", type=str, default="white")
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


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


def _expanded_bounds(
    bounds: tuple[float, float, float, float, float, float],
    factor: float = 1.05,
) -> tuple[float, float, float, float, float, float]:
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    hx = 0.5 * max(bounds[1] - bounds[0], 1e-3) * float(factor)
    hy = 0.5 * max(bounds[3] - bounds[2], 1e-3) * float(factor)
    hz = 0.5 * max(bounds[5] - bounds[4], 1e-3) * float(factor)
    return (cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


def _clip_value(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _ui_text_color(background: str) -> str:
    try:
        rgb = np.array(pv.Color(background).float_rgb, dtype=np.float64)
        luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    except Exception:
        luminance = 0.5
    return "black" if luminance >= 0.55 else "white"


def _ensure_component_arrays(dataset: pv.DataSet) -> None:
    if "vel" not in dataset.point_data:
        return
    vel = np.asarray(dataset.point_data["vel"])
    if vel.ndim != 2 or vel.shape[1] < 3:
        return
    if "vx" not in dataset.point_data:
        dataset.point_data["vx"] = vel[:, 0]
    if "vy" not in dataset.point_data:
        dataset.point_data["vy"] = vel[:, 1]
    if "vz" not in dataset.point_data:
        dataset.point_data["vz"] = vel[:, 2]
    if "speed" not in dataset.point_data:
        dataset.point_data["speed"] = np.linalg.norm(vel[:, :3], axis=1)


@dataclass
class ViewerState:
    scalar: str
    clip_bounds: tuple[float, float, float, float, float, float]
    clip_enabled: bool = True
    show_hud: bool = True
    camera_initialized: bool = False
    box_visible: bool = True


class RoiSignViewer:
    def __init__(self, grid: pv.StructuredGrid, args: argparse.Namespace):
        if "mask" not in grid.point_data:
            raise ValueError("ROI sign QC requires a vessel mask. Provide --mask <mask.nii.gz>.")

        self.grid = grid
        self.args = args
        _ensure_component_arrays(self.grid)

        self.plotter = pv.Plotter(
            shape=(2, 2),
            off_screen=bool(args.off_screen),
            window_size=(int(args.window_width), int(args.window_height)),
        )
        self.plotter.set_background(args.background)
        self.ui_color = _ui_text_color(args.background)

        vessel = self.grid.contour(isosurfaces=[0.5], scalars="mask")
        if vessel.n_points == 0:
            raise ValueError("Mask contour is empty. Verify mask threshold/alignment.")
        self.vessel_surface = vessel.sample(self.grid)
        _ensure_component_arrays(self.vessel_surface)

        default_bounds = _parse_clip_bounds(args.clip_box_bounds)
        if default_bounds is None:
            default_bounds = _expanded_bounds(self.vessel_surface.bounds, factor=1.03)

        self.state = ViewerState(
            scalar=str(args.scalar),
            clip_bounds=default_bounds,
            clip_enabled=not bool(args.start_without_clip),
        )

        self.actors: dict[str, object] = {}
        self.box_widget = None

    def _active_bounds(self) -> tuple[float, float, float, float, float, float]:
        if self.state.clip_enabled:
            return self.state.clip_bounds
        return _expanded_bounds(self.vessel_surface.bounds, factor=1.0)

    def _center_from_bounds(self, bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
        return (
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        )

    def _scalar_meta(self, scalar: str) -> tuple[str, tuple[float, float], str]:
        mask = np.asarray(self.grid.point_data["mask"]) > 0.5
        if scalar == "speed":
            values = np.asarray(self.grid.point_data["speed"])[mask]
            provided = None
            if self.args.speed_vmin is not None and self.args.speed_vmax is not None:
                provided = (float(self.args.speed_vmin), float(self.args.speed_vmax))
            clim = compute_speed_clim(values, provided=provided, q_low=2.0, q_high=98.5)
            return (self.args.speed_cmap, clim, "Speed (m/s)")

        values = np.asarray(self.grid.point_data[scalar])[mask]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(np.abs(finite), float(self.args.signed_percentile)))
            vmax = max(vmax, 1e-6)
        title = {"vx": "Vx (m/s)", "vy": "Vy (m/s)", "vz": "Vz (m/s)"}[scalar]
        return (self.args.component_cmap, (-vmax, vmax), title)

    def _remove_actor(self, key: str) -> None:
        actor = self.actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor, reset_camera=False)

    def _extract_roi_surface(self) -> pv.PolyData:
        surf = self.vessel_surface
        if self.state.clip_enabled:
            surf = surf.clip_box(bounds=self.state.clip_bounds, invert=False)
        _ensure_component_arrays(surf)
        return surf

    def _extract_slice(self, normal: str) -> pv.DataSet:
        bounds = self._active_bounds()
        center = self._center_from_bounds(bounds)
        slc = self.grid.slice(normal=normal, origin=center)
        if self.state.clip_enabled:
            slc = slc.clip_box(bounds=bounds, invert=False)
        _ensure_component_arrays(slc)
        if slc.n_points > 0 and "mask" in slc.point_data:
            masked = slc.threshold([0.5, 1.0], scalars="mask")
            if masked.n_points > 0:
                slc = masked
        return slc

    def _slice_glyphs(self, slc: pv.DataSet) -> pv.PolyData | None:
        if not bool(self.args.show_slice_glyphs):
            return None
        if slc.n_points == 0 or "vel" not in slc.point_data:
            return None
        stride = max(1, int(self.args.glyph_stride))
        ids = np.arange(0, slc.n_points, stride, dtype=np.int64)
        if ids.size == 0:
            return None
        pts = slc.extract_points(ids, include_cells=False)
        if pts.n_points == 0:
            return None
        return pts.glyph(orient="vel", scale=False, factor=float(self.args.glyph_factor))

    def _add_hud(self, note: str = "") -> None:
        self.plotter.subplot(0, 0)
        if not self.state.show_hud:
            self.plotter.add_text("", name="roi_hud", position="upper_left")
            return
        b = self.state.clip_bounds
        center = self._center_from_bounds(b)
        lines = [
            "ROI Sign QC",
            f"scalar={self.state.scalar} clip={'on' if self.state.clip_enabled else 'off'} box={'on' if self.state.box_visible else 'off'}",
            f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm",
            f"bounds={b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f},{b[4]:.1f},{b[5]:.1f}",
            "Keys: 1/2/3/4 = speed/vx/vy/vz | c clip | b box | h HUD | r reset camera",
        ]
        if note:
            lines.append(note)
        self.plotter.add_text("\n".join(lines), name="roi_hud", position="upper_left", font_size=10, color=self.ui_color)

    def _update_3d(self, note: str = "") -> None:
        cmap, clim, title = self._scalar_meta(self.state.scalar)
        roi = self._extract_roi_surface()

        self.plotter.subplot(0, 0)
        self._remove_actor("ctx")
        self._remove_actor("roi")

        self.actors["ctx"] = self.plotter.add_mesh(
            self.vessel_surface,
            color="lightgray",
            opacity=float(_clip_value(self.args.context_opacity, 0.0, 1.0)),
            smooth_shading=True,
        )

        if roi.n_points > 0:
            self.actors["roi"] = self.plotter.add_mesh(
                roi,
                scalars=self.state.scalar,
                cmap=cmap,
                clim=clim,
                opacity=float(_clip_value(self.args.vessel_opacity, 0.0, 1.0)),
                smooth_shading=True,
                scalar_bar_args={
                    "title": title,
                    "vertical": True,
                    "position_x": 0.86,
                    "position_y": 0.25,
                    "width": 0.10,
                    "height": 0.60,
                    "title_font_size": 10,
                    "label_font_size": 9,
                    "color": self.ui_color,
                },
            )

        self.plotter.add_text("3D ROI", position="upper_edge", font_size=11, name="title_3d", color=self.ui_color)
        self.plotter.add_axes()

        if not self.state.camera_initialized:
            fit_camera_to_bounds(
                plotter=self.plotter,
                bounds=self.vessel_surface.bounds,
                azimuth_deg=self.args.camera_azimuth,
                elevation_deg=self.args.camera_elevation,
                distance_scale=self.args.camera_distance_scale,
                zoom=self.args.camera_zoom,
            )
            self.state.camera_initialized = True

        self._add_hud(note)

    def _apply_slice_camera(self, normal: str) -> None:
        if normal == "x":
            self.plotter.view_yz()
        elif normal == "y":
            self.plotter.view_xz()
        else:
            self.plotter.view_xy()
        try:
            self.plotter.enable_parallel_projection()
        except Exception:
            pass
        self.plotter.reset_camera()

    def _update_slice(self, subplot_rc: tuple[int, int], normal: str, label: str) -> None:
        cmap, clim, title = self._scalar_meta(self.state.scalar)
        slc = self._extract_slice(normal=normal)

        key_main = f"slice_{normal}"
        key_glyph = f"slice_glyph_{normal}"

        self.plotter.subplot(subplot_rc[0], subplot_rc[1])
        self._remove_actor(key_main)
        self._remove_actor(key_glyph)

        if slc.n_points > 0:
            self.actors[key_main] = self.plotter.add_mesh(
                slc,
                scalars=self.state.scalar,
                cmap=cmap,
                clim=clim,
                opacity=float(_clip_value(self.args.slice_opacity, 0.0, 1.0)),
                show_scalar_bar=False,
            )
            glyphs = self._slice_glyphs(slc)
            if glyphs is not None and glyphs.n_points > 0:
                self.actors[key_glyph] = self.plotter.add_mesh(
                    glyphs,
                    color="black" if self.ui_color == "black" else "white",
                    opacity=0.85,
                )

        self.plotter.add_text(f"{label} ({title})", position="upper_edge", font_size=11, name=f"title_{normal}", color=self.ui_color)
        self.plotter.add_axes()
        self._apply_slice_camera(normal)

    def _rebuild(self, note: str = "", reset_camera: bool = False) -> None:
        if reset_camera:
            self.state.camera_initialized = False
        self._update_3d(note=note)
        self._update_slice((0, 1), "x", "Slice X")
        self._update_slice((1, 0), "y", "Slice Y")
        self._update_slice((1, 1), "z", "Slice Z")
        self.plotter.render()

    def _set_scalar(self, name: str) -> None:
        self.state.scalar = name
        self._rebuild(note=f"scalar -> {name}")

    def _toggle_clip(self) -> None:
        self.state.clip_enabled = not self.state.clip_enabled
        self._rebuild(note=f"clip -> {'on' if self.state.clip_enabled else 'off'}")

    def _toggle_hud(self) -> None:
        self.state.show_hud = not self.state.show_hud
        self._rebuild(note="HUD toggled")

    def _toggle_box_visibility(self) -> None:
        if self.box_widget is None:
            return
        self.state.box_visible = not self.state.box_visible
        if self.state.box_visible:
            self.box_widget.On()
        else:
            self.box_widget.Off()
        self._rebuild(note=f"box -> {'on' if self.state.box_visible else 'off'}")

    def _on_box_updated(self, poly: pv.PolyData) -> None:
        self.state.clip_bounds = tuple(float(v) for v in poly.bounds)
        self._rebuild(note="ROI updated")

    def _register_interactions(self) -> None:
        self.plotter.add_key_event("1", lambda: self._set_scalar("speed"))
        self.plotter.add_key_event("2", lambda: self._set_scalar("vx"))
        self.plotter.add_key_event("3", lambda: self._set_scalar("vy"))
        self.plotter.add_key_event("4", lambda: self._set_scalar("vz"))
        self.plotter.add_key_event("c", self._toggle_clip)
        self.plotter.add_key_event("h", self._toggle_hud)
        self.plotter.add_key_event("b", self._toggle_box_visibility)
        self.plotter.add_key_event("r", lambda: self._rebuild(note="camera reset", reset_camera=True))

        # Fallback zoom shortcuts when touchpad zoom is not captured by VTK.
        self.plotter.add_key_event("=", lambda: (self.plotter.camera.Zoom(1.15), self.plotter.render()))
        self.plotter.add_key_event("z", lambda: (self.plotter.camera.Zoom(1.15), self.plotter.render()))
        self.plotter.add_key_event("-", lambda: (self.plotter.camera.Zoom(1.0 / 1.15), self.plotter.render()))
        self.plotter.add_key_event("x", lambda: (self.plotter.camera.Zoom(1.0 / 1.15), self.plotter.render()))

        if self.args.interactive_clip_box and not self.args.off_screen and self.args.screenshot is None:
            self.plotter.subplot(0, 0)
            self.box_widget = self.plotter.add_box_widget(
                callback=self._on_box_updated,
                bounds=self.state.clip_bounds,
                rotation_enabled=bool(self.args.clip_widget_rotation),
                outline_translation=not bool(self.args.clip_widget_no_translation),
                interaction_event="end",
            )
            self.state.box_visible = True

    def show(self) -> None:
        self._register_interactions()
        self._rebuild(note="ready", reset_camera=True)

        if self.args.screenshot is not None:
            self.args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            self.plotter.show(auto_close=False)
            self.plotter.screenshot(str(self.args.screenshot))
            self.plotter.close()
            print(f"ROI sign QC image written to: {self.args.screenshot.resolve()}")
        else:
            self.plotter.show()


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)
    grid, _ = make_frame_grid(flow, args.frame, zero_velocity_below=args.zero_velocity_below)

    viewer = RoiSignViewer(grid=grid, args=args)
    viewer.show()


if __name__ == "__main__":
    main()
