#!/usr/bin/env python3
"""Interactive PyVista aneurysm sphere selector for CoW segmentations.

Displays the vessel surface in a 2×2 layout (3D view + three orthogonal
slices).  Left-click anywhere on the 3D surface to set the sphere centre;
use keyboard shortcuts to adjust the radius; press  s  to save the result.

Usage
-----
  python code/inference/select_aneurysm_roi.py \\
      --seg  data/paired_dataset/cow_segmentation_ens301_current/001_20240313/cow_seg_final.nii.gz \\
      [--bg  data/paired_dataset/hr_7t_in_3t_masked/001_20240313/input_mag_raw.nii.gz] \\
      [--out /tmp/aneurysm_001.nii.gz]

Keys
----
  left-click          pick sphere centre on the vessel surface
  = / +               grow sphere radius +1 mm
  - (minus)           shrink sphere radius -1 mm
  ] / [               grow / shrink radius by 0.5 mm
  s                   save aneurysm mask to --out (prints JSON summary)
  r                   reset camera (3D view only)
  h                   toggle HUD
  q / Escape          quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv

# flowviz lives in the sibling visualization package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualization"))
from flowviz.common import build_structured_grid, fit_camera_to_bounds, load_ras_canonical


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactively select an aneurysm ROI sphere from a CoW segmentation."
    )
    p.add_argument("--seg", required=True, help="Binary segmentation NIfTI (cow_seg_final.nii.gz).")
    p.add_argument("--bg", default="", help="Optional background magnitude NIfTI for slice textures.")
    p.add_argument(
        "--out", default="",
        help="Output path for aneurysm mask.  Default: <seg_dir>/aneurysm_roi.nii.gz.",
    )
    p.add_argument("--radius", type=float, default=5.0, help="Initial sphere radius in mm (default 5.0).")
    p.add_argument("--radius-step", type=float, default=1.0, help="Radius step per = / - key press (default 1.0 mm).")
    p.add_argument("--vessel-opacity", type=float, default=0.55)
    p.add_argument("--sphere-opacity", type=float, default=0.30)
    p.add_argument("--background", type=str, default="white")
    p.add_argument("--window-width", type=int, default=1600)
    p.add_argument("--window-height", type=int, default=900)
    p.add_argument("--camera-azimuth", type=float, default=35.0)
    p.add_argument("--camera-elevation", type=float, default=18.0)
    p.add_argument("--camera-distance-scale", type=float, default=1.5)
    p.add_argument("--camera-zoom", type=float, default=1.1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ui_text_color(background: str) -> str:
    try:
        rgb = np.array(pv.Color(background).float_rgb, dtype=np.float64)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    except Exception:
        luminance = 0.5
    return "black" if luminance >= 0.55 else "white"


def _load_seg(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Return (binary_mask uint8, affine float64, RAS-canonical image)."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return (data >= 0.5).astype(np.uint8), np.array(img.affine, dtype=np.float64), img


def _load_bg(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Load background magnitude as float32, crop to target_shape if needed."""
    data, _ = load_ras_canonical(path)
    if data.ndim == 4:
        data = data[..., 0]
    if data.shape[:3] != target_shape:
        tx = min(int(data.shape[0]), target_shape[0])
        ty = min(int(data.shape[1]), target_shape[1])
        tz = min(int(data.shape[2]), target_shape[2])
        data = data[:tx, :ty, :tz]
    return data.astype(np.float32)


def _build_grid(
    seg_mask: np.ndarray, affine: np.ndarray, bg: np.ndarray | None
) -> pv.StructuredGrid:
    zeros = np.zeros(seg_mask.shape, dtype=np.float32)
    mag = bg if bg is not None else zeros
    return build_structured_grid(mag, zeros, zeros, zeros, affine, mask_t=seg_mask)


def _make_ring(center: np.ndarray, radius: float, axis: str, n_pts: int = 72) -> pv.PolyData:
    """Circle ring in world space perpendicular to the given axis."""
    t = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    c0, c1, c2 = float(center[0]), float(center[1]), float(center[2])
    if axis == "x":
        pts = np.stack([np.full_like(t, c0), c1 + radius * np.cos(t), c2 + radius * np.sin(t)], axis=1)
    elif axis == "y":
        pts = np.stack([c0 + radius * np.cos(t), np.full_like(t, c1), c2 + radius * np.sin(t)], axis=1)
    else:
        pts = np.stack([c0 + radius * np.cos(t), c1 + radius * np.sin(t), np.full_like(t, c2)], axis=1)
    pd = pv.PolyData()
    pd.points = pts.astype(np.float32)
    n = len(pts)
    pd.lines = np.array([[2, i, (i + 1) % n] for i in range(n)], dtype=np.intp).ravel()
    return pd


def _compute_aneurysm_mask(
    seg_mask: np.ndarray,
    affine: np.ndarray,
    center_world: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    nx, ny, nz = seg_mask.shape
    i, j, k = np.mgrid[:nx, :ny, :nz]
    ijk1 = np.stack([i, j, k, np.ones_like(i)], axis=-1).astype(np.float64)
    xyz = (ijk1 @ affine.T)[..., :3]
    dist_sq = ((xyz - center_world) ** 2).sum(axis=-1)
    return np.logical_and(seg_mask > 0, dist_sq <= radius_mm ** 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Viewer state
# ---------------------------------------------------------------------------

@dataclass
class _State:
    center_world: np.ndarray
    radius_mm: float
    show_hud: bool = True
    camera_initialized: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class AneurysmSelector:
    def __init__(
        self,
        grid: pv.StructuredGrid,
        seg_mask: np.ndarray,
        affine: np.ndarray,
        ref_img: nib.Nifti1Image,
        out_path: Path,
        has_bg: bool,
        args: argparse.Namespace,
    ) -> None:
        self.grid = grid
        self.seg_mask = seg_mask
        self.affine = affine
        self.ref_img = ref_img
        self.out_path = out_path
        self.has_bg = has_bg
        self.args = args

        vessel = grid.contour(isosurfaces=[0.5], scalars="mask")
        if vessel.n_points == 0:
            raise ValueError("Mask contour is empty — check segmentation threshold.")
        self.vessel_surface = vessel

        self.state = _State(
            center_world=np.array(vessel.center, dtype=np.float64),
            radius_mm=float(args.radius),
        )
        self.actors: dict[str, object] = {}

        self.plotter = pv.Plotter(
            shape=(2, 2),
            window_size=(int(args.window_width), int(args.window_height)),
        )
        self.plotter.set_background(args.background)
        self.ui_color = _ui_text_color(args.background)

    # -----------------------------------------------------------------------
    # Actor management
    # -----------------------------------------------------------------------

    def _remove_actor(self, key: str) -> None:
        actor = self.actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor, reset_camera=False)

    def _sphere_bounds(self) -> tuple[float, float, float, float, float, float]:
        c, r = self.state.center_world, self.state.radius_mm
        return (c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r)

    def _safe_origin(self) -> list[float]:
        """Clip sphere centre to grid bounds so slice() always intersects."""
        c = self.state.center_world
        b = self.grid.bounds
        return [
            float(max(b[0], min(b[1], c[0]))),
            float(max(b[2], min(b[3], c[1]))),
            float(max(b[4], min(b[5], c[2]))),
        ]

    # -----------------------------------------------------------------------
    # HUD
    # -----------------------------------------------------------------------

    def _add_hud(self) -> None:
        self.plotter.subplot(0, 0)
        if not self.state.show_hud:
            self.plotter.add_text("", name="hud", position="upper_left")
            return
        c = self.state.center_world
        n_vox = int(_compute_aneurysm_mask(self.seg_mask, self.affine, c, self.state.radius_mm).sum())
        lines = [
            "Aneurysm ROI Selector",
            f"centre  ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm",
            f"radius  {self.state.radius_mm:.1f} mm   voxels {n_vox}",
            f"out  {self.out_path.name}",
            "left-click: centre  |  = / - : radius  |  s: save  |  r: camera  |  h: HUD",
        ]
        if self.state.note:
            lines.append(self.state.note)
        self.plotter.add_text(
            "\n".join(lines), name="hud", position="upper_left", font_size=9, color=self.ui_color
        )

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _update_3d(self) -> None:
        self.plotter.subplot(0, 0)
        for key in ("vessel_ctx", "sphere", "roi_hl"):
            self._remove_actor(key)

        self.actors["vessel_ctx"] = self.plotter.add_mesh(
            self.vessel_surface,
            color="lightgray",
            opacity=float(self.args.vessel_opacity),
            smooth_shading=True,
        )

        sphere_poly = pv.Sphere(
            center=self.state.center_world.tolist(),
            radius=self.state.radius_mm,
            theta_resolution=30,
            phi_resolution=30,
        )
        self.actors["sphere"] = self.plotter.add_mesh(
            sphere_poly,
            color="orange",
            opacity=float(self.args.sphere_opacity),
            smooth_shading=True,
        )

        roi_clip = self.vessel_surface.clip_box(bounds=self._sphere_bounds(), invert=False)
        if roi_clip.n_points > 0:
            self.actors["roi_hl"] = self.plotter.add_mesh(
                roi_clip, color="tomato", opacity=0.92, smooth_shading=True
            )

        self.plotter.add_text(
            "3D – Aneurysm ROI  (left-click to set centre)",
            position="upper_edge", font_size=10, name="title_3d", color=self.ui_color,
        )
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

        self._add_hud()

    def _update_slice(self, subplot_rc: tuple[int, int], axis: str, label: str) -> None:
        self.plotter.subplot(subplot_rc[0], subplot_rc[1])
        self._remove_actor(f"slice_{axis}")
        self._remove_actor(f"ring_{axis}")

        origin = self._safe_origin()
        slc = self.grid.slice(normal=axis, origin=origin)
        if slc.n_points > 0:
            scalar = "mag" if self.has_bg else "mask"
            cmap = "gray" if self.has_bg else "binary"
            self.actors[f"slice_{axis}"] = self.plotter.add_mesh(
                slc, scalars=scalar, cmap=cmap, show_scalar_bar=False, opacity=0.9,
            )

        ring = _make_ring(self.state.center_world, self.state.radius_mm, axis)
        self.actors[f"ring_{axis}"] = self.plotter.add_mesh(
            ring, color="orange", line_width=3, style="wireframe",
        )

        self.plotter.add_text(label, position="upper_edge", font_size=10, name=f"title_{axis}", color=self.ui_color)
        self.plotter.add_axes()

        if axis == "x":
            self.plotter.view_yz()
        elif axis == "y":
            self.plotter.view_xz()
        else:
            self.plotter.view_xy()
        try:
            self.plotter.enable_parallel_projection()
        except Exception:
            pass
        self.plotter.reset_camera()

    def _rebuild(self, reset_camera: bool = False, note: str = "") -> None:
        if reset_camera:
            self.state.camera_initialized = False
        self.state.note = note
        self._update_3d()
        self._update_slice((0, 1), "x", "Slice YZ  (Sagittal)")
        self._update_slice((1, 0), "y", "Slice XZ  (Coronal)")
        self._update_slice((1, 1), "z", "Slice XY  (Axial)")
        self.plotter.render()

    # -----------------------------------------------------------------------
    # Interactions
    # -----------------------------------------------------------------------

    def _on_pick(self, *args) -> None:
        """Accept both (ndarray,) and (PolyData,) callback signatures."""
        pt = None
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.ndim == 1 and len(arg) >= 3:
                pt = arg[:3].astype(np.float64)
                break
            if hasattr(arg, "points") and hasattr(arg, "n_points") and arg.n_points > 0:
                pt = np.asarray(arg.points[0], dtype=np.float64)
                break
            if isinstance(arg, (list, tuple)) and len(arg) >= 3:
                try:
                    pt = np.array(arg[:3], dtype=np.float64)
                    break
                except Exception:
                    pass
        if pt is None:
            return
        self.state.center_world = pt
        self._rebuild(note=f"centre → ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")

    def _adjust_radius(self, delta: float) -> None:
        self.state.radius_mm = max(0.5, self.state.radius_mm + delta)
        self._rebuild(note=f"radius → {self.state.radius_mm:.1f} mm")

    def _save(self) -> None:
        aneurysm = _compute_aneurysm_mask(
            self.seg_mask, self.affine, self.state.center_world, self.state.radius_mm
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = nib.Nifti1Image(aneurysm, self.ref_img.affine, self.ref_img.header)
        nib.save(out_img, str(self.out_path))
        c = self.state.center_world
        info = {
            "out_path": str(self.out_path),
            "center_world_mm": [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)],
            "radius_mm": round(float(self.state.radius_mm), 3),
            "foreground_voxels": int(aneurysm.sum()),
        }
        print(json.dumps(info, indent=2))
        self._rebuild(note=f"SAVED  ({info['foreground_voxels']} voxels)")

    def _toggle_hud(self) -> None:
        self.state.show_hud = not self.state.show_hud
        self._rebuild(note="")

    def _register_interactions(self) -> None:
        self.plotter.subplot(0, 0)
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            use_mesh=False,
            show_message=True,
            left_clicking=True,
        )

        step = float(self.args.radius_step)
        self.plotter.add_key_event("=", lambda: self._adjust_radius(+step))
        self.plotter.add_key_event("+", lambda: self._adjust_radius(+step))
        self.plotter.add_key_event("-", lambda: self._adjust_radius(-step))
        self.plotter.add_key_event("]", lambda: self._adjust_radius(+0.5))
        self.plotter.add_key_event("[", lambda: self._adjust_radius(-0.5))
        self.plotter.add_key_event("s", self._save)
        self.plotter.add_key_event("r", lambda: self._rebuild(reset_camera=True, note="camera reset"))
        self.plotter.add_key_event("h", self._toggle_hud)

    def show(self) -> None:
        self._register_interactions()
        self._rebuild(reset_camera=True, note="ready — left-click to set centre")
        self.plotter.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    seg_path = Path(args.seg).expanduser().resolve()
    if not seg_path.is_file():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else seg_path.parent / "aneurysm_roi.nii.gz"
    )

    seg_mask, affine, ref_img = _load_seg(seg_path)
    if seg_mask.sum() == 0:
        raise ValueError("Segmentation mask is empty after thresholding.")

    bg: np.ndarray | None = None
    if args.bg:
        bg_path = Path(args.bg).expanduser().resolve()
        if bg_path.is_file():
            bg = _load_bg(bg_path, seg_mask.shape[:3])
            print(f"Background loaded: {bg_path}")
        else:
            print(f"Warning: background not found, continuing without it: {bg_path}")

    print(f"Segmentation: {seg_path}  ({int(seg_mask.sum())} fg voxels)")
    print(f"Output will be written to: {out_path}")

    grid = _build_grid(seg_mask, affine, bg)

    selector = AneurysmSelector(
        grid=grid,
        seg_mask=seg_mask,
        affine=affine,
        ref_img=ref_img,
        out_path=out_path,
        has_bg=(bg is not None),
        args=args,
    )
    selector.show()


if __name__ == "__main__":
    main()
