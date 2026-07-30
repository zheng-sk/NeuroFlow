#!/usr/bin/env python3
"""Interactive PyVista aneurysm box selector for CoW segmentations.

Displays a large 3D vessel surface with an interactive box widget and three
smaller orthogonal slices on the right. Drag the box handles around the
aneurysm; press  s  to save the selected mask.

Usage
-----
  python code/inference/select_aneurysm_roi.py \\
      --seg  data/paired_dataset/cow_segmentation_ens301_current/subject_001/cow_seg_final.nii.gz \\
      [--bg  data/paired_dataset/hr_7t_in_3t_masked/subject_001/input_mag_raw.nii.gz] \\
      [--out /tmp/aneurysm_001.nii.gz]

Keys
----
  drag box handles    move/resize selected aneurysm ROI
  u                   update slice previews after moving the box
  s                   save aneurysm mask to --out (prints JSON summary)
  r                   reset camera (3D view only)
  h                   toggle HUD
  q / Escape          quit without saving
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


def _configure_pyvista_env() -> None:
    """Set environment variables before importing PyVista for desktop selection."""
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    os.environ.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
    os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_PREFIX", "")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/neuroflow_matplotlib")


_configure_pyvista_env()

try:
    import pyvista as pv
except Exception as exc:
    raise RuntimeError(
        "select_aneurysm_roi requires pyvista. Install the visualization extra "
        'with:  pip install -e ".[viz]"'
    ) from exc

pv.OFF_SCREEN = False
if hasattr(pv, "set_jupyter_backend"):
    try:
        pv.set_jupyter_backend(None)
    except Exception:
        pass

# flowviz lives in the sibling visualization package
from neuroflow.visualization.flowviz.common import build_structured_grid, fit_camera_to_bounds, load_ras_canonical


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactively select an aneurysm ROI box from a CoW segmentation."
    )
    p.add_argument("--seg", required=True, help="Binary segmentation NIfTI (cow_seg_final.nii.gz).")
    p.add_argument("--bg", default="", help="Optional background magnitude NIfTI for slice textures.")
    p.add_argument(
        "--out", default="",
        help="Output path for aneurysm mask.  Default: <seg_dir>/aneurysm_roi.nii.gz.",
    )
    p.add_argument("--box-size", type=float, default=10.0, help="Initial cubic box side length in mm (default 10.0).")
    p.add_argument("--vessel-opacity", type=float, default=0.55)
    p.add_argument("--roi-opacity", type=float, default=0.92)
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
    """Load background magnitude as float32, crop/pad to target_shape if needed."""
    data, _ = load_ras_canonical(path)
    if data.ndim == 4:
        data = data[..., 0]
    if data.shape[:3] != target_shape:
        out = np.zeros(target_shape, dtype=np.float32)
        tx = min(int(data.shape[0]), int(target_shape[0]))
        ty = min(int(data.shape[1]), int(target_shape[1]))
        tz = min(int(data.shape[2]), int(target_shape[2]))
        out[:tx, :ty, :tz] = data[:tx, :ty, :tz]
        data = out
    return data.astype(np.float32)


def _build_grid(
    seg_mask: np.ndarray, affine: np.ndarray, bg: np.ndarray | None
) -> pv.StructuredGrid:
    zeros = np.zeros(seg_mask.shape, dtype=np.float32)
    mag = bg if bg is not None else zeros
    return build_structured_grid(mag, zeros, zeros, zeros, affine, mask_t=seg_mask)


def _center_from_bounds(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=np.float64,
    )


def _clip_bounds_to_grid(
    bounds: tuple[float, float, float, float, float, float],
    grid_bounds: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    x0, x1, y0, y1, z0, z1 = [float(v) for v in bounds]
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if z0 > z1:
        z0, z1 = z1, z0
    return (
        max(grid_bounds[0], min(grid_bounds[1], x0)),
        max(grid_bounds[0], min(grid_bounds[1], x1)),
        max(grid_bounds[2], min(grid_bounds[3], y0)),
        max(grid_bounds[2], min(grid_bounds[3], y1)),
        max(grid_bounds[4], min(grid_bounds[5], z0)),
        max(grid_bounds[4], min(grid_bounds[5], z1)),
    )


def _compute_aneurysm_mask(
    seg_mask: np.ndarray,
    affine: np.ndarray,
    bounds_world: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    nx, ny, nz = seg_mask.shape
    i, j, k = np.mgrid[:nx, :ny, :nz]
    ijk1 = np.stack([i, j, k, np.ones_like(i)], axis=-1).astype(np.float64)
    xyz = (ijk1 @ affine.T)[..., :3]
    x0, x1, y0, y1, z0, z1 = bounds_world
    inside = (
        (xyz[..., 0] >= x0)
        & (xyz[..., 0] <= x1)
        & (xyz[..., 1] >= y0)
        & (xyz[..., 1] <= y1)
        & (xyz[..., 2] >= z0)
        & (xyz[..., 2] <= z1)
    )
    return np.logical_and(seg_mask > 0, inside).astype(np.uint8)


def _mask_centroid_world(seg_mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords = np.argwhere(seg_mask > 0)
    if coords.size == 0:
        raise ValueError("Cannot compute centroid of an empty segmentation.")
    centroid_ijk = coords.astype(np.float64).mean(axis=0)
    return (np.append(centroid_ijk, 1.0) @ affine.T)[:3].astype(np.float64)


# ---------------------------------------------------------------------------
# Viewer state
# ---------------------------------------------------------------------------

@dataclass
class _State:
    bounds_world: tuple[float, float, float, float, float, float]
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

        center = _mask_centroid_world(seg_mask, affine)
        half = 0.5 * max(float(args.box_size), 0.5)
        initial_bounds = (
            center[0] - half,
            center[0] + half,
            center[1] - half,
            center[1] + half,
            center[2] - half,
            center[2] + half,
        )
        self.state = _State(bounds_world=_clip_bounds_to_grid(initial_bounds, grid.bounds))
        self.actors: dict[str, object] = {}
        self.box_widget = None

        self.plotter = pv.Plotter(
            shape="1|3",
            notebook=False,
            off_screen=False,
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

    def _safe_origin(self) -> list[float]:
        """Clip box centre to grid bounds so slice() always intersects."""
        c = _center_from_bounds(self.state.bounds_world)
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
        self.plotter.subplot(0)
        if not self.state.show_hud:
            self.plotter.add_text("", name="hud", position="upper_left")
            return
        b = self.state.bounds_world
        c = _center_from_bounds(b)
        n_vox = int(_compute_aneurysm_mask(self.seg_mask, self.affine, b).sum())
        dims = (b[1] - b[0], b[3] - b[2], b[5] - b[4])
        lines = [
            "Aneurysm ROI Selector",
            f"centre  ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm",
            f"box mm  ({dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f})   voxels {n_vox}",
            f"out  {self.out_path.name}",
            "drag box handles  |  u: update views  |  s: save  |  h: HUD  |  q: quit",
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
        self.plotter.subplot(0)
        for key in ("vessel_ctx", "roi_hl"):
            self._remove_actor(key)

        self.actors["vessel_ctx"] = self.plotter.add_mesh(
            self.vessel_surface,
            color="lightgray",
            opacity=float(self.args.vessel_opacity),
            smooth_shading=True,
        )

        roi_clip = self.vessel_surface.clip_box(bounds=self.state.bounds_world, invert=False)
        if roi_clip.n_points > 0:
            self.actors["roi_hl"] = self.plotter.add_mesh(
                roi_clip, color="tomato", opacity=float(self.args.roi_opacity), smooth_shading=True
            )

        self.plotter.add_text(
            "3D Aneurysm ROI Box",
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

    def _update_slice(self, subplot_idx: int, axis: str, label: str) -> None:
        self.plotter.subplot(subplot_idx)
        self._remove_actor(f"slice_{axis}")
        self._remove_actor(f"slice_roi_{axis}")

        origin = self._safe_origin()
        slc = self.grid.slice(normal=axis, origin=origin)
        if slc.n_points > 0:
            scalar = "mag" if self.has_bg else "mask"
            cmap = "gray" if self.has_bg else "binary"
            self.actors[f"slice_{axis}"] = self.plotter.add_mesh(
                slc, scalars=scalar, cmap=cmap, show_scalar_bar=False, opacity=0.9,
            )

            roi_slc = slc.clip_box(bounds=self.state.bounds_world, invert=False)
            if roi_slc.n_points > 0:
                self.actors[f"slice_roi_{axis}"] = self.plotter.add_mesh(
                    roi_slc, color="tomato", opacity=0.55, show_scalar_bar=False,
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
        self._update_slice(1, "x", "Slice YZ  (Sagittal)")
        self._update_slice(2, "y", "Slice XZ  (Coronal)")
        self._update_slice(3, "z", "Slice XY  (Axial)")
        self.plotter.render()

    def _refresh_previews(self, note: str = "") -> None:
        self.state.note = note
        self._add_hud()
        self._update_slice(1, "x", "Slice YZ  (Sagittal)")
        self._update_slice(2, "y", "Slice XZ  (Coronal)")
        self._update_slice(3, "z", "Slice XY  (Axial)")
        self.plotter.render()

    def _reset_3d_camera(self) -> None:
        self.plotter.subplot(0)
        fit_camera_to_bounds(
            plotter=self.plotter,
            bounds=self.vessel_surface.bounds,
            azimuth_deg=self.args.camera_azimuth,
            elevation_deg=self.args.camera_elevation,
            distance_scale=self.args.camera_distance_scale,
            zoom=self.args.camera_zoom,
        )
        self.state.note = "camera reset"
        self._add_hud()
        self.plotter.render()

    # -----------------------------------------------------------------------
    # Interactions
    # -----------------------------------------------------------------------

    def _on_box_updated(self, poly: pv.PolyData) -> None:
        self.state.bounds_world = _clip_bounds_to_grid(tuple(float(v) for v in poly.bounds), self.grid.bounds)
        self.state.note = "ROI box updated - press u to refresh slice previews or s to save"

    def _save(self) -> None:
        aneurysm = _compute_aneurysm_mask(self.seg_mask, self.affine, self.state.bounds_world)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = nib.Nifti1Image(aneurysm, self.ref_img.affine, self.ref_img.header)
        nib.save(out_img, str(self.out_path))
        b = self.state.bounds_world
        c = _center_from_bounds(b)
        info = {
            "out_path": str(self.out_path),
            "source_seg_path": str(Path(self.args.seg).expanduser().resolve()),
            "center_world_mm": [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)],
            "bounds_world_mm": [round(float(v), 3) for v in b],
            "box_size_mm": [round(float(b[1] - b[0]), 3), round(float(b[3] - b[2]), 3), round(float(b[5] - b[4]), 3)],
            "foreground_voxels": int(aneurysm.sum()),
        }
        sidecar_path = self.out_path.with_suffix("")
        if sidecar_path.suffix == ".nii":
            sidecar_path = sidecar_path.with_suffix("")
        sidecar_path = sidecar_path.with_name(sidecar_path.name + "_selection.json")
        sidecar_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        info["selection_json"] = str(sidecar_path)
        print(json.dumps(info, indent=2))
        self.state.note = f"SAVED  ({info['foreground_voxels']} voxels)"

    def _toggle_hud(self) -> None:
        self.state.show_hud = not self.state.show_hud
        self._add_hud()
        self.plotter.render()

    def _register_interactions(self) -> None:
        self.plotter.subplot(0)
        self.box_widget = self.plotter.add_box_widget(
            callback=self._on_box_updated,
            bounds=self.state.bounds_world,
            factor=1.0,
            rotation_enabled=False,
            color="yellow",
            use_planes=False,
            outline_translation=True,
            interaction_event="end",
        )
        self.plotter.add_key_event("s", self._save)
        self.plotter.add_key_event("u", lambda: self._refresh_previews(note="views refreshed"))
        self.plotter.add_key_event("r", self._reset_3d_camera)
        self.plotter.add_key_event("h", self._toggle_hud)
        self.plotter.add_key_event("q", lambda: self.plotter.iren.terminate_app())
        self.plotter.add_key_event("Escape", lambda: self.plotter.iren.terminate_app())

    def show(self) -> None:
        self._rebuild(reset_camera=True, note="ready - drag the yellow box around the aneurysm")
        self._register_interactions()
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
