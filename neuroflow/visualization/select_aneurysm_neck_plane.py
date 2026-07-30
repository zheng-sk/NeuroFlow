#!/usr/bin/env python3
"""Interactive PyVista neck-plane selector for aneurysm separation.

This tool is intended for a rough ROI that includes the aneurysm and a small
part of the connecting parent artery. Move/rotate the plane to the neck, flip
the kept side if needed, then save an aneurysm-only mask plus a neck JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label, marching_cubes, regionprops


def _configure_pyvista_env() -> None:
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    os.environ.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
    os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_PREFIX", "")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/neuroflow_matplotlib")


_configure_pyvista_env()

try:
    import pyvista as pv
except Exception as exc:
    raise RuntimeError(
        "select_aneurysm_neck_plane.py requires pyvista in the active environment."
    ) from exc

pv.OFF_SCREEN = False
if hasattr(pv, "set_jupyter_backend"):
    try:
        pv.set_jupyter_backend(None)
    except Exception:
        pass

from neuroflow.visualization.flowviz.common import build_structured_grid, fit_camera_to_bounds, load_ras_canonical


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select aneurysm neck using a movable 3D cutting plane.")
    p.add_argument("--roi", required=True, help="Rough aneurysm ROI mask including a small artery connection.")
    p.add_argument("--bg", default="", help="Optional background magnitude NIfTI for slice context.")
    p.add_argument("--out", required=True, help="Output aneurysm-only NIfTI mask.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dilate", type=int, default=0, help="Optional dilation iterations after cutting, usually 0.")
    p.add_argument("--neck-thickness-mm", type=float, default=0.75, help="Thickness around plane used to estimate neck area.")
    p.add_argument("--background", default="white")
    p.add_argument("--window-width", type=int, default=1600)
    p.add_argument("--window-height", type=int, default=950)
    p.add_argument("--camera-azimuth", type=float, default=35.0)
    p.add_argument("--camera-elevation", type=float, default=18.0)
    p.add_argument("--camera-distance-scale", type=float, default=1.7)
    p.add_argument("--camera-zoom", type=float, default=1.05)
    return p.parse_args()


def _ui_text_color(background: str) -> str:
    try:
        rgb = np.asarray(pv.Color(background).float_rgb, dtype=np.float64)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    except Exception:
        luminance = 0.5
    return "black" if luminance >= 0.55 else "white"


def _load_mask(path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data >= float(threshold), np.asarray(img.affine, dtype=np.float64), img


def _load_bg(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    data, _ = load_ras_canonical(path)
    if data.ndim == 4:
        data = data[..., 0]
    if data.shape[:3] != target_shape:
        out = np.zeros(target_shape, dtype=np.float32)
        sx = min(int(data.shape[0]), int(target_shape[0]))
        sy = min(int(data.shape[1]), int(target_shape[1]))
        sz = min(int(data.shape[2]), int(target_shape[2]))
        out[:sx, :sy, :sz] = data[:sx, :sy, :sz]
        data = out
    return data.astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask > 0)
    props = regionprops(lab)
    if not props:
        return np.zeros(mask.shape, dtype=bool)
    main = max(props, key=lambda p: p.area)
    return lab == main.label


def _voxel_to_world(points_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ijk, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (np.concatenate([pts, ones], axis=1) @ affine.T)[:, :3]


def _world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    inv = np.linalg.inv(affine)
    return (np.concatenate([pts, ones], axis=1) @ inv.T)[:, :3]


def _mask_centroid_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0).astype(np.float64)
    return _voxel_to_world(coords, affine).mean(axis=0)


def _principal_normal(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords_world = _voxel_to_world(np.argwhere(mask > 0).astype(np.float64), affine)
    centered = coords_world - coords_world.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[0]
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def _surface_from_mask(mask: np.ndarray, affine: np.ndarray) -> pv.PolyData:
    if int(mask.sum()) < 4:
        return pv.PolyData()
    verts_ijk, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    verts = _voxel_to_world(verts_ijk, affine)
    vtk_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    return pv.PolyData(verts, vtk_faces)


def _build_grid(mask: np.ndarray, affine: np.ndarray, bg: np.ndarray | None) -> pv.StructuredGrid:
    zeros = np.zeros(mask.shape, dtype=np.float32)
    mag = bg if bg is not None else zeros
    return build_structured_grid(mag, zeros, zeros, zeros, affine, mask_t=mask.astype(np.uint8))


def _signed_world(mask: np.ndarray, affine: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0).astype(np.float64)
    coords_world = _voxel_to_world(coords, affine)
    signed = (coords_world - origin[None, :]) @ normal
    return coords, coords_world, signed


def _cut_mask(
    roi_mask: np.ndarray,
    affine: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    keep_sign: float,
    dilate: int,
) -> np.ndarray:
    coords, _, signed = _signed_world(roi_mask, affine, origin, normal)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    tol = float(np.min(zooms)) * 0.5
    keep = signed * float(keep_sign) >= -tol
    out = np.zeros(roi_mask.shape, dtype=bool)
    keep_coords = coords[keep].astype(int)
    if keep_coords.size:
        out[keep_coords[:, 0], keep_coords[:, 1], keep_coords[:, 2]] = True
    out = _largest_component(out)
    if int(dilate) > 0:
        out = binary_dilation(out, iterations=int(dilate)) & roi_mask
    return out


def _neck_intersection(
    roi_mask: np.ndarray,
    affine: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    thickness_mm: float,
) -> tuple[np.ndarray, float, float]:
    coords, _, signed = _signed_world(roi_mask, affine, origin, normal)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    half_thickness = max(float(thickness_mm) * 0.5, float(np.min(zooms)) * 0.5)
    near = np.abs(signed) <= half_thickness
    neck_vox = coords[near].astype(int)
    if neck_vox.size == 0:
        near = np.abs(signed) <= float(thickness_mm)
        neck_vox = coords[near].astype(int)
    area = float(len(neck_vox) * (np.min(zooms) ** 2))
    diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else float("nan")
    return neck_vox, area, diameter


def _farthest_pair(points_world: np.ndarray) -> tuple[int, int]:
    if len(points_world) < 2:
        return 0, 0
    diffs = points_world[:, None, :] - points_world[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(dist2)), dist2.shape))


def _save_outputs(
    roi_mask: np.ndarray,
    sac_mask: np.ndarray,
    affine: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: Path,
    source_roi_path: Path,
    origin: np.ndarray,
    normal: np.ndarray,
    keep_sign: float,
    thickness_mm: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(sac_mask.astype(np.uint8), ref_img.affine, ref_img.header), str(out_path))

    oriented_normal = np.asarray(normal, dtype=np.float64) * float(keep_sign)
    oriented_normal = oriented_normal / max(float(np.linalg.norm(oriented_normal)), 1e-12)
    neck_vox, neck_area, neck_diameter = _neck_intersection(roi_mask, affine, origin, normal, thickness_mm)
    if len(neck_vox) >= 2:
        neck_world = _voxel_to_world(neck_vox.astype(np.float64), affine)
        i, j = _farthest_pair(neck_world)
        neck_points_vox = neck_vox[[i, j]].astype(np.float64)
        neck_points_world = neck_world[[i, j]]
    else:
        neck_points_vox = _world_to_voxel(np.asarray(origin, dtype=np.float64)[None, :], affine).repeat(2, axis=0)
        neck_points_world = np.asarray(origin, dtype=np.float64)[None, :].repeat(2, axis=0)

    sac_coords = np.argwhere(sac_mask > 0).astype(np.float64)
    sac_world = _voxel_to_world(sac_coords, affine)
    signed = (sac_world - origin[None, :]) @ oriented_normal
    if signed.size:
        dome_idx = int(np.argmax(signed))
        sac_side_vox = sac_coords[dome_idx]
        sac_side_world = sac_world[dome_idx]
        height_mm = float(np.max(signed))
    else:
        sac_side_vox = neck_points_vox[0]
        sac_side_world = neck_points_world[0]
        height_mm = float("nan")

    slice_index = int(np.rint(neck_vox[:, 2].mean())) if len(neck_vox) else int(np.rint(neck_points_vox[0, 2]))
    info: dict[str, Any] = {
        "selection_mode": "manual_3d_plane",
        "source_roi_path": str(source_roi_path),
        "out_path": str(out_path),
        "view": "axial",
        "slice_index": slice_index,
        "neck_points_vox": [[round(float(v), 3) for v in neck_points_vox[0]], [round(float(v), 3) for v in neck_points_vox[1]]],
        "sac_side_point_vox": [round(float(v), 3) for v in sac_side_vox],
        "neck_points_world_mm": [[round(float(v), 4) for v in neck_points_world[0]], [round(float(v), 4) for v in neck_points_world[1]]],
        "sac_side_point_world_mm": [round(float(v), 4) for v in sac_side_world],
        "neck_plane_point_world_mm": [round(float(v), 4) for v in origin],
        "neck_plane_normal_world": [round(float(v), 8) for v in oriented_normal],
        "manual_neck_width_mm": round(float(neck_diameter), 4) if math.isfinite(neck_diameter) else None,
        "plane_neck_area_mm2": round(float(neck_area), 6),
        "plane_neck_equivalent_diameter_mm": round(float(neck_diameter), 4) if math.isfinite(neck_diameter) else None,
        "plane_neck_voxel_count": int(len(neck_vox)),
        "plane_neck_thickness_mm": round(float(thickness_mm), 4),
        "plane_neck_voxels_ijk": neck_vox.astype(int).tolist(),
        "height_from_neck_plane_mm": round(height_mm, 4) if math.isfinite(height_mm) else None,
        "roi_voxels": int(roi_mask.sum()),
        "sac_voxels": int(sac_mask.sum()),
    }
    sidecar = out_path.with_suffix("")
    if sidecar.suffix == ".nii":
        sidecar = sidecar.with_suffix("")
    sidecar = sidecar.with_name(sidecar.name + "_neck.json")
    info["neck_json"] = str(sidecar)
    sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


@dataclass
class PlaneState:
    origin: np.ndarray
    normal: np.ndarray
    keep_sign: float = 1.0
    show_hud: bool = True
    camera_initialized: bool = False
    note: str = "move/rotate plane to the neck"


class NeckPlaneSelector:
    def __init__(
        self,
        roi_mask: np.ndarray,
        affine: np.ndarray,
        ref_img: nib.Nifti1Image,
        bg: np.ndarray | None,
        roi_path: Path,
        out_path: Path,
        args: argparse.Namespace,
    ) -> None:
        self.roi_mask = roi_mask.astype(bool)
        self.affine = affine
        self.ref_img = ref_img
        self.bg = bg
        self.roi_path = roi_path
        self.out_path = out_path
        self.args = args
        self.grid = _build_grid(self.roi_mask, self.affine, self.bg)
        self.full_surface = _surface_from_mask(self.roi_mask, self.affine)
        if self.full_surface.n_points == 0:
            raise ValueError("ROI surface is empty.")

        self.state = PlaneState(
            origin=_mask_centroid_world(self.roi_mask, self.affine),
            normal=_principal_normal(self.roi_mask, self.affine),
        )
        self.plotter = pv.Plotter(
            window_size=(int(args.window_width), int(args.window_height)),
            notebook=False,
            off_screen=False,
        )
        self.plotter.set_background(str(args.background))
        self.ui_color = _ui_text_color(str(args.background))
        self.actors: dict[str, Any] = {}
        self.plane_widget = None

    def _remove_actor(self, key: str) -> None:
        actor = self.actors.pop(key, None)
        if actor is not None:
            self.plotter.remove_actor(actor, reset_camera=False)

    def _current_cut(self) -> np.ndarray:
        return _cut_mask(
            self.roi_mask,
            self.affine,
            self.state.origin,
            self.state.normal,
            self.state.keep_sign,
            int(self.args.dilate),
        )

    def _hud(self, sac_mask: np.ndarray, neck_area: float, neck_diam: float) -> str:
        side = "+" if self.state.keep_sign > 0 else "-"
        lines = [
            "3D Neck Plane Selector",
            "Move/rotate plane to the aneurysm neck.",
            f"kept side: {side} normal | kept voxels: {int(sac_mask.sum())}/{int(self.roi_mask.sum())}",
            f"neck area: {neck_area:.3f} mm2 | equivalent diameter: {neck_diam:.3f} mm",
            "f flip side | s save | r reset camera | h HUD | q quit",
        ]
        if self.state.note:
            lines.append(self.state.note)
        return "\n".join(lines)

    def _draw_plane_patch(self) -> None:
        self._remove_actor("plane_patch")
        span = max(float(max(self.full_surface.length, 1.0)), 3.0)
        plane = pv.Plane(
            center=tuple(float(v) for v in self.state.origin),
            direction=tuple(float(v) for v in self.state.normal),
            i_size=span,
            j_size=span,
        )
        self.actors["plane_patch"] = self.plotter.add_mesh(
            plane,
            color="#06b6d4",
            opacity=0.22,
            show_edges=True,
            edge_color="#0891b2",
            pickable=False,
        )

    def rebuild(self, reset_camera: bool = False) -> None:
        for key in ("full_surface", "kept_surface", "neck_pts", "hud"):
            self._remove_actor(key)

        sac_mask = self._current_cut()
        kept_surface = _surface_from_mask(sac_mask, self.affine)
        neck_vox, neck_area, neck_diam = _neck_intersection(
            self.roi_mask,
            self.affine,
            self.state.origin,
            self.state.normal,
            float(self.args.neck_thickness_mm),
        )

        self.actors["full_surface"] = self.plotter.add_mesh(
            self.full_surface,
            color="lightgray",
            opacity=0.27,
            smooth_shading=True,
            pickable=False,
        )
        if kept_surface.n_points > 0:
            self.actors["kept_surface"] = self.plotter.add_mesh(
                kept_surface,
                color="tomato",
                opacity=0.86,
                smooth_shading=True,
                pickable=False,
            )
        if len(neck_vox):
            neck_world = _voxel_to_world(neck_vox.astype(np.float64), self.affine)
            self.actors["neck_pts"] = self.plotter.add_mesh(
                pv.PolyData(neck_world),
                color="#06b6d4",
                point_size=12,
                render_points_as_spheres=True,
                pickable=False,
            )

        self._draw_plane_patch()
        self.plotter.add_axes()
        if self.state.show_hud:
            self.actors["hud"] = self.plotter.add_text(
                self._hud(sac_mask, neck_area, neck_diam),
                position="upper_left",
                font_size=9,
                color=self.ui_color,
                name="hud",
            )

        if reset_camera or not self.state.camera_initialized:
            fit_camera_to_bounds(
                plotter=self.plotter,
                bounds=self.full_surface.bounds,
                azimuth_deg=float(self.args.camera_azimuth),
                elevation_deg=float(self.args.camera_elevation),
                distance_scale=float(self.args.camera_distance_scale),
                zoom=float(self.args.camera_zoom),
            )
            self.state.camera_initialized = True
        self.plotter.render()

    def on_plane(self, normal, origin) -> None:
        normal_arr = np.asarray(normal, dtype=np.float64)
        normal_arr = normal_arr / max(float(np.linalg.norm(normal_arr)), 1e-12)
        self.state.normal = normal_arr
        self.state.origin = np.asarray(origin, dtype=np.float64)
        self.state.note = "plane updated"
        self.rebuild(reset_camera=False)

    def flip(self) -> None:
        self.state.keep_sign *= -1.0
        self.state.note = "kept side flipped"
        self.rebuild(reset_camera=False)

    def save(self) -> None:
        sac_mask = self._current_cut()
        _save_outputs(
            roi_mask=self.roi_mask,
            sac_mask=sac_mask,
            affine=self.affine,
            ref_img=self.ref_img,
            out_path=self.out_path,
            source_roi_path=self.roi_path,
            origin=self.state.origin,
            normal=self.state.normal,
            keep_sign=self.state.keep_sign,
            thickness_mm=float(self.args.neck_thickness_mm),
        )
        self.state.note = f"saved {self.out_path.name}"
        self.rebuild(reset_camera=False)

    def toggle_hud(self) -> None:
        self.state.show_hud = not self.state.show_hud
        self.rebuild(reset_camera=False)

    def reset_camera(self) -> None:
        self.state.camera_initialized = False
        self.state.note = "camera reset"
        self.rebuild(reset_camera=True)

    def show(self) -> None:
        self.rebuild(reset_camera=True)
        self.plane_widget = self.plotter.add_plane_widget(
            callback=self.on_plane,
            normal=tuple(float(v) for v in self.state.normal),
            origin=tuple(float(v) for v in self.state.origin),
            bounds=self.full_surface.bounds,
            factor=1.15,
            color="#06b6d4",
            implicit=True,
            outline_translation=True,
            origin_translation=True,
            normal_rotation=True,
            interaction_event="end",
        )
        self.plotter.add_key_event("f", self.flip)
        self.plotter.add_key_event("s", self.save)
        self.plotter.add_key_event("r", self.reset_camera)
        self.plotter.add_key_event("h", self.toggle_hud)
        self.plotter.add_key_event("q", lambda: self.plotter.iren.terminate_app())
        self.plotter.add_key_event("Escape", lambda: self.plotter.iren.terminate_app())
        self.plotter.show()


def main() -> None:
    args = _parse_args()
    roi_path = Path(args.roi).expanduser().resolve()
    bg_path = Path(args.bg).expanduser().resolve() if args.bg else None
    out_path = Path(args.out).expanduser().resolve()
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI not found: {roi_path}")

    roi_mask, affine, ref_img = _load_mask(roi_path, float(args.threshold))
    roi_mask = _largest_component(roi_mask)
    if int(roi_mask.sum()) == 0:
        raise ValueError("ROI mask is empty after threshold/largest-component filtering.")

    bg = None
    if bg_path is not None and bg_path.is_file():
        bg = _load_bg(bg_path, roi_mask.shape)
        print(f"Background loaded: {bg_path}")

    print(f"ROI: {roi_path} ({int(roi_mask.sum())} fg voxels)")
    print(f"Output will be written to: {out_path}")
    print("Move the cyan plane to the aneurysm neck, press f to flip kept side, press s to save.")

    selector = NeckPlaneSelector(
        roi_mask=roi_mask,
        affine=affine,
        ref_img=ref_img,
        bg=bg,
        roi_path=roi_path,
        out_path=out_path,
        args=args,
    )
    selector.show()


if __name__ == "__main__":
    main()
