#!/usr/bin/env python3
"""Component-level QA for Vx/Vy/Vz values inside a vessel mask."""

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
    fit_camera_to_bounds,
    load_flow_data_from_args,
    merge_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Vx/Vy/Vz values at CoW points with stats, CSV export, and a 2x2 PyVista panel."
    )
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument(
        "--aggregate",
        choices=["frame", "mean", "maxabs"],
        default="frame",
        help="How to combine temporal data before visualization.",
    )
    parser.add_argument(
        "--report-all-frames",
        action="store_true",
        help="Print per-frame statistics in addition to the selected aggregate.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional CSV path with per-point CoW values (i,j,k,x,y,z,vx,vy,vz,speed).",
    )

    parser.add_argument("--signed-percentile", type=float, default=99.0, help="Percentile for signed component color limits.")
    parser.add_argument("--speed-vmin", type=float, default=None)
    parser.add_argument("--speed-vmax", type=float, default=None)
    parser.add_argument("--component-cmap", type=str, default="coolwarm")
    parser.add_argument("--speed-cmap", type=str, default="turbo")

    parser.add_argument("--show-glyphs", action="store_true", help="Overlay velocity glyphs in the speed panel.")
    parser.add_argument("--glyph-stride", type=int, default=50, help="Use one every N mask points for glyphs.")
    parser.add_argument("--glyph-factor", type=float, default=1.6, help="Glyph size factor.")

    parser.add_argument("--smooth-iters", type=int, default=8, help="Surface smoothing iterations.")
    parser.add_argument("--background", type=str, default="black")
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)

    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-elevation", type=float, default=18.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.5)
    parser.add_argument("--camera-zoom", type=float, default=1.2)
    return parser.parse_args()


def _aggregate_component(volume_4d: np.ndarray, mode: str, frame: int) -> np.ndarray:
    if mode == "frame":
        return extract_frame(volume_4d, frame)
    if mode == "mean":
        return np.mean(volume_4d, axis=-1)
    idx = np.argmax(np.abs(volume_4d), axis=-1)[..., np.newaxis]
    return np.take_along_axis(volume_4d, idx, axis=-1)[..., 0]


def _aggregate_mask(mask_4d: np.ndarray, mode: str, frame: int) -> np.ndarray:
    if mode == "frame":
        return extract_frame(mask_4d, frame) > 0
    return np.max(mask_4d, axis=-1) > 0


def _component_stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {
            "n": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "pos_frac": 0.0,
            "neg_frac": 0.0,
            "zero_frac": 1.0,
        }
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p01": float(np.percentile(v, 1.0)),
        "p50": float(np.percentile(v, 50.0)),
        "p99": float(np.percentile(v, 99.0)),
        "pos_frac": float(np.mean(v > 0)),
        "neg_frac": float(np.mean(v < 0)),
        "zero_frac": float(np.mean(np.isclose(v, 0.0, atol=1e-6))),
    }


def _print_stats_block(label: str, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mask: np.ndarray) -> None:
    mask = mask > 0
    vx_m = vx[mask]
    vy_m = vy[mask]
    vz_m = vz[mask]
    speed_m = np.sqrt(vx_m * vx_m + vy_m * vy_m + vz_m * vz_m)

    print(f"\n[{label}] Points in mask: {int(mask.sum())}")
    for name, data in (("Vx", vx_m), ("Vy", vy_m), ("Vz", vz_m), ("Speed", speed_m)):
        st = _component_stats(data)
        print(
            f"  {name}: n={st['n']} min={st['min']:.5f} max={st['max']:.5f} "
            f"mean={st['mean']:.5f} std={st['std']:.5f} "
            f"p01={st['p01']:.5f} p50={st['p50']:.5f} p99={st['p99']:.5f} "
            f"pos={st['pos_frac']:.3f} neg={st['neg_frac']:.3f} zero={st['zero_frac']:.3f}"
        )


def _signed_clim(values: np.ndarray, percentile: float) -> tuple[float, float]:
    abs_v = np.abs(values[np.isfinite(values)])
    if abs_v.size == 0:
        return (-1.0, 1.0)
    vmax = float(np.percentile(abs_v, float(percentile)))
    vmax = max(vmax, 1e-6)
    return (-vmax, vmax)


def _export_csv(
    out_path: Path,
    affine: np.ndarray,
    mask: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> None:
    idx = np.argwhere(mask > 0).astype(np.int32)
    if idx.size == 0:
        raise ValueError("Mask is empty. Cannot export per-point CSV.")

    ijk = idx.astype(np.float64)
    aff = np.asarray(affine, dtype=np.float64)
    xyz = np.einsum("ij,nj->ni", aff[:3, :3], ijk) + aff[:3, 3]
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite coordinates after affine transform. Verify input affine matrices.")

    vv_x = vx[mask > 0]
    vv_y = vy[mask > 0]
    vv_z = vz[mask > 0]
    speed = np.sqrt(vv_x * vv_x + vv_y * vv_y + vv_z * vv_z)

    table = np.column_stack([idx, xyz, vv_x, vv_y, vv_z, speed])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(out_path),
        table,
        delimiter=",",
        header="i,j,k,x_mm,y_mm,z_mm,vx,vy,vz,speed",
        comments="",
        fmt=["%d", "%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
    )
    print(f"Per-point CSV written to: {out_path.resolve()}")


def _show_or_save(plotter: pv.Plotter, args: argparse.Namespace) -> None:
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Component QA image written to: {args.screenshot.resolve()}")
    else:
        plotter.show()


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    if flow.mask is None:
        raise ValueError("Component QA requires a CoW mask. Provide --mask <mask.nii.gz>.")

    vx = _aggregate_component(flow.vx, args.aggregate, args.frame)
    vy = _aggregate_component(flow.vy, args.aggregate, args.frame)
    vz = _aggregate_component(flow.vz, args.aggregate, args.frame)
    mag = _aggregate_component(flow.mag, "frame" if args.aggregate == "frame" else "mean", args.frame)
    mask = _aggregate_mask(flow.mask, args.aggregate, args.frame).astype(np.uint8)

    if args.zero_velocity_below > 0:
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        low = speed < float(args.zero_velocity_below)
        vx = vx.copy()
        vy = vy.copy()
        vz = vz.copy()
        vx[low] = 0.0
        vy[low] = 0.0
        vz[low] = 0.0

    _print_stats_block(f"Aggregate={args.aggregate}", vx, vy, vz, mask)

    if args.report_all_frames:
        nt = flow.vx.shape[-1]
        print(f"\nPer-frame report ({nt} frames):")
        for t in range(nt):
            mask_t = extract_frame(flow.mask, t).astype(bool)
            _print_stats_block(
                label=f"Frame={t}",
                vx=extract_frame(flow.vx, t),
                vy=extract_frame(flow.vy, t),
                vz=extract_frame(flow.vz, t),
                mask=mask_t,
            )

    if args.export_csv is not None:
        _export_csv(args.export_csv, flow.affine, mask.astype(bool), vx, vy, vz)

    grid = build_structured_grid(mag, vx, vy, vz, flow.affine, mask_t=mask)
    grid.point_data["vx"] = vx.ravel(order="F")
    grid.point_data["vy"] = vy.ravel(order="F")
    grid.point_data["vz"] = vz.ravel(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="mask")
    if surface.n_points == 0:
        raise ValueError("Mask contour is empty. Check mask alignment/threshold.")
    if args.smooth_iters > 0:
        surface = surface.smooth(n_iter=int(args.smooth_iters), relaxation_factor=0.1)
    surface = surface.sample(grid)

    vx_clim = _signed_clim(surface.point_data["vx"], args.signed_percentile)
    vy_clim = _signed_clim(surface.point_data["vy"], args.signed_percentile)
    vz_clim = _signed_clim(surface.point_data["vz"], args.signed_percentile)

    provided_speed = None
    if args.speed_vmin is not None and args.speed_vmax is not None:
        provided_speed = (float(args.speed_vmin), float(args.speed_vmax))
    speed_clim = compute_speed_clim(surface.point_data["speed"], provided=provided_speed, q_low=2.0, q_high=98.5)

    glyphs = None
    if args.show_glyphs:
        mask_ids = np.flatnonzero(grid.point_data["mask"] > 0.5)
        stride = max(1, int(args.glyph_stride))
        ids = mask_ids[::stride]
        if ids.size > 0:
            pts = grid.extract_points(ids.astype(np.int64), include_cells=False)
            glyphs = pts.glyph(orient="vel", scale="speed", factor=float(args.glyph_factor))

    p = pv.Plotter(shape=(2, 2), off_screen=args.off_screen, window_size=(1700, 1200))
    p.set_background(args.background)

    panels = [
        ("Vx (signed)", "vx", vx_clim, args.component_cmap),
        ("Vy (signed)", "vy", vy_clim, args.component_cmap),
        ("Vz (signed)", "vz", vz_clim, args.component_cmap),
        ("Speed (m/s)", "speed", speed_clim, args.speed_cmap),
    ]

    b_list = [surface.bounds]
    if glyphs is not None and glyphs.n_points > 0:
        b_list.append(glyphs.bounds)
    merged_bounds = merge_bounds(b_list)

    for k, (title, scalar_name, clim, cmap) in enumerate(panels):
        row = k // 2
        col = k % 2
        p.subplot(row, col)
        p.add_text(title, font_size=12)
        p.add_mesh(
            surface,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            smooth_shading=True,
            scalar_bar_args={"title": title, "vertical": False},
        )
        if scalar_name == "speed" and glyphs is not None and glyphs.n_points > 0:
            p.add_mesh(glyphs, scalars="speed", cmap=args.speed_cmap, clim=speed_clim)
        fit_camera_to_bounds(
            plotter=p,
            bounds=merged_bounds,
            azimuth_deg=args.camera_azimuth,
            elevation_deg=args.camera_elevation,
            distance_scale=args.camera_distance_scale,
            zoom=args.camera_zoom,
        )
        p.add_axes()

    _show_or_save(p, args)


if __name__ == "__main__":
    main()
