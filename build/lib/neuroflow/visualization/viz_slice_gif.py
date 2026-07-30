#!/usr/bin/env python3
"""Create a simple magnitude slice GIF over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

from neuroflow.visualization.flowviz.common import add_input_args, build_structured_grid, extract_frame, load_flow_data_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render magnitude slice GIF over time.")
    add_input_args(parser)
    parser.add_argument("--gif-path", type=Path, required=True)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--slice-normal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--background", type=str, default="white")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    nt = flow.mag.shape[-1]
    p = pv.Plotter(off_screen=True)
    p.set_background(args.background)
    args.gif_path.parent.mkdir(parents=True, exist_ok=True)
    p.open_gif(str(args.gif_path), fps=max(1, int(args.gif_fps)))

    for t in range(nt):
        mag_t = extract_frame(flow.mag, t)
        vx_t = extract_frame(flow.vx, t)
        vy_t = extract_frame(flow.vy, t)
        vz_t = extract_frame(flow.vz, t)
        mask_t = extract_frame(flow.mask, t) if flow.mask is not None else None

        grid = build_structured_grid(mag_t, vx_t, vy_t, vz_t, flow.affine, mask_t=mask_t)
        slc = grid.slice(normal=args.slice_normal)

        p.clear()
        p.add_mesh(slc, scalars="mag", opacity=0.75, cmap="gray")
        p.add_text(f"Frame {t + 1}/{nt}", font_size=14)
        p.write_frame()

    p.close()
    print(f"Slice GIF written to: {args.gif_path.resolve()}")


if __name__ == "__main__":
    main()
