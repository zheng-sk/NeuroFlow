#!/usr/bin/env python3
"""Legacy wrapper for modular PyVista visualization scripts.

Use these new entrypoints directly:
- code/visualization/viz_streamlines.py
- code/visualization/viz_flow_gif.py
- code/visualization/viz_slice_gif.py
- code/visualization/viz_surface.py
"""

from __future__ import annotations

import sys


def _read_mode(argv: list[str]) -> str:
    if "--mode" in argv:
        idx = argv.index("--mode")
        if idx + 1 < len(argv):
            return argv[idx + 1].strip().lower()
    return "panel"


def main() -> None:
    mode = _read_mode(sys.argv[1:])

    if mode in {"gif"}:
        from viz_slice_gif import main as entry

        print("[flow_qc_pyvista] delegating to viz_slice_gif.py")
    elif mode in {"flow-gif"}:
        from viz_flow_gif import main as entry

        print("[flow_qc_pyvista] delegating to viz_flow_gif.py")
    elif mode in {"particle-gif"}:
        from viz_particle_gif import main as entry

        print("[flow_qc_pyvista] delegating to viz_particle_gif.py")
    elif mode in {"direction-qa"}:
        from viz_direction_qa import main as entry

        print("[flow_qc_pyvista] delegating to viz_direction_qa.py")
    elif mode in {"affine-direction-qa"}:
        from viz_affine_direction_qc import main as entry

        print("[flow_qc_pyvista] delegating to viz_affine_direction_qc.py")
    elif mode in {"flux-qa"}:
        from viz_flux_planes_qc import main as entry

        print("[flow_qc_pyvista] delegating to viz_flux_planes_qc.py")
    elif mode in {"component-qa"}:
        from viz_component_qc import main as entry

        print("[flow_qc_pyvista] delegating to viz_component_qc.py")
    elif mode in {"roi-sign-qa", "roi-sign", "roi-2d-3d"}:
        from viz_roi_sign_qc import main as entry

        print("[flow_qc_pyvista] delegating to viz_roi_sign_qc.py")
    elif mode in {"surface"}:
        from viz_surface import main as entry

        print("[flow_qc_pyvista] delegating to viz_surface.py")
    else:
        from viz_streamlines import main as entry

        print("[flow_qc_pyvista] delegating to viz_streamlines.py")

    entry()


if __name__ == "__main__":
    main()
