import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

# Configure PyVista environment BEFORE import to ensure proper initialization.
# This must happen before pyvista is imported as it reads these at import time.
def _configure_pyvista_env() -> None:
    """Set environment variables for PyVista desktop rendering before import."""
    if "PYVISTA_OFF_SCREEN" in os.environ:
        val = str(os.environ.get("PYVISTA_OFF_SCREEN", "")).strip().lower()
        if val in {"1", "true", "yes", "on"}:
            print("[info] Overriding PYVISTA_OFF_SCREEN for interactive ROI selection.")
    os.environ["PYVISTA_OFF_SCREEN"] = "false"
    # Some VTK builds honor this variable and may force off-screen behavior.
    os.environ.pop("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", None)
    # Ensure trame/jupyter backends are not used
    os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_PREFIX", "")

_configure_pyvista_env()

import numpy as np

try:
    import pyvista as pv
except Exception as exc:
    raise RuntimeError(
        "select_metric_roi_bbox.py requires pyvista. Install it in your environment (pip install pyvista)."
    ) from exc

# Apply runtime configuration after import
pv.OFF_SCREEN = False
if hasattr(pv, "set_jupyter_backend"):
    try:
        pv.set_jupyter_backend(None)
    except Exception:
        pass


def _configure_pyvista_desktop() -> None:
    """Re-apply desktop rendering config (called before plotter creation)."""
    pv.OFF_SCREEN = False
    # Force the global plotter theme to use the default backend
    if hasattr(pv, "global_theme"):
        pv.global_theme.notebook = False


def _pyvista_interactive_available() -> bool:
    """Best-effort preflight check to avoid hard crashes on headless/OpenGL-broken sessions."""
    fn = getattr(pv, "system_supports_plotting", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return True


def _load_payload(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _pick_mask_volume(mask_txyz: np.ndarray, mode: str, frame_index: int) -> np.ndarray:
    if mask_txyz.ndim != 4:
        raise ValueError(f"Expected mask shape [T,X,Y,Z], got {mask_txyz.shape}")

    t_count = int(mask_txyz.shape[0])
    if mode == "frame":
        f = int(np.clip(frame_index, 0, t_count - 1))
        return (mask_txyz[f] > 0.5).astype(np.uint8)
    if mode == "intersection":
        return np.all(mask_txyz > 0.5, axis=0).astype(np.uint8)
    return np.any(mask_txyz > 0.5, axis=0).astype(np.uint8)


def _mask_bbox_xyz(mask_xyz: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    pts = np.argwhere(mask_xyz > 0)
    if pts.size == 0:
        raise ValueError("Mask volume is empty after temporal reduction.")

    x0 = int(pts[:, 0].min())
    x1 = int(pts[:, 0].max()) + 1
    y0 = int(pts[:, 1].min())
    y1 = int(pts[:, 1].max()) + 1
    z0 = int(pts[:, 2].min())
    z1 = int(pts[:, 2].max()) + 1
    return x0, x1, y0, y1, z0, z1


def _bbox_xyz_to_world_bounds(
    bbox_xyz: Sequence[int], spacing_xyz: Sequence[float]
) -> Tuple[float, float, float, float, float, float]:
    x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_xyz]
    sx, sy, sz = [float(v) for v in spacing_xyz]
    return (
        float(x0) * sx,
        float(x1) * sx,
        float(y0) * sy,
        float(y1) * sy,
        float(z0) * sz,
        float(z1) * sz,
    )


def _world_bounds_to_bbox_xyz(
    bounds: Sequence[float], spacing_xyz: Sequence[float], shape_xyz: Sequence[int]
) -> Tuple[int, int, int, int, int, int]:
    bx0, bx1, by0, by1, bz0, bz1 = [float(v) for v in bounds]
    sx, sy, sz = [max(float(v), 1e-8) for v in spacing_xyz]
    nx, ny, nz = [int(v) for v in shape_xyz]

    x0 = int(np.floor(min(bx0, bx1) / sx))
    x1 = int(np.ceil(max(bx0, bx1) / sx))
    y0 = int(np.floor(min(by0, by1) / sy))
    y1 = int(np.ceil(max(by0, by1) / sy))
    z0 = int(np.floor(min(bz0, bz1) / sz))
    z1 = int(np.ceil(max(bz0, bz1) / sz))

    x0 = max(0, min(x0, nx - 1))
    y0 = max(0, min(y0, ny - 1))
    z0 = max(0, min(z0, nz - 1))
    x1 = max(x0 + 1, min(x1, nx))
    y1 = max(y0 + 1, min(y1, ny))
    z1 = max(z0 + 1, min(z1, nz))

    return x0, x1, y0, y1, z0, z1


def _map_hr_to_lr_bbox(
    bbox_hr: Sequence[int], hr_shape: Sequence[int], lr_shape: Sequence[int]
) -> Tuple[int, int, int, int, int, int]:
    x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_hr]
    hx, hy, hz = [int(v) for v in hr_shape]
    lx, ly, lz = [int(v) for v in lr_shape]

    def _map(a0: int, a1: int, h: int, l: int) -> Tuple[int, int]:
        b0 = int(np.floor((float(a0) * l) / max(h, 1)))
        b1 = int(np.ceil((float(a1) * l) / max(h, 1)))
        b0 = max(0, min(b0, l - 1))
        b1 = max(b0 + 1, min(b1, l))
        return b0, b1

    lx0, lx1 = _map(x0, x1, hx, lx)
    ly0, ly1 = _map(y0, y1, hy, ly)
    lz0, lz1 = _map(z0, z1, hz, lz)
    return lx0, lx1, ly0, ly1, lz0, lz1


def _extract_bounds_from_widget_output(obj: Any) -> Tuple[float, float, float, float, float, float]:
    if hasattr(obj, "bounds"):
        b = obj.bounds
        if len(b) == 6:
            return tuple(float(v) for v in b)
    if isinstance(obj, (tuple, list, np.ndarray)) and len(obj) == 6:
        return tuple(float(v) for v in obj)
    raise ValueError("Could not extract bounds from box widget callback output.")


def _bounds_from_box_widget(box_widget: Any) -> Tuple[float, float, float, float, float, float]:
    """Read current world bounds directly from a vtkBoxWidget."""
    poly = pv.PolyData()
    box_widget.GetPolyData(poly)
    b = tuple(float(v) for v in poly.bounds)
    if len(b) != 6:
        raise ValueError("Invalid bounds from vtkBoxWidget.")
    return b


def _build_mask_surface(mask_xyz: np.ndarray, spacing_xyz: Sequence[float]) -> "pv.PolyData":
    nx, ny, nz = [int(v) for v in mask_xyz.shape]
    sx, sy, sz = [float(v) for v in spacing_xyz]

    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(sx, sy, sz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.cell_data["mask"] = mask_xyz.astype(np.uint8).ravel(order="F")
    th = grid.threshold(0.5, scalars="mask")
    try:
        surf = th.extract_surface(algorithm="dataset_surface").triangulate()
    except TypeError:
        # Backward compatibility with older PyVista versions.
        surf = th.extract_surface().triangulate()
    if surf.n_cells == 0:
        raise ValueError("Could not extract surface from selected mask volume.")
    return surf


def _select_bbox_3d(
    mask_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
    initial_bbox_xyz: Sequence[int],
) -> Tuple[int, int, int, int, int, int]:
    surface = _build_mask_surface(mask_xyz, spacing_xyz)
    initial_bounds_world = _bbox_xyz_to_world_bounds(initial_bbox_xyz, spacing_xyz)

    state: Dict[str, Any] = {
        "accepted": False,
        "bounds_world": tuple(initial_bounds_world),
    }

    # Ensure desktop rendering mode is active before creating plotter
    _configure_pyvista_desktop()

    # Create plotter and explicitly initialize the render window
    plotter = pv.Plotter(window_size=(1280, 900), notebook=False, off_screen=False)

    # Force render window initialization by accessing iren (interactor)
    # This ensures the window is created before we try to show it
    try:
        _ = plotter.iren
        # Also force window creation
        if plotter.render_window is None:
            raise RuntimeError("render_window is None after plotter creation")
    except Exception as init_err:
        plotter.close()
        raise RuntimeError(
            "PyVista could not initialize an interactive render window. "
            "Ensure you are running in a desktop session with GUI access. "
            f"Details: {init_err}"
        ) from init_err

    plotter.set_background("#0b1020")
    plotter.add_mesh(surface, color="#67e8f9", opacity=0.35, show_edges=False)
    plotter.add_axes()

    instructions = (
        "3D ROI Box Selection\n"
        "- Drag box handles to adjust ROI\n"
        "- Press Enter/Space/Q to accept\n"
        "- Press Esc to cancel"
    )
    plotter.add_text(instructions, position="upper_left", font_size=11, color="white")

    def _box_callback(widget_output, box_widget=None):
        try:
            if box_widget is not None:
                state["bounds_world"] = _bounds_from_box_widget(box_widget)
            else:
                state["bounds_world"] = _extract_bounds_from_widget_output(widget_output)
        except Exception:
            pass

    box_widget = plotter.add_box_widget(
        callback=_box_callback,
        bounds=initial_bounds_world,
        factor=1.0,
        rotation_enabled=False,
        color="yellow",
        # Keep PolyData callback payload so bounds are available in callback output.
        use_planes=False,
        outline_translation=True,
        # Also pass the vtkBoxWidget to robustly read final bounds at accept time.
        pass_widget=True,
    )

    def _stop_interaction() -> None:
        # Avoid closing the render window directly inside a key callback,
        # which can crash on some VTK/macOS combinations.
        try:
            if plotter.iren is not None:
                plotter.iren.terminate_app()
        except Exception:
            pass

    def _accept() -> None:
        try:
            state["bounds_world"] = _bounds_from_box_widget(box_widget)
        except Exception:
            pass
        state["accepted"] = True
        _stop_interaction()

    def _cancel() -> None:
        state["accepted"] = False
        _stop_interaction()

    # Key aliases vary across OS/keyboard layouts.
    for key in ("Return", "Enter", "KP_Enter", "space", "q"):
        plotter.add_key_event(key, _accept)
    plotter.add_key_event("Escape", _cancel)

    try:
        plotter.show(title="ROI Bounding Box Selector (3D)", auto_close=False)
    except AttributeError as exc:
        if "IsCurrent" in str(exc) or "NoneType" in str(exc):
            raise RuntimeError(
                "PyVista could not create an interactive render window "
                "(render_window is None). Make sure you are running in a desktop session "
                "with GUI access and that off-screen rendering is disabled. "
                "You can try: unset PYVISTA_OFF_SCREEN"
            ) from exc
        raise
    finally:
        try:
            plotter.close()
        except Exception:
            pass

    if not bool(state.get("accepted", False)):
        raise KeyboardInterrupt("ROI selection canceled by user.")

    bbox_xyz = _world_bounds_to_bbox_xyz(
        bounds=state["bounds_world"],
        spacing_xyz=spacing_xyz,
        shape_xyz=mask_xyz.shape,
    )
    if tuple(int(v) for v in bbox_xyz) == tuple(int(v) for v in initial_bbox_xyz):
        print("[warn] Selected bbox is identical to initial bbox.")
    return bbox_xyz


def _select_bbox_napari(
    mask_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
    initial_bbox_xyz: Sequence[int],
) -> Tuple[int, int, int, int, int, int]:
    try:
        import napari
    except Exception as exc:
        raise RuntimeError(
            "Napari backend requested but napari is not installed. Install with: pip install napari[all]"
        ) from exc

    init = _clip_bbox_xyz(initial_bbox_xyz, mask_xyz.shape)
    ix0, ix1, iy0, iy1, iz0, iz1 = [int(v) for v in init]
    # Two corners in data coordinates [x, y, z].
    corners = np.asarray(
        [
            [ix0, iy0, iz0],
            [max(ix0 + 1, ix1 - 1), max(iy0 + 1, iy1 - 1), max(iz0 + 1, iz1 - 1)],
        ],
        dtype=np.float32,
    )

    state: Dict[str, Any] = {"accepted": False, "bbox_xyz": init}
    viewer = napari.Viewer(ndisplay=3, title="ROI Bounding Box Selector (Napari)")
    viewer.add_image(
        mask_xyz.astype(np.float32),
        name="mask",
        rendering="iso",
        iso_threshold=0.5,
        opacity=0.35,
        scale=tuple(float(v) for v in spacing_xyz),
    )
    points = viewer.add_points(
        corners,
        name="bbox_corners",
        ndim=3,
        size=8,
        face_color="yellow",
        edge_color="black",
    )
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Napari ROI selection\n"
        "- Move/Add exactly 2 points (opposite bbox corners)\n"
        "- Enter: accept\n"
        "- Esc: cancel/close"
    )

    @viewer.bind_key("Enter")
    def _accept(_viewer) -> None:
        pts = np.asarray(points.data, dtype=np.float64)
        if pts.shape[0] < 2:
            print("[warn] Add at least 2 points to define bbox corners.")
            return
        p0 = pts.min(axis=0)
        p1 = pts.max(axis=0)
        x0 = int(np.floor(p0[0]))
        y0 = int(np.floor(p0[1]))
        z0 = int(np.floor(p0[2]))
        x1 = int(np.ceil(p1[0])) + 1
        y1 = int(np.ceil(p1[1])) + 1
        z1 = int(np.ceil(p1[2])) + 1
        state["bbox_xyz"] = _clip_bbox_xyz((x0, x1, y0, y1, z0, z1), mask_xyz.shape)
        state["accepted"] = True
        _viewer.close()

    @viewer.bind_key("Escape")
    def _cancel(_viewer) -> None:
        state["accepted"] = False
        _viewer.close()

    napari.run()
    if not bool(state.get("accepted", False)):
        raise KeyboardInterrupt("ROI selection canceled by user.")
    return tuple(int(v) for v in state["bbox_xyz"])


def _clip_bbox_xyz(bbox_xyz: Sequence[int], shape_xyz: Sequence[int]) -> Tuple[int, int, int, int, int, int]:
    x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_xyz]
    nx, ny, nz = [int(v) for v in shape_xyz]
    x0 = max(0, min(x0, nx - 1))
    y0 = max(0, min(y0, ny - 1))
    z0 = max(0, min(z0, nz - 1))
    x1 = max(x0 + 1, min(x1, nx))
    y1 = max(y0 + 1, min(y1, ny))
    z1 = max(z0 + 1, min(z1, nz))
    return x0, x1, y0, y1, z0, z1


def _select_bbox_manual(initial_bbox_xyz: Sequence[int], shape_xyz: Sequence[int]) -> Tuple[int, int, int, int, int, int]:
    default_bbox = _clip_bbox_xyz(initial_bbox_xyz, shape_xyz)
    print("\nManual ROI fallback")
    print("Enter bbox as six integers: x0 x1 y0 y1 z0 z1")
    print(f"Default bbox: {list(default_bbox)}")
    while True:
        raw = input("bbox> ").strip()
        if not raw:
            yn = input("Use default bbox? [y/N]: ").strip().lower()
            if yn in {"y", "yes"}:
                return default_bbox
            print("Please enter a custom bbox.")
            continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 6:
            print(f"[warn] Expected 6 integers, got {len(parts)}: {raw}")
            continue
        bbox = tuple(int(v) for v in parts)
        return _clip_bbox_xyz(bbox, shape_xyz)


def _bbox_tag(bbox_xyz: Sequence[int]) -> str:
    x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox_xyz]
    return f"bbox_x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}"


def main() -> None:
    _configure_pyvista_desktop()

    parser = argparse.ArgumentParser(
        description=(
            "Interactive 3D ROI bounding-box selector for SR/UQ metrics. "
            "Loads mask from analysis_payload.npz, opens a PyVista box widget, exports ROI JSON, "
            "and can optionally run generate_sr_uq_report.py."
        )
    )
    parser.add_argument("--payload-npz", required=True, help="Path to analysis_payload.npz")
    parser.add_argument("--out-json", required=True, help="Output path for ROI JSON")
    parser.add_argument(
        "--temporal-mode",
        default="union",
        choices=["union", "intersection", "frame"],
        help="How to collapse temporal mask before 3D selection.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index used when --temporal-mode=frame")
    parser.add_argument(
        "--padding-vox",
        type=int,
        default=2,
        help="Padding (in voxels) added around the default initial box from mask extent.",
    )

    parser.add_argument("--run-report", action="store_true", help="Run generate_sr_uq_report.py after ROI export.")
    parser.add_argument("--metadata-json", default="", help="Optional metadata JSON path for report generation.")
    parser.add_argument("--report-out-dir", default="", help="Optional output directory for report generation.")
    parser.add_argument(
        "--selector-mode",
        default="auto",
        choices=["auto", "3d", "napari", "manual"],
        help=(
            "ROI selector mode: 3d (PyVista), napari (3D points/corners), "
            "manual (terminal bbox), auto (3d -> napari -> manual)."
        ),
    )

    args = parser.parse_args()

    payload_path = Path(args.payload_npz).resolve()
    payload = _load_payload(str(payload_path))
    if "mask" not in payload:
        raise ValueError(f"Payload does not contain 'mask': {payload_path}")
    if "gt_norm" not in payload or "lr_norm" not in payload:
        raise ValueError("Payload must include 'gt_norm' and 'lr_norm' to map HR/LR bounding boxes.")

    mask_txyz = payload["mask"].astype(np.float32)
    hr_shape_xyz = tuple(int(v) for v in payload["gt_norm"].shape[2:])
    lr_shape_xyz = tuple(int(v) for v in payload["lr_norm"].shape[2:])
    hr_spacing = [float(v) for v in payload.get("hr_spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).tolist()]
    lr_spacing = [float(v) for v in payload.get("lr_spacing", np.array([1.0, 1.0, 1.0], dtype=np.float32)).tolist()]

    if tuple(mask_txyz.shape[1:]) != hr_shape_xyz:
        raise ValueError(
            f"Mask shape {tuple(mask_txyz.shape[1:])} does not match HR shape {hr_shape_xyz}. "
            "ROI is defined in HR voxel space."
        )

    mask_3d = _pick_mask_volume(mask_txyz, mode=str(args.temporal_mode), frame_index=int(args.frame_index))
    if int(mask_3d.sum()) == 0:
        raise ValueError("Selected temporal mask volume is empty.")

    x0, x1, y0, y1, z0, z1 = _mask_bbox_xyz(mask_3d)
    pad = max(0, int(args.padding_vox))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    z0 = max(0, z0 - pad)
    x1 = min(hr_shape_xyz[0], x1 + pad)
    y1 = min(hr_shape_xyz[1], y1 + pad)
    z1 = min(hr_shape_xyz[2], z1 + pad)

    initial_bbox = (x0, x1, y0, y1, z0, z1)
    selector_mode = str(args.selector_mode).strip().lower()
    used_mode: Optional[str] = None
    if selector_mode == "manual":
        bbox_hr = list(_select_bbox_manual(initial_bbox, hr_shape_xyz))
        used_mode = "manual"
    elif selector_mode == "napari":
        print("Opening Napari ROI selector...")
        bbox_hr = list(_select_bbox_napari(mask_3d, spacing_xyz=hr_spacing, initial_bbox_xyz=initial_bbox))
        used_mode = "napari_3d_points"
    else:
        can_pyvista = _pyvista_interactive_available()
        if selector_mode == "3d" and not can_pyvista:
            raise RuntimeError(
                "PyVista interactive plotting is not available in this session "
                "(headless display or missing OpenGL context). "
                "Use --selector-mode napari or --selector-mode manual."
            )

        if can_pyvista:
            print("Opening 3D ROI selector...")
            try:
                bbox_hr = list(_select_bbox_3d(mask_3d, spacing_xyz=hr_spacing, initial_bbox_xyz=initial_bbox))
                used_mode = "pyvista_3d_box_widget"
            except Exception as exc:
                if selector_mode == "3d":
                    raise
                print(f"[warn] 3D selector failed: {exc}")
                try:
                    print("Opening Napari ROI selector...")
                    bbox_hr = list(_select_bbox_napari(mask_3d, spacing_xyz=hr_spacing, initial_bbox_xyz=initial_bbox))
                    used_mode = "napari_3d_points"
                except Exception as exc2:
                    print(f"[warn] Napari selector failed: {exc2}")
                    bbox_hr = list(_select_bbox_manual(initial_bbox, hr_shape_xyz))
                    used_mode = "manual"
        else:
            print("[warn] PyVista interactive plotting is unavailable in this session.")
            try:
                print("Opening Napari ROI selector...")
                bbox_hr = list(_select_bbox_napari(mask_3d, spacing_xyz=hr_spacing, initial_bbox_xyz=initial_bbox))
                used_mode = "napari_3d_points"
            except Exception as exc2:
                print(f"[warn] Napari selector failed: {exc2}")
                bbox_hr = list(_select_bbox_manual(initial_bbox, hr_shape_xyz))
                used_mode = "manual"
    bbox_lr = list(_map_hr_to_lr_bbox(bbox_hr, hr_shape=hr_shape_xyz, lr_shape=lr_shape_xyz))

    roi_mask = np.zeros(hr_shape_xyz, dtype=np.uint8)
    roi_mask[bbox_hr[0] : bbox_hr[1], bbox_hr[2] : bbox_hr[3], bbox_hr[4] : bbox_hr[5]] = 1
    mask_vox = int(mask_3d.sum())
    mask_in_roi = int((mask_3d > 0).astype(np.uint8)[roi_mask > 0].sum())

    out = {
        "payload_path": str(payload_path),
        "selection_mode": str(used_mode or "unknown"),
        "temporal_mode": str(args.temporal_mode),
        "frame_index": int(args.frame_index),
        "bbox_hr_xyz": bbox_hr,
        "bbox_hr_size_xyz": [int(bbox_hr[1] - bbox_hr[0]), int(bbox_hr[3] - bbox_hr[2]), int(bbox_hr[5] - bbox_hr[4])],
        "bbox_lr_xyz": bbox_lr,
        "bbox_lr_size_xyz": [int(bbox_lr[1] - bbox_lr[0]), int(bbox_lr[3] - bbox_lr[2]), int(bbox_lr[5] - bbox_lr[4])],
        "hr_shape_xyz": [int(v) for v in hr_shape_xyz],
        "lr_shape_xyz": [int(v) for v in lr_shape_xyz],
        "hr_spacing_mm": hr_spacing,
        "lr_spacing_mm": lr_spacing,
        "mask_voxels_selected_volume": mask_vox,
        "mask_voxels_inside_bbox": mask_in_roi,
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nROI exported:")
    print(f"- {out_path}")
    print(f"- HR bbox xyz: {bbox_hr}")
    print(f"- LR bbox xyz: {bbox_lr}")
    print(f"- mask voxels in ROI: {mask_in_roi}/{mask_vox}")

    default_out_dir = payload_path.parent
    bbox_tag = _bbox_tag(bbox_hr)
    auto_bbox_out_dir = default_out_dir.parent / f"{default_out_dir.name}_{bbox_tag}"
    default_meta = default_out_dir / "inference_metadata.json"
    print("\nSuggested command:")
    print(
        "python code/inference/generate_sr_uq_report.py "
        f"--payload-npz {payload_path} "
        f"--metadata-json {default_meta} "
        f"--out-dir {auto_bbox_out_dir} "
        f"--roi-json {out_path}"
    )

    if args.run_report:
        report_script = (Path(__file__).resolve().parent / "generate_sr_uq_report.py").resolve()
        report_out_dir = Path(args.report_out_dir).resolve() if args.report_out_dir else auto_bbox_out_dir
        report_out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = Path(args.metadata_json).resolve() if args.metadata_json else default_meta

        cmd = [
            sys.executable,
            str(report_script),
            "--payload-npz",
            str(payload_path),
            "--out-dir",
            str(report_out_dir),
            "--roi-json",
            str(out_path),
        ]
        if meta_path.exists():
            cmd.extend(["--metadata-json", str(meta_path)])

        print("\nRunning ROI report:")
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
