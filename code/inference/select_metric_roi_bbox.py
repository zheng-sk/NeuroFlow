import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector


def _load_payload(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _pick_mask_volume(mask_txyz: np.ndarray, mode: str, frame_index: int) -> np.ndarray:
    if mask_txyz.ndim != 4:
        raise ValueError(f"Expected mask shape [T,X,Y,Z], got {mask_txyz.shape}")

    t = mask_txyz.shape[0]
    if mode == "frame":
        f = int(np.clip(frame_index, 0, t - 1))
        return (mask_txyz[f] > 0.5).astype(np.uint8)
    if mode == "intersection":
        return np.all(mask_txyz > 0.5, axis=0).astype(np.uint8)
    return np.any(mask_txyz > 0.5, axis=0).astype(np.uint8)


def _clip_range(lo: float, hi: float, size: int) -> Tuple[int, int]:
    a = int(np.floor(min(lo, hi)))
    b = int(np.ceil(max(lo, hi)))
    a = max(0, min(a, size - 1))
    b = max(a + 1, min(b, size))
    return a, b


def _select_rect(proj2d: np.ndarray, title: str, xlabel: str, ylabel: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if proj2d.ndim != 2:
        raise ValueError(f"Expected 2D projection, got {proj2d.shape}")
    h, w = proj2d.shape

    selected: Dict[str, Any] = {"extent": None, "cancelled": False}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(proj2d, origin="lower", cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    ax.text(
        0.01,
        1.02,
        "Arrastra para seleccionar. Enter=confirmar, r=reset, Esc=cancelar",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    def _on_select(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        selected["extent"] = (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)

    rect = RectangleSelector(
        ax,
        _on_select,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="pixels",
        interactive=True,
        drag_from_anywhere=True,
    )

    def _on_key(event):
        key = str(event.key).lower() if event.key is not None else ""
        if key == "r":
            rect.set_active(False)
            rect.set_active(True)
            selected["extent"] = None
            fig.canvas.draw_idle()
        elif key == "enter":
            if selected["extent"] is None:
                print("No hay selección todavía.")
                return
            plt.close(fig)
        elif key == "escape":
            selected["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show()

    if selected["cancelled"]:
        raise KeyboardInterrupt("Selección cancelada por usuario.")
    if selected["extent"] is None:
        raise RuntimeError("No se recibió selección de bbox.")

    x0f, x1f, y0f, y1f = selected["extent"]
    col_rng = _clip_range(x0f, x1f, w)
    row_rng = _clip_range(y0f, y1f, h)
    return row_rng, col_rng


def _merge_ranges(name: str, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    lo_i = max(a[0], b[0])
    hi_i = min(a[1], b[1])
    if lo_i < hi_i:
        return lo_i, hi_i
    lo_u = min(a[0], b[0])
    hi_u = max(a[1], b[1])
    print(f"[warn] Rango {name} inconsistente entre vistas. Usando unión: [{lo_u}, {hi_u})")
    return lo_u, hi_u


def _map_hr_to_lr_bbox(bbox_hr: Sequence[int], hr_shape: Sequence[int], lr_shape: Sequence[int]) -> Tuple[int, int, int, int, int, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Selector interactivo de ROI bbox para métricas SR/UQ. "
            "Carga la máscara del payload, permite seleccionar bbox 3D y exporta JSON."
        )
    )
    parser.add_argument("--payload-npz", required=True, help="Path a analysis_payload.npz")
    parser.add_argument("--out-json", required=True, help="Path de salida para bbox ROI (JSON)")
    parser.add_argument(
        "--temporal-mode",
        default="union",
        choices=["union", "intersection", "frame"],
        help="Cómo colapsar máscara temporal para la selección interactiva.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Frame usado cuando --temporal-mode=frame")
    parser.add_argument("--run-report", action="store_true", help="Si se activa, ejecuta generate_sr_uq_report.py al finalizar.")
    parser.add_argument("--metadata-json", default="", help="Metadata JSON para generate_sr_uq_report.py (opcional).")
    parser.add_argument("--report-out-dir", default="", help="Output dir para generate_sr_uq_report.py (opcional).")
    args = parser.parse_args()

    payload_path = Path(args.payload_npz).resolve()
    payload = _load_payload(str(payload_path))
    if "mask" not in payload:
        raise ValueError(f"Payload sin 'mask': {payload_path}")
    if "gt_norm" not in payload or "lr_norm" not in payload:
        raise ValueError("Payload debe incluir 'gt_norm' y 'lr_norm' para mapear HR/LR.")

    mask_txyz = payload["mask"].astype(np.float32)
    hr_shape_xyz = tuple(int(v) for v in payload["gt_norm"].shape[2:])
    lr_shape_xyz = tuple(int(v) for v in payload["lr_norm"].shape[2:])
    hr_spacing = [float(v) for v in payload.get("hr_spacing", np.array([1.0, 1.0, 1.0])).tolist()]
    lr_spacing = [float(v) for v in payload.get("lr_spacing", np.array([1.0, 1.0, 1.0])).tolist()]

    if tuple(mask_txyz.shape[1:]) != hr_shape_xyz:
        raise ValueError(
            f"Mask shape {tuple(mask_txyz.shape[1:])} no coincide con HR shape {hr_shape_xyz}. "
            "La ROI se define en espacio HR."
        )

    mask_3d = _pick_mask_volume(mask_txyz, mode=str(args.temporal_mode), frame_index=int(args.frame_index))
    if int(mask_3d.sum()) == 0:
        raise ValueError("La máscara elegida está vacía.")

    # Proyecciones para bbox 3D:
    # XY: max sobre Z, XZ: max sobre Y, YZ: max sobre X
    proj_xy = np.max(mask_3d, axis=2)  # [X,Y]
    proj_xz = np.max(mask_3d, axis=1)  # [X,Z]
    proj_yz = np.max(mask_3d, axis=0)  # [Y,Z]

    print("Selecciona bbox en vista XY...")
    xy_rows_x, xy_cols_y = _select_rect(proj_xy, "ROI Selection - XY (max over Z)", xlabel="Y", ylabel="X")
    print("Selecciona bbox en vista XZ...")
    xz_rows_x, xz_cols_z = _select_rect(proj_xz, "ROI Selection - XZ (max over Y)", xlabel="Z", ylabel="X")
    print("Selecciona bbox en vista YZ...")
    yz_rows_y, yz_cols_z = _select_rect(proj_yz, "ROI Selection - YZ (max over X)", xlabel="Z", ylabel="Y")

    x0, x1 = _merge_ranges("X", xy_rows_x, xz_rows_x)
    y0, y1 = _merge_ranges("Y", xy_cols_y, yz_rows_y)
    z0, z1 = _merge_ranges("Z", xz_cols_z, yz_cols_z)

    bbox_hr = [int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)]
    bbox_lr = list(_map_hr_to_lr_bbox(bbox_hr, hr_shape=hr_shape_xyz, lr_shape=lr_shape_xyz))

    roi_mask = np.zeros(hr_shape_xyz, dtype=np.uint8)
    roi_mask[x0:x1, y0:y1, z0:z1] = 1
    mask_vox = int(mask_3d.sum())
    mask_in_roi = int((mask_3d > 0).astype(np.uint8)[roi_mask > 0].sum())

    out = {
        "payload_path": str(payload_path),
        "temporal_mode": str(args.temporal_mode),
        "frame_index": int(args.frame_index),
        "bbox_hr_xyz": bbox_hr,
        "bbox_hr_size_xyz": [int(x1 - x0), int(y1 - y0), int(z1 - z0)],
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

    print("\nROI guardada:")
    print(f"- {out_path}")
    print(f"- bbox HR xyz: {bbox_hr}")
    print(f"- bbox LR xyz: {bbox_lr}")
    print(f"- mask voxels in ROI: {mask_in_roi}/{mask_vox}")

    default_out_dir = payload_path.parent
    default_meta = default_out_dir / "inference_metadata.json"
    print("\nComando sugerido para reporte con ROI:")
    print(
        "python code/inference/generate_sr_uq_report.py "
        f"--payload-npz {payload_path} "
        f"--metadata-json {default_meta} "
        f"--out-dir {default_out_dir} "
        f"--roi-json {out_path}"
    )

    if args.run_report:
        report_script = (Path(__file__).resolve().parent / "generate_sr_uq_report.py").resolve()
        report_out_dir = Path(args.report_out_dir).resolve() if args.report_out_dir else default_out_dir
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

        print("\nEjecutando reporte ROI:")
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
