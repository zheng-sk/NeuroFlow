#!/usr/bin/env python3
"""Affine and direction QA by ranking sign/permutation candidates for Vx/Vy/Vz."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from neuroflow.visualization.flowviz.common import add_common_preproc_args, add_input_args, load_flow_data_from_args


@dataclass
class FrameBundle:
    t: int
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    mask: np.ndarray
    boundary: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank Vx/Vy/Vz sign/permutation candidates using physics-based metrics "
            "(divergence, wall-normal component) and affine orientation consistency."
        )
    )
    add_input_args(parser)
    add_common_preproc_args(parser)

    parser.add_argument("--all-frames", action="store_true", help="Use all temporal frames.")
    parser.add_argument("--frame-step", type=int, default=4, help="Frame stride when --all-frames is not used.")
    parser.add_argument("--max-frames", type=int, default=6, help="Max number of frames when --all-frames is not used.")
    parser.add_argument("--no-permutations", action="store_true", help="Only test sign flips (no axis permutation).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to print.")

    parser.add_argument("--wall-grad-threshold", type=float, default=1e-4, help="Boundary gradient threshold for wall normal estimation.")
    parser.add_argument("--affine-lambda", type=float, default=0.8, help="Weight of affine-consistency term in final score.")
    parser.add_argument("--export-ranking-csv", type=Path, default=None, help="Optional output CSV with all candidates and metrics.")
    return parser.parse_args()


def _selected_frames(nt: int, args: argparse.Namespace) -> list[int]:
    if nt <= 0:
        return []
    if args.all_frames:
        return list(range(nt))
    step = max(1, int(args.frame_step))
    frames = list(range(0, nt, step))
    cap = max(1, int(args.max_frames))
    return frames[:cap]


def _bbox_from_mask(mask: np.ndarray, margin: int = 2) -> tuple[slice, slice, slice]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, np.array(mask.shape))
    return (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2])))


def _boundary_normals(mask: np.ndarray, affine: np.ndarray, grad_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g0, g1, g2 = np.gradient(mask.astype(np.float32), edge_order=1)
    a_inv_t = np.linalg.inv(np.asarray(affine[:3, :3], dtype=np.float64)).T

    gx = a_inv_t[0, 0] * g0 + a_inv_t[0, 1] * g1 + a_inv_t[0, 2] * g2
    gy = a_inv_t[1, 0] * g0 + a_inv_t[1, 1] * g1 + a_inv_t[1, 2] * g2
    gz = a_inv_t[2, 0] * g0 + a_inv_t[2, 1] * g1 + a_inv_t[2, 2] * g2

    gn = np.sqrt(gx * gx + gy * gy + gz * gz)
    boundary = (mask > 0) & (gn > float(grad_thr))
    denom = np.where(gn > 0, gn, 1.0)
    nx = gx / denom
    ny = gy / denom
    nz = gz / denom
    return boundary, nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)


def _build_frames(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mask_4d: np.ndarray, affine: np.ndarray, frame_ids: list[int], grad_thr: float) -> list[FrameBundle]:
    bundles: list[FrameBundle] = []
    for t in frame_ids:
        mask_t = mask_4d[..., t] > 0
        if not np.any(mask_t):
            continue
        s0, s1, s2 = _bbox_from_mask(mask_t, margin=2)
        vx_t = vx[s0, s1, s2, t].astype(np.float32, copy=False)
        vy_t = vy[s0, s1, s2, t].astype(np.float32, copy=False)
        vz_t = vz[s0, s1, s2, t].astype(np.float32, copy=False)
        m_t = mask_t[s0, s1, s2]
        boundary, nx, ny, nz = _boundary_normals(m_t, affine, grad_thr=grad_thr)
        bundles.append(FrameBundle(t=t, vx=vx_t, vy=vy_t, vz=vz_t, mask=m_t, boundary=boundary, nx=nx, ny=ny, nz=nz))
    return bundles


def _orientation_matrix_from_affines(src_affine: np.ndarray, dst_affine: np.ndarray) -> np.ndarray:
    src_ornt = nib.orientations.io_orientation(src_affine)
    dst_ornt = nib.orientations.io_orientation(dst_affine)
    xform = nib.orientations.ornt_transform(src_ornt, dst_ornt)
    m = np.zeros((3, 3), dtype=np.int8)
    for out_ax in range(3):
        in_ax = int(xform[out_ax, 0])
        sign = int(xform[out_ax, 1])
        m[out_ax, in_ax] = sign
    return m


def _affine_predicted_component_matrix(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    can_img = nib.as_closest_canonical(img)
    return _orientation_matrix_from_affines(img.affine, can_img.affine)


def _all_signed_permutation_mats(with_permutations: bool) -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    perms = list(itertools.permutations([0, 1, 2])) if with_permutations else [(0, 1, 2)]
    for p in perms:
        for s in itertools.product([-1, 1], repeat=3):
            m = np.zeros((3, 3), dtype=np.int8)
            for out_ax, in_ax in enumerate(p):
                m[out_ax, in_ax] = int(s[out_ax])
            mats.append(m)
    return mats


def _matrix_label(m: np.ndarray) -> str:
    names = ["Vx", "Vy", "Vz"]
    out = []
    for ax_name, row in zip(["x", "y", "z"], m):
        in_ax = int(np.argmax(np.abs(row)))
        sign = int(np.sign(row[in_ax]))
        prefix = "+" if sign >= 0 else "-"
        out.append(f"{ax_name}={prefix}{names[in_ax]}")
    return " ".join(out)


def _physics_metrics_for_candidate(
    bundles: list[FrameBundle],
    tmat: np.ndarray,
    affine_3x3: np.ndarray,
    zero_velocity_below: float,
) -> tuple[float, float, float, float]:
    a_inv = np.linalg.inv(np.asarray(affine_3x3, dtype=np.float64))
    div_l1_vals = []
    div_mean_abs_vals = []
    wall_ratio_vals = []
    wall_signed_abs_vals = []

    tfloat = tmat.astype(np.float32)

    for fb in bundles:
        v0 = np.stack([fb.vx, fb.vy, fb.vz], axis=0)
        v = np.einsum("ab,bxyz->axyz", tfloat, v0)
        speed = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        valid = fb.mask.copy()
        if zero_velocity_below > 0:
            valid &= speed >= float(zero_velocity_below)

        if not np.any(valid):
            continue

        d00, d01, d02 = np.gradient(v[0], edge_order=1)
        d10, d11, d12 = np.gradient(v[1], edge_order=1)
        d20, d21, d22 = np.gradient(v[2], edge_order=1)
        div = (
            d00 * a_inv[0, 0] + d01 * a_inv[1, 0] + d02 * a_inv[2, 0]
            + d10 * a_inv[0, 1] + d11 * a_inv[1, 1] + d12 * a_inv[2, 1]
            + d20 * a_inv[0, 2] + d21 * a_inv[1, 2] + d22 * a_inv[2, 2]
        )
        div_mask = div[valid]
        div_l1_vals.append(float(np.mean(np.abs(div_mask))))
        div_mean_abs_vals.append(float(np.abs(np.mean(div_mask))))

        bw = fb.boundary & valid
        if np.any(bw):
            vn = v[0] * fb.nx + v[1] * fb.ny + v[2] * fb.nz
            vn_bw = vn[bw]
            sp_bw = speed[bw]
            wall_ratio_vals.append(float(np.mean(np.abs(vn_bw) / (sp_bw + 1e-6))))
            wall_signed_abs_vals.append(float(np.abs(np.mean(vn_bw))))

    if not div_l1_vals:
        return (1e9, 1e9, 1e9, 1e9)
    return (
        float(np.mean(div_l1_vals)),
        float(np.mean(div_mean_abs_vals)),
        float(np.mean(wall_ratio_vals)) if wall_ratio_vals else 1e9,
        float(np.mean(wall_signed_abs_vals)) if wall_signed_abs_vals else 1e9,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    mu = float(np.mean(values))
    sd = float(np.std(values))
    if sd < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sd


def main() -> None:
    args = parse_args()
    flow = load_flow_data_from_args(args)

    if flow.mask is None:
        raise ValueError("This QA requires --mask to evaluate directions inside CoW.")

    nt = int(flow.vx.shape[-1])
    frame_ids = _selected_frames(nt, args)
    print(f"Frames used for ranking: {frame_ids}")

    bundles = _build_frames(
        vx=flow.vx,
        vy=flow.vy,
        vz=flow.vz,
        mask_4d=flow.mask,
        affine=flow.affine,
        frame_ids=frame_ids,
        grad_thr=float(args.wall_grad_threshold),
    )
    if not bundles:
        raise ValueError("No valid mask frames found after selection.")

    pred_vx = _affine_predicted_component_matrix(flow.paths["vx"])
    pred_vy = _affine_predicted_component_matrix(flow.paths["vy"])
    pred_vz = _affine_predicted_component_matrix(flow.paths["vz"])
    if not (np.array_equal(pred_vx, pred_vy) and np.array_equal(pred_vx, pred_vz)):
        print("Warning: affine orientation transforms differ across Vx/Vy/Vz files.")
    pred_mat = pred_vx

    print("Affine-predicted component transform (if components are voxel-axis encoded):")
    print(pred_mat)
    print(f"Label: {_matrix_label(pred_mat)}")

    candidates = _all_signed_permutation_mats(with_permutations=not args.no_permutations)
    rows = []
    for mat in candidates:
        div_l1, div_mean_abs, wall_ratio, wall_signed_abs = _physics_metrics_for_candidate(
            bundles=bundles,
            tmat=mat,
            affine_3x3=flow.affine[:3, :3],
            zero_velocity_below=float(args.zero_velocity_below),
        )
        affine_dist = float(np.linalg.norm(mat.astype(np.float64) - pred_mat.astype(np.float64)))
        rows.append(
            {
                "label": _matrix_label(mat),
                "matrix": mat.copy(),
                "div_l1": div_l1,
                "div_mean_abs": div_mean_abs,
                "wall_ratio": wall_ratio,
                "wall_signed_abs": wall_signed_abs,
                "affine_dist": affine_dist,
            }
        )

    div_l1_arr = np.array([r["div_l1"] for r in rows], dtype=np.float64)
    div_mean_arr = np.array([r["div_mean_abs"] for r in rows], dtype=np.float64)
    wall_ratio_arr = np.array([r["wall_ratio"] for r in rows], dtype=np.float64)
    wall_signed_arr = np.array([r["wall_signed_abs"] for r in rows], dtype=np.float64)
    affine_arr = np.array([r["affine_dist"] for r in rows], dtype=np.float64)

    physics_score = _zscore(div_l1_arr) + _zscore(div_mean_arr) + _zscore(wall_ratio_arr) + 0.5 * _zscore(wall_signed_arr)
    final_score = physics_score + float(args.affine_lambda) * _zscore(affine_arr)

    for i, r in enumerate(rows):
        r["physics_score"] = float(physics_score[i])
        r["final_score"] = float(final_score[i])

    by_physics = sorted(rows, key=lambda x: x["physics_score"])
    by_final = sorted(rows, key=lambda x: x["final_score"])
    k = max(1, int(args.top_k))

    print("\nTop candidates by physics-only score (lower is better):")
    for rank, r in enumerate(by_physics[:k], start=1):
        print(
            f"{rank:2d}. {r['label']:<24} physics={r['physics_score']:+.4f} "
            f"div_l1={r['div_l1']:.6f} div_mean_abs={r['div_mean_abs']:.6f} "
            f"wall_ratio={r['wall_ratio']:.6f} wall_signed_abs={r['wall_signed_abs']:.6f} "
            f"affine_dist={r['affine_dist']:.3f}"
        )

    print("\nTop candidates by final score (physics + affine):")
    for rank, r in enumerate(by_final[:k], start=1):
        print(
            f"{rank:2d}. {r['label']:<24} final={r['final_score']:+.4f} "
            f"physics={r['physics_score']:+.4f} affine_dist={r['affine_dist']:.3f}"
        )

    best_phys = by_physics[0]
    best_final = by_final[0]
    print("\nBest by physics:")
    print(f"  {best_phys['label']}  (score={best_phys['physics_score']:+.4f})")
    print("Best by final (recommended):")
    print(f"  {best_final['label']}  (score={best_final['final_score']:+.4f})")
    print("Note: a global sign inversion (+/- all components) cannot be resolved reliably without known inlet/outlet reference.")

    if args.export_ranking_csv is not None:
        out = args.export_ranking_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        header = "rank_final,rank_physics,label,div_l1,div_mean_abs,wall_ratio,wall_signed_abs,affine_dist,physics_score,final_score,matrix"
        order_final = {id(r): i + 1 for i, r in enumerate(by_final)}
        order_phys = {id(r): i + 1 for i, r in enumerate(by_physics)}
        lines = [header]
        for r in sorted(rows, key=lambda x: x["final_score"]):
            mat_str = np.array2string(r["matrix"], separator=" ", max_line_width=80).replace("\n", " ")
            lines.append(
                f"{order_final[id(r)]},{order_phys[id(r)]},{r['label']},"
                f"{r['div_l1']:.8f},{r['div_mean_abs']:.8f},{r['wall_ratio']:.8f},"
                f"{r['wall_signed_abs']:.8f},{r['affine_dist']:.8f},"
                f"{r['physics_score']:.8f},{r['final_score']:.8f},\"{mat_str}\""
            )
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Ranking CSV written to: {out.resolve()}")


if __name__ == "__main__":
    main()
