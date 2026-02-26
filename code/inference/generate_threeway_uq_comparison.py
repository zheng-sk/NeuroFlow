import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


REPORT_DPI = 320
FONT_FAMILY = "Times New Roman"
FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
VIOLIN_UPPER_PERCENTILE = 95.0


def _setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="ticks", context="paper", font=FONT_FAMILY, palette="colorblind")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "semibold",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
            "savefig.dpi": REPORT_DPI,
        }
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_metrics_csv(metrics_dir: Path, filename: str) -> Path:
    candidates = [filename]
    if filename == "bland_altman_stats.csv":
        candidates.extend(["Denoising_bland_altman_stats.csv", "denoising_bland_altman_stats.csv"])
    elif filename == "table2_like_temporal_mean.csv":
        candidates.extend(["Denoising_table2_like_temporal_mean.csv", "denoising_table2_like_temporal_mean.csv"])
    elif filename == "table2_like_per_frame_all_slices.csv":
        candidates.extend(
            ["Denoising_table2_like_per_frame_all_slices.csv", "denoising_table2_like_per_frame_all_slices.csv"]
        )
    for name in candidates:
        p = metrics_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing metrics csv in {metrics_dir}: tried {candidates}")


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _load_payload_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(str(path))
    return {k: z[k] for k in z.files}


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _finite(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _mean(x: Sequence[float] | np.ndarray) -> float:
    arr = _finite(x)
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(x: Sequence[float] | np.ndarray) -> float:
    arr = _finite(x)
    return float(np.median(arr)) if arr.size else float("nan")


def _percentile(x: Sequence[float] | np.ndarray, p: float) -> float:
    arr = _finite(x)
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def _winsorize_upper(x: Sequence[float] | np.ndarray, p: float) -> np.ndarray:
    arr = _finite(x)
    if arr.size == 0:
        return arr
    hi = _percentile(arr, p)
    if not np.isfinite(hi):
        return arr
    return np.minimum(arr, hi)


def _legend_handles(methods: List[str], method_colors: Dict[str, str]) -> List[Patch]:
    return [Patch(facecolor=method_colors[m], edgecolor="#111827", linewidth=0.8, label=m) for m in methods]


def _method_key(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _metric_label(domain: str, component: str = "") -> str:
    d = str(domain or "").strip()
    c = str(component or "").strip()
    if d == "speed_intraluminal":
        base = "Speed (Intraluminal)"
    elif d == "flow_temporal":
        base = "Flow (Temporal)"
    elif d == "velocity_component_peak":
        base = "Velocity Peak"
    else:
        base = d.replace("_", " ").title()
    if c and c.lower() not in {"nan", "none"}:
        return f"{base} ({c})"
    return base


def _save_figure(fig: plt.Figure, out_path: Path, top: float = 0.94) -> None:
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def _detect_metrics_dir(report_dir: Path, preferred_mode: str) -> Path:
    metrics_root = report_dir / "metrics"
    direct = metrics_root / preferred_mode
    if direct.exists():
        return direct
    children = [p for p in metrics_root.iterdir() if p.is_dir()] if metrics_root.exists() else []
    if len(children) == 1:
        return children[0]
    raise FileNotFoundError(f"Could not resolve metrics dir under {metrics_root} (preferred={preferred_mode}).")


def _payload_path_from_metrics(metrics_dir: Path) -> Path:
    summary_path = metrics_dir / "summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json in {metrics_dir}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    p = str(data.get("payload_path", "")).strip()
    if not p:
        raise ValueError(f"summary_metrics.json has no payload_path in {summary_path}")
    payload_path = Path(p)
    if not payload_path.exists():
        raise FileNotFoundError(f"payload_path from summary does not exist: {payload_path}")
    return payload_path


def _common_frame_indices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.int32).ravel()
    b = np.asarray(b, dtype=np.int32).ravel()
    b_pos = {int(v): i for i, v in enumerate(b.tolist())}
    common = [int(v) for v in a.tolist() if int(v) in b_pos]
    if not common:
        raise ValueError("No common frame indices between payloads.")
    a_pos = {int(v): i for i, v in enumerate(a.tolist())}
    idx_a = np.asarray([a_pos[v] for v in common], dtype=np.int64)
    idx_b = np.asarray([b_pos[v] for v in common], dtype=np.int64)
    return np.asarray(common, dtype=np.int32), idx_a, idx_b


def _crop_to_xyz(arr: np.ndarray, shape_xyz: Tuple[int, int, int]) -> np.ndarray:
    sx, sy, sz = [int(v) for v in shape_xyz]
    lead = (slice(None),) * (arr.ndim - 3)
    return np.asarray(arr[lead + (slice(0, sx), slice(0, sy), slice(0, sz))], dtype=arr.dtype)


def _common_xyz(arrs: List[np.ndarray]) -> Tuple[int, int, int]:
    if not arrs:
        raise ValueError("No arrays provided for common xyz resolution.")
    sx = min(int(a.shape[-3]) for a in arrs)
    sy = min(int(a.shape[-2]) for a in arrs)
    sz = min(int(a.shape[-1]) for a in arrs)
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"Invalid common xyz shape: {(sx, sy, sz)}")
    return sx, sy, sz


def _build_threeway_raw_bundle(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    reference_label: str,
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
) -> Dict[str, Any]:
    dns_payload_path = _payload_path_from_metrics(dns_metrics_dir)
    sr_payload_path = _payload_path_from_metrics(sr_metrics_dir)
    dns = _load_payload_npz(dns_payload_path)
    sr = _load_payload_npz(sr_payload_path)

    dns_gt = np.asarray(dns["gt_norm"], dtype=np.float32)
    dns_lr = np.asarray(dns["lr_norm"], dtype=np.float32)
    dns_pred = np.asarray(dns["pred_norm"], dtype=np.float32)
    dns_mask = np.asarray(dns["mask"], dtype=np.float32)
    dns_venc = np.asarray(dns["venc"], dtype=np.float32).ravel()
    sr_pred = np.asarray(sr["pred_norm"], dtype=np.float32)
    sr_mask = np.asarray(sr["mask"], dtype=np.float32)
    sr_venc = np.asarray(sr["venc"], dtype=np.float32).ravel()

    dns_frames = np.asarray(dns.get("frame_indices", np.arange(dns_gt.shape[0], dtype=np.int32)), dtype=np.int32)
    sr_frames = np.asarray(sr.get("frame_indices", np.arange(sr_pred.shape[0], dtype=np.int32)), dtype=np.int32)
    common_frames, idx_dns, idx_sr = _common_frame_indices(dns_frames, sr_frames)

    dns_gt = dns_gt[idx_dns]
    dns_lr = dns_lr[idx_dns]
    dns_pred = dns_pred[idx_dns]
    dns_mask = dns_mask[idx_dns]
    dns_venc = dns_venc[idx_dns]
    sr_pred = sr_pred[idx_sr]
    sr_mask = sr_mask[idx_sr]
    sr_venc = sr_venc[idx_sr]

    dns_ref_uv = dns_gt[:, :3]
    dns_base_uv = dns_lr[:, :3]
    dns_den_uv = dns_pred[:, :3]
    sr_sup_uv = sr_pred[:, :3]

    # Broadcast VENC across [T, C, X, Y, Z].
    venc_dns = dns_venc.reshape((-1,) + (1,) * (dns_ref_uv.ndim - 1))
    venc_sr = sr_venc.reshape((-1,) + (1,) * (sr_sup_uv.ndim - 1))
    ref_phys_uv = dns_ref_uv * venc_dns
    base_phys_uv = dns_base_uv * venc_dns
    den_phys_uv = dns_den_uv * venc_dns
    sup_phys_uv = sr_sup_uv * venc_sr

    xyz = _common_xyz([dns_ref_uv, dns_base_uv, dns_den_uv, sr_sup_uv, dns_mask, sr_mask])
    dns_ref_uv = _crop_to_xyz(dns_ref_uv, xyz)
    dns_base_uv = _crop_to_xyz(dns_base_uv, xyz)
    dns_den_uv = _crop_to_xyz(dns_den_uv, xyz)
    sr_sup_uv = _crop_to_xyz(sr_sup_uv, xyz)
    ref_phys_uv = _crop_to_xyz(ref_phys_uv, xyz)
    base_phys_uv = _crop_to_xyz(base_phys_uv, xyz)
    den_phys_uv = _crop_to_xyz(den_phys_uv, xyz)
    sup_phys_uv = _crop_to_xyz(sup_phys_uv, xyz)
    dns_mask = _crop_to_xyz(dns_mask, xyz)
    sr_mask = _crop_to_xyz(sr_mask, xyz)

    mask = (dns_mask > 0.5) & (sr_mask > 0.5)
    if int(mask.sum()) == 0:
        mask = (dns_mask > 0.5) | (sr_mask > 0.5)
    if int(mask.sum()) == 0:
        raise ValueError("Unified mask is empty when building 3-way raw bundle.")

    def _to_4ch(uvw: np.ndarray) -> np.ndarray:
        mag = np.sqrt(np.maximum(0.0, np.sum(uvw.astype(np.float64) ** 2, axis=1))).astype(np.float32)
        return np.concatenate([uvw.astype(np.float32), mag[:, None]], axis=1)

    norm_4ch = {
        reference_label: _to_4ch(dns_ref_uv),
        baseline_label: _to_4ch(dns_base_uv),
        denoised_label: _to_4ch(dns_den_uv),
        superres_label: _to_4ch(sr_sup_uv),
    }
    phys_4ch = {
        reference_label: _to_4ch(ref_phys_uv),
        baseline_label: _to_4ch(base_phys_uv),
        denoised_label: _to_4ch(den_phys_uv),
        superres_label: _to_4ch(sup_phys_uv),
    }
    return {
        "frame_indices": common_frames,
        "mask": mask.astype(bool),
        "norm_4ch": norm_4ch,
        "phys_4ch": phys_4ch,
        "payload_paths": {
            "dns": str(dns_payload_path),
            "sr": str(sr_payload_path),
        },
    }

def _load_run_context(metrics_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(metrics_dir / "run_context.csv")
    return rows[0] if rows else {}


def _select_method_rows(
    rows: List[Dict[str, Any]],
    baseline_method_name: str,
    model_method_name: str,
    target_baseline_label: str,
    target_model_label: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        m = str(r.get("method", "")).strip()
        if m == baseline_method_name:
            rr = dict(r)
            rr["method"] = target_baseline_label
            out.append(rr)
        elif m == model_method_name:
            rr = dict(r)
            rr["method"] = target_model_label
            out.append(rr)
    return out


def _merge_velocity_metrics(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    dns_ctx: Dict[str, Any],
    sr_ctx: Dict[str, Any],
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in ["mean_velocity_metrics.csv", "peak_velocity_metrics.csv", "correlation_metrics.csv"]:
        dns_rows = _read_csv(dns_metrics_dir / name)
        sr_rows = _read_csv(sr_metrics_dir / name)

        dns_base = str(dns_ctx.get("baseline_label", "3T"))
        dns_model = str(dns_ctx.get("sr_label", "3T Denoised"))
        sr_base = str(sr_ctx.get("baseline_label", "3T"))
        sr_model = str(sr_ctx.get("sr_label", "3T SR"))

        merged = []
        merged.extend(_select_method_rows(dns_rows, dns_base, dns_model, baseline_label, denoised_label))
        merged.extend(_select_method_rows(sr_rows, sr_base, sr_model, baseline_label, superres_label))

        # Keep one baseline source (from denoising run) to avoid duplicated 3T rows.
        baseline_seen: set[Tuple[Any, ...]] = set()
        deduped: List[Dict[str, Any]] = []
        for r in merged:
            key = (
                r.get("domain", ""),
                r.get("region", ""),
                r.get("component", ""),
                r.get("method", ""),
                r.get("frame_payload_index", ""),
                r.get("frame_source_index", ""),
            )
            if str(r.get("method", "")) == baseline_label:
                if key in baseline_seen:
                    continue
                baseline_seen.add(key)
            deduped.append(r)
        out[name] = deduped
    return out


def _merge_flow_average_metrics(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    dns_ctx: Dict[str, Any],
    sr_ctx: Dict[str, Any],
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
) -> List[Dict[str, Any]]:
    dns_rows = _read_csv(dns_metrics_dir / "flow_average_metrics.csv")
    sr_rows = _read_csv(sr_metrics_dir / "flow_average_metrics.csv")
    dns_base = str(dns_ctx.get("baseline_label", "3T"))
    dns_model = str(dns_ctx.get("sr_label", "3T Denoised"))
    sr_base = str(sr_ctx.get("baseline_label", "3T"))
    sr_model = str(sr_ctx.get("sr_label", "3T SR"))

    merged = []
    merged.extend(_select_method_rows(dns_rows, dns_base, dns_model, baseline_label, denoised_label))
    merged.extend(_select_method_rows(sr_rows, sr_base, sr_model, baseline_label, superres_label))

    baseline_seen = False
    out: List[Dict[str, Any]] = []
    for r in merged:
        if str(r.get("method", "")) == baseline_label:
            if baseline_seen:
                continue
            baseline_seen = True
        out.append(r)
    return out


def _merge_flow_per_frame_metrics(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    dns_ctx: Dict[str, Any],
    sr_ctx: Dict[str, Any],
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
    reference_label: str,
) -> List[Dict[str, Any]]:
    dns_rows = _read_csv(dns_metrics_dir / "flow_metrics_per_frame.csv")
    sr_rows = _read_csv(sr_metrics_dir / "flow_metrics_per_frame.csv")
    dns_base = str(dns_ctx.get("baseline_label", "3T"))
    dns_model = str(dns_ctx.get("sr_label", "3T Denoised"))
    dns_ref = str(dns_ctx.get("reference_label", "7T"))
    sr_model = str(sr_ctx.get("sr_label", "3T SR"))
    sr_ref = str(sr_ctx.get("reference_label", "7T"))

    out: List[Dict[str, Any]] = []
    # Keep baseline/reference from denoising side, add SR model from superresolution side.
    for r in dns_rows:
        rr = dict(r)
        m = str(rr.get("method", "")).strip()
        if m == dns_ref:
            rr["method"] = reference_label
            out.append(rr)
        elif m == dns_base:
            rr["method"] = baseline_label
            out.append(rr)
        elif m == dns_model:
            rr["method"] = denoised_label
            out.append(rr)
    for r in sr_rows:
        rr = dict(r)
        m = str(rr.get("method", "")).strip()
        if m == sr_model:
            rr["method"] = superres_label
            out.append(rr)
        elif m == sr_ref:
            # add missing reference rows only if a frame wasn't present from dns
            rr["method"] = reference_label
            key = (rr.get("frame_payload_index", ""), rr.get("frame_source_index", ""), rr.get("method", ""))
            if not any((x.get("frame_payload_index", ""), x.get("frame_source_index", ""), x.get("method", "")) == key for x in out):
                out.append(rr)
    return out


def _merge_voxel_distribution_stats(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    dns_ctx: Dict[str, Any],
    sr_ctx: Dict[str, Any],
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
    reference_label: str,
) -> List[Dict[str, Any]]:
    dns_rows = _read_csv(dns_metrics_dir / "voxel_distribution_stats.csv")
    sr_rows = _read_csv(sr_metrics_dir / "voxel_distribution_stats.csv")
    dns_base = str(dns_ctx.get("baseline_label", "3T"))
    dns_model = str(dns_ctx.get("sr_label", "3T Denoised"))
    dns_ref = str(dns_ctx.get("reference_label", "7T"))
    sr_model = str(sr_ctx.get("sr_label", "3T SR"))

    out: List[Dict[str, Any]] = []
    for r in dns_rows:
        rr = dict(r)
        m = str(rr.get("method", "")).strip()
        if m == dns_ref:
            rr["method"] = reference_label
            out.append(rr)
        elif m == dns_base:
            rr["method"] = baseline_label
            out.append(rr)
        elif m == dns_model:
            rr["method"] = denoised_label
            out.append(rr)
    for r in sr_rows:
        rr = dict(r)
        m = str(rr.get("method", "")).strip()
        if m == sr_model:
            rr["method"] = superres_label
            out.append(rr)
    return out


def _merge_significance(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    denoised_label: str,
    superres_label: str,
) -> List[Dict[str, Any]]:
    dns_rows = _read_csv(dns_metrics_dir / "significance_pvalues.csv")
    sr_rows = _read_csv(sr_metrics_dir / "significance_pvalues.csv")
    dns_map = {(r.get("analysis", ""), r.get("region", "")): r for r in dns_rows}
    sr_map = {(r.get("analysis", ""), r.get("region", "")): r for r in sr_rows}
    keys = sorted(set(dns_map.keys()) | set(sr_map.keys()))
    out: List[Dict[str, Any]] = []
    for k in keys:
        d = dns_map.get(k, {})
        s = sr_map.get(k, {})
        out.append(
            {
                "analysis": k[0],
                "region": k[1],
                f"wilcoxon_p_value_{denoised_label.lower().replace(' ', '_')}": d.get("wilcoxon_p_value", ""),
                f"n_voxels_{denoised_label.lower().replace(' ', '_')}": d.get("n_voxels", ""),
                f"wilcoxon_p_value_{superres_label.lower().replace(' ', '_')}": s.get("wilcoxon_p_value", ""),
                f"n_voxels_{superres_label.lower().replace(' ', '_')}": s.get("n_voxels", ""),
            }
        )
    return out


def _merge_single_method_table(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    filename: str,
    dns_ctx: Dict[str, Any],
    sr_ctx: Dict[str, Any],
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
    keep_baseline_once: bool = True,
) -> List[Dict[str, Any]]:
    dns_rows = _read_csv(_resolve_metrics_csv(dns_metrics_dir, filename))
    sr_rows = _read_csv(_resolve_metrics_csv(sr_metrics_dir, filename))
    dns_base = str(dns_ctx.get("baseline_label", "3T"))
    dns_model = str(dns_ctx.get("sr_label", "3T Denoised"))
    sr_base = str(sr_ctx.get("baseline_label", "3T"))
    sr_model = str(sr_ctx.get("sr_label", "3T SR"))
    merged = []
    merged.extend(_select_method_rows(dns_rows, dns_base, dns_model, baseline_label, denoised_label))
    merged.extend(_select_method_rows(sr_rows, sr_base, sr_model, baseline_label, superres_label))
    if not keep_baseline_once:
        return merged
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in merged:
        m = str(r.get("method", ""))
        if m != baseline_label:
            out.append(r)
            continue
        key = tuple(sorted((k, str(v)) for k, v in r.items() if k != "method"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _merge_table2_like(
    dns_metrics_dir: Path,
    sr_metrics_dir: Path,
    baseline_label: str,
    denoised_label: str,
    superres_label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dns_all = _read_csv(dns_metrics_dir / "table2_like_all_slices.csv")
    sr_all = _read_csv(sr_metrics_dir / "table2_like_all_slices.csv")
    dns_tm = _read_csv(_resolve_metrics_csv(dns_metrics_dir, "table2_like_temporal_mean.csv"))
    sr_tm = _read_csv(_resolve_metrics_csv(sr_metrics_dir, "table2_like_temporal_mean.csv"))

    def _key_all(r: Dict[str, Any]) -> Tuple[str, str]:
        return (str(r.get("slice_index", "")), str(r.get("variable", "")))

    def _key_tm(r: Dict[str, Any]) -> Tuple[str, str]:
        return (str(r.get("slice_index", "")), str(r.get("variable", "")))

    dns_all_map = {_key_all(r): r for r in dns_all}
    sr_all_map = {_key_all(r): r for r in sr_all}
    dns_tm_map = {_key_tm(r): r for r in dns_tm}
    sr_tm_map = {_key_tm(r): r for r in sr_tm}

    all_keys = sorted(set(dns_all_map.keys()) & set(sr_all_map.keys()))
    tm_keys = sorted(set(dns_tm_map.keys()) & set(sr_tm_map.keys()))

    merged_all: List[Dict[str, Any]] = []
    baseline_delta_all: List[float] = []
    ref_delta_all: List[float] = []
    for k in all_keys:
        d = dns_all_map[k]
        s = sr_all_map[k]
        ref_d = _to_float(d.get("ref"))
        ref_s = _to_float(s.get("ref"))
        base_d = _to_float(d.get("baseline"))
        base_s = _to_float(s.get("baseline"))
        den = _to_float(d.get("sr"))
        sr = _to_float(s.get("sr"))
        if np.isfinite(base_d) and np.isfinite(base_s):
            baseline_delta_all.append(abs(base_d - base_s))
        if np.isfinite(ref_d) and np.isfinite(ref_s):
            ref_delta_all.append(abs(ref_d - ref_s))
        ref_use = ref_d if np.isfinite(ref_d) else ref_s
        base_use = base_d if np.isfinite(base_d) else base_s
        ae_base = abs(base_use - ref_use) if np.isfinite(base_use) and np.isfinite(ref_use) else float("nan")
        ae_den = abs(den - ref_use) if np.isfinite(den) and np.isfinite(ref_use) else float("nan")
        ae_sr = abs(sr - ref_use) if np.isfinite(sr) and np.isfinite(ref_use) else float("nan")
        merged_all.append(
            {
                "slice_index": k[0],
                "variable": k[1],
                "ref": ref_use,
                f"value_{baseline_label.lower().replace(' ', '_')}": base_use,
                f"value_{denoised_label.lower().replace(' ', '_')}": den,
                f"value_{superres_label.lower().replace(' ', '_')}": sr,
                f"abs_error_{baseline_label.lower().replace(' ', '_')}": ae_base,
                f"abs_error_{denoised_label.lower().replace(' ', '_')}": ae_den,
                f"abs_error_{superres_label.lower().replace(' ', '_')}": ae_sr,
            }
        )

    merged_tm: List[Dict[str, Any]] = []
    baseline_delta_tm: List[float] = []
    ref_delta_tm: List[float] = []
    for k in tm_keys:
        d = dns_tm_map[k]
        s = sr_tm_map[k]
        ref_d = _to_float(d.get("ref_mean_over_frames"))
        ref_s = _to_float(s.get("ref_mean_over_frames"))
        base_d = _to_float(d.get("baseline_mean_over_frames"))
        base_s = _to_float(s.get("baseline_mean_over_frames"))
        den = _to_float(d.get("sr_mean_over_frames"))
        sr = _to_float(s.get("sr_mean_over_frames"))
        n_frames = int(_to_float(d.get("n_frames"), default=0))
        if np.isfinite(base_d) and np.isfinite(base_s):
            baseline_delta_tm.append(abs(base_d - base_s))
        if np.isfinite(ref_d) and np.isfinite(ref_s):
            ref_delta_tm.append(abs(ref_d - ref_s))
        ref_use = ref_d if np.isfinite(ref_d) else ref_s
        base_use = base_d if np.isfinite(base_d) else base_s
        ae_base = abs(base_use - ref_use) if np.isfinite(base_use) and np.isfinite(ref_use) else float("nan")
        ae_den = abs(den - ref_use) if np.isfinite(den) and np.isfinite(ref_use) else float("nan")
        ae_sr = abs(sr - ref_use) if np.isfinite(sr) and np.isfinite(ref_use) else float("nan")
        merged_tm.append(
            {
                "slice_index": k[0],
                "variable": k[1],
                "n_frames": n_frames,
                "ref_mean_over_frames": ref_use,
                f"value_{baseline_label.lower().replace(' ', '_')}_mean_over_frames": base_use,
                f"value_{denoised_label.lower().replace(' ', '_')}_mean_over_frames": den,
                f"value_{superres_label.lower().replace(' ', '_')}_mean_over_frames": sr,
                f"abs_error_{baseline_label.lower().replace(' ', '_')}_mean_over_frames": ae_base,
                f"abs_error_{denoised_label.lower().replace(' ', '_')}_mean_over_frames": ae_den,
                f"abs_error_{superres_label.lower().replace(' ', '_')}_mean_over_frames": ae_sr,
            }
        )

    qc = {
        "n_intersection_rows_table2_all_slices": len(all_keys),
        "n_intersection_rows_table2_temporal_mean": len(tm_keys),
        "baseline_abs_delta_all_slices_mean": _mean(baseline_delta_all),
        "baseline_abs_delta_temporal_mean_mean": _mean(baseline_delta_tm),
        "reference_abs_delta_all_slices_mean": _mean(ref_delta_all),
        "reference_abs_delta_temporal_mean_mean": _mean(ref_delta_tm),
    }
    return merged_all, merged_tm, qc


def _table2_abs_error_summary(
    merged_rows: List[Dict[str, Any]],
    methods: List[str],
    mean_suffix: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_var: Dict[str, Dict[str, List[float]]] = {}
    for r in merged_rows:
        var = str(r.get("variable", "")).strip()
        if not var:
            continue
        by_var.setdefault(var, {m: [] for m in methods})
        for m in methods:
            key = f"abs_error_{m.lower().replace(' ', '_')}{mean_suffix}"
            v = _to_float(r.get(key))
            if np.isfinite(v):
                by_var[var][m].append(v)
    for var in sorted(by_var.keys()):
        for m in methods:
            vals = by_var[var][m]
            out.append(
                {
                    "variable": var,
                    "method": m,
                    "n": int(len(vals)),
                    "mean_abs_error": _mean(vals),
                    "median_abs_error": _median(vals),
                    "p95_abs_error": _percentile(vals, 95.0),
                }
            )
    return out


def _save_bar_velocity_errors(
    rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    region_order = ["core", "wall", "intraluminal"]
    metric_defs = [("mae", "MAE [m/s]"), ("rmse", "RMSE [m/s]")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax, (metric_key, ylab) in zip(axes, metric_defs):
        x = np.arange(len(region_order), dtype=np.float64)
        width = 0.24
        for i, m in enumerate(methods):
            vals = []
            for reg in region_order:
                vv = [
                    _to_float(r.get(metric_key))
                    for r in rows
                    if str(r.get("region", "")) == reg and str(r.get("method", "")) == m
                ]
                vals.append(_mean(vv))
            ax.bar(
                x + (i - 1) * width,
                [0.0 if not np.isfinite(v) else v for v in vals],
                width=width,
                label=m,
                color=method_colors[m],
                edgecolor="#111827",
                linewidth=0.8,
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([r.capitalize() for r in region_order])
        ax.set_ylabel(ylab)
        ax.set_xlabel("Region")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
    axes[0].set_title("MAE")
    axes[1].set_title("RMSE")
    fig.suptitle(title)
    fig.legend(_legend_handles(methods, method_colors), methods, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save_figure(fig, out_path)


def _save_velocity_metric_single(
    rows: List[Dict[str, Any]],
    out_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    region_order = ["core", "wall", "intraluminal"]
    x = np.arange(len(region_order), dtype=np.float64)
    width = 0.24
    vals_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for m in methods:
        for reg in region_order:
            vv = [
                _to_float(r.get(metric_key))
                for r in rows
                if str(r.get("region", "")) == reg and str(r.get("method", "")) == m
            ]
            vals_by_method[m].append(_mean(vv))
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.9))
    for i, m in enumerate(methods):
        ax.bar(
            x + (i - 1) * width,
            [0.0 if not np.isfinite(v) else v for v in vals_by_method[m]],
            width=width,
            color=method_colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
            label=m,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in region_order])
    ax.set_xlabel("Region")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_flow_bar(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    metric_defs = [
        ("mean_abs_err_ml_s", "Mean Absolute Error [ml/s]"),
        ("mean_relative_err_pct", "Mean Relative Error [%]"),
        ("rmse_over_time_ml_s", "RMSE over Time [ml/s]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    x = np.arange(len(methods), dtype=np.float64)
    for ax, (metric_key, title) in zip(axes, metric_defs):
        vals = []
        for m in methods:
            vv = [_to_float(r.get(metric_key)) for r in rows if str(r.get("method", "")) == m]
            vals.append(_mean(vv))
        ax.bar(
            x,
            [0.0 if not np.isfinite(v) else v for v in vals],
            color=[method_colors[m] for m in methods],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=14)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
    fig.suptitle("Flow Metrics: Three-way Comparison")
    fig.legend(_legend_handles(methods, method_colors), methods, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save_figure(fig, out_path)


def _save_flow_profile_over_time(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
    reference_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.0))
    order = [reference_label] + methods
    colors = {reference_label: "#374151", **method_colors}
    for m in order:
        by_frame: Dict[int, List[float]] = {}
        for r in rows:
            if str(r.get("method", "")) != m:
                continue
            f = int(_to_float(r.get("frame_source_index"), default=-1))
            q = _to_float(r.get("Q_ml_s"))
            if f < 0 or not np.isfinite(q):
                continue
            by_frame.setdefault(f, []).append(q)
        if not by_frame:
            continue
        xs = sorted(by_frame.keys())
        ys = [_mean(by_frame[k]) for k in xs]
        ax.plot(xs, ys, marker="o", markersize=4.0, linewidth=2.0, color=colors[m], label=m)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Flow [ml/s]")
    ax.set_title("Flow Profile Over Time (Three-way)")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(loc="best", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_table2_violin_family(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
    family_tag: str,
    family_title: str,
    prefixes: List[str],
    temporal_mean: bool,
) -> None:
    if sns is None:
        return
    velocity_pref = ["Mean velocity [m/s]", "SD velocity [m/s]", "Skewness velocity", "Kurtosis velocity"]
    vorticity_pref = ["Mean vorticity [1/s]", "SD vorticity [1/s]", "Skewness vorticity", "Kurtosis vorticity"]
    pref = velocity_pref if family_tag == "velocity" else vorticity_pref
    vars_present = sorted(set(str(r.get("variable", "")) for r in rows))
    var_order = [v for v in pref if v in vars_present and any(v.startswith(p) for p in prefixes)]
    if not var_order:
        return
    suffix = "_mean_over_frames" if temporal_mean else ""
    disp = {v: v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order}

    x_vals: List[str] = []
    y_vals: List[float] = []
    h_vals: List[str] = []
    for v in var_order:
        for m in methods:
            key = f"abs_error_{m.lower().replace(' ', '_')}{suffix}"
            vals = [_to_float(r.get(key)) for r in rows if str(r.get("variable", "")) == v]
            vals = _winsorize_upper(vals, VIOLIN_UPPER_PERCENTILE)
            if vals.size == 0:
                continue
            x_vals.extend([disp[v]] * int(vals.size))
            y_vals.extend(vals.tolist())
            h_vals.extend([m] * int(vals.size))
    if not y_vals:
        return

    fig_w = max(7.0, 1.35 * len(var_order) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.6))
    sns.violinplot(
        x=x_vals,
        y=y_vals,
        hue=h_vals,
        order=[disp[v] for v in var_order],
        cut=0,
        inner="quartile",
        linewidth=1.0,
        palette=method_colors,
        ax=ax,
        dodge=True,
    )
    tprefix = "Temporal-Mean " if temporal_mean else ""
    ax.set_title(f"{tprefix}Absolute Error Distribution ({family_title})", pad=14)
    ax.set_xlabel("Variable")
    ax.set_ylabel(f"Absolute Error vs 7T (winsorized at P{int(VIOLIN_UPPER_PERCENTILE)})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(
        handles=_legend_handles(methods, method_colors),
        labels=methods,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        frameon=False,
    )
    ax.tick_params(axis="x", rotation=12)
    _save_figure(fig, out_path, top=0.90)


def _save_temporal_mean_bar_by_variable(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    var_order = sorted(set(str(r.get("variable", "")) for r in rows))
    x = np.arange(len(var_order), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(max(11.5, 0.9 * len(var_order) + 3.0), 5.8))
    for i, m in enumerate(methods):
        vals = []
        key = f"abs_error_{m.lower().replace(' ', '_')}_mean_over_frames"
        for v in var_order:
            vv = [_to_float(r.get(key)) for r in rows if str(r.get("variable", "")) == v]
            vals.append(_mean(vv))
        ax.bar(
            x + (i - 1) * width,
            [0.0 if not np.isfinite(v) else v for v in vals],
            width=width,
            label=m,
            color=method_colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order], rotation=18, ha="right")
    ax.set_ylabel("Mean Absolute Error vs 7T")
    ax.set_title("Temporal-Mean Absolute Error by Variable (Three-way)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, frameon=False, loc="upper right", title="Method")
    _save_figure(fig, out_path)


def _save_temporal_mean_bar_by_variable_family(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
    family: str,
) -> None:
    var_order = [v for v in sorted(set(str(r.get("variable", "")) for r in rows)) if family.lower() in v.lower()]
    if not var_order:
        return
    x = np.arange(len(var_order), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(max(8.8, 1.0 * len(var_order) + 3.2), 5.6))
    for i, m in enumerate(methods):
        vals = []
        key = f"abs_error_{m.lower().replace(' ', '_')}_mean_over_frames"
        for v in var_order:
            vv = [_to_float(r.get(key)) for r in rows if str(r.get("variable", "")) == v]
            vals.append(_mean(vv))
        ax.bar(
            x + (i - 1) * width,
            [0.0 if not np.isfinite(v) else v for v in vals],
            width=width,
            label=m,
            color=method_colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order], rotation=16, ha="right")
    ax.set_ylabel("Mean Absolute Error vs 7T")
    ax.set_title(f"Temporal-Mean Absolute Error by {family.title()} (Three-way)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, frameon=False, loc="upper right", title="Method")
    _save_figure(fig, out_path)


def _save_relative_error_bars(
    rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    region_order = ["core", "wall", "intraluminal"]
    x = np.arange(len(region_order), dtype=np.float64)
    width = 0.24
    vals_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for m in methods:
        for reg in region_order:
            vv = [
                _to_float(r.get("relative_error_pct"))
                for r in rows
                if str(r.get("region", "")) == reg and str(r.get("method", "")) == m
            ]
            vals_by_method[m].append(_mean(vv))

    raw = [v for m in methods for v in vals_by_method[m] if np.isfinite(v)]
    cap = _percentile(raw, 95.0) if raw else float("nan")
    use_cap = bool(np.isfinite(cap) and cap > 0)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.0))
    for i, m in enumerate(methods):
        vals = vals_by_method[m]
        if use_cap:
            vals = [min(v, cap) if np.isfinite(v) else v for v in vals]
        ax.bar(
            x + (i - 1) * width,
            [0.0 if not np.isfinite(v) else v for v in vals],
            width=width,
            color=method_colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
            label=m,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in region_order])
    ylab = "Relative Error [%] vs 7T"
    if use_cap:
        ylab = f"Relative Error [%] vs 7T (capped at P95={cap:.2g})"
    ax.set_ylabel(ylab)
    ax.set_xlabel("Region")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_slice_abs_error_boxplot(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    vars_of_interest = ["Mean velocity [m/s]", "SD velocity [m/s]", "Mean vorticity [1/s]"]
    var_order = [v for v in vars_of_interest if any(str(r.get("variable", "")) == v for r in rows)]
    if not var_order:
        return
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.4))
    x = np.arange(len(var_order), dtype=np.float64)
    width = 0.24
    for i, m in enumerate(methods):
        pos = x + (i - 1) * width
        for xi, var_name in enumerate(var_order):
            key = f"abs_error_{_method_key(m)}"
            vals = [_to_float(r.get(key)) for r in rows if str(r.get("variable", "")) == var_name]
            vals = _finite(vals)
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                vals,
                positions=[pos[xi]],
                widths=width * 0.9,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#111827", "linewidth": 1.1},
                boxprops={"edgecolor": "#111827", "linewidth": 0.9},
                whiskerprops={"color": "#111827", "linewidth": 0.9},
                capprops={"color": "#111827", "linewidth": 0.9},
                flierprops={"marker": "o", "markersize": 3.2, "markerfacecolor": method_colors[m], "markeredgecolor": "#111827", "alpha": 0.65},
            )
            for box in bp["boxes"]:
                box.set_facecolor(method_colors[m])
                box.set_alpha(0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" [m/s]", "").replace(" [1/s]", "") for v in var_order], rotation=12)
    ax.set_xlabel("Hemodynamic Parameter")
    ax.set_ylabel("Absolute Error vs 7T")
    ax.set_title("Distribution of Absolute Errors Across All Slices")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_flow_abs_error_over_time(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 5.0))
    for m in methods:
        by_frame: Dict[int, List[float]] = {}
        for r in rows:
            if str(r.get("method", "")) != m:
                continue
            f = int(_to_float(r.get("frame_source_index"), default=-1))
            e = _to_float(r.get("abs_err_vs_ref_profile_ml_s"))
            if f < 0 or not np.isfinite(e):
                continue
            by_frame.setdefault(f, []).append(e)
        if not by_frame:
            continue
        xs = sorted(by_frame.keys())
        ys = [_mean(by_frame[k]) for k in xs]
        ax.plot(xs, ys, marker="o", markersize=4.0, linewidth=2.0, color=method_colors[m], label=m)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Absolute Flow Error [ml/s] vs 7T profile")
    ax.set_title("Temporal Flow Absolute Error (Three-way)")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_correlation_metric_bar(
    rows: List[Dict[str, Any]],
    out_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    # Filter low-support points for robustness.
    filt = [r for r in rows if _to_float(r.get("n")) > 100 and str(r.get("method", "")) in methods]
    labels = []
    for r in filt:
        lbl = _metric_label(str(r.get("domain", "")), str(r.get("component", "")))
        if lbl not in labels:
            labels.append(lbl)
    if not labels:
        return
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig_w = max(10.0, 1.05 * len(labels) + 3.8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.2))
    for i, m in enumerate(methods):
        vals = []
        for lbl in labels:
            vv = [
                _to_float(r.get(metric_key))
                for r in filt
                if str(r.get("method", "")) == m and _metric_label(str(r.get("domain", "")), str(r.get("component", ""))) == lbl
            ]
            vals.append(_mean(vv))
        ax.bar(
            x + (i - 1) * width,
            [0.0 if not np.isfinite(v) else v for v in vals],
            width=width,
            color=method_colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
            label=m,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods, method_colors), labels=methods, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_voxel_std_bars(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    reference_label: str,
    method_colors: Dict[str, str],
) -> None:
    methods_all = [reference_label] + methods
    colors = dict(method_colors)
    colors[reference_label] = "#374151"
    channel_order = []
    for r in rows:
        ch = str(r.get("channel", "")).strip()
        if ch and ch not in channel_order:
            channel_order.append(ch)
    if not channel_order:
        return
    x = np.arange(len(channel_order), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(1, 1, figsize=(max(8.2, 1.1 * len(channel_order) + 4.2), 5.2))
    for i, m in enumerate(methods_all):
        vals = []
        for ch in channel_order:
            vv = [_to_float(r.get("std")) for r in rows if str(r.get("channel", "")) == ch and str(r.get("method", "")) == m]
            vals.append(_mean(vv))
        ax.bar(
            x + (i - (len(methods_all) - 1) / 2.0) * width,
            [0.0 if not np.isfinite(v) else v for v in vals],
            width=width,
            color=colors[m],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
            label=m,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(channel_order)
    ax.set_xlabel("Velocity Component Channel")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Voxel Distribution Std Dev (Three-way)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(handles=_legend_handles(methods_all, colors), labels=methods_all, loc="upper right", frameon=False, title="Method")
    _save_figure(fig, out_path)


def _save_voxel_hist_full(
    norm_4ch: Dict[str, np.ndarray],
    mask: np.ndarray,
    out_dir: Path,
    method_order: List[str],
    method_colors: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    channel_names = ["u", "v", "w", "mag"]
    stats_rows: List[Dict[str, Any]] = []
    saved_names: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def _sample_vals(vals: np.ndarray, seed: int) -> np.ndarray:
        vals = _finite(vals)
        if vals.size <= 220000:
            return vals
        rng = np.random.default_rng(seed)
        idx = rng.choice(vals.size, size=220000, replace=False)
        return vals[idx]

    # Combined 2x2 histogram panel.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    axes = axes.ravel()
    for ci, ch in enumerate(channel_names):
        ax = axes[ci]
        for mi, m in enumerate(method_order):
            arr = np.asarray(norm_4ch[m][:, ci], dtype=np.float32)
            vals = _sample_vals(arr[mask], seed=41 + 13 * ci + mi)
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=120,
                density=True,
                alpha=0.30,
                color=method_colors[m],
                edgecolor="none",
                label=m,
            )
            stats_rows.append(
                {
                    "channel": ch,
                    "method": m,
                    "count": int(vals.size),
                    "mean": _mean(vals),
                    "std": float(np.std(vals)) if vals.size > 1 else float("nan"),
                    "median": _median(vals),
                    "p05": _percentile(vals, 5.0),
                    "p95": _percentile(vals, 95.0),
                    "min": float(np.min(vals)) if vals.size else float("nan"),
                    "max": float(np.max(vals)) if vals.size else float("nan"),
                }
            )
        ax.set_title(f"{ch.upper()} in-mask voxel distribution")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Density")
        ax.grid(axis="y", linestyle="--", alpha=0.30)
        if sns is not None:
            sns.despine(ax=ax)
    handles = _legend_handles(method_order, method_colors)
    fig.legend(handles, method_order, loc="upper center", ncol=min(4, len(method_order)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Voxel-value distribution inside vessel mask (Three-way)", y=0.985)
    combined_name = "three_way_voxel_histogram_in_mask_full.png"
    _save_figure(fig, out_dir / combined_name, top=0.93)
    saved_names.append(combined_name)

    # Per-channel standalone histograms.
    for ci, ch in enumerate(channel_names):
        fig_ch, ax_ch = plt.subplots(1, 1, figsize=(7.0, 4.8))
        for mi, m in enumerate(method_order):
            arr = np.asarray(norm_4ch[m][:, ci], dtype=np.float32)
            vals = _sample_vals(arr[mask], seed=97 + 17 * ci + mi)
            if vals.size == 0:
                continue
            ax_ch.hist(
                vals,
                bins=120,
                density=True,
                alpha=0.30,
                color=method_colors[m],
                edgecolor="none",
                label=m,
            )
        ax_ch.set_title(f"{ch.upper()} in-mask voxel distribution")
        ax_ch.set_xlabel("Normalized value")
        ax_ch.set_ylabel("Density")
        ax_ch.grid(axis="y", linestyle="--", alpha=0.30)
        if sns is not None:
            sns.despine(ax=ax_ch)
        ax_ch.legend(handles=_legend_handles(method_order, method_colors), labels=method_order, frameon=False, loc="upper right", title="Method")
        name = f"three_way_voxel_histogram_{ch}_in_mask_full.png"
        _save_figure(fig_ch, out_dir / name)
        saved_names.append(name)
    return stats_rows, saved_names


def _plot_ba_panel(
    ax,
    ref_vals: np.ndarray,
    test_vals: np.ndarray,
    ref_label: str,
    test_label: str,
    color: str,
    seed: int,
) -> Dict[str, float]:
    r0 = np.asarray(ref_vals, dtype=np.float64).ravel()
    t0 = np.asarray(test_vals, dtype=np.float64).ravel()
    n = min(r0.size, t0.size)
    r0 = r0[:n]
    t0 = t0[:n]
    m = np.isfinite(r0) & np.isfinite(t0)
    r = r0[m]
    t = t0[m]
    n = int(r.size)
    if n < 3:
        ax.set_title(f"{test_label} vs {ref_label} (insufficient)")
        return {"n": float(n), "bias": float("nan"), "sd_diff": float("nan"), "loa_low": float("nan"), "loa_high": float("nan")}
    mean_v = 0.5 * (t + r)
    diff_v = t - r
    if mean_v.size > 60000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(mean_v.size, size=60000, replace=False)
    else:
        idx = np.arange(mean_v.size)
    ax.scatter(mean_v[idx], diff_v[idx], s=18, marker="o", alpha=0.34, color=color, edgecolors="white", linewidths=0.25)
    bias = float(np.mean(diff_v))
    sd = float(np.std(diff_v, ddof=1)) if diff_v.size > 1 else float("nan")
    loa_low = bias - 1.96 * sd if np.isfinite(sd) else float("nan")
    loa_high = bias + 1.96 * sd if np.isfinite(sd) else float("nan")
    ax.axhline(bias, color="#b91c1c", linestyle="-", linewidth=1.2)
    ax.axhline(loa_low, color="#111827", linestyle="--", linewidth=1.0)
    ax.axhline(loa_high, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title(f"{test_label} vs {ref_label}")
    ax.set_xlabel("Mean [m/s]")
    ax.set_ylabel("Difference [m/s]")
    ax.grid(axis="both", linestyle=":", alpha=0.25)
    if sns is not None:
        sns.despine(ax=ax)
    return {
        "n": float(n),
        "bias": float(bias),
        "sd_diff": float(sd),
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
    }


def _save_bland_altman_full(
    phys_4ch: Dict[str, np.ndarray],
    mask: np.ndarray,
    out_dir: Path,
    reference_label: str,
    methods: List[str],
    method_colors: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_names: List[str] = []
    rows: List[Dict[str, Any]] = []
    comp_idx = {"u": 0, "v": 1, "w": 2, "mag": 3}
    ref_arr = np.asarray(phys_4ch[reference_label], dtype=np.float32)

    def _vals(arr4: np.ndarray, cidx: int) -> np.ndarray:
        return np.asarray(arr4[:, cidx], dtype=np.float32)[mask]

    # Intraluminal speed (mag) full + singles.
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.8), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for i, m in enumerate(methods):
        test = _vals(phys_4ch[m], comp_idx["mag"])
        ref = _vals(ref_arr, comp_idx["mag"])
        st = _plot_ba_panel(axes[i], ref, test, reference_label, m, method_colors[m], seed=111 + i)
        rows.append(
            {
                "domain": "speed_intraluminal",
                "region": "intraluminal",
                "component": "",
                "method": m,
                "frame_payload_index": -1,
                "frame_source_index": -1,
                **st,
            }
        )
        fig_s, ax_s = plt.subplots(1, 1, figsize=(5.2, 4.8))
        _plot_ba_panel(ax_s, ref, test, reference_label, m, method_colors[m], seed=211 + i)
        s_name = f"three_way_bland_altman_speed_intraluminal_{_method_key(m)}.png"
        _save_figure(fig_s, out_dir / s_name)
        saved_names.append(s_name)
    fig.suptitle("Bland-Altman: intraluminal speed (all frames, in-mask voxels)")
    full_speed_name = "three_way_bland_altman_speed_intraluminal_full.png"
    _save_figure(fig, out_dir / full_speed_name)
    saved_names.append(full_speed_name)

    # Velocity components all-frames.
    for ci_name, ci in comp_idx.items():
        fig_c, axes_c = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.8), sharey=True)
        if len(methods) == 1:
            axes_c = [axes_c]
        for i, m in enumerate(methods):
            test = _vals(phys_4ch[m], ci)
            ref = _vals(ref_arr, ci)
            st = _plot_ba_panel(axes_c[i], ref, test, reference_label, m, method_colors[m], seed=301 + 17 * ci + i)
            rows.append(
                {
                    "domain": "velocity_component_all_frames",
                    "region": "intraluminal",
                    "component": ci_name,
                    "method": m,
                    "frame_payload_index": -1,
                    "frame_source_index": -1,
                    **st,
                }
            )
            fig_s, ax_s = plt.subplots(1, 1, figsize=(5.2, 4.8))
            _plot_ba_panel(ax_s, ref, test, reference_label, m, method_colors[m], seed=401 + 19 * ci + i)
            s_name = f"three_way_bland_altman_velocity_component_{ci_name}_{_method_key(m)}.png"
            _save_figure(fig_s, out_dir / s_name)
            saved_names.append(s_name)
        fig_c.suptitle(f"Bland-Altman: {ci_name.upper()} (all frames, in-mask voxels)")
        full_name = f"three_way_bland_altman_velocity_component_{ci_name}_allframes_full.png"
        _save_figure(fig_c, out_dir / full_name)
        saved_names.append(full_name)
    return rows, saved_names


def _save_significance_plot(
    rows: List[Dict[str, Any]],
    out_path: Path,
    denoised_label: str,
    superres_label: str,
) -> None:
    k_den = f"wilcoxon_p_value_{_method_key(denoised_label)}"
    k_sr = f"wilcoxon_p_value_{_method_key(superres_label)}"
    if not rows or k_den not in rows[0] or k_sr not in rows[0]:
        return
    labels = [f"{r.get('analysis', '')} | {str(r.get('region', '')).capitalize()}" for r in rows]
    vals_den = []
    vals_sr = []
    for r in rows:
        p_d = max(_to_float(r.get(k_den), default=1.0), 1e-300)
        p_s = max(_to_float(r.get(k_sr), default=1.0), 1e-300)
        vals_den.append(-np.log10(p_d))
        vals_sr.append(-np.log10(p_s))
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig_w = max(12.0, 0.7 * len(labels) + 4.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.2))
    ax.bar(x - 0.5 * width, vals_den, width=width, color="#0072B2", edgecolor="#111827", linewidth=0.8, alpha=0.9, label=denoised_label)
    ax.bar(x + 0.5 * width, vals_sr, width=width, color="#009E73", edgecolor="#111827", linewidth=0.8, alpha=0.9, label=superres_label)
    ax.axhline(-np.log10(0.05), color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Wilcoxon Significance (Baseline Error vs Enhanced Error)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if sns is not None:
        sns.despine(ax=ax)
    ax.legend(frameon=False, loc="upper right", title="Method")
    _save_figure(fig, out_path)


def _save_flow_peak_bars(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    metric_defs = [
        ("peak_abs_err_ml_s", "Peak Absolute Error [ml/s]"),
        ("peak_relative_err_pct", "Peak Relative Error [%]"),
        ("rmse_over_time_ml_s", "RMSE over Time [ml/s]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9))
    x = np.arange(len(methods), dtype=np.float64)
    for ax, (mk, title) in zip(axes, metric_defs):
        vals = []
        for m in methods:
            vv = [_to_float(r.get(mk)) for r in rows if str(r.get("method", "")) == m]
            vals.append(_mean(vv))
        ax.bar(
            x,
            [0.0 if not np.isfinite(v) else v for v in vals],
            color=[method_colors[m] for m in methods],
            edgecolor="#111827",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
    fig.suptitle("Flow Peak Metrics (Three-way)")
    fig.legend(_legend_handles(methods, method_colors), methods, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    _save_figure(fig, out_path)


def _save_bland_altman_summary(
    rows: List[Dict[str, Any]],
    out_path: Path,
    methods: List[str],
    method_colors: Dict[str, str],
) -> None:
    filt = [r for r in rows if str(r.get("method", "")) in methods]
    labels = []
    for r in filt:
        lbl = _metric_label(str(r.get("domain", "")), str(r.get("component", "")))
        if lbl not in labels:
            labels.append(lbl)
    if not labels:
        return
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig_w = max(11.5, 1.0 * len(labels) + 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.3))
    specs = [
        ("bias", "Bias [method - ref]"),
        ("loa_width", "LoA Width"),
    ]
    for ax, (mk, ylab) in zip(axes, specs):
        for i, m in enumerate(methods):
            vals = []
            for lbl in labels:
                vv = []
                for r in filt:
                    if str(r.get("method", "")) != m:
                        continue
                    if _metric_label(str(r.get("domain", "")), str(r.get("component", ""))) != lbl:
                        continue
                    if mk == "loa_width":
                        lo = _to_float(r.get("loa_low"))
                        hi = _to_float(r.get("loa_high"))
                        if np.isfinite(lo) and np.isfinite(hi):
                            vv.append(hi - lo)
                    else:
                        vv.append(_to_float(r.get("bias")))
                vals.append(_mean(vv))
            ax.bar(
                x + (i - 1) * width,
                [0.0 if not np.isfinite(v) else v for v in vals],
                width=width,
                color=method_colors[m],
                edgecolor="#111827",
                linewidth=0.8,
                alpha=0.9,
                label=m,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if sns is not None:
            sns.despine(ax=ax)
    axes[0].set_title("Bland-Altman Bias")
    axes[1].set_title("Bland-Altman Limits of Agreement Width")
    fig.suptitle("Bland-Altman Summary (Three-way)", y=0.965)
    fig.legend(_legend_handles(methods, method_colors), methods, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.085))
    _save_figure(fig, out_path, top=0.84)


def _write_html_report(
    out_dir: Path,
    fig_rel_paths: List[Tuple[str, str]],
    metrics_rel_paths: List[str],
    qc: Dict[str, Any],
    labels: Dict[str, str],
    method_colors: Dict[str, str],
) -> None:
    color_items = "".join(
        [
            f'<span class="pill"><span class="swatch" style="background:{method_colors[k]};"></span>{k}</span>'
            for k in [labels["baseline"], labels["denoised"], labels["superres"]]
            if k in method_colors
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Three-way UQ Comparison Report</title>
  <style>
    body {{ font-family: 'Times New Roman', Times, serif; margin: 24px; color: #111827; }}
    h1, h2, h3 {{ margin: 0.3rem 0 0.6rem; }}
    p {{ margin: 0.3rem 0 0.8rem; }}
    .muted {{ color: #4b5563; }}
    .pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #f3f4f6; margin-right: 8px; }}
    .swatch {{ display:inline-block; width:10px; height:10px; border:1px solid #111827; margin-right:6px; vertical-align:middle; }}
    .imgbox {{ margin: 0.9rem 0 1.5rem; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
    code {{ background: #f9fafb; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Three-way UQ Comparison Report</h1>
  <p class="muted">Comparison against 7T reference for <b>{labels['baseline']}</b>, <b>{labels['denoised']}</b>, and <b>{labels['superres']}</b>.</p>
  <p>
    <span class="pill">DPI: {REPORT_DPI}</span>
    <span class="pill">Font: Times New Roman</span>
    <span class="pill">Violin cap: P{int(VIOLIN_UPPER_PERCENTILE)}</span>
  </p>
  <p>{color_items}</p>

  <h2>QC Checks</h2>
  <pre>{json.dumps(qc, indent=2)}</pre>

  <h2>Figures</h2>
  {"".join([f'<div class="imgbox"><h3>{t}</h3><img src="{p}" alt="{t}" /></div>' for t, p in fig_rel_paths])}

  <h2>Saved Tables</h2>
  <ul>
    {"".join([f"<li><code>{p}</code></li>" for p in metrics_rel_paths])}
  </ul>
</body>
</html>
"""
    (out_dir / "report_three_way.html").write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate three-way comparison report: 3T vs 7T, 3T Denoised vs 7T, 3T Superresolution vs 7T.")
    p.add_argument("--dns-report-dir", required=True, help="Path to denoising report root (contains metrics/denoising).")
    p.add_argument("--sr-report-dir", required=True, help="Path to superresolution report root (contains metrics/superresolution).")
    p.add_argument("--out-dir", required=True, help="Output directory for merged 3-way report.")
    p.add_argument("--baseline-label", default="3T", help="Display label for baseline method.")
    p.add_argument("--denoised-label", default="3T Denoised", help="Display label for denoised method.")
    p.add_argument("--superres-label", default="3T Superresolution", help="Display label for superresolution method.")
    args = p.parse_args()

    _setup_style()

    dns_report_dir = Path(args.dns_report_dir)
    sr_report_dir = Path(args.sr_report_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dns_metrics_dir = _detect_metrics_dir(dns_report_dir, preferred_mode="denoising")
    sr_metrics_dir = _detect_metrics_dir(sr_report_dir, preferred_mode="superresolution")

    dns_ctx = _load_run_context(dns_metrics_dir)
    sr_ctx = _load_run_context(sr_metrics_dir)

    baseline_label = str(args.baseline_label).strip() or "3T"
    denoised_label = str(args.denoised_label).strip() or "3T Denoised"
    superres_label = str(args.superres_label).strip() or "3T Superresolution"
    reference_label = str(dns_ctx.get("reference_label", "7T")).strip() or "7T"
    methods = [baseline_label, denoised_label, superres_label]
    method_colors = {
        baseline_label: "#D55E00",
        denoised_label: "#0072B2",
        superres_label: "#009E73",
    }

    metrics_dir = out_dir / "metrics" / "three_way"
    figs_dir = out_dir / "figures" / "three_way"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    velocity_sets = _merge_velocity_metrics(
        dns_metrics_dir,
        sr_metrics_dir,
        dns_ctx,
        sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
    )
    flow_avg = _merge_flow_average_metrics(
        dns_metrics_dir,
        sr_metrics_dir,
        dns_ctx,
        sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
    )
    flow_per_frame = _merge_flow_per_frame_metrics(
        dns_metrics_dir,
        sr_metrics_dir,
        dns_ctx,
        sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
        reference_label=reference_label,
    )
    voxel_stats = _merge_voxel_distribution_stats(
        dns_metrics_dir,
        sr_metrics_dir,
        dns_ctx,
        sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
        reference_label=reference_label,
    )
    sig_rows = _merge_significance(
        dns_metrics_dir,
        sr_metrics_dir,
        denoised_label=denoised_label,
        superres_label=superres_label,
    )
    raw_bundle: Optional[Dict[str, Any]] = None
    raw_voxel_rows: List[Dict[str, Any]] = []
    raw_voxel_figs: List[str] = []
    raw_ba_rows: List[Dict[str, Any]] = []
    raw_ba_figs: List[str] = []
    raw_bundle_error = ""
    try:
        raw_bundle = _build_threeway_raw_bundle(
            dns_metrics_dir=dns_metrics_dir,
            sr_metrics_dir=sr_metrics_dir,
            reference_label=reference_label,
            baseline_label=baseline_label,
            denoised_label=denoised_label,
            superres_label=superres_label,
        )
    except Exception as exc:
        raw_bundle_error = str(exc)
    flow_peak_rows = _merge_single_method_table(
        dns_metrics_dir,
        sr_metrics_dir,
        filename="flow_peak_metrics.csv",
        dns_ctx=dns_ctx,
        sr_ctx=sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
        keep_baseline_once=True,
    )
    bland_altman_rows = _merge_single_method_table(
        dns_metrics_dir,
        sr_metrics_dir,
        filename="bland_altman_stats.csv",
        dns_ctx=dns_ctx,
        sr_ctx=sr_ctx,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
        keep_baseline_once=True,
    )
    table2_all, table2_tm, table2_qc = _merge_table2_like(
        dns_metrics_dir,
        sr_metrics_dir,
        baseline_label=baseline_label,
        denoised_label=denoised_label,
        superres_label=superres_label,
    )

    _write_csv(
        metrics_dir / "mean_velocity_metrics_three_way.csv",
        velocity_sets["mean_velocity_metrics.csv"],
        [
            "domain",
            "region",
            "method",
            "frame_payload_index",
            "frame_source_index",
            "n",
            "mae",
            "rmse",
            "relative_error_pct",
            "cosine_similarity",
        ],
    )
    _write_csv(
        metrics_dir / "peak_velocity_metrics_three_way.csv",
        velocity_sets["peak_velocity_metrics.csv"],
        [
            "domain",
            "region",
            "method",
            "frame_payload_index",
            "frame_source_index",
            "n",
            "mae",
            "rmse",
            "relative_error_pct",
            "cosine_similarity",
        ],
    )
    _write_csv(
        metrics_dir / "correlation_metrics_three_way.csv",
        velocity_sets["correlation_metrics.csv"],
        [
            "domain",
            "region",
            "component",
            "method",
            "frame_payload_index",
            "frame_source_index",
            "n",
            "slope",
            "intercept",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
            "r2_linear",
            "rmse",
            "bias",
        ],
    )
    _write_csv(
        metrics_dir / "flow_average_metrics_three_way.csv",
        flow_avg,
        [
            "domain",
            "method",
            "mean_ref_flow_ml_s",
            "mean_method_flow_ml_s",
            "mean_abs_err_ml_s",
            "mean_relative_err_pct",
            "rmse_over_time_ml_s",
        ],
    )
    _write_csv(
        metrics_dir / "flow_peak_metrics_three_way.csv",
        flow_peak_rows,
        [
            "domain",
            "method",
            "peak_frame_payload_index",
            "peak_frame_source_index",
            "peak_ref_flow_ml_s",
            "peak_method_flow_ml_s",
            "peak_abs_err_ml_s",
            "peak_relative_err_pct",
            "rmse_over_time_ml_s",
            "relative_err_over_time_pct",
        ],
    )
    _write_csv(
        metrics_dir / "bland_altman_stats_three_way.csv",
        bland_altman_rows,
        [
            "domain",
            "region",
            "component",
            "method",
            "frame_payload_index",
            "frame_source_index",
            "n",
            "bias",
            "sd_diff",
            "loa_low",
            "loa_high",
        ],
    )
    _write_csv(
        metrics_dir / "flow_metrics_per_frame_three_way.csv",
        flow_per_frame,
        [
            "frame_payload_index",
            "frame_source_index",
            "method",
            "flow_method",
            "Q_ml_s",
            "abs_err_vs_qref_ml_s",
            "abs_err_vs_ref_profile_ml_s",
        ],
    )
    _write_csv(
        metrics_dir / "voxel_distribution_stats_three_way.csv",
        voxel_stats,
        ["channel", "method", "count", "mean", "std", "median", "p05", "p95", "min", "max"],
    )
    if raw_bundle is not None:
        raw_voxel_rows, raw_voxel_figs = _save_voxel_hist_full(
            norm_4ch=raw_bundle["norm_4ch"],
            mask=np.asarray(raw_bundle["mask"], dtype=bool),
            out_dir=figs_dir,
            method_order=[reference_label, baseline_label, denoised_label, superres_label],
            method_colors={
                reference_label: "#374151",
                baseline_label: method_colors[baseline_label],
                denoised_label: method_colors[denoised_label],
                superres_label: method_colors[superres_label],
            },
        )
        _write_csv(
            metrics_dir / "voxel_distribution_stats_three_way_unified_raw.csv",
            raw_voxel_rows,
            ["channel", "method", "count", "mean", "std", "median", "p05", "p95", "min", "max"],
        )
        raw_ba_rows, raw_ba_figs = _save_bland_altman_full(
            phys_4ch=raw_bundle["phys_4ch"],
            mask=np.asarray(raw_bundle["mask"], dtype=bool),
            out_dir=figs_dir,
            reference_label=reference_label,
            methods=methods,
            method_colors=method_colors,
        )
        _write_csv(
            metrics_dir / "bland_altman_stats_three_way_unified_raw.csv",
            raw_ba_rows,
            [
                "domain",
                "region",
                "component",
                "method",
                "frame_payload_index",
                "frame_source_index",
                "n",
                "bias",
                "sd_diff",
                "loa_low",
                "loa_high",
            ],
        )
    sig_cols = list(sig_rows[0].keys()) if sig_rows else ["analysis", "region"]
    _write_csv(metrics_dir / "significance_pvalues_three_way.csv", sig_rows, sig_cols)

    all_cols = list(table2_all[0].keys()) if table2_all else [
        "slice_index",
        "variable",
        "ref",
        f"value_{baseline_label.lower().replace(' ', '_')}",
        f"value_{denoised_label.lower().replace(' ', '_')}",
        f"value_{superres_label.lower().replace(' ', '_')}",
        f"abs_error_{baseline_label.lower().replace(' ', '_')}",
        f"abs_error_{denoised_label.lower().replace(' ', '_')}",
        f"abs_error_{superres_label.lower().replace(' ', '_')}",
    ]
    tm_cols = list(table2_tm[0].keys()) if table2_tm else [
        "slice_index",
        "variable",
        "n_frames",
        "ref_mean_over_frames",
        f"value_{baseline_label.lower().replace(' ', '_')}_mean_over_frames",
        f"value_{denoised_label.lower().replace(' ', '_')}_mean_over_frames",
        f"value_{superres_label.lower().replace(' ', '_')}_mean_over_frames",
        f"abs_error_{baseline_label.lower().replace(' ', '_')}_mean_over_frames",
        f"abs_error_{denoised_label.lower().replace(' ', '_')}_mean_over_frames",
        f"abs_error_{superres_label.lower().replace(' ', '_')}_mean_over_frames",
    ]
    _write_csv(metrics_dir / "table2_like_all_slices_three_way.csv", table2_all, all_cols)
    _write_csv(metrics_dir / "table2_like_temporal_mean_three_way.csv", table2_tm, tm_cols)

    summary_all = _table2_abs_error_summary(table2_all, methods=methods, mean_suffix="")
    summary_tm = _table2_abs_error_summary(table2_tm, methods=methods, mean_suffix="_mean_over_frames")
    _write_csv(
        metrics_dir / "table2_abs_error_summary_three_way.csv",
        summary_all,
        ["variable", "method", "n", "mean_abs_error", "median_abs_error", "p95_abs_error"],
    )
    _write_csv(
        metrics_dir / "table2_temporal_mean_abs_error_summary_three_way.csv",
        summary_tm,
        ["variable", "method", "n", "mean_abs_error", "median_abs_error", "p95_abs_error"],
    )

    (metrics_dir / "merge_qc_three_way.json").write_text(json.dumps(table2_qc, indent=2), encoding="utf-8")

    # Figures
    mean_rows = velocity_sets["mean_velocity_metrics.csv"]
    peak_rows = velocity_sets["peak_velocity_metrics.csv"]
    _save_bar_velocity_errors(
        mean_rows,
        figs_dir / "three_way_mean_velocity_mae_rmse.png",
        title="Mean Velocity Error (Three-way)",
        methods=methods,
        method_colors=method_colors,
    )
    _save_velocity_metric_single(
        mean_rows,
        figs_dir / "three_way_mean_velocity_mae.png",
        metric_key="mae",
        title="Mean Velocity MAE (Three-way)",
        y_label="MAE [m/s]",
        methods=methods,
        method_colors=method_colors,
    )
    _save_velocity_metric_single(
        mean_rows,
        figs_dir / "three_way_mean_velocity_rmse.png",
        metric_key="rmse",
        title="Mean Velocity RMSE (Three-way)",
        y_label="RMSE [m/s]",
        methods=methods,
        method_colors=method_colors,
    )
    _save_relative_error_bars(
        mean_rows,
        figs_dir / "three_way_mean_velocity_relative_error_pct.png",
        title="Mean Velocity Relative Error (Three-way)",
        methods=methods,
        method_colors=method_colors,
    )
    _save_bar_velocity_errors(
        peak_rows,
        figs_dir / "three_way_peak_velocity_mae_rmse.png",
        title="Peak Velocity Error (Three-way)",
        methods=methods,
        method_colors=method_colors,
    )
    _save_velocity_metric_single(
        peak_rows,
        figs_dir / "three_way_peak_velocity_mae.png",
        metric_key="mae",
        title="Peak Velocity MAE (Three-way)",
        y_label="MAE [m/s]",
        methods=methods,
        method_colors=method_colors,
    )
    _save_velocity_metric_single(
        peak_rows,
        figs_dir / "three_way_peak_velocity_rmse.png",
        metric_key="rmse",
        title="Peak Velocity RMSE (Three-way)",
        y_label="RMSE [m/s]",
        methods=methods,
        method_colors=method_colors,
    )
    _save_relative_error_bars(
        peak_rows,
        figs_dir / "three_way_peak_velocity_relative_error_pct.png",
        title="Peak Velocity Relative Error (Three-way)",
        methods=methods,
        method_colors=method_colors,
    )
    _save_slice_abs_error_boxplot(
        table2_all,
        figs_dir / "three_way_slice_abs_error_boxplot.png",
        methods=methods,
        method_colors=method_colors,
    )
    _save_flow_bar(flow_avg, figs_dir / "three_way_flow_error_bars.png", methods=methods, method_colors=method_colors)
    _save_flow_profile_over_time(
        flow_per_frame,
        figs_dir / "three_way_flow_profile_over_time.png",
        methods=methods,
        method_colors=method_colors,
        reference_label=reference_label,
    )
    _save_flow_abs_error_over_time(
        flow_per_frame,
        figs_dir / "three_way_flow_abs_error_over_time.png",
        methods=methods,
        method_colors=method_colors,
    )
    _save_correlation_metric_bar(
        velocity_sets["correlation_metrics.csv"],
        figs_dir / "three_way_correlation_pearson_r.png",
        metric_key="pearson_r",
        title="Correlation (Pearson r) Three-way",
        y_label="Pearson r",
        methods=methods,
        method_colors=method_colors,
    )
    _save_correlation_metric_bar(
        velocity_sets["correlation_metrics.csv"],
        figs_dir / "three_way_correlation_rmse.png",
        metric_key="rmse",
        title="Correlation (RMSE) Three-way",
        y_label="RMSE",
        methods=methods,
        method_colors=method_colors,
    )
    _save_voxel_std_bars(
        voxel_stats,
        figs_dir / "three_way_voxel_distribution_std.png",
        methods=methods,
        reference_label=reference_label,
        method_colors=method_colors,
    )
    _save_significance_plot(
        sig_rows,
        figs_dir / "three_way_significance_pvalues.png",
        denoised_label=denoised_label,
        superres_label=superres_label,
    )
    _save_flow_peak_bars(
        flow_peak_rows,
        figs_dir / "three_way_flow_peak_metrics.png",
        methods=methods,
        method_colors=method_colors,
    )
    _save_bland_altman_summary(
        bland_altman_rows,
        figs_dir / "three_way_bland_altman_summary.png",
        methods=methods,
        method_colors=method_colors,
    )
    _save_temporal_mean_bar_by_variable(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_by_variable.png",
        methods=methods,
        method_colors=method_colors,
    )
    _save_temporal_mean_bar_by_variable_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_by_velocity.png",
        methods=methods,
        method_colors=method_colors,
        family="velocity",
    )
    _save_temporal_mean_bar_by_variable_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_by_vorticity.png",
        methods=methods,
        method_colors=method_colors,
        family="vorticity",
    )

    _save_table2_violin_family(
        table2_all,
        figs_dir / "three_way_abs_error_violin_velocity_scale.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="velocity",
        family_title="Velocity: Mean & SD",
        prefixes=["Mean", "SD"],
        temporal_mean=False,
    )
    _save_table2_violin_family(
        table2_all,
        figs_dir / "three_way_abs_error_violin_velocity_shape.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="velocity",
        family_title="Velocity: Skewness & Kurtosis",
        prefixes=["Skewness", "Kurtosis"],
        temporal_mean=False,
    )
    _save_table2_violin_family(
        table2_all,
        figs_dir / "three_way_abs_error_violin_vorticity_scale.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="vorticity",
        family_title="Vorticity: Mean & SD",
        prefixes=["Mean", "SD"],
        temporal_mean=False,
    )
    _save_table2_violin_family(
        table2_all,
        figs_dir / "three_way_abs_error_violin_vorticity_shape.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="vorticity",
        family_title="Vorticity: Skewness & Kurtosis",
        prefixes=["Skewness", "Kurtosis"],
        temporal_mean=False,
    )
    _save_table2_violin_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_violin_velocity_scale.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="velocity",
        family_title="Temporal-Mean Velocity: Mean & SD",
        prefixes=["Mean", "SD"],
        temporal_mean=True,
    )
    _save_table2_violin_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_violin_velocity_shape.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="velocity",
        family_title="Temporal-Mean Velocity: Skewness & Kurtosis",
        prefixes=["Skewness", "Kurtosis"],
        temporal_mean=True,
    )
    _save_table2_violin_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_violin_vorticity_scale.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="vorticity",
        family_title="Temporal-Mean Vorticity: Mean & SD",
        prefixes=["Mean", "SD"],
        temporal_mean=True,
    )
    _save_table2_violin_family(
        table2_tm,
        figs_dir / "three_way_temporal_mean_abs_error_violin_vorticity_shape.png",
        methods=methods,
        method_colors=method_colors,
        family_tag="vorticity",
        family_title="Temporal-Mean Vorticity: Skewness & Kurtosis",
        prefixes=["Skewness", "Kurtosis"],
        temporal_mean=True,
    )

    figure_entries = [
        ("Mean Velocity Error (MAE/RMSE)", "figures/three_way/three_way_mean_velocity_mae_rmse.png"),
        ("Mean Velocity MAE", "figures/three_way/three_way_mean_velocity_mae.png"),
        ("Mean Velocity RMSE", "figures/three_way/three_way_mean_velocity_rmse.png"),
        ("Mean Velocity Relative Error [%]", "figures/three_way/three_way_mean_velocity_relative_error_pct.png"),
        ("Peak Velocity Error (MAE/RMSE)", "figures/three_way/three_way_peak_velocity_mae_rmse.png"),
        ("Peak Velocity MAE", "figures/three_way/three_way_peak_velocity_mae.png"),
        ("Peak Velocity RMSE", "figures/three_way/three_way_peak_velocity_rmse.png"),
        ("Peak Velocity Relative Error [%]", "figures/three_way/three_way_peak_velocity_relative_error_pct.png"),
        ("Slice-wise Absolute Error Boxplot", "figures/three_way/three_way_slice_abs_error_boxplot.png"),
        ("Flow Error Bars", "figures/three_way/three_way_flow_error_bars.png"),
        ("Flow Profile Over Time", "figures/three_way/three_way_flow_profile_over_time.png"),
        ("Temporal Flow Absolute Error", "figures/three_way/three_way_flow_abs_error_over_time.png"),
        ("Correlation Pearson r", "figures/three_way/three_way_correlation_pearson_r.png"),
        ("Correlation RMSE", "figures/three_way/three_way_correlation_rmse.png"),
        ("Voxel Distribution Std Dev", "figures/three_way/three_way_voxel_distribution_std.png"),
        ("Flow Peak Metrics", "figures/three_way/three_way_flow_peak_metrics.png"),
        ("Bland-Altman Summary", "figures/three_way/three_way_bland_altman_summary.png"),
        ("Wilcoxon Significance", "figures/three_way/three_way_significance_pvalues.png"),
        ("Temporal-Mean Absolute Error by Variable", "figures/three_way/three_way_temporal_mean_abs_error_by_variable.png"),
        ("Temporal-Mean Absolute Error by Velocity", "figures/three_way/three_way_temporal_mean_abs_error_by_velocity.png"),
        ("Temporal-Mean Absolute Error by Vorticity", "figures/three_way/three_way_temporal_mean_abs_error_by_vorticity.png"),
        ("Absolute Error Violin (Velocity Mean/SD)", "figures/three_way/three_way_abs_error_violin_velocity_scale.png"),
        ("Absolute Error Violin (Velocity Skewness/Kurtosis)", "figures/three_way/three_way_abs_error_violin_velocity_shape.png"),
        ("Absolute Error Violin (Vorticity Mean/SD)", "figures/three_way/three_way_abs_error_violin_vorticity_scale.png"),
        ("Absolute Error Violin (Vorticity Skewness/Kurtosis)", "figures/three_way/three_way_abs_error_violin_vorticity_shape.png"),
        ("Temporal-Mean Violin (Velocity Mean/SD)", "figures/three_way/three_way_temporal_mean_abs_error_violin_velocity_scale.png"),
        ("Temporal-Mean Violin (Velocity Skewness/Kurtosis)", "figures/three_way/three_way_temporal_mean_abs_error_violin_velocity_shape.png"),
        ("Temporal-Mean Violin (Vorticity Mean/SD)", "figures/three_way/three_way_temporal_mean_abs_error_violin_vorticity_scale.png"),
        ("Temporal-Mean Violin (Vorticity Skewness/Kurtosis)", "figures/three_way/three_way_temporal_mean_abs_error_violin_vorticity_shape.png"),
    ]
    if raw_bundle is not None:
        figure_entries.extend(
            [
                ("Voxel Histogram In-mask (Full)", "figures/three_way/three_way_voxel_histogram_in_mask_full.png"),
                ("Bland-Altman Speed Full", "figures/three_way/three_way_bland_altman_speed_intraluminal_full.png"),
                ("Bland-Altman U Full", "figures/three_way/three_way_bland_altman_velocity_component_u_allframes_full.png"),
                ("Bland-Altman V Full", "figures/three_way/three_way_bland_altman_velocity_component_v_allframes_full.png"),
                ("Bland-Altman W Full", "figures/three_way/three_way_bland_altman_velocity_component_w_allframes_full.png"),
                ("Bland-Altman MAG Full", "figures/three_way/three_way_bland_altman_velocity_component_mag_allframes_full.png"),
            ]
        )
    metrics_paths = [
        "metrics/three_way/mean_velocity_metrics_three_way.csv",
        "metrics/three_way/peak_velocity_metrics_three_way.csv",
        "metrics/three_way/correlation_metrics_three_way.csv",
        "metrics/three_way/flow_average_metrics_three_way.csv",
        "metrics/three_way/flow_peak_metrics_three_way.csv",
        "metrics/three_way/flow_metrics_per_frame_three_way.csv",
        "metrics/three_way/bland_altman_stats_three_way.csv",
        "metrics/three_way/voxel_distribution_stats_three_way.csv",
        "metrics/three_way/significance_pvalues_three_way.csv",
        "metrics/three_way/table2_like_all_slices_three_way.csv",
        "metrics/three_way/table2_like_temporal_mean_three_way.csv",
        "metrics/three_way/table2_abs_error_summary_three_way.csv",
        "metrics/three_way/table2_temporal_mean_abs_error_summary_three_way.csv",
        "metrics/three_way/merge_qc_three_way.json",
    ]
    if raw_bundle is not None:
        metrics_paths.extend(
            [
                "metrics/three_way/voxel_distribution_stats_three_way_unified_raw.csv",
                "metrics/three_way/bland_altman_stats_three_way_unified_raw.csv",
            ]
        )
    elif raw_bundle_error:
        table2_qc["raw_bundle_warning"] = raw_bundle_error
    _write_html_report(
        out_dir,
        fig_rel_paths=figure_entries,
        metrics_rel_paths=metrics_paths,
        qc=table2_qc,
        labels={"baseline": baseline_label, "denoised": denoised_label, "superres": superres_label},
        method_colors=method_colors,
    )

    print("Three-way report generated.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
