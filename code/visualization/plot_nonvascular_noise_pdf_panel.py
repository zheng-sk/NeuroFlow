#!/usr/bin/env python3
"""Plot non-vascular noise PDFs for all patients in a 2x2 panel.

The panel contains:
- Magnitude (input_mag_raw)
- Phase Vx
- Phase Vy
- Phase Vz

Each subplot overlays one PDF curve per patient.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from scipy import stats


CHANNEL_FILES = {
    "Magnitude": "input_mag_raw.nii.gz",
    "Phase Vx": "Vx.nii.gz",
    "Phase Vy": "Vy.nii.gz",
    "Phase Vz": "Vz.nii.gz",
}


@dataclass(frozen=True)
class Config:
    images_root: Path
    masks_root: Path
    use_analysis_mask: bool
    mask_threshold: float
    time_mode: str
    frame_index: int
    bins: int
    disable_clip: bool
    clip_low: float
    clip_high: float
    magnitude_range_upper_percentile: float
    exclude_zero: bool
    zero_eps: float


@dataclass(frozen=True)
class FitResult:
    channel: str
    model: str
    params: tuple[float, ...]
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    n_samples: int


FIT_MODELS = {
    "normal": ("Normal", stats.norm),
    "laplace": ("Laplace", stats.laplace),
    "student_t": ("StudentT", stats.t),
    "cauchy": ("Cauchy", stats.cauchy),
    "logistic": ("Logistic", stats.logistic),
    "gennorm": ("GeneralizedNormal", stats.gennorm),
    "skewnorm": ("SkewNormal", stats.skewnorm),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract non-vascular intensity PDFs patient-by-patient for "
            "magnitude and phase images, and plot a global 2x2 panel."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/paired_dataset/lr_3t"),
        help="Root folder with patient subfolders containing input_mag_raw, Vx, Vy, Vz.",
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        default=Path("data/paired_dataset/cow_segmentation"),
        help="Root folder with patient subfolders containing cow_seg_final and optional analysis_mask.",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=Path("output/noise_pdf/nonvascular_noise_pdf_panel_lr3t.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/noise_pdf/nonvascular_noise_pdf_panel_lr3t.csv"),
        help="Optional CSV with per-bin PDF values. Use empty string to disable.",
    )

    parser.add_argument("--use-analysis-mask", action="store_true", help="Restrict region to analysis_mask before removing vessels.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)

    parser.add_argument("--time-mode", choices=["all", "frame"], default="all")
    parser.add_argument("--frame-index", type=int, default=0)

    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--disable-clip", action="store_true", help="Disable percentile clipping before histogram.")
    parser.add_argument("--clip-low", type=float, default=0.0, help="Lower percentile for clipping (ignored with --disable-clip).")
    parser.add_argument("--clip-high", type=float, default=100.0, help="Upper percentile for clipping (ignored with --disable-clip).")
    parser.add_argument(
        "--magnitude-range-upper-percentile",
        type=float,
        default=99.0,
        help=(
            "Upper percentile used to define the histogram range for Magnitude "
            "(use 100.0 to use absolute max)."
        ),
    )

    parser.add_argument("--exclude-zero", action="store_true", help="Exclude near-zero values before histogramming.")
    parser.add_argument("--zero-eps", type=float, default=1e-8)
    parser.add_argument(
        "--fit-models",
        type=str,
        default="normal,laplace,student_t,cauchy,logistic,gennorm,skewnorm",
        help=(
            "Comma-separated distribution candidates: "
            "normal, laplace, student_t, cauchy, logistic, gennorm, skewnorm."
        ),
    )
    parser.add_argument(
        "--fit-max-samples",
        type=int,
        default=0,
        help="Max samples per channel used for distribution fitting. Use 0 to use all samples.",
    )
    parser.add_argument(
        "--overlay-best-fit",
        action="store_true",
        help="Overlay best-fit PDF (by AIC) in each subplot.",
    )
    parser.add_argument(
        "--overlay-all-fits",
        action="store_true",
        help="Overlay all fitted candidate PDFs in each subplot.",
    )
    parser.add_argument(
        "--overlay-exaggerated-fit",
        action="store_true",
        help="Overlay exaggerated distribution derived from best fit (wider + flatter).",
    )
    parser.add_argument(
        "--exaggerated-profile-mode",
        type=str,
        default="envelope",
        choices=["envelope", "bestfit_hat"],
        help=(
            "How to build exaggerated overlay: "
            "`envelope` (from patient PDF envelope) or `bestfit_hat` (from best-fit deformation)."
        ),
    )
    parser.add_argument(
        "--exaggerated-scale-mult",
        type=float,
        default=2.5,
        help="Scale multiplier applied to best-fit spread for exaggerated overlay.",
    )
    parser.add_argument(
        "--exaggerated-uniform-mix",
        type=float,
        default=0.35,
        help="Mix factor in [0,1] between widened best-fit PDF and uniform PDF (hat-like flattening).",
    )
    parser.add_argument(
        "--exaggerated-envelope-quantile",
        type=float,
        default=1.0,
        help=(
            "Envelope quantile in [0,1] across patient PDFs (1.0=max envelope). "
            "Used with --exaggerated-profile-mode envelope."
        ),
    )
    parser.add_argument(
        "--exaggerated-envelope-gain",
        type=float,
        default=1.0,
        help="Amplitude gain applied to envelope profile (used with envelope mode).",
    )
    parser.add_argument(
        "--exaggerated-smooth-window",
        type=int,
        default=31,
        help="Odd smoothing window (in x samples) for exaggerated overlay.",
    )
    parser.add_argument(
        "--exaggerated-side-expand",
        type=float,
        default=1.25,
        help=(
            "Side expansion factor (>=1). >1 broadens the distribution laterally "
            "by stretching around the center."
        ),
    )
    parser.add_argument(
        "--exaggerated-edge-boost",
        type=float,
        default=0.12,
        help="Extra lift added near left/right borders (tails).",
    )
    parser.add_argument(
        "--exaggerated-edge-power",
        type=float,
        default=2.0,
        help="Shape of edge boost profile (>0). Higher means boost concentrated nearer borders.",
    )
    parser.add_argument(
        "--exaggerated-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize exaggerated overlay to integrate to 1. "
            "Disabled by default to keep envelope-like visual profile."
        ),
    )
    parser.add_argument(
        "--out-fit-csv",
        type=Path,
        default=Path("output/noise_pdf/nonvascular_noise_fit_summary.csv"),
        help="CSV with fit metrics per channel/model. Use empty string to disable.",
    )

    return parser.parse_args()


def resolve_existing(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path("..") / path
    if alt.exists():
        return alt
    return path


def list_patient_ids(images_root: Path, masks_root: Path) -> list[str]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root.resolve()}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks root not found: {masks_root.resolve()}")

    image_ids = {p.name for p in images_root.iterdir() if p.is_dir()}
    mask_ids = {p.name for p in masks_root.iterdir() if p.is_dir()}

    common = sorted(image_ids & mask_ids)
    if not common:
        raise ValueError("No overlapping patient IDs between image and mask roots.")
    return common


def load_nifti_float(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    p = resolve_existing(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path} (resolved: {p.resolve()})")
    img = nib.load(str(p))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    return img, arr


def select_mask_3d(mask_data: np.ndarray, time_index: int = 0) -> np.ndarray:
    if mask_data.ndim == 3:
        return mask_data
    if mask_data.ndim == 4:
        if not (0 <= time_index < mask_data.shape[-1]):
            raise ValueError(f"time_index={time_index} out of range for mask shape={mask_data.shape}")
        return np.asarray(mask_data[..., time_index], dtype=np.float32)
    raise ValueError(f"Invalid mask shape={mask_data.shape}. Expected 3D or 4D.")


def align_mask_to_reference(mask_img: nib.Nifti1Image, mask_3d: np.ndarray, ref_img: nib.Nifti1Image) -> np.ndarray:
    mask_3d_img = nib.Nifti1Image(mask_3d, mask_img.affine, mask_img.header)
    same_shape = tuple(mask_3d_img.shape[:3]) == tuple(ref_img.shape[:3])
    same_affine = np.allclose(mask_3d_img.affine, ref_img.affine, atol=1e-4)
    if not (same_shape and same_affine):
        mask_3d_img = nibproc.resample_from_to(mask_3d_img, (ref_img.shape[:3], ref_img.affine), order=0)
    return np.asarray(mask_3d_img.dataobj, dtype=np.float32)


def to_binary(mask_data: np.ndarray, thr: float) -> np.ndarray:
    return (mask_data >= float(thr)).astype(np.uint8)


def extract_region_values(image_data: np.ndarray, region_mask_3d: np.ndarray, time_mode: str, frame_index: int) -> np.ndarray:
    mask_bool = region_mask_3d.astype(bool)

    if image_data.ndim == 3:
        return image_data[mask_bool]

    if image_data.ndim != 4:
        raise ValueError(f"Expected 3D/4D image, got shape={image_data.shape}")

    if time_mode == "frame":
        if not (0 <= frame_index < image_data.shape[-1]):
            raise ValueError(f"frame_index={frame_index} out of range for shape={image_data.shape}")
        return image_data[..., frame_index][mask_bool]

    return image_data[mask_bool].reshape(-1)


def clean_values(values: np.ndarray, exclude_zero: bool, zero_eps: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if exclude_zero:
        x = x[np.abs(x) > float(zero_eps)]
    return x


def clip_values(values: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [p_lo, p_hi])
    return np.clip(values, lo, hi)


def load_nonvascular_mask_for_patient(patient_id: str, cfg: Config) -> tuple[nib.Nifti1Image, np.ndarray]:
    patient_img_dir = cfg.images_root / patient_id
    patient_mask_dir = cfg.masks_root / patient_id

    mag_img, _ = load_nifti_float(patient_img_dir / CHANNEL_FILES["Magnitude"])

    vmask_img, vmask_raw = load_nifti_float(patient_mask_dir / "cow_seg_final.nii.gz")
    vmask_3d = select_mask_3d(vmask_raw, time_index=0)
    vmask_3d = align_mask_to_reference(vmask_img, vmask_3d, mag_img)
    vascular_mask = to_binary(vmask_3d, cfg.mask_threshold)

    if cfg.use_analysis_mask:
        amask_img, amask_raw = load_nifti_float(patient_mask_dir / "analysis_mask.nii.gz")
        amask_3d = select_mask_3d(amask_raw, time_index=0)
        amask_3d = align_mask_to_reference(amask_img, amask_3d, mag_img)
        analysis_mask = to_binary(amask_3d, cfg.mask_threshold)
    else:
        analysis_mask = np.ones(mag_img.shape[:3], dtype=np.uint8)

    nonvascular_mask = ((analysis_mask > 0) & (vascular_mask == 0)).astype(np.uint8)
    return mag_img, nonvascular_mask


def extract_values_for_channel(patient_id: str, channel_file: str, nonvascular_mask: np.ndarray, cfg: Config) -> np.ndarray:
    patient_img_dir = cfg.images_root / patient_id
    _, arr = load_nifti_float(patient_img_dir / channel_file)

    raw_values = extract_region_values(
        image_data=arr,
        region_mask_3d=nonvascular_mask,
        time_mode=cfg.time_mode,
        frame_index=cfg.frame_index,
    )
    clean = clean_values(raw_values, exclude_zero=cfg.exclude_zero, zero_eps=cfg.zero_eps)
    if cfg.disable_clip:
        return clean
    return clip_values(clean, cfg.clip_low, cfg.clip_high)


def estimate_global_ranges(patient_ids: list[str], cfg: Config) -> dict[str, tuple[float, float]]:
    per_channel_lows: dict[str, list[float]] = {k: [] for k in CHANNEL_FILES}
    per_channel_highs: dict[str, list[float]] = {k: [] for k in CHANNEL_FILES}

    for patient_id in patient_ids:
        _, nonvascular_mask = load_nonvascular_mask_for_patient(patient_id, cfg)
        voxels = int(nonvascular_mask.sum())
        if voxels == 0:
            print(f"[WARN] Empty non-vascular mask for patient {patient_id}. Skipping patient.")
            continue

        for channel_name, channel_file in CHANNEL_FILES.items():
            values = extract_values_for_channel(patient_id, channel_file, nonvascular_mask, cfg)
            if values.size == 0:
                print(f"[WARN] No valid values for {patient_id} {channel_name}. Skipping this curve.")
                continue
            per_channel_lows[channel_name].append(float(np.min(values)))
            if channel_name == "Magnitude" and cfg.magnitude_range_upper_percentile < 100.0:
                hi = float(np.percentile(values, cfg.magnitude_range_upper_percentile))
            else:
                hi = float(np.max(values))
            per_channel_highs[channel_name].append(hi)

    ranges: dict[str, tuple[float, float]] = {}
    for channel_name in CHANNEL_FILES:
        lows = per_channel_lows[channel_name]
        highs = per_channel_highs[channel_name]
        if not lows or not highs:
            raise ValueError(f"No valid data found for channel: {channel_name}")

        lo = min(lows)
        hi = max(highs)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite histogram range for channel: {channel_name}")
        if hi <= lo:
            hi = lo + 1e-6
        ranges[channel_name] = (lo, hi)

    return ranges


def compute_patient_histograms(
    patient_ids: list[str],
    cfg: Config,
    ranges: dict[str, tuple[float, float]],
) -> tuple[
    dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]],
    dict[str, list[np.ndarray]],
]:
    out: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]] = {k: [] for k in CHANNEL_FILES}
    channel_values: dict[str, list[np.ndarray]] = {k: [] for k in CHANNEL_FILES}

    for patient_id in patient_ids:
        _, nonvascular_mask = load_nonvascular_mask_for_patient(patient_id, cfg)
        if int(nonvascular_mask.sum()) == 0:
            continue

        for channel_name, channel_file in CHANNEL_FILES.items():
            values = extract_values_for_channel(patient_id, channel_file, nonvascular_mask, cfg)
            if values.size == 0:
                continue

            lo, hi = ranges[channel_name]
            counts, edges = np.histogram(values, bins=cfg.bins, range=(lo, hi), density=False)
            widths = np.diff(edges)
            total = max(int(counts.sum()), 1)
            probs = counts / total
            pdf = probs / widths
            centers = edges[:-1] + widths / 2.0
            out[channel_name].append((patient_id, centers, pdf, counts.astype(np.int64)))
            channel_values[channel_name].append(values)

    return out, channel_values


def parse_fit_models(raw: str) -> list[str]:
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("fit_models cannot be empty.")
    invalid = [x for x in names if x not in FIT_MODELS]
    if invalid:
        raise ValueError(f"Unknown fit model(s): {invalid}. Allowed: {list(FIT_MODELS)}")
    return names


def _safe_fit_values(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    if max_samples <= 0 or x.size <= max_samples:
        return x
    idx = rng.choice(x.size, size=max_samples, replace=False)
    return x[idx]


def fit_distribution_models(
    channel_values: dict[str, list[np.ndarray]],
    model_names: list[str],
    max_samples: int,
    rng_seed: int = 42,
) -> dict[str, list[FitResult]]:
    rng = np.random.default_rng(rng_seed)
    out: dict[str, list[FitResult]] = {k: [] for k in CHANNEL_FILES}

    for channel_name in CHANNEL_FILES:
        if not channel_values[channel_name]:
            continue
        pooled = np.concatenate(channel_values[channel_name], axis=0)
        x = _safe_fit_values(pooled, max_samples=max_samples, rng=rng)
        n = int(x.size)
        if n < 20:
            print(f"[WARN] Too few samples to fit {channel_name} (n={n}).")
            continue
        if max_samples > 0 and pooled.size > n:
            print(f"[INFO] Fit sampling for {channel_name}: pooled={pooled.size} -> used={n}")
        else:
            print(f"[INFO] Fit uses all samples for {channel_name}: n={n}")

        for model_name in model_names:
            pretty, dist = FIT_MODELS[model_name]
            try:
                params = tuple(float(v) for v in dist.fit(x))
                logpdf = np.asarray(dist.logpdf(x, *params), dtype=np.float64)
                if not np.all(np.isfinite(logpdf)):
                    raise FloatingPointError("Non-finite logpdf values.")

                nll = -float(np.sum(logpdf))
                k = len(params)
                aic = 2.0 * k + 2.0 * nll
                bic = k * np.log(max(n, 1)) + 2.0 * nll
                ks = stats.kstest(x, dist.cdf, args=params)

                out[channel_name].append(
                    FitResult(
                        channel=channel_name,
                        model=pretty,
                        params=params,
                        aic=float(aic),
                        bic=float(bic),
                        ks_stat=float(ks.statistic),
                        ks_pvalue=float(ks.pvalue),
                        n_samples=n,
                    )
                )
            except Exception as exc:  # keep batch robust across channels/models
                print(f"[WARN] Fit failed for {channel_name} model={pretty}: {exc}")

        out[channel_name].sort(key=lambda r: r.aic)

    return out


def save_fit_summary_csv(out_csv: Path, fit_results: dict[str, list[FitResult]]) -> None:
    rows = ["channel,model,n_samples,aic,bic,ks_stat,ks_pvalue,params"]
    for channel_name in CHANNEL_FILES:
        for r in fit_results[channel_name]:
            param_text = ";".join(f"{p:.9g}" for p in r.params)
            rows.append(
                f"{channel_name},{r.model},{r.n_samples},{r.aic:.9g},{r.bic:.9g},"
                f"{r.ks_stat:.9g},{r.ks_pvalue:.9g},{param_text}"
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Fit summary CSV written to: {out_csv.resolve()}")


def _dist_from_pretty_name(pretty_name: str):
    for _, (pretty, dist) in FIT_MODELS.items():
        if pretty == pretty_name:
            return dist
    return None


def _build_exaggerated_pdf(
    best_fit: FitResult,
    x_line: np.ndarray,
    lo: float,
    hi: float,
    scale_mult: float,
    uniform_mix: float,
    smooth_window: int,
    side_expand: float,
    edge_boost: float,
    edge_power: float,
    normalize: bool,
) -> np.ndarray:
    dist = _dist_from_pretty_name(best_fit.model)
    if dist is None:
        return np.zeros_like(x_line, dtype=np.float64)

    params = list(best_fit.params)
    if len(params) < 2:
        return np.zeros_like(x_line, dtype=np.float64)

    # scipy continuous distributions generally use (...shape, loc, scale)
    # so scaling the last parameter widens the spread.
    params[-1] = float(params[-1]) * max(float(scale_mult), 1e-6)

    try:
        y_wide = np.asarray(dist.pdf(x_line, *tuple(params)), dtype=np.float64)
    except Exception:
        return np.zeros_like(x_line, dtype=np.float64)
    y_wide[~np.isfinite(y_wide)] = 0.0
    y_wide = np.clip(y_wide, 0.0, None)

    span = max(float(hi) - float(lo), 1e-12)
    y_uniform = np.full_like(x_line, fill_value=1.0 / span, dtype=np.float64)
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    y_mix = (1.0 - mix) * y_wide + mix * y_uniform
    y_mix = _expand_sides_and_edges(
        y_mix,
        x_line=x_line,
        side_expand=side_expand,
        edge_boost=edge_boost,
        edge_power=edge_power,
    )
    y_mix = _smooth_1d(y_mix, smooth_window)

    if normalize:
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(y_mix, x_line))
        else:
            area = float(np.trapz(y_mix, x_line))
        if area > 0.0 and np.isfinite(area):
            y_mix = y_mix / area
    return y_mix


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return y
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(y, kernel, mode="same")


def _expand_sides_and_edges(
    y: np.ndarray,
    x_line: np.ndarray,
    side_expand: float,
    edge_boost: float,
    edge_power: float,
) -> np.ndarray:
    y0 = np.asarray(y, dtype=np.float64)
    span = max(float(x_line[-1]) - float(x_line[0]), 1e-12)
    mid = 0.5 * (float(x_line[0]) + float(x_line[-1]))

    s = max(float(side_expand), 1e-6)
    if abs(s - 1.0) > 1e-9:
        x_query = mid + (x_line - mid) / s
        y1 = np.interp(x_query, x_line, y0, left=0.0, right=0.0)
    else:
        y1 = y0.copy()

    b = max(float(edge_boost), 0.0)
    if b > 0.0:
        p = max(float(edge_power), 1e-6)
        edge = np.abs((x_line - mid) / (0.5 * span))
        edge = np.clip(edge, 0.0, 1.0) ** p
        amp = max(float(np.max(y1)), 1e-12)
        y1 = y1 + b * amp * edge

    y1 = np.clip(y1, 0.0, None)
    return y1


def _build_exaggerated_envelope(
    entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    x_line: np.ndarray,
    quantile: float,
    gain: float,
    smooth_window: int,
    side_expand: float,
    edge_boost: float,
    edge_power: float,
    normalize: bool,
) -> np.ndarray:
    curves = []
    for _pid, centers, pdf, _counts in entries:
        curves.append(np.interp(x_line, centers, pdf, left=0.0, right=0.0))
    if not curves:
        return np.zeros_like(x_line, dtype=np.float64)
    y_stack = np.stack(curves, axis=0)
    q = float(np.clip(quantile, 0.0, 1.0))
    y_env = np.quantile(y_stack, q=q, axis=0)
    y_env = np.clip(y_env, 0.0, None) * max(float(gain), 0.0)
    y_env = _expand_sides_and_edges(
        y_env,
        x_line=x_line,
        side_expand=side_expand,
        edge_boost=edge_boost,
        edge_power=edge_power,
    )
    y_env = _smooth_1d(y_env, smooth_window)

    if normalize:
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(y_env, x_line))
        else:
            area = float(np.trapz(y_env, x_line))
        if area > 0.0 and np.isfinite(area):
            y_env = y_env / area
    return y_env


def save_csv(
    out_csv: Path,
    hist_data: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]],
) -> None:
    rows = ["channel,patient_id,bin_center,pdf,count"]

    for channel_name, entries in hist_data.items():
        for patient_id, centers, pdf, counts in entries:
            for c, p, n in zip(centers, pdf, counts):
                rows.append(f"{channel_name},{patient_id},{float(c):.9g},{float(p):.9g},{int(n)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"CSV written to: {out_csv.resolve()}")


def plot_panel(
    hist_data: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]],
    fit_results: dict[str, list[FitResult]],
    out_fig: Path,
    cfg: Config,
    overlay_best_fit: bool,
    overlay_all_fits: bool,
    overlay_exaggerated_fit: bool,
    exaggerated_profile_mode: str,
    exaggerated_scale_mult: float,
    exaggerated_uniform_mix: float,
    exaggerated_envelope_quantile: float,
    exaggerated_envelope_gain: float,
    exaggerated_smooth_window: int,
    exaggerated_side_expand: float,
    exaggerated_edge_boost: float,
    exaggerated_edge_power: float,
    exaggerated_normalize: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    channel_names = list(CHANNEL_FILES.keys())
    for idx, channel_name in enumerate(channel_names):
        ax = axes[idx]
        entries = hist_data[channel_name]
        best_label = None
        if fit_results.get(channel_name):
            best_fit = fit_results[channel_name][0]
            best_label = f"{best_fit.model} (AIC={best_fit.aic:.3g})"

        if not entries:
            title = f"{channel_name} (no data)"
            if best_label is not None:
                title += f"\nBest fit: {best_label}"
            ax.set_title(title)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("PDF")
            ax.grid(alpha=0.25)
            continue

        for patient_id, centers, pdf, _ in entries:
            ax.plot(centers, pdf, linewidth=1.5, alpha=0.9, label=patient_id)

        if overlay_all_fits and fit_results[channel_name]:
            x_line = np.linspace(float(entries[0][1][0]), float(entries[0][1][-1]), 800)
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(fit_results[channel_name]))))
            linestyles = ["--", "-.", ":", "-"]
            for idx_fit, fit in enumerate(fit_results[channel_name]):
                dist = _dist_from_pretty_name(fit.model)
                if dist is None:
                    continue
                y_line = np.asarray(dist.pdf(x_line, *fit.params), dtype=np.float64)
                y_line[~np.isfinite(y_line)] = 0.0
                ax.plot(
                    x_line,
                    y_line,
                    linestyle=linestyles[idx_fit % len(linestyles)],
                    color=colors[idx_fit % len(colors)],
                    linewidth=2.0,
                    alpha=0.95,
                    label="_nolegend_",
                )
        elif (overlay_best_fit or overlay_exaggerated_fit) and fit_results[channel_name]:
            best = fit_results[channel_name][0]
            dist = _dist_from_pretty_name(best.model)
            if dist is not None:
                x_line = np.linspace(float(entries[0][1][0]), float(entries[0][1][-1]), 800)
                y_line = np.asarray(dist.pdf(x_line, *best.params), dtype=np.float64)
                y_line[~np.isfinite(y_line)] = 0.0
                ax.plot(x_line, y_line, "k--", linewidth=2.0, alpha=0.95, label="_nolegend_")

        if overlay_exaggerated_fit and fit_results[channel_name]:
            lo = float(entries[0][1][0])
            hi = float(entries[0][1][-1])
            x_line = np.linspace(lo, hi, 800)
            if exaggerated_profile_mode == "envelope":
                y_ex = _build_exaggerated_envelope(
                    entries=entries,
                    x_line=x_line,
                    quantile=exaggerated_envelope_quantile,
                    gain=exaggerated_envelope_gain,
                    smooth_window=exaggerated_smooth_window,
                    side_expand=exaggerated_side_expand,
                    edge_boost=exaggerated_edge_boost,
                    edge_power=exaggerated_edge_power,
                    normalize=exaggerated_normalize,
                )
            else:
                best = fit_results[channel_name][0]
                y_ex = _build_exaggerated_pdf(
                    best_fit=best,
                    x_line=x_line,
                    lo=lo,
                    hi=hi,
                    scale_mult=exaggerated_scale_mult,
                    uniform_mix=exaggerated_uniform_mix,
                    smooth_window=exaggerated_smooth_window,
                    side_expand=exaggerated_side_expand,
                    edge_boost=exaggerated_edge_boost,
                    edge_power=exaggerated_edge_power,
                    normalize=exaggerated_normalize,
                )
            ax.plot(
                x_line,
                y_ex,
                color="#d62728",
                linestyle="-.",
                linewidth=2.2,
                alpha=0.98,
                label="_nolegend_",
            )

        title = f"{channel_name}: non-vascular noise PDF"
        if best_label is not None:
            title += f"\nBest fit: {best_label}"
        ax.set_title(title)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("PDF")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    pairs = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    unique = {l: h for h, l in pairs}
    fig.legend(unique.values(), unique.keys(), loc="center right", title="Patient", frameon=False)

    subtitle = (
        f"time_mode={cfg.time_mode}, frame_index={cfg.frame_index}, bins={cfg.bins}, "
        f"clip={'disabled' if cfg.disable_clip else f'[{cfg.clip_low},{cfg.clip_high}]'}, "
        f"mag_range_hi_p={cfg.magnitude_range_upper_percentile}, use_analysis_mask={cfg.use_analysis_mask}"
    )
    if overlay_exaggerated_fit:
        subtitle += (
            f", exaggerated(mode={exaggerated_profile_mode}, side_expand={exaggerated_side_expand:.3g}, "
            f"edge_boost={exaggerated_edge_boost:.3g}, normalize={exaggerated_normalize})"
        )
    fig.suptitle("Global non-vascular noise distributions (all patients)\n" + subtitle, fontsize=12)
    plt.tight_layout(rect=[0.0, 0.0, 0.86, 0.95])

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"Figure written to: {out_fig.resolve()}")


def main() -> None:
    args = parse_args()

    out_csv = None
    if isinstance(args.out_csv, Path) and str(args.out_csv).strip() != "":
        out_csv = args.out_csv
    out_fit_csv = None
    if isinstance(args.out_fit_csv, Path) and str(args.out_fit_csv).strip() != "":
        out_fit_csv = args.out_fit_csv

    cfg = Config(
        images_root=resolve_existing(args.images_root),
        masks_root=resolve_existing(args.masks_root),
        use_analysis_mask=bool(args.use_analysis_mask),
        mask_threshold=float(args.mask_threshold),
        time_mode=str(args.time_mode),
        frame_index=int(args.frame_index),
        bins=int(args.bins),
        disable_clip=bool(args.disable_clip),
        clip_low=float(args.clip_low),
        clip_high=float(args.clip_high),
        magnitude_range_upper_percentile=float(args.magnitude_range_upper_percentile),
        exclude_zero=bool(args.exclude_zero),
        zero_eps=float(args.zero_eps),
    )

    if (not cfg.disable_clip) and cfg.clip_high <= cfg.clip_low:
        raise ValueError("clip_high must be greater than clip_low.")
    if not (0.0 < cfg.magnitude_range_upper_percentile <= 100.0):
        raise ValueError("magnitude_range_upper_percentile must be in (0, 100].")
    if args.fit_max_samples < 0:
        raise ValueError("fit_max_samples must be >= 0 (0 means use all samples)")
    if 0 < args.fit_max_samples < 100:
        raise ValueError("fit_max_samples must be 0 or >= 100")
    if args.exaggerated_scale_mult <= 0:
        raise ValueError("exaggerated_scale_mult must be > 0.")
    if not (0.0 <= args.exaggerated_uniform_mix <= 1.0):
        raise ValueError("exaggerated_uniform_mix must be in [0, 1].")
    if not (0.0 <= args.exaggerated_envelope_quantile <= 1.0):
        raise ValueError("exaggerated_envelope_quantile must be in [0, 1].")
    if args.exaggerated_smooth_window < 1:
        raise ValueError("exaggerated_smooth_window must be >= 1.")
    if args.exaggerated_side_expand < 1.0:
        raise ValueError("exaggerated_side_expand must be >= 1.0.")
    if args.exaggerated_edge_boost < 0.0:
        raise ValueError("exaggerated_edge_boost must be >= 0.")
    if args.exaggerated_edge_power <= 0.0:
        raise ValueError("exaggerated_edge_power must be > 0.")
    if args.exaggerated_profile_mode not in {"envelope", "bestfit_hat"}:
        raise ValueError("exaggerated_profile_mode must be one of: envelope, bestfit_hat.")

    patient_ids = list_patient_ids(cfg.images_root, cfg.masks_root)
    print(f"Patients found: {len(patient_ids)} -> {patient_ids}")
    if args.overlay_all_fits and args.overlay_best_fit:
        print("[INFO] overlay_all_fits enabled: overlay_best_fit will be ignored.")

    ranges = estimate_global_ranges(patient_ids, cfg)
    for name, (lo, hi) in ranges.items():
        print(f"Range {name}: [{lo:.6g}, {hi:.6g}]")

    hist_data, channel_values = compute_patient_histograms(patient_ids, cfg, ranges)
    fit_model_names = parse_fit_models(args.fit_models)
    fit_results = fit_distribution_models(
        channel_values=channel_values,
        model_names=fit_model_names,
        max_samples=int(args.fit_max_samples),
        rng_seed=42,
    )
    for channel_name in CHANNEL_FILES:
        if fit_results[channel_name]:
            best = fit_results[channel_name][0]
            print(
                f"Best fit {channel_name}: {best.model} "
                f"(AIC={best.aic:.3g}, BIC={best.bic:.3g}, KS={best.ks_stat:.4f}, p={best.ks_pvalue:.3g})"
            )

    plot_panel(
        hist_data=hist_data,
        fit_results=fit_results,
        out_fig=resolve_existing(args.out_fig),
        cfg=cfg,
        overlay_best_fit=bool(args.overlay_best_fit),
        overlay_all_fits=bool(args.overlay_all_fits),
        overlay_exaggerated_fit=bool(args.overlay_exaggerated_fit),
        exaggerated_profile_mode=str(args.exaggerated_profile_mode),
        exaggerated_scale_mult=float(args.exaggerated_scale_mult),
        exaggerated_uniform_mix=float(args.exaggerated_uniform_mix),
        exaggerated_envelope_quantile=float(args.exaggerated_envelope_quantile),
        exaggerated_envelope_gain=float(args.exaggerated_envelope_gain),
        exaggerated_smooth_window=int(args.exaggerated_smooth_window),
        exaggerated_side_expand=float(args.exaggerated_side_expand),
        exaggerated_edge_boost=float(args.exaggerated_edge_boost),
        exaggerated_edge_power=float(args.exaggerated_edge_power),
        exaggerated_normalize=bool(args.exaggerated_normalize),
    )

    if out_csv is not None:
        save_csv(resolve_existing(out_csv), hist_data)
    if out_fit_csv is not None:
        save_fit_summary_csv(resolve_existing(out_fit_csv), fit_results)


if __name__ == "__main__":
    main()
