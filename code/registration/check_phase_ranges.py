#!/usr/bin/env python3
"""Check phase/velocity NIfTI ranges and print per-file diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("nibabel is required. Install with: pip install nibabel") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):
        return iterable


DEFAULT_PHASE_NAMES = [
    "Vx.nii.gz",
    "Vy.nii.gz",
    "Vz.nii.gz",
    "input_phase_x_raw.nii.gz",
    "input_phase_y_raw.nii.gz",
    "input_phase_z_raw.nii.gz",
    "phaseX_7T_in_3T.nii.gz",
    "phaseY_7T_in_3T.nii.gz",
    "phaseZ_7T_in_3T.nii.gz",
]


@dataclass(frozen=True)
class FileCheck:
    path: str
    shape: str
    ndim: int
    min_value: float
    max_value: float
    mean_value: float
    p01: float
    p50: float
    p99: float
    guessed_mode: str
    status: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate phase/velocity ranges in NIfTI files.")
    parser.add_argument("--input-root", required=True, help="Root folder with NIfTI files")
    parser.add_argument(
        "--phase-names",
        default=",".join(DEFAULT_PHASE_NAMES),
        help="Comma-separated filenames to search for (default includes Vx/Vy/Vz and phase variants)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search under input-root")

    parser.add_argument(
        "--expected-mode",
        choices=["auto", "raw", "velocity"],
        default="auto",
        help="Expected signal type: raw phase (0..4096), velocity (approx -VENC..VENC), or auto-detect",
    )
    parser.add_argument("--raw-min", type=float, default=0.0, help="Expected minimum for raw phase")
    parser.add_argument("--raw-max", type=float, default=4096.0, help="Expected maximum for raw phase")
    parser.add_argument("--raw-margin", type=float, default=256.0, help="Allowed margin for raw phase bounds")

    parser.add_argument("--venc", type=float, default=None, help="Expected VENC in m/s for velocity mode")
    parser.add_argument(
        "--vel-margin-ratio",
        type=float,
        default=0.20,
        help="Allowed relative margin around VENC (default: 0.20 => +/-20%%)",
    )
    parser.add_argument(
        "--vel-hard-limit",
        type=float,
        default=10.0,
        help="Fallback absolute limit in m/s when --venc is not provided",
    )

    parser.add_argument("--csv-report", default=None, help="Optional CSV output path")
    parser.add_argument("--verbose", action="store_true", help="Print all files, including PASS")
    return parser.parse_args()


def discover_files(input_root: str, names: list[str], recursive: bool) -> list[str]:
    wanted = {name.strip() for name in names if name.strip()}
    found: list[str] = []

    if recursive:
        for root, _dirs, files in os.walk(input_root):
            for filename in files:
                if filename in wanted:
                    found.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(input_root):
            path = os.path.join(input_root, filename)
            if os.path.isfile(path) and filename in wanted:
                found.append(path)

    return sorted(found)


def guess_mode(min_value: float, max_value: float) -> str:
    if min_value >= 0.0 and max_value > 1000.0 and max_value <= 8192.0:
        return "raw"
    return "velocity"


def assess_range(
    min_value: float,
    max_value: float,
    expected_mode: str,
    guessed_mode: str,
    raw_min: float,
    raw_max: float,
    raw_margin: float,
    venc: float | None,
    vel_margin_ratio: float,
    vel_hard_limit: float,
) -> tuple[str, str]:
    mode = guessed_mode if expected_mode == "auto" else expected_mode

    if mode == "raw":
        low = raw_min - raw_margin
        high = raw_max + raw_margin
        if min_value < low or max_value > high:
            return "FAIL", f"Outside raw range [{low:.2f}, {high:.2f}]"
        return "PASS", f"Within raw range [{low:.2f}, {high:.2f}]"

    # velocity mode
    if venc is not None:
        limit = abs(venc) * (1.0 + vel_margin_ratio)
        if max(abs(min_value), abs(max_value)) > limit:
            return "FAIL", f"Outside velocity range +/-{limit:.4f} m/s (from VENC={venc})"
        return "PASS", f"Within velocity range +/-{limit:.4f} m/s (from VENC={venc})"

    # no venc: use broad hard safety check
    limit = abs(vel_hard_limit)
    if max(abs(min_value), abs(max_value)) > limit:
        return "WARN", f"Large values for velocity mode: exceeds +/-{limit:.4f} m/s (no VENC provided)"
    return "PASS", f"Within fallback velocity limit +/-{limit:.4f} m/s (no VENC provided)"


def check_file(path: str, args: argparse.Namespace) -> FileCheck:
    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return FileCheck(
            path=path,
            shape=str(tuple(data.shape)),
            ndim=data.ndim,
            min_value=float("nan"),
            max_value=float("nan"),
            mean_value=float("nan"),
            p01=float("nan"),
            p50=float("nan"),
            p99=float("nan"),
            guessed_mode="unknown",
            status="FAIL",
            message="No finite voxels",
        )

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    mean_value = float(np.mean(finite))
    p01, p50, p99 = [float(x) for x in np.percentile(finite, [1, 50, 99])]
    guessed = guess_mode(min_value, max_value)
    status, message = assess_range(
        min_value=min_value,
        max_value=max_value,
        expected_mode=args.expected_mode,
        guessed_mode=guessed,
        raw_min=args.raw_min,
        raw_max=args.raw_max,
        raw_margin=args.raw_margin,
        venc=args.venc,
        vel_margin_ratio=args.vel_margin_ratio,
        vel_hard_limit=args.vel_hard_limit,
    )

    return FileCheck(
        path=path,
        shape=str(tuple(data.shape)),
        ndim=data.ndim,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        p01=p01,
        p50=p50,
        p99=p99,
        guessed_mode=guessed,
        status=status,
        message=message,
    )


def write_csv(path: str, checks: list[FileCheck]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "path",
                "shape",
                "ndim",
                "min",
                "max",
                "mean",
                "p01",
                "p50",
                "p99",
                "guessed_mode",
                "status",
                "message",
            ]
        )
        for item in checks:
            writer.writerow(
                [
                    item.path,
                    item.shape,
                    item.ndim,
                    f"{item.min_value:.6f}",
                    f"{item.max_value:.6f}",
                    f"{item.mean_value:.6f}",
                    f"{item.p01:.6f}",
                    f"{item.p50:.6f}",
                    f"{item.p99:.6f}",
                    item.guessed_mode,
                    item.status,
                    item.message,
                ]
            )


def main() -> None:
    args = parse_args()
    names = [item.strip() for item in args.phase_names.split(",") if item.strip()]

    files = discover_files(args.input_root, names, args.recursive)
    if not files:
        raise SystemExit("No matching phase files found.")

    print(f"Found {len(files)} files to inspect.")
    checks: list[FileCheck] = []
    for path in tqdm(files, desc="Checking ranges", unit="file"):
        checks.append(check_file(path, args))

    pass_count = sum(1 for item in checks if item.status == "PASS")
    warn_count = sum(1 for item in checks if item.status == "WARN")
    fail_count = sum(1 for item in checks if item.status == "FAIL")

    print("\nPer-file results:")
    for item in checks:
        if not args.verbose and item.status == "PASS":
            continue
        print(
            f"[{item.status}] {Path(item.path).name} | mode={item.guessed_mode} "
            f"| min={item.min_value:.5f} max={item.max_value:.5f} "
            f"| p01/p50/p99={item.p01:.5f}/{item.p50:.5f}/{item.p99:.5f}"
        )
        if item.status != "PASS":
            print(f"         {item.message}")

    print("\nSummary:")
    print(f"Total: {len(checks)}")
    print(f"PASS:  {pass_count}")
    print(f"WARN:  {warn_count}")
    print(f"FAIL:  {fail_count}")

    if args.csv_report:
        write_csv(args.csv_report, checks)
        print(f"CSV report: {args.csv_report}")

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
