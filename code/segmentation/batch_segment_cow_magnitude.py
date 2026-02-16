#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_SCRIPT = REPO_ROOT / "code" / "segmentation" / "segment_cow_crops.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "topcow-claim-models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cow_segmentation_mag_only"


def _iter_glob(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    return root.rglob(pattern) if recursive else root.glob(pattern)


def find_magnitude_files(input_root: Path, recursive: bool, patterns: List[str]) -> List[Path]:
    seen = set()
    matches: List[Path] = []
    for pattern in patterns:
        for path in _iter_glob(input_root, pattern, recursive):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(resolved)
    return sorted(matches)


def build_patient_id(mag_file: Path, input_root: Path) -> str:
    if mag_file.parent != input_root:
        return mag_file.parent.name
    stem = mag_file.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    return stem


def run_case(args: argparse.Namespace, mag_file: Path, output_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(args.segment_script),
        "--input",
        str(mag_file),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(output_dir),
        "--projection-method",
        args.projection_method,
        "--projection-percentile",
        str(args.projection_percentile),
        "--projection-topk",
        str(args.projection_topk),
        "--ensemble-mode",
        "ai",
        "--no-classic-cow",
        "--no-postprocess",
    ]

    print(f"[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segmenta CoW paciente por paciente usando SOLO imagenes de magnitud, "
            "forzando AI-only y sin postproceso para evitar expansion de mascara."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Carpeta raiz con casos/pacientes (contiene NIfTI).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta de salida (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Carpeta del modelo nnU-Net (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--segment-script",
        type=Path,
        default=DEFAULT_SEGMENT_SCRIPT,
        help=f"Ruta a segment_cow_crops.py (default: {DEFAULT_SEGMENT_SCRIPT}).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar archivos de magnitud en subcarpetas.",
    )
    parser.add_argument(
        "--mag-pattern",
        action="append",
        default=None,
        help=(
            "Glob para seleccionar solo magnitud. Repetir el flag para varios patrones. "
            "Defaults: mag*.nii.gz, input_mag*.nii.gz"
        ),
    )
    parser.add_argument(
        "--projection-method",
        choices=["max", "percentile", "topk_mean"],
        default="max",
        help="Metodo de proyeccion temporal 4D->3D.",
    )
    parser.add_argument(
        "--projection-percentile",
        type=float,
        default=95.0,
        help="Percentil para --projection-method percentile.",
    )
    parser.add_argument(
        "--projection-topk",
        type=int,
        default=3,
        help="Top-k para --projection-method topk_mean.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Detener ejecucion si falla algun paciente.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mag_pattern:
        args.mag_pattern = ["mag*.nii.gz", "input_mag*.nii.gz"]

    if not args.input_root.exists():
        raise FileNotFoundError(f"No existe input-root: {args.input_root}")
    if not args.segment_script.exists():
        raise FileNotFoundError(f"No existe segment script: {args.segment_script}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"No existe model-dir: {args.model_dir}")

    mag_files = find_magnitude_files(args.input_root, args.recursive, args.mag_pattern)
    if not mag_files:
        raise RuntimeError(
            "No se encontraron archivos de magnitud. "
            f"input_root={args.input_root}, patterns={args.mag_pattern}, recursive={args.recursive}"
        )

    print(f"Se encontraron {len(mag_files)} volumenes de magnitud")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, mag_file in enumerate(mag_files, start=1):
        patient_id = build_patient_id(mag_file, args.input_root)
        case_output_dir = args.output_dir / patient_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(mag_files)}] Paciente: {patient_id}")
        print(f"Magnitud: {mag_file}")
        print(f"Salida:   {case_output_dir}")

        code = run_case(args, mag_file, case_output_dir)
        if code != 0:
            failed.append((patient_id, str(mag_file), code))
            print(f"[ERROR] paciente={patient_id} returncode={code}")
            if args.stop_on_error:
                break

    if failed:
        print("\nFallaron estos pacientes:")
        for patient_id, mag_file, code in failed:
            print(f"- {patient_id}: {mag_file} (code={code})")
        return 1

    print("\nCompletado: todas las mascaras fueron segmentadas (AI-only, sin postproceso).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
