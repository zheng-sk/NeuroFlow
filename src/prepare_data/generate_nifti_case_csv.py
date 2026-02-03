import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


NIFTI_COLUMNS = [
    "lr_u",
    "lr_v",
    "lr_w",
    "lr_mag_u",
    "lr_mag_v",
    "lr_mag_w",
    "hr_u",
    "hr_v",
    "hr_w",
    "mask",
    "venc",
]


def strip_known_extensions(filename: str) -> str:
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def read_unique_source_target_pairs(csv_path: str) -> List[Tuple[str, str]]:
    pairs = []
    seen = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain 'source' and 'target' columns "
                "(legacy patch CSV format)."
            )
        for row in reader:
            src = row["source"].strip()
            tgt = row["target"].strip()
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _resolve_build_path(root: str, rel_or_name: str) -> str:
    if os.path.isabs(rel_or_name):
        return rel_or_name
    return os.path.normpath(os.path.join(root, rel_or_name))


def _format_pattern(pattern: str, context: Dict[str, str]) -> str:
    try:
        return pattern.format(**context)
    except KeyError as exc:
        raise ValueError(f"Pattern '{pattern}' uses unknown placeholder: {exc}") from exc


def build_case_row(
    source: str,
    target: str,
    lr_root: str,
    hr_root: str,
    patterns: Dict[str, str],
    venc: float,
) -> Dict[str, str]:
    lr_stem = strip_known_extensions(source)
    hr_stem = strip_known_extensions(target)
    context = {
        "source": source,
        "target": target,
        "source_name": Path(source).name,
        "target_name": Path(target).name,
        "lr_stem": lr_stem,
        "hr_stem": hr_stem,
    }

    row = {
        "lr_u": _resolve_build_path(lr_root, _format_pattern(patterns["lr_u"], context)),
        "lr_v": _resolve_build_path(lr_root, _format_pattern(patterns["lr_v"], context)),
        "lr_w": _resolve_build_path(lr_root, _format_pattern(patterns["lr_w"], context)),
        "lr_mag_u": _resolve_build_path(lr_root, _format_pattern(patterns["lr_mag_u"], context)),
        "lr_mag_v": _resolve_build_path(lr_root, _format_pattern(patterns["lr_mag_v"], context)),
        "lr_mag_w": _resolve_build_path(lr_root, _format_pattern(patterns["lr_mag_w"], context)),
        "hr_u": _resolve_build_path(hr_root, _format_pattern(patterns["hr_u"], context)),
        "hr_v": _resolve_build_path(hr_root, _format_pattern(patterns["hr_v"], context)),
        "hr_w": _resolve_build_path(hr_root, _format_pattern(patterns["hr_w"], context)),
        "mask": _resolve_build_path(hr_root, _format_pattern(patterns["mask"], context)),
        "venc": f"{float(venc):.6f}",
    }
    return row


def maybe_relativize_paths(row: Dict[str, str], output_csv: str, absolute_paths: bool) -> Dict[str, str]:
    if absolute_paths:
        return row
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    converted = dict(row)
    for key in NIFTI_COLUMNS:
        if key == "venc":
            continue
        converted[key] = os.path.relpath(row[key], out_dir)
    return converted


def validate_row_paths(row: Dict[str, str], strict_exists: bool, allow_missing_mask: bool):
    missing = []
    for key in NIFTI_COLUMNS:
        if key == "venc":
            continue
        if key == "mask" and allow_missing_mask:
            if not row[key] or not os.path.exists(row[key]):
                row[key] = ""
            continue
        if not os.path.exists(row[key]):
            missing.append(f"{key}: {row[key]}")
    if strict_exists and missing:
        raise FileNotFoundError("Missing expected NIfTI files:\n  " + "\n  ".join(missing))


def write_nifti_case_csv(output_csv: str, rows: Iterable[Dict[str, str]]):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NIFTI_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in NIFTI_COLUMNS})


def convert_legacy_split(
    legacy_csv: str,
    output_csv: str,
    lr_root: str,
    hr_root: str,
    patterns: Dict[str, str],
    venc: float,
    strict_exists: bool,
    allow_missing_mask: bool,
    absolute_paths: bool,
):
    pairs = read_unique_source_target_pairs(legacy_csv)
    rows = []
    for source, target in pairs:
        row = build_case_row(
            source=source,
            target=target,
            lr_root=lr_root,
            hr_root=hr_root,
            patterns=patterns,
            venc=venc,
        )
        validate_row_paths(row, strict_exists=strict_exists, allow_missing_mask=allow_missing_mask)
        rows.append(maybe_relativize_paths(row, output_csv=output_csv, absolute_paths=absolute_paths))

    write_nifti_case_csv(output_csv, rows)
    print(f"{legacy_csv}: {len(rows)} unique cases -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate NIfTI case CSV(s) for trainer_nifti.py from legacy patch-index CSVs "
            "(e.g., data/train.csv and data/validate.csv)."
        )
    )
    parser.add_argument("--legacy-train-csv", type=str, required=True, help="Legacy train patch CSV.")
    parser.add_argument("--legacy-val-csv", type=str, required=True, help="Legacy validation patch CSV.")
    parser.add_argument("--output-train-csv", type=str, required=True, help="Output NIfTI train case CSV.")
    parser.add_argument("--output-val-csv", type=str, required=True, help="Output NIfTI val case CSV.")
    parser.add_argument("--lr-root", type=str, required=True, help="Root directory for LR NIfTI files.")
    parser.add_argument("--hr-root", type=str, required=True, help="Root directory for HR NIfTI files.")
    parser.add_argument("--venc", type=float, default=0.0, help="Default venc written to output rows (0 means auto at load time).")
    parser.add_argument("--strict-exists", action="store_true", help="Fail if any expected path does not exist.")
    parser.add_argument("--allow-missing-mask", action="store_true", help="Allow missing mask files and write empty mask field.")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute paths instead of CSV-relative paths.")

    # Filename patterns, placeholders: {lr_stem}, {hr_stem}, {source}, {target}, {source_name}, {target_name}
    parser.add_argument("--lr-u-pattern", type=str, default="{lr_stem}_u.nii.gz")
    parser.add_argument("--lr-v-pattern", type=str, default="{lr_stem}_v.nii.gz")
    parser.add_argument("--lr-w-pattern", type=str, default="{lr_stem}_w.nii.gz")
    parser.add_argument("--lr-mag-u-pattern", type=str, default="{lr_stem}_mag_u.nii.gz")
    parser.add_argument("--lr-mag-v-pattern", type=str, default="{lr_stem}_mag_v.nii.gz")
    parser.add_argument("--lr-mag-w-pattern", type=str, default="{lr_stem}_mag_w.nii.gz")
    parser.add_argument("--hr-u-pattern", type=str, default="{hr_stem}_u.nii.gz")
    parser.add_argument("--hr-v-pattern", type=str, default="{hr_stem}_v.nii.gz")
    parser.add_argument("--hr-w-pattern", type=str, default="{hr_stem}_w.nii.gz")
    parser.add_argument("--mask-pattern", type=str, default="{hr_stem}_mask.nii.gz")
    args = parser.parse_args()

    patterns = {
        "lr_u": args.lr_u_pattern,
        "lr_v": args.lr_v_pattern,
        "lr_w": args.lr_w_pattern,
        "lr_mag_u": args.lr_mag_u_pattern,
        "lr_mag_v": args.lr_mag_v_pattern,
        "lr_mag_w": args.lr_mag_w_pattern,
        "hr_u": args.hr_u_pattern,
        "hr_v": args.hr_v_pattern,
        "hr_w": args.hr_w_pattern,
        "mask": args.mask_pattern,
    }

    convert_legacy_split(
        legacy_csv=args.legacy_train_csv,
        output_csv=args.output_train_csv,
        lr_root=os.path.abspath(args.lr_root),
        hr_root=os.path.abspath(args.hr_root),
        patterns=patterns,
        venc=args.venc,
        strict_exists=args.strict_exists,
        allow_missing_mask=args.allow_missing_mask,
        absolute_paths=args.absolute_paths,
    )
    convert_legacy_split(
        legacy_csv=args.legacy_val_csv,
        output_csv=args.output_val_csv,
        lr_root=os.path.abspath(args.lr_root),
        hr_root=os.path.abspath(args.hr_root),
        patterns=patterns,
        venc=args.venc,
        strict_exists=args.strict_exists,
        allow_missing_mask=args.allow_missing_mask,
        absolute_paths=args.absolute_paths,
    )
    print("Done.")


if __name__ == "__main__":
    main()
