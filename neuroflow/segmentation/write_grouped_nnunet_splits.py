#!/usr/bin/env python3
"""Write grouped nnU-Net splits so related samples from the same patient stay in the same fold."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create nnU-Net splits_final.json with grouping by patient/case id."
    )
    p.add_argument("--images-tr-dir", required=True, help="Dataset imagesTr directory.")
    p.add_argument("--preprocessed-dataset-dir", required=True, help="Dataset preprocessed directory where splits_final.json will be written.")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--sample-suffixes",
        default="3t,x05",
        help="Comma-separated suffixes to strip from case ids when forming patient groups.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def strip_channel_suffix(filename: str) -> str:
    if not filename.endswith(".nii.gz"):
        return filename
    stem = filename[:-7]
    if len(stem) >= 5 and stem[-5] == "_" and stem[-4:].isdigit():
        return stem[:-5]
    return stem


def derive_group(case_token: str, suffixes: list[str]) -> str:
    for suffix in suffixes:
        suffix_token = f"_{suffix}"
        if case_token.endswith(suffix_token):
            return case_token[: -len(suffix_token)]
    return case_token


def main() -> int:
    args = parse_args()

    images_tr_dir = Path(args.images_tr_dir).resolve()
    preprocessed_dir = Path(args.preprocessed_dataset_dir).resolve()
    splits_path = preprocessed_dir / "splits_final.json"

    if not images_tr_dir.is_dir():
        raise FileNotFoundError(f"imagesTr directory not found: {images_tr_dir}")
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed dataset directory not found: {preprocessed_dir}")
    if splits_path.exists() and not args.overwrite:
        raise FileExistsError(f"{splits_path} already exists. Use --overwrite to replace it.")

    suffixes = [s.strip() for s in args.sample_suffixes.split(",") if s.strip()]
    case_tokens = sorted({strip_channel_suffix(p.name) for p in images_tr_dir.glob("*.nii.gz")})
    if not case_tokens:
        raise RuntimeError(f"No training image files found in {images_tr_dir}")

    grouped: dict[str, list[str]] = {}
    for token in case_tokens:
        grouped.setdefault(derive_group(token, suffixes), []).append(token)

    group_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    folds = [[] for _ in range(args.num_folds)]
    for idx, group_id in enumerate(group_ids):
        folds[idx % args.num_folds].append(group_id)

    split_payload = []
    for fold_idx in range(args.num_folds):
        val_groups = set(folds[fold_idx])
        train_cases = []
        val_cases = []
        for group_id in group_ids:
            target = val_cases if group_id in val_groups else train_cases
            target.extend(sorted(grouped[group_id]))
        split_payload.append({"train": train_cases, "val": val_cases})
        if args.verbose:
            print(
                f"[fold {fold_idx}] train_groups={len(group_ids) - len(val_groups)} "
                f"val_groups={len(val_groups)} train_cases={len(train_cases)} val_cases={len(val_cases)}"
            )

    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
        f.write("\n")

    print("Grouped nnU-Net splits written")
    print(f"imagesTr:          {images_tr_dir}")
    print(f"preprocessed dir:  {preprocessed_dir}")
    print(f"splits path:       {splits_path}")
    print(f"groups:            {len(group_ids)}")
    print(f"cases:             {len(case_tokens)}")
    print(f"folds:             {args.num_folds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
