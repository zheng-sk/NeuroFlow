#!/usr/bin/env python3
"""
Deprecated entrypoint.

Use `convert2nifti2.py` directly. This wrapper exists for backward compatibility.
"""

from convert2nifti2 import process_with_fallback


def main() -> None:
    print("[INFO] `convert2nifti.py` is deprecated. Running `convert2nifti2.py` logic instead.")
    process_with_fallback(
        input_root="../data/sorted_patients",
        output_root="../data/nifti_patients",
        dcm2niix_path="dcm2niix",
        canonicalize=True,
    )


if __name__ == "__main__":
    main()
