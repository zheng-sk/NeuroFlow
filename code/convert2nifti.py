#!/usr/bin/env python3
"""Deprecated entrypoint kept for backward compatibility."""

from conversion.dicom_to_nifti import main as run_main


def main() -> None:
    print("[INFO] `convert2nifti.py` is deprecated. Delegating to `conversion/dicom_to_nifti.py`.")
    run_main()


if __name__ == "__main__":
    main()
