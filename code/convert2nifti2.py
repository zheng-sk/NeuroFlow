#!/usr/bin/env python3
"""Compatibility wrapper for `conversion/dicom_to_nifti.py`."""

from conversion.dicom_to_nifti import main, process_with_fallback

__all__ = ["main", "process_with_fallback"]


if __name__ == "__main__":
    main()
