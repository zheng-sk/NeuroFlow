#!/usr/bin/env python3
"""Compatibility exports for `registration/core.py`."""

from registration.core import apply_transforms_to_file, apply_transforms_to_velocity_triplet, register_4d_nifti

__all__ = [
    "register_4d_nifti",
    "apply_transforms_to_file",
    "apply_transforms_to_velocity_triplet",
]
