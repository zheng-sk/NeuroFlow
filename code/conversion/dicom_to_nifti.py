#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert2nifti2.py

DICOM -> NIfTI with:
- dcm2niix first (when it succeeds without data loss)
- fallback to direct 4D DICOM reading, building affine from DICOM metadata (IOP/IPP/PixelSpacing),
  then converting LPS -> RAS (NIfTI/Slicer)
- progress reporting (tqdm when available; plain prints otherwise)
- robustness for:
    * mixed Rows/Cols/IOP/PixelSpacing in one folder -> uses dominant geometry and warns
    * missing frames -> leaves zeros (legacy-compatible behavior)
    * avoids np.stack shape failures
- standardized outputs so Python/NiiVue/3D Slicer agree:
    * enforces consistent qform/sform
    * for MAGNITUDE (scalar): canonicalizes to RAS+ (permutations/flips allowed)
    * for PHASE components (Phase_Vx/Vy/Vz):
        - canonicalizes to RAS+ (permutations/flips allowed by default)
        - applies LPS->RAS component sign convention: Vx and Vy invert sign, Vz does not
          (for both dcm2niix and direct-read outputs).

IMPORTANT (raw phase):
- Does NOT apply modality LUT / rescale slope-intercept; raw phase values are preserved.
  If you need VENC scaling, do it downstream in a controlled step.
"""

import os
import re
import glob
import shutil
import subprocess
import time
import argparse
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from pydicom.uid import generate_uid

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "sorted_patients"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "nifti_patients"

# --------------------------
# LPS -> RAS constants
# --------------------------
LPS2RAS_4 = np.diag([-1.0, -1.0,  1.0, 1.0]).astype(np.float64)

# --------------------------
# Progress (optional tqdm)
# --------------------------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _now() -> float:
    return time.time()


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or np.isinf(seconds):
        return " ?.?s"
    if seconds < 60:
        return f"{seconds:5.1f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:3d}m{s:04.1f}s"


class SimpleProgress:
    def __init__(self, total: int, desc: str = "", every: int = 25):
        self.total = int(total)
        self.desc = desc
        self.every = max(1, int(every))
        self.n = 0
        self.t0 = _now()

    def update(self, k: int = 1):
        self.n += k
        if (self.n % self.every == 0) or (self.n == self.total):
            dt = _now() - self.t0
            rate = self.n / dt if dt > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else float("inf")
            pct = 100.0 * self.n / self.total if self.total > 0 else 100.0
            print(f"         {self.desc} {self.n}/{self.total} ({pct:5.1f}%) | {rate:7.2f}/s | ETA {_fmt_eta(eta)}")

    def close(self):
        pass


def make_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, leave=False)
    return SimpleProgress(total=total, desc=desc, every=max(1, total // 20))


# --------------------------
# Regex / series type helpers
# --------------------------
def is_vx(name: str) -> bool:
    return bool(re.search(r"Phase_?Vx", name, re.IGNORECASE))


def is_vy(name: str) -> bool:
    return bool(re.search(r"Phase_?Vy", name, re.IGNORECASE))


def is_vz(name: str) -> bool:
    return bool(re.search(r"Phase_?Vz", name, re.IGNORECASE))


def is_phase_component(name: str) -> bool:
    base = os.path.basename(name)
    return is_vx(base) or is_vy(base) or is_vz(base)


def apply_lps2ras_component_sign(data: np.ndarray, series_name: str) -> np.ndarray:
    """
    Convention for vector components when converting LPS -> RAS:
      - X (L/R) changes sign
      - Y (P/A) changes sign
      - Z (I/S) does not change

    For unsigned RAW phase (0..~4096, centered near 2048), negate around the raw
    center instead of applying `-data`, so raw encoding remains valid.
    """
    if not (is_vx(series_name) or is_vy(series_name)):
        return data

    out = data.astype(np.float32, copy=False)
    mn = float(np.nanmin(out))
    mx = float(np.nanmax(out))
    is_unsigned_raw = (mn >= 0.0) and (mx > 1000.0) and (mx <= 8192.0)
    if is_unsigned_raw:
        # Typical center is 2048 for 12-bit raw phase. Keep 8192 support as a guard.
        center = 4096.0 if mx > 5000.0 else 2048.0
        out = (2.0 * center) - out
    else:
        out = -out
    return out.astype(np.float32, copy=False)


# --------------------------
# DICOM affine (LPS) and conversion to RAS
# --------------------------
def dicom_affine_from_iop_ipp(iop, ipp, pixel_spacing, slice_spacing) -> np.ndarray:
    """
    Builds affine in patient LPS coordinates from:
      - IOP (6): row_cos, col_cos (DICOM)
      - IPP (3): voxel position [row=0,col=0] for the reference slice
      - PixelSpacing [row_spacing, col_spacing]
      - slice spacing (mm) along the slice normal

    If data is stored as (X=cols, Y=rows, Z=slices):
      axis0 (X) = row_cos * col_spacing
      axis1 (Y) = -col_cos * row_spacing
      axis2 (Z) = normal  * slice_spacing

    Note: the sign on axis1 follows the same in-plane convention observed in
    dcm2niix outputs for these Siemens 4D-flow slices.
    """
    iop = np.asarray(iop, dtype=np.float64).reshape(-1)
    ipp = np.asarray(ipp, dtype=np.float64).reshape(-1)
    ps = np.asarray(pixel_spacing, dtype=np.float64).reshape(-1)

    if iop.size != 6 or ipp.size != 3 or ps.size != 2:
        raise ValueError("Invalid IOP/IPP/PixelSpacing sizes")

    row_cos = iop[0:3]
    col_cos = iop[3:6]
    slc_cos = np.cross(row_cos, col_cos)

    row_cos /= np.linalg.norm(row_cos) + 1e-12
    col_cos /= np.linalg.norm(col_cos) + 1e-12
    slc_cos /= np.linalg.norm(slc_cos) + 1e-12

    row_spacing = float(ps[0])
    col_spacing = float(ps[1])

    aff = np.eye(4, dtype=np.float64)
    aff[0:3, 0] = row_cos * col_spacing
    aff[0:3, 1] = -col_cos * row_spacing
    aff[0:3, 2] = slc_cos * float(slice_spacing)
    aff[0:3, 3] = ipp
    return aff


def lps_to_ras_affine(aff_lps: np.ndarray) -> np.ndarray:
    return LPS2RAS_4 @ aff_lps


# --------------------------
# Consistent qform/sform
# --------------------------
def force_qsform(img: nib.Nifti1Image, code: int = 1) -> nib.Nifti1Image:
    hdr = img.header.copy()
    hdr.set_qform(img.affine, code=code)
    hdr.set_sform(img.affine, code=code)
    return nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine, hdr)


# --------------------------
# Controlled reorientation
# --------------------------
def _apply_ornt_to_img(img: nib.Nifti1Image, transform: np.ndarray) -> nib.Nifti1Image:
    """
    Apply a (3x2) orientation transform to spatial axes of a 3D/4D image,
    and update affine consistently.
    """
    data = img.get_fdata(dtype=np.float32)
    data2 = nib.orientations.apply_orientation(data, transform)
    aff2 = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape[:3])
    out = nib.Nifti1Image(data2, aff2, img.header.copy())
    out = force_qsform(out, code=1)
    return out


def canonicalize_scalar_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    For scalar images: full canonical RAS+ (permutations/flips allowed).
    """
    img2 = nib.as_closest_canonical(img)
    img2 = force_qsform(img2, code=1)
    return img2


def flip_only_to_ras_if_possible(img: nib.Nifti1Image):
    """
    For phase components: avoid axis permutations because that can break Vx/Vy/Vz semantics when files are separate.

    So:
      - inspect current orientation codes
      - if axis order is already (X,Y,Z), apply flips only
      - if permutation is required, return (None, warning) for caller handling.
    """
    ax = nib.aff2axcodes(img.affine)  # e.g. ('L','P','S') o ('R','A','S')
    # We require axis order X,Y,Z (no axis permutation):
    # X must be L/R, Y must be A/P, Z must be S/I
    ok_order = (ax[0] in ("L", "R")) and (ax[1] in ("A", "P")) and (ax[2] in ("S", "I"))
    if not ok_order:
        return None, f"Would require axis permutation (axcodes={ax}). For Phase_Vx/Vy/Vz this is not safe with separate files."

    # Build transform with flips only (no permutations).
    # transform[:, 0] is input axis index [0, 1, 2], transform[:, 1] is axis sign (+1/-1).
    transform = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int64)

    # If X is 'L', flip to make it 'R'
    if ax[0] == "L":
        transform[0, 1] = -1
    # If Y is 'P', flip to make it 'A'
    if ax[1] == "P":
        transform[1, 1] = -1
    # If Z is 'I', flip to make it 'S'
    if ax[2] == "I":
        transform[2, 1] = -1

    if np.all(transform == np.array([[0, 1], [1, 1], [2, 1]])):
        # Already RAS-oriented
        out = force_qsform(img, code=1)
        return out, None

    out = _apply_ornt_to_img(img, transform)
    return out, None


# --------------------------
# Time parsing (for sorting)
# --------------------------
def parse_acquisition_time(ds) -> float:
    t = ds.get("AcquisitionTime", None)
    if t is None:
        return np.nan
    try:
        s = str(t).strip()
        if len(s) < 6:
            return np.nan
        hh = int(s[0:2])
        mm = int(s[2:4])
        ss = float(s[4:])
        return 3600 * hh + 60 * mm + ss
    except Exception:
        return np.nan


def time_key(ds):
    trig = ds.get("TriggerTime", None)
    if trig is not None:
        try:
            return ("TriggerTime", float(trig))
        except Exception:
            pass

    tpi = ds.get("TemporalPositionIdentifier", None)
    if tpi is not None:
        try:
            return ("TemporalPositionIdentifier", int(tpi))
        except Exception:
            pass

    acq = parse_acquisition_time(ds)
    if not np.isnan(acq):
        return ("AcquisitionTime", float(acq))

    inst = ds.get("InstanceNumber", None)
    try:
        return ("InstanceNumber", int(inst) if inst is not None else 0)
    except Exception:
        return ("InstanceNumber", 0)


# --------------------------
# Direct 4D DICOM read (raw phase / raw magnitude)
# --------------------------
def read_dicom_series_4d_build_nifti(
    dicom_dir: str,
    out_nii_gz: str,
    series_name_for_phase_logic: str,
    canonicalize: bool = True,
    phase_flips_only: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Read DICOMs and build volume as (X=cols, Y=rows, Z, T).
    - Does not apply LUT or rescale (raw phase preserved).
    - Affine is computed from DICOM (LPS) and converted to RAS.
    - For Phase_Vx/Vy: sign is inverted during LPS->RAS conversion.
    """
    files = glob.glob(os.path.join(dicom_dir, "*"))
    files = [f for f in files if f.upper().endswith((".IMA", ".DCM"))]
    if len(files) == 0:
        if verbose:
            print("      [!] No .IMA/.DCM DICOM files found in:", dicom_dir)
        return False

    if verbose:
        print(f"      -> Direct read: {len(files)} DICOMs (affine from DICOM metadata, no padding/cropping)...")

    # 1) Read headers (without pixel data) for geometry and temporal sorting.
    meta = []
    pbar = make_progress(len(files), "reading headers")
    try:
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            rows = int(ds.get("Rows", 0) or 0)
            cols = int(ds.get("Columns", 0) or 0)
            iop = ds.get("ImageOrientationPatient", None)
            ipp = ds.get("ImagePositionPatient", None)
            ps = ds.get("PixelSpacing", None)

            if iop is None or ipp is None or ps is None or rows <= 0 or cols <= 0:
                pbar.update(1)
                continue

            iop = np.array([float(x) for x in iop], dtype=np.float64)
            ipp = np.array([float(x) for x in ipp], dtype=np.float64)
            ps = np.array([float(x) for x in ps], dtype=np.float64)

            key_geom = (
                rows,
                cols,
                tuple(np.round(iop, 6)),
                tuple(np.round(ps, 6)),
            )

            meta.append(
                {
                    "path": f,
                    "rows": rows,
                    "cols": cols,
                    "iop": iop,
                    "ipp": ipp,
                    "ps": ps,
                    "geom_key": key_geom,
                    "inst": int(ds.get("InstanceNumber", 0) or 0),
                    "tkey": time_key(ds),
                }
            )
            pbar.update(1)
    finally:
        if hasattr(pbar, "close"):
            pbar.close()

    if len(meta) == 0:
        if verbose:
            print("      [!] Could not extract enough metadata (IOP/IPP/PixelSpacing).")
        return False

    # 2) Keep dominant geometry (e.g., mixed 448 vs 484 acquisitions)
    counts = Counter([m["geom_key"] for m in meta])
    main_geom, main_n = counts.most_common(1)[0]

    if len(counts) > 1 and verbose:
        others = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        msg = ", ".join([f"{k[0]}x{k[1]}:{v}" for k, v in others[:6]])
        print(f"      [!] Warning: mixed geometries detected. Using {main_geom[0]}x{main_geom[1]} ({main_n}). Others: {msg}")

    meta = [m for m in meta if m["geom_key"] == main_geom]
    if len(meta) == 0:
        if verbose:
            print("      [!] No files left after filtering to dominant geometry.")
        return False

    rows = main_geom[0]
    cols = main_geom[1]
    iop0 = np.array(main_geom[2], dtype=np.float64)
    ps0 = np.array(main_geom[3], dtype=np.float64)

    # 3) Z ordering: project IPP onto slice normal
    row_cos = iop0[0:3]
    col_cos = iop0[3:6]
    slc_cos = np.cross(row_cos, col_cos)
    slc_cos /= np.linalg.norm(slc_cos) + 1e-12

    z_coord = [float(np.dot(m["ipp"], slc_cos)) for m in meta]
    zq = np.round(z_coord, 3)
    for m, z in zip(meta, zq):
        m["zq"] = float(z)

    z_values = sorted(list(set([m["zq"] for m in meta])))
    num_z = len(z_values)

    # slice spacing
    if num_z > 1:
        dz = np.diff(np.array(z_values, dtype=np.float64))
        slice_spacing = float(np.median(np.abs(dz)))
        if slice_spacing <= 0:
            slice_spacing = 1.0
    else:
        slice_spacing = 1.0

    # 4) T
    tkeys = [m["tkey"] for m in meta]
    t_types = [tk[0] for tk in tkeys]
    ttype = Counter(t_types).most_common(1)[0][0]

    tvals = [tk[1] for tk in tkeys if tk[0] == ttype]
    unique_t = sorted(list(set([round(float(x), 3) for x in tvals]))) if len(tvals) > 0 else [0.0]

    use_explicit_time = len(unique_t) > 1
    if not use_explicit_time:
        num_t = max(1, int(round(len(meta) / max(1, num_z))))
        unique_t = list(range(num_t))
    else:
        num_t = len(unique_t)

    if verbose:
        print(f"      -> Geometry: rows={rows} cols={cols} | Z={num_z} | T={num_t}")

    # 5) Build volume as (X=cols, Y=rows, Z, T) using int16.
    # For exact dtype preservation, inspect the first frame dtype and branch accordingly.
    vol = np.zeros((cols, rows, num_z, num_t), dtype=np.int16)

    z_to_idx = {z: i for i, z in enumerate(z_values)}
    if use_explicit_time:
        t_to_idx = {round(float(t), 3): i for i, t in enumerate(unique_t)}

    meta_sorted = sorted(meta, key=lambda m: (m["tkey"][0], float(m["tkey"][1]), m["zq"], m["inst"]))
    pbar2 = make_progress(len(meta_sorted), "filling volume")

    # Reference IPP for affine: smallest time and slice position
    ref_item = min(meta_sorted, key=lambda m: (m["tkey"][0], float(m["tkey"][1]), m["zq"], m["inst"]))
    ipp0 = np.array(ref_item["ipp"], dtype=np.float64)

    missing = 0
    try:
        if not use_explicit_time:
            per_z_counter = defaultdict(int)

        for m in meta_sorted:
            # z index
            if m["zq"] not in z_to_idx:
                missing += 1
                pbar2.update(1)
                continue
            z_idx = z_to_idx[m["zq"]]

            # t index
            if use_explicit_time:
                t_val = round(float(m["tkey"][1]), 3) if m["tkey"][0] == ttype else round(float(unique_t[0]), 3)
                if t_val not in t_to_idx:
                    arr = np.array(list(t_to_idx.keys()), dtype=np.float64)
                    t_val = float(arr[np.argmin(np.abs(arr - float(t_val)))])
                t_idx = t_to_idx[t_val]
            else:
                t_idx = per_z_counter[z_idx]
                if t_idx >= num_t:
                    missing += 1
                    pbar2.update(1)
                    continue
                per_z_counter[z_idx] += 1

            ds = pydicom.dcmread(m["path"], stop_before_pixels=False, force=True)
            arr = np.asarray(ds.pixel_array)

            if arr.shape != (rows, cols):
                missing += 1
                pbar2.update(1)
                continue

            # Keep raw values (no LUT/rescale), but store as (cols, rows) to
            # match the affine convention used above.
            vol[:, :, z_idx, t_idx] = arr.T.astype(np.int16, copy=False)
            pbar2.update(1)
    finally:
        if hasattr(pbar2, "close"):
            pbar2.close()

    expected = num_z * num_t
    nonzero_frames = np.sum(np.any(vol != 0, axis=(0, 1)))
    empty_frames = expected - int(nonzero_frames)
    if empty_frames > 0 and verbose:
        print(f"      [!] Warning: missing {empty_frames} frames (slice/time). Frames remain zero-filled (legacy-compatible behavior).")

    # Match dcm2niix in-plane convention for these Siemens exports.
    vol = np.flip(vol, axis=1)

    # 6) affine LPS -> RAS
    aff_lps = dicom_affine_from_iop_ipp(iop0, ipp0, ps0, slice_spacing)
    # Compensate translation for the in-plane axis-1 flip above.
    aff_lps[0:3, 3] = aff_lps[0:3, 3] - aff_lps[0:3, 1] * float(rows - 1)
    aff_ras = lps_to_ras_affine(aff_lps)

    # 7) Component values: LPS->RAS flips sign for Vx/Vy
    vol_out = vol.astype(np.float32)  # allow negatives without overflow if input was unsigned
    if is_phase_component(series_name_for_phase_logic):
        vol_out = apply_lps2ras_component_sign(vol_out, series_name_for_phase_logic)

    img = nib.Nifti1Image(vol_out, aff_ras)
    img = force_qsform(img, code=1)

    # 8) Standardize final orientation
    if canonicalize:
        if is_phase_component(series_name_for_phase_logic):
            if phase_flips_only:
                img2, warn = flip_only_to_ras_if_possible(img)
                if img2 is None:
                    if verbose:
                        print(f"      [!] WARNING: {warn}")
                        print("          -> Keeping non-canonical orientation (without permutation) due to --phase-flips-only.")
                    img2 = img
                img = force_qsform(img2, code=1)
            else:
                img = canonicalize_scalar_ras(img)
        else:
            img = canonicalize_scalar_ras(img)

    # 9) Save
    os.makedirs(os.path.dirname(out_nii_gz), exist_ok=True)
    nib.save(img, out_nii_gz)
    if verbose:
        print(f"      -> Saved: {out_nii_gz} | shape={img.shape} | AxCodes={nib.aff2axcodes(img.affine)}")
    return True


# --------------------------
# dcm2niix helpers
# --------------------------
def run_dcm2niix(
    dcm2niix_path: str,
    dicom_dir: str,
    out_dir: str,
    fname: str = "%p_%s_%r_%t",
):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        dcm2niix_path,
        "-z",
        "y",
        "-m",
        "y",
        "-f",
        fname,
        "-o",
        out_dir,
        dicom_dir,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nii = sorted(glob.glob(os.path.join(out_dir, "*.nii.gz")))
    return p.returncode, nii, p.stdout.decode(errors="ignore"), p.stderr.decode(errors="ignore")


def build_uid_repaired_dicom_copy(src_dicom_dir: str, dst_dicom_dir: str) -> int:
    """
    Copy a DICOM series and normalize grouping IDs so dcm2niix can merge
    single-frame objects into a volumetric output.
    Orientation/direction tags are not modified.
    """
    os.makedirs(dst_dicom_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dicom_dir, "*")))
    files = [f for f in files if f.upper().endswith((".IMA", ".DCM"))]
    if len(files) == 0:
        return 0

    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    series_uid = str(ds0.get("SeriesInstanceUID", generate_uid()))
    study_uid = str(ds0.get("StudyInstanceUID", generate_uid()))
    frame_uid = str(ds0.get("FrameOfReferenceUID", generate_uid()))

    written = 0
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            ds.SeriesInstanceUID = series_uid
            ds.StudyInstanceUID = study_uid
            ds.FrameOfReferenceUID = frame_uid
            out_f = os.path.join(dst_dicom_dir, os.path.basename(f))
            ds.save_as(out_f, write_like_original=True)
            written += 1
        except Exception:
            continue
    return written


def dcm2niix_output_ok(nifti_path: str, num_dicoms: int) -> bool:
    """
    Simple data-loss heuristic:
      - if NIfTI has Z*T much smaller than #DICOMs, temporal collapse likely occurred
    """
    try:
        img = nib.load(nifti_path)
        sh = img.shape
        # Accept any volumetric output from dcm2niix; some scanners export
        # duplicated DICOM objects per frame, so raw DICOM count can be much
        # larger than Z*T even when conversion is correct.
        if len(sh) < 3:
            return False
        z = int(sh[2])
        t = int(sh[3]) if len(sh) >= 4 else 1
        return (z > 1) and (t >= 1)
    except Exception:
        return False


def get_nifti_axcodes(nifti_path: str):
    try:
        return nib.aff2axcodes(nib.load(nifti_path).affine)
    except Exception:
        return None


def choose_dcm2niix_candidate(nifti_paths):
    """
    Pick the best volumetric candidate among dcm2niix outputs.
    Returns (path_or_none, info_list_sorted_by_size_desc).
    """
    info = []
    for p in nifti_paths:
        try:
            sh = nib.load(p).shape
            if len(sh) < 3:
                continue
            z = int(sh[2])
            t = int(sh[3]) if len(sh) >= 4 else 1
            if z <= 1:
                continue
            size4 = int(np.prod(sh[:4])) if len(sh) >= 4 else int(np.prod(sh[:3]))
            info.append((size4, t, z, p, sh))
        except Exception:
            continue

    info.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    if not info:
        return None, info
    return info[0][3], info


def summarize_nifti_shapes(nifti_paths, max_items: int = 5) -> str:
    """
    Compact shape summary for debugging dcm2niix multi-file outputs.
    """
    counts = Counter()
    for p in nifti_paths:
        try:
            counts[tuple(int(x) for x in nib.load(p).shape)] += 1
        except Exception:
            continue
    if not counts:
        return "unknown"
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join([f"{sh}:{n}" for sh, n in items[:max_items]])


def parse_instance_from_generated_name(nifti_path: str):
    """
    Extract likely DICOM InstanceNumber from dcm2niix output filename.
    For names like <proto>_<SeriesNumber>_<InstanceNumber>_<AcqTime>.nii.gz,
    this returns the second-to-last numeric token.
    """
    base = os.path.basename(nifti_path)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    parts = base.split("_")
    numeric = [p for p in parts if re.fullmatch(r"\d+", p)]
    if len(numeric) >= 2:
        try:
            return int(numeric[-2])
        except Exception:
            return None
    if len(numeric) == 1:
        try:
            return int(numeric[0])
        except Exception:
            return None
    return None


def try_stack_dcm2niix_2d_outputs(
    generated_niftis,
    dicom_dir: str,
    out_nii_gz: str,
    series_name_for_phase_logic: str,
    canonicalize: bool = True,
    phase_flips_only: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Attempt to reconstruct a 4D volume from many single-slice dcm2niix NIfTIs.
    This is intended for series where dcm2niix emits one file per DICOM frame.
    """
    single_slice = []
    for p in generated_niftis:
        try:
            sh = nib.load(p).shape
        except Exception:
            continue
        if len(sh) == 2:
            single_slice.append(p)
        elif len(sh) == 3 and int(sh[2]) == 1:
            single_slice.append(p)

    if len(single_slice) < 2:
        return False

    # Build DICOM frame layout (same geometry/time logic as direct-read fallback).
    files = glob.glob(os.path.join(dicom_dir, "*"))
    files = [f for f in files if f.upper().endswith((".IMA", ".DCM"))]
    if len(files) == 0:
        return False

    meta = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        rows = int(ds.get("Rows", 0) or 0)
        cols = int(ds.get("Columns", 0) or 0)
        iop = ds.get("ImageOrientationPatient", None)
        ipp = ds.get("ImagePositionPatient", None)
        ps = ds.get("PixelSpacing", None)
        if iop is None or ipp is None or ps is None or rows <= 0 or cols <= 0:
            continue
        iop = np.array([float(x) for x in iop], dtype=np.float64)
        ipp = np.array([float(x) for x in ipp], dtype=np.float64)
        ps = np.array([float(x) for x in ps], dtype=np.float64)
        key_geom = (
            rows,
            cols,
            tuple(np.round(iop, 6)),
            tuple(np.round(ps, 6)),
        )
        meta.append(
            {
                "rows": rows,
                "cols": cols,
                "iop": iop,
                "ipp": ipp,
                "ps": ps,
                "geom_key": key_geom,
                "inst": int(ds.get("InstanceNumber", 0) or 0),
                "tkey": time_key(ds),
            }
        )

    if len(meta) == 0:
        return False

    counts = Counter([m["geom_key"] for m in meta])
    main_geom, _ = counts.most_common(1)[0]
    meta = [m for m in meta if m["geom_key"] == main_geom]
    if len(meta) == 0:
        return False

    rows = main_geom[0]
    cols = main_geom[1]
    iop0 = np.array(main_geom[2], dtype=np.float64)
    ps0 = np.array(main_geom[3], dtype=np.float64)

    row_cos = iop0[0:3]
    col_cos = iop0[3:6]
    slc_cos = np.cross(row_cos, col_cos)
    slc_cos /= np.linalg.norm(slc_cos) + 1e-12

    z_coord = [float(np.dot(m["ipp"], slc_cos)) for m in meta]
    zq = np.round(z_coord, 3)
    for m, z in zip(meta, zq):
        m["zq"] = float(z)
    z_values = sorted(list(set([m["zq"] for m in meta])))
    num_z = len(z_values)
    if num_z <= 0:
        return False

    if num_z > 1:
        dz = np.diff(np.array(z_values, dtype=np.float64))
        slice_spacing = float(np.median(np.abs(dz)))
        if slice_spacing <= 0:
            slice_spacing = 1.0
    else:
        slice_spacing = 1.0

    tkeys = [m["tkey"] for m in meta]
    t_types = [tk[0] for tk in tkeys]
    ttype = Counter(t_types).most_common(1)[0][0]
    tvals = [tk[1] for tk in tkeys if tk[0] == ttype]
    unique_t = sorted(list(set([round(float(x), 3) for x in tvals]))) if len(tvals) > 0 else [0.0]
    use_explicit_time = len(unique_t) > 1
    if not use_explicit_time:
        num_t = max(1, int(round(len(meta) / max(1, num_z))))
        unique_t = list(range(num_t))
    else:
        num_t = len(unique_t)

    z_to_idx = {z: i for i, z in enumerate(z_values)}
    if use_explicit_time:
        t_to_idx = {round(float(t), 3): i for i, t in enumerate(unique_t)}

    meta_sorted = sorted(meta, key=lambda m: (m["tkey"][0], float(m["tkey"][1]), m["zq"], m["inst"]))
    ref_item = min(meta_sorted, key=lambda m: (m["tkey"][0], float(m["tkey"][1]), m["zq"], m["inst"]))
    ipp0 = np.array(ref_item["ipp"], dtype=np.float64)

    # Map InstanceNumber -> list of (z,t) frame slots.
    inst_to_slots = defaultdict(list)
    per_z_counter = defaultdict(int)
    for m in meta_sorted:
        z_idx = z_to_idx.get(m["zq"], None)
        if z_idx is None:
            continue
        if use_explicit_time:
            t_val = round(float(m["tkey"][1]), 3) if m["tkey"][0] == ttype else round(float(unique_t[0]), 3)
            if t_val not in t_to_idx:
                arr = np.array(list(t_to_idx.keys()), dtype=np.float64)
                t_val = float(arr[np.argmin(np.abs(arr - float(t_val)))])
            t_idx = t_to_idx[t_val]
        else:
            t_idx = per_z_counter[z_idx]
            if t_idx >= num_t:
                continue
            per_z_counter[z_idx] += 1
        inst_to_slots[int(m["inst"])].append((z_idx, t_idx))

    vol = np.zeros((cols, rows, num_z, num_t), dtype=np.float32)
    filled = 0

    ordered_slices = sorted(
        single_slice,
        key=lambda p: (parse_instance_from_generated_name(p) is None, parse_instance_from_generated_name(p), p),
    )
    for p in ordered_slices:
        inst = parse_instance_from_generated_name(p)
        if inst is None:
            continue
        slots = inst_to_slots.get(int(inst), None)
        if not slots:
            continue
        z_idx, t_idx = slots.pop(0)
        try:
            im = nib.as_closest_canonical(nib.load(p))
            arr = np.asarray(im.dataobj)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            if arr.ndim != 2:
                continue
        except Exception:
            continue

        if arr.shape == (cols, rows):
            frame_xy = arr
        elif arr.shape == (rows, cols):
            frame_xy = arr.T
        else:
            continue

        vol[:, :, z_idx, t_idx] = frame_xy.astype(np.float32, copy=False)
        filled += 1

    expected = num_z * num_t
    if expected <= 0:
        return False
    if filled == 0:
        if verbose:
            print("      [dcm2niix-2D stack] no frames mapped -> fallback to direct-read.")
        return False
    empty_frames = expected - filled

    aff_lps = dicom_affine_from_iop_ipp(iop0, ipp0, ps0, slice_spacing)
    aff_lps[0:3, 3] = aff_lps[0:3, 3] - aff_lps[0:3, 1] * float(rows - 1)
    aff_ras = lps_to_ras_affine(aff_lps)

    img = nib.Nifti1Image(vol, aff_ras)
    img = force_qsform(img, code=1)

    if canonicalize:
        if is_phase_component(series_name_for_phase_logic):
            if phase_flips_only:
                img2, warn = flip_only_to_ras_if_possible(img)
                if img2 is None:
                    if verbose:
                        print(f"      [!] WARNING: {warn}")
                        print("          -> Keeping non-canonical orientation (without permutation) due to --phase-flips-only.")
                    img2 = img
                img = force_qsform(img2, code=1)
            else:
                img = canonicalize_scalar_ras(img)
        else:
            img = canonicalize_scalar_ras(img)

    if is_phase_component(series_name_for_phase_logic):
        data = img.get_fdata(dtype=np.float32)
        data = apply_lps2ras_component_sign(data, series_name_for_phase_logic)
        hdr = img.header.copy()
        hdr.set_data_dtype(np.float32)
        img = nib.Nifti1Image(data, img.affine, hdr)
        img = force_qsform(img, code=1)

    os.makedirs(os.path.dirname(out_nii_gz), exist_ok=True)
    nib.save(img, out_nii_gz)
    if verbose:
        if empty_frames > 0:
            print(
                f"      [dcm2niix-2D stack] Warning: missing {empty_frames} frames (slice/time). "
                "Frames remain zero-filled (legacy-compatible behavior)."
            )
        print(
            f"      [dcm2niix-2D stack] OK | frames={filled}/{expected} | "
            f"shape={img.shape} | AxCodes={nib.aff2axcodes(img.affine)}"
        )
    return True


def standardize_from_dcm2niix(
    nifti_in: str,
    nifti_out: str,
    series_name: str,
    canonicalize: bool = True,
    phase_flips_only: bool = False,
):
    """
    dcm2niix usually writes RAS already. This step:
      - enforces qform/sform
      - for scalars: canonicalizes to RAS+
      - for phase components: canonicalizes to RAS+ by default (or flips-only with
        --phase-flips-only), then applies Vx/Vy sign correction.
    """
    img = nib.load(nifti_in)
    img = force_qsform(img, code=1)

    if canonicalize:
        if is_phase_component(series_name):
            if phase_flips_only:
                img2, warn = flip_only_to_ras_if_possible(img)
                if img2 is None:
                    print(f"      [!] WARNING: {warn}")
                    print("          -> Keeping current orientation (no permutation) due to --phase-flips-only.")
                    img2 = img
                img = force_qsform(img2, code=1)
            else:
                img = canonicalize_scalar_ras(img)
        else:
            img = canonicalize_scalar_ras(img)

    if is_phase_component(series_name):
        data = img.get_fdata(dtype=np.float32)
        data = apply_lps2ras_component_sign(data, series_name)
        hdr = img.header.copy()
        hdr.set_data_dtype(np.float32)
        img = nib.Nifti1Image(data, img.affine, hdr)
        img = force_qsform(img, code=1)

    nib.save(img, nifti_out)


# --------------------------
# Main pipeline
# --------------------------
def process_with_fallback(
    input_root: str,
    output_root: str,
    dcm2niix_path: str = "dcm2niix",
    canonicalize: bool = False,
    phase_flips_only: bool = False,
    stack_dcm2niix_2d: bool = False,
    uid_repair_for_dcm2niix: bool = True,
):
    print("--- Processing: dcm2niix + direct-read fallback (RAS + phase-safe) ---")

    # Check dcm2niix availability
    try:
        subprocess.run([dcm2niix_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        print("Error: dcm2niix was not found.")
        return

    patients = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]
    for pi, patient_id in enumerate(patients, start=1):
        print(f"[{pi}/{len(patients)}] {patient_id}...")
        patient_dicom_dir = os.path.join(input_root, patient_id)
        patient_nifti_dir = os.path.join(output_root, patient_id)
        os.makedirs(patient_nifti_dir, exist_ok=True)

        series_dirs = [s for s in sorted(os.listdir(patient_dicom_dir))
                  if os.path.isdir(os.path.join(patient_dicom_dir, s)) and "Unknown" not in s]

        for si, series_name in enumerate(series_dirs, start=1):
            dicom_series_path = os.path.join(patient_dicom_dir, series_name)
            nifti_final_path = os.path.join(patient_nifti_dir, f"{series_name}.nii.gz")

            dicoms = [f for f in os.listdir(dicom_series_path) if f.upper().endswith((".IMA", ".DCM"))]
            num_dicoms = len(dicoms)
            print(f"   - [{si}/{len(series_dirs)}] {series_name} | DICOMs={num_dicoms}")

            # 1) dcm2niix
            temp_dir = os.path.join(patient_nifti_dir, f"temp_{series_name}")
            # Ensure no stale outputs from previous interrupted runs.
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            code, generated, dcm2_stdout, dcm2_stderr = run_dcm2niix(
                dcm2niix_path,
                dicom_series_path,
                temp_dir,
            )

            use_direct_read = True
            if code == 0 and len(generated) >= 1:
                chosen, info = choose_dcm2niix_candidate(generated)
                if chosen is not None and dcm2niix_output_ok(chosen, num_dicoms):
                    ax_raw = get_nifti_axcodes(chosen)
                    if len(generated) == 1:
                        print("      [dcm2niix] OK")
                    else:
                        size4, t, z, p, sh = info[0]
                        print(
                            f"      [dcm2niix] Produced {len(generated)} files; using largest volumetric output "
                            f"{os.path.basename(p)} shape={sh}."
                        )
                    print(f"      [dcm2niix] Raw AxCodes={ax_raw}")
                    if ax_raw is not None and ax_raw != ("R", "A", "S") and not canonicalize:
                        print("      [!] WARNING: dcm2niix output is not RAS and canonicalization is disabled.")
                    standardize_from_dcm2niix(
                        chosen,
                        nifti_final_path,
                        series_name=series_name,
                        canonicalize=canonicalize,
                        phase_flips_only=phase_flips_only,
                    )
                    use_direct_read = False
                else:
                    if len(generated) > 1:
                        shp = summarize_nifti_shapes(generated)
                        print(f"      [dcm2niix] Produced {len(generated)} NIfTI files but no reliable volumetric candidate.")
                        print(f"      [dcm2niix] Output shapes summary: {shp}")
                        repaired_used = False
                        if uid_repair_for_dcm2niix:
                            fix_in = tempfile.mkdtemp(prefix="d2n_uidfix_in_")
                            fix_out = tempfile.mkdtemp(prefix="d2n_uidfix_out_")
                            try:
                                written = build_uid_repaired_dicom_copy(dicom_series_path, fix_in)
                                if written > 0:
                                    code2, generated2, dcm2_stdout2, dcm2_stderr2 = run_dcm2niix(
                                        dcm2niix_path,
                                        fix_in,
                                        fix_out,
                                    )
                                    chosen2, info2 = choose_dcm2niix_candidate(generated2)
                                    if code2 == 0 and chosen2 is not None and dcm2niix_output_ok(chosen2, num_dicoms):
                                        ax_raw2 = get_nifti_axcodes(chosen2)
                                        if len(generated2) == 1:
                                            print("      [dcm2niix uid-repair] OK")
                                        else:
                                            _, _, _, p2, sh2 = info2[0]
                                            print(
                                                "      [dcm2niix uid-repair] Produced multiple files; using "
                                                f"{os.path.basename(p2)} shape={sh2}."
                                            )
                                        print(f"      [dcm2niix uid-repair] Raw AxCodes={ax_raw2}")
                                        if ax_raw2 is not None and ax_raw2 != ("R", "A", "S") and not canonicalize:
                                            print("      [!] WARNING: uid-repair dcm2niix output is not RAS and canonicalization is disabled.")
                                        standardize_from_dcm2niix(
                                            chosen2,
                                            nifti_final_path,
                                            series_name=series_name,
                                            canonicalize=canonicalize,
                                            phase_flips_only=phase_flips_only,
                                        )
                                        use_direct_read = False
                                        repaired_used = True
                                    else:
                                        shp2 = summarize_nifti_shapes(generated2)
                                        print(f"      [dcm2niix uid-repair] Not enough: shapes={shp2}.")
                            finally:
                                shutil.rmtree(fix_in, ignore_errors=True)
                                shutil.rmtree(fix_out, ignore_errors=True)

                        if (not repaired_used) and stack_dcm2niix_2d:
                            stacked_ok = try_stack_dcm2niix_2d_outputs(
                                generated,
                                dicom_series_path,
                                nifti_final_path,
                                series_name_for_phase_logic=series_name,
                                canonicalize=canonicalize,
                                phase_flips_only=phase_flips_only,
                                verbose=True,
                            )
                            if stacked_ok:
                                use_direct_read = False
                            else:
                                print("      -> dcm2niix 2D stack not usable; switching to direct-read fallback.")
                                use_direct_read = True
                        elif not repaired_used:
                            print("      -> switching to direct-read fallback (stack disabled).")
                            use_direct_read = True
                    else:
                        print("      [dcm2niix] Potential data loss detected -> switching to direct-read fallback.")
                        use_direct_read = True
            else:
                print("      [dcm2niix] Failed or produced no NIfTI -> switching to direct-read fallback.")
                if code != 0:
                    merged_msg = "\n".join(
                        [line.strip() for line in (dcm2_stderr + "\n" + dcm2_stdout).splitlines() if line.strip()]
                    )
                    if merged_msg:
                        # Show only the last lines to keep logs readable.
                        tail = "\n".join(merged_msg.splitlines()[-5:])
                        print(f"      [dcm2niix] exit_code={code}. Last messages:\n{tail}")
                use_direct_read = True

            # 2) direct-read fallback
            if use_direct_read:
                ok = read_dicom_series_4d_build_nifti(
                    dicom_series_path,
                    nifti_final_path,
                    series_name_for_phase_logic=series_name,
                    canonicalize=canonicalize,
                    phase_flips_only=phase_flips_only,
                    verbose=True,
                )
                if not ok:
                    print(f"      [!] Critical failure in direct-read fallback: {series_name}")

            # cleanup temp
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------
# MAIN
# --------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sorted DICOM folders to NIfTI with dcm2niix + robust fallback.")
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root folder containing per-patient DICOM folders",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Destination folder for per-patient NIfTI outputs",
    )
    parser.add_argument("--dcm2niix", default="dcm2niix", help="Path to dcm2niix executable")
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Recanonicalize outputs to RAS+ (disabled by default).",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-flips-only",
        action="store_true",
        help="For phase components, avoid axis permutations and use flips-only orientation fixes.",
    )
    parser.add_argument(
        "--stack-dcm2niix-2d",
        action="store_true",
        help="Enable stacking dcm2niix single-slice outputs into 4D before direct-read fallback (disabled by default).",
    )
    parser.add_argument(
        "--no-uid-repair-for-dcm2niix",
        action="store_true",
        help="Disable temporary DICOM UID normalization before retrying dcm2niix when it outputs many single-slice files.",
    )
    parser.add_argument(
        "--no-stack-dcm2niix-2d",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    process_with_fallback(
        input_root=args.input_root,
        output_root=args.output_root,
        dcm2niix_path=args.dcm2niix,
        canonicalize=(args.canonicalize and not args.no_canonicalize),
        phase_flips_only=args.phase_flips_only,
        stack_dcm2niix_2d=(args.stack_dcm2niix_2d and not args.no_stack_dcm2niix_2d),
        uid_repair_for_dcm2niix=(not args.no_uid_repair_for_dcm2niix),
    )


if __name__ == "__main__":
    main()
