import argparse
import json
import os
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_MODEL = REPO_ROOT / "models" / "yolo-cow-detection.pt"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data" / "reports" / "cow_bboxes.json"


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data, img


def save_nifti(data, output_path, reference_img):
    header = reference_img.header.copy()
    if tuple(header.get_data_shape()) != tuple(data.shape):
        header.set_data_shape(data.shape)
    out = nib.Nifti1Image(data, reference_img.affine, header)
    nib.save(out, output_path)


def maybe_mip_4d(data):
    if data.ndim == 4:
        return np.max(data, axis=-1), True
    if data.ndim == 3:
        return data, False
    raise ValueError(f"Unsupported NIfTI shape {data.shape}. Expected 3D or 4D.")


def detect_bbox_with_yolo(volume_3d, model):
    bbox_centers = []
    bbox_widths = []
    bbox_heights = []
    slices_with_roi = []

    for slice_idx in range(volume_3d.shape[2]):
        slice_data = volume_3d[:, :, slice_idx]
        smin, smax = np.min(slice_data), np.max(slice_data)
        if smax > smin:
            norm_slice = (slice_data - smin) / (smax - smin) * 255.0
        else:
            norm_slice = np.zeros_like(slice_data)

        norm_slice = norm_slice.astype(np.uint8)
        slice_rgb = cv2.cvtColor(norm_slice, cv2.COLOR_GRAY2RGB)
        results = model.predict(source=slice_rgb, verbose=False)

        for result in results:
            if len(result.boxes.xyxy) > 0:
                slices_with_roi.append(slice_idx)
            for bbox in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, bbox)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min
                bbox_centers.append((center_x, center_y, slice_idx))
                bbox_widths.append(width)
                bbox_heights.append(height)

    return bbox_centers, bbox_widths, bbox_heights, slices_with_roi


def compute_crop_dict(volume_shape, bbox_centers, bbox_widths, bbox_heights, slices_with_roi, margin_xy, margin_z):
    if not bbox_centers:
        return {
            "detected": False,
            "size": [int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])],
            "location": [0, 0, 0],
        }

    median_center_x = int(np.median([c[0] for c in bbox_centers]))
    median_center_y = int(np.median([c[1] for c in bbox_centers]))
    median_center_z = int(np.median([c[2] for c in bbox_centers]))
    median_width = int(np.median(bbox_widths))
    median_height = int(np.median(bbox_heights))
    crop_z_size = len(set(slices_with_roi))

    crop_x_min = max(0, median_center_x - median_width // 2)
    crop_x_max = min(volume_shape[1], median_center_x + median_width // 2)
    crop_y_min = max(0, median_center_y - median_height // 2)
    crop_y_max = min(volume_shape[0], median_center_y + median_height // 2)
    crop_z_min = max(0, median_center_z - crop_z_size // 2)
    crop_z_max = min(volume_shape[2], median_center_z + crop_z_size // 2)

    y_start = max(0, crop_y_min - margin_xy)
    y_end = min(volume_shape[0], crop_y_max + margin_xy)
    x_start = max(0, crop_x_min - margin_xy)
    x_end = min(volume_shape[1], crop_x_max + margin_xy)
    z_start = max(0, crop_z_min - margin_z)
    z_end = min(volume_shape[2], crop_z_max + margin_z)

    return {
        "detected": True,
        "size": [int(y_end - y_start), int(x_end - x_start), int(z_end - z_start)],
        "location": [int(y_start), int(x_start), int(z_start)],
    }


def collect_inputs(path_str, recursive=False):
    path = Path(path_str)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        pattern = "**/*.nii.gz" if recursive else "*.nii.gz"
        files = sorted([str(p) for p in path.glob(pattern)])
        return files
    raise FileNotFoundError(f"Input path not found: {path_str}")


def main():
    parser = argparse.ArgumentParser(description="Compute CoW bounding boxes with YOLO (no segmentation).")
    parser.add_argument("--input", required=True, help="Path to a .nii.gz file or a folder of .nii.gz files.")
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL), help="Path to YOLO weights.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path.")
    parser.add_argument("--margin-xy", type=int, default=20, help="Bounding box margin for X/Y.")
    parser.add_argument("--margin-z", type=int, default=8, help="Bounding box margin for Z.")
    parser.add_argument("--save-crops-dir", default=None, help="Optional folder to save cropped ROI volumes.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .nii.gz files in input dir.")
    args = parser.parse_args()

    model = YOLO(args.yolo_model)
    input_files = collect_inputs(args.input, recursive=args.recursive)
    if len(input_files) == 0:
        hint = " Try --recursive if your cases are in nested folders." if not args.recursive else ""
        raise RuntimeError(f"No .nii.gz files found in input.{hint}")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    if args.save_crops_dir is not None:
        os.makedirs(args.save_crops_dir, exist_ok=True)

    results = []
    for nifti_path in input_files:
        data, img = load_nifti(nifti_path)
        vol3d, used_mip = maybe_mip_4d(data)

        bbox_centers, bbox_widths, bbox_heights, slices_with_roi = detect_bbox_with_yolo(vol3d, model)
        crop_dict = compute_crop_dict(
            vol3d.shape,
            bbox_centers,
            bbox_widths,
            bbox_heights,
            slices_with_roi,
            args.margin_xy,
            args.margin_z,
        )

        item = {
            "image": nifti_path,
            "input_shape": list(data.shape),
            "prediction_shape": list(vol3d.shape),
            "used_mip_from_4d": used_mip,
            "bbox": crop_dict,
        }
        results.append(item)
        print(f"{os.path.basename(nifti_path)} -> {crop_dict}")

        if args.save_crops_dir is not None:
            y0, x0, z0 = crop_dict["location"]
            sy, sx, sz = crop_dict["size"]
            crop = vol3d[y0:y0 + sy, x0:x0 + sx, z0:z0 + sz]
            out_name = os.path.basename(nifti_path).replace(".nii.gz", "_bbox_crop.nii.gz")
            out_path = os.path.join(args.save_crops_dir, out_name)
            save_nifti(crop, out_path, img)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
