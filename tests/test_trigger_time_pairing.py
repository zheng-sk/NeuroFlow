import csv
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "registration"))

import numpy as np

try:
    from Network.NiftiPatchDataset import _StackNormalizeFieldsd, load_nifti_case_table
except Exception:  # pragma: no cover - environment guard
    _StackNormalizeFieldsd = None
    load_nifti_case_table = None

import export_paired_lr_hr_dataset as export_paired_lr_hr_dataset
import generate_trigger_time_frame_pairs as generate_trigger_time_frame_pairs


def make_4d_channel(values):
    arr = np.zeros((1, 2, 2, 2, len(values)), dtype=np.float32)
    for idx, value in enumerate(values):
        arr[0, ..., idx] = float(value)
    return arr


@unittest.skipIf(_StackNormalizeFieldsd is None, "MONAI dataset stack is unavailable in this environment.")
class LoaderPairingTests(unittest.TestCase):
    def test_explicit_lr_hr_indices_override_random_time_frame(self):
        transform = _StackNormalizeFieldsd(
            mag_scale=1.0,
            mag_norm_mode="divisor",
            mask_threshold=0.5,
            include_hr_mag=True,
            raw_phase_input=False,
            random_time_frame=True,
        )
        mask = np.zeros((1, 2, 2, 2, 7), dtype=np.float32)
        mask[0, ..., 5] = 1.0
        data = {
            "lr_u": make_4d_channel([10, 11, 12, 13, 14, 15, 16]),
            "lr_v": make_4d_channel([20, 21, 22, 23, 24, 25, 26]),
            "lr_w": make_4d_channel([30, 31, 32, 33, 34, 35, 36]),
            "hr_u": make_4d_channel([100, 101, 102, 103, 104, 105, 106]),
            "hr_v": make_4d_channel([200, 201, 202, 203, 204, 205, 206]),
            "hr_w": make_4d_channel([300, 301, 302, 303, 304, 305, 306]),
            "lr_mag_u": make_4d_channel([1, 2, 3, 4, 5, 6, 7]),
            "lr_mag_v": make_4d_channel([11, 12, 13, 14, 15, 16, 17]),
            "lr_mag_w": make_4d_channel([21, 22, 23, 24, 25, 26, 27]),
            "hr_mag": make_4d_channel([1000, 1001, 1002, 1003, 1004, 1005, 1006]),
            "mask": mask,
            "venc": 1.0,
            "venc_u": 1.0,
            "venc_v": 1.0,
            "venc_w": 1.0,
            "time_start": 0,
            "time_end": 7,
            "time_index": 1,
            "lr_time_index": 3,
            "hr_time_index": 5,
        }

        for _ in range(3):
            sample = transform(data)
            self.assertAlmostEqual(float(sample["lr_vel"][0, 0, 0, 0]), 13.0)
            self.assertAlmostEqual(float(sample["hr_vel"][0, 0, 0, 0]), 105.0)
            self.assertAlmostEqual(float(sample["hr_vel"][1, 0, 0, 0]), 205.0)
            self.assertTrue(np.all(sample["mask"] == 1.0))

    def test_legacy_time_index_behavior_is_preserved(self):
        transform = _StackNormalizeFieldsd(
            mag_scale=1.0,
            mag_norm_mode="divisor",
            mask_threshold=0.5,
            include_hr_mag=False,
            raw_phase_input=False,
            random_time_frame=False,
        )
        mask = np.zeros((1, 2, 2, 2, 4), dtype=np.float32)
        mask[0, ..., 2] = 1.0
        data = {
            "lr_u": make_4d_channel([10, 11, 12, 13]),
            "lr_v": make_4d_channel([20, 21, 22, 23]),
            "lr_w": make_4d_channel([30, 31, 32, 33]),
            "hr_u": make_4d_channel([100, 101, 102, 103]),
            "hr_v": make_4d_channel([200, 201, 202, 203]),
            "hr_w": make_4d_channel([300, 301, 302, 303]),
            "lr_mag_u": make_4d_channel([1, 2, 3, 4]),
            "lr_mag_v": make_4d_channel([5, 6, 7, 8]),
            "lr_mag_w": make_4d_channel([9, 10, 11, 12]),
            "mask": mask,
            "venc": 1.0,
            "venc_u": 1.0,
            "venc_v": 1.0,
            "venc_w": 1.0,
            "time_start": 0,
            "time_end": 4,
            "time_index": 2,
        }

        sample = transform(data)
        self.assertAlmostEqual(float(sample["lr_vel"][0, 0, 0, 0]), 12.0)
        self.assertAlmostEqual(float(sample["hr_vel"][0, 0, 0, 0]), 102.0)
        self.assertTrue(np.all(sample["mask"] == 1.0))

    def test_trigger_time_map_restores_random_lr_sampling_with_mapped_hr_frame(self):
        transform = _StackNormalizeFieldsd(
            mag_scale=1.0,
            mag_norm_mode="divisor",
            mask_threshold=0.5,
            include_hr_mag=True,
            raw_phase_input=False,
            random_time_frame=True,
        )
        mask = np.ones((1, 2, 2, 2, 7), dtype=np.float32)
        data = {
            "lr_u": make_4d_channel([10, 11, 12, 13, 14, 15, 16]),
            "lr_v": make_4d_channel([20, 21, 22, 23, 24, 25, 26]),
            "lr_w": make_4d_channel([30, 31, 32, 33, 34, 35, 36]),
            "hr_u": make_4d_channel([100, 101, 102, 103, 104, 105, 106]),
            "hr_v": make_4d_channel([200, 201, 202, 203, 204, 205, 206]),
            "hr_w": make_4d_channel([300, 301, 302, 303, 304, 305, 306]),
            "lr_mag_u": make_4d_channel([1, 2, 3, 4, 5, 6, 7]),
            "lr_mag_v": make_4d_channel([11, 12, 13, 14, 15, 16, 17]),
            "lr_mag_w": make_4d_channel([21, 22, 23, 24, 25, 26, 27]),
            "hr_mag": make_4d_channel([1000, 1001, 1002, 1003, 1004, 1005, 1006]),
            "mask": mask,
            "venc": 1.0,
            "venc_u": 1.0,
            "venc_v": 1.0,
            "venc_w": 1.0,
            "time_start": 0,
            "time_end": 4,
            "hr_time_index_map": [0, 2, 3, 5],
        }

        seen = set()
        for _ in range(20):
            sample = transform(data)
            lr_value = int(round(float(sample["lr_vel"][0, 0, 0, 0])))
            hr_value = int(round(float(sample["hr_vel"][0, 0, 0, 0])))
            seen.add((lr_value, hr_value))

        self.assertTrue(seen)
        self.assertEqual(seen, {(10, 100), (11, 102), (12, 103), (13, 105)})

    def test_load_nifti_case_table_parses_pair_columns(self):
        csv_path = REPO_ROOT / "tests" / "_tmp_pair_columns.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "lr_u",
                        "lr_v",
                        "lr_w",
                        "lr_mag_u",
                        "lr_mag_v",
                        "lr_mag_w",
                        "hr_u",
                        "hr_v",
                        "hr_w",
                        "hr_mag",
                        "mask",
                        "venc",
                        "time_start",
                        "time_end",
                        "time_index",
                        "lr_time_index",
                        "hr_time_index",
                        "lr_trigger_time_ms",
                        "hr_trigger_time_ms",
                        "pairing_method",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "lr_u": "case/lr_u.nii.gz",
                        "lr_v": "case/lr_v.nii.gz",
                        "lr_w": "case/lr_w.nii.gz",
                        "lr_mag_u": "case/lr_mag.nii.gz",
                        "lr_mag_v": "case/lr_mag.nii.gz",
                        "lr_mag_w": "case/lr_mag.nii.gz",
                        "hr_u": "case/hr_u.nii.gz",
                        "hr_v": "case/hr_v.nii.gz",
                        "hr_w": "case/hr_w.nii.gz",
                        "hr_mag": "case/hr_mag.nii.gz",
                        "mask": "case/mask.nii.gz",
                        "venc": "0.9",
                        "time_start": "",
                        "time_end": "",
                        "time_index": "3",
                        "lr_time_index": "3",
                        "hr_time_index": "5",
                        "lr_trigger_time_ms": "255.992",
                        "hr_trigger_time_ms": "317.834",
                        "pairing_method": "trigger_time_nearest",
                    }
                )

            cases = load_nifti_case_table(str(csv_path), include_hr_mag=True)
            self.assertEqual(len(cases), 1)
            case = cases[0]
            self.assertEqual(case["lr_time_index"], 3)
            self.assertEqual(case["hr_time_index"], 5)
            self.assertAlmostEqual(case["lr_trigger_time_ms"], 255.992)
            self.assertAlmostEqual(case["hr_trigger_time_ms"], 317.834)
            self.assertEqual(case["pairing_method"], "trigger_time_nearest")
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_load_nifti_case_table_parses_case_mapping_columns(self):
        csv_path = REPO_ROOT / "tests" / "_tmp_case_map_columns.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "lr_u",
                        "lr_v",
                        "lr_w",
                        "lr_mag_u",
                        "lr_mag_v",
                        "lr_mag_w",
                        "hr_u",
                        "hr_v",
                        "hr_w",
                        "hr_mag",
                        "mask",
                        "venc",
                        "time_start",
                        "time_end",
                        "time_index",
                        "hr_time_index_map",
                        "lr_trigger_time_ms_map",
                        "hr_trigger_time_ms_map",
                        "pairing_method",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "lr_u": "case/lr_u.nii.gz",
                        "lr_v": "case/lr_v.nii.gz",
                        "lr_w": "case/lr_w.nii.gz",
                        "lr_mag_u": "case/lr_mag.nii.gz",
                        "lr_mag_v": "case/lr_mag.nii.gz",
                        "lr_mag_w": "case/lr_mag.nii.gz",
                        "hr_u": "case/hr_u.nii.gz",
                        "hr_v": "case/hr_v.nii.gz",
                        "hr_w": "case/hr_w.nii.gz",
                        "hr_mag": "case/hr_mag.nii.gz",
                        "mask": "case/mask.nii.gz",
                        "venc": "0.9",
                        "time_start": "0",
                        "time_end": "9",
                        "time_index": "",
                        "hr_time_index_map": "0;2;3;5;7;8;10;11;13",
                        "lr_trigger_time_ms_map": "36.570;109.711;182.852",
                        "hr_trigger_time_ms_map": "28.894;144.470;202.258",
                        "pairing_method": "trigger_time_nearest",
                    }
                )

            cases = load_nifti_case_table(str(csv_path), include_hr_mag=True)
            self.assertEqual(len(cases), 1)
            case = cases[0]
            self.assertEqual(case["hr_time_index_map"], [0, 2, 3, 5, 7, 8, 10, 11, 13])
            self.assertEqual(case["lr_trigger_time_ms_map"], [36.57, 109.711, 182.852])
            self.assertEqual(case["hr_trigger_time_ms_map"], [28.894, 144.47, 202.258])
            self.assertEqual(case["pairing_method"], "trigger_time_nearest")
        finally:
            if csv_path.exists():
                csv_path.unlink()


class TriggerTimePairingHelperTests(unittest.TestCase):
    def test_case_001_nearest_mapping_matches_expected(self):
        lr = [36.570350646973, 109.71105194092, 182.85174560547, 255.9924621582, 329.13314819336, 402.27383422852, 475.41455078125, 548.55529785156, 621.69598388672]
        hr = [28.893999099731, 86.681999206543, 144.4700012207, 202.25799560547, 260.04598999023, 317.83401489258, 375.62200927734, 433.41000366211, 491.19799804688, 548.98596191406, 606.77398681641, 664.56195068359, 722.34997558594, 780.13793945313]
        mapping = generate_trigger_time_frame_pairs.build_nearest_frame_mapping(lr, hr)
        self.assertEqual(mapping, [0, 2, 3, 5, 7, 8, 10, 11, 13])

    def test_case_002_nearest_mapping_is_identity(self):
        lr = [38.982, 116.945, 194.908, 272.871, 350.834, 428.798, 506.761, 584.724, 662.687, 740.650]
        hr = [37.287, 111.861, 186.436, 261.010, 335.584, 410.159, 484.733, 559.307, 633.882, 708.456]
        mapping = generate_trigger_time_frame_pairs.build_nearest_frame_mapping(lr, hr)
        self.assertEqual(mapping, list(range(10)))

    def test_case_map_row_resets_time_range_and_serializes_mapping(self):
        row = {
            "lr_u": "data/paired_dataset/lr_3t/001_20240313/Vx.nii.gz",
            "hr_u": "data/paired_dataset/hr_7t_in_3t_masked/001_20240313/Vx.nii.gz",
            "time_start": "0",
            "time_end": "8",
            "time_index": "",
        }
        lr = [36.570350646973, 109.71105194092, 182.85174560547]
        hr = [28.893999099731, 86.681999206543, 144.4700012207, 202.25799560547]
        mapped = generate_trigger_time_frame_pairs.build_case_map_row(row, lr, hr)
        self.assertEqual(mapped["time_start"], "0")
        self.assertEqual(mapped["time_end"], "3")
        self.assertEqual(mapped["time_index"], "")
        self.assertEqual(mapped["hr_time_index_map"], "0;2;3")
        self.assertEqual(mapped["pairing_method"], "trigger_time_nearest")


class ExporterTimeRangeTests(unittest.TestCase):
    def test_resolve_time_end_uses_exclusive_semantics(self):
        lr_path = REPO_ROOT / "data" / "paired_dataset" / "lr_3t" / "001_20240313" / "Vx.nii.gz"
        hr_path = REPO_ROOT / "data" / "paired_dataset" / "hr_7t_in_3t" / "001_20240313" / "Vx.nii.gz"
        self.assertEqual(export_paired_lr_hr_dataset.resolve_time_end(str(lr_path), str(hr_path)), 9)


if __name__ == "__main__":
    unittest.main()
