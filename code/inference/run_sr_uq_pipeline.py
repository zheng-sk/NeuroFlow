import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SR full-volume inference + uncertainty quantification report in one command."
    )
    parser.add_argument("--case-csv", required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--time-axis", type=int, default=-1)
    parser.add_argument("--frame-index", type=int, nargs="*", default=None)
    parser.add_argument("--use-csv-frame-selection", action="store_true")

    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--res-increase", type=int, default=2)
    parser.add_argument("--low-resblock", type=int, default=8)
    parser.add_argument("--hi-resblock", type=int, default=4)
    parser.add_argument("--predict-mag", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--raw-phase-input", dest="raw_phase_input", action="store_true", default=True)
    parser.add_argument("--already-velocity-input", dest="raw_phase_input", action="store_false")
    parser.add_argument("--legacy-invert-uv-sign-on-raw", action="store_true")
    parser.add_argument("--raw-center", type=float, default=2048.0)
    parser.add_argument("--raw-scale", type=float, default=2048.0)
    parser.add_argument("--mag-scale", type=float, default=4095.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)

    parser.add_argument("--flow-axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--selected-frame", type=int, default=0)
    parser.add_argument("--max-display-slices", type=int, default=12)
    parser.add_argument("--mask-min-slice-voxels", type=int, default=25)
    parser.add_argument("--q-ref", type=float, default=float("nan"))
    parser.add_argument("--cca-range", type=str, default="")
    parser.add_argument("--mu-pa-s", type=float, default=0.0035)
    parser.add_argument("--max-wall-points", type=int, default=30000)
    parser.add_argument("--report-title", default="4D Flow SR Uncertainty Quantification Report")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    infer_py = str((Path(__file__).resolve().parent / "run_sr_inference_case.py"))
    report_py = str((Path(__file__).resolve().parent / "generate_sr_uq_report.py"))

    infer_cmd = [
        sys.executable,
        infer_py,
        "--case-csv",
        args.case_csv,
        "--case-index",
        str(args.case_index),
        "--model-path",
        args.model_path,
        "--out-dir",
        str(out_dir),
        "--patch-size",
        str(args.patch_size),
        "--sw-batch-size",
        str(args.sw_batch_size),
        "--overlap",
        str(args.overlap),
        "--res-increase",
        str(args.res_increase),
        "--low-resblock",
        str(args.low_resblock),
        "--hi-resblock",
        str(args.hi_resblock),
        "--time-axis",
        str(args.time_axis),
        "--raw-center",
        str(args.raw_center),
        "--raw-scale",
        str(args.raw_scale),
        "--mag-scale",
        str(args.mag_scale),
        "--mask-threshold",
        str(args.mask_threshold),
    ]
    if args.frame_index:
        infer_cmd.append("--frame-index")
        infer_cmd.extend([str(x) for x in args.frame_index])
    if args.use_csv_frame_selection:
        infer_cmd.append("--use-csv-frame-selection")
    if args.predict_mag is not None:
        infer_cmd.append("--predict-mag" if args.predict_mag else "--no-predict-mag")
    if args.raw_phase_input:
        infer_cmd.append("--raw-phase-input")
    else:
        infer_cmd.append("--already-velocity-input")
    if args.legacy_invert_uv_sign_on_raw:
        infer_cmd.append("--legacy-invert-uv-sign-on-raw")

    _run(infer_cmd)

    payload_npz = out_dir / "analysis_payload.npz"
    meta_json = out_dir / "inference_metadata.json"

    report_cmd = [
        sys.executable,
        report_py,
        "--payload-npz",
        str(payload_npz),
        "--metadata-json",
        str(meta_json),
        "--out-dir",
        str(out_dir),
        "--flow-axis",
        str(args.flow_axis),
        "--selected-frame",
        str(args.selected_frame),
        "--max-display-slices",
        str(args.max_display_slices),
        "--mask-min-slice-voxels",
        str(args.mask_min_slice_voxels),
        "--mu-pa-s",
        str(args.mu_pa_s),
        "--max-wall-points",
        str(args.max_wall_points),
        "--report-title",
        args.report_title,
    ]
    if args.cca_range:
        report_cmd.extend(["--cca-range", args.cca_range])
    if str(args.q_ref) != "nan":
        report_cmd.extend(["--q-ref", str(args.q_ref)])

    _run(report_cmd)

    print("\nPipeline finished.")
    print(f"Report: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
