import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def _require_if(value: str, *, cond: bool, name: str) -> None:
    if cond and not str(value).strip():
        raise ValueError(f"Missing required argument --{name} for this execution mode.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run denoising + superresolution UQ workflows and then generate a unified 3-way comparison report "
            "(3T vs 7T, 3T Denoised vs 7T, 3T Superresolution vs 7T)."
        )
    )
    parser.add_argument("--case-csv", default="", help="Case CSV used by run_sr_uq_pipeline.py")
    parser.add_argument("--case-index", type=int, default=0, help="Case index in CSV")

    parser.add_argument("--dns-model-path", default="", help="Model path for denoising workflow")
    parser.add_argument("--sr-model-path", default="", help="Model path for superresolution workflow")

    parser.add_argument("--dns-out-dir", required=True, help="Output directory for denoising workflow report")
    parser.add_argument("--sr-out-dir", required=True, help="Output directory for superresolution workflow report")
    parser.add_argument("--threeway-out-dir", required=True, help="Output directory for 3-way comparison report")

    parser.add_argument("--dns-res-increase", type=int, default=1, help="res_increase for denoising workflow (default: 1)")
    parser.add_argument("--sr-res-increase", type=int, default=2, help="res_increase for superresolution workflow (default: 2)")
    parser.add_argument(
        "--use-dns-baseline-for-sr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use denoising payload as fixed baseline 3T source in SR report (recommended for fair comparisons).",
    )

    parser.add_argument("--skip-dns", action="store_true", help="Skip running denoising workflow (reuse existing dns-out-dir).")
    parser.add_argument("--skip-sr", action="store_true", help="Skip running superresolution workflow (reuse existing sr-out-dir).")
    parser.add_argument("--skip-threeway", action="store_true", help="Skip final 3-way report generation.")

    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used to launch sub-scripts.")
    parser.add_argument("--threeway-baseline-label", default="3T")
    parser.add_argument("--threeway-denoised-label", default="3T Denoised")
    parser.add_argument("--threeway-superres-label", default="3T Superresolution")

    # Unknown args are forwarded to each run_sr_uq_pipeline.py call.
    args, forward_args = parser.parse_known_args()

    need_workflow_inputs = (not args.skip_dns) or (not args.skip_sr)
    _require_if(args.case_csv, cond=need_workflow_inputs, name="case-csv")
    _require_if(args.dns_model_path, cond=(not args.skip_dns), name="dns-model-path")
    _require_if(args.sr_model_path, cond=(not args.skip_sr), name="sr-model-path")

    script_dir = Path(__file__).resolve().parent
    run_single = script_dir / "run_sr_uq_pipeline.py"
    run_compare = script_dir / "generate_threeway_uq_comparison.py"

    dns_out = Path(args.dns_out_dir).resolve()
    sr_out = Path(args.sr_out_dir).resolve()
    threeway_out = Path(args.threeway_out_dir).resolve()

    if not args.skip_dns:
        cmd_dns = [
            str(args.python_exe),
            str(run_single),
            *forward_args,
            "--case-csv",
            str(args.case_csv),
            "--case-index",
            str(args.case_index),
            "--model-path",
            str(args.dns_model_path),
            "--out-dir",
            str(dns_out),
            "--res-increase",
            str(args.dns_res_increase),
            "--task-mode",
            "denoising",
        ]
        _run(cmd_dns)

    if not args.skip_sr:
        cmd_sr = [
            str(args.python_exe),
            str(run_single),
            *forward_args,
            "--case-csv",
            str(args.case_csv),
            "--case-index",
            str(args.case_index),
            "--model-path",
            str(args.sr_model_path),
            "--out-dir",
            str(sr_out),
            "--res-increase",
            str(args.sr_res_increase),
            "--task-mode",
            "superresolution",
        ]
        if bool(args.use_dns_baseline_for_sr):
            dns_payload = dns_out / "analysis_payload.npz"
            if not dns_payload.exists():
                raise FileNotFoundError(
                    f"Expected DNS payload for fixed baseline at {dns_payload}. "
                    "Run DNS first or pass --no-use-dns-baseline-for-sr."
                )
            cmd_sr.extend(["--baseline-payload-npz", str(dns_payload)])
        _run(cmd_sr)

    if not args.skip_threeway:
        if not dns_out.exists():
            raise FileNotFoundError(f"DNS output directory does not exist: {dns_out}")
        if not sr_out.exists():
            raise FileNotFoundError(f"SR output directory does not exist: {sr_out}")
        cmd_threeway = [
            str(args.python_exe),
            str(run_compare),
            "--dns-report-dir",
            str(dns_out),
            "--sr-report-dir",
            str(sr_out),
            "--out-dir",
            str(threeway_out),
            "--baseline-label",
            str(args.threeway_baseline_label),
            "--denoised-label",
            str(args.threeway_denoised_label),
            "--superres-label",
            str(args.threeway_superres_label),
        ]
        _run(cmd_threeway)

    print("\nThree-way pipeline finished.")
    print(f"DNS report dir: {dns_out}")
    print(f"SR report dir: {sr_out}")
    print(f"Three-way report: {threeway_out / 'report_three_way.html'}")


if __name__ == "__main__":
    main()

