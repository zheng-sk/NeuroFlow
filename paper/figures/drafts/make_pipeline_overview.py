"""Render the end-to-end pipeline overview figure (F2) as a clean
matplotlib box-and-arrow schematic. No external dependencies beyond
matplotlib. Output: fig_pipeline_overview.png at 300 dpi."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

LANES = [
    {
        "title": "1. Data acquisition",
        "blocks": [
            ("DICOM→NIfTI", "dicom_to_nifti.py"),
            ("3T LR (1.0 mm)", ""),
            ("7T HR (0.5 mm)", ""),
        ],
        "color": "#4F7AA8",
    },
    {
        "title": "2. Preprocessing",
        "blocks": [
            ("Temporal\nregistration", "batch_register_magnitude.py"),
            ("YOLO CoW\ncrop", "yolo_crop_patient_pairs.py"),
            ("7T→3T\nregistration", "batch_register_7T_to_3T.py"),
            ("nnU-Net\nResEnc-M seg.", "segment_cow_crops.py"),
        ],
        "color": "#5C9E7E",
    },
    {
        "title": "3. Super-resolution",
        "blocks": [
            ("SR4DFlowNet\n(LR u,v,w + mag)", "src/Network/SR4DFlowNet.py"),
            ("LOO CV\n(7 folds)", "run_loo_xval.sh"),
        ],
        "color": "#D08A4E",
    },
    {
        "title": "4. Inference",
        "blocks": [
            ("MONAI sliding\nwindow (overlap 0.25)", "predict_nifti.py"),
            ("HR velocity\n(u,v,w) NIfTI", ""),
        ],
        "color": "#B85C6B",
    },
    {
        "title": "5. Analysis & evaluation",
        "blocks": [
            ("Aneurysm ROI\n& neck", "select_aneurysm_roi.py"),
            ("Shape\nmetrics", "calculate_aneurysm_shape_metrics.py"),
            ("SR-UQ\nreport", "generate_sr_uq_report.py"),
        ],
        "color": "#7E5BA6",
    },
]

LANE_HEIGHT = 1.8
LANE_PAD = 0.45
BLOCK_HEIGHT = 1.1
BLOCK_PAD_X = 0.35
LEFT_LABEL_W = 2.9
FIG_W = 18
FIG_H = LANE_HEIGHT * len(LANES) + 1.0


def draw_block(ax, x, y, w, h, text, sub, face):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.0,
            edgecolor="#222222",
            facecolor=face,
            alpha=0.95,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.12 if sub else 0.0),
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        color="white",
        weight="bold",
    )
    if sub:
        ax.text(
            x + w / 2,
            y + 0.16,
            sub,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            family="monospace",
            alpha=0.9,
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_axis_off()

    total_w = FIG_W - LEFT_LABEL_W - 0.6
    lane_centers_y = []
    for idx, lane in enumerate(reversed(LANES)):
        y0 = 0.5 + idx * LANE_HEIGHT
        lane_centers_y.append(y0 + BLOCK_HEIGHT / 2)
        ax.text(
            0.3,
            y0 + BLOCK_HEIGHT / 2,
            lane["title"],
            ha="left",
            va="center",
            fontsize=12,
            weight="bold",
            color="#222",
        )
        n = len(lane["blocks"])
        block_w = (total_w - (n + 1) * BLOCK_PAD_X) / n
        for j, (text, sub) in enumerate(lane["blocks"]):
            bx = LEFT_LABEL_W + BLOCK_PAD_X + j * (block_w + BLOCK_PAD_X)
            draw_block(ax, bx, y0, block_w, BLOCK_HEIGHT, text, sub, lane["color"])

    lane_centers_y = lane_centers_y[::-1]
    arrow_x = LEFT_LABEL_W + total_w / 2
    for i in range(len(lane_centers_y) - 1):
        y_top = lane_centers_y[i] - BLOCK_HEIGHT / 2
        y_bot = lane_centers_y[i + 1] + BLOCK_HEIGHT / 2
        ax.add_patch(
            FancyArrowPatch(
                (arrow_x, y_top),
                (arrow_x, y_bot),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.4,
                color="#444",
            )
        )

    ax.text(
        FIG_W / 2,
        FIG_H - 0.35,
        "NeuroFlow end-to-end pipeline",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color="#222",
    )

    out = Path(__file__).parent / "fig_pipeline_overview.png"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
