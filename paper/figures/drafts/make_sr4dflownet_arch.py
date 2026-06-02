"""Render the SR4DFlowNet architecture diagram (F3) using matplotlib.
Dual-pathway residual CNN: magnitude/speed and phase encoders fused via
1x1 conv, 8 LR ResBlocks, trilinear x2 upsample, 4 HR ResBlocks, three
1x1 conv heads for (u_hat, v_hat, w_hat) plus optional magnitude head."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PHASE_COLOR = "#4F7AA8"
MAG_COLOR = "#5C9E7E"
FUSE_COLOR = "#7E5BA6"
LR_COLOR = "#D08A4E"
UP_COLOR = "#C44569"
HR_COLOR = "#D08A4E"
HEAD_COLOR = "#3B4D61"


def box(ax, x, y, w, h, text, face, sub=None, fontsize=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face,
            edgecolor="#222",
            linewidth=0.9,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.08 if sub else 0),
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        weight="bold",
    )
    if sub:
        ax.text(
            x + w / 2,
            y + 0.13,
            sub,
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            alpha=0.9,
        )


def arrow(ax, x0, y0, x1, y1, color="#444"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=1.2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    # Inputs
    box(ax, 0.3, 5.0, 1.6, 1.0, "Phase\n(u,v,w)", PHASE_COLOR, sub="3×16³")
    box(ax, 0.3, 2.6, 1.6, 1.0, "Magnitude\n+ speed", MAG_COLOR, sub="3×16³")

    # Encoder ResBlocks (3 per pathway)
    for i in range(3):
        x = 2.4 + i * 1.05
        box(ax, x, 5.0, 0.95, 1.0, f"ResBlock\n64ch", PHASE_COLOR, fontsize=8)
        box(ax, x, 2.6, 0.95, 1.0, f"ResBlock\n64ch", MAG_COLOR, fontsize=8)

    # arrows from inputs through encoder
    for i in range(4):
        x_from = 1.9 + i * 1.05
        x_to = x_from + 0.45
        arrow(ax, x_from, 5.5, x_to, 5.5)
        arrow(ax, x_from, 3.1, x_to, 3.1)

    # Fusion 1x1
    box(ax, 5.9, 3.8, 1.4, 1.4, "1×1 Conv\nfusion", FUSE_COLOR, sub="128→64")
    arrow(ax, 5.7, 5.5, 6.6, 5.2)
    arrow(ax, 5.7, 3.1, 6.6, 3.8)

    # LR ResBlocks (8) shown as a stack
    box(ax, 7.7, 3.8, 2.5, 1.4, "8 × LR ResBlock\n(64 ch, LeakyReLU)", LR_COLOR, sub="16³×64")
    arrow(ax, 7.3, 4.5, 7.7, 4.5)

    # Upsample
    box(ax, 10.5, 3.8, 1.6, 1.4, "Trilinear\nupsample ×2", UP_COLOR, sub="32³×64")
    arrow(ax, 10.2, 4.5, 10.5, 4.5)

    # HR ResBlocks (4)
    box(ax, 12.4, 3.8, 2.3, 1.4, "4 × HR ResBlock\n(64 ch)", HR_COLOR, sub="32³×64")
    arrow(ax, 12.1, 4.5, 12.4, 4.5)

    # Heads
    head_x = 15.2
    box(ax, head_x, 5.4, 1.6, 0.8, "1×1 Conv  →  û", HEAD_COLOR, fontsize=9)
    box(ax, head_x, 4.3, 1.6, 0.8, "1×1 Conv  →  v̂", HEAD_COLOR, fontsize=9)
    box(ax, head_x, 3.2, 1.6, 0.8, "1×1 Conv  →  ŵ", HEAD_COLOR, fontsize=9)
    box(ax, head_x, 2.0, 1.6, 0.8, "1×1 Conv  →  M̂\n(optional)", HEAD_COLOR, fontsize=8)
    for yh in (5.8, 4.7, 3.6, 2.4):
        arrow(ax, 14.7, 4.5, head_x, yh)

    # Title and footer
    ax.text(9, 6.6, "SR4DFlowNet architecture", ha="center", va="center",
            fontsize=14, weight="bold", color="#222")
    ax.text(
        9,
        1.2,
        "Composite masked loss: L = MSE_fluid + 0.3 · MSE_non-fluid (+ λ_m · MSE_mag)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#333",
    )
    ax.text(
        9,
        0.6,
        "Inputs: 16³ patches at LR  ·  Output: 32³ patches at 2× resolution",
        ha="center",
        va="center",
        fontsize=9,
        color="#555",
    )

    out = Path(__file__).parent / "fig_sr4dflownet_arch.png"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
