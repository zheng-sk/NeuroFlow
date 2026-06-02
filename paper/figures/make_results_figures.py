"""
Distill-style results figures for the NeuroFlow paper.
Generates:
  fig_results_velocity_geometry.png  — 2-panel: MSE_vel + geometry (Dice/sMSD/HD)
  fig_results_background.png         — background MSE comparison (excl. 4DFlowNet)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── palette (distill / paper style) ─────────────────────────────────────────
C_ORIG   = "#4F7AA8"   # Original baseline — blue
C_DNS    = "#5C9E7E"   # CBAM-DNS — green
C_SRES   = "#7E5BA6"   # CBAM-SupRes — purple
C_4DFLOW = "#B85C6B"   # 4DFlowNet — coral
C_MASKED = "#D08A4E"   # masked-input tint — gold

BG       = "white"
GRID     = "#EEEEEE"
TICK_COL = "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#AAAAAA",
    "axes.linewidth": 0.8,
    "xtick.color": TICK_COL,
    "ytick.color": TICK_COL,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "legend.frameon": False,
})

VENC = 0.9  # m/s (for axis label only)

# ── data ────────────────────────────────────────────────────────────────────
# (mean, sd) pairs

dns_vel = {
    "A (Original / Synth)":   (35.9, 11.5),
    "B (CBAM-DNS / Synth)":   (35.7, 12.6),
    "E (CBAM-DNS / Masked)":  (34.5, 11.6),
}
sr_vel = {
    "C (Original / Synth)":     (41.5, 15.1),
    "D (CBAM-SupRes / Synth)":  (40.1, 14.3),
    "F (CBAM-SupRes / Masked)": (40.8, 13.9),
    "G (4DFlowNet / Orig)":     (62.1, 21.4),
}

dns_geo = {
    "A": dict(dice=(0.826,0.039), smsd=(0.376,0.088), hd=(14.67,3.75)),
    "B": dict(dice=(0.831,0.032), smsd=(0.359,0.074), hd=(13.67,5.28)),
    "E": dict(dice=(0.839,0.022), smsd=(0.300,0.037), hd=(14.12,4.85)),
}
sr_geo = {
    "C": dict(dice=(0.778,0.037), smsd=(0.434,0.063), hd=(15.28,3.73)),
    "D": dict(dice=(0.781,0.041), smsd=(0.417,0.077), hd=(14.53,3.92)),
    "F": dict(dice=(0.786,0.033), smsd=(0.370,0.049), hd=(14.01,4.95)),
}

dns_bg = {
    "A (Original / Synth)":  (0.622, 0.191),
    "B (CBAM-DNS / Synth)":  (0.672, 0.248),
    "E (CBAM-DNS / Masked)": (0.338, 0.099),
}
sr_bg = {
    "C (Original / Synth)":     (0.916, 0.664),
    "D (CBAM-SupRes / Synth)":  (0.588, 0.152),
    "F (CBAM-SupRes / Masked)": (0.346, 0.113),
}

# colour per config
_col = {
    "A": C_ORIG,   "B": C_DNS,   "E": C_DNS,
    "C": C_ORIG,   "D": C_SRES,  "F": C_SRES,
    "G": C_4DFLOW,
}
# hatch for masked
_hatch = {"E": "///", "F": "///"}


# ── helper ──────────────────────────────────────────────────────────────────
def bar_groups(ax, groups, values, errors, colors, hatches=None,
               ylabel="", title="", show_legend=False):
    """
    groups : list of str labels
    values / errors : parallel lists
    """
    x = np.arange(len(groups))
    w = 0.55
    bars = ax.bar(x, values, width=w, color=colors,
                  edgecolor="white", linewidth=0.5,
                  yerr=errors, capsize=3,
                  error_kw=dict(elinewidth=0.8, ecolor="#888888", capthick=0.8))
    if hatches:
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — velocity MSE + geometry (Dice, sMSD, HD)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5),
                         gridspec_kw=dict(hspace=0.55, wspace=0.38))

# helper for geo panel
def geo_bar(ax, geo_dict, metric, ylabel, title, ylim=None):
    keys  = list(geo_dict.keys())
    means = [geo_dict[k][metric][0] for k in keys]
    sds   = [geo_dict[k][metric][1] for k in keys]
    cols  = [_col[k] for k in keys]
    htch  = [_hatch.get(k, "") for k in keys]
    bar_groups(ax, keys, means, sds, cols, htch, ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(ylim)

# ── row 0: Velocity MSE ─────────────────────────────────────────────────────
# DNS
ax = axes[0, 0]
keys  = list(dns_vel.keys())
ids   = [k.split(" ")[0] for k in keys]
means = [dns_vel[k][0] for k in keys]
sds   = [dns_vel[k][1] for k in keys]
cols  = [_col[i] for i in ids]
htch  = [_hatch.get(i,"") for i in ids]
bar_groups(ax, ids, means, sds, cols, htch,
           ylabel=r"MSE$_\mathrm{vel}$ ($\times10^{-3}$)",
           title="Velocity MSE — Denoising (×1)")
ax.set_ylim(0, 60)

# SR (excl 4DFlowNet for scale — plotted with note)
ax = axes[0, 1]
sr_keys_main = ["C (Original / Synth)", "D (CBAM-SupRes / Synth)", "F (CBAM-SupRes / Masked)"]
ids_sr = ["C", "D", "F"]
means_sr = [sr_vel[k][0] for k in sr_keys_main]
sds_sr   = [sr_vel[k][1] for k in sr_keys_main]
cols_sr  = [_col[i] for i in ids_sr]
htch_sr  = [_hatch.get(i,"") for i in ids_sr]
bar_groups(ax, ids_sr, means_sr, sds_sr, cols_sr, htch_sr,
           ylabel="", title="Velocity MSE — SR (×2)\n(excl. 4DFlowNet)")
ax.set_ylim(0, 60)
# reference line for 4DFlowNet
ax.axhline(62.1, color=C_4DFLOW, lw=1.2, ls="--", label="G 4DFlowNet: 62.1")
ax.legend(fontsize=6.5, loc="upper right")

# SR incl 4DFlowNet full scale
ax = axes[0, 2]
keys_all  = ["C","D","F","G"]
labels_sr = ["C","D","F","G"]
means_all = [sr_vel[k+" (Original / Synth)" if k=="C" else
               k+" (CBAM-SupRes / Synth)" if k=="D" else
               k+" (CBAM-SupRes / Masked)" if k=="F" else
               k+" (4DFlowNet / Orig)"][0]
             for k in keys_all]
# simplify lookup
_sv = {"C": sr_vel["C (Original / Synth)"],
       "D": sr_vel["D (CBAM-SupRes / Synth)"],
       "F": sr_vel["F (CBAM-SupRes / Masked)"],
       "G": sr_vel["G (4DFlowNet / Orig)"]}
means_all = [_sv[k][0] for k in keys_all]
sds_all   = [_sv[k][1] for k in keys_all]
cols_all  = [_col[k] for k in keys_all]
htch_all  = [_hatch.get(k,"") for k in keys_all]
bar_groups(ax, labels_sr, means_all, sds_all, cols_all, htch_all,
           ylabel="", title="Velocity MSE — SR (×2)\n(incl. 4DFlowNet)")

# ── row 1: geometry ──────────────────────────────────────────────────────────
# geometry panels: combine DNS + SR configs
axes[1,0].cla()
all_geo = {**{"A": dns_geo["A"], "B": dns_geo["B"], "E": dns_geo["E"]},
           **{"C": sr_geo["C"],  "D": sr_geo["D"],  "F": sr_geo["F"]}}
keys_geo = list(all_geo.keys())

geo_bar(axes[1,0], all_geo, "dice",  "Dice (↑)",       "Dice — All configs", ylim=(0.6,1.0))
geo_bar(axes[1,1], all_geo, "smsd",  "sMSD (mm, ↓)",   "sMSD — All configs")
geo_bar(axes[1,2], all_geo, "hd",    "HD (mm, ↓)",     "HD — All configs")

# ── vertical group separators ────────────────────────────────────────────────
for ax in axes[1,:]:
    ax.axvline(2.5, color="#BBBBBB", lw=1.0, ls=":")

# ── shared legend ────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=C_ORIG,   label="Original"),
    mpatches.Patch(color=C_DNS,    label="CBAM-DNS"),
    mpatches.Patch(color=C_SRES,   label="CBAM-SupRes"),
    mpatches.Patch(color=C_4DFLOW, label="4DFlowNet"),
    mpatches.Patch(facecolor="white", edgecolor="grey",
                   hatch="///",   label="Masked input"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))

# DNS / SR group labels
for ax in axes[1,:]:
    ax.text(1.0, ax.get_ylim()[1]*0.97, "Denoising", ha="center",
            fontsize=6.5, color="#555555", style="italic")
    ax.text(4.0, ax.get_ylim()[1]*0.97, "SR", ha="center",
            fontsize=6.5, color="#555555", style="italic")

fig.subplots_adjust(bottom=0.14)
out1 = "fig_results_velocity_geometry.png"
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — background MSE (excl. 4DFlowNet for scale)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.5),
                           gridspec_kw=dict(wspace=0.4))

for ax, title, bg_dict in zip(
        axes2,
        ["Background MSE — Denoising (×1)",
         "Background MSE — SR (×2)"],
        [dns_bg, sr_bg]):

    keys  = list(bg_dict.keys())
    ids   = [k.split(" ")[0] for k in keys]
    means = [bg_dict[k][0] for k in keys]
    sds   = [bg_dict[k][1] for k in keys]
    cols  = [_col[i] for i in ids]
    htch  = [_hatch.get(i,"") for i in ids]
    bar_groups(ax, ids, means, sds, cols, htch,
               ylabel=r"MSE$_\mathrm{bg}$ ($\times10^{-3}$)",
               title=title)

# 4DFlowNet note
axes2[1].text(0.98, 0.96,
              "4DFlowNet (G): 65.5 ± 24.4",
              transform=axes2[1].transAxes, ha="right", va="top",
              fontsize=6.5, color=C_4DFLOW,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_4DFLOW, lw=0.8))

fig2.legend(handles=legend_patches[:4], loc="lower center", ncol=4,
            fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig2.subplots_adjust(bottom=0.22)
out2 = "fig_results_background.png"
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out2}")
