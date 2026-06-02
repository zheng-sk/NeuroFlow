"""
Presentation GIF — NeuroFlow CoW reconstruction comparison.

Four panels (z-scroll, ping-pong):
  3T Input  |  DNS Prediction (×1)  |  SR Prediction (×2)  |  7T Reference (masked)

All data from analysis_payload.npz (pre-normalized, channels 0-2 = u,v,w).
Speed = sqrt(u²+v²+w²) * VENC  →  m/s, displayed with a shared colour scale.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

BASE   = Path(__file__).parent.parent.parent
OUT    = Path(__file__).parent / "fig_results_gif.gif"
P      = "001_20240313"
VENC   = 0.9  # m/s
FOLD   = "fold_000"

r1_npz = BASE / f"output/inference_only_masked/r1_noise_{FOLD}_{P}_case000/analysis_payload.npz"
x2_npz = BASE / f"output/inference_only_masked/x2_noise_{FOLD}_{P}_case000/analysis_payload.npz"

# ── load & compute speed (temporal mean) ─────────────────────────────────────
def speed_from_npz(npz_path, key):
    d = np.load(npz_path, allow_pickle=True)
    arr = d[key]                          # (frames, channels, x, y, z)
    u, v, w = arr[:, 0], arr[:, 1], arr[:, 2]
    speed = np.sqrt(u**2 + v**2 + w**2) * VENC   # m/s, (frames, x, y, z)
    return speed.mean(0)                   # (x, y, z)

def masked_speed(npz_path, key):
    d = np.load(npz_path, allow_pickle=True)
    arr  = d[key]
    mask = d["mask"]                      # (frames, x, y, z)
    u, v, w = arr[:, 0], arr[:, 1], arr[:, 2]
    speed = np.sqrt(u**2 + v**2 + w**2) * VENC
    return (speed * mask).mean(0)

print("Loading …")
s_3t  = speed_from_npz(r1_npz, "lr_norm")    # (147, 99, 48) native res
s_dns = speed_from_npz(r1_npz, "pred_norm")  # (147, 99, 48)
s_sr  = speed_from_npz(x2_npz, "pred_norm")  # (148, 100, 48) ← crop to ref shape
s_gt  = masked_speed(r1_npz,   "gt_norm")    # (147, 99, 48) masked 7T

# Crop SR to same XY footprint as reference
nx, ny, nz = s_3t.shape
s_sr = s_sr[:nx, :ny, :]

print(f"  3T  speed  p99={np.percentile(s_3t, 99):.3f} m/s  max={s_3t.max():.3f}")
print(f"  DNS speed  p99={np.percentile(s_dns,99):.3f} m/s  max={s_dns.max():.3f}")
print(f"  SR  speed  p99={np.percentile(s_sr, 99):.3f} m/s  max={s_sr.max():.3f}")
print(f"  GT  speed  p99={np.percentile(s_gt[s_gt>0], 99):.3f} m/s  max={s_gt.max():.3f}")

# Common display scale: 99th pct of intraluminal GT signal
vmax = np.percentile(s_gt[s_gt > 0.001], 99)
print(f"  Shared vmax: {vmax:.4f} m/s")

def norm(x):
    return np.clip(x / vmax, 0, 1)

vols = [norm(s_3t), norm(s_dns), norm(s_sr), norm(s_gt)]

# ── find z-range with vessels ─────────────────────────────────────────────────
gt_raw = np.load(r1_npz, allow_pickle=True)
mask_mean = gt_raw["mask"].mean(0)   # (x, y, z)
vessel_per_z = (mask_mean > 0.1).sum(axis=(0, 1))
valid_z = np.where(vessel_per_z > 15)[0]
z0, z1 = int(valid_z.min()), int(valid_z.max())
print(f"  Vessel z-range: {z0}–{z1}  ({z1-z0+1} slices)")

# ── colour palette ────────────────────────────────────────────────────────────
titles  = ["3T Input", "DNS Prediction (×1)", "SR Prediction (×2)", "7T Reference (masked)"]
colors  = ["#4F7AA8",  "#5C9E7E",              "#7E5BA6",             "#D08A4E"]

# ── build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 4, figsize=(13.5, 3.6),
    facecolor="#0D0D0D",
    gridspec_kw=dict(wspace=0.04, left=0.01, right=0.98,
                     top=0.87, bottom=0.09))

ims = []
for ax, title, color, vol in zip(axes, titles, colors, vols):
    ax.set_facecolor("#0D0D0D")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    ax.set_title(title, color=color, fontsize=9.5, fontweight="bold", pad=5)
    im = ax.imshow(vol[:, :, z0].T,
                   cmap="hot", vmin=0, vmax=1,
                   aspect="auto", origin="lower", interpolation="bilinear")
    ims.append(im)

# Shared colour bar
sm = ScalarMappable(cmap="hot", norm=Normalize(0, vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                    fraction=0.025, pad=0.04, aspect=50)
cbar.set_label("Speed (m/s)", color="#AAAAAA", fontsize=8)
cbar.ax.tick_params(colors="#AAAAAA", labelsize=7)

# Header label
fig.text(0.5, 0.97,
         "NeuroFlow  ·  Circle of Willis 4D Flow reconstruction  ·  Patient 001",
         ha="center", color="#CCCCCC", fontsize=9, style="italic")

slice_text = fig.text(0.5, 0.01, "", ha="center",
                      color="#888888", fontsize=7.5)

# ── animation ────────────────────────────────────────────────────────────────
frames_fwd = list(range(z0, z1 + 1))
frames     = frames_fwd + frames_fwd[-2:0:-1]   # ping-pong, no duplicate endpoints

def update(z):
    for im, vol in zip(ims, vols):
        im.set_data(vol[:, :, z].T)
    slice_text.set_text(f"axial  z = {z + 1} / {nz}")
    return ims + [slice_text]

anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=True)

print(f"Writing {OUT}  ({len(frames)} frames) …")
writer = PillowWriter(fps=8)
anim.save(str(OUT), writer=writer, dpi=100)
plt.close(fig)
print(f"Done  →  {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
