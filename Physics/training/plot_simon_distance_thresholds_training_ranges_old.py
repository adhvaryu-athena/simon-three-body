import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# =============================================================================
# SIMON distance thresholds
# =============================================================================

EPS = 3e-4
R_SOFT_MIN = 5e-4
ADAPTIVE = 0.05
NN_THRESHOLD = 0.15

TRAIN_60_START = 0.5 * EPS       # 1.5e-4 AU
TRAIN_60_END = 50.0 * EPS        # 1.5e-2 AU
TRAIN_40_START = 50.0 * EPS      # 1.5e-2 AU
TRAIN_40_END = 10.0              # 10 AU

X_MIN = 1e-8
X_MAX = 1e5


# =============================================================================
# System positions
# =============================================================================

systems = [
    ("Didymos-Dimorphos", 7.91e-9),
    (r"Near-softening stress $q$", 1.00e-3),
    ("Earth-Moon", 2.57e-3),
    (r"Scaled Gliese inner $q$", 5.04e-2),
    (r"Compact eccentric $q$", 8.00e-2),
    (r"Scaled Alpha Cen AB $q$", 1.13e-1),
    ("Sun-Earth", 1.00e0),
    (r"Gliese 667 inner $q$", 5.29e0),
    (r"Alpha Cen AB $q$", 1.13e1),
    (r"Gliese 667 outer $q$", 1.25e2),
    ("Proxima approx. sep.", 1.30e4),
]

labels = [s[0] for s in systems]
xvals = np.array([s[1] for s in systems])
yvals = np.arange(len(systems), 0, -1)


# =============================================================================
# Plot
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

fig, ax = plt.subplots(figsize=(13.5, 7.0))

ax.set_xscale("log")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0.15, len(systems) + 1.25)


# =============================================================================
# Background regions
# =============================================================================

ax.axvspan(X_MIN, R_SOFT_MIN, color="#f3c8cc", alpha=0.45)
ax.axvspan(R_SOFT_MIN, ADAPTIVE, color="#cfe8d6", alpha=0.55)
ax.axvspan(ADAPTIVE, NN_THRESHOLD, color="#fff0b8", alpha=0.65)
ax.axvspan(NN_THRESHOLD, X_MAX, color="#e6e6e6", alpha=0.55)


# =============================================================================
# Threshold lines
# =============================================================================

thresholds = [
    (EPS, "eps = 3e-4", "red", ":"),
    (R_SOFT_MIN, "r_soft_min = 5e-4", "green", ":"),
    (ADAPTIVE, "adaptive\n0.05", "tab:orange", "--"),
    (NN_THRESHOLD, "NN threshold\n0.15", "tab:blue", ":"),
]

for x, text, color, style in thresholds:
    ax.axvline(x, color=color, linestyle=style, linewidth=1.4, alpha=0.95)

# Place top labels manually to avoid overlap.
top_y = len(systems) + 0.75
ax.text(EPS * 0.85, top_y, "eps = 3e-4",
        ha="right", va="bottom", color="black", fontsize=9)
ax.text(R_SOFT_MIN * 1.15, top_y, "r_soft_min = 5e-4",
        ha="left", va="bottom", color="black", fontsize=9)
ax.text(ADAPTIVE, top_y, "adaptive\n0.05",
        ha="center", va="bottom", color="black", fontsize=9)
ax.text(NN_THRESHOLD * 1.08, top_y, "NN threshold\n0.15",
        ha="left", va="bottom", color="black", fontsize=9)


# =============================================================================
# Scatter points and value labels
# =============================================================================

ax.scatter(xvals, yvals, s=52, color="#0d1b3d", zorder=5)

for x, y in zip(xvals, yvals):
    ax.text(
        x * 1.18,
        y,
        f"{x:.2e}",
        ha="left",
        va="center",
        fontsize=9,
        color="#0d1b3d",
        zorder=6,
    )


# =============================================================================
# Training arrows
# =============================================================================

arrow_y = 0.82

# 60% training arrow: exactly 1.5e-4 to 1.5e-2 AU
ax.annotate(
    "",
    xy=(TRAIN_60_END, arrow_y),
    xytext=(TRAIN_60_START, arrow_y),
    arrowprops=dict(
        arrowstyle="<->",
        color="forestgreen",
        lw=2.0,
        shrinkA=0,
        shrinkB=0,
    ),
    zorder=7,
)

ax.text(
    np.sqrt(TRAIN_60_START * TRAIN_60_END),
    arrow_y + 0.18,
    "60% training samples:\n"
    r"$0.5\epsilon \leq r \leq 50\epsilon$ ="
    "\n"
    r"$1.5\times10^{-4}$ to $1.5\times10^{-2}$ AU",
    ha="center",
    va="bottom",
    fontsize=9,
    color="forestgreen",
)

# 40% training arrow: exactly 1.5e-2 to 10 AU
ax.annotate(
    "",
    xy=(TRAIN_40_END, arrow_y),
    xytext=(TRAIN_40_START, arrow_y),
    arrowprops=dict(
        arrowstyle="<->",
        color="#003cb3",
        lw=2.0,
        shrinkA=0,
        shrinkB=0,
    ),
    zorder=7,
)

ax.text(
    np.sqrt(TRAIN_40_START * TRAIN_40_END),
    arrow_y + 0.18,
    "40% training samples:\n"
    r"$50\epsilon \leq r \leq 10$ AU ="
    "\n"
    r"$1.5\times10^{-2}$ to $10$ AU",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#003cb3",
)


# =============================================================================
# Axes formatting
# =============================================================================

ax.set_yticks(yvals)
ax.set_yticklabels([f"{i+1}. {lab}" for i, lab in enumerate(labels)])

ax.set_xlabel("Representative pair separation / periastron (AU, log scale)")
ax.set_title(
    "Figure 1 revised — Distance position of discussed systems relative to SIMON thresholds",
    pad=18,
)

ax.grid(True, which="both", axis="x", alpha=0.18)
ax.grid(True, which="major", axis="y", alpha=0.10)


# =============================================================================
# Legend, bottom-left
# =============================================================================

legend_handles = [
    Patch(facecolor="#f3c8cc", edgecolor="black", alpha=0.45,
          label=r"Safety fallback: $r < 5\times10^{-4}$ AU"),
    Patch(facecolor="#cfe8d6", edgecolor="black", alpha=0.55,
          label="NN useful: correction can matter"),
    Patch(facecolor="#fff0b8", edgecolor="black", alpha=0.65,
          label="NN eligible but correction usually tiny"),
    Patch(facecolor="#e6e6e6", edgecolor="black", alpha=0.55,
          label="Far field: exact Newtonian"),
    Line2D([0], [0], color="forestgreen", lw=2.0,
           marker="<", markersize=6, label=r"60% train: $0.5\epsilon \leq r \leq 50\epsilon$"),
    Line2D([0], [0], color="#003cb3", lw=2.0,
           marker="<", markersize=6, label=r"40% train: $50\epsilon \leq r \leq 10$ AU"),
]

ax.legend(
    handles=legend_handles,
    loc="lower left",
    bbox_to_anchor=(0.01, 0.02),
    frameon=True,
    framealpha=0.95,
    borderpad=0.8,
)


# =============================================================================
# Save
# =============================================================================

fig.tight_layout()

out_png = "simon_distance_thresholds_training_ranges.png"
out_pdf = "simon_distance_thresholds_training_ranges.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print()
print("Sanity check:")
print(f"  epsilon              = {EPS:.3e} AU")
print(f"  r_soft_min           = {R_SOFT_MIN:.3e} AU")
print(f"  60% arrow start      = {TRAIN_60_START:.3e} AU")
print(f"  60% arrow end        = {TRAIN_60_END:.3e} AU")
print(f"  40% arrow start      = {TRAIN_40_START:.3e} AU")
print(f"  40% arrow end        = {TRAIN_40_END:.3e} AU")