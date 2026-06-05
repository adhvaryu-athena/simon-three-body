"""
plot_simon_distance_thresholds_training_ranges_clear_v2.py

Clear two-panel version of the SIMON distance-threshold plot.

Main layout fixes:
  1. No large right-side legend. Zone explanations are written directly as callouts.
  2. Top-panel threshold text is moved to the right empty space with arrows.
  3. Bottom training-range panel has more height and bottom margin, so labels are not cut.
  4. Training arrows are separated into rows to avoid overlap.

Run:
    python plot_simon_distance_thresholds_training_ranges_clear_v2.py

Outputs:
    simon_distance_thresholds_training_ranges_clear_v2.png
    simon_distance_thresholds_training_ranges_clear_v2.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# =============================================================================
# Constants -- match SIMON / pair_eval_after_adaptive settings
# =============================================================================

EPS          = 3e-4          # softening length, cfg.eps
R_SOFT_MIN   = 5e-4          # safety gate threshold on r_soft
ADAPT_THRESH = 0.05          # adaptive sub-stepping threshold
NN_THRESH    = 500.0 * EPS   # 0.15 AU, NN activation threshold
MAX_SUBSTEPS = 16

# r value where r_soft = R_SOFT_MIN:
# sqrt(r^2 + eps^2) = R_SOFT_MIN  ->  r = sqrt(R_SOFT_MIN^2 - EPS^2)
R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))  # 4e-4 AU

# OLD analytic training ranges from train_pair_correction_new.py
OLD_60_START = 0.5 * EPS       # 1.5e-4 AU
OLD_60_END   = 50.0 * EPS      # 1.5e-2 AU
OLD_40_START = 50.0 * EPS      # 1.5e-2 AU
OLD_40_END   = 10.0            # 10 AU

# NEW proposed trajectory-trained ranges
NEW_SUB_START   = R_GATE                 # 4e-4 AU
NEW_SUB_END     = ADAPT_THRESH           # 0.05 AU
NEW_MACRO_START = ADAPT_THRESH + 0.002   # 0.052 AU
NEW_MACRO_END   = NN_THRESH - 0.002      # 0.148 AU

# Representative systems / scales
systems = [
    ("Didymos-Dimorphos",           7.91e-9),
    (r"Near-softening stress $q$",  1.00e-3),
    ("Earth-Moon",                  2.57e-3),
    (r"Scaled Gliese inner $q$",    5.04e-2),
    (r"Compact eccentric $q$",      8.00e-2),
    (r"Scaled Alpha Cen AB $q$",    1.13e-1),
    ("Sun-Earth",                   1.00e0),
    (r"Gliese 667 inner $q$",       5.29e0),
    (r"Alpha Cen AB $q$",           1.13e1),
    (r"Gliese 667 outer $q$",       1.25e2),
    ("Proxima approx. sep.",        1.30e4),
]

labels = [s[0] for s in systems]
xvals  = np.array([s[1] for s in systems], dtype=float)
yvals  = np.arange(len(systems), 0, -1)

X_MIN = 1e-8
X_MAX = 1e5

# =============================================================================
# Styling
# =============================================================================

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

ZONE_RED    = "#f3c8cc"
ZONE_GREEN  = "#cfe8d6"
ZONE_YELLOW = "#fff0b8"
ZONE_GREY   = "#e6e6e6"
POINT_COLOR = "#0d1b3d"

# =============================================================================
# Helper functions
# =============================================================================

def c_analytic(r):
    r = np.asarray(r, dtype=float)
    rs = np.sqrt(r * r + EPS * EPS)
    return (rs / r) ** 3


def add_zones(ax):
    """Add background zones and threshold lines to an axis."""
    ax.axvspan(X_MIN,        R_GATE,       color=ZONE_RED,    alpha=0.45, zorder=0)
    ax.axvspan(R_GATE,       ADAPT_THRESH, color=ZONE_GREEN,  alpha=0.50, zorder=0)
    ax.axvspan(ADAPT_THRESH, NN_THRESH,    color=ZONE_YELLOW, alpha=0.62, zorder=0)
    ax.axvspan(NN_THRESH,    X_MAX,        color=ZONE_GREY,   alpha=0.50, zorder=0)

    ax.axvline(EPS,          color="red",        linestyle=":",  linewidth=1.7, zorder=3)
    ax.axvline(R_GATE,       color="darkgreen",  linestyle="-.", linewidth=1.7, zorder=3)
    ax.axvline(ADAPT_THRESH, color="darkorange", linestyle="--", linewidth=1.7, zorder=3)
    ax.axvline(NN_THRESH,    color="tab:blue",   linestyle=":",  linewidth=1.9, zorder=3)


def callout(ax, x_point, y_point, x_text, y_text, text, color):
    """Arrow callout used to keep threshold labels out of crowded zones."""
    ax.annotate(
        text,
        xy=(x_point, y_point),
        xytext=(x_text, y_text),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=8.5,
        color=color,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.0,
            shrinkA=2,
            shrinkB=2,
            connectionstyle="arc3,rad=0.10",
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec=color,
            alpha=0.88,
        ),
        zorder=10,
    )


def draw_range(ax, x_start, x_end, y, color, text, text_y_offset=0.22):
    """Draw a two-sided arrow and put text above it."""
    ax.annotate(
        "",
        xy=(x_end, y),
        xytext=(x_start, y),
        arrowprops=dict(
            arrowstyle="<->",
            color=color,
            lw=2.3,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )
    x_mid = np.sqrt(x_start * x_end)
    ax.text(
        x_mid,
        y + text_y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.84),
        zorder=7,
    )


def format_log_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=14))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="both", axis="x", alpha=0.16)
    ax.grid(True, which="major", axis="y", alpha=0.10)

# =============================================================================
# Figure setup
# =============================================================================

fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    figsize=(18, 11),
    sharex=True,
    gridspec_kw={"height_ratios": [4.2, 1.7], "hspace": 0.08},
)

fig.suptitle(
    "SIMON distance thresholds, zone descriptions, and training ranges\n"
    "clear layout: systems above, training ranges below",
    y=0.965,
    fontsize=15,
)

# Leave enough bottom space so the lower panel is never cropped.
fig.subplots_adjust(left=0.15, right=0.965, top=0.905, bottom=0.125)

# =============================================================================
# Top panel: systems and zones
# =============================================================================

add_zones(ax_top)
format_log_axis(ax_top)

ax_top.set_ylim(0.25, len(systems) + 2.25)
ax_top.set_yticks(yvals)
ax_top.set_yticklabels([f"{i+1}. {lab}" for i, lab in enumerate(labels)])

# Short zone labels only. Detailed text is moved to right-side callouts.
zone_y = len(systems) + 1.73
ax_top.text(np.sqrt(X_MIN * R_GATE), zone_y, "Zone 1\nhard fallback",
            ha="center", va="top", fontsize=8.5, color="#8b3a3a")
ax_top.text(np.sqrt(R_GATE * ADAPT_THRESH), zone_y, "Zone 2\nsub-step + NN",
            ha="center", va="top", fontsize=8.5, color="#2f6b3f")
ax_top.text(np.sqrt(ADAPT_THRESH * NN_THRESH), zone_y, "Zone 3\nNN macro-step",
            ha="center", va="top", fontsize=8.5, color="#8a6500")
ax_top.text(25.0, zone_y, "Zone 4\nfar field",
            ha="center", va="top", fontsize=8.5, color="#555555")

# System points and numeric labels.
ax_top.scatter(xvals, yvals, s=52, color=POINT_COLOR, zorder=5)
for x, y in zip(xvals, yvals):
    ax_top.text(
        x * 1.16,
        y,
        f"{x:.2e}",
        ha="left",
        va="center",
        fontsize=9,
        color=POINT_COLOR,
        zorder=6,
    )

# Threshold callouts moved to the right side of the close-encounter area.
# x_text is in the far-field / open region, avoiding the crowded threshold lines.
callout(
    ax_top,
    EPS,
    len(systems) + 0.95,
    0.42,
    len(systems) + 1.10,
    "ε = 3×10⁻⁴ AU\nsoftening length\ninside fallback zone",
    "red",
)
callout(
    ax_top,
    R_GATE,
    len(systems) + 0.35,
    0.42,
    len(systems) + 0.35,
    "r_gate = 4×10⁻⁴ AU\nr_soft = r_soft_min\nNN usable above this",
    "darkgreen",
)
callout(
    ax_top,
    ADAPT_THRESH,
    len(systems) - 0.25,
    0.42,
    len(systems) - 0.40,
    "adapt_thresh = 0.05 AU\nsub-stepping fires below this",
    "darkorange",
)
callout(
    ax_top,
    NN_THRESH,
    len(systems) - 0.85,
    0.42,
    len(systems) - 1.15,
    "nn_thresh = 0.15 AU\nNN fires below this",
    "tab:blue",
)

# c_analytic labels near bottom of top panel.
c_labels = [
    (R_GATE,      r"$c_{ana}=1.953$"),
    (1.0e-3,      r"$c_{ana}=1.138$"),
    (5.0e-3,      r"$c_{ana}=1.005$"),
    (ADAPT_THRESH, r"$c_{ana}=1.000$"),
]
for x, txt in c_labels:
    ax_top.text(
        x,
        0.58,
        txt,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        zorder=8,
    )

ax_top.set_ylabel("")
ax_top.tick_params(axis="x", labelbottom=False)

# =============================================================================
# Bottom panel: training ranges
# =============================================================================

add_zones(ax_bottom)
format_log_axis(ax_bottom)

# More vertical space than previous version, so nothing is cut.
ax_bottom.set_ylim(0.05, 5.10)
ax_bottom.set_yticks([4.25, 3.20, 2.10, 1.05])
ax_bottom.set_yticklabels(["OLD 60%", "OLD 40%", "NEW sub-step", "NEW macro-step"])

# Draw separated rows. Text sits above arrows inside the bottom panel.
draw_range(
    ax_bottom,
    OLD_60_START,
    OLD_60_END,
    4.25,
    "forestgreen",
    "0.5ε–50ε  (1.5e-4–1.5e-2 AU)\nanalytic softening zone",
    text_y_offset=0.16,
)

draw_range(
    ax_bottom,
    OLD_40_START,
    OLD_40_END,
    3.20,
    "#003cb3",
    "50ε–10 AU  (1.5e-2–10 AU)\nfar-field samples",
    text_y_offset=0.16,
)

draw_range(
    ax_bottom,
    NEW_SUB_START,
    NEW_SUB_END,
    2.10,
    "#b35900",
    "4e-4–0.05 AU\neffective sub_dt as 4th input\n$c_{analytic}$: 1.00–1.95",
    text_y_offset=0.16,
)

draw_range(
    ax_bottom,
    NEW_MACRO_START,
    NEW_MACRO_END,
    1.05,
    "#8b0000",
    "0.052–0.148 AU\ndt as 4th input\n$c_{optimal}$ ≈ 0.80–0.93",
    text_y_offset=0.16,
)

ax_bottom.set_xlabel(
    "Representative pair separation / periastron (AU, log scale)",
    labelpad=12,
)

# Small explanatory note in the far-right blank space of bottom panel.
ax_bottom.text(
    22,
    4.80,
    "Reading guide:\nTop panel = physical thresholds and example systems\nBottom panel = old vs new training ranges",
    ha="left",
    va="top",
    fontsize=8.5,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#999999", alpha=0.88),
)

# =============================================================================
# Save
# =============================================================================

out_png = "simon_distance_thresholds_training_ranges_clear_v2.png"
out_pdf = "simon_distance_thresholds_training_ranges_clear_v2.pdf"

# Do not use bbox_inches='tight' here; the margins are set manually to avoid cropping.
fig.savefig(out_png, dpi=300)
fig.savefig(out_pdf)
plt.close(fig)

print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
