"""
plot_simon_distance_thresholds_training_ranges.py  -- REVISED

Changes from original:
  1. Red zone boundary corrected: uses r_gate = 4e-4 AU (where r_soft = r_soft_min)
     NOT r_soft_min = 5e-4 AU directly as an r-value.
  2. Green zone relabelled: "Sub-step + NN overlap" (both adaptive sub-stepping
     AND NN active simultaneously), lower boundary corrected to r_gate.
  3. Yellow zone relabelled: "NN macro-step zone" — this is where the NEW
     trajectory-trained correction is significant (c_optimal 0.80-0.93).
     Old label "correction usually tiny" was based on old analytic training.
  4. Training arrows updated: OLD model arrows kept (labelled OLD), NEW proposed
     training ranges added as separate arrows showing both sub-zones.
  5. eps line annotated as sitting INSIDE the hard fallback zone (eps < r_gate).
  6. c_analytic values annotated at key zone boundaries.

All threshold values read directly from pair_eval_after_adaptive.py:
    cfg.eps          = 3e-4
    cfg.r_soft_min   = 5e-4
    adapt_thresh     = 0.05
    nn_thresh        = 500 * cfg.eps = 0.15
    max_substeps     = 16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# =============================================================================
# Constants -- read directly from pair_eval_after_adaptive.py
# =============================================================================

EPS          = 3e-4          # cfg.eps
R_SOFT_MIN   = 5e-4          # cfg.r_soft_min  (gate fires when r_soft < this)
ADAPT_THRESH = 0.05          # adapt_thresh
NN_THRESH    = 500.0 * EPS   # nn_thresh = 0.15 AU
MAX_SUBSTEPS = 16            # max_substeps

# Issue 1 fix: r_gate is the r-value where r_soft = R_SOFT_MIN
# r_soft = sqrt(r^2 + eps^2) = R_SOFT_MIN  =>  r = sqrt(R_SOFT_MIN^2 - EPS^2)
R_GATE = np.sqrt(R_SOFT_MIN**2 - EPS**2)   # = 4e-4 AU exactly

# =============================================================================
# OLD training ranges (train_pair_correction_new.py)
# =============================================================================

OLD_60_START = 0.5 * EPS     # 1.5e-4 AU
OLD_60_END   = 50.0 * EPS    # 1.5e-2 AU
OLD_40_START = 50.0 * EPS    # 1.5e-2 AU
OLD_40_END   = 10.0          # 10 AU

# =============================================================================
# NEW proposed training ranges (generate_encounter_data.py, extended)
# =============================================================================

# Macro-step zone: where the new trajectory-trained c is targeted
NEW_MACRO_START = ADAPT_THRESH + 0.002   # 0.052 AU  (as in generate_encounter_data.py)
NEW_MACRO_END   = NN_THRESH - 0.002      # 0.148 AU

# Sub-step overlap zone: needs effective sub_dt as 4th input
NEW_SUB_START   = R_GATE                 # 4e-4 AU (lower bound of NN-active zone)
NEW_SUB_END     = ADAPT_THRESH           # 0.05 AU

# =============================================================================
# Systems
# =============================================================================

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
xvals  = np.array([s[1] for s in systems])
yvals  = np.arange(len(systems), 0, -1)

X_MIN = 1e-8
X_MAX = 1e5

# =============================================================================
# Plot setup
# =============================================================================

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

fig, ax = plt.subplots(figsize=(15.0, 8.0))

ax.set_xscale("log")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-0.5, len(systems) + 2.8)

# =============================================================================
# Background zones -- boundaries are r-values, not r_soft values
# Zone 1 (red):    r < R_GATE = 4e-4 AU   Hard fallback: NN fires but
#                                          r_soft < r_soft_min → output rejected
# Zone 2 (green):  R_GATE to ADAPT_THRESH  Sub-step overlap: both adaptive
#                                          sub-stepping AND NN active
# Zone 3 (yellow): ADAPT_THRESH to NN_THRESH  NN macro-step zone: NN active,
#                                              no sub-stepping
# Zone 4 (grey):   > NN_THRESH             Far field: exact Newtonian only
# =============================================================================

ax.axvspan(X_MIN,        R_GATE,       color="#f3c8cc", alpha=0.50)   # red
ax.axvspan(R_GATE,       ADAPT_THRESH, color="#cfe8d6", alpha=0.55)   # green
ax.axvspan(ADAPT_THRESH, NN_THRESH,    color="#fff0b8", alpha=0.65)   # yellow
ax.axvspan(NN_THRESH,    X_MAX,        color="#e6e6e6", alpha=0.55)   # grey

# =============================================================================
# Threshold vertical lines
# =============================================================================

top_y    = len(systems) + 0.85
label_y  = len(systems) + 1.05

# eps -- note: sits INSIDE the hard fallback zone (eps < R_GATE)
ax.axvline(EPS, color="red", linestyle=":", linewidth=1.4, alpha=0.9)
ax.text(EPS * 0.82, top_y,
        r"$\varepsilon$ = 3×10⁻⁴" + "\n(softening length)\n[inside fallback zone]",
        ha="right", va="bottom", color="red", fontsize=8)

# R_GATE -- actual lower boundary of NN-active zone (r where r_soft = r_soft_min)
ax.axvline(R_GATE, color="darkgreen", linestyle="-.", linewidth=1.6, alpha=0.95)
ax.text(R_GATE * 1.12, top_y,
        r"$r_{gate}$ = 4×10⁻⁴" + "\n(r_soft = r_soft_min)\nNN active above here",
        ha="left", va="bottom", color="darkgreen", fontsize=8)

# ADAPT_THRESH -- adaptive sub-stepping boundary
ax.axvline(ADAPT_THRESH, color="tab:orange", linestyle="--", linewidth=1.6, alpha=0.95)
ax.text(ADAPT_THRESH * 1.05, top_y,
        "adapt_thresh\n= 0.05 AU\nsub-stepping\nfires below here",
        ha="left", va="bottom", color="darkorange", fontsize=8)

# NN_THRESH -- NN activation boundary
ax.axvline(NN_THRESH, color="tab:blue", linestyle=":", linewidth=1.6, alpha=0.95)
ax.text(NN_THRESH * 1.05, top_y,
        "nn_thresh\n= 0.15 AU\nNN fires\nbelow here",
        ha="left", va="bottom", color="tab:blue", fontsize=8)

# =============================================================================
# c_analytic annotations at key boundaries (to show softening significance)
# =============================================================================

def c_analytic(r):
    rs = np.sqrt(r**2 + EPS**2)
    return (rs / r)**3

annot_color = "#555555"
for r_val, yoff in [(R_GATE, -0.38), (1e-3, -0.38), (5e-3, -0.38), (ADAPT_THRESH, -0.38)]:
    c_val = c_analytic(r_val)
    ax.text(r_val, 0.30 + yoff,
            f"c_ana\n={c_val:.3f}",
            ha="center", va="top", fontsize=7.5, color=annot_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# =============================================================================
# System scatter points
# =============================================================================

ax.scatter(xvals, yvals, s=52, color="#0d1b3d", zorder=5)

for x, y in zip(xvals, yvals):
    ax.text(x * 1.18, y, f"{x:.2e}",
            ha="left", va="center", fontsize=9, color="#0d1b3d", zorder=6)

# =============================================================================
# Training arrows
# Arrow rows:
#   Row A (y = -0.10): OLD 60% training
#   Row B (y = -0.50): OLD 40% training
#   Row C (y =  0.35): NEW sub-step zone
#   Row D (y =  0.00): NEW macro-step zone
# =============================================================================

def draw_arrow(ax, x_start, x_end, y_pos, color, label_text, label_above=True):
    ax.annotate("",
        xy=(x_end, y_pos), xytext=(x_start, y_pos),
        arrowprops=dict(arrowstyle="<->", color=color, lw=2.0,
                        shrinkA=0, shrinkB=0),
        zorder=7)
    x_mid = np.sqrt(x_start * x_end)
    y_text = y_pos + 0.22 if label_above else y_pos - 0.22
    va = "bottom" if label_above else "top"
    ax.text(x_mid, y_text, label_text,
            ha="center", va=va, fontsize=8, color=color)

# OLD model arrows (row -0.10 and -0.50)
draw_arrow(ax, OLD_60_START, OLD_60_END, -0.10, "forestgreen",
           "OLD 60%: 0.5ε–50ε\n(1.5×10⁻⁴ – 1.5×10⁻² AU)\nsoftening zone",
           label_above=False)

draw_arrow(ax, OLD_40_START, OLD_40_END, -0.50, "#003cb3",
           "OLD 40%: 50ε–10 AU\n(1.5×10⁻² – 10 AU)\nfar field",
           label_above=False)

# NEW proposed training arrows (rows 0.35 and 0.70)
draw_arrow(ax, NEW_SUB_START, NEW_SUB_END, 0.72, "#b35900",
           "NEW sub-step zone: 4×10⁻⁴ – 0.05 AU\n(use effective sub_dt as 4th input)\nc_analytic: 1.00–1.95 here",
           label_above=True)

draw_arrow(ax, NEW_MACRO_START, NEW_MACRO_END, 0.35, "#8b0000",
           "NEW macro-step zone: 0.052–0.148 AU\n(dt as 4th input; c_optimal = 0.80–0.93)\nNEW training targets this zone",
           label_above=True)

# =============================================================================
# Axes
# =============================================================================

ax.set_yticks(yvals)
ax.set_yticklabels([f"{i+1}. {lab}" for i, lab in enumerate(labels)])

ax.set_xlabel("Representative pair separation / periastron (AU, log scale)")
ax.set_title(
    "SIMON distance thresholds, zone descriptions, and training ranges\n"
    "(OLD analytic training vs NEW proposed trajectory-trained)",
    pad=14,
)

ax.grid(True, which="both", axis="x", alpha=0.18)
ax.grid(True, which="major", axis="y", alpha=0.10)

# =============================================================================
# Legend
# =============================================================================

legend_handles = [
    # Zones
    Patch(facecolor="#f3c8cc", edgecolor="grey", alpha=0.7,
          label=r"Zone 1: Hard fallback  $r < 4\times10^{-4}$ AU"
                "\n  NN fires but r_soft < r_soft_min → output rejected,"
                "\n  exact Newtonian used; eps=3×10⁻⁴ sits here"),
    Patch(facecolor="#cfe8d6", edgecolor="grey", alpha=0.7,
          label=r"Zone 2: Sub-step + NN overlap  $4\times10^{-4}$ – 0.05 AU"
                "\n  Both adaptive sub-stepping AND NN active;"
                "\n  c_analytic: 1.00–1.95; softening significant at low end"),
    Patch(facecolor="#fff0b8", edgecolor="grey", alpha=0.8,
          label="Zone 3: NN macro-step zone  0.05 – 0.15 AU\n"
                "  NN active, no sub-stepping; c_analytic≈1.0;\n"
                "  NEW training: c_optimal=0.80–0.93 (leapfrog discretisation error)"),
    Patch(facecolor="#e6e6e6", edgecolor="grey", alpha=0.7,
          label="Zone 4: Far field  > 0.15 AU\n"
                "  NN not called; exact Newtonian throughout"),
    # Threshold lines
    Line2D([0],[0], color="red",       ls=":",  lw=1.4,
           label=r"$\varepsilon$ = 3×10⁻⁴ AU (softening length; inside Zone 1)"),
    Line2D([0],[0], color="darkgreen", ls="-.", lw=1.6,
           label=r"$r_{gate}$ = 4×10⁻⁴ AU (r_soft=r_soft_min; Zone 1/2 boundary)"),
    Line2D([0],[0], color="darkorange",ls="--", lw=1.6,
           label="adapt_thresh = 0.05 AU (Zone 2/3 boundary; sub-stepping turns off)"),
    Line2D([0],[0], color="tab:blue",  ls=":",  lw=1.6,
           label="nn_thresh = 0.15 AU (Zone 3/4 boundary; NN turns off)"),
    # Training arrows
    Line2D([0],[0], color="forestgreen", lw=2.0,
           label="OLD 60% training: softening zone 1.5×10⁻⁴–1.5×10⁻² AU"),
    Line2D([0],[0], color="#003cb3",     lw=2.0,
           label="OLD 40% training: far field 1.5×10⁻²–10 AU"),
    Line2D([0],[0], color="#b35900",     lw=2.0,
           label="NEW sub-step zone training: 4×10⁻⁴–0.05 AU (effective sub_dt as input)"),
    Line2D([0],[0], color="#8b0000",     lw=2.0,
           label="NEW macro-step zone training: 0.052–0.148 AU (dt as 4th input)"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=True,
    framealpha=0.97,
    borderpad=0.9,
    handlelength=2.2,
)

# =============================================================================
# Save
# =============================================================================

fig.tight_layout(rect=[0, 0, 0.72, 1.0])   # leave room for right-side legend

out_png = "simon_distance_thresholds_training_ranges_revised.png"
out_pdf = "simon_distance_thresholds_training_ranges_revised.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print()
print("=== Sanity check (all from code) ===")
print(f"  eps              = {EPS:.3e} AU  (cfg.eps)")
print(f"  r_soft_min       = {R_SOFT_MIN:.3e} AU  (cfg.r_soft_min, gate on r_soft)")
print(f"  r_gate           = {R_GATE:.3e} AU  (r where r_soft = r_soft_min)")
print(f"  eps < r_gate?    = {EPS < R_GATE}  (eps is inside hard fallback zone)")
print(f"  adapt_thresh     = {ADAPT_THRESH:.3f} AU")
print(f"  nn_thresh        = {NN_THRESH:.3f} AU  (500 * eps)")
print(f"  max_substeps     = {MAX_SUBSTEPS}")
print()
print("=== Zone boundaries ===")
print(f"  Zone 1 (hard fallback):    r < {R_GATE:.2e} AU")
print(f"  Zone 2 (sub-step + NN):    {R_GATE:.2e} <= r < {ADAPT_THRESH}")
print(f"  Zone 3 (NN macro-step):    {ADAPT_THRESH} <= r < {NN_THRESH}")
print(f"  Zone 4 (far field):        r >= {NN_THRESH}")
print()
print("=== c_analytic at zone boundaries ===")
for r_val in [R_GATE, 1e-3, 5e-3, ADAPT_THRESH, NN_THRESH]:
    rs = np.sqrt(r_val**2 + EPS**2)
    c  = (rs / r_val)**3
    print(f"  r={r_val:.2e} AU: c_analytic={c:.4f}")
print()
print("=== OLD training ===")
print(f"  60%: ({OLD_60_START:.2e}, {OLD_60_END:.2e}) AU")
print(f"  40%: ({OLD_40_START:.2e}, {OLD_40_END:.2e}) AU")
print()
print("=== NEW proposed training ===")
print(f"  Sub-step zone:  ({NEW_SUB_START:.2e}, {NEW_SUB_END:.3f}) AU  [effective sub_dt as 4th input]")
print(f"  Macro-step zone:({NEW_MACRO_START:.3f},  {NEW_MACRO_END:.3f}) AU  [dt as 4th input]")
