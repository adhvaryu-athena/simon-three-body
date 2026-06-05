# adaptive_comparison.py
#
# Reads perf_summary files from before and after adaptive sub-stepping runs.
# Generates a comparison figure and table for the V3 paper (Step 6).
#
# INPUT  (read-only, never modified):
#   C:\Aarush\Physics\set 2_before_adaptive\hybrid_eval_out_v3\perf_summary_T100_before_adaptive.txt
#   C:\Aarush\Physics\set 3_after_adaptive\hybrid_eval_out_v3\perf_summary_T100_after_adaptive.txt
#
# OUTPUT (written to):
#   C:\Aarush\Physics\Results for paper\output\
#     adaptive_error_vs_dt.png          <- error vs dt comparison (main figure)
#     adaptive_divergence_slope.png     <- before/after lambda bar chart
#     adaptive_step_time.png            <- per-step time bar chart
#     adaptive_comparison_table.txt     <- full numbers table for paper
#   + copies of existing PNGs from both sets into the output folder
#
# Run:  python adaptive_comparison.py

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Paths ──────────────────────────────────────────────────────────────────────

BEFORE_DIR = r"C:\Aarush\Physics\set 2_before_adaptive\hybrid_eval_out_v3_dt04"
AFTER_DIR  = r"C:\Aarush\Physics\set 3_after_adaptive\hybrid_eval_out_v3_dt04"
OUT_DIR    = r"C:\Aarush\Physics\Results for paper\output"

BEFORE_SUMMARY = os.path.join(BEFORE_DIR, "perf_summary_T100_before_adaptive_dt04.txt")
AFTER_SUMMARY  = os.path.join(AFTER_DIR,  "perf_summary_T100_after_adaptive_dt04.txt")

# Existing PNGs to copy into output for convenience
BEFORE_PNGS = [
    "model_vs_rebound_divergence_T100_before_adaptive.png",
    "speed_accuracy_frontier_T100_before_adaptive.png",
    "comp_cost_frontier_T100_before_adaptive.png",
    "traj_overlay_xy_subplots_all_bodies_T100_before_adaptive.png",
    "traj_overlay_timeseries_all_bodies_T100_before_adaptive.png",
]
AFTER_PNGS = [
    "model_vs_rebound_divergence_T100_after_adaptive.png",
    "speed_accuracy_frontier_T100_after_adaptive.png",
    "comp_cost_frontier_T100_after_adaptive.png",
    "traj_overlay_xy_subplots_all_bodies_T100_after_adaptive.png",
    "traj_overlay_timeseries_all_bodies_T100_after_adaptive.png",
]

# ── Colours ────────────────────────────────────────────────────────────────────
C_BEFORE = "#DC2626"   # red  — before adaptive (problem)
C_AFTER  = "#16A34A"   # green — after adaptive  (solution)
C_IAS    = "#2563A6"   # blue — ias15 reference


# ── Parser ─────────────────────────────────────────────────────────────────────
def parse_perf_summary(path):
    """
    Parse a perf_summary_T100_*.txt file.
    Returns a dict with scalar metrics and a list of frontier rows.
    """
    data = {
        "ias15_time": None,
        "divergence_slope": None,
        "rep_step_time_us": None,
        "frontier": []   # list of {dt, final_err, step_time_us, speedup}
    }

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    in_frontier = False
    header_seen = False
    for line in lines:
        line = line.rstrip()

        if "total_time_sec:" in line and "Baseline" not in line and data["ias15_time"] is None:
            # Could be ias15 baseline
            pass
        if line.strip().startswith("total_time_sec:") and data["ias15_time"] is None:
            data["ias15_time"] = float(line.split(":")[1].strip())
        elif "divergence_slope_1_per_yr:" in line:
            data["divergence_slope"] = float(line.split(":")[1].strip())
        elif "time_per_step_sec:" in line:
            data["rep_step_time_us"] = float(line.split(":")[1].strip()) * 1e6
        elif line.startswith("dt\t"):
            in_frontier = True
            header_seen = True
            continue
        elif in_frontier and header_seen and line.strip() and not line.startswith("Notes"):
            parts = line.split("\t")
            if len(parts) >= 7:
                try:
                    data["frontier"].append({
                        "dt":           float(parts[0]),
                        "final_err":    float(parts[1]),
                        "step_time_us": float(parts[3]) * 1e6,
                        "speedup_str":  parts[7].strip() if len(parts) > 7 else parts[6].strip(),
                    })
                except (ValueError, IndexError):
                    pass

    return data


# ── Load data ──────────────────────────────────────────────────────────────────
print("[adaptive_comparison] Reading perf_summary files ...")
before = parse_perf_summary(BEFORE_SUMMARY)
after  = parse_perf_summary(AFTER_SUMMARY)

# Extract frontier arrays
b_dts  = [r["dt"]        for r in before["frontier"]]
b_errs = [r["final_err"] for r in before["frontier"]]
a_dts  = [r["dt"]        for r in after["frontier"]]
a_errs = [r["final_err"] for r in after["frontier"]]

b_slope = before["divergence_slope"]
a_slope = after["divergence_slope"]
b_step  = before["rep_step_time_us"]
a_step  = after["rep_step_time_us"]

print(f"  Before: slope={b_slope:.4f}/yr  worst_err={max(b_errs):.1f}  step={b_step:.1f} us")
print(f"  After:  slope={a_slope:.4f}/yr  worst_err={max(a_errs):.3f}  step={a_step:.1f} us")

worst_before = max(b_errs)
worst_after  = max(a_errs)
improvement  = worst_before / worst_after
slope_pct    = (b_slope - a_slope) / b_slope * 100
step_overhead = (a_step - b_step) / b_step * 100

print(f"\n  Error improvement:   {improvement:.0f}x  ({worst_before:.1f} -> {worst_after:.3f})")
print(f"  Slope reduction:     {slope_pct:.1f}%")
print(f"  Step time overhead:  {step_overhead:.1f}%  ({b_step:.1f} -> {a_step:.1f} us)")

# ── Create output folder ───────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
print(f"\n[adaptive_comparison] Output folder: {OUT_DIR}")


# ── Figure 1: Error vs dt (main comparison figure) ────────────────────────────
# Local font override for this paper figure only.
# This changes only chart appearance, not any parsed values or calculations.
with plt.rc_context({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.semilogy(b_dts, b_errs, "o-", color=C_BEFORE, lw=2.0, ms=7,
                label="Before adaptive sub-stepping", zorder=3)
    ax.semilogy(a_dts, a_errs, "s-", color=C_AFTER, lw=2.0, ms=7,
                label="After adaptive sub-stepping", zorder=3)

    # Annotate the worst-case point before adaptive sub-stepping.
    worst_idx = b_errs.index(max(b_errs))
    ax.annotate(
        f"Worst-case\n{worst_before:.0f} AU",
        xy=(b_dts[worst_idx], b_errs[worst_idx]),
        xytext=(b_dts[worst_idx] * 1.18, b_errs[worst_idx] * 0.42),
        color=C_BEFORE,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BEFORE, lw=1.0)
    )

    # Annotate the worst-case point after adaptive sub-stepping.
    worst_idx_a = a_errs.index(max(a_errs))
    ax.annotate(
        f"Worst-case\n{worst_after:.2f} AU",
        xy=(a_dts[worst_idx_a], a_errs[worst_idx_a]),
        xytext=(a_dts[worst_idx_a] * 0.55, a_errs[worst_idx_a] * 3.0),
        color=C_AFTER,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_AFTER, lw=1.0)
    )

    # Improvement arrow: visual annotation only.
    ax.annotate(
        "",
        xy=(b_dts[worst_idx], a_errs[worst_idx_a]),
        xytext=(b_dts[worst_idx], b_errs[worst_idx]),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2, alpha=0.8)
    )
    ax.text(
        b_dts[worst_idx] * 1.22,
        np.sqrt(b_errs[worst_idx] * a_errs[worst_idx_a]) * 0.55,
        f"{improvement:.0f}x reduction",
        color="gray",
        fontsize=8,
        va="center"
    )

    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Final RMS position error vs ias15 at T = 100 yr (AU)")

    # No chart title: the LaTeX caption explains the figure.
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xscale("log")

    out_path = os.path.join(OUT_DIR, "adaptive_error_vs_dt.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

print(f"  Saved adaptive_error_vs_dt.png")


# ── Figure 2: Divergence slope comparison (bar chart) ─────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))

bars = ax.bar(
    ["Before\nadaptive", "After\nadaptive"],
    [b_slope, a_slope],
    color=[C_BEFORE, C_AFTER],
    width=0.45, edgecolor="white", linewidth=1.2
)

# Value labels on bars
for bar, val in zip(bars, [b_slope, a_slope]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f"{val:.3f}/yr",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Divergence rate \u03bb (1/yr)")
ax.set_title(f"Divergence Rate: {slope_pct:.0f}% Reduction\nfrom Adaptive Sub-Stepping")
ax.set_ylim(0, max(b_slope, a_slope) * 1.3)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Reduction annotation
ax.annotate(
    f"-{slope_pct:.0f}%",
    xy=(1, a_slope + 0.01),
    xytext=(0.5, (b_slope + a_slope)/2),
    fontsize=12, color="gray", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2)
)

out_path = os.path.join(OUT_DIR, "adaptive_divergence_slope.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"  Saved adaptive_divergence_slope.png")


# ── Figure 3: Per-step time (bar chart) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))

bars = ax.bar(
    ["Before\nadaptive", "After\nadaptive"],
    [b_step, a_step],
    color=[C_BEFORE, C_AFTER],
    width=0.45, edgecolor="white", linewidth=1.2
)

for bar, val in zip(bars, [b_step, a_step]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.2,
            f"{val:.1f} \u03bcs",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Per-step time (\u03bcs)")
ax.set_title(f"Per-Step Overhead: +{step_overhead:.0f}%\nfrom Adaptive Sub-Stepping")
ax.set_ylim(0, max(b_step, a_step) * 1.3)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate(
    f"+{step_overhead:.0f}%\noverhead",
    xy=(1, a_step),
    xytext=(0.55, (b_step + a_step) * 0.7),
    fontsize=10, color="gray", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2)
)

out_path = os.path.join(OUT_DIR, "adaptive_step_time.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"  Saved adaptive_step_time.png")


# ── Figure 4: Combined panel figure ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: Error vs dt
ax = axes[0]
ax.semilogy(b_dts, b_errs, 'o-', color=C_BEFORE, lw=2, ms=7, label="Before adaptive")
ax.semilogy(a_dts, a_errs, 's-', color=C_AFTER,  lw=2, ms=7, label="After adaptive")
ax.set_xlabel("Timestep dt (yr)"); ax.set_ylabel("Final RMS error vs ias15 (AU)")
ax.set_title("(a)  Final Error vs Timestep")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3); ax.set_xscale("log")
# Annotate worst cases
ax.annotate(f"{worst_before:.0f} AU", xy=(b_dts[worst_idx], b_errs[worst_idx]),
            xytext=(b_dts[worst_idx]*1.3, b_errs[worst_idx]*0.25),
            color=C_BEFORE, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_BEFORE))
ax.annotate(f"{worst_after:.2f} AU", xy=(a_dts[worst_idx_a], a_errs[worst_idx_a]),
            xytext=(a_dts[worst_idx_a]*0.4, a_errs[worst_idx_a]*4),
            color=C_AFTER, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_AFTER))

# Panel 2: Divergence slope
ax = axes[1]
bars2 = ax.bar(["Before", "After"], [b_slope, a_slope],
               color=[C_BEFORE, C_AFTER], width=0.4, edgecolor="white")
for bar, val in zip(bars2, [b_slope, a_slope]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("\u03bb (1/yr)"); ax.set_title(f"(b)  Divergence Rate\n(-{slope_pct:.0f}% reduction)")
ax.set_ylim(0, b_slope*1.35); ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 3: Per-step time
ax = axes[2]
bars3 = ax.bar(["Before", "After"], [b_step, a_step],
               color=[C_BEFORE, C_AFTER], width=0.4, edgecolor="white")
for bar, val in zip(bars3, [b_step, a_step]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{val:.1f} \u03bcs", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Per-step time (\u03bcs)")
ax.set_title(f"(c)  Per-Step Overhead\n(+{step_overhead:.0f}% cost)")
ax.set_ylim(0, a_step*1.35); ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Adaptive Sub-Stepping: Before vs After Comparison  (T = 100 yr)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "adaptive_comparison_combined.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"  Saved adaptive_comparison_combined.png")


# ── Table ──────────────────────────────────────────────────────────────────────
table_path = os.path.join(OUT_DIR, "adaptive_comparison_table.txt")
with open(table_path, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("ADAPTIVE SUB-STEPPING: BEFORE vs AFTER COMPARISON\n")
    f.write("T = 100 yr  |  m = [1.0, 0.01, 0.005] M_sun\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"{'Metric':<32} {'Before':>12} {'After':>12} {'Change':>10}\n")
    f.write("-" * 68 + "\n")

    rows = [
        ("Bodies bounded at T=100yr?",   "NO (ejection)",   "YES",            ""),
        ("Divergence slope lambda (1/yr)",f"{b_slope:.4f}",  f"{a_slope:.4f}", f"-{slope_pct:.0f}%"),
        ("Worst-case error (AU, any dt)", f"{worst_before:.1f}", f"{worst_after:.3f}", f"{improvement:.0f}x lower"),
        ("Error range across dt values",
         f"[{min(b_errs):.2f}, {worst_before:.1f}]",
         f"[{min(a_errs):.3f}, {worst_after:.3f}]", ""),
        ("Best speedup vs ias15",
         f"{max(float(r['speedup_str'].rstrip('x')) for r in before['frontier'] if r['speedup_str'].endswith('x')):.2f}x",
         f"{max(float(r['speedup_str'].rstrip('x')) for r in after['frontier']  if r['speedup_str'].endswith('x')):.2f}x",
         ""),
        ("Per-step time (rep dt=0.02)",  f"{b_step:.1f} us",  f"{a_step:.1f} us",  f"+{step_overhead:.0f}%"),
        ("Adaptive activations",         "0% (no mechanism)", "0.2-1.1% of steps", ""),
    ]

    for metric, bval, aval, change in rows:
        f.write(f"{metric:<32} {bval:>12} {aval:>12} {change:>10}\n")

    f.write("\n" + "=" * 65 + "\n")
    f.write("KEY NUMBERS FOR PAPER:\n")
    f.write(f"  Worst-case error reduction: {worst_before:.1f} -> {worst_after:.3f} AU  ({improvement:.0f}x improvement)\n")
    f.write(f"  Divergence slope reduction: {b_slope:.3f} -> {a_slope:.3f} /yr  ({slope_pct:.0f}% improvement)\n")
    f.write(f"  Per-step overhead added:    {b_step:.1f} -> {a_step:.1f} us  (+{step_overhead:.0f}%)\n")
    f.write(f"  Bodies bounded:             NO -> YES\n")
    f.write("\nSOURCE FILES:\n")
    f.write(f"  Before: {BEFORE_SUMMARY}\n")
    f.write(f"  After:  {AFTER_SUMMARY}\n")

print(f"  Saved adaptive_comparison_table.txt")


# ── Copy existing PNGs into output folder ─────────────────────────────────────
print("\n[adaptive_comparison] Copying existing PNGs ...")
for fname in BEFORE_PNGS:
    src = os.path.join(BEFORE_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT_DIR, fname))
        print(f"  Copied {fname}")
    else:
        print(f"  WARNING: not found: {src}")

for fname in AFTER_PNGS:
    src = os.path.join(AFTER_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT_DIR, fname))
        print(f"  Copied {fname}")
    else:
        print(f"  WARNING: not found: {src}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("DONE.  Files written to:")
print(f"  {OUT_DIR}")
print()
print("NEW figures generated:")
print("  adaptive_error_vs_dt.png          <- main paper figure")
print("  adaptive_comparison_combined.png  <- 3-panel summary figure")
print("  adaptive_divergence_slope.png     <- bar chart lambda")
print("  adaptive_step_time.png            <- bar chart overhead")
print("  adaptive_comparison_table.txt     <- numbers table for paper")
print()
print("Existing PNGs copied from before/after folders.")
print("=" * 65)
