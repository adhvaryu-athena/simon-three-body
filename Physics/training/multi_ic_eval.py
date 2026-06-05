# multi_ic_eval.py
#
# Runs SIMON across 4 distinct three-body initial conditions (ICs).
# Tests whether results generalise beyond the single IC used in V2.
#
# REQUIRES in same folder:
#   pair_eval_after_adaptive.py
#   pair_correction_nn.pt
#
# PRODUCES in multi_ic_out/:
#   multi_ic_summary.txt          <- numbers table for paper
#   multi_ic_divergence.png       <- divergence curve per IC (4-panel)
#   multi_ic_results.png          <- summary bar charts
#   multi_ic_trajectories.png     <- trajectory overlay per IC (4-panel)
#
# Run: python multi_ic_eval.py
#
# Expected runtime: ~2 minutes (4 ICs x ~30s each)

import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

sys.path.insert(0, r"C:\Aarush\Physics\set 3_after_adaptive")
from pair_eval_after_adaptive import (
    PairCorrectionNN, HybridConfig,
    simulate_leapfrog_hybrid,
    simulate_rebound_ias15,
    rms_sep, fit_log_slope,
)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "pair_correction_nn.pt"
OUT_DIR    = "multi_ic_out"
DT_REF     = 0.04      # reference dt — same as V2 paper
T          = 100.0     # simulation horizon
N_SAMPLES  = 5000      # same as V2
EJECTION_THRESHOLD = 10.0   # AU — final RMS above this = ejection

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("[multi_ic] Loading SIMON model ...")
model = PairCorrectionNN(hidden=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
cfg = HybridConfig()
print(f"  {MODEL_PATH}  params={sum(p.numel() for p in model.parameters())}")

# ── Initial conditions ─────────────────────────────────────────────────────────
# All ICs use G=1 units (masses in solar masses, distances in AU, time in years
# where 2*pi years = 1 orbit for a 1 Msun system at 1 AU).
# Each IC is physically bound (total energy < 0, verified below).
# CoM centering is applied to each before use.
#
# IC design rationale:
#   IC1 (default): V2's exact IC. Known to work. m=[1, 0.01, 0.005] — hierarchical mass ratio.
#   IC2 (near-equal mass): m=[1, 0.5, 0.25]. More symmetric masses → more chaotic dynamics.
#                          All three bodies interact at comparable gravitational strength.
#   IC3 (tight binary): Same masses as IC1 but body 1 starts at 0.5 AU instead of 1.0 AU.
#                       Tests SIMON under more frequent/aggressive close encounters.
#   IC4 (hierarchical): One body far (5 AU) from a tight inner pair.
#                       Approaches the restricted 3-body limit. Tests a different regime.

ICS = {
    "IC1_default": {
        "label":   "IC1: Default (V2)",
        "desc":    "m=[1.0, 0.01, 0.005] M\u2609  |  Same as V2 paper",
        "m":   np.array([1.0, 0.01, 0.005]),
        "x0":  np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64),
        "v0":  np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64),
        "color": "#2563A6",
    },
    "IC2_near_equal": {
        "label":   "IC2: Near-Equal Mass",
        "desc":    "m=[1.0, 0.5, 0.25] M\u2609  |  All bodies interact comparably",
        "m":   np.array([1.0, 0.5, 0.25]),
        "x0":  np.array([[0, 0, 0], [1, 0, 0], [-0.5, 0.8, 0]], dtype=np.float64),
        "v0":  np.array([[0, 0, 0], [0, 0.6, 0], [-0.4, -0.3, 0]], dtype=np.float64),
        "color": "#16A34A",
    },
    "IC3_tight_binary": {
        "label":   "IC3: Tight Inner Pair",
        "desc":    "m=[1.0, 0.01, 0.005] M\u2609  |  Bodies 0,1 start at 0.5 AU",
        "m":   np.array([1.0, 0.01, 0.005]),
        "x0":  np.array([[0, 0, 0], [0.5, 0, 0], [0, 2.5, 0]], dtype=np.float64),
        "v0":  np.array([[0, 0, 0], [0, 1.3, 0], [-0.4, 0, 0]], dtype=np.float64),
        "color": "#DC2626",
    },
    "IC4_hierarchical": {
        "label":   "IC4: Hierarchical",
        "desc":    "m=[1.0, 0.01, 0.005] M\u2609  |  Body 2 starts at 5 AU",
        "m":   np.array([1.0, 0.01, 0.005]),
        "x0":  np.array([[0, 0, 0], [1, 0, 0], [0, 5.0, 0]], dtype=np.float64),
        "v0":  np.array([[0, 0, 0], [0, 1.0, 0], [-0.12, 0, 0]], dtype=np.float64),
        "color": "#9333EA",
    },
}

# ── Apply CoM centering ─────────────────────────────────────────────────────────
for name, ic in ICS.items():
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]
    M  = m.sum()
    ic["x0"] = x0 - (m[:, None] * x0).sum(0) / M
    ic["v0"] = v0 - (m[:, None] * v0).sum(0) / M

# ── Run simulations ────────────────────────────────────────────────────────────
print(f"\n[multi_ic] Running 4 ICs  |  dt={DT_REF}  |  T={T}yr  |  n_samples={N_SAMPLES}")
print(f"{'='*65}")

results = {}

for name, ic in ICS.items():
    print(f"\n[{ic['label']}]")
    print(f"  {ic['desc']}")
    m  = ic["m"]
    x0 = ic["x0"]
    v0 = ic["v0"]

    # Compute total energy to verify bound system
    KE = 0.5 * np.sum(m[:, None] * v0**2)
    PE = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(x0[i] - x0[j])
            PE -= cfg.G * m[i] * m[j] / r
    E_total = KE + PE
    print(f"  Total energy E = {E_total:.4f}  ({'BOUND' if E_total < 0 else 'UNBOUND!'})")

    # ias15 ground truth
    t0 = time.perf_counter()
    tr, pr, vr, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, T, N_SAMPLES)
    t_ias = time.perf_counter() - t0
    print(f"  ias15:   {t_ias:.3f}s")

    # SIMON
    t0 = time.perf_counter()
    tm, pm, vm, perf_m = simulate_leapfrog_hybrid(x0, v0, m, model, cfg, DT_REF, T, N_SAMPLES)
    t_simon = time.perf_counter() - t0
    speedup = t_ias / max(perf_m["total_time_sec"], 1e-12)
    print(f"  SIMON:   {t_simon:.3f}s  speedup={speedup:.2f}x  "
          f"NN_frac={perf_m['avg_fallback_frac']:.4f}")

    # Metrics
    delta = rms_sep(pm, pr)
    slope, win = fit_log_slope(tr, delta)
    final_rms  = float(delta[-1])
    bounded    = final_rms < EJECTION_THRESHOLD

    print(f"  lambda:  {slope:.4f}/yr  |  final_RMS: {final_rms:.4f} AU  |  "
          f"bounded: {'YES' if bounded else 'NO (ejection)'}")

    results[name] = {
        "ic":        ic,
        "tr":        tr,
        "pr":        pr,
        "pm":        pm,
        "delta":     delta,
        "slope":     slope,
        "win":       win,
        "final_rms": final_rms,
        "bounded":   bounded,
        "speedup":   speedup,
        "nn_frac":   perf_m["avg_fallback_frac"],
        "E_total":   E_total,
    }

# ── Statistics ─────────────────────────────────────────────────────────────────
all_slopes  = [r["slope"]     for r in results.values()]
all_rms     = [r["final_rms"] for r in results.values()]
all_bounded = [r["bounded"]   for r in results.values()]
all_speedup = [r["speedup"]   for r in results.values()]

# Only include bounded ICs in stats
bounded_slopes = [s for s, b in zip(all_slopes, all_bounded) if b]
bounded_rms    = [r for r, b in zip(all_rms,    all_bounded) if b]

mean_slope  = np.mean(bounded_slopes) if bounded_slopes else float("nan")
std_slope   = np.std(bounded_slopes)  if bounded_slopes else float("nan")
mean_rms    = np.mean(bounded_rms)    if bounded_rms    else float("nan")
std_rms     = np.std(bounded_rms)     if bounded_rms    else float("nan")

print(f"\n{'='*65}")
print("MULTI-IC SUMMARY")
print(f"{'='*65}")
print(f"  {'IC':<22} {'lambda(/yr)':>12} {'final_RMS(AU)':>14} {'bounded':>9} {'speedup':>9}")
print(f"  {'-'*68}")
for name, r in results.items():
    b = "YES" if r["bounded"] else "NO"
    print(f"  {r['ic']['label']:<22} {r['slope']:>12.4f} {r['final_rms']:>14.4f} "
          f"{b:>9} {r['speedup']:>8.2f}x")
print(f"  {'-'*68}")
print(f"  {'Mean (bounded only)':<22} {mean_slope:>12.4f} {mean_rms:>14.4f}")
print(f"  {'Std  (bounded only)':<22} {std_slope:>12.4f} {std_rms:>14.4f}")
print(f"  {'Frac bounded':<22} {sum(all_bounded)}/{len(all_bounded)}")

# ── Figure 1: Divergence curves (4-panel) ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()

for ax, (name, r) in zip(axes, results.items()):
    col   = r["ic"]["color"]
    label = r["ic"]["label"]
    ax.semilogy(r["tr"], r["delta"], color=col, lw=1.8, alpha=0.9)
    ax.axvspan(r["win"][0], r["win"][1], alpha=0.12, color="gray", label="fit window")
    status = "BOUNDED" if r["bounded"] else "EJECTION"
    status_col = "#166534" if r["bounded"] else "#DC2626"
    ax.set_title(f"{label}\n\u03bb = {r['slope']:.4f}/yr  |  "
                 f"final RMS = {r['final_rms']:.2f} AU  |  "
                 f"[{status}]", color=status_col, fontsize=10)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error (AU)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("SIMON Divergence Rate Across 4 Initial Conditions  (T = 100 yr, dt = 0.04 yr)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
div_path = os.path.join(OUT_DIR, "multi_ic_divergence.png")
plt.savefig(div_path, dpi=300)
plt.close()
print(f"\n[multi_ic] Saved {div_path}")

# ── Figure 2: Summary bar charts ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
names  = [r["ic"]["label"].replace(": ", "\n") for r in results.values()]
colors = [r["ic"]["color"] for r in results.values()]
slopes = [r["slope"]     for r in results.values()]
rmss   = [r["final_rms"] for r in results.values()]
speeds = [r["speedup"]   for r in results.values()]

# Panel 1: lambda
ax = axes[0]
bars = ax.bar(names, slopes, color=colors, edgecolor="white", linewidth=1.0, width=0.5)
ax.axhline(mean_slope, color="black", lw=1.5, linestyle="--",
           label=f"Mean = {mean_slope:.4f}/yr")
for bar, val in zip(bars, slopes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylabel("\u03bb (1/yr)")
ax.set_title(f"Divergence Rate \u03bb\nMean = {mean_slope:.4f} \u00b1 {std_slope:.4f}/yr")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 2: final RMS (log scale)
ax = axes[1]
bars2 = ax.bar(names, rmss, color=colors, edgecolor="white", linewidth=1.0, width=0.5)
ax.set_yscale("log")
for bar, val, b in zip(bars2, rmss, all_bounded):
    col = "#166534" if b else "#DC2626"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=col)
ax.axhline(EJECTION_THRESHOLD, color=EJECTION_THRESHOLD and "#DC2626",
           lw=1.2, linestyle=":", label=f"Ejection threshold ({EJECTION_THRESHOLD} AU)")
ax.set_ylabel("Final RMS error (AU, log scale)")
ax.set_title(f"Final RMS Error at T=100yr\n({sum(all_bounded)}/{len(all_bounded)} bounded)")
ax.legend(fontsize=9); ax.grid(True, which="both", axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 3: speedup
ax = axes[2]
bars3 = ax.bar(names, speeds, color=colors, edgecolor="white", linewidth=1.0, width=0.5)
ax.axhline(1.0, color="black", lw=1.2, linestyle="--", label="ias15 (1.00x)")
for bar, val in zip(bars3, speeds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.2f}x", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylabel("Speedup vs ias15")
ax.set_title(f"Computational Speedup\n(dt = {DT_REF} yr)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("SIMON Performance Summary Across 4 Initial Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
res_path = os.path.join(OUT_DIR, "multi_ic_results.png")
plt.savefig(res_path, dpi=300)
plt.close()
print(f"[multi_ic] Saved {res_path}")

# ── Figure 3: Trajectory overlays (4-panel, body 1 only for clarity) ──────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, (name, r) in zip(axes, results.items()):
    col   = r["ic"]["color"]
    label = r["ic"]["label"]
    pr = r["pr"]; pm = r["pm"]
    # Show body 1 (middle mass) — most informative for all ICs
    ax.plot(pr[:, 1, 0], pr[:, 1, 1], "-",  color="#2563A6", lw=1.5, alpha=0.9,
            label="ias15")
    ax.plot(pm[:, 1, 0], pm[:, 1, 1], "--", color=col, lw=1.2, alpha=0.9,
            label="SIMON")
    status = "BOUNDED" if r["bounded"] else "EJECTION"
    status_col = "#166534" if r["bounded"] else "#DC2626"
    ax.set_title(f"{label}  [{status}]", color=status_col, fontsize=10)
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

fig.suptitle("Body 1 Trajectory Overlay (ias15 vs SIMON)  —  4 Initial Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout()
traj_path = os.path.join(OUT_DIR, "multi_ic_trajectories.png")
plt.savefig(traj_path, dpi=300)
plt.close()
print(f"[multi_ic] Saved {traj_path}")

# ── Summary text file ──────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "multi_ic_summary.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 68 + "\n")
    f.write("MULTI-INITIAL-CONDITION EVALUATION SUMMARY\n")
    f.write(f"SIMON  |  T={T}yr  |  dt={DT_REF}yr  |  n_samples={N_SAMPLES}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write("=" * 68 + "\n\n")

    f.write("INITIAL CONDITIONS:\n")
    for name, ic in ICS.items():
        r = results[name]
        f.write(f"\n  {ic['label']}\n")
        f.write(f"    {ic['desc']}\n")
        f.write(f"    m  = {ic['m'].tolist()}\n")
        f.write(f"    E_total = {r['E_total']:.4f}  (negative = bound)\n")

    f.write("\n\nRESULTS:\n")
    f.write(f"\n  {'IC':<24} {'lambda(/yr)':>12} {'final_RMS':>12} "
            f"{'bounded':>9} {'speedup':>9} {'NN_frac':>9}\n")
    f.write(f"  {'-'*77}\n")
    for name, r in results.items():
        b = "YES" if r["bounded"] else "NO"
        f.write(f"  {r['ic']['label']:<24} {r['slope']:>12.4f} "
                f"{r['final_rms']:>12.4f} {b:>9} "
                f"{r['speedup']:>8.2f}x {r['nn_frac']:>9.4f}\n")
    f.write(f"  {'-'*77}\n")
    f.write(f"  {'Mean (bounded only)':<24} {mean_slope:>12.4f} {mean_rms:>12.4f}\n")
    f.write(f"  {'Std  (bounded only)':<24} {std_slope:>12.4f} {std_rms:>12.4f}\n\n")

    f.write(f"  Fraction bounded: {sum(all_bounded)}/{len(all_bounded)}\n\n")

    f.write("=" * 68 + "\n")
    f.write("KEY NUMBERS FOR PAPER:\n")
    f.write(f"  lambda range:  [{min(bounded_slopes):.4f}, {max(bounded_slopes):.4f}] /yr\n")
    f.write(f"  lambda mean:   {mean_slope:.4f} +/- {std_slope:.4f} /yr\n")
    f.write(f"  RMS range:     [{min(bounded_rms):.3f}, {max(bounded_rms):.3f}] AU\n")
    f.write(f"  Bodies bounded: {sum(all_bounded)}/{len(all_bounded)} ICs\n")
    f.write(f"  Speedup range: [{min(all_speedup):.2f}x, {max(all_speedup):.2f}x]\n")

print(f"[multi_ic] Saved {txt_path}")
print(f"\n{'='*65}")
print("[multi_ic] Done.")
print(f"  lambda: {mean_slope:.4f} +/- {std_slope:.4f} /yr  (bounded ICs only)")
print(f"  bounded: {sum(all_bounded)}/{len(all_bounded)} ICs")
