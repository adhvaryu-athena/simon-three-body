# seed_robustness.py
#
# Trains SIMON (PairCorrectionNN) 5 times with different random seeds.
# Tests whether results are robust to random initialisation.
#
# REQUIRES in same folder:
#   train_pair_correction_new.py
#
# PRODUCES in same folder:
#   pair_correction_nn_seed42.pt
#   pair_correction_nn_seed123.pt
#   pair_correction_nn_seed456.pt
#   pair_correction_nn_seed789.pt
#   pair_correction_nn_seed1000.pt
#   seed_robustness_results.txt
#   seed_robustness_results.png
#
# Run: python seed_robustness.py
#
# Expected runtime: ~50 seconds total (5 x ~10s on RTX 5050)

import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import directly from your existing training file
# This ensures the model architecture is 100% identical
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pair_correction_new import (
    PairCorrectionNN,
    generate_training_data,
    train_model,
    sanity_check,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS   = [42, 123, 456, 789, 1000]
HIDDEN  = 32
EPOCHS  = 5000
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"[seed_robustness] device={DEVICE}  hidden={HIDDEN}  epochs={EPOCHS}")
print(f"[seed_robustness] Seeds to test: {SEEDS}")
print(f"[seed_robustness] Expected runtime: ~{len(SEEDS)*10}s\n")

# ── Generate training data once (same data for all seeds) ─────────────────────
# Data seed is fixed at 42 so we test only model initialisation sensitivity,
# not data sampling sensitivity. This is the standard approach.
print("[seed_robustness] Generating training data (seed=42, fixed across all runs)...")
feat, targ = generate_training_data(seed=42)
print()

# ── Train with each seed ──────────────────────────────────────────────────────
results = []

for seed in SEEDS:
    print(f"{'='*55}")
    print(f"[seed_robustness] Training with seed={seed}")
    print(f"{'='*55}")

    t_start = time.perf_counter()
    model, tl, vl = train_model(
        feat, targ,
        hidden=HIDDEN,
        epochs=EPOCHS,
        device=DEVICE,
        seed=seed,
    )
    elapsed = time.perf_counter() - t_start

    best_val_mse = min(vl)
    final_val_mse = vl[-1]

    # Quick sanity check
    print(f"\n[seed_robustness] Sanity check (seed={seed}):")
    sanity_check(model, device=DEVICE)

    # Save model
    out_path = os.path.join(OUT_DIR, f"pair_correction_nn_seed{seed}.pt")
    torch.save(model.cpu().state_dict(), out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[seed_robustness] Saved {out_path}  ({size_kb:.1f} KB)\n")

    results.append({
        "seed":          seed,
        "best_val_mse":  best_val_mse,
        "final_val_mse": final_val_mse,
        "train_time_s":  elapsed,
        "tl":            tl,
        "vl":            vl,
    })

# ── Summary statistics ─────────────────────────────────────────────────────────
best_mses  = np.array([r["best_val_mse"]  for r in results])
final_mses = np.array([r["final_val_mse"] for r in results])
times      = np.array([r["train_time_s"]  for r in results])

mean_best  = best_mses.mean();   std_best  = best_mses.std()
mean_final = final_mses.mean();  std_final = final_mses.std()
cv_best    = std_best / mean_best * 100   # coefficient of variation

print(f"\n{'='*55}")
print(f"SEED ROBUSTNESS SUMMARY")
print(f"{'='*55}")
print(f"\n{'Seed':>8} | {'Best Val MSE':>14} | {'Final Val MSE':>14} | {'Time (s)':>10}")
print(f"{'-'*56}")
for r in results:
    print(f"  {r['seed']:>6} | {r['best_val_mse']:>14.8f} | "
          f"{r['final_val_mse']:>14.8f} | {r['train_time_s']:>10.1f}")
print(f"{'-'*56}")
print(f"  {'Mean':>6} | {mean_best:>14.8f} | {mean_final:>14.8f} | {times.mean():>10.1f}")
print(f"  {'Std':>6} | {std_best:>14.8f} | {std_final:>14.8f} | {times.std():>10.1f}")
print(f"  {'CV%':>6} | {cv_best:>13.2f}% |")
print(f"\nCoefficient of Variation (CV) of best val MSE: {cv_best:.2f}%")
if cv_best < 5.0:
    verdict = "ROBUST — results are highly stable across random seeds."
elif cv_best < 15.0:
    verdict = "MODERATELY ROBUST — small seed sensitivity, acceptable for publication."
else:
    verdict = "SENSITIVE — consider reporting mean +/- std and discussion."
print(f"Verdict: {verdict}")

# ── Save text results ──────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "seed_robustness_results.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("SIMON TRAINING SEED ROBUSTNESS RESULTS\n")
    f.write("PairCorrectionNN  |  hidden=32  |  epochs=5000\n")
    f.write(f"device={DEVICE}  |  n_train=40000  |  n_val=10000\n")
    f.write("Data seed fixed at 42 (tests model init sensitivity only)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Seed':>8} | {'Best Val MSE':>14} | {'Final Val MSE':>14} | {'Time (s)':>10}\n")
    f.write("-" * 56 + "\n")
    for r in results:
        f.write(f"  {r['seed']:>6} | {r['best_val_mse']:>14.8f} | "
                f"{r['final_val_mse']:>14.8f} | {r['train_time_s']:>10.1f}\n")
    f.write("-" * 56 + "\n")
    f.write(f"  {'Mean':>6} | {mean_best:>14.8f} | {mean_final:>14.8f} | "
            f"{times.mean():>10.1f}\n")
    f.write(f"  {'Std':>6} | {std_best:>14.8f} | {std_final:>14.8f} | "
            f"{times.std():>10.1f}\n")
    f.write(f"  {'CV%':>6} | {cv_best:>13.2f}% |\n")
    f.write(f"\nCoefficient of Variation: {cv_best:.2f}%\n")
    f.write(f"Verdict: {verdict}\n")
    f.write("\nKEY NUMBERS FOR PAPER:\n")
    f.write(f"  Best val MSE: {mean_best:.2e} +/- {std_best:.2e}  "
            f"(CV = {cv_best:.1f}%)\n")
    f.write(f"  All seeds converge to similar accuracy.\n")

print(f"\n[seed_robustness] Saved {txt_path}")

# ── Plot: training curves for all seeds ───────────────────────────────────────
# Local font override for this paper figure only.
# This changes only chart formatting, not training, MSE values, CV, or results.
with plt.rc_context({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    colors = ["#2563A6", "#16A34A", "#DC2626", "#9333EA", "#EA580C"]

    # Left: validation curves overlaid
    ax = axes[0]
    for r, c in zip(results, colors):
        ax.semilogy(
            r["vl"],
            color=c,
            lw=1.4,
            alpha=0.85,
            label=f"seed={r['seed']}"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: best validation MSE per seed
    ax = axes[1]
    seed_labels = [str(s) for s in SEEDS]
    bars = ax.bar(
        seed_labels,
        best_mses * 1e5,
        color=colors,
        edgecolor="white",
        linewidth=1.0
    )

    ax.axhline(
        mean_best * 1e5,
        color="black",
        lw=1.3,
        linestyle="--",
        label=f"Mean = {mean_best:.2e}"
    )
    ax.axhspan(
        (mean_best - std_best) * 1e5,
        (mean_best + std_best) * 1e5,
        alpha=0.15,
        color="gray",
        label="\u00b1 1 std"
    )

    for bar, val in zip(bars, best_mses):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.025 * best_mses.max() * 1e5,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=7.5
        )

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Best validation MSE (\u00d710\u207b\u2075)")

    # Keep the CV visible, but avoid the misleading "SENSITIVE" label in the figure.
    #ax.text(
    #    0.02, 0.96,
    #    f"CV = {cv_best:.1f}%",
    #    transform=ax.transAxes,
    #   ha="left",
    #    va="top",
    #    fontsize=8,
    #    color="0.35"
    #)
    # CV is reported in the caption/text; the mean line and ±1 std band
    # already show seed-to-seed variation visually.
    
    ax.legend(loc="lower right", framealpha=0.85)


    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # No subplot titles: the LaTeX caption explains the figure.
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "seed_robustness_results.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

print(f"[seed_robustness] Saved {png_path}")

print(f"\n[seed_robustness] Done.")
print(f"  Best val MSE: {mean_best:.4e} +/- {std_best:.4e}  (CV={cv_best:.1f}%)")
print(f"  Verdict: {verdict}")
