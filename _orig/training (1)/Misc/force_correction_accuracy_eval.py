# force_correction_accuracy_eval.py
#
# Experiment purpose:
#   Directly test whether SIMON's scalar NN correction improves local force accuracy
#   in the close-encounter regime where the NN is actually invoked.
#
# This experiment compares:
#   1. Exact Newtonian force magnitude
#   2. No-NN softened force magnitude
#   3. SIMON NN-corrected softened force magnitude
#
# This is NOT a trajectory rollout.
# It is a force-level validation designed to isolate the NN's accuracy contribution
# without contamination from chaotic long-horizon divergence.
#
# Required file in same folder:
#   pair_correction_nn.pt

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "pair_correction_nn.pt"
OUT_DIR = "force_correction_accuracy_out"

EPS = 3e-4
G = 1.0

# Same close-pair threshold used in SIMON:
# NN is invoked only when r < 500*eps = 0.15 AU.
NN_THRESHOLD = 500.0 * EPS

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# SIMON scalar correction model
# -----------------------------------------------------------------------------
class PairCorrectionNN(nn.Module):
    def __init__(self, hidden=32, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("input_mean", torch.zeros(3))
        self.register_buffer("input_std", torch.ones(3))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)
# -----------------------------------------------------------------------------
# Load trained SIMON model
# -----------------------------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put this script in the same folder "
            f"as pair_correction_nn.pt or update MODEL_PATH."
        )

    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Force magnitude models
# -----------------------------------------------------------------------------
def exact_newtonian_force_mag(r, mi, mj):
    """
    Exact Newtonian pairwise force magnitude:
        F = G mi mj / r^2
    """
    return G * mi * mj / (r**2)


def softened_force_mag(r, mi, mj):
    """
    No-NN softened force magnitude:
        F_soft = G mi mj r / (r^2 + eps^2)^(3/2)

    This is the magnitude of:
        G mi mj / r_soft^3 * r_vec
    """
    return G * mi * mj * r / ((r**2 + EPS**2) ** 1.5)


def simon_corrected_force_mag(model, r, mi, mj):
    """
    SIMON NN-corrected softened force magnitude:
        F_SIMON = c * F_soft

    where the NN predicts log(c) from:
        [log(r_soft), log(mi), log(mj)]
    """
    r_soft = np.sqrt(r**2 + EPS**2)

    features = np.stack(
        [
            np.log(r_soft),
            np.full_like(r, np.log(mi)),
            np.full_like(r, np.log(mj)),
        ],
        axis=1,
    ).astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        log_c = model(x).cpu().numpy()

    c = np.exp(log_c)
    return c * softened_force_mag(r, mi, mj), c

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_force_error(r_values, rel_err_soft, rel_err_simon, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    # -------------------------------------------------------------------------
    # Key regime markers
    # -------------------------------------------------------------------------
    safety_cutoff = 5e-4
    useful_cutoff = 10.0 * EPS       # r/eps = 10
    far_field_cutoff = NN_THRESHOLD  # 500*eps = 0.15 AU

    x_min = np.min(r_values)
    x_max = np.max(r_values)

    # Show only the main "NN useful" region as a light guide.
    # This avoids clutter while highlighting where the correction actually matters.
    ax.axvspan(
        safety_cutoff,
        useful_cutoff,
        alpha=0.12,
        color="green",
        label=r"NN useful region: $5\times10^{-4}\leq r \leq 10\epsilon$",
    )

    # -------------------------------------------------------------------------
    # Force-error curves
    # -------------------------------------------------------------------------
    ax.loglog(
        r_values,
        rel_err_soft,
        "--",
        lw=2.1,
        color="tab:blue",
        label="No-NN softened force",
    )

    ax.loglog(
        r_values,
        rel_err_simon,
        "-",
        lw=2.1,
        color="tab:orange",
        label="SIMON NN-corrected force",
    )

    # -------------------------------------------------------------------------
    # Vertical threshold lines
    # -------------------------------------------------------------------------
    ax.axvline(
        EPS,
        linestyle=":",
        lw=1.3,
        color="black",
        alpha=0.70,
    )

    ax.axvline(
        safety_cutoff,
        linestyle="-.",
        lw=1.4,
        color="black",
        alpha=0.75,
    )

    ax.axvline(
        useful_cutoff,
        linestyle=":",
        lw=1.3,
        color="black",
        alpha=0.70,
    )

    ax.axvline(
        far_field_cutoff,
        linestyle=":",
        lw=1.3,
        color="black",
        alpha=0.70,
    )

    # -------------------------------------------------------------------------
    # Small text labels for thresholds
    # -------------------------------------------------------------------------
    label_y = 1.35

    ax.text(
        EPS,
        label_y,
        r"$\epsilon$",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
    )

    ax.text(
        safety_cutoff,
        label_y,
        r"$r_{\rm soft,min}$",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
    )

    ax.text(
        useful_cutoff,
        label_y,
        r"$10\epsilon$",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
    )

    ax.text(
        far_field_cutoff * 0.92,
        label_y,
        r"$500\epsilon$",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
    )

    # Add two short regime annotations instead of a large coloured-region legend.
    ax.text(
        1.15e-3,
        2.8e-2,
        "correction\ncan matter",
        fontsize=8,
        ha="center",
        va="center",
    )

    ax.text(
        3.2e-2,
        2.8e-2,
        "NN eligible,\nbut correction usually tiny",
        fontsize=8,
        ha="center",
        va="center",
    )

    # -------------------------------------------------------------------------
    # Axes and legend
    # -------------------------------------------------------------------------
    ax.set_xlabel("Pair separation r (AU)")
    ax.set_ylabel("Relative force-magnitude error")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1e-5, 2.0)

    ax.grid(True, which="both", alpha=0.22)

    ax.legend(
        framealpha=0.90,
        fontsize=8,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    print("[force_correction_accuracy_eval] Loading SIMON model ...")
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {MODEL_PATH} with {n_params} parameters")

    # Representative mass pairs inside SIMON's training range.
    # The analytic correction is mostly separation-dependent, so this checks
    # that the trained NN does not become sensitive to a single mass scale.
    mass_pairs = [
        (1.0, 0.01),    # main paper mass scale
        (1.0, 0.5),     # compact stellar binary scale
        (0.7, 0.4),     # Gliese-like stellar scale
    ]

    # Use the first pair for the main figure and detailed CSV.
    mi = mass_pairs[0][0]
    mj = mass_pairs[0][1]

    # Sample separations across and below the NN activation threshold.
    # The dense log spacing focuses on the close-encounter regime.
    r_values = np.logspace(
        np.log10(1e-5),
        np.log10(0.3),
        1200,
        dtype=np.float64,
    )

    F_exact = exact_newtonian_force_mag(r_values, mi, mj)
    F_soft = softened_force_mag(r_values, mi, mj)
    F_simon, c_pred = simon_corrected_force_mag(model, r_values, mi, mj)

    # Relative force magnitude errors against exact Newtonian force.
    rel_err_soft = np.abs(F_soft - F_exact) / np.maximum(np.abs(F_exact), 1e-30)
    rel_err_simon = np.abs(F_simon - F_exact) / np.maximum(np.abs(F_exact), 1e-30)

    # Main close regime where the NN is invoked in SIMON.
    close_mask = r_values < NN_THRESHOLD

    # Operational correction regime:
    # exclude extremely tiny separations below r_soft_min, where SIMON's safety
    # fallback/gating logic becomes important in actual rollouts.
    r_soft_min = 5e-4
    operational_mask = (r_values >= r_soft_min) & (r_values < NN_THRESHOLD)

    # Transition regime where softening is significant but not extreme.
    transition_mask = (r_values >= EPS) & (r_values < NN_THRESHOLD)

    mean_soft_close = float(np.mean(rel_err_soft[close_mask]))
    mean_simon_close = float(np.mean(rel_err_simon[close_mask]))

    median_soft_close = float(np.median(rel_err_soft[close_mask]))
    median_simon_close = float(np.median(rel_err_simon[close_mask]))

    max_soft_close = float(np.max(rel_err_soft[close_mask]))
    max_simon_close = float(np.max(rel_err_simon[close_mask]))

    mean_soft_operational = float(np.mean(rel_err_soft[operational_mask]))
    mean_simon_operational = float(np.mean(rel_err_simon[operational_mask]))

    median_soft_operational = float(np.median(rel_err_soft[operational_mask]))
    median_simon_operational = float(np.median(rel_err_simon[operational_mask]))

    max_soft_operational = float(np.max(rel_err_soft[operational_mask]))
    max_simon_operational = float(np.max(rel_err_simon[operational_mask]))

    improvement_mean_operational = mean_soft_operational / max(mean_simon_operational, 1e-30)
    improvement_median_operational = median_soft_operational / max(median_simon_operational, 1e-30)

    mean_soft_transition = float(np.mean(rel_err_soft[transition_mask]))
    mean_simon_transition = float(np.mean(rel_err_simon[transition_mask]))

    median_soft_transition = float(np.median(rel_err_soft[transition_mask]))
    median_simon_transition = float(np.median(rel_err_simon[transition_mask]))

    improvement_mean_transition = mean_soft_transition / max(mean_simon_transition, 1e-30)
    improvement_median_transition = median_soft_transition / max(median_simon_transition, 1e-30)

    improvement_mean = mean_soft_close / max(mean_simon_close, 1e-30)
    improvement_median = median_soft_close / max(median_simon_close, 1e-30)

    print("\n[close regime: r < 0.15 AU]")
    print(f"  Mean relative error, No-NN softened: {mean_soft_close:.6e}")
    print(f"  Mean relative error, SIMON corrected: {mean_simon_close:.6e}")
    print(f"  Mean error reduction factor: {improvement_mean:.2f}x")
    print()
    print(f"  Median relative error, No-NN softened: {median_soft_close:.6e}")
    print(f"  Median relative error, SIMON corrected: {median_simon_close:.6e}")
    print(f"  Median error reduction factor: {improvement_median:.2f}x")
    print()
    print(f"  Max relative error, No-NN softened: {max_soft_close:.6e}")
    print(f"  Max relative error, SIMON corrected: {max_simon_close:.6e}")

    print("\n[operational correction regime: 5e-4 AU <= r < 0.15 AU]")
    print(f"  Mean relative error, No-NN softened: {mean_soft_operational:.6e}")
    print(f"  Mean relative error, SIMON corrected: {mean_simon_operational:.6e}")
    print(f"  Mean error reduction factor: {improvement_mean_operational:.2f}x")
    print()
    print(f"  Median relative error, No-NN softened: {median_soft_operational:.6e}")
    print(f"  Median relative error, SIMON corrected: {median_simon_operational:.6e}")
    print(f"  Median error reduction factor: {improvement_median_operational:.2f}x")
    print()
    print(f"  Max relative error, No-NN softened: {max_soft_operational:.6e}")
    print(f"  Max relative error, SIMON corrected: {max_simon_operational:.6e}")

    print("\n[transition regime: eps <= r < 0.15 AU]")
    print(f"  Mean relative error, No-NN softened: {mean_soft_transition:.6e}")
    print(f"  Mean relative error, SIMON corrected: {mean_simon_transition:.6e}")
    print(f"  Mean error reduction factor: {improvement_mean_transition:.2f}x")
    print()
    print(f"  Median relative error, No-NN softened: {median_soft_transition:.6e}")
    print(f"  Median relative error, SIMON corrected: {median_simon_transition:.6e}")
    print(f"  Median error reduction factor: {improvement_median_transition:.2f}x")

    plot_path = os.path.join(OUT_DIR, "force_correction_relative_error.png")
    plot_force_error(r_values, rel_err_soft, rel_err_simon, plot_path)
    print(f"Saved plot to {plot_path}")

    plot_pdf_path = os.path.join(OUT_DIR, "force_correction_relative_error.pdf")
    plot_force_error(r_values, rel_err_soft, rel_err_simon, plot_pdf_path)
    print(f"Saved plot to {plot_pdf_path}")

    csv_path = os.path.join(OUT_DIR, "force_correction_accuracy_data.csv")
    data = np.column_stack(
        [
            r_values,
            F_exact,
            F_soft,
            F_simon,
            c_pred,
            rel_err_soft,
            rel_err_simon,
        ]
    )
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "r_AU,F_exact,F_no_nn_softened,F_simon_corrected,"
            "c_pred,rel_err_no_nn_softened,rel_err_simon_corrected"
        ),
        comments="",
    )
    
    print(f"Saved data to {csv_path}")

    # -------------------------------------------------------------------------
    # Multi-mass robustness check
    # -------------------------------------------------------------------------
    mass_pair_rows = []

    for mi_test, mj_test in mass_pairs:
        F_exact_test = exact_newtonian_force_mag(r_values, mi_test, mj_test)
        F_soft_test = softened_force_mag(r_values, mi_test, mj_test)
        F_simon_test, _ = simon_corrected_force_mag(model, r_values, mi_test, mj_test)

        rel_err_soft_test = np.abs(F_soft_test - F_exact_test) / np.maximum(
            np.abs(F_exact_test), 1e-30
        )
        rel_err_simon_test = np.abs(F_simon_test - F_exact_test) / np.maximum(
            np.abs(F_exact_test), 1e-30
        )

        mean_soft_op = float(np.mean(rel_err_soft_test[operational_mask]))
        mean_simon_op = float(np.mean(rel_err_simon_test[operational_mask]))
        median_soft_op = float(np.median(rel_err_soft_test[operational_mask]))
        median_simon_op = float(np.median(rel_err_simon_test[operational_mask]))
        max_soft_op = float(np.max(rel_err_soft_test[operational_mask]))
        max_simon_op = float(np.max(rel_err_simon_test[operational_mask]))

        mass_pair_rows.append(
            {
                "mi": mi_test,
                "mj": mj_test,
                "mean_soft": mean_soft_op,
                "mean_simon": mean_simon_op,
                "mean_reduction": mean_soft_op / max(mean_simon_op, 1e-30),
                "median_soft": median_soft_op,
                "median_simon": median_simon_op,
                "median_reduction": median_soft_op / max(median_simon_op, 1e-30),
                "max_soft": max_soft_op,
                "max_simon": max_simon_op,
            }
        )

    print("\n[multi-mass robustness: operational correction regime]")
    for row in mass_pair_rows:
        print(
            f"  mi={row['mi']:.3g}, mj={row['mj']:.3g}: "
            f"mean reduction={row['mean_reduction']:.2f}x, "
            f"max error {row['max_soft']:.3e} -> {row['max_simon']:.3e}"
        )

    mass_csv_path = os.path.join(OUT_DIR, "force_correction_mass_robustness.csv")
    mass_data = np.array(
        [
            [
                row["mi"],
                row["mj"],
                row["mean_soft"],
                row["mean_simon"],
                row["mean_reduction"],
                row["median_soft"],
                row["median_simon"],
                row["median_reduction"],
                row["max_soft"],
                row["max_simon"],
            ]
            for row in mass_pair_rows
        ],
        dtype=np.float64,
    )
    np.savetxt(
        mass_csv_path,
        mass_data,
        delimiter=",",
        header=(
            "mi,mj,mean_rel_err_no_nn,mean_rel_err_simon,mean_reduction,"
            "median_rel_err_no_nn,median_rel_err_simon,median_reduction,"
            "max_rel_err_no_nn,max_rel_err_simon"
        ),
        comments="",
    )
    print(f"Saved mass robustness data to {mass_csv_path}")

    summary_path = os.path.join(OUT_DIR, "force_correction_accuracy_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FORCE-LEVEL NN CORRECTION ACCURACY VALIDATION\n")
        f.write("=" * 72 + "\n\n")
        f.write("Purpose:\n")
        f.write(
            "This experiment directly compares local pairwise force magnitude "
            "accuracy in the close-encounter regime where SIMON invokes the NN.\n\n"
        )
        f.write(f"Masses tested: mi={mi}, mj={mj}\n")
        f.write(f"Softening eps={EPS}\n")
        f.write(f"NN threshold={NN_THRESHOLD} AU\n\n")
        f.write("Close regime: r < 0.15 AU\n")
        f.write(f"Mean relative error, No-NN softened: {mean_soft_close:.8e}\n")
        f.write(f"Mean relative error, SIMON corrected: {mean_simon_close:.8e}\n")
        f.write(f"Mean error reduction factor: {improvement_mean:.4f}x\n\n")
        f.write(f"Median relative error, No-NN softened: {median_soft_close:.8e}\n")
        f.write(f"Median relative error, SIMON corrected: {median_simon_close:.8e}\n")
        f.write(f"Median error reduction factor: {improvement_median:.4f}x\n\n")
        f.write(f"Max relative error, No-NN softened: {max_soft_close:.8e}\n")
        f.write(f"Max relative error, SIMON corrected: {max_simon_close:.8e}\n")

        f.write("\n" + "-" * 72 + "\n")
        f.write("Operational correction regime: 5e-4 AU <= r < 0.15 AU\n")
        f.write(
            "This excludes the extreme safety-fallback region and corresponds "
            "to the regime where the NN correction is intended to operate.\n\n"
        )
        f.write(f"Mean relative error, No-NN softened: {mean_soft_operational:.8e}\n")
        f.write(f"Mean relative error, SIMON corrected: {mean_simon_operational:.8e}\n")
        f.write(f"Mean error reduction factor: {improvement_mean_operational:.4f}x\n\n")
        f.write(f"Median relative error, No-NN softened: {median_soft_operational:.8e}\n")
        f.write(f"Median relative error, SIMON corrected: {median_simon_operational:.8e}\n")
        f.write(f"Median error reduction factor: {improvement_median_operational:.4f}x\n\n")
        f.write(f"Max relative error, No-NN softened: {max_soft_operational:.8e}\n")
        f.write(f"Max relative error, SIMON corrected: {max_simon_operational:.8e}\n")

        f.write("\n" + "-" * 72 + "\n")
        f.write("Transition regime: eps <= r < 0.15 AU\n")
        f.write(
            "This regime begins at the softening length eps and measures the "
            "region where softening error is significant but not dominated by "
            "extreme near-zero separations.\n\n"
        )
        f.write(f"Mean relative error, No-NN softened: {mean_soft_transition:.8e}\n")
        f.write(f"Mean relative error, SIMON corrected: {mean_simon_transition:.8e}\n")
        f.write(f"Mean error reduction factor: {improvement_mean_transition:.4f}x\n\n")
        
        f.write(f"Median relative error, No-NN softened: {median_soft_transition:.8e}\n")
        f.write(f"Median relative error, SIMON corrected: {median_simon_transition:.8e}\n")
        f.write(f"Median error reduction factor: {improvement_median_transition:.4f}x\n")

        f.write("\n" + "-" * 72 + "\n")
        f.write("Multi-mass robustness: operational correction regime\n")
        f.write("Regime: 5e-4 AU <= r < 0.15 AU\n\n")
        f.write(
            "mi,mj,mean_no_nn,mean_simon,mean_reduction,"
            "median_no_nn,median_simon,median_reduction,"
            "max_no_nn,max_simon\n"
        )

        for row in mass_pair_rows:
            f.write(
                f"{row['mi']:.8e},"
                f"{row['mj']:.8e},"
                f"{row['mean_soft']:.8e},"
                f"{row['mean_simon']:.8e},"
                f"{row['mean_reduction']:.4f}x,"
                f"{row['median_soft']:.8e},"
                f"{row['median_simon']:.8e},"
                f"{row['median_reduction']:.4f}x,"
                f"{row['max_soft']:.8e},"
                f"{row['max_simon']:.8e}\n"
            )

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()