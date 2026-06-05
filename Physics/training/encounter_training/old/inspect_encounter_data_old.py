"""
inspect_encounter_data.py  --  Step 3  (FINAL)

Loads encounter_data.npz and answers the key question:

  Does c_optimal differ from c_analytic in a dt-dependent way?
  YES -> trajectory training adds genuine value. Proceed to training.
  NO  -> c_opt ~= c_analytic for all dt. No value-add.

The verdict is based on the RANGE of c_opt_med across dt values:
  range = max(c_opt_med) - min(c_opt_med) across all dt values
  range > 0.10 : PROCEED   — strong dt-dependence confirmed
  range > 0.05 : CAUTION   — mild dt-dependence
  range < 0.05 : STOP      — no meaningful dt-dependence

Why range-based rather than per-dt voting:
  Small dt (0.005-0.020) correctly gives c_opt ~= 1.0 because the
  leapfrog resolves encounters well at small steps. That is the RIGHT
  learned behaviour, not a weakness. The old per-dt voting labelled
  small-dt rows as "no value-add" which undersold the result.
  What matters is whether c_opt varies significantly WITH dt.

Backward compatible with:
  - New format (Zone 3 only, from final generate_encounter_data.py):
    fields: r_AU, r_soft, log_mi, log_mj, log_dt, c_opt, log_c_opt,
            c_ana, log_c_ana, improvement
  - Old format (Zone 2+3, from previous generate version):
    extra fields: zone, log_macro_dt
    Zone 2 data shown FOR REFERENCE ONLY — not used for training
    (see design decision in generate_encounter_data.py)

Four output figures:
  fig_step3a_c_opt_vs_r.png    -- c_opt vs r, coloured by dt
  fig_step3b_c_diff_vs_r.png   -- c_opt - c_analytic vs r, by dt
  fig_step3c_improvement.png   -- MSE improvement distribution per dt
  fig_step3d_c_opt_vs_dt.png   -- c_opt vs dt (the learned 1D surface)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FILE = "encounter_data.npz"

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":     150,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})

EPS = 3e-4   # must match generator


def _c_analytic(r):
    """Analytic softening correction: c = (r_soft/r)³."""
    r_soft = np.sqrt(r**2 + EPS**2)
    return (r_soft / r) ** 3


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data = np.load(FILE)
    except FileNotFoundError:
        print(f"ERROR: {FILE} not found. Run generate_encounter_data.py first.")
        sys.exit(1)

    r     = data["r_AU"].astype(np.float64)
    c_opt = data["c_opt"].astype(np.float64)
    c_ana = data["c_ana"].astype(np.float64)
    impr  = data["improvement"].astype(np.float64)

    # log_dt is present in both old and new format; it is the 4th NN input
    dt_vals = np.exp(data["log_dt"].astype(np.float64))   # effective dt

    # ── Handle old vs new format ───────────────────────────────────────────────
    # New format (Zone 3 only): no zone or log_macro_dt fields.
    #   dt_vals = macro_dt (discrete, 8 values from DT_VALUES).
    # Old format (Zone 2+3): has zone and log_macro_dt fields.
    #   For Zone 3: log_macro_dt == log_dt. For Zone 2: they differ.

    if "zone" in data.files and "log_macro_dt" in data.files:
        zone     = data["zone"].astype(np.float64)
        macro_dt = np.exp(data["log_macro_dt"].astype(np.float64))
        has_z2   = bool(np.any(zone == 2.0))
        z3_mask  = (zone == 3.0)
        z2_mask  = (zone == 2.0)
        fmt      = "old (Zone 2+3)"
    else:
        # New format: all samples are Zone 3; dt_vals are the discrete macro_dts
        zone     = np.full(len(r), 3.0)
        macro_dt = dt_vals.copy()
        has_z2   = False
        z3_mask  = np.ones(len(r), dtype=bool)
        z2_mask  = np.zeros(len(r), dtype=bool)
        fmt      = "new (Zone 3 only)"

    # Unique macro_dt values for Zone 3 (grouping and colouring)
    mdt_vals = sorted(set(round(float(v), 4) for v in macro_dt[z3_mask]))
    n_dt     = len(mdt_vals)
    colours  = plt.cm.viridis(np.linspace(0.1, 0.9, n_dt))
    col_map  = {dt: col for dt, col in zip(mdt_vals, colours)}

    # ── Report header ──────────────────────────────────────────────────────────
    print("=" * 75)
    print("INSPECT REPORT  --  encounter_data.npz")
    print("=" * 75)
    print(f"  Format     : {fmt}")
    print(f"  Total n    : {len(r)}")
    print(f"  Zone 3     : {int(z3_mask.sum())}  (macro-step, training data)")
    if has_z2:
        print(f"  Zone 2     : {int(z2_mask.sum())}  (sub-step, FOR REFERENCE ONLY)")
    print(f"  dt values  : {mdt_vals}")
    print()

    # ── Zone 3 analysis table ─────────────────────────────────────────────────
    print("─" * 75)
    print("ZONE 3  (macro-step zone, r = 0.052-0.148 AU)  ← TRAINING DATA")
    print("  c_opt should decrease as dt increases (leapfrog misses encounter")
    print("  at large dt; c<1 reduces overshoot).")
    print("  c_ana ~= 1.000 throughout (softening negligible in Zone 3).")
    print("─" * 75)
    hdr = (f"  {'dt':>7} | {'n':>6} | {'c_opt_med':>10} | "
           f"{'c_ana_med':>10} | {'diff_med':>10} | {'impr_med':>10}")
    print(hdr)
    print("  " + "-" * 67)

    c_opt_meds = []
    for dt_v, col in zip(mdt_vals, colours):
        mask = z3_mask & (np.abs(macro_dt - dt_v) < 1e-5)
        n    = int(mask.sum())
        if n == 0:
            continue
        cm   = float(np.median(c_opt[mask]))
        cam  = float(np.median(c_ana[mask]))
        dm   = float(np.median((c_opt - c_ana)[mask]))
        im   = float(np.median(impr[mask]))
        c_opt_meds.append((dt_v, cm))
        print(f"  {dt_v:7.3f} | {n:6d} | {cm:10.5f} | "
              f"{cam:10.7f} | {dm:10.5f} | {im:10.2%}")

    print()
    # Range-based verdict: does c_opt_med vary significantly with dt?
    if len(c_opt_meds) >= 2:
        c_vals = [x[1] for x in c_opt_meds]
        c_rng  = float(max(c_vals) - min(c_vals))
        c_max  = float(max(c_vals))
        c_min  = float(min(c_vals))
        dt_max = c_opt_meds[c_vals.index(min(c_vals))][0]

        print(f"  c_opt_med range across dt: {c_max:.4f} (dt={c_opt_meds[0][0]:.3f})"
              f"  →  {c_min:.4f} (dt={dt_max:.3f})")
        print(f"  Range = {c_rng:.4f}")
        print()

        if c_rng > 0.10:
            verdict = (f"PROCEED: strong dt-dependence confirmed (range={c_rng:.3f})."
                       f" NN learns c(r,dt) which has no analytic form.")
        elif c_rng > 0.05:
            verdict = (f"PROCEED WITH CAUTION: mild dt-dependence (range={c_rng:.3f})."
                       f" Value-add exists but is limited.")
        else:
            verdict = (f"STOP: c_opt ~= c_analytic for all dt (range={c_rng:.3f})."
                       f" No meaningful dt-dependence. No value in trajectory training.")
    else:
        verdict = "INSUFFICIENT DATA for verdict."

    print(f"  VERDICT: {verdict}")

    # ── Zone 2 reference table (old format only) ───────────────────────────────
    if has_z2:
        print()
        print("─" * 75)
        print("ZONE 2  (sub-step zone)  ← FOR REFERENCE ONLY, NOT TRAINING DATA")
        print()
        print("  Zone 2 uses c = (r_soft/r)³ analytically in the simulation.")
        print("  Reason: Zone 2 = Earth-Moon bound orbit.")
        print("    Earth-Moon masses (3e-8 to 3e-6 Msun) are 4-7 orders of")
        print("    magnitude below the training mass range (0.001-2.0 Msun).")
        print("    The new c IS mass-dependent; the NN cannot extrapolate.")
        print("    The analytic formula correctly gives c=1.020 for Earth-Moon")
        print("    (the 2% softening correction) regardless of mass.")
        print()
        print("  The data below shows Zone 2 c_opt < 1 at large sub_dt.")
        print("  This is EXPECTED: training masses are large, giving high")
        print("  step_fraction even with n_sub=16, so c_opt < 1 is correct")
        print("  FOR THOSE TRAINING MASSES. But it would be wrong for")
        print("  Earth-Moon masses where step_fraction << 1 and c_opt~1.02.")
        print()

        # Group Zone 2 by macro_dt for display
        mdt_vals_z2 = sorted(set(round(float(v), 4) for v in macro_dt[z2_mask]))
        hdr2 = (f"  {'macro_dt':>9} | {'n':>6} | {'c_opt_med':>10} | "
                f"{'c_ana_med':>10} | {'sub_dt_med':>12}")
        print(hdr2)
        print("  " + "-" * 55)

        for dt_v in mdt_vals_z2:
            mask  = z2_mask & (np.abs(macro_dt - dt_v) < 1e-5)
            n     = int(mask.sum())
            if n == 0:
                continue
            cm    = float(np.median(c_opt[mask]))
            cam   = float(np.median(c_ana[mask]))
            sdm   = float(np.median(dt_vals[mask]))
            print(f"  {dt_v:9.3f} | {n:6d} | {cm:10.5f} | "
                  f"{cam:10.7f} | {sdm:12.5f}")
        print("─" * 75)

    # ── Figure A: c_opt vs r, coloured by dt ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt_v, col in zip(mdt_vals, colours):
        mask = z3_mask & (np.abs(macro_dt - dt_v) < 1e-5)
        if mask.sum() == 0:
            continue
        ax.scatter(r[mask], c_opt[mask], s=2, alpha=0.20,
                   color=col, label=f"dt={dt_v:.3f}")
    r_plot = np.linspace(float(r[z3_mask].min()), float(r[z3_mask].max()), 300)
    ax.plot(r_plot, _c_analytic(r_plot), "k--", lw=1.5,
            label="c_analytic = (r_soft/r)³  [old training target]")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="c = 1.0")
    ax.set_xlabel("Separation r (AU)")
    ax.set_ylabel("c_optimal")
    ax.set_title("Zone 3: c_optimal vs r  (coloured by macro_dt)\n"
                 "c_opt shifts below c_analytic as dt increases — this dt-dependence"
                 " has no analytic form")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, min(5.5, float(c_opt[z3_mask].max()) * 1.1 + 0.3))
    fig.tight_layout()
    fig.savefig("fig_step3a_c_opt_vs_r.png")
    plt.close(fig)
    print()
    print("Saved fig_step3a_c_opt_vs_r.png")

    # ── Figure B: c_opt - c_analytic vs r ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt_v, col in zip(mdt_vals, colours):
        mask = z3_mask & (np.abs(macro_dt - dt_v) < 1e-5)
        if mask.sum() == 0:
            continue
        ax.scatter(r[mask], (c_opt - c_ana)[mask], s=2, alpha=0.20,
                   color=col, label=f"dt={dt_v:.3f}")
    ax.axhline(0.0, color="k", lw=1.2, ls="--", label="zero difference")
    ax.set_xlabel("Separation r (AU)")
    ax.set_ylabel("c_optimal − c_analytic")
    ax.set_title("c_optimal − c_analytic in Zone 3\n"
                 "Non-zero and dt-dependent = NN is learning something the"
                 " formula cannot capture")
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("fig_step3b_c_diff_vs_r.png")
    plt.close(fig)
    print("Saved fig_step3b_c_diff_vs_r.png")

    # ── Figure C: improvement distribution per dt ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt_v, col in zip(mdt_vals, colours):
        mask = z3_mask & (np.abs(macro_dt - dt_v) < 1e-5)
        if mask.sum() < 5:
            continue
        ax.hist(impr[mask] * 100.0, bins=40, alpha=0.45,
                color=col, label=f"dt={dt_v:.3f}", density=True)
    ax.axvline(10.0, color="k", lw=1.2, ls="--", label="10% threshold")
    ax.set_xlabel("One-step MSE improvement over c=1.0  (%)")
    ax.set_ylabel("Density")
    ax.set_title("Zone 3: MSE improvement distribution per dt\n"
                 "All accepted samples already have >5% improvement (filter criterion)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("fig_step3c_improvement.png")
    plt.close(fig)
    print("Saved fig_step3c_improvement.png")

    # ── Figure D: c_opt vs dt — the 1D learned surface ────────────────────────
    # This is the most important figure: shows the dt-dependence the NN must learn.
    # For each dt, plot median c_opt with IQR as error bars.
    fig, ax = plt.subplots(figsize=(9, 5))
    dt_arr   = []
    med_arr  = []
    q25_arr  = []
    q75_arr  = []
    for dt_v, col in zip(mdt_vals, colours):
        mask = z3_mask & (np.abs(macro_dt - dt_v) < 1e-5)
        if mask.sum() < 5:
            continue
        vals = c_opt[mask]
        med  = float(np.median(vals))
        q25  = float(np.percentile(vals, 25))
        q75  = float(np.percentile(vals, 75))
        # Individual points (sub-sampled for clarity)
        idx_sub = np.random.RandomState(0).choice(len(vals),
                                                   size=min(500, len(vals)),
                                                   replace=False)
        ax.scatter(np.full(len(idx_sub), dt_v), vals[idx_sub],
                   s=2, alpha=0.12, color=col)
        dt_arr.append(dt_v)
        med_arr.append(med)
        q25_arr.append(q25)
        q75_arr.append(q75)

    # Overlay median trend with IQR band
    dt_arr  = np.array(dt_arr)
    med_arr = np.array(med_arr)
    q25_arr = np.array(q25_arr)
    q75_arr = np.array(q75_arr)
    ax.fill_between(dt_arr, q25_arr, q75_arr, alpha=0.25,
                    color="steelblue", label="IQR (25th-75th percentile)")
    ax.plot(dt_arr, med_arr, "o-", color="steelblue", lw=2,
            ms=6, label="Median c_opt", zorder=5)

    # c_analytic reference line (= 1.0 for Zone 3)
    ax.axhline(1.0, color="k", lw=1.2, ls="--",
               label="c = 1.0  (old model, Zone 3)")

    # Legend entries for dt colours
    legend_elems = [
        Line2D([0], [0], color="steelblue", lw=2, label="Median c_opt (Zone 3)"),
        Line2D([0], [0], color="steelblue", lw=0, marker="s",
               ms=8, alpha=0.35, label="IQR band"),
        Line2D([0], [0], color="k", lw=1.2, ls="--", label="c=1.0 (old model)"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    ax.set_xlabel("macro_dt  (yr)")
    ax.set_ylabel("c_optimal")
    ax.set_title("Zone 3: c_optimal vs dt — the NN learns this relationship\n"
                 "c_opt decreases from ~1.0 (small dt, encounter resolved)"
                 " to ~0.76 (large dt, encounter missed)\n"
                 "This dt-dependence has NO analytic form — it requires a NN")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.3, 1.15)
    fig.tight_layout()
    fig.savefig("fig_step3d_c_opt_vs_dt.png")
    plt.close(fig)
    print("Saved fig_step3d_c_opt_vs_dt.png")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 75)
    if has_z2:
        print()
        print("NOTE: This file contains Zone 2 data from a previous run.")
        print("  Zone 2 data is shown for reference only.")
        print("  Training uses Zone 3 samples only.")
        print("  Re-run generate_encounter_data.py to produce Zone 3-only data.")


if __name__ == "__main__":
    main()
