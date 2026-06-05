"""
multi_ic_gate_eval.py

Cross-IC evaluation driver for the gated v4 Zone-3 NN.

Runs the same frontier sweep across 7 ICs from the phase-1 paper (IC1-IC7),
in three modes per (IC, dt):
  - NN gate-ON   : trajectory NN active, encounter gate enabled (current best)
  - NN gate-OFF  : trajectory NN active, encounter gate disabled (un-gated)
  - no-NN        : Zone 3 uses c=1 (no NN at all)

For each IC, IAS15 is run once and reused as reference across all dts.

Output:
  out_dir/
    summary_by_ic.txt          # per-IC tables + cross-IC matrix
    summary_cross_ic.csv       # machine-readable cross-IC matrix
    per_ic_csv/                # one frontier CSV per IC
      IC1_default.csv
      IC2_near_equal.csv
      ...
    worst_dt_logs/             # per-IC firing log at the worst-dt for NN gate-on
      IC1_default_dt0.1.csv
      ...

This script does NOT modify pair_eval_v4_zone3_frozen_timeavg.py. It imports
the evaluator's building blocks and re-uses them. Run from the same
directory as the evaluator file so the import succeeds.
"""

import os
import argparse
import csv
import time
import numpy as np
import torch

# Import the evaluator building blocks (the file must be on sys.path or in cwd)
from pair_eval_v4_zone3_frozen_timeavg import (
    PairCorrectionNNv4,
    HybridConfig,
    simulate_leapfrog_v4_zone3,
    simulate_rebound_ias15,
    rms_sep,
    load_v4_model,
)


# =============================================================================
# IC definitions — byte-identical to multi_ic_eval_v3.py
# =============================================================================
ICS = {
    "IC1_default": {
        "label": "IC1: Default",
        "regime": "Moderate three-body scattering",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64),
        "stable": True,
    },
    "IC2_near_equal": {
        "label": "IC2: Near-Equal Mass",
        "regime": "Strongly interacting -- PHYSICALLY UNSTABLE (IAS15 also ejects)",
        "m":  np.array([1.0, 0.5, 0.25]),
        "x0": np.array([[0,0,0],[1,0,0],[-0.5,0.8,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.6,0],[-0.4,-0.3,0]], dtype=np.float64),
        "stable": False,
    },
    "IC3_tight": {
        "label": "IC3: Tight Inner Pair",
        "regime": "Tight inner pair -- continuous close encounters",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.5,0,0],[0,2.5,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.3,0],[-0.4,0,0]], dtype=np.float64),
        "stable": True,
    },
    "IC4_hierarchical": {
        "label": "IC4: Hierarchical",
        "regime": "Hierarchical -- weakly coupled outer body",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,5.0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.0,0],[-0.12,0,0]], dtype=np.float64),
        "stable": True,
    },
    "IC5_high_ecc": {
        "label": "IC5: High Ecc (unstable)",
        "regime": "High eccentricity -- PHYSICALLY UNSTABLE (IAS15 ejects at t=34.8 yr)",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.3,0,0],[0,2.0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,2.2,0],[-0.3,0,0]], dtype=np.float64),
        "stable": False,
    },
    "IC6_near_circular": {
        "label": "IC6: Near-Circular",
        "regime": "Near-circular coplanar -- low chaos, NN rarely invoked",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1.5,0,0],[-3.0,0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.816,0],[0,-0.577,0]], dtype=np.float64),
        "stable": True,
    },
    "IC7_high_ecc_stable": {
        "label": "IC7: High Ecc (redesign)",
        "regime": "High eccentricity -- redesigned for stability (e~0.52)",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.6,0,0],[0,4.0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.6,0],[-0.15,0,0]], dtype=np.float64),
        "stable": True,
    },
}


def com_center(x0, v0, m):
    """Apply center-of-mass centering (same recipe as evaluator)."""
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M
    return x0, v0


def write_log_csv(rows, path):
    """Write firing log CSV. If rows empty, write headers only."""
    if rows:
        headers = list(rows[0].keys())
    else:
        headers = [
            "step_idx", "substep_idx", "t_yr", "dt_step",
            "pair_i", "pair_j", "r_au", "v_rad_norm", "v_tan_norm",
            "m_i", "m_j", "c_pred", "c_anal_softening",
            "gate_fallback", "strong_approach", "n_zone3_pairs_this_step",
            "encounter_gate_suppressed",
        ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def classify_verdict(delta_pct):
    """Classify NN-vs-no-NN delta% into a verdict label.
    Negative delta means NN helps; positive means NN hurts."""
    if delta_pct > 500.0:
        return "CATASTROPHIC"
    elif delta_pct > 10.0:
        return "NN_hurts"
    elif delta_pct < -10.0:
        return "NN_helps"
    elif delta_pct < -0.5:
        return "NN_helps_mild"
    elif delta_pct > 0.5:
        return "NN_hurts_mild"
    else:
        return "tie"


def fmt_delta(d):
    if abs(d) > 1000:
        return f"{d:+.0f}%"
    elif abs(d) > 100:
        return f"{d:+.1f}%"
    else:
        return f"{d:+.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pair_correction_nn_v4_bounded.pt")
    parser.add_argument("--out_dir", default="multi_ic_gate_out")
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--dts", default="0.005,0.01,0.02,0.04,0.08,0.1")
    parser.add_argument(
        "--ics", default=None,
        help="Comma-separated IC keys to run (default: all 7). "
             "Example: --ics IC1_default,IC3_tight")
    args = parser.parse_args()

    dts = [float(x) for x in args.dts.split(",") if x.strip()]

    if args.ics:
        ic_keys = [k.strip() for k in args.ics.split(",")]
        for k in ic_keys:
            if k not in ICS:
                raise SystemExit(
                    f"Unknown IC '{k}'. Valid: {list(ICS.keys())}")
    else:
        ic_keys = list(ICS.keys())

    os.makedirs(args.out_dir, exist_ok=True)
    per_ic_dir = os.path.join(args.out_dir, "per_ic_csv")
    worst_log_dir = os.path.join(args.out_dir, "worst_dt_logs")
    os.makedirs(per_ic_dir, exist_ok=True)
    os.makedirs(worst_log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model once via the evaluator's own helper. This handles the
    # log_c_min/log_c_max buffer extraction correctly.
    # ------------------------------------------------------------------
    print(f"[multi_ic] Loading model from {args.model_path}")
    model = load_v4_model(args.model_path)
    cfg = HybridConfig()

    # ------------------------------------------------------------------
    # Master results dictionary: results[ic_key][dt][mode] = metrics dict
    # ------------------------------------------------------------------
    results = {ic: {dt: {} for dt in dts} for ic in ic_keys}
    # Also store: results[ic]["ias15"] = perf dict (one-time)
    for ic in ic_keys:
        results[ic]["ias15_time_sec"] = None

    # Store firing logs only for the worst-NN dt per IC
    # (collected on the fly and pruned)
    all_logs = {ic: {dt: None for dt in dts} for ic in ic_keys}

    t0_global = time.perf_counter()
    for ic_key in ic_keys:
        ic = ICS[ic_key]
        print(f"\n[multi_ic] ===== {ic['label']} =====")
        print(f"[multi_ic] {ic['regime']}")
        x0, v0 = com_center(ic["x0"].copy(), ic["v0"].copy(), ic["m"])
        m = ic["m"]

        # IAS15 reference for this IC (one run, reused across all dts)
        t_ias = time.perf_counter()
        times_r, p_ref, _, perf_ref = simulate_rebound_ias15(
            x0, v0, m, cfg.G, args.T, args.n_samples)
        t_ias = time.perf_counter() - t_ias
        results[ic_key]["ias15_time_sec"] = perf_ref["total_time_sec"]
        print(f"[multi_ic] IAS15 done: {perf_ref['total_time_sec']:.3f}s")

        for dt in dts:
            # NN gate-ON
            rows_on = []
            _, p_on, _, perf_on = simulate_leapfrog_v4_zone3(
                x0, v0, m, model, cfg, dt, args.T, args.n_samples,
                use_zone3_nn=True, use_encounter_gate=True,
                nn_log_rows=rows_on,
            )
            rms_on = rms_sep(p_on, p_ref)
            tar_on = float(np.sqrt(np.mean(rms_on ** 2)))
            fe_on = float(rms_on[-1])
            all_logs[ic_key][dt] = rows_on

            # NN gate-OFF
            _, p_off, _, perf_off = simulate_leapfrog_v4_zone3(
                x0, v0, m, model, cfg, dt, args.T, args.n_samples,
                use_zone3_nn=True, use_encounter_gate=False,
            )
            rms_off = rms_sep(p_off, p_ref)
            tar_off = float(np.sqrt(np.mean(rms_off ** 2)))
            fe_off = float(rms_off[-1])

            # no-NN
            _, p_no, _, perf_no = simulate_leapfrog_v4_zone3(
                x0, v0, m, model, cfg, dt, args.T, args.n_samples,
                use_zone3_nn=False,
            )
            rms_no = rms_sep(p_no, p_ref)
            tar_no = float(np.sqrt(np.mean(rms_no ** 2)))
            fe_no = float(rms_no[-1])

            denom_noNN = max(abs(tar_no), 1e-30)
            denom_off = max(abs(tar_off), 1e-30)
            delta_on_vs_noNN = 100.0 * (tar_on - tar_no) / denom_noNN
            delta_on_vs_off = 100.0 * (tar_on - tar_off) / denom_off

            speed_on = perf_ref["total_time_sec"] / max(perf_on["total_time_sec"], 1e-12)
            speed_no = perf_ref["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)

            results[ic_key][dt] = {
                "tar_on": tar_on,
                "tar_off": tar_off,
                "tar_no": tar_no,
                "fe_on": fe_on,
                "fe_off": fe_off,
                "fe_no": fe_no,
                "delta_on_vs_noNN": delta_on_vs_noNN,
                "delta_on_vs_off": delta_on_vs_off,
                "verdict": classify_verdict(delta_on_vs_noNN),
                "n_supp": int(perf_on["zone3_suppressed_pairs"]),
                "n_active_firings": int(perf_on["frozen_steps_with_zone3"]),
                "speedup_on": speed_on,
                "speedup_no": speed_no,
                "zone3_nn_frac_on": perf_on["zone3_nn_frac"],
            }

            print(f"  dt={dt:>6}: "
                  f"NN_on={tar_on:.3e}  NN_off={tar_off:.3e}  no_NN={tar_no:.3e}  "
                  f"Δ_on_vs_noNN={fmt_delta(delta_on_vs_noNN)} "
                  f"[{results[ic_key][dt]['verdict']}]")

        # Identify worst-dt for this IC (largest positive delta_on_vs_noNN).
        # If NN never hurts on this IC, pick the most NN-help dt (most negative) instead.
        deltas = [(dt, results[ic_key][dt]["delta_on_vs_noNN"]) for dt in dts]
        # Worst = highest delta (most regression)
        worst_dt, worst_delta = max(deltas, key=lambda x: x[1])
        # Write firing log for worst-dt
        log_path = os.path.join(
            worst_log_dir, f"{ic_key}_dt{worst_dt}.csv")
        write_log_csv(all_logs[ic_key][worst_dt], log_path)
        print(f"  worst dt = {worst_dt}  (Δ={fmt_delta(worst_delta)})  "
              f"log -> {os.path.basename(log_path)}  "
              f"({len(all_logs[ic_key][worst_dt])} rows)")

    t_total = time.perf_counter() - t0_global

    # ------------------------------------------------------------------
    # Per-IC frontier CSVs
    # ------------------------------------------------------------------
    for ic_key in ic_keys:
        path = os.path.join(per_ic_dir, f"{ic_key}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dt", "tar_NN_on", "tar_NN_off", "tar_no_NN",
                "final_err_NN_on", "final_err_NN_off", "final_err_no_NN",
                "delta_on_vs_noNN_pct", "delta_on_vs_off_pct",
                "verdict", "n_supp_pairs", "n_active_firings",
                "speedup_NN_on", "speedup_no_NN", "zone3_nn_frac_on",
            ])
            for dt in dts:
                r = results[ic_key][dt]
                w.writerow([
                    dt, f"{r['tar_on']:.6e}", f"{r['tar_off']:.6e}",
                    f"{r['tar_no']:.6e}",
                    f"{r['fe_on']:.6e}", f"{r['fe_off']:.6e}", f"{r['fe_no']:.6e}",
                    f"{r['delta_on_vs_noNN']:+.2f}",
                    f"{r['delta_on_vs_off']:+.2f}",
                    r["verdict"], r["n_supp"], r["n_active_firings"],
                    f"{r['speedup_on']:.2f}", f"{r['speedup_no']:.2f}",
                    f"{r['zone3_nn_frac_on']:.6f}",
                ])

    # ------------------------------------------------------------------
    # Cross-IC summary text file
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary_by_ic.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("Multi-IC evaluation of gated v4 Zone-3 NN\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"T = {args.T} yr,  n_samples = {args.n_samples}\n")
        f.write(f"dts = {dts}\n")
        f.write(f"Total wall-clock time = {t_total:.1f} s\n\n")

        f.write("Modes:\n")
        f.write("  NN_on  = trajectory NN with encounter gate ENABLED\n")
        f.write("  NN_off = trajectory NN with encounter gate DISABLED (un-gated)\n")
        f.write("  no_NN  = c=1 in Zone 3 (softened force without NN)\n\n")
        f.write("Verdict thresholds on delta_on_vs_noNN%:\n")
        f.write("  CATASTROPHIC: > 500%\n")
        f.write("  NN_hurts:     > 10%\n")
        f.write("  NN_hurts_mild: > 0.5%\n")
        f.write("  tie: -0.5 .. +0.5%\n")
        f.write("  NN_helps_mild: < -0.5%\n")
        f.write("  NN_helps:     < -10%\n\n")

        # Per-IC blocks
        for ic_key in ic_keys:
            ic = ICS[ic_key]
            f.write("-" * 78 + "\n")
            f.write(f"{ic['label']}  -- {ic['regime']}\n")
            f.write(f"  m = {list(ic['m'])}, stable = {ic['stable']}\n")
            ias_t = results[ic_key]["ias15_time_sec"]
            f.write(f"  IAS15 time = {ias_t:.3f} s\n")
            f.write(f"  {'dt':>7}  {'NN_on':>10}  {'NN_off':>10}  "
                    f"{'no_NN':>10}  {'on-vs-noNN':>12}  "
                    f"{'on-vs-off':>11}  {'verdict':>16}\n")
            for dt in dts:
                r = results[ic_key][dt]
                f.write(f"  {dt:>7.4f}  "
                        f"{r['tar_on']:>10.4e}  {r['tar_off']:>10.4e}  "
                        f"{r['tar_no']:>10.4e}  "
                        f"{fmt_delta(r['delta_on_vs_noNN']):>12}  "
                        f"{fmt_delta(r['delta_on_vs_off']):>11}  "
                        f"{r['verdict']:>16}\n")
            f.write("\n")

        # Cross-IC matrix of delta_on_vs_noNN
        f.write("=" * 78 + "\n")
        f.write("CROSS-IC MATRIX: delta time_avg_rms (NN_on - no_NN)/no_NN, in %\n")
        f.write("  Negative = NN helps, Positive = NN hurts.\n")
        f.write("  Catastrophic (>500%) is bracketed in [].\n")
        f.write("=" * 78 + "\n")
        f.write(f"  {'IC':<22}")
        for dt in dts:
            f.write(f"  dt={dt:<6}")
        f.write(f"  {'verdict_count':<25}\n")
        for ic_key in ic_keys:
            ic = ICS[ic_key]
            stable_marker = "" if ic["stable"] else " (unstable)"
            label = ic_key + stable_marker
            f.write(f"  {label:<22}")
            for dt in dts:
                d = results[ic_key][dt]["delta_on_vs_noNN"]
                if d > 500:
                    cell = f"[{d:+.0f}%]"
                elif abs(d) > 100:
                    cell = f"{d:+.0f}%"
                elif abs(d) > 10:
                    cell = f"{d:+.1f}%"
                else:
                    cell = f"{d:+.2f}%"
                f.write(f"  {cell:>8}")
            # Verdict counts for this IC
            verdicts = [results[ic_key][dt]["verdict"] for dt in dts]
            nh = sum(1 for v in verdicts if "NN_helps" in v)
            nhrt = sum(1 for v in verdicts if "NN_hurts" in v or v == "CATASTROPHIC")
            ntie = sum(1 for v in verdicts if v == "tie")
            ncat = sum(1 for v in verdicts if v == "CATASTROPHIC")
            f.write(f"  helps={nh}/hurts={nhrt}/tie={ntie}/cat={ncat}\n")
        f.write("\n")

        # Cross-IC per-dt aggregate (bounded ICs only)
        bounded_ics = [k for k in ic_keys if ICS[k]["stable"]]
        f.write("=" * 78 + "\n")
        f.write(f"PER-DT AGGREGATE OVER BOUNDED ICs ({len(bounded_ics)} ICs): "
                f"{bounded_ics}\n")
        f.write("=" * 78 + "\n")
        f.write(f"  {'dt':>7}  {'med_delta':>12}  {'max_delta':>12}  "
                f"{'helps':>5} {'hurts':>5} {'tie':>5} {'cat':>5}\n")
        for dt in dts:
            deltas_b = [results[ic][dt]["delta_on_vs_noNN"] for ic in bounded_ics]
            verdicts_b = [results[ic][dt]["verdict"] for ic in bounded_ics]
            med = float(np.median(deltas_b))
            mx = float(max(deltas_b))
            nh = sum(1 for v in verdicts_b if "NN_helps" in v)
            nhrt = sum(1 for v in verdicts_b if "NN_hurts" in v or v == "CATASTROPHIC")
            ntie = sum(1 for v in verdicts_b if v == "tie")
            ncat = sum(1 for v in verdicts_b if v == "CATASTROPHIC")
            f.write(f"  {dt:>7.4f}  {fmt_delta(med):>12}  {fmt_delta(mx):>12}  "
                    f"{nh:>5} {nhrt:>5} {ntie:>5} {ncat:>5}\n")
        f.write("\n")

        # Top-level summary
        all_verdicts_b = []
        for ic in bounded_ics:
            for dt in dts:
                all_verdicts_b.append(results[ic][dt]["verdict"])
        nh = sum(1 for v in all_verdicts_b if "NN_helps" in v)
        nhrt = sum(1 for v in all_verdicts_b if "NN_hurts" in v or v == "CATASTROPHIC")
        ntie = sum(1 for v in all_verdicts_b if v == "tie")
        ncat = sum(1 for v in all_verdicts_b if v == "CATASTROPHIC")
        total = len(all_verdicts_b)
        f.write("=" * 78 + "\n")
        f.write("OVERALL VERDICT COUNTS over bounded ICs x all dts "
                f"(N = {total}):\n")
        f.write(f"  NN helps      : {nh:>3} / {total}  ({100*nh/total:.1f}%)\n")
        f.write(f"  NN hurts      : {nhrt:>3} / {total}  ({100*nhrt/total:.1f}%)\n")
        f.write(f"  tie           : {ntie:>3} / {total}  ({100*ntie/total:.1f}%)\n")
        f.write(f"  catastrophic  : {ncat:>3} / {total}  ({100*ncat/total:.1f}%)\n")
        f.write("=" * 78 + "\n")

    # Cross-IC CSV
    cross_path = os.path.join(args.out_dir, "summary_cross_ic.csv")
    with open(cross_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ic"] + [f"dt={dt}" for dt in dts] + ["stable"])
        for ic_key in ic_keys:
            row = [ic_key]
            for dt in dts:
                row.append(f"{results[ic_key][dt]['delta_on_vs_noNN']:+.2f}%")
            row.append(ICS[ic_key]["stable"])
            w.writerow(row)

    print(f"\n[multi_ic] DONE in {t_total:.1f}s")
    print(f"[multi_ic] Summary text: {summary_path}")
    print(f"[multi_ic] Cross-IC CSV: {cross_path}")
    print(f"[multi_ic] Per-IC CSVs:  {per_ic_dir}/")
    print(f"[multi_ic] Worst-dt logs: {worst_log_dir}/")


if __name__ == "__main__":
    main()
