r"""
pair_eval_v4_zone3_event_audit.py

Purpose
-------
Audit the actual Zone 3 NN trigger events for the optimized frozen-c Phase 2
evaluator. This file is intentionally narrow: it only answers the next question:

    Why does dt=0.04 get worse while dt=0.08 improves?

Starting point
--------------
This script reuses the model class, weight extraction, v4 forward pass, IAS15
reference solver, and RMS helper from:

    pair_eval_v4_zone3_frozen_timeavg_3way.py

Place this file in the same folder:

    C:\Aarush\Physics\training\encounter_training

Typical run
-----------
    python -B pair_eval_v4_zone3_event_audit.py --model_path pair_correction_nn_v4_bounded.pt

Outputs
-------
Creates:
    zone3_event_audit_out\audit_summary.txt
    zone3_event_audit_out\zone3_events_dt0p040.csv
    zone3_event_audit_out\zone3_events_dt0p080.csv
    zone3_event_audit_out\audit_rms_dt0p040.png
    zone3_event_audit_out\audit_rms_dt0p080.png

What gets logged
----------------
For each Zone 3 trigger event:
    time, step index, pair, r, v_rad_norm, v_tan_norm, c_pred,
    strong_approach flag, gate fallback flag,
    v4/noNN RMS before the event, after the event, and after short windows.

Important
---------
This script is diagnostic only. It does not replace the evaluator used for the
main Phase 2 frontier table.
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the checked 3-way evaluator components.
from pair_eval_v4_zone3_frozen_timeavg_3way import (
    HybridConfig,
    load_v4_model,
    extract_weights_numpy_v4,
    v4_forward_numpy,
    simulate_rebound_ias15,
    rms_sep,
)


PAIR_LABELS = ["0-1", "0-2", "1-2"]


def dt_token(dt: float) -> str:
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def make_ic1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IC1 from pair_eval_after_adaptive_timeavg.py / 3-way evaluator."""
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(axis=0) / M
    v0 = v0 - (m[:, None] * v0).sum(axis=0) / M
    return x0, v0, m


def pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ii.append(i)
            jj.append(j)
    return np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64)


def simulate_audit(
    x0: np.ndarray,
    v0: np.ndarray,
    m: np.ndarray,
    model,
    cfg: HybridConfig,
    dt: float,
    T: float,
    n_samples: int,
    use_zone3_nn: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]], Dict[str, float]]:
    """
    Frozen-c leapfrog simulation with event logging.

    This mirrors the Phase 2 frozen-c deployment rule:
      - Zone 2 direct Newtonian + substeps, no NN.
      - Zone 3 frozen c for each macro/sub-step.
      - Optional no-Zone-3-NN baseline: Zone 3 uses c=1 times F_soft.
    """
    w = extract_weights_numpy_v4(model)
    N = x0.shape[0]
    ii, jj = pair_indices(N)
    P = len(ii)

    G = cfg.G
    eps2 = cfg.eps * cfg.eps
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi = m_f[ii]
    mj = m_f[jj]
    Gmimj = G * mi * mj
    inv_mi = 1.0 / mi
    inv_mj = 1.0 / mj
    log_mi = np.log(mi + 1e-30).astype(np.float32)
    log_mj = np.log(mj + 1e-30).astype(np.float32)

    times = np.linspace(0.0, T, n_samples)
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    events: List[Dict[str, float]] = []
    stats = {
        "pair_evals": 0,
        "zone2_pairs": 0,
        "zone3_pairs": 0,
        "zone3_applied_pairs": 0,
        "zone3_gate_pairs": 0,
        "strong_zone3_pairs": 0,
        "steps": 0,
        "total_substeps": 0,
        "steps_with_zone3": 0,
    }

    def geom(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        return rij, r2, r

    def build_features(rij, r2, r, vel, step_dt, zmask):
        n_z = int(np.sum(zmask))
        r_soft = np.sqrt(r2[zmask] + eps2)
        rij_z = rij[zmask]
        r_z = r[zmask]
        r_hat = rij_z / (r_z[:, None] + 1e-30)
        vij = vel[jj] - vel[ii]
        vij_z = vij[zmask]
        v_rad = np.einsum("ij,ij->i", vij_z, r_hat)
        v_tan_vec = vij_z - v_rad[:, None] * r_hat
        v_tan = np.sqrt(np.einsum("ij,ij->i", v_tan_vec, v_tan_vec) + 1e-30)
        v_scale = np.sqrt(G * (mi[zmask] + mj[zmask]) / (r_z + 1e-30))
        v_rad_norm = v_rad / (v_scale + 1e-30)
        v_tan_norm = v_tan / (v_scale + 1e-30)

        nn_in = np.empty((n_z, 6), dtype=np.float32)
        nn_in[:, 0] = np.log(r_soft + 1e-30).astype(np.float32)
        nn_in[:, 1] = log_mi[zmask]
        nn_in[:, 2] = log_mj[zmask]
        nn_in[:, 3] = np.float32(np.log(float(step_dt) + 1e-30))
        nn_in[:, 4] = v_rad_norm.astype(np.float32)
        nn_in[:, 5] = v_tan_norm.astype(np.float32)
        return nn_in, v_rad_norm, v_tan_norm

    def freeze_state(pos, vel, step_dt, t_start, step_index, sub_index):
        rij, r2, r = geom(pos)
        frozen_c = np.ones(P, dtype=np.float64)
        eligible = np.zeros(P, dtype=bool)
        fallback = np.zeros(P, dtype=bool)
        strong = np.zeros(P, dtype=bool)
        vrn_all = np.full(P, np.nan, dtype=np.float64)
        vtn_all = np.full(P, np.nan, dtype=np.float64)

        zmask = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        stats["zone2_pairs"] += int(np.sum((r >= cfg.zone1_r_gate) & (r < cfg.adapt_thresh)))
        stats["zone3_pairs"] += int(np.sum(zmask))

        if np.any(zmask):
            eligible[zmask] = True
            stats["steps_with_zone3"] += 1
            if use_zone3_nn:
                nn_in, vrn, vtn = build_features(rij, r2, r, vel, step_dt, zmask)
                c = v4_forward_numpy(nn_in, w)
                r_soft = np.sqrt(r2[zmask] + eps2)
                fb = ((r_soft < cfg.r_soft_min) | (c < cfg.c_min) |
                      (c > cfg.c_max) | ~np.isfinite(c))
                frozen_c[zmask] = c
                fallback[zmask] = fb
                strong[zmask] = vrn < -0.6
                vrn_all[zmask] = vrn
                vtn_all[zmask] = vtn
                stats["zone3_applied_pairs"] += int(np.sum(zmask))
                stats["zone3_gate_pairs"] += int(np.sum(fb))
                stats["strong_zone3_pairs"] += int(np.sum(vrn < -0.6))
            else:
                nn_in, vrn, vtn = build_features(rij, r2, r, vel, step_dt, zmask)
                vrn_all[zmask] = vrn
                vtn_all[zmask] = vtn
                stats["zone3_applied_pairs"] += int(np.sum(zmask))

            for p in np.where(zmask)[0]:
                events.append({
                    "mode": label,
                    "dt": float(dt),
                    "step_dt": float(step_dt),
                    "step_index": int(step_index),
                    "sub_index": int(sub_index),
                    "t_start": float(t_start),
                    "t_end": float(t_start + step_dt),
                    "pair_index": int(p),
                    "pair": PAIR_LABELS[p],
                    "r_start": float(r[p]),
                    "v_rad_norm": float(vrn_all[p]),
                    "v_tan_norm": float(vtn_all[p]),
                    "c_pred": float(frozen_c[p]) if use_zone3_nn else 1.0,
                    "gate_fallback": bool(fallback[p]) if use_zone3_nn else False,
                    "strong_approach": bool(strong[p]) if use_zone3_nn else bool(vrn_all[p] < -0.6),
                })

        return rij, r2, r, frozen_c, eligible, fallback, strong

    def compute_acc_from_geom(rij, r2, r, frozen_c, eligible, fallback):
        F_scalar = Gmimj / (r2 * r + 1e-30)
        zmask = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        apply = eligible & zmask
        if np.any(apply):
            F_soft = Gmimj[apply] / ((r2[apply] + eps2) ** 1.5 + 1e-30)
            if use_zone3_nn:
                c = frozen_c[apply]
                fb = fallback[apply]
                F_scalar[apply] = np.where(fb, F_scalar[apply], c * F_soft)
            else:
                F_scalar[apply] = F_soft
        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        stats["pair_evals"] += P
        return acc

    def compute_acc(pos, frozen_c, eligible, fallback):
        rij, r2, r = geom(pos)
        return compute_acc_from_geom(rij, r2, r, frozen_c, eligible, fallback)

    def min_pair_dist(pos):
        _, r2, _ = geom(pos)
        return float(np.sqrt(np.min(r2) + 1e-30))

    def one_frozen_step(x_in, v_in, step_dt, t_start, step_index, sub_index):
        rij0, r20, r0, frozen_c, eligible, fallback, _strong = freeze_state(
            x_in, v_in, step_dt, t_start, step_index, sub_index
        )
        a0 = compute_acc_from_geom(rij0, r20, r0, frozen_c, eligible, fallback)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        a1 = compute_acc(x_new, frozen_c, eligible, fallback)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new

    # Initial sample.
    si = 0
    t_cur = 0.0
    next_t = times[si]
    while si < n_samples and t_cur >= next_t - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < n_samples:
            next_t = times[si]

    n_steps = int(math.ceil(T / dt))
    for step in range(n_steps):
        if t_cur >= T - 1e-12:
            break
        macro_dt = min(float(dt), T - t_cur)
        r_min = min_pair_dist(x)
        if r_min < cfg.adapt_thresh:
            n_sub = min(cfg.max_substeps, max(2, int(math.ceil(cfg.adapt_thresh / max(r_min, 1e-30)))))
            sub_dt = macro_dt / n_sub
            for sub in range(n_sub):
                x, v = one_frozen_step(x, v, sub_dt, t_cur, step, sub)
                t_cur += sub_dt
                stats["total_substeps"] += 1
        else:
            x, v = one_frozen_step(x, v, macro_dt, t_cur, step, -1)
            t_cur += macro_dt
        stats["steps"] += 1

        while si < n_samples and t_cur >= next_t - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                next_t = times[si]

    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    total_pair_evals = max(int(stats["pair_evals"]), 1)
    perf = {
        "steps": float(stats["steps"]),
        "total_substeps": float(stats["total_substeps"]),
        "zone2_frac": float(stats["zone2_pairs"] / total_pair_evals),
        "zone3_frac": float(stats["zone3_pairs"] / total_pair_evals),
        "zone3_applied_frac": float(stats["zone3_applied_pairs"] / total_pair_evals),
        "zone3_gate_frac": float(stats["zone3_gate_pairs"] / max(stats["zone3_applied_pairs"], 1)),
        "strong_zone3_frac": float(stats["strong_zone3_pairs"] / max(stats["zone3_applied_pairs"], 1)),
        "steps_with_zone3": float(stats["steps_with_zone3"]),
        "events": float(len(events)),
    }
    return times, pos_out, vel_out, events, perf


def annotate_events(events: List[Dict[str, float]], times: np.ndarray,
                    rms_v4: np.ndarray, rms_no: np.ndarray) -> None:
    """Add local error information to each v4 event."""
    windows = [0.0, 0.2, 1.0, 5.0]
    for ev in events:
        t0 = float(ev["t_start"])
        ib = max(0, int(np.searchsorted(times, t0, side="right") - 1))
        ev["v4_rms_before"] = float(rms_v4[ib])
        ev["noNN_rms_before"] = float(rms_no[ib])
        for w in windows:
            ta = min(times[-1], t0 + float(ev["step_dt"]) + w)
            ia = min(len(times) - 1, int(np.searchsorted(times, ta, side="left")))
            key = "after_step" if w == 0.0 else f"after_{str(w).replace('.', 'p')}yr"
            ev[f"v4_rms_{key}"] = float(rms_v4[ia])
            ev[f"noNN_rms_{key}"] = float(rms_no[ia])
            ev[f"v4_minus_noNN_{key}"] = float(rms_v4[ia] - rms_no[ia])


def write_events_csv(path: str, events: List[Dict[str, float]]) -> None:
    if not events:
        with open(path, "w", encoding="utf-8") as f:
            f.write("no Zone 3 events\n")
        return
    keys = [
        "mode", "dt", "step_dt", "step_index", "sub_index", "t_start", "t_end",
        "pair_index", "pair", "r_start", "v_rad_norm", "v_tan_norm", "c_pred",
        "gate_fallback", "strong_approach",
        "v4_rms_before", "noNN_rms_before",
        "v4_rms_after_step", "noNN_rms_after_step", "v4_minus_noNN_after_step",
        "v4_rms_after_0p2yr", "noNN_rms_after_0p2yr", "v4_minus_noNN_after_0p2yr",
        "v4_rms_after_1p0yr", "noNN_rms_after_1p0yr", "v4_minus_noNN_after_1p0yr",
        "v4_rms_after_5p0yr", "noNN_rms_after_5p0yr", "v4_minus_noNN_after_5p0yr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for ev in events:
            writer.writerow({k: ev.get(k, "") for k in keys})


def plot_rms_with_events(path: str, times: np.ndarray, rms_v4: np.ndarray, rms_no: np.ndarray,
                         events: List[Dict[str, float]], dt: float) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(times, rms_no, lw=1.2, label="revised no Zone 3 NN")
    ax.plot(times, rms_v4, lw=1.2, label="v4 Zone 3 NN")
    for ev in events:
        color_alpha = 0.28 if ev.get("strong_approach") else 0.14
        ax.axvline(ev["t_start"], lw=0.8, alpha=color_alpha)
    ax.set_yscale("log")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS separation vs IAS15 (AU, log)")
    ax.set_title(f"Zone 3 event audit, dt={dt:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def run_one_dt(dt: float, args, model, cfg, x0, v0, m, ref_pos, times, out_dir: str) -> Dict[str, float]:
    print(f"[audit] running dt={dt:g}: v4...")
    tv4, pv4, _vv4, ev_v4, perf_v4 = simulate_audit(
        x0, v0, m, model, cfg, dt, args.T, args.n_samples, True, "v4"
    )
    print(f"[audit] running dt={dt:g}: no-Zone-3-NN baseline...")
    tno, pno, _vno, _ev_no, perf_no = simulate_audit(
        x0, v0, m, model, cfg, dt, args.T, args.n_samples, False, "no_zone3_nn"
    )
    if not np.allclose(tv4, times) or not np.allclose(tno, times):
        raise RuntimeError("internal time-grid mismatch")

    rms_v4 = rms_sep(pv4, ref_pos)
    rms_no = rms_sep(pno, ref_pos)
    annotate_events(ev_v4, times, rms_v4, rms_no)

    tok = dt_token(dt)
    write_events_csv(os.path.join(out_dir, f"zone3_events_dt{tok}.csv"), ev_v4)
    plot_rms_with_events(os.path.join(out_dir, f"audit_rms_dt{tok}.png"), times, rms_v4, rms_no, ev_v4, dt)

    # Top events by v4 disadvantage after 1 year.
    top = sorted(ev_v4, key=lambda e: e.get("v4_minus_noNN_after_1p0yr", -1e99), reverse=True)[:8]
    txt_path = os.path.join(out_dir, f"zone3_event_summary_dt{tok}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Zone 3 event audit summary for dt={dt:g}\n")
        f.write("=" * 72 + "\n")
        f.write(f"v4_final_err: {float(rms_v4[-1]):.6e}\n")
        f.write(f"noNN_final_err: {float(rms_no[-1]):.6e}\n")
        f.write(f"v4_time_avg_rms: {float(np.sqrt(np.mean(rms_v4**2))):.6e}\n")
        f.write(f"noNN_time_avg_rms: {float(np.sqrt(np.mean(rms_no**2))):.6e}\n")
        f.write(f"v4_events: {len(ev_v4)}\n")
        f.write(f"v4_steps_with_zone3: {perf_v4['steps_with_zone3']:.0f}\n")
        f.write(f"v4_zone3_applied_frac: {perf_v4['zone3_applied_frac']:.6f}\n")
        f.write(f"v4_gate_frac: {perf_v4['zone3_gate_frac']:.6f}\n")
        f.write(f"v4_strong_zone3_frac: {perf_v4['strong_zone3_frac']:.6f}\n\n")
        if ev_v4:
            c_vals = np.array([e["c_pred"] for e in ev_v4], dtype=float)
            vr_vals = np.array([e["v_rad_norm"] for e in ev_v4], dtype=float)
            f.write(f"c_pred min/median/max: {np.min(c_vals):.6f} / {np.median(c_vals):.6f} / {np.max(c_vals):.6f}\n")
            f.write(f"v_rad_norm min/median/max: {np.min(vr_vals):.6f} / {np.median(vr_vals):.6f} / {np.max(vr_vals):.6f}\n\n")
        f.write("Top events by v4_minus_noNN_after_1yr:\n")
        f.write("time\tpair\tr\tvr_norm\tvt_norm\tc\tstrong\tv4_minus_noNN_after_1yr\n")
        for e in top:
            f.write(f"{e['t_start']:.6f}\t{e['pair']}\t{e['r_start']:.6e}\t"
                    f"{e['v_rad_norm']:.6f}\t{e['v_tan_norm']:.6f}\t{e['c_pred']:.6f}\t"
                    f"{int(e['strong_approach'])}\t{e.get('v4_minus_noNN_after_1p0yr', np.nan):.6e}\n")

    return {
        "dt": float(dt),
        "v4_final_err": float(rms_v4[-1]),
        "noNN_final_err": float(rms_no[-1]),
        "v4_time_avg_rms": float(np.sqrt(np.mean(rms_v4**2))),
        "noNN_time_avg_rms": float(np.sqrt(np.mean(rms_no**2))),
        "v4_minus_noNN_time_avg": float(np.sqrt(np.mean(rms_v4**2)) - np.sqrt(np.mean(rms_no**2))),
        "events": float(len(ev_v4)),
        "steps_with_zone3": float(perf_v4["steps_with_zone3"]),
        "zone3_applied_frac": float(perf_v4["zone3_applied_frac"]),
        "gate_frac": float(perf_v4["zone3_gate_frac"]),
        "strong_zone3_frac": float(perf_v4["strong_zone3_frac"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit actual Zone 3 NN trigger events for dt=0.04 vs dt=0.08.")
    parser.add_argument("--model_path", default="pair_correction_nn_v4_bounded.pt")
    parser.add_argument("--out_dir", default="zone3_event_audit_out")
    parser.add_argument("--dts", default="0.04,0.08", help="Comma-separated dt values to audit.")
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_v4_model(args.model_path)
    cfg = HybridConfig()
    x0, v0, m = make_ic1()

    print("[audit] IAS15 baseline...")
    times, ref_pos, _ref_vel, ref_perf = simulate_rebound_ias15(x0, v0, m, cfg.G, args.T, args.n_samples)
    print(f"[audit] IAS15 time={ref_perf['total_time_sec']:.3f}s")

    rows = []
    dts = [float(x.strip()) for x in args.dts.split(",") if x.strip()]
    for dt in dts:
        rows.append(run_one_dt(dt, args, model, cfg, x0, v0, m, ref_pos, times, args.out_dir))

    summary_path = os.path.join(args.out_dir, "audit_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Zone 3 event audit: dt=0.04 failure vs dt=0.08 improvement\n")
        f.write("=" * 78 + "\n")
        f.write(f"T_years: {args.T}\n")
        f.write(f"n_samples: {args.n_samples}\n")
        f.write(f"model: {args.model_path}\n")
        f.write("method: Zone 2 direct Newtonian + substeps; Zone 3 frozen v4 NN vs no-Zone-3-NN baseline\n\n")
        f.write("dt\tv4_time_avg\tnoNN_time_avg\tv4_minus_noNN\tv4_final\tnoNN_final\tevents\tsteps_with_zone3\tzone3_frac\tgate_frac\tstrong_frac\n")
        for r in rows:
            f.write(f"{r['dt']}\t{r['v4_time_avg_rms']:.6e}\t{r['noNN_time_avg_rms']:.6e}\t"
                    f"{r['v4_minus_noNN_time_avg']:.6e}\t{r['v4_final_err']:.6e}\t{r['noNN_final_err']:.6e}\t"
                    f"{int(r['events'])}\t{int(r['steps_with_zone3'])}\t{r['zone3_applied_frac']:.6f}\t"
                    f"{r['gate_frac']:.6f}\t{r['strong_zone3_frac']:.6f}\n")
        f.write("\nInterpretation guide:\n")
        f.write("- Positive v4_minus_noNN means v4 is worse than the revised no-Zone-3-NN baseline.\n")
        f.write("- Check the per-dt CSV files to see whether one event or several events cause the difference.\n")
        f.write("- The PNG plots show RMS divergence with vertical lines at Zone 3 NN event times.\n")

    print(f"[audit] wrote {args.out_dir}")
    print(f"[audit] summary: {summary_path}")
    print("[audit] done.")


if __name__ == "__main__":
    main()
