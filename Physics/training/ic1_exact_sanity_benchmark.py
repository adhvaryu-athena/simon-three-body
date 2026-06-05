"""
ic1_exact_sanity_benchmark.py

Exact-IC1 Sanity Benchmark for SIMON
====================================

Purpose:
    Check whether the controlled benchmark framework reproduces the known IC1
    speed--accuracy behaviour before doing any perturbation experiments.

    This script runs the exact IC1 initial condition from multi_ic_eval_v3.py at
    two operating timesteps:
        dt = 0.04 yr  (conservative IC1 point)
        dt = 0.08 yr  (speed-frontier IC1 point)

    It repeats the exact same IC several times only to reduce runtime noise in
    the speed measurement. Classification, lambda, and RMS should be identical
    up to floating-point noise across repeats.

Interpretation:
    If exact IC1 is bounded in both ias15 and SIMON and the speedups are close
    to the paper's speed-frontier pattern, then the perturbation failures are
    caused by the perturbation neighbourhood, not by a code mismatch.

Outputs:
    ic1_exact_sanity_out/ic1_exact_sanity_data.csv
    ic1_exact_sanity_out/ic1_exact_sanity_summary.txt

Run:
    python ic1_exact_sanity_benchmark.py

Requirements:
    pair_correction_nn.pt in the same folder as this script, plus rebound,
    torch, numpy.
"""

import os
import time
import math
import csv
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import rebound


# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "pair_correction_nn.pt"
OUT_DIR = "ic1_exact_sanity_out"
os.makedirs(OUT_DIR, exist_ok=True)

T_SIM = 100.0
N_SAMPLES = 5000              # dense-output protocol used in multi-IC/frontier tests
EJECTION_THRESHOLD = 10.0
DT_VALUES = [0.04, 0.08]
N_REPEATS_PER_DT = 5          # repeat exact IC1 to reduce timing noise only

# Exact IC1 from multi_ic_eval_v3.py / paper speed frontier.
BASE_M = np.array([1.0, 0.01, 0.005], dtype=np.float64)
BASE_X0 = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.2, 0.0],
], dtype=np.float64)
BASE_V0 = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-0.9, 0.0, 0.0],
], dtype=np.float64)

# Keep False to match the paper/multi_ic_eval_v3 style exactly:
# ias15 is moved to COM inside REBOUND, while SIMON uses the supplied state.
# Set True only if you want both solvers explicitly started from the same
# COM-centred coordinate system for a secondary check.
CENTER_TO_COM_BEFORE_BOTH = False

# Reference seven-IC lambda from the paper, used only for context in summary.
REF_LAMBDA_MEAN = 0.0916
REF_LAMBDA_STD = 0.0682


# =============================================================================
# SIMON model
# =============================================================================
class PairCorrectionNN(nn.Module):
    def __init__(self, hidden=32, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("input_mean", torch.zeros(3))
        self.register_buffer("input_std", torch.ones(3))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    mc_samples: int = 1
    unc_rel_thresh: float = 0.25
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


def extract_weights_numpy(model):
    sd = model.state_dict()
    return {
        "mean": sd["input_mean"].cpu().numpy().astype(np.float32),
        "std": sd["input_std"].cpu().numpy().astype(np.float32) + 1e-8,
        "w0T": sd["net.0.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b0": sd["net.0.bias"].cpu().numpy().astype(np.float32),
        "w1T": sd["net.2.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b1": sd["net.2.bias"].cpu().numpy().astype(np.float32),
        "w2T": sd["net.4.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b2": sd["net.4.bias"].cpu().numpy().astype(np.float32),
        "w3T": sd["net.6.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b3": sd["net.6.bias"].cpu().numpy().astype(np.float32),
    }


# =============================================================================
# Diagnostics
# =============================================================================
def center_to_com(x, v, m):
    mtot = float(np.sum(m))
    x_com = np.sum(x * m[:, None], axis=0) / mtot
    v_com = np.sum(v * m[:, None], axis=0) / mtot
    return x - x_com, v - v_com


def total_energy(x, v, m, G=1.0):
    ke = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    pe = 0.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            rij = float(np.linalg.norm(x[j] - x[i]))
            pe -= G * float(m[i] * m[j]) / max(rij, 1e-30)
    return ke + pe


def min_pair_distance(x):
    dmin = float("inf")
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dmin = min(dmin, float(np.linalg.norm(x[j] - x[i])))
    return dmin


def rms_sep(a, b):
    d = a - b
    per_body = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(per_body * per_body, axis=1))


def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T = times[-1]
    t0 = t0_frac * T
    t1 = t1_frac * T
    mask = (times >= t0) & (times <= t1)
    if int(np.sum(mask)) < 4:
        return float("nan")
    x = times[mask]
    y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean()
    y0 = y.mean()
    denom = np.sum((x - x0) ** 2) + 1e-30
    return float(np.sum((x - x0) * (y - y0)) / denom)


def is_bounded(pos, threshold=EJECTION_THRESHOLD):
    final_rmax = float(np.max(np.linalg.norm(pos[-1], axis=1)))
    return final_rmax < threshold


# =============================================================================
# Integrators
# =============================================================================
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples):
    """SIMON leapfrog integrator with analytic force direction and adaptive sub-stepping."""
    w = extract_weights_numpy(model)
    w_mean = w["mean"]
    w_std = w["std"]
    w0T = w["w0T"]; b0 = w["b0"]
    w1T = w["w1T"]; b1 = w["b1"]
    w2T = w["w2T"]; b2 = w["b2"]
    w3T = w["w3T"]; b3 = w["b3"]

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i)
            jj.append(j)
    ii = np.array(ii)
    jj = np.array(jj)
    P = len(ii)

    G = cfg.G
    eps2 = cfg.eps * cfg.eps
    c_min = cfg.c_min
    c_max = cfg.c_max
    r_soft_min = cfg.r_soft_min
    nn_thresh = 500.0 * cfg.eps

    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]
    mj_arr = m_f[jj]
    Gmimj = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        invr3 = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close_mask = r < nn_thresh
        n_close = int(np.sum(close_mask))
        if n_close > 0:
            r_soft_close = np.sqrt(r2[close_mask] + eps2)
            denom = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom

            nn_in = np.empty((n_close, 3), dtype=np.float32)
            nn_in[:, 0] = np.log(r_soft_close + 1e-30).astype(np.float32)
            nn_in[:, 1] = log_mi[close_mask]
            nn_in[:, 2] = log_mj[close_mask]

            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w1T + b1; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w2T + b2; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            log_c = (h @ w3T + b3).ravel()
            c = np.exp(log_c).astype(np.float64)

            fallback = ((r_soft_close < r_soft_min) |
                        (c < c_min) | (c > c_max) | ~np.isfinite(c))
            F_corrected = np.where(fallback, F_scalar[close_mask], c * F_soft_close)
            F_scalar[close_mask] = F_corrected

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc, n_close

    adapt_thresh = 0.05
    max_substeps = 16

    def min_pair_dist_current(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return float(np.sqrt(np.min(r2) + 1e-30))

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, nf = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, nf

    a, n_close = compute_acc(x)
    close_pair_sum = n_close
    pair_sum = P
    total_substeps = 0

    si = 0
    nt = times[0]
    t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < n_samples:
            nt = times[si]

    steps = 0
    dt_f = float(dt)
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist_current(x)
        if r_min < adapt_thresh:
            n_sub = min(max_substeps, max(2, int(math.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                close_pair_sum += nf
                pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x = x + dt_f * vh
            a, nf = compute_acc(x)
            v = vh + 0.5 * dt_f * a
            close_pair_sum += nf
            pair_sum += P
            total_substeps += 1

        t_cur += dt_f
        steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]
        if t_cur >= T - 1e-12:
            break

    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    total_time = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps": steps,
        "dt": dt,
        "T_years": T,
        "n_samples": n_samples,
        "total_time_sec": total_time,
        "time_per_step_sec": total_time / max(steps, 1),
        "nn_eligible_pair_frac": close_pair_sum / max(pair_sum, 1),
        "avg_pairs_per_step": P,
        "total_substeps": total_substeps,
    }


def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    """REBOUND ias15 reference integrator."""
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


# =============================================================================
# Output helpers
# =============================================================================
def save_csv(rows):
    path = os.path.join(OUT_DIR, "ic1_exact_sanity_data.csv")
    fieldnames = [
        "dt", "repeat", "bounded_ias15", "bounded_simon", "agree",
        "lambda", "final_rms", "mean_rms", "max_rms", "speedup",
        "t_ias15", "t_simon", "nn_eligible_pair_frac", "total_substeps",
        "steps", "E0", "min_initial_sep", "center_to_com_before_both",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved {path}")


def summarize_dt(rows, dt):
    subset = [r for r in rows if abs(r["dt"] - dt) < 1e-12]
    n = len(subset)
    n_agree = sum(1 for r in subset if r["agree"])
    n_ref_ej = sum(1 for r in subset if not r["bounded_ias15"])
    n_sim_ej = sum(1 for r in subset if not r["bounded_simon"])
    speed_vals = np.array([r["speedup"] for r in subset], dtype=float)
    lambda_vals = np.array([r["lambda"] for r in subset if np.isfinite(r["lambda"])], dtype=float)
    final_rms_vals = np.array([r["final_rms"] for r in subset if np.isfinite(r["final_rms"])], dtype=float)
    total_ias15 = float(np.sum([r["t_ias15"] for r in subset]))
    total_simon = float(np.sum([r["t_simon"] for r in subset]))
    pop_speedup = total_ias15 / max(total_simon, 1e-12)

    lines = [
        f"dt = {dt:.2f} yr",
        f"  repeats                 : {n}",
        f"  agreement               : {n_agree}/{n} = {n_agree / max(n, 1):.1%}",
        f"  ias15 ejections         : {n_ref_ej}/{n}",
        f"  SIMON ejections         : {n_sim_ej}/{n}",
        f"  mean speedup            : {np.mean(speed_vals):.3f}x +/- {np.std(speed_vals, ddof=1) if len(speed_vals) > 1 else 0.0:.3f}",
        f"  median speedup          : {np.median(speed_vals):.3f}x",
        f"  total/population speedup: {pop_speedup:.3f}x",
        f"  total ias15 time        : {total_ias15:.4f} s",
        f"  total SIMON time        : {total_simon:.4f} s",
    ]
    if len(lambda_vals) > 0:
        lines.extend([
            f"  lambda                  : {np.mean(lambda_vals):.4f} +/- {np.std(lambda_vals, ddof=1) if len(lambda_vals) > 1 else 0.0:.4f} /yr",
            f"  final RMS               : {np.mean(final_rms_vals):.4f} AU",
        ])
    else:
        lines.append("  lambda/final RMS        : not computed because both solvers were not both bounded")
    return lines


def write_summary(rows):
    lines = [
        "=" * 78,
        "EXACT IC1 SANITY BENCHMARK — SUMMARY",
        "=" * 78,
        "",
        "SETUP",
        f"  Exact IC1 only; repeats per dt = {N_REPEATS_PER_DT}",
        f"  T = {T_SIM:.1f} yr, samples/IC = {N_SAMPLES}",
        f"  dt values tested = {DT_VALUES}",
        f"  Ejection threshold = {EJECTION_THRESHOLD:.1f} AU",
        f"  CENTER_TO_COM_BEFORE_BOTH = {CENTER_TO_COM_BEFORE_BOTH}",
        f"  Initial total energy = {rows[0]['E0']:.6g}",
        f"  Initial min pair distance = {rows[0]['min_initial_sep']:.6g} AU",
        "",
        "RESULTS",
    ]
    for dt in DT_VALUES:
        lines.extend(summarize_dt(rows, dt))
        lines.append("")

    ok_by_dt = []
    for dt in DT_VALUES:
        subset = [r for r in rows if abs(r["dt"] - dt) < 1e-12]
        ok = all(r["agree"] for r in subset) and all(r["bounded_ias15"] and r["bounded_simon"] for r in subset)
        ok_by_dt.append(ok)

    lines.extend([
        "INTERPRETATION",
        f"  Exact IC1 classification sanity pass? {'YES' if all(ok_by_dt) else 'NO'}",
        "  If this is YES, the perturbation-benchmark failures are not caused by a",
        "  basic code mismatch in the exact IC1 setup. They instead indicate that the",
        "  perturbations moved the system outside the locally safe IC1 operating region.",
        "  If this is NO, compare this script line-by-line with the original speed-frontier",
        "  / multi_ic_eval_v3 code before making any paper claim.",
        "",
        "OUTPUT FILES",
        "  ic1_exact_sanity_data.csv",
        "  ic1_exact_sanity_summary.txt",
        "=" * 78,
    ])

    path = os.path.join(OUT_DIR, "ic1_exact_sanity_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}\n")
    for line in lines:
        print("  " + line)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 78)
    print("Exact IC1 sanity benchmark")
    print("=" * 78)
    print(f"Output directory: {OUT_DIR}")
    print(f"T={T_SIM}, N_SAMPLES={N_SAMPLES}, dt values={DT_VALUES}")
    print()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Place pair_correction_nn.pt in the same "
            f"folder as this script or update MODEL_PATH."
        )

    print("[1/3] Loading SIMON model ...")
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig()
    print(f"  Loaded {MODEL_PATH} with {sum(p.numel() for p in model.parameters())} parameters")

    x0 = BASE_X0.copy()
    v0 = BASE_V0.copy()
    m = BASE_M.copy()
    if CENTER_TO_COM_BEFORE_BOTH:
        x0, v0 = center_to_com(x0, v0, m)
    E0 = total_energy(x0, v0, m, cfg.G)
    dmin0 = min_pair_distance(x0)

    print("\n[2/3] Running exact IC1 repeats ...")
    rows = []
    for dt in DT_VALUES:
        for rep in range(N_REPEATS_PER_DT):
            print(f"  dt={dt:.2f}, repeat {rep + 1}/{N_REPEATS_PER_DT} ...", end="", flush=True)
            times_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(
                x0, v0, m, cfg.G, T_SIM, N_SAMPLES
            )
            times_sim, pos_sim, vel_sim, perf_sim = simulate_leapfrog_hybrid(
                x0, v0, m, model, cfg, dt, T_SIM, N_SAMPLES
            )

            bounded_ref = is_bounded(pos_ref)
            bounded_sim = is_bounded(pos_sim)
            agree = (bounded_ref == bounded_sim)
            t_ias15 = float(perf_ref["total_time_sec"])
            t_simon = float(perf_sim["total_time_sec"])
            speedup = t_ias15 / max(t_simon, 1e-12)

            if bounded_ref and bounded_sim:
                delta = rms_sep(pos_sim, pos_ref)
                lam = fit_log_slope(times_ref, delta)
                final_rms = float(delta[-1])
                mean_rms = float(np.mean(delta))
                max_rms = float(np.max(delta))
            else:
                lam = float("nan")
                final_rms = float("nan")
                mean_rms = float("nan")
                max_rms = float("nan")

            rows.append({
                "dt": float(dt),
                "repeat": int(rep),
                "bounded_ias15": bool(bounded_ref),
                "bounded_simon": bool(bounded_sim),
                "agree": bool(agree),
                "lambda": float(lam),
                "final_rms": float(final_rms),
                "mean_rms": float(mean_rms),
                "max_rms": float(max_rms),
                "speedup": float(speedup),
                "t_ias15": t_ias15,
                "t_simon": t_simon,
                "nn_eligible_pair_frac": float(perf_sim["nn_eligible_pair_frac"]),
                "total_substeps": int(perf_sim["total_substeps"]),
                "steps": int(perf_sim["steps"]),
                "E0": float(E0),
                "min_initial_sep": float(dmin0),
                "center_to_com_before_both": bool(CENTER_TO_COM_BEFORE_BOTH),
            })
            print(f" agree={agree} bounded(ref/SIMON)=({bounded_ref}/{bounded_sim}) speedup={speedup:.2f}x")

    print("\n[3/3] Saving outputs ...")
    save_csv(rows)
    write_summary(rows)
    print(f"\nAll outputs saved in: {OUT_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
