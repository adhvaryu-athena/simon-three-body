"""
ic1_perturbation_speed_benchmark_dt04.py

Controlled IC1-Like Speed--Accuracy Benchmark for SIMON
=======================================================

Purpose:
    Test whether the speed--accuracy result observed for IC1 at dt=0.04 yr is
    a one-off numerical accident or survives small, pre-defined local
    perturbations of the IC1 initial condition.

    This is NOT a broad population validation and NOT a hard-scattering random
    stress test. It is a local robustness benchmark around the same moderate
    three-body scattering regime used for the paper's speed--accuracy frontier,
    but at the conservative dt=0.04 yr setting after dt=0.08 proved too aggressive.

Benchmark question:
    For a small IC1-like neighbourhood, does SIMON at dt=0.04 yr:
        1. remain in the same bounded/ejected class as ias15?
        2. keep lambda and RMS separation controlled for both-bounded cases?
        3. retain >1x speedup under the same dense-output protocol used by the
           multi-IC / speed-frontier experiments?

IC generation:
    Base IC1:
        m  = [1.0, 0.01, 0.005] Msun
        x0 = [[0,0,0], [1,0,0], [0,1.2,0]] AU
        v0 = [[0,0,0], [0,1,0], [-0.9,0,0]] AU/yr

    The benchmark includes IC1 itself plus N_BENCH-1 local perturbations:
        - masses are perturbed log-normally within small bounds;
        - body-1 and body-2 radii are perturbed locally around 1.0 and 1.2 AU;
        - angular positions are perturbed locally around 0 and pi/2;
        - tangential speeds are perturbed locally around 1.0 and 0.9 AU/yr;
        - small radial velocity components are allowed.

    All filters are pre-integration filters only. No IC is selected or rejected
    based on SIMON or ias15 outcome.

Simulation:
    T=100 yr, dt=0.04 yr, n_samples=5000.
    The dense output count intentionally matches the multi-IC / speed-frontier
    protocol, so the runtime comparison tests the same evaluation workflow used
    to obtain the IC1 conservative-timestep result.

Outputs in ic1_perturb_benchmark_dt04/:
    ic1_perturb_data.csv
    ic1_perturb_summary.txt
    fig1_agreement_matrix.png
    fig2_speedup_distribution.png
    fig3_lambda_distribution.png
    fig4_rms_distribution.png
    fig5_speedup_vs_rms.png

Run:
    python ic1_perturbation_speed_benchmark_dt04.py

Requirements:
    pair_correction_nn.pt in the same directory as this script.
    rebound, torch, numpy, matplotlib.
"""

import os
import time
import math
import csv
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rebound

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif"],
    "font.size":       11,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH         = "pair_correction_nn.pt"
OUT_DIR            = "ic1_perturb_benchmark_dt04"
os.makedirs(OUT_DIR, exist_ok=True)

# Simulation parameters: intentionally aligned with the dense-output protocol
# used in multi_ic_eval_v3.py / the speed--accuracy frontier.
DT                 = 0.04          # yr -- conservative IC1-like timestep
T_SIM              = 100.0         # yr
N_SAMPLES          = 5000          # dense output, matching multi-IC protocol
EJECTION_THRESHOLD = 10.0          # AU -- same bounded/ejected criterion

# Benchmark population size
N_BENCH            = 50            # IC1 + 49 local perturbations
SEED               = 42
MAX_ATTEMPTS       = 200 * N_BENCH

# IC1 local perturbation scales. These are deliberately small so the benchmark
# tests local robustness of the IC1 speed frontier rather than a new population.
MASS_LOG_SIGMA     = 0.08          # ~8% one-sigma multiplicative perturbation
RADIUS_LOG_SIGMA   = 0.08          # ~8% one-sigma radial perturbation
ANGLE_SIGMA        = 0.12          # radians, local angular perturbation
VTAN_LOG_SIGMA     = 0.08          # ~8% one-sigma tangential speed perturbation
VRAD_FRAC_SIGMA    = 0.04          # radial velocity as fraction of tangential speed

# Hard pre-integration safety filters. These prevent pathological immediate
# collisions while preserving the IC1-like moderate-scattering geometry.
MIN_INITIAL_SEPARATION = 0.50      # AU
MAX_INITIAL_RADIUS     = 1.60      # AU for each non-central body
MIN_INITIAL_RADIUS     = 0.70      # AU for each non-central body
MAX_ATTEMPTED_MASS     = 2.00      # Msun, SIMON training upper bound
MIN_ATTEMPTED_MASS     = 0.001     # Msun, SIMON training lower bound

# Reference values from the paper's seven-IC bounded subset.
REF_LAMBDA_MEAN = 0.0916
REF_LAMBDA_STD  = 0.0682

# Base IC1 from multi_ic_eval_v3.py.
BASE_M  = np.array([1.0, 0.01, 0.005], dtype=np.float64)
BASE_X0 = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.2, 0.0]], dtype=np.float64)
BASE_V0 = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-0.9, 0.0, 0.0]], dtype=np.float64)


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
        self.register_buffer("input_std",  torch.ones(3))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


@dataclass
class HybridConfig:
    G:              float = 1.0
    eps:            float = 3e-4
    mc_samples:     int   = 1
    unc_rel_thresh: float = 0.25
    c_min:          float = 0.2
    c_max:          float = 5.0
    r_soft_min:     float = 5e-4


def extract_weights_numpy(model):
    sd = model.state_dict()
    return {
        "mean": sd["input_mean"].cpu().numpy().astype(np.float32),
        "std":  sd["input_std"].cpu().numpy().astype(np.float32) + 1e-8,
        "w0T":  sd["net.0.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b0":   sd["net.0.bias"].cpu().numpy().astype(np.float32),
        "w1T":  sd["net.2.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b1":   sd["net.2.bias"].cpu().numpy().astype(np.float32),
        "w2T":  sd["net.4.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b2":   sd["net.4.bias"].cpu().numpy().astype(np.float32),
        "w3T":  sd["net.6.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b3":   sd["net.6.bias"].cpu().numpy().astype(np.float32),
    }


# =============================================================================
# Integrators
# =============================================================================
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples):
    """SIMON leapfrog integrator with analytic force direction and adaptive sub-stepping."""
    w = extract_weights_numpy(model)
    w_mean = w["mean"]; w_std  = w["std"]
    w0T = w["w0T"];     b0 = w["b0"]
    w1T = w["w1T"];     b1 = w["b1"]
    w2T = w["w2T"];     b2 = w["b2"]
    w3T = w["w3T"];     b3 = w["b3"]

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii); jj = np.array(jj); P = len(ii)

    G          = cfg.G
    eps2       = cfg.eps * cfg.eps
    c_min      = cfg.c_min
    c_max      = cfg.c_max
    r_soft_min = cfg.r_soft_min
    nn_thresh  = 500.0 * cfg.eps   # 0.15 AU

    x      = x0.astype(np.float64).copy()
    v      = v0.astype(np.float64).copy()
    m_f    = m.astype(np.float64)
    mi_arr = m_f[ii]
    mj_arr = m_f[jj]
    Gmimj  = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times   = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum("ij,ij->i", rij, rij)
        r   = np.sqrt(r2 + 1e-30)

        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))
        if n_close > 0:
            r_soft_close = np.sqrt(r2[close_mask] + eps2)
            denom        = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom

            nn_in       = np.empty((n_close, 3), dtype=np.float32)
            nn_in[:, 0] = np.log(r_soft_close + 1e-30).astype(np.float32)
            nn_in[:, 1] = log_mi[close_mask]
            nn_in[:, 2] = log_mj[close_mask]

            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w1T + b1;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w2T + b2;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            log_c = (h @ w3T + b3).ravel()
            c     = np.exp(log_c).astype(np.float64)

            fallback = ((r_soft_close < r_soft_min) |
                        (c < c_min) | (c > c_max) | ~np.isfinite(c))
            F_corrected = np.where(fallback, F_scalar[close_mask], c * F_soft_close)
            F_scalar[close_mask] = F_corrected

        F_vec = F_scalar[:, None] * rij
        acc   = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc, n_close

    adapt_thresh = 0.05
    max_substeps = 16

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum("ij,ij->i", rij, rij)
        return float(np.sqrt(np.min(r2) + 1e-30))

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh    = v_in + 0.5 * sub_dt * a_in
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
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(math.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                close_pair_sum += nf
                pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
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
    pos   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0    = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


# =============================================================================
# Diagnostics and IC generation
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
    d  = a - b
    pb = np.sqrt(np.sum(d ** 2, axis=-1))
    return np.sqrt(np.mean(pb ** 2, axis=1))


def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T_   = times[-1]
    t0 = t0_frac * T_
    t1 = t1_frac * T_
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


def accept_ic(x, v, m):
    if np.any(m < MIN_ATTEMPTED_MASS) or np.any(m > MAX_ATTEMPTED_MASS):
        return False
    radii = np.linalg.norm(x[1:], axis=1)
    if np.any(radii < MIN_INITIAL_RADIUS) or np.any(radii > MAX_INITIAL_RADIUS):
        return False
    if min_pair_distance(x) < MIN_INITIAL_SEPARATION:
        return False
    if not np.isfinite(total_energy(x, v, m)):
        return False
    if total_energy(x, v, m) >= 0.0:
        return False
    return True


def generate_perturbed_ic(rng, ic_id):
    """Generate one local IC1-like perturbation using only pre-integration rules."""
    if ic_id == 0:
        m = BASE_M.copy()
        x = BASE_X0.copy()
        v = BASE_V0.copy()
        x, v = center_to_com(x, v, m)
        return x, v, m, total_energy(x, v, m), min_pair_distance(x), "IC1_base"

    for _ in range(MAX_ATTEMPTS):
        # Masses remain close to IC1 but inside the scalar NN training range.
        m = BASE_M * np.exp(rng.normal(0.0, MASS_LOG_SIGMA, size=3))

        # Local polar-coordinate perturbation around IC1 geometry.
        r1 = 1.0 * np.exp(rng.normal(0.0, RADIUS_LOG_SIGMA))
        r2 = 1.2 * np.exp(rng.normal(0.0, RADIUS_LOG_SIGMA))
        th1 = 0.0 + rng.normal(0.0, ANGLE_SIGMA)
        th2 = 0.5 * np.pi + rng.normal(0.0, ANGLE_SIGMA)

        rhat1 = np.array([np.cos(th1), np.sin(th1), 0.0])
        that1 = np.array([-np.sin(th1), np.cos(th1), 0.0])
        rhat2 = np.array([np.cos(th2), np.sin(th2), 0.0])
        that2 = np.array([-np.sin(th2), np.cos(th2), 0.0])

        x = np.zeros((3, 3), dtype=np.float64)
        x[1] = r1 * rhat1
        x[2] = r2 * rhat2

        vtan1 = 1.0 * np.exp(rng.normal(0.0, VTAN_LOG_SIGMA))
        vtan2 = 0.9 * np.exp(rng.normal(0.0, VTAN_LOG_SIGMA))
        vrad1 = rng.normal(0.0, VRAD_FRAC_SIGMA) * vtan1
        vrad2 = rng.normal(0.0, VRAD_FRAC_SIGMA) * vtan2

        v = np.zeros((3, 3), dtype=np.float64)
        v[1] = vtan1 * that1 + vrad1 * rhat1
        v[2] = vtan2 * that2 + vrad2 * rhat2

        x, v = center_to_com(x, v, m)
        if accept_ic(x, v, m):
            return x, v, m, total_energy(x, v, m), min_pair_distance(x), "IC1_perturbed"

    raise RuntimeError("Unable to generate a valid IC1-like perturbation. Relax filters.")


def make_benchmark_ics():
    rng = np.random.default_rng(SEED)
    ics = []
    for ic_id in range(N_BENCH):
        ics.append(generate_perturbed_ic(rng, ic_id))
    return ics


# =============================================================================
# Plots and output
# =============================================================================
def save_csv(rows):
    path = os.path.join(OUT_DIR, "ic1_perturb_data.csv")
    fieldnames = [
        "ic_id", "kind", "m0", "m1", "m2", "E0", "min_initial_sep",
        "bounded_ias15", "bounded_simon", "agree", "lambda", "final_rms",
        "mean_rms", "max_rms", "speedup", "t_ias15", "t_simon",
        "nn_eligible_pair_frac", "total_substeps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved {path}")


def plot_agreement_matrix(rows):
    valid = [r for r in rows if r["bounded_ias15"] is not None]
    both_bounded = sum(1 for r in valid if r["bounded_ias15"] and r["bounded_simon"])
    ref_ej_sim_bnd = sum(1 for r in valid if (not r["bounded_ias15"]) and r["bounded_simon"])
    ref_bnd_sim_ej = sum(1 for r in valid if r["bounded_ias15"] and (not r["bounded_simon"]))
    both_ejected = sum(1 for r in valid if (not r["bounded_ias15"]) and (not r["bounded_simon"]))
    mat = np.array([[both_bounded, ref_bnd_sim_ej],
                    [ref_ej_sim_bnd, both_ejected]], dtype=int)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["SIMON\nbounded", "SIMON\nejected"])
    ax.set_yticklabels(["ias15\nbounded", "ias15\nejected"])
    ax.set_title("IC1-like benchmark: outcome agreement")
    total = max(len(valid), 1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}\n({mat[i, j] / total:.1%})",
                    ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_agreement_matrix.png")
    fig.savefig(path)
    plt.close(fig)


def plot_speedup_distribution(rows):
    vals = np.array([r["speedup"] for r in rows if np.isfinite(r["speedup"])], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.hist(vals, bins=min(12, max(5, len(vals) // 4)), edgecolor="black", alpha=0.75)
    ax.axvline(1.0, linestyle="--", linewidth=1.2, label="1x")
    if len(vals) > 0:
        ax.axvline(np.mean(vals), linestyle="-", linewidth=1.2,
                   label=f"mean={np.mean(vals):.2f}x")
    ax.set_xlabel("Speedup over ias15 (t_ias15 / t_SIMON)")
    ax.set_ylabel("Number of ICs")
    ax.set_title("IC1-like benchmark speedup distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_speedup_distribution.png")
    fig.savefig(path)
    plt.close(fig)


def plot_lambda_distribution(rows):
    vals = np.array([r["lambda"] for r in rows if np.isfinite(r["lambda"])], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if len(vals) > 0:
        ax.hist(vals, bins=min(12, max(5, len(vals) // 4)), edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(vals), linestyle="-", linewidth=1.2,
                   label=f"mean={np.mean(vals):.3f}/yr")
        ax.axvline(REF_LAMBDA_MEAN, linestyle="--", linewidth=1.2,
                   label="7-IC mean")
    ax.set_xlabel(r"Divergence-rate proxy $\lambda$ (yr$^{-1}$)")
    ax.set_ylabel("Both-bounded ICs")
    ax.set_title("IC1-like benchmark lambda distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_lambda_distribution.png")
    fig.savefig(path)
    plt.close(fig)


def plot_rms_distribution(rows):
    vals = np.array([r["final_rms"] for r in rows if np.isfinite(r["final_rms"])], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if len(vals) > 0:
        ax.hist(vals, bins=min(12, max(5, len(vals) // 4)), edgecolor="black", alpha=0.75)
        ax.axvline(np.median(vals), linestyle="-", linewidth=1.2,
                   label=f"median={np.median(vals):.2f} AU")
    ax.set_xlabel("Final RMS separation from ias15 (AU)")
    ax.set_ylabel("Both-bounded ICs")
    ax.set_title("IC1-like benchmark RMS distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_rms_distribution.png")
    fig.savefig(path)
    plt.close(fig)


def plot_speedup_vs_rms(rows):
    xs = []
    ys = []
    for r in rows:
        if np.isfinite(r["speedup"]) and np.isfinite(r["final_rms"]):
            xs.append(r["final_rms"])
            ys.append(r["speedup"])
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if len(xs) > 0:
        ax.scatter(xs, ys, s=32, alpha=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Final RMS separation from ias15 (AU)")
    ax.set_ylabel("Speedup over ias15")
    ax.set_title("Speedup vs RMS for both-bounded ICs")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_speedup_vs_rms.png")
    fig.savefig(path)
    plt.close(fig)


def write_summary(rows, n_failed):
    valid = [r for r in rows if r["bounded_ias15"] is not None]
    n_valid = len(valid)
    n_total = n_valid + n_failed

    both_bounded = sum(1 for r in valid if r["bounded_ias15"] and r["bounded_simon"])
    ref_ej_sim_bnd = sum(1 for r in valid if (not r["bounded_ias15"]) and r["bounded_simon"])
    ref_bnd_sim_ej = sum(1 for r in valid if r["bounded_ias15"] and (not r["bounded_simon"]))
    both_ejected = sum(1 for r in valid if (not r["bounded_ias15"]) and (not r["bounded_simon"]))

    n_ref_ej = sum(1 for r in valid if not r["bounded_ias15"])
    n_sim_ej = sum(1 for r in valid if not r["bounded_simon"])
    n_agree = sum(1 for r in valid if r["agree"])

    speed_vals = np.array([r["speedup"] for r in valid if np.isfinite(r["speedup"])], dtype=float)
    lam_vals = np.array([r["lambda"] for r in valid if np.isfinite(r["lambda"])], dtype=float)
    final_rms_vals = np.array([r["final_rms"] for r in valid if np.isfinite(r["final_rms"])], dtype=float)
    mean_rms_vals = np.array([r["mean_rms"] for r in valid if np.isfinite(r["mean_rms"])], dtype=float)
    max_rms_vals = np.array([r["max_rms"] for r in valid if np.isfinite(r["max_rms"])], dtype=float)

    total_t_ias15 = float(np.nansum([r["t_ias15"] for r in valid]))
    total_t_simon = float(np.nansum([r["t_simon"] for r in valid]))
    population_speedup = total_t_ias15 / max(total_t_simon, 1e-12)

    speed_claim_ok = (n_valid > 0 and
                      n_agree / n_valid >= 0.90 and
                      both_bounded >= 0.70 * n_valid and
                      population_speedup > 1.0 and
                      np.median(speed_vals) > 1.0)

    lines = [
        "=" * 74,
        "CONTROLLED IC1-LIKE SPEED--ACCURACY BENCHMARK — SUMMARY",
        "=" * 74,
        "",
        "SETUP",
        f"  N_BENCH        : {N_BENCH}",
        f"  Valid ICs      : {n_valid}  (failed during simulation: {n_failed})",
        f"  T=100yr, dt={DT:.2f}yr  (conservative IC1-like timestep)",
        f"  Samples/IC     : {N_SAMPLES}  (dense-output protocol)",
        "  Population     : IC1 base + local pre-integration perturbations",
        f"  Perturbations  : mass sigma={MASS_LOG_SIGMA:.2f}, radius sigma={RADIUS_LOG_SIGMA:.2f}, "
        f"angle sigma={ANGLE_SIGMA:.2f} rad, vtan sigma={VTAN_LOG_SIGMA:.2f}",
        f"  IC filters     : min pair distance >= {MIN_INITIAL_SEPARATION:.2f} AU, "
        f"radii in [{MIN_INITIAL_RADIUS:.2f},{MAX_INITIAL_RADIUS:.2f}] AU, E_total < 0",
        f"  Seed           : {SEED}",
        "",
        "OUTCOME CLASSIFICATION",
        f"  ias15 ejection rate  : {n_ref_ej / max(n_valid, 1):.1%}  ({n_ref_ej}/{n_valid} ICs)",
        f"  SIMON ejection rate  : {n_sim_ej / max(n_valid, 1):.1%}  ({n_sim_ej}/{n_valid} ICs)",
        f"  Agreement rate       : {n_agree / max(n_valid, 1):.1%}  ({n_agree}/{n_valid} ICs)",
        "",
        "  Confusion matrix (ias15 rows x SIMON cols):",
        f"    Both bounded        : {both_bounded:4d}  ({both_bounded / max(n_valid, 1):.1%})",
        f"    ias15 ej, SIMON bnd : {ref_ej_sim_bnd:4d}  ({ref_ej_sim_bnd / max(n_valid, 1):.1%})",
        f"    ias15 bnd, SIMON ej : {ref_bnd_sim_ej:4d}  ({ref_bnd_sim_ej / max(n_valid, 1):.1%})",
        f"    Both ejected        : {both_ejected:4d}  ({both_ejected / max(n_valid, 1):.1%})",
        "",
        "DIVERGENCE RATE AND RMS ERROR (both-bounded ICs only)",
    ]

    if len(lam_vals) > 0:
        lines += [
            f"  Population lambda : {np.mean(lam_vals):.4f} +/- {np.std(lam_vals, ddof=1) if len(lam_vals) > 1 else 0.0:.4f} /yr  (N={len(lam_vals)})",
            f"  7-IC reference    : {REF_LAMBDA_MEAN:.4f} +/- {REF_LAMBDA_STD:.4f} /yr",
            f"  Final RMS         : {np.mean(final_rms_vals):.4f} +/- {np.std(final_rms_vals, ddof=1) if len(final_rms_vals) > 1 else 0.0:.4f} AU  "
            f"(median {np.median(final_rms_vals):.4f} AU)",
            f"  Mean RMS          : {np.mean(mean_rms_vals):.4f} AU",
            f"  Max RMS           : {np.mean(max_rms_vals):.4f} AU",
        ]
    else:
        lines += ["  No both-bounded ICs available for lambda/RMS statistics."]

    lines += [
        "",
        "COMPUTATIONAL PERFORMANCE",
        f"  Mean speedup      : {np.mean(speed_vals):.3f}x +/- {np.std(speed_vals, ddof=1) if len(speed_vals) > 1 else 0.0:.3f}",
        f"  Median speedup    : {np.median(speed_vals):.3f}x",
        f"  Total ias15       : {total_t_ias15:.3f} s",
        f"  Total SIMON       : {total_t_simon:.3f} s",
        f"  Population speedup: {population_speedup:.3f}x",
        "",
        "INTERPRETATION GATE",
        f"  Use for speed claim? : {'YES' if speed_claim_ok else 'NO'}",
        "  Gate condition       : agreement >= 90%, most cases both-bounded, median speedup > 1x, population speedup > 1x",
        "",
        "PAPER-SAFE FRAMING",
        "  If the gate is YES, this benchmark supports the narrow claim that the IC1",
        "  speed-frontier result is locally robust to small pre-defined perturbations.",
        "  If the gate is NO, treat the result as evidence that even the conservative dt=0.04",
        "  operating point needs tighter calibration or a narrower IC1 neighbourhood.",
        "",
        "OUTPUT FILES",
        "  ic1_perturb_data.csv",
        "  fig1_agreement_matrix.png",
        "  fig2_speedup_distribution.png",
        "  fig3_lambda_distribution.png",
        "  fig4_rms_distribution.png",
        "  fig5_speedup_vs_rms.png",
        "=" * 74,
    ]

    path = os.path.join(OUT_DIR, "ic1_perturb_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {path}\n")
    for line in lines:
        print("  " + line)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 74)
    print("Controlled IC1-like speed--accuracy benchmark at dt=0.04")
    print("=" * 74)
    print(f"Output directory: {OUT_DIR}")
    print(f"N_BENCH={N_BENCH}, T={T_SIM}, dt={DT}, N_SAMPLES={N_SAMPLES}")
    print()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Place pair_correction_nn.pt in the same "
            f"folder as this script or update MODEL_PATH."
        )

    print("[1/4] Loading SIMON model ...")
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig()
    print(f"  Loaded {MODEL_PATH} with {sum(p.numel() for p in model.parameters())} parameters")

    print("\n[2/4] Generating IC1-like benchmark set ...")
    ics = make_benchmark_ics()
    print(f"  Generated {len(ics)} ICs using only pre-integration filters")

    print("\n[3/4] Running ias15 and SIMON ...")
    rows = []
    n_failed = 0
    for ic_id, (x0, v0, m, E0, dmin0, kind) in enumerate(ics):
        print(f"  IC {ic_id + 1:02d}/{len(ics)} [{kind}] ...", end="", flush=True)
        try:
            times_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(
                x0, v0, m, cfg.G, T_SIM, N_SAMPLES
            )
            times_sim, pos_sim, vel_sim, perf_sim = simulate_leapfrog_hybrid(
                x0, v0, m, model, cfg, DT, T_SIM, N_SAMPLES
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
                "ic_id": ic_id,
                "kind": kind,
                "m0": float(m[0]), "m1": float(m[1]), "m2": float(m[2]),
                "E0": float(E0),
                "min_initial_sep": float(dmin0),
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
            })
            print(f" agree={agree} speedup={speedup:.2f}x")
        except Exception as exc:
            n_failed += 1
            print(f" FAILED: {exc}")
            rows.append({
                "ic_id": ic_id,
                "kind": kind,
                "m0": float(m[0]), "m1": float(m[1]), "m2": float(m[2]),
                "E0": float(E0),
                "min_initial_sep": float(dmin0),
                "bounded_ias15": None,
                "bounded_simon": None,
                "agree": None,
                "lambda": float("nan"),
                "final_rms": float("nan"),
                "mean_rms": float("nan"),
                "max_rms": float("nan"),
                "speedup": float("nan"),
                "t_ias15": float("nan"),
                "t_simon": float("nan"),
                "nn_eligible_pair_frac": float("nan"),
                "total_substeps": -1,
            })

    print("\n[4/4] Saving outputs ...")
    save_csv(rows)
    plot_agreement_matrix(rows)
    plot_speedup_distribution(rows)
    plot_lambda_distribution(rows)
    plot_rms_distribution(rows)
    plot_speedup_vs_rms(rows)
    write_summary(rows, n_failed)

    print(f"\nAll outputs saved in: {OUT_DIR}/")
    print("=" * 74)


if __name__ == "__main__":
    main()
