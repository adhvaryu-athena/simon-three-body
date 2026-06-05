"""
population_validation08.py

Experiment B: Moderate-Scattering Stress Population for SIMON
=============================================================

Purpose:
    Run a pre-defined moderate-scattering stress population to test whether the
    speed--accuracy behaviour observed on IC1 generalises beyond one hand-picked
    configuration. This is intentionally not a broad hierarchical population.

    The benchmark asks:

    1. Does SIMON agree with ias15 on bounded/ejected classification?
    2. For configurations bounded in both solvers, are lambda and RMS separation
       controlled relative to the reference trajectory?
    3. Does SIMON achieve speedup over ias15 at dt=0.08 yr on compact,
       interaction-rich three-body cases?

    Use the result for speed claims only if classification agreement remains high.

IC generation:
    Masses sampled log-uniformly from within SIMON's training range [0.001, 2.0] Msun:
        m0 (primary):   log-uniform [0.50, 2.00] Msun
        m1 (secondary): log-uniform [0.005, 0.50] Msun
        m2 (tertiary):  log-uniform [0.001, 0.10] Msun
    Positions: compact, non-pathological, interaction-rich geometry.
    Velocities: tangential, scaled from circular velocity by random factor.
    Accept if: E_total < 0, initial separations are safe, and the estimated
       outer periapsis crosses into the inner-orbit interaction zone.

Simulation:
    T=100yr, dt=0.08yr, n_samples=500
    EJECTION_THRESHOLD=10 AU (same as main paper experiments)

Outputs (in population_validation_stress_dt08/):
    pop_data.csv                    -- per-IC raw results (for supplementary)
    pop_summary.txt                 -- statistics for paper paragraph
    fig1_agreement_matrix.png       -- 2x2 outcome matrix (paper-ready)
    fig2_speedup_distribution.png   -- speedup histogram across 500 ICs
    fig3_lambda_distribution.png    -- lambda distribution (both-bounded ICs only)
    fig4_rms_distribution.png       -- RMS-separation distribution (both-bounded ICs only)

Paper paragraph:
    Use pop_summary.txt verbatim in Section 5.2.x.

Run:
    python population_validation08.py
    Expected runtime: a few minutes; stress cases may be slower than the broad population.

Requirements:
    pair_correction_nn.pt   (in same directory as this script)
    rebound, torch, numpy, matplotlib
"""

import os
import time
import math
import csv

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
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
OUT_DIR            = "population_validation_stress_dt08"
os.makedirs(OUT_DIR, exist_ok=True)

# Simulation parameters
DT                 = 0.08          # yr -- speed-oriented stress-test timestep
T_SIM              = 100.0         # yr
N_SAMPLES          = 500           # samples per IC (0.2 yr spacing)
EJECTION_THRESHOLD = 10.0          # AU -- same as main paper

# Population parameters
N_POP              = 500           # number of random ICs to evaluate
SEED               = 42            # random seed for reproducibility

# Mass ranges -- all within SIMON training range [0.001, 2.0] Msun
M0_RANGE = (0.50, 2.00)    # primary: main-sequence stars
M1_RANGE = (0.005, 0.50)   # secondary: planets to companion stars
M2_RANGE = (0.001, 0.10)   # tertiary: planets to low-mass companions

# Separation ranges for Experiment B stress population.
# These deliberately create compact, interaction-rich systems, unlike the broad
# filtered hierarchical population. The filters below prevent pathological
# immediate collisions while allowing moderate scattering.
R1_RANGE = (0.8, 2.0)      # inner separation (AU)
R2_RANGE = (1.5, 4.0)      # outer initial radius (AU), forced to be > r1 + gap
R2_GAP_MIN = 0.25          # AU; keeps body 2 initially outside body 1

# Velocity scale factors (as fraction of circular velocity).
# Wider than the broad population to create eccentric, interacting cases, but
# still below/near escape to avoid a trivial mostly-unbound sample.
F1_RANGE = (0.75, 1.30)    # inner body: moderate eccentricity
F2_RANGE = (0.55, 1.15)    # outer body: eccentric/crossing trajectories allowed

# IC rejection thresholds for a fair pre-integration stress benchmark.
MIN_INITIAL_SEPARATION = 0.25   # AU; avoid immediate numerical collisions
MIN_INNER_PERIAPSIS   = 0.15   # AU; avoid persistent sub-softening plunges
MIN_OUTER_PERIAPSIS   = 0.15   # AU; avoid pathological primary plunges
MAX_OUTER_PERIAPSIS_FACTOR = 1.35  # require q_outer <= 1.35*r1 for interaction
MAX_ATTEMPTS = 300 * N_POP     # stress filters are stricter than broad population

# Reference values from 7-IC experiment (for comparison in summary)
REF_LAMBDA_MEAN = 0.0916   # /yr (mean lambda over bounded ICs in 7-IC study)
REF_LAMBDA_STD  = 0.0682   # /yr


# =============================================================================
# Model (verbatim from multi_ic_eval_v3.py)
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
        return self.net(
            (x - self.input_mean) / (self.input_std + 1e-8)
        ).squeeze(-1)


@dataclass
class HybridConfig:
    G:             float = 1.0
    eps:           float = 3e-4
    mc_samples:    int   = 1
    unc_rel_thresh:float = 0.25
    c_min:         float = 0.2
    c_max:         float = 5.0
    r_soft_min:    float = 5e-4


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
# Integrators (verbatim from multi_ic_eval_v3.py with FP guard added)
# =============================================================================
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples):
    """SIMON leapfrog integrator. Floating-point guard added."""
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

    x     = x0.astype(np.float64).copy()
    v     = v0.astype(np.float64).copy()
    m_f   = m.astype(np.float64)
    mi_arr = m_f[ii]; mj_arr = m_f[jj]
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
            nn_in        = np.empty((n_close, 3), dtype=np.float32)
            nn_in[:, 0]  = np.log(r_soft_close + 1e-30).astype(np.float32)
            nn_in[:, 1]  = log_mi[close_mask]
            nn_in[:, 2]  = log_mj[close_mask]
            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w1T + b1;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w2T + b2;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            log_c = (h @ w3T + b3).ravel()
            c     = np.exp(log_c).astype(np.float64)
            fb    = ((r_soft_close < r_soft_min) |
                     (c < c_min) | (c > c_max) | ~np.isfinite(c))
            F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_close)
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

    a, n_fb = compute_acc(x)
    fb_sum  = n_fb; pair_sum = P
    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt)
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(math.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
        t_cur += dt_f; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12:
            break

    # Floating-point guard: fill any unwritten trailing samples
    while si < n_samples:
        pos_out[si] = x; vel_out[si] = v; si += 1

    total_time = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps":             steps,
        "total_time_sec":    total_time,
        "avg_fallback_frac": fb_sum / max(pair_sum, 1),
    }


def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    """REBOUND ias15 reference integrator."""
    sim = rebound.Simulation()
    sim.integrator = "ias15"; sim.G = G
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


def rms_sep(a, b):
    """Per-timestep RMS position separation between two trajectories."""
    d  = a - b
    pb = np.sqrt(np.sum(d ** 2, axis=-1))
    return np.sqrt(np.mean(pb ** 2, axis=1))


def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    """Fit log-linear slope to divergence curve (divergence rate lambda)."""
    T_   = times[-1]; t0 = t0_frac * T_; t1 = t1_frac * T_
    mask = (times >= t0) & (times <= t1)
    if mask.sum() < 4:
        return float("nan")
    x  = times[mask]
    y  = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean(); y0 = y.mean()
    slope = float(
        np.sum((x - x0) * (y - y0)) / (np.sum((x - x0) ** 2) + 1e-30)
    )
    return slope


def estimate_periapsis(r, v, mu):
    """
    Estimate two-body periapsis for a tangential launch at radius r.

    Uses:
        vis-viva:  v^2 = mu * (2/r - 1/a)
        h = r v
        e^2 = 1 - h^2/(mu a)
        q = a(1-e)

    Returns None if the two-body orbit is unbound.
    """
    denom = 2.0 * mu / r - v * v
    if denom <= 0.0:
        return None

    a = mu / denom
    if a <= 0.0 or not np.isfinite(a):
        return None

    h = r * v
    e_sq = 1.0 - (h * h) / (mu * a)
    e = np.sqrt(max(0.0, e_sq))
    q = a * (1.0 - e)

    if not np.isfinite(q):
        return None
    return float(q)


def is_bounded(pos, threshold=EJECTION_THRESHOLD):
    """True if all bodies remain within threshold AU of the CoM at final time."""
    # Use the final-position check consistent with the main paper
    final_rmax = float(np.max(np.linalg.norm(pos[-1], axis=1)))
    return final_rmax < threshold


# =============================================================================
# Random IC generator
# =============================================================================
def generate_random_ic(rng, G=1.0):
    """
    Generate one compact moderate-scattering initial condition.

    This is Experiment B, not a broad random population. The filter is defined
    entirely from the initial geometry before integration:
        - total energy must be negative;
        - initial pair separations must be safe;
        - inner periapsis must avoid persistent near-collision behaviour;
        - outer periapsis must enter the inner interaction zone, making the
          problem harder for adaptive reference integration than a smooth
          hierarchical orbit.

    Returns a CoM-centred IC tuple plus diagnostic parameters, or None if the
    candidate fails the pre-integration stress-population filters.
    """
    # Sample masses log-uniformly within SIMON training range
    m0 = np.exp(rng.uniform(np.log(M0_RANGE[0]), np.log(M0_RANGE[1])))
    m1 = np.exp(rng.uniform(np.log(M1_RANGE[0]), np.log(M1_RANGE[1])))
    m2 = np.exp(rng.uniform(np.log(M2_RANGE[0]), np.log(M2_RANGE[1])))
    m  = np.array([m0, m1, m2])

    # Sample separations and random orbital angles
    r1     = rng.uniform(R1_RANGE[0], R1_RANGE[1])
    theta1 = rng.uniform(0.0, 2.0 * np.pi)
    r2_low = max(R2_RANGE[0], r1 + R2_GAP_MIN)
    if r2_low >= R2_RANGE[1]:
        return None
    r2     = rng.uniform(r2_low, R2_RANGE[1])
    theta2 = rng.uniform(0.0, 2.0 * np.pi)

    # Positions (in xy plane; z=0)
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [r1 * np.cos(theta1), r1 * np.sin(theta1), 0.0],
        [r2 * np.cos(theta2), r2 * np.sin(theta2), 0.0],
    ])

    # Circular velocities using the relevant two-body gravitational parameters.
    # These are still approximate because the full system is three-body, but this
    # is a better IC filter than using m0 alone when m1 is non-negligible.
    mu1 = G * (m0 + m1)
    mu2 = G * (m0 + m1 + m2)

    v_circ1 = np.sqrt(mu1 / r1)
    v_circ2 = np.sqrt(mu2 / r2)

    # Eccentricity scale factors
    f1 = rng.uniform(F1_RANGE[0], F1_RANGE[1])
    f2 = rng.uniform(F2_RANGE[0], F2_RANGE[1])

    v1 = f1 * v_circ1
    v2 = f2 * v_circ2

    # Reject ICs outside the intended moderate-scattering stress regime.
    q1_est = estimate_periapsis(r1, v1, mu1)
    if q1_est is None or q1_est < MIN_INNER_PERIAPSIS:
        return None

    q2_est = estimate_periapsis(r2, v2, mu2)
    if q2_est is None or q2_est < MIN_OUTER_PERIAPSIS:
        return None
    if q2_est > MAX_OUTER_PERIAPSIS_FACTOR * r1:
        return None

    # Tangential velocity: perpendicular to radius vector in xy plane
    # r_hat = [cos(theta), sin(theta), 0]
    # t_hat = [-sin(theta), cos(theta), 0]
    v0 = np.array([
        [0.0, 0.0, 0.0],
        [-v1 * np.sin(theta1),  v1 * np.cos(theta1),  0.0],
        [-v2 * np.sin(theta2),  v2 * np.cos(theta2),  0.0],
    ])

    # Centre-of-mass frame
    M  = m.sum()
    x0 -= (m[:, None] * x0).sum(axis=0) / M
    v0 -= (m[:, None] * v0).sum(axis=0) / M

    # Total energy
    KE = 0.5 * np.sum(m * np.sum(v0 ** 2, axis=1))
    PE = 0.0
    min_r = 1e9
    for i in range(3):
        for j in range(i + 1, 3):
            r = float(np.linalg.norm(x0[i] - x0[j]))
            if r < min_r:
                min_r = r
            PE -= G * m[i] * m[j] / r
    E_total = KE + PE

    # Reject if unbound or initial separation too small for stable integration
    if E_total >= 0.0:
        return None
    if min_r < MIN_INITIAL_SEPARATION:
        return None

    return (x0, v0, m, float(E_total), float(r1), float(r2),
            float(q1_est), float(q2_est), float(f1), float(f2), float(min_r))


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 65)
    print("EXPERIMENT B — MODERATE-SCATTERING STRESS POPULATION")
    print(f"  N_POP={N_POP}  T={T_SIM}yr  dt={DT}yr  seed={SEED}")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────
    print("\n[1/4] Loading SIMON model ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. "
            f"Run this script from the directory containing pair_correction_nn.pt."
        )
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {MODEL_PATH} ({n_params} parameters)")

    # ── Generate 500 random ICs ───────────────────────────────────────
    print(f"\n[2/4] Generating {N_POP} random ICs (seed={SEED}) ...")
    rng = np.random.RandomState(SEED)
    ics = []
    attempts = 0
    
    while len(ics) < N_POP and attempts < MAX_ATTEMPTS:
        attempts += 1
        result = generate_random_ic(rng)
        if result is not None:
            ics.append(result)

    if len(ics) < N_POP:
        raise RuntimeError(
            f"Only generated {len(ics)} valid ICs after {attempts} attempts. "
            f"Relax the IC filters or increase MAX_ATTEMPTS."
        )

    print(f"  Generated {N_POP} bound, compact stress ICs from {attempts} attempts "
          f"(acceptance rate {N_POP/attempts:.1%})")

    # ── Run simulations ───────────────────────────────────────────────
    print(f"\n[3/4] Running {N_POP} × 2 integrations "
          f"(SIMON + ias15 each) ...")
    print(f"  Progress printed every 50 ICs.\n")

    records = []
    t_wall_start = time.perf_counter()

    for ic_idx, (x0, v0, m, E0, r1_init, r2_init, q1_est, q2_est, f1, f2, min_r0) in enumerate(ics):
        if (ic_idx + 1) % 50 == 0 or ic_idx == 0:
            elapsed = time.perf_counter() - t_wall_start
            rate    = (ic_idx + 1) / max(elapsed, 1e-6)
            eta     = (N_POP - ic_idx - 1) / max(rate, 1e-6)
            print(f"  IC {ic_idx+1:4d}/{N_POP}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s")

        try:
            # ias15 reference
            _, pos_ref, _, perf_ref = simulate_rebound_ias15(
                x0, v0, m, cfg.G, T_SIM, N_SAMPLES)
            t_ias15 = perf_ref["total_time_sec"]

            # SIMON
            times, pos_sim, _, perf_sim = simulate_leapfrog_hybrid(
                x0, v0, m, model, cfg, DT, T_SIM, N_SAMPLES)
            t_simon  = perf_sim["total_time_sec"]
            
            nn_eligible_frac = perf_sim["avg_fallback_frac"]
            speedup = t_ias15 / max(t_simon, 1e-12)

            # Outcome: bounded/ejected
            bounded_ref   = is_bounded(pos_ref)
            bounded_simon = is_bounded(pos_sim)
            agree         = (bounded_ref == bounded_simon)

            # Lambda and RMS separation are meaningful only when both
            # integrators keep the configuration bounded. Post-ejection RMS is
            # dominated by chaotic branch divergence, so it is not used for the
            # speed--accuracy stress metric.
            if bounded_ref and bounded_simon:
                delta = rms_sep(pos_sim, pos_ref)
                lam = fit_log_slope(times, delta)
                final_rms = float(delta[-1])
                mean_rms = float(np.mean(delta))
                max_rms = float(np.max(delta))
            else:
                lam = float("nan")
                final_rms = float("nan")
                mean_rms = float("nan")
                max_rms = float("nan")

            records.append({
                "ic_id":           ic_idx,
                "m0":              m[0], "m1": m[1], "m2": m[2],

                "r1_init":         r1_init, "r2_init": r2_init,
                "q1_est":          q1_est, "q2_est": q2_est,
                "f1":              f1, "f2": f2, "min_r0": min_r0,
                "E0":              E0,

                "bounded_ias15":   bounded_ref,
                "bounded_simon":   bounded_simon,
                "agree":           agree,
                "lambda":          lam,
                "final_rms":       final_rms,
                "mean_rms":        mean_rms,
                "max_rms":         max_rms,
                "speedup":         speedup,
                "nn_eligible_frac": nn_eligible_frac,
                "t_ias15":         t_ias15,
                "t_simon":         t_simon,
            })

        except Exception as exc:
            print(f"  WARNING: IC {ic_idx} failed ({exc}); skipping.")
            records.append({
                
                "ic_id": ic_idx, "m0": m[0], "m1": m[1], "m2": m[2],
                "r1_init": r1_init, "r2_init": r2_init,
                "q1_est": q1_est, "q2_est": q2_est,
                "f1": f1, "f2": f2, "min_r0": min_r0, "E0": E0,
                "bounded_ias15": None, "bounded_simon": None, "agree": None,
                "lambda": float("nan"), "final_rms": float("nan"),
                "mean_rms": float("nan"), "max_rms": float("nan"),
                "speedup": float("nan"), "nn_eligible_frac": float("nan"),
                "t_ias15": float("nan"), "t_simon": float("nan"),

            })

    total_wall = time.perf_counter() - t_wall_start
    print(f"\n  All {N_POP} ICs complete in {total_wall:.1f}s "
          f"({total_wall/N_POP:.2f}s per IC)")

    # ── Compute statistics ────────────────────────────────────────────
    print("\n[4/4] Computing statistics and generating figures ...")

    # Filter valid records
    valid = [r for r in records if r["agree"] is not None]
    n_valid = len(valid)

    # Ejection rates
    n_ej_ias15 = sum(1 for r in valid if not r["bounded_ias15"])
    n_ej_simon = sum(1 for r in valid if not r["bounded_simon"])
    ej_rate_ias15 = n_ej_ias15 / n_valid
    ej_rate_simon = n_ej_simon / n_valid

    # Agreement
    n_agree   = sum(1 for r in valid if r["agree"])
    agree_rate = n_agree / n_valid

    # Confusion matrix: (ias15_outcome × simon_outcome)
    # both_bound  = ias15 bounded  AND simon bounded
    # ias_ej_sim_bound = ias15 ejects, simon bounded
    # ias_bound_sim_ej = ias15 bounded, simon ejects
    # both_ej = both eject
    both_bound       = sum(1 for r in valid if r["bounded_ias15"] and r["bounded_simon"])
    ias_ej_sim_bound = sum(1 for r in valid if (not r["bounded_ias15"]) and r["bounded_simon"])
    ias_bound_sim_ej = sum(1 for r in valid if r["bounded_ias15"] and (not r["bounded_simon"]))
    both_ej          = sum(1 for r in valid if (not r["bounded_ias15"]) and (not r["bounded_simon"]))

    # Lambda for configurations bounded in both integrators
    lam_vals = [r["lambda"] for r in valid
                if r["bounded_ias15"] and r["bounded_simon"] and np.isfinite(r["lambda"])]
    
    lam_mean = float(np.mean(lam_vals)) if lam_vals else float("nan")
    lam_std  = float(np.std(lam_vals))  if lam_vals else float("nan")

    # RMS statistics for configurations bounded in both integrators
    final_rms_vals = [r["final_rms"] for r in valid if np.isfinite(r.get("final_rms", float("nan")))]
    mean_rms_vals  = [r["mean_rms"]  for r in valid if np.isfinite(r.get("mean_rms", float("nan")))]
    max_rms_vals   = [r["max_rms"]   for r in valid if np.isfinite(r.get("max_rms", float("nan")))]
    final_rms_mean = float(np.mean(final_rms_vals)) if final_rms_vals else float("nan")
    final_rms_std  = float(np.std(final_rms_vals))  if final_rms_vals else float("nan")
    final_rms_med  = float(np.median(final_rms_vals)) if final_rms_vals else float("nan")
    mean_rms_mean  = float(np.mean(mean_rms_vals)) if mean_rms_vals else float("nan")
    max_rms_mean   = float(np.mean(max_rms_vals)) if max_rms_vals else float("nan")

    # Speedup statistics
    sp_vals = [r["speedup"] for r in valid if np.isfinite(r["speedup"])]
    sp_mean = float(np.mean(sp_vals)) if sp_vals else float("nan")
    sp_std  = float(np.std(sp_vals))  if sp_vals else float("nan")
    sp_med  = float(np.median(sp_vals)) if sp_vals else float("nan")

    # Total compute time
    t_tot_ias15 = sum(r["t_ias15"] for r in valid if np.isfinite(r["t_ias15"]))
    t_tot_simon = sum(r["t_simon"] for r in valid if np.isfinite(r["t_simon"]))
    pop_speedup = t_tot_ias15 / max(t_tot_simon, 1e-12)

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "pop_data.csv")

    fieldnames = ["ic_id", "m0", "m1", "m2", "r1_init", "r2_init",
                  "q1_est", "q2_est", "f1", "f2", "min_r0", "E0",
                  "bounded_ias15", "bounded_simon", "agree", "lambda",
                  "final_rms", "mean_rms", "max_rms",
                  "speedup", "nn_eligible_frac", "t_ias15", "t_simon"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"  Saved {csv_path}")

    # ── Figure 1: Agreement matrix ────────────────────────────────────
    fig1_path = os.path.join(OUT_DIR, "fig1_agreement_matrix.png")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    matrix = np.array([[both_bound, ias_bound_sim_ej],
                       [ias_ej_sim_bound, both_ej]], dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            pct = 100.0 * matrix[i, j] / n_valid
            ax.text(j, i, f"{int(matrix[i,j])}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=13,
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
                    fontweight="bold")
    labels = ["Bounded", "Ejected"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("SIMON outcome", fontsize=12)
    ax.set_ylabel("ias15 outcome", fontsize=12)
    ax.set_title(
        f"Outcome Agreement Matrix  (N={n_valid})\n"
        f"Agreement rate: {agree_rate:.1%}  |  "
        f"Ejection rate: ias15={ej_rate_ias15:.1%}, SIMON={ej_rate_simon:.1%}",
        fontsize=10, pad=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig1_path}")

    # ── Figure 2: Speedup distribution ───────────────────────────────
    fig2_path = os.path.join(OUT_DIR, "fig2_speedup_distribution.png")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(sp_vals, bins=30, color="#2563A6", alpha=0.80, edgecolor="white",
            linewidth=0.5)
    ax.axvline(1.0,     color="black",   lw=1.5, ls="--",
               label="ias15 baseline (1.0×)")
    ax.axvline(sp_mean, color="#DC2626", lw=1.5, ls="-",
               label=f"Mean speedup ({sp_mean:.2f}×)")
    ax.axvline(sp_med,  color="#16A34A", lw=1.5, ls=":",
               label=f"Median speedup ({sp_med:.2f}×)")
    ax.set_xlabel("Speedup vs ias15", fontsize=12)
    ax.set_ylabel("Number of configurations", fontsize=12)
    ax.set_title(
        f"SIMON Speedup Distribution  (N={len(sp_vals)} valid ICs)\n"
        f"Mean {sp_mean:.2f}× ± {sp_std:.2f}  |  "
        f"Population total: SIMON {t_tot_simon:.0f}s vs ias15 {t_tot_ias15:.0f}s "
        f"(total speed ratio {pop_speedup:.2f}×)",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig2_path}")

    # ── Figure 3: Lambda distribution (bounded ICs only) ─────────────
    fig3_path = os.path.join(OUT_DIR, "fig3_lambda_distribution.png")
    if lam_vals:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(lam_vals, bins=30, color="#9333EA", alpha=0.80,
                edgecolor="white", linewidth=0.5,
                label=f"Population (N={len(lam_vals)} bounded ICs)")
        ax.axvline(lam_mean, color="#DC2626", lw=1.8, ls="-",
                   label=f"Population mean ({lam_mean:.3f} /yr)")
        ax.axvspan(REF_LAMBDA_MEAN - REF_LAMBDA_STD,
                   REF_LAMBDA_MEAN + REF_LAMBDA_STD,
                   alpha=0.20, color="orange",
                   label=f"7-IC reference: {REF_LAMBDA_MEAN:.3f}±{REF_LAMBDA_STD:.3f} /yr")
        ax.axvline(REF_LAMBDA_MEAN, color="orange", lw=1.5, ls="--")
        ax.set_xlabel("Divergence rate λ (yr⁻¹)", fontsize=12)
        ax.set_ylabel("Number of configurations", fontsize=12)
        ax.set_title(
            f"Divergence Rate Distribution — Bounded ICs\n"
            f"Population: λ = {lam_mean:.3f} ± {lam_std:.3f} /yr  |  "
            f"7-IC reference: {REF_LAMBDA_MEAN:.3f} ± {REF_LAMBDA_STD:.3f} /yr",
            fontsize=10, pad=8,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig3_path}")

    # ── Figure 4: RMS distribution (bounded in both solvers) ─────────
    fig4_path = os.path.join(OUT_DIR, "fig4_rms_distribution.png")
    if final_rms_vals:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(final_rms_vals, bins=30, color="#0F766E", alpha=0.80,
                edgecolor="white", linewidth=0.5,
                label=f"Both-bounded ICs (N={len(final_rms_vals)})")
        ax.axvline(final_rms_mean, color="#DC2626", lw=1.8, ls="-",
                   label=f"Mean final RMS ({final_rms_mean:.2f} AU)")
        ax.axvline(final_rms_med, color="#16A34A", lw=1.5, ls=":",
                   label=f"Median final RMS ({final_rms_med:.2f} AU)")
        ax.set_xlabel("Final RMS separation from ias15 (AU)", fontsize=12)
        ax.set_ylabel("Number of configurations", fontsize=12)
        ax.set_title(
            f"Trajectory Error Distribution — Both-Bounded Stress ICs\n"
            f"Final RMS: {final_rms_mean:.2f} ± {final_rms_std:.2f} AU  |  "
            f"Mean time-averaged RMS: {mean_rms_mean:.2f} AU",
            fontsize=10, pad=8,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig4_path}")

    # ── Write summary ─────────────────────────────────────────────────
    lam_consistent = (
        abs(lam_mean - REF_LAMBDA_MEAN) < 2 * (REF_LAMBDA_STD + lam_std)
        if lam_vals else False
    )
    summary_path = os.path.join(OUT_DIR, "pop_summary.txt")
    lines = [
        "=" * 68,
        "EXPERIMENT B — MODERATE-SCATTERING STRESS POPULATION — SUMMARY",
        "=" * 68,
        "",
        "SETUP",
        f"  N_POP          : {N_POP}",
        f"  Valid ICs       : {n_valid}  (failed: {N_POP - n_valid})",
        f"  T=100yr, dt=0.08yr  (speed-oriented stress-test timestep)",
        f"  Masses          : m0 log-U[0.50,2.00], m1 log-U[0.005,0.50],",
        f"                    m2 log-U[0.001,0.10] Msun (within training range)",

        f"  Separations     : r1 U[{R1_RANGE[0]:.1f},{R1_RANGE[1]:.1f}] AU,  "
        f"r2 U[{R2_RANGE[0]:.1f},{R2_RANGE[1]:.1f}] AU",
        f"  Velocity factors: f1 U[{F1_RANGE[0]:.1f},{F1_RANGE[1]:.1f}],  "
        f"f2 U[{F2_RANGE[0]:.1f},{F2_RANGE[1]:.1f}]",
        f"  IC filters      : q_inner >= {MIN_INNER_PERIAPSIS:.2f} AU, "
        f"{MIN_OUTER_PERIAPSIS:.2f} <= q_outer <= "
        f"{MAX_OUTER_PERIAPSIS_FACTOR:.2f} r1, "
        f"initial min pair distance >= {MIN_INITIAL_SEPARATION:.2f} AU",

        f"  Seed            : {SEED}",
        "",
        "OUTCOME CLASSIFICATION",
        f"  ias15 ejection rate  : {ej_rate_ias15:.1%}  ({n_ej_ias15}/{n_valid} ICs)",
        f"  SIMON ejection rate  : {ej_rate_simon:.1%}  ({n_ej_simon}/{n_valid} ICs)",
        f"  Agreement rate       : {agree_rate:.1%}  ({n_agree}/{n_valid} ICs)",
        "",
        "  Confusion matrix (ias15 rows × SIMON cols):",
        f"    Both bounded    : {both_bound:4d}  ({100.*both_bound/n_valid:.1f}%)",
        f"    ias15 ej, SIMON bnd : {ias_ej_sim_bound:4d}  ({100.*ias_ej_sim_bound/n_valid:.1f}%)",
        f"    ias15 bnd, SIMON ej : {ias_bound_sim_ej:4d}  ({100.*ias_bound_sim_ej/n_valid:.1f}%)",
        f"    Both ejected    : {both_ej:4d}  ({100.*both_ej/n_valid:.1f}%)",
        "",
        "DIVERGENCE RATE AND RMS ERROR (both-bounded ICs only)",
        f"  Population λ    : {lam_mean:.4f} ± {lam_std:.4f} /yr  "
        f"(N={len(lam_vals)})",
        f"  7-IC reference  : {REF_LAMBDA_MEAN:.4f} ± {REF_LAMBDA_STD:.4f} /yr",
        f"  Consistent      : {lam_consistent}",
        f"  Final RMS       : {final_rms_mean:.4f} ± {final_rms_std:.4f} AU  "
        f"(median {final_rms_med:.4f} AU)",
        f"  Mean RMS        : {mean_rms_mean:.4f} AU",
        f"  Max RMS         : {max_rms_mean:.4f} AU",
        "",
        "COMPUTATIONAL PERFORMANCE",
        f"  Mean speedup    : {sp_mean:.3f}× ± {sp_std:.3f}",
        f"  Median speedup  : {sp_med:.3f}×",
        f"  Total ias15     : {t_tot_ias15:.1f} s  ({t_tot_ias15/60:.2f} min)",
        f"  Total SIMON     : {t_tot_simon:.1f} s  ({t_tot_simon/60:.2f} min)",
        f"  Population speedup : {pop_speedup:.3f}×",
        "",
        "PAPER PARAGRAPH (use only if agreement and speedup are acceptable):",
        "─" * 68,
        f"To test whether the speed--accuracy behaviour observed on IC1",
        f"generalises beyond a single hand-selected case, we constructed a",
        f"moderate-scattering stress population of {n_valid} compact three-body",
        f"configurations using only pre-integration geometric filters. Masses",
        f"were sampled log-uniformly from [{M0_RANGE[0]:.2f}--{M0_RANGE[1]:.1f}],",
        f"[{M1_RANGE[0]:.3f}--{M1_RANGE[1]:.2f}], and",
        f"[{M2_RANGE[0]:.3f}--{M2_RANGE[1]:.2f}]~M_{{\\odot}}, keeping all bodies",
        f"inside the scalar network's training range. The stress filter required",
        f"negative total energy, safe initial separations, and an estimated outer",
        f"periapsis inside the inner interaction zone, so the sample is designed",
        f"to test moderate three-body scattering rather than broad hierarchical",
        f"population performance.",
        f"At $dt=0.08$~yr, SIMON and \\texttt{{ias15}} agreed on the",
        f"bounded/ejected classification for {agree_rate:.1%} of configurations",
        f"(\\texttt{{ias15}} ejection rate {ej_rate_ias15:.1%}; SIMON ejection",
        f"rate {ej_rate_simon:.1%}). For the {len(lam_vals)} configurations",
        f"bounded in both solvers, the mean divergence-rate proxy was",
        f"$\\lambda = {lam_mean:.3f} \\pm {lam_std:.3f}$~yr$^{{-1}}$ and the",
        f"mean final RMS separation from \\texttt{{ias15}} was",
        f"{final_rms_mean:.3f} \\pm {final_rms_std:.3f}~AU. The mean per-IC",
        f"speed ratio relative to \\texttt{{ias15}} was {sp_mean:.2f}\\times",
        f"(median {sp_med:.2f}\\times; population total {pop_speedup:.2f}\\times).",
        "─" * 68,
        "",
        "OUTPUTS",
        f"  pop_data.csv              -- per-IC raw data",
        f"  fig1_agreement_matrix.png -- 2x2 outcome matrix",
        f"  fig2_speedup_distribution.png",
        f"  fig3_lambda_distribution.png",
        f"  fig4_rms_distribution.png",
        "=" * 68,
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {summary_path}")
    print()
    for line in lines:
        print(f"  {line}")

    print(f"\nAll outputs in: {OUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
