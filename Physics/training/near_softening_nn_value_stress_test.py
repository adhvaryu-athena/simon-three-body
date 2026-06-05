"""
near_softening_nn_value_stress_test.py

Trajectory-level NN-value stress test for SIMON.

Purpose
-------
This experiment deliberately creates a short three-body rollout where one pair
passes near the softening scale, so the scalar NN correction is actually used in
the regime where it can matter.

It compares:
  1. ias15 reference
  2. SIMON with NN correction
  3. No-NN baseline with the same analytic force direction and adaptive sub-stepping

The only difference between SIMON and No-NN is:
  SIMON: close-pair correction factor c is predicted by the NN
  No-NN: close-pair correction factor is fixed to c = 1

Default near-softening design
-----------------------------
Inner pair:
  m0 = 0.010 Msun, m1 = 0.005 Msun
  a_in = 0.020 AU, e_in = 0.95
  q_in = a_in(1-e_in) = 0.001 AU

This is above SIMON's safety fallback threshold r_soft_min = 5e-4 AU,
but close enough to eps = 3e-4 AU that pure softening has a meaningful force
magnitude error. That is exactly the regime where the trained scalar correction
should help.

Run:
  python near_softening_nn_value_stress_test.py

Useful alternatives:
  python near_softening_nn_value_stress_test.py --dt 5e-5 --T 1.0
  python near_softening_nn_value_stress_test.py --dt 2e-4 --T 1.0

Required in same folder:
  pair_correction_nn.pt

Outputs:
  trajectory_level_nn_value_out/near_softening_nn_value_stress_test/
      summary_near_softening_nn_value.txt
      near_softening_metrics.csv
      near_softening_timeseries.npz
      rms_error_early.png
      rms_error_full.png
      inner_separation.png
      exact_energy_drift.png
      xy_trajectories.png
"""

import os
import csv
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rebound
except ImportError as exc:
    raise ImportError(
        "rebound is required. Install it in your cenv, for example: pip install rebound"
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH_DEFAULT = "pair_correction_nn.pt"
OUT_DIR_DEFAULT = os.path.join(
    "trajectory_level_nn_value_out",
    "near_softening_nn_value_stress_test",
)

G = 1.0
EPS = 3e-4
NN_THRESHOLD = 500.0 * EPS       # 0.15 AU
ADAPT_THRESH = 0.05
MAX_SUBSTEPS = 16
R_SOFT_MIN = 5e-4
C_MIN = 0.2
C_MAX = 5.0
EJECTION_AU = 10.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})


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


@dataclass
class HybridConfig:
    G: float = G
    eps: float = EPS
    c_min: float = C_MIN
    c_max: float = C_MAX
    r_soft_min: float = R_SOFT_MIN
    nn_threshold: float = NN_THRESHOLD
    adapt_threshold: float = ADAPT_THRESH
    max_substeps: int = MAX_SUBSTEPS


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run this script from the folder containing "
            "pair_correction_nn.pt, or pass --model_path."
        )
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


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


def simon_forward_numpy(nn_in, w):
    h = (nn_in - w["mean"]) / w["std"]
    h = h @ w["w0T"] + w["b0"]
    s = 1.0 / (1.0 + np.exp(-h))
    h = h * s
    h = h @ w["w1T"] + w["b1"]
    s = 1.0 / (1.0 + np.exp(-h))
    h = h * s
    h = h @ w["w2T"] + w["b2"]
    s = 1.0 / (1.0 + np.exp(-h))
    h = h * s
    return (h @ w["w3T"] + w["b3"]).ravel()


# -----------------------------------------------------------------------------
# Initial condition construction
# -----------------------------------------------------------------------------
def rotation_matrix(inc, Omega, omega):
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(omega), np.sin(omega)
    RzO = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    Rxi = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    Rzo = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
    return RzO @ Rxi @ Rzo


def kepler_relative_state(a, e, inc, Omega, omega, f, mu):
    """Return relative r, v from Keplerian elements for a bound ellipse."""
    if not (0.0 <= e < 1.0):
        raise ValueError("Only bound elliptical orbits with 0 <= e < 1 are supported.")
    if a <= 0.0:
        raise ValueError("Semimajor axis must be positive.")
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(f))
    r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0], dtype=np.float64)
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0], dtype=np.float64)
    R = rotation_matrix(inc, Omega, omega)
    return R @ r_pf, R @ v_pf


def move_to_center_of_mass(x0, v0, m):
    M = float(np.sum(m))
    x_com = np.sum(m[:, None] * x0, axis=0) / M
    v_com = np.sum(m[:, None] * v0, axis=0) / M
    return x0 - x_com, v0 - v_com


def make_near_softening_triple():
    """
    Construct a compact Jacobi-coordinate triple.

    Body 0 and body 1 form an eccentric inner binary with q_in = 0.001 AU.
    Body 2 orbits the inner-binary centre of mass.
    """
    m0 = 0.010
    m1 = 0.005
    m2 = 0.003
    m = np.array([m0, m1, m2], dtype=np.float64)

    # Inner pair: deliberately close to softening, but above safety fallback.
    a_in = 0.020
    e_in = 0.95
    inc_in = np.deg2rad(0.0)
    Omega_in = np.deg2rad(0.0)
    omega_in = np.deg2rad(0.0)
    f_in = np.deg2rad(0.0)  # start at periastron, q=0.001 AU

    # Outer body: compact but still hierarchical. Starting at apastron keeps the
    # third body away from the initial inner close passage.
    a_out = 0.200
    e_out = 0.20
    inc_out = np.deg2rad(45.0)
    Omega_out = np.deg2rad(35.0)
    omega_out = np.deg2rad(80.0)
    f_out = np.deg2rad(180.0)  # start at outer apastron

    M01 = m0 + m1
    Mtot = M01 + m2

    r01, v01 = kepler_relative_state(
        a_in, e_in, inc_in, Omega_in, omega_in, f_in, mu=G * M01
    )
    r2_rel, v2_rel = kepler_relative_state(
        a_out, e_out, inc_out, Omega_out, omega_out, f_out, mu=G * Mtot
    )

    # Inner binary around its own COM.
    x0_inner = -(m1 / M01) * r01
    x1_inner = +(m0 / M01) * r01
    v0_inner = -(m1 / M01) * v01
    v1_inner = +(m0 / M01) * v01

    # Inner COM and body 2 around total COM.
    x_inner_com = -(m2 / Mtot) * r2_rel
    x2 = +(M01 / Mtot) * r2_rel
    v_inner_com = -(m2 / Mtot) * v2_rel
    v2 = +(M01 / Mtot) * v2_rel

    x0 = x_inner_com + x0_inner
    x1 = x_inner_com + x1_inner
    v0 = v_inner_com + v0_inner
    v1 = v_inner_com + v1_inner

    x = np.vstack([x0, x1, x2]).astype(np.float64)
    v = np.vstack([v0, v1, v2]).astype(np.float64)
    x, v = move_to_center_of_mass(x, v, m)

    params = {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "a_in": a_in,
        "e_in": e_in,
        "q_in": a_in * (1.0 - e_in),
        "Q_in": a_in * (1.0 + e_in),
        "period_in": 2.0 * np.pi * np.sqrt(a_in**3 / (G * M01)),
        "a_out": a_out,
        "e_out": e_out,
        "q_out": a_out * (1.0 - e_out),
        "Q_out": a_out * (1.0 + e_out),
        "period_out": 2.0 * np.pi * np.sqrt(a_out**3 / (G * Mtot)),
        "mutual_inclination_deg": 45.0,
    }
    return x, v, m, params


# -----------------------------------------------------------------------------
# Integrators
# -----------------------------------------------------------------------------
def simulate_ias15(x0, v0, m, G_value, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G_value
    for i in range(len(m)):
        sim.add(
            m=float(m[i]),
            x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
            vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]),
        )
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


def simulate_hybrid(x0, v0, m, w, cfg, dt, T, n_samples, mode):
    """
    mode='simon': use NN-predicted c for close pairs.
    mode='no_nn': use c=1 for close pairs.
    Everything else is identical.
    """
    if mode not in {"simon", "no_nn"}:
        raise ValueError("mode must be 'simon' or 'no_nn'")

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i)
            jj.append(j)
    ii = np.array(ii, dtype=np.int64)
    jj = np.array(jj, dtype=np.int64)
    P = len(ii)

    eps2 = cfg.eps * cfg.eps
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]
    mj_arr = m_f[jj]
    Gmimj = cfg.G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    stats = {
        "close_evals": 0,
        "nn_applied_evals": 0,
        "fallback_evals": 0,
        "pair_evals": 0,
        "c_sum": 0.0,
        "c_min": np.inf,
        "c_max": -np.inf,
        "min_pair_distance": np.inf,
        "total_substeps": 0,
    }

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        stats["min_pair_distance"] = min(stats["min_pair_distance"], float(np.min(r)))
        stats["pair_evals"] += P

        # Default far field: exact Newtonian.
        F_scalar = Gmimj / (r2 * r + 1e-30)

        close_mask = r < cfg.nn_threshold
        n_close = int(np.sum(close_mask))
        stats["close_evals"] += n_close

        if n_close > 0:
            r2_c = r2[close_mask]
            r_soft_c = np.sqrt(r2_c + eps2)
            F_soft_c = Gmimj[close_mask] / ((r2_c + eps2) ** 1.5 + 1e-30)

            if mode == "simon":
                nn_in = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close_mask]
                nn_in[:, 2] = log_mj[close_mask]
                log_c = simon_forward_numpy(nn_in, w)
                c = np.exp(log_c).astype(np.float64)
                fallback = (
                    (r_soft_c < cfg.r_soft_min)
                    | (c < cfg.c_min)
                    | (c > cfg.c_max)
                    | ~np.isfinite(c)
                )
                applied = ~fallback
                if np.any(applied):
                    c_applied = c[applied]
                    stats["nn_applied_evals"] += int(np.sum(applied))
                    stats["c_sum"] += float(np.sum(c_applied))
                    stats["c_min"] = min(stats["c_min"], float(np.min(c_applied)))
                    stats["c_max"] = max(stats["c_max"], float(np.max(c_applied)))
                stats["fallback_evals"] += int(np.sum(fallback))
                F_scalar[close_mask] = np.where(fallback, F_scalar[close_mask], c * F_soft_c)
            else:
                fallback = r_soft_c < cfg.r_soft_min
                stats["fallback_evals"] += int(np.sum(fallback))
                # No-NN baseline: pure softened gravity except same safety fallback.
                F_scalar[close_mask] = np.where(fallback, F_scalar[close_mask], F_soft_c)

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return float(np.sqrt(np.min(r2) + 1e-30))

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new

    a = compute_acc(x)

    si = 0
    nt = times[0]
    t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < n_samples:
            nt = times[si]

    ejected = False
    ejection_time = np.nan
    t_start = time.perf_counter()
    steps_done = 0

    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < cfg.adapt_threshold:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_threshold / r_min))))
            sub_dt = float(dt) / n_sub
            for _ in range(n_sub):
                x, v, a = leapfrog_substep(x, v, a, sub_dt)
            stats["total_substeps"] += n_sub
        else:
            vh = v + 0.5 * float(dt) * a
            x = x + float(dt) * vh
            a = compute_acc(x)
            v = vh + 0.5 * float(dt) * a
            stats["total_substeps"] += 1

        t_cur += float(dt)
        steps_done += 1

        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]

        max_com = float(np.max(np.linalg.norm(x, axis=1)))
        if (not np.isfinite(max_com)) or max_com > EJECTION_AU:
            ejected = True
            ejection_time = float(t_cur)
            while si < n_samples:
                pos_out[si] = x
                vel_out[si] = v
                si += 1
            break

        if t_cur >= T - 1e-12:
            break

    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    elapsed = time.perf_counter() - t_start
    c_mean = stats["c_sum"] / max(stats["nn_applied_evals"], 1)
    if not np.isfinite(stats["c_min"]):
        stats["c_min"] = np.nan
        stats["c_max"] = np.nan
        c_mean = np.nan

    perf = {
        "mode": mode,
        "steps": steps_done,
        "planned_steps": n_steps,
        "dt": float(dt),
        "T_years": float(T),
        "total_time_sec": elapsed,
        "time_per_step_sec": elapsed / max(steps_done, 1),
        "pair_evals": int(stats["pair_evals"]),
        "close_evals": int(stats["close_evals"]),
        "close_pair_frac": stats["close_evals"] / max(stats["pair_evals"], 1),
        "nn_applied_evals": int(stats["nn_applied_evals"]),
        "nn_applied_frac": stats["nn_applied_evals"] / max(stats["pair_evals"], 1),
        "fallback_evals": int(stats["fallback_evals"]),
        "fallback_frac": stats["fallback_evals"] / max(stats["pair_evals"], 1),
        "c_mean": float(c_mean),
        "c_min": float(stats["c_min"]),
        "c_max": float(stats["c_max"]),
        "min_pair_distance": float(stats["min_pair_distance"]),
        "total_substeps": int(stats["total_substeps"]),
        "ejected": bool(ejected),
        "ejection_time": float(ejection_time),
    }
    return times, pos_out, vel_out, perf


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rms_sep(a, b):
    d = a - b
    per_body = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(per_body * per_body, axis=1))


def timeavg_rms(delta):
    return float(np.sqrt(np.mean(delta * delta)))


def pair_distance(pos, i, j):
    return np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=1)


def max_distance_from_com(pos):
    return np.max(np.linalg.norm(pos, axis=2), axis=1)


def compute_energy_trajectory(pos, vel, m, G_value=G, softened=False, eps=EPS):
    m = m.astype(np.float64)
    KE = 0.5 * np.einsum("kij,i->k", vel * vel, m)
    PE = np.zeros(pos.shape[0], dtype=np.float64)
    eps2 = eps * eps
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            diff = pos[:, i, :] - pos[:, j, :]
            r2 = np.einsum("ki,ki->k", diff, diff)
            if softened:
                r = np.sqrt(r2 + eps2)
            else:
                r = np.sqrt(r2 + 1e-30)
            PE -= G_value * m[i] * m[j] / r
    return KE + PE


def energy_drift_pct(pos, vel, m, softened=False):
    E = compute_energy_trajectory(pos, vel, m, G_value=G, softened=softened, eps=EPS)
    if not np.all(np.isfinite(E)):
        return np.full(E.shape, np.nan), np.inf, np.nan
    E0 = float(E[0])
    dE = (E - E0) / max(abs(E0), 1e-30)
    return dE * 100.0, float(np.max(np.abs(dE)) * 100.0), float(dE[-1] * 100.0)


def window_metric(times, delta, t_end):
    mask = times <= min(t_end, times[-1]) + 1e-12
    if not np.any(mask):
        return np.nan
    return timeavg_rms(delta[mask])


def summarize_model(label, pos, vel, pos_ref, m, perf):
    delta = rms_sep(pos, pos_ref)
    inner_sep = pair_distance(pos, 0, 1)
    max_com = max_distance_from_com(pos)
    exact_drift_series, exact_max_drift, exact_final_drift = energy_drift_pct(
        pos, vel, m, softened=False
    )
    soft_drift_series, soft_max_drift, soft_final_drift = energy_drift_pct(
        pos, vel, m, softened=True
    )
    return {
        "label": label,
        "bounded": bool(np.all(np.isfinite(pos)) and np.max(max_com) < EJECTION_AU),
        "ejected": bool(perf.get("ejected", False)),
        "runtime_sec": float(perf["total_time_sec"]),
        "close_pair_frac": float(perf["close_pair_frac"]),
        "nn_applied_frac": float(perf["nn_applied_frac"]),
        "fallback_frac": float(perf["fallback_frac"]),
        "min_pair_distance": float(perf["min_pair_distance"]),
        "c_mean": float(perf.get("c_mean", np.nan)),
        "c_min": float(perf.get("c_min", np.nan)),
        "c_max": float(perf.get("c_max", np.nan)),
        "total_substeps": int(perf["total_substeps"]),
        "full_timeavg_rms": timeavg_rms(delta),
        "final_rms": float(delta[-1]),
        "max_rms": float(np.max(delta)),
        "exact_max_energy_drift_pct": exact_max_drift,
        "exact_final_energy_drift_pct": exact_final_drift,
        "soft_max_energy_drift_pct": soft_max_drift,
        "soft_final_energy_drift_pct": soft_final_drift,
        "delta": delta,
        "inner_sep": inner_sep,
        "max_com": max_com,
        "exact_energy_drift_pct_series": exact_drift_series,
        "soft_energy_drift_pct_series": soft_drift_series,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_rms(times, delta_s, delta_n, out_path, xlim=None, title=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(times, np.maximum(delta_s, 1e-14), "-", lw=1.8, label="SIMON with NN")
    ax.semilogy(times, np.maximum(delta_n, 1e-14), "--", lw=1.8, label="No-NN baseline")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_inner_separation(times, r_ref, r_s, r_n, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(times, r_ref, "-", lw=1.6, label="ias15 inner separation")
    ax.semilogy(times, r_s, "--", lw=1.4, label="SIMON with NN")
    ax.semilogy(times, r_n, ":", lw=1.6, label="No-NN baseline")
    ax.axhline(EPS, linestyle=":", lw=1.2, label=r"softening $\epsilon$")
    ax.axhline(R_SOFT_MIN, linestyle="-.", lw=1.2, label=r"safety $r_{\rm soft,min}$")
    ax.axhline(NN_THRESHOLD, linestyle=":", lw=1.2, label="NN threshold")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Inner-pair separation r01 (AU)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_energy(times, dE_s, dE_n, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(times, dE_s, "-", lw=1.5, label="SIMON with NN")
    ax.plot(times, dE_n, "--", lw=1.5, label="No-NN baseline")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"Exact energy drift $\Delta E/E_0$ (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xy(pr, ps, pn, out_path):
    N = pr.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(4.8 * N, 4.2))
    if N == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(pr[:, i, 0], pr[:, i, 1], "-", lw=1.5, label="ias15")
        ax.plot(ps[:, i, 0], ps[:, i, 1], "--", lw=1.2, label="SIMON")
        ax.plot(pn[:, i, 0], pn[:, i, 1], ":", lw=1.4, label="No-NN")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_title(f"Body {i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(framealpha=0.85, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=4000)
    args = parser.parse_args()

    if args.dt <= 0 or args.T <= 0 or args.n_samples < 2:
        raise ValueError("Require dt > 0, T > 0, and n_samples >= 2.")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[near_softening_nn_value_stress_test] Loading SIMON model ...")
    model = load_model(args.model_path)
    w = extract_weights_numpy(model)
    cfg = HybridConfig()
    print(f"  Loaded {args.model_path} with {sum(p.numel() for p in model.parameters())} parameters")

    x0, v0, m, params = make_near_softening_triple()
    print("\n[initial condition]")
    print(f"  masses = {m}")
    print(f"  inner periastron q_in = {params['q_in']:.8f} AU")
    print(f"  inner apastron Q_in = {params['Q_in']:.8f} AU")
    print(f"  inner period = {params['period_in']:.8f} yr")
    print(f"  outer periastron q_out = {params['q_out']:.8f} AU")
    print(f"  outer period = {params['period_out']:.8f} yr")
    print(f"  softening eps = {EPS:.8e} AU")
    print(f"  safety r_soft_min = {R_SOFT_MIN:.8e} AU")
    print(f"  NN threshold = {NN_THRESHOLD:.8f} AU")
    print(f"  dt = {args.dt:.8e} yr, T = {args.T:.6f} yr, n_samples = {args.n_samples}")

    print("\n[running ias15 reference]")
    tr, pr, vr, perf_r = simulate_ias15(x0, v0, m, G, args.T, args.n_samples)
    print(f"  ias15 runtime: {perf_r['total_time_sec']:.3f}s")

    print("\n[running SIMON with NN]")
    ts, ps, vs, perf_s = simulate_hybrid(
        x0, v0, m, w, cfg, args.dt, args.T, args.n_samples, mode="simon"
    )
    print(
        f"  SIMON runtime: {perf_s['total_time_sec']:.3f}s | "
        f"close_pair_frac={perf_s['close_pair_frac']:.6f} | "
        f"nn_applied_frac={perf_s['nn_applied_frac']:.6f} | "
        f"fallback_frac={perf_s['fallback_frac']:.6f} | "
        f"min_r={perf_s['min_pair_distance']:.8e} AU | "
        f"c_mean={perf_s['c_mean']:.4f} | ejected={perf_s['ejected']}"
    )

    print("\n[running No-NN baseline]")
    tn, pn, vn, perf_n = simulate_hybrid(
        x0, v0, m, w, cfg, args.dt, args.T, args.n_samples, mode="no_nn"
    )
    print(
        f"  No-NN runtime: {perf_n['total_time_sec']:.3f}s | "
        f"close_pair_frac={perf_n['close_pair_frac']:.6f} | "
        f"fallback_frac={perf_n['fallback_frac']:.6f} | "
        f"min_r={perf_n['min_pair_distance']:.8e} AU | ejected={perf_n['ejected']}"
    )

    res_s = summarize_model("SIMON_with_NN", ps, vs, pr, m, perf_s)
    res_n = summarize_model("No_NN_baseline", pn, vn, pr, m, perf_n)

    # Reference inner separation for diagnostic plots.
    inner_ref = pair_distance(pr, 0, 1)
    delta_s = res_s["delta"]
    delta_n = res_n["delta"]

    window_defs = [
        ("0_0p02yr", 0.02),
        ("0_0p05yr", 0.05),
        ("0_0p10yr", 0.10),
        ("0_0p25yr", 0.25),
        ("0_0p50yr", 0.50),
        ("0_1p00yr", 1.00),
        ("0_1inner_period", params["period_in"]),
        ("0_2inner_periods", 2.0 * params["period_in"]),
        ("0_4inner_periods", 4.0 * params["period_in"]),
    ]
    window_rows = []
    print("\n[accuracy summary]")
    for name, tend in window_defs:
        if tend > args.T + 1e-12:
            continue
        s_val = window_metric(tr, delta_s, tend)
        n_val = window_metric(tr, delta_n, tend)
        ratio = n_val / max(s_val, 1e-30)
        reduction_pct = (1.0 - s_val / max(n_val, 1e-30)) * 100.0
        window_rows.append((name, tend, s_val, n_val, ratio, reduction_pct))
        print(
            f"  {name}: SIMON={s_val:.6e}, No-NN={n_val:.6e}, "
            f"NoNN/SIMON={ratio:.3f}x, reduction={reduction_pct:.2f}%"
        )

    print("\n[boundedness and energy]")
    for r in [res_s, res_n]:
        print(
            f"  {r['label']}: bounded={r['bounded']}, ejected={r['ejected']}, "
            f"final_RMS={r['final_rms']:.6e}, full_timeavg_RMS={r['full_timeavg_rms']:.6e}, "
            f"exact_max_energy_drift={r['exact_max_energy_drift_pct']:.6e}%"
        )

    # Save summary.
    summary_path = os.path.join(args.out_dir, "summary_near_softening_nn_value.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("NEAR-SOFTENING TRAJECTORY-LEVEL NN-VALUE STRESS TEST\n")
        f.write("=" * 78 + "\n\n")
        f.write("Purpose:\n")
        f.write(
            "  Test whether SIMON's scalar NN correction improves trajectory accuracy "
            "when a three-body rollout repeatedly enters the near-softening regime.\n\n"
        )
        f.write("Design:\n")
        f.write("  ias15 reference vs SIMON with NN vs No-NN baseline.\n")
        f.write("  Adaptive sub-stepping and analytic force direction remain ON for both SIMON variants.\n")
        f.write("  The only model difference is c predicted by NN vs c fixed to 1.\n\n")
        f.write("Initial condition parameters:\n")
        for k, v_param in params.items():
            f.write(f"  {k}: {v_param}\n")
        f.write(f"  eps: {EPS}\n")
        f.write(f"  NN threshold: {NN_THRESHOLD} AU\n")
        f.write(f"  adaptive threshold: {ADAPT_THRESH} AU\n")
        f.write(f"  safety threshold r_soft_min: {R_SOFT_MIN} AU\n\n")
        f.write("Run configuration:\n")
        f.write(f"  dt: {args.dt} yr\n")
        f.write(f"  T: {args.T} yr\n")
        f.write(f"  n_samples: {args.n_samples}\n")
        f.write(f"  ias15 runtime: {perf_r['total_time_sec']:.6f} s\n\n")
        f.write("Core results:\n")
        f.write(
            "  Model, bounded, ejected, close_pair_frac, nn_applied_frac, fallback_frac, "
            "min_pair_distance, c_mean, full_time_avg_rms, final_rms, "
            "exact_max_energy_drift_pct, soft_max_energy_drift_pct\n"
        )
        for r in [res_s, res_n]:
            f.write(
                f"  {r['label']}, {r['bounded']}, {r['ejected']}, "
                f"{r['close_pair_frac']:.8e}, {r['nn_applied_frac']:.8e}, "
                f"{r['fallback_frac']:.8e}, {r['min_pair_distance']:.8e}, "
                f"{r['c_mean']:.8e}, {r['full_timeavg_rms']:.8e}, "
                f"{r['final_rms']:.8e}, {r['exact_max_energy_drift_pct']:.8e}, "
                f"{r['soft_max_energy_drift_pct']:.8e}\n"
            )
        f.write("\nEarly-window trajectory accuracy:\n")
        for name, tend, s_val, n_val, ratio, reduction_pct in window_rows:
            f.write(
                f"  {name}: SIMON={s_val:.8e}, No-NN={n_val:.8e}, "
                f"NoNN/SIMON={ratio:.4f}x, reduction={reduction_pct:.2f}%\n"
            )
        f.write("\nInterpretation guide:\n")
        f.write(
            "  If both SIMON and No-NN remain bounded but SIMON has lower early-window RMS, "
            "the experiment supports the claim that the NN correction improves accuracy "
            "rather than providing the primary stability mechanism.\n"
        )

    # Save metrics CSV.
    metrics_path = os.path.join(args.out_dir, "near_softening_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "bounded", "ejected", "runtime_sec", "close_pair_frac",
            "nn_applied_frac", "fallback_frac", "min_pair_distance", "c_mean", "c_min", "c_max",
            "total_substeps", "full_timeavg_rms", "final_rms", "max_rms",
            "exact_max_energy_drift_pct", "exact_final_energy_drift_pct",
            "soft_max_energy_drift_pct", "soft_final_energy_drift_pct",
        ])
        for r in [res_s, res_n]:
            writer.writerow([
                r["label"], r["bounded"], r["ejected"], r["runtime_sec"],
                r["close_pair_frac"], r["nn_applied_frac"], r["fallback_frac"],
                r["min_pair_distance"], r["c_mean"], r["c_min"], r["c_max"],
                r["total_substeps"], r["full_timeavg_rms"], r["final_rms"], r["max_rms"],
                r["exact_max_energy_drift_pct"], r["exact_final_energy_drift_pct"],
                r["soft_max_energy_drift_pct"], r["soft_final_energy_drift_pct"],
            ])

    # Save NPZ time series.
    npz_path = os.path.join(args.out_dir, "near_softening_timeseries.npz")
    np.savez_compressed(
        npz_path,
        times=tr,
        pos_ias15=pr,
        vel_ias15=vr,
        pos_simon=ps,
        vel_simon=vs,
        pos_no_nn=pn,
        vel_no_nn=vn,
        delta_simon=delta_s,
        delta_no_nn=delta_n,
        inner_ref=inner_ref,
        inner_simon=res_s["inner_sep"],
        inner_no_nn=res_n["inner_sep"],
        exact_energy_drift_simon=res_s["exact_energy_drift_pct_series"],
        exact_energy_drift_no_nn=res_n["exact_energy_drift_pct_series"],
        soft_energy_drift_simon=res_s["soft_energy_drift_pct_series"],
        soft_energy_drift_no_nn=res_n["soft_energy_drift_pct_series"],
        masses=m,
        x0=x0,
        v0=v0,
    )

    # Plots.
    plot_rms(
        tr, delta_s, delta_n,
        os.path.join(args.out_dir, "rms_error_early.png"),
        xlim=(0.0, min(0.25, args.T)),
        title="Near-softening stress test: early RMS error",
    )
    plot_rms(
        tr, delta_s, delta_n,
        os.path.join(args.out_dir, "rms_error_full.png"),
        xlim=(0.0, args.T),
        title="Near-softening stress test: full RMS error",
    )
    plot_inner_separation(
        tr, inner_ref, res_s["inner_sep"], res_n["inner_sep"],
        os.path.join(args.out_dir, "inner_separation.png"),
    )
    plot_energy(
        tr, res_s["exact_energy_drift_pct_series"], res_n["exact_energy_drift_pct_series"],
        os.path.join(args.out_dir, "exact_energy_drift.png"),
    )
    plot_xy(pr, ps, pn, os.path.join(args.out_dir, "xy_trajectories.png"))

    print("\n[outputs]")
    print(f"  Summary: {summary_path}")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Time series NPZ: {npz_path}")
    print(f"  Figures saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
