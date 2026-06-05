"""
real_system_sun_earth_moon_no_nn.py

Real-system Sun-Earth-Moon validation for the NN-correction ablation.

Experiment:
    ias15 reference
    Full SIMON          = adaptive ON + analytic direction + NN scalar correction
    No-NN baseline      = adaptive ON + analytic direction + pure softened close-pair force (c=1)

Purpose:
    Test whether the learned scalar correction matters on real JPL Horizons
    Sun-Earth-Moon ephemeris data, while keeping the stabilising structural
    constraints fixed: analytic force direction and adaptive sub-stepping.

Main outputs:
    real_system_validation/sun_earth_moon_no_nn_T1/
        summary_no_nn_real_system.txt
        no_nn_results.csv
        no_nn_results.npz
        timeavg_rms_vs_dt.png
        max_rms_vs_dt.png
        energy_drift_vs_dt.png
        runtime_vs_dt.png
        speed_ratio_vs_dt.png
        earth_moon_range_vs_dt.png
        selected_dt016_distance_comparison.png
        selected_dt016_xy_trajectories.png

Run:
    python real_system_sun_earth_moon_no_nn.py

Requirements in the same folder:
    pair_correction_nn.pt
"""

import os
import csv
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from astroquery.jplhorizons import Horizons

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rebound


# JPL Horizons can be slow on the first query.
Horizons.TIMEOUT = 120


# -----------------------------------------------------------------------------
# Units and real-system constants
# -----------------------------------------------------------------------------
DAYS_PER_YEAR = 365.25
G_REAL = 4.0 * np.pi**2


# -----------------------------------------------------------------------------
# JPL Horizons real initial conditions
# -----------------------------------------------------------------------------
def get_horizons_vector(body_id, epoch="2026-01-01"):
    """
    Fetch Solar-System-barycentric JPL Horizons state vector for one body.

    location='@0' means Solar System barycentre.
    Output position is AU and velocity is AU/day.
    """
    epochs = {
        "start": epoch,
        "stop": "2026-01-02",
        "step": "1d",
    }

    print(f"Fetching JPL Horizons vector for body {body_id} ...")
    obj = Horizons(id=body_id, location="@0", epochs=epochs)
    vec = obj.vectors()

    pos = np.array([
        float(vec["x"][0]),
        float(vec["y"][0]),
        float(vec["z"][0]),
    ], dtype=np.float64)

    vel_au_per_day = np.array([
        float(vec["vx"][0]),
        float(vec["vy"][0]),
        float(vec["vz"][0]),
    ], dtype=np.float64)

    return pos, vel_au_per_day * DAYS_PER_YEAR


def move_to_center_of_mass(x0, v0, m):
    """Shift positions and velocities to the centre-of-mass frame."""
    M = np.sum(m)
    x_com = np.sum(m[:, None] * x0, axis=0) / M
    v_com = np.sum(m[:, None] * v0, axis=0) / M
    return x0 - x_com, v0 - v_com


def load_sun_earth_moon_initial_conditions(epoch="2026-01-01"):
    """
    Fetch real Sun-Earth-Moon initial conditions from JPL Horizons.

    Body IDs:
      Sun   = 10
      Earth = 399
      Moon  = 301
    """
    sun_x, sun_v = get_horizons_vector("10", epoch)
    earth_x, earth_v = get_horizons_vector("399", epoch)
    moon_x, moon_v = get_horizons_vector("301", epoch)

    x0 = np.vstack([sun_x, earth_x, moon_x])
    v0 = np.vstack([sun_v, earth_v, moon_v])

    # Masses in solar masses.
    m = np.array([
        1.0,                 # Sun
        3.003489614915e-6,   # Earth
        3.694303349e-8,      # Moon
    ], dtype=np.float64)

    return (*move_to_center_of_mass(x0, v0, m), m)


# -----------------------------------------------------------------------------
# SIMON model and configuration
# -----------------------------------------------------------------------------
class PairCorrectionNN(nn.Module):
    """Same scalar correction network used in the SIMON experiments."""
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
    G: float = G_REAL
    eps: float = 3e-4
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


def extract_weights_numpy(model):
    """Convert trained PyTorch weights into NumPy arrays for fast SIMON inference."""
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
    """SIMON forward pass. nn_in shape: (n_close, 3). Returns log(c)."""
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
# Integrators
# -----------------------------------------------------------------------------
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    """Reference integration using REBOUND ias15."""
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G

    for i in range(len(m)):
        sim.add(
            m=float(m[i]),
            x=float(x0[i, 0]),
            y=float(x0[i, 1]),
            z=float(x0[i, 2]),
            vx=float(v0[i, 0]),
            vy=float(v0[i, 1]),
            vz=float(v0[i, 2]),
        )

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

    perf = {"total_time_sec": time.perf_counter() - t0}
    return times, pos, vel, perf


def simulate_hybrid_real_system(
    x0,
    v0,
    m,
    model,
    cfg,
    dt,
    T,
    n_samples,
    model_type="simon",
):
    """
    Hybrid rollout for real Sun-Earth-Moon initial conditions.

    model_type='simon':
        Close pairs (r < 0.15 AU):
            F = c * G*m_i*m_j/r_soft^3 * r_ij
            c is predicted by the NN and gated to [0.2, 5.0].
        Direction is always exact analytic geometry.

    model_type='no_nn':
        Close pairs (r < 0.15 AU):
            F = G*m_i*m_j/r_soft^3 * r_ij
            c = 1 always; NN is never called.
        Extreme-close fallback r_soft < r_soft_min uses exact Newtonian,
        matching the existing no_nn_ablation.py logic.

    Both modes use adaptive sub-stepping with the same threshold and max_substeps.
    """
    if model_type not in {"simon", "no_nn"}:
        raise ValueError("model_type must be 'simon' or 'no_nn'")

    w = extract_weights_numpy(model) if model_type == "simon" else None

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

    nn_thresh = 500.0 * cfg.eps      # 0.15 AU
    adapt_thresh = 0.05              # same threshold as paper experiments
    max_substeps = 16

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)

        # Default: exact unsoftened Newtonian force for all pairs.
        invr3 = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close_mask = r < nn_thresh
        n_close = int(np.sum(close_mask))
        n_nn = 0
        n_fallback = 0

        if n_close > 0:
            r_soft_close = np.sqrt(r2[close_mask] + eps2)
            denom = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom

            if model_type == "simon":
                nn_in = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_close + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close_mask]
                nn_in[:, 2] = log_mj[close_mask]

                log_c = simon_forward_numpy(nn_in, w)
                c = np.exp(log_c).astype(np.float64)

                fallback = (
                    (r_soft_close < r_soft_min)
                    | (c < c_min)
                    | (c > c_max)
                    | ~np.isfinite(c)
                )
                n_fallback = int(np.sum(fallback))
                n_nn = int(n_close - n_fallback)

                F_scalar[close_mask] = np.where(
                    fallback,
                    F_scalar[close_mask],
                    c * F_soft_close,
                )

            else:  # model_type == "no_nn"
                # Pure softened gravity for close pairs, c = 1 always.
                # Extreme-close fallback to exact Newtonian is identical to
                # the existing no_nn_ablation.py convention.
                fallback = r_soft_close < r_soft_min
                n_fallback = int(np.sum(fallback))
                n_nn = 0
                F_scalar[close_mask] = np.where(
                    fallback,
                    F_scalar[close_mask],
                    F_soft_close,
                )

        F_vec = F_scalar[:, None] * rij

        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]

        return acc, n_close, n_nn, n_fallback

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, n_close, n_nn, n_fallback = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, n_close, n_nn, n_fallback

    a, n_close, n_nn, n_fallback = compute_acc(x)

    close_sum = n_close
    nn_sum = n_nn
    fallback_sum = n_fallback
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

    t0 = time.perf_counter()

    for _ in range(n_steps):
        r_min = min_pair_dist(x)

        # Adaptive ON for both Full SIMON and No-NN baseline.
        if r_min < adapt_thresh:
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = float(dt) / n_sub
            for _ in range(n_sub):
                x, v, a, n_close, n_nn, n_fallback = leapfrog_substep(x, v, a, sub_dt)
                close_sum += n_close
                nn_sum += n_nn
                fallback_sum += n_fallback
                pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * float(dt) * a
            x = x + float(dt) * vh
            a, n_close, n_nn, n_fallback = compute_acc(x)
            v = vh + 0.5 * float(dt) * a
            close_sum += n_close
            nn_sum += n_nn
            fallback_sum += n_fallback
            pair_sum += P
            total_substeps += 1

        t_cur += float(dt)

        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]

        if t_cur >= T - 1e-12:
            break

    # Guard against floating-point accumulation leaving final samples unwritten.
    # This prevents the old zero-sample energy-drift artifact.
    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    perf = {
        "steps": n_steps,
        "total_time_sec": time.perf_counter() - t0,
        "avg_close_pair_frac": close_sum / max(pair_sum, 1),
        "avg_nn_frac": nn_sum / max(pair_sum, 1),
        "avg_fallback_frac": fallback_sum / max(pair_sum, 1),
        "total_substeps": total_substeps,
    }

    return times, pos_out, vel_out, perf


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rms_sep(a, b):
    """Per-time RMS position separation across bodies."""
    d = a - b
    per_body = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(per_body**2, axis=1))


def paper_time_avg_rms(delta):
    """RMS_{0:T} = sqrt(mean_t delta(t)^2)."""
    return float(np.sqrt(np.mean(delta**2)))


def pair_distance(pos_arr, i, j):
    return np.linalg.norm(pos_arr[:, i, :] - pos_arr[:, j, :], axis=1)


def max_distance_from_com(pos_arr):
    return np.max(np.linalg.norm(pos_arr, axis=2), axis=1)


def compute_energy_trajectory(pos_arr, vel_arr, m, G, eps):
    """
    Compute total mechanical energy E(t) = KE(t) + PE_softened(t).

    Softened PE follows the corrected energy-drift scaling script:
        PE = -G*m_i*m_j/sqrt(r^2 + eps^2)
    This avoids pathological values for failed close-pair rollouts and is
    effectively Newtonian when r >> eps.
    """
    m = m.astype(np.float64)
    pos_arr = pos_arr.astype(np.float64, copy=False)
    vel_arr = vel_arr.astype(np.float64, copy=False)

    KE = 0.5 * np.einsum("kij,i->k", vel_arr**2, m)

    PE = np.zeros(pos_arr.shape[0], dtype=np.float64)
    eps2 = float(eps) ** 2
    N = len(m)
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos_arr[:, i, :] - pos_arr[:, j, :]
            r2 = np.einsum("ki,ki->k", diff, diff)
            r_soft = np.sqrt(r2 + eps2)
            PE -= G * m[i] * m[j] / r_soft

    return KE + PE


def energy_drift_metrics(pos_arr, vel_arr, m, G, eps):
    """
    dE(t) = (E(t) - E(0)) / |E(0)|.
    Returns max, final, and RMS energy drift in percent.
    """
    E = compute_energy_trajectory(pos_arr, vel_arr, m, G=G, eps=eps)

    if not np.all(np.isfinite(E)):
        return {
            "energy_E0": np.nan,
            "max_energy_drift_pct": np.inf,
            "final_energy_drift_pct": np.nan,
            "rms_energy_drift_pct": np.inf,
            "energy_drift_pct_series": np.full_like(E, np.nan, dtype=np.float64),
            "energy_finite": False,
        }

    E0 = float(E[0])
    denom = max(abs(E0), 1e-30)
    dE = (E - E0) / denom
    dE_pct = dE * 100.0

    return {
        "energy_E0": E0,
        "max_energy_drift_pct": float(np.max(np.abs(dE_pct))),
        "final_energy_drift_pct": float(dE_pct[-1]),
        "rms_energy_drift_pct": float(np.sqrt(np.mean(dE_pct**2))),
        "energy_drift_pct_series": dE_pct.astype(np.float64),
        "energy_finite": True,
    }


def summarize_rollout(pos_model, vel_model, pos_ref, perf, label, dt, m, G, eps, bounded_threshold=10.0):
    """Compute the metrics used in the real-system No-NN ablation."""
    delta = rms_sep(pos_model, pos_ref)
    sun_earth = pair_distance(pos_model, 0, 1)
    earth_moon = pair_distance(pos_model, 1, 2)
    max_com = max_distance_from_com(pos_model)

    finite = bool(np.all(np.isfinite(pos_model)) and np.all(np.isfinite(vel_model)))
    bounded = bool(finite and np.max(max_com) < bounded_threshold)
    energy = energy_drift_metrics(pos_model, vel_model, m, G=G, eps=eps)

    return {
        "label": label,
        "dt": float(dt),
        "runtime_sec": float(perf["total_time_sec"]),
        "steps": int(perf["steps"]),
        "total_substeps": int(perf["total_substeps"]),
        "avg_close_pair_frac": float(perf["avg_close_pair_frac"]),
        "avg_nn_frac": float(perf["avg_nn_frac"]),
        "avg_fallback_frac": float(perf["avg_fallback_frac"]),
        "time_avg_rms": paper_time_avg_rms(delta),
        "final_rms": float(delta[-1]),
        "max_rms": float(np.max(delta)),
        "sun_earth_min": float(np.min(sun_earth)),
        "sun_earth_max": float(np.max(sun_earth)),
        "earth_moon_min": float(np.min(earth_moon)),
        "earth_moon_max": float(np.max(earth_moon)),
        "max_com": float(np.max(max_com)),
        "max_energy_drift_pct": energy["max_energy_drift_pct"],
        "final_energy_drift_pct": energy["final_energy_drift_pct"],
        "rms_energy_drift_pct": energy["rms_energy_drift_pct"],
        "energy_E0": energy["energy_E0"],
        "energy_finite": energy["energy_finite"],
        "bounded": bounded,
        "finite": finite,
        "delta": delta,
        "sun_earth": sun_earth,
        "earth_moon": earth_moon,
        "max_com_series": max_com,
        "energy_drift_pct_series": energy["energy_drift_pct_series"],
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_metric_vs_dt(rows_simon, rows_no_nn, out_path, metric, ylabel, logy=True):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    y_simon = np.array([r[metric] for r in rows_simon], dtype=float)
    y_no_nn = np.array([r[metric] for r in rows_no_nn], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_simon, "o-", lw=1.8, label="Full SIMON")
    ax.plot(dts, y_no_nn, "s--", lw=1.8, label="No-NN baseline")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_runtime_vs_dt(rows_simon, rows_no_nn, ias15_time, out_path):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    y_simon = np.array([r["runtime_sec"] for r in rows_simon], dtype=float)
    y_no_nn = np.array([r["runtime_sec"] for r in rows_no_nn], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_simon, "o-", lw=1.8, label="Full SIMON")
    ax.plot(dts, y_no_nn, "s--", lw=1.8, label="No-NN baseline")
    ax.axhline(ias15_time, lw=1.4, linestyle=":", label="ias15 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Runtime (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_speed_ratio_vs_dt(rows_simon, rows_no_nn, ias15_time, out_path):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    y_simon = np.array([ias15_time / max(r["runtime_sec"], 1e-30) for r in rows_simon], dtype=float)
    y_no_nn = np.array([ias15_time / max(r["runtime_sec"], 1e-30) for r in rows_no_nn], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_simon, "o-", lw=1.8, label="Full SIMON")
    ax.plot(dts, y_no_nn, "s--", lw=1.8, label="No-NN baseline")
    ax.axhline(1.0, lw=1.2, linestyle=":", label="same runtime as ias15")
    ax.set_xscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Speed ratio: ias15 runtime / model runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_earth_moon_range_vs_dt(ref_earth_moon, rows_simon, rows_no_nn, out_path):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    simon_min = np.array([r["earth_moon_min"] for r in rows_simon], dtype=float)
    simon_max = np.array([r["earth_moon_max"] for r in rows_simon], dtype=float)
    no_nn_min = np.array([r["earth_moon_min"] for r in rows_no_nn], dtype=float)
    no_nn_max = np.array([r["earth_moon_max"] for r in rows_no_nn], dtype=float)

    ref_min = float(np.min(ref_earth_moon))
    ref_max = float(np.max(ref_earth_moon))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(dts, simon_min, simon_max, alpha=0.20, label="Full SIMON range")
    ax.plot(dts, simon_min, "o-", lw=1.2)
    ax.plot(dts, simon_max, "o-", lw=1.2)
    ax.fill_between(dts, no_nn_min, no_nn_max, alpha=0.20, label="No-NN range")
    ax.plot(dts, no_nn_min, "s--", lw=1.2)
    ax.plot(dts, no_nn_max, "s--", lw=1.2)
    ax.axhline(ref_min, lw=1.2, linestyle=":", label="ias15 min/max")
    ax.axhline(ref_max, lw=1.2, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Earth-Moon distance range (AU)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pair_distances_three(times, ref_se, simon_se, no_nn_se, ref_em, simon_em, no_nn_em, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    axes[0].plot(times, ref_se, lw=1.8, label="ias15")
    axes[0].plot(times, simon_se, "--", lw=1.8, label="Full SIMON")
    axes[0].plot(times, no_nn_se, ":", lw=1.8, label="No-NN")
    axes[0].set_ylabel("Sun-Earth distance (AU)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    axes[1].plot(times, ref_em, lw=1.8, label="ias15")
    axes[1].plot(times, simon_em, "--", lw=1.8, label="Full SIMON")
    axes[1].plot(times, no_nn_em, ":", lw=1.8, label="No-NN")
    axes[1].set_xlabel("Time (yr)")
    axes[1].set_ylabel("Earth-Moon distance (AU)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xy_trajectories_three(pos_ref, pos_simon, pos_no_nn, out_path):
    """Sun and Earth in barycentric frame; Moon relative to Earth."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].plot(pos_ref[:, 0, 0], pos_ref[:, 0, 1], lw=1.5, label="ias15")
    axes[0].plot(pos_simon[:, 0, 0], pos_simon[:, 0, 1], "--", lw=1.5, label="Full SIMON")
    axes[0].plot(pos_no_nn[:, 0, 0], pos_no_nn[:, 0, 1], ":", lw=1.5, label="No-NN")
    axes[0].set_title("Sun")
    axes[0].set_xlabel("x (AU)")
    axes[0].set_ylabel("y (AU)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    axes[1].plot(pos_ref[:, 1, 0], pos_ref[:, 1, 1], lw=1.5, label="ias15")
    axes[1].plot(pos_simon[:, 1, 0], pos_simon[:, 1, 1], "--", lw=1.5, label="Full SIMON")
    axes[1].plot(pos_no_nn[:, 1, 0], pos_no_nn[:, 1, 1], ":", lw=1.5, label="No-NN")
    axes[1].set_title("Earth")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("y (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    moon_rel_ref = pos_ref[:, 2, :] - pos_ref[:, 1, :]
    moon_rel_simon = pos_simon[:, 2, :] - pos_simon[:, 1, :]
    moon_rel_no_nn = pos_no_nn[:, 2, :] - pos_no_nn[:, 1, :]

    axes[2].plot(moon_rel_ref[:, 0], moon_rel_ref[:, 1], lw=1.5, label="ias15")
    axes[2].plot(moon_rel_simon[:, 0], moon_rel_simon[:, 1], "--", lw=1.5, label="Full SIMON")
    axes[2].plot(moon_rel_no_nn[:, 0], moon_rel_no_nn[:, 1], ":", lw=1.5, label="No-NN")
    axes[2].set_title("Moon relative to Earth")
    axes[2].set_xlabel("x relative to Earth (AU)")
    axes[2].set_ylabel("y relative to Earth (AU)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------
def write_csv(rows_simon, rows_no_nn, ias15_time, out_path):
    fieldnames = [
        "mode", "dt", "runtime_sec", "speed_vs_ias15", "steps", "total_substeps",
        "avg_close_pair_frac", "avg_nn_frac", "avg_fallback_frac",
        "time_avg_rms", "final_rms", "max_rms",
        "max_energy_drift_pct", "final_energy_drift_pct", "rms_energy_drift_pct",
        "sun_earth_min", "sun_earth_max", "earth_moon_min", "earth_moon_max",
        "max_com", "bounded", "finite", "energy_finite",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in [rows_simon, rows_no_nn]:
            for r in rows:
                row = {k: r[k] for k in fieldnames if k in r}
                row["mode"] = r["label"]
                row["speed_vs_ias15"] = ias15_time / max(r["runtime_sec"], 1e-30)
                writer.writerow(row)


def write_summary(
    out_path,
    T,
    n_samples,
    dt_values,
    epoch,
    x0,
    v0,
    m,
    perf_ref,
    ref_sun_earth,
    ref_earth_moon,
    ref_max_com,
    ref_energy,
    rows_simon,
    rows_no_nn,
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Real Sun-Earth-Moon Full SIMON vs No-NN ablation\n")
        f.write("=" * 86 + "\n\n")
        f.write(f"Epoch                   = {epoch}\n")
        f.write(f"T                       = {T:.2f} yr\n")
        f.write(f"n_samples               = {n_samples}\n")
        f.write("dt_values               = " + ", ".join(f"{d:.4f}" for d in dt_values) + " yr\n")
        f.write(f"G_REAL                  = {G_REAL:.8f}\n")
        f.write(f"ias15 runtime            = {perf_ref['total_time_sec']:.6f} s\n\n")

        f.write("Experiment definition:\n")
        f.write("- Full SIMON: adaptive ON + analytic direction + NN scalar correction.\n")
        f.write("- No-NN baseline: adaptive ON + analytic direction + softened close-pair force with c=1.\n")
        f.write("- Both modes use identical initial conditions, dt values, adaptive sub-stepping, and safety fallback.\n\n")

        f.write("Initial-condition sanity checks after COM centering:\n")
        f.write(f"Sun-Earth distance      = {np.linalg.norm(x0[1] - x0[0]):.6f} AU\n")
        f.write(f"Earth-Moon distance     = {np.linalg.norm(x0[2] - x0[1]):.6f} AU\n")
        f.write(f"Masses                  = {m}\n\n")

        f.write("ias15 reference ranges over simulation:\n")
        f.write(f"Sun-Earth distance      = {np.min(ref_sun_earth):.6f} to {np.max(ref_sun_earth):.6f} AU\n")
        f.write(f"Earth-Moon distance     = {np.min(ref_earth_moon):.6f} to {np.max(ref_earth_moon):.6f} AU\n")
        f.write(f"Max distance from COM   = {np.max(ref_max_com):.6f} AU\n")
        f.write(f"Max energy drift        = {ref_energy['max_energy_drift_pct']:.6e}%\n")
        f.write(f"Final energy drift      = {ref_energy['final_energy_drift_pct']:.6e}%\n\n")

        f.write("Main table:\n")
        f.write(
            f"{'mode':<14} {'dt':>8} {'runtime(s)':>12} {'speed':>9} "
            f"{'substeps':>10} {'NN_frac':>9} {'RMS0T(AU)':>13} {'Emax(%)':>12} "
            f"{'final(AU)':>12} {'max(AU)':>12} {'EM_min':>10} {'EM_max':>10} {'bounded':>8}\n"
        )
        f.write("-" * 154 + "\n")
        for rows in [rows_simon, rows_no_nn]:
            for r in rows:
                speed = perf_ref["total_time_sec"] / max(r["runtime_sec"], 1e-30)
                f.write(
                    f"{r['label']:<14} {r['dt']:>8.4f} {r['runtime_sec']:>12.6f} {speed:>9.3f} "
                    f"{r['total_substeps']:>10d} {r['avg_nn_frac']:>9.6f} "
                    f"{r['time_avg_rms']:>13.6e} {r['max_energy_drift_pct']:>12.6e} "
                    f"{r['final_rms']:>12.6e} {r['max_rms']:>12.6e} "
                    f"{r['earth_moon_min']:>10.6f} {r['earth_moon_max']:>10.6f} "
                    f"{str(r['bounded']):>8}\n"
                )
            f.write("\n")

        f.write("Interpretation guide:\n")
        f.write("- RMS0T is the paper-consistent time-averaged RMS: sqrt(mean_t delta(t)^2).\n")
        f.write("- Emax is max_t |(E(t)-E(0))/E(0)| in percent using softened PE.\n")
        f.write("- NN_frac is non-zero only for Full SIMON; it is 0 for the No-NN baseline by definition.\n")
        f.write("- This ablation keeps adaptive sub-stepping ON for both models, isolating the learned scalar correction.\n")


def save_npz(
    out_path,
    times,
    x0,
    v0,
    m,
    pos_ref,
    vel_ref,
    rows_simon,
    rows_no_nn,
    pos_simon_all,
    pos_no_nn_all,
    vel_simon_all,
    vel_no_nn_all,
    dt_values,
    T,
    G,
    ref_energy,
):
    np.savez_compressed(
        out_path,
        times=times,
        x0=x0,
        v0=v0,
        masses=m,
        pos_ias15=pos_ref,
        vel_ias15=vel_ref,
        pos_full_simon=pos_simon_all,
        pos_no_nn=pos_no_nn_all,
        vel_full_simon=vel_simon_all,
        vel_no_nn=vel_no_nn_all,
        dt_values=np.array(dt_values, dtype=np.float64),
        T=T,
        G_REAL=G,
        max_energy_drift_pct_ref=ref_energy["max_energy_drift_pct"],
        energy_drift_pct_ref=ref_energy["energy_drift_pct_series"],
        time_avg_rms_full_simon=np.array([r["time_avg_rms"] for r in rows_simon], dtype=np.float64),
        time_avg_rms_no_nn=np.array([r["time_avg_rms"] for r in rows_no_nn], dtype=np.float64),
        final_rms_full_simon=np.array([r["final_rms"] for r in rows_simon], dtype=np.float64),
        final_rms_no_nn=np.array([r["final_rms"] for r in rows_no_nn], dtype=np.float64),
        max_rms_full_simon=np.array([r["max_rms"] for r in rows_simon], dtype=np.float64),
        max_rms_no_nn=np.array([r["max_rms"] for r in rows_no_nn], dtype=np.float64),
        max_energy_drift_pct_full_simon=np.array([r["max_energy_drift_pct"] for r in rows_simon], dtype=np.float64),
        max_energy_drift_pct_no_nn=np.array([r["max_energy_drift_pct"] for r in rows_no_nn], dtype=np.float64),
        final_energy_drift_pct_full_simon=np.array([r["final_energy_drift_pct"] for r in rows_simon], dtype=np.float64),
        final_energy_drift_pct_no_nn=np.array([r["final_energy_drift_pct"] for r in rows_no_nn], dtype=np.float64),
        energy_drift_pct_full_simon=np.stack([r["energy_drift_pct_series"] for r in rows_simon], axis=0),
        energy_drift_pct_no_nn=np.stack([r["energy_drift_pct_series"] for r in rows_no_nn], axis=0),
        runtime_full_simon=np.array([r["runtime_sec"] for r in rows_simon], dtype=np.float64),
        runtime_no_nn=np.array([r["runtime_sec"] for r in rows_no_nn], dtype=np.float64),
        substeps_full_simon=np.array([r["total_substeps"] for r in rows_simon], dtype=np.int64),
        substeps_no_nn=np.array([r["total_substeps"] for r in rows_no_nn], dtype=np.int64),
        avg_nn_frac_full_simon=np.array([r["avg_nn_frac"] for r in rows_simon], dtype=np.float64),
        avg_nn_frac_no_nn=np.array([r["avg_nn_frac"] for r in rows_no_nn], dtype=np.float64),
        bounded_full_simon=np.array([r["bounded"] for r in rows_simon], dtype=bool),
        bounded_no_nn=np.array([r["bounded"] for r in rows_no_nn], dtype=bool),
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    MODEL_PATH = "pair_correction_nn.pt"
    EPOCH = "2026-01-01"

    T = 1.0
    N_SAMPLES = 1000
    DT_VALUES = [0.001, 0.002, 0.004, 0.008, 0.010, 0.012, 0.014, 0.016]

    BASE_OUT_DIR = "real_system_validation"
    CASE_NAME = "sun_earth_moon_no_nn_T1"
    OUT_DIR = os.path.join(BASE_OUT_DIR, CASE_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[real_system_no_nn] Loading JPL Horizons Sun-Earth-Moon initial conditions.")
    x0, v0, m = load_sun_earth_moon_initial_conditions(epoch=EPOCH)

    print("\nSanity checks:")
    print(f"Sun-Earth distance  = {np.linalg.norm(x0[1] - x0[0]):.6f} AU")
    print(f"Earth-Moon distance = {np.linalg.norm(x0[2] - x0[1]):.6f} AU")
    print(f"G_REAL              = {G_REAL:.8f}")

    print("\n[real_system_no_nn] Loading SIMON model.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put this script in the same folder as pair_correction_nn.pt."
        )
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig(G=G_REAL)

    print("\n[real_system_no_nn] Running ias15 reference once.")
    t_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(
        x0, v0, m, cfg.G, T, N_SAMPLES
    )
    print(f"  ias15 runtime = {perf_ref['total_time_sec']:.6f} s")

    ref_sun_earth = pair_distance(pos_ref, 0, 1)
    ref_earth_moon = pair_distance(pos_ref, 1, 2)
    ref_max_com = max_distance_from_com(pos_ref)
    ref_energy = energy_drift_metrics(pos_ref, vel_ref, m, G=cfg.G, eps=cfg.eps)
    print(f"  ias15 max energy drift = {ref_energy['max_energy_drift_pct']:.6e}%")

    rows_simon = []
    rows_no_nn = []
    pos_simon_all = []
    pos_no_nn_all = []
    vel_simon_all = []
    vel_no_nn_all = []

    print("\n[real_system_no_nn] Running Full SIMON vs No-NN sweep.")
    print(f"  T={T:.2f} yr | n_samples={N_SAMPLES} | dt_values={DT_VALUES}")
    print("  " + "-" * 128)
    print(
        f"  {'dt':>8} | {'mode':<12} | {'runtime(s)':>10} | {'speed':>7} | "
        f"{'NN_frac':>8} | {'RMS0T(AU)':>12} | {'Emax(%)':>10} | "
        f"{'EM range (AU)':>25} | {'bounded':>7}"
    )
    print("  " + "-" * 128)

    for dt in DT_VALUES:
        # Full SIMON
        _, pos_simon, vel_simon, perf_simon = simulate_hybrid_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES, model_type="simon"
        )
        row_simon = summarize_rollout(
            pos_simon, vel_simon, pos_ref, perf_simon,
            "full_SIMON", dt, m, cfg.G, cfg.eps,
        )
        rows_simon.append(row_simon)
        pos_simon_all.append(pos_simon)
        vel_simon_all.append(vel_simon)

        speed_simon = perf_ref["total_time_sec"] / max(row_simon["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'Full SIMON':<12} | {row_simon['runtime_sec']:>10.4f} | "
            f"{speed_simon:>7.3f} | {row_simon['avg_nn_frac']:>8.5f} | "
            f"{row_simon['time_avg_rms']:>12.5e} | {row_simon['max_energy_drift_pct']:>10.3e} | "
            f"{row_simon['earth_moon_min']:.6f} to {row_simon['earth_moon_max']:.6f} | "
            f"{str(row_simon['bounded']):>7}"
        )

        # No-NN baseline
        _, pos_no_nn, vel_no_nn, perf_no_nn = simulate_hybrid_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES, model_type="no_nn"
        )
        row_no_nn = summarize_rollout(
            pos_no_nn, vel_no_nn, pos_ref, perf_no_nn,
            "no_NN", dt, m, cfg.G, cfg.eps,
        )
        rows_no_nn.append(row_no_nn)
        pos_no_nn_all.append(pos_no_nn)
        vel_no_nn_all.append(vel_no_nn)

        speed_no_nn = perf_ref["total_time_sec"] / max(row_no_nn["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'No-NN':<12} | {row_no_nn['runtime_sec']:>10.4f} | "
            f"{speed_no_nn:>7.3f} | {row_no_nn['avg_nn_frac']:>8.5f} | "
            f"{row_no_nn['time_avg_rms']:>12.5e} | {row_no_nn['max_energy_drift_pct']:>10.3e} | "
            f"{row_no_nn['earth_moon_min']:.6f} to {row_no_nn['earth_moon_max']:.6f} | "
            f"{str(row_no_nn['bounded']):>7}"
        )

    print("  " + "-" * 128)

    pos_simon_all = np.stack(pos_simon_all, axis=0)
    pos_no_nn_all = np.stack(pos_no_nn_all, axis=0)
    vel_simon_all = np.stack(vel_simon_all, axis=0)
    vel_no_nn_all = np.stack(vel_no_nn_all, axis=0)

    # Save numeric outputs.
    summary_path = os.path.join(OUT_DIR, "summary_no_nn_real_system.txt")
    csv_path = os.path.join(OUT_DIR, "no_nn_results.csv")
    npz_path = os.path.join(OUT_DIR, "no_nn_results.npz")

    write_summary(
        summary_path,
        T,
        N_SAMPLES,
        DT_VALUES,
        EPOCH,
        x0,
        v0,
        m,
        perf_ref,
        ref_sun_earth,
        ref_earth_moon,
        ref_max_com,
        ref_energy,
        rows_simon,
        rows_no_nn,
    )
    write_csv(rows_simon, rows_no_nn, perf_ref["total_time_sec"], csv_path)
    save_npz(
        npz_path,
        t_ref,
        x0,
        v0,
        m,
        pos_ref,
        vel_ref,
        rows_simon,
        rows_no_nn,
        pos_simon_all,
        pos_no_nn_all,
        vel_simon_all,
        vel_no_nn_all,
        DT_VALUES,
        T,
        cfg.G,
        ref_energy,
    )

    # Save plots.
    plot_metric_vs_dt(
        rows_simon,
        rows_no_nn,
        os.path.join(OUT_DIR, "timeavg_rms_vs_dt.png"),
        "time_avg_rms",
        "Time-averaged RMS vs ias15 (AU)",
        logy=True,
    )
    plot_metric_vs_dt(
        rows_simon,
        rows_no_nn,
        os.path.join(OUT_DIR, "max_rms_vs_dt.png"),
        "max_rms",
        "Max RMS vs ias15 (AU)",
        logy=True,
    )
    plot_metric_vs_dt(
        rows_simon,
        rows_no_nn,
        os.path.join(OUT_DIR, "energy_drift_vs_dt.png"),
        "max_energy_drift_pct",
        "Max energy drift |ΔE/E0| (%)",
        logy=True,
    )
    plot_runtime_vs_dt(
        rows_simon,
        rows_no_nn,
        perf_ref["total_time_sec"],
        os.path.join(OUT_DIR, "runtime_vs_dt.png"),
    )
    plot_speed_ratio_vs_dt(
        rows_simon,
        rows_no_nn,
        perf_ref["total_time_sec"],
        os.path.join(OUT_DIR, "speed_ratio_vs_dt.png"),
    )
    plot_earth_moon_range_vs_dt(
        ref_earth_moon,
        rows_simon,
        rows_no_nn,
        os.path.join(OUT_DIR, "earth_moon_range_vs_dt.png"),
    )

    # Selected representative dt plot: use the largest tested dt, where any
    # model differences should be most visible but adaptive stepping is still ON.
    selected_dt = 0.016
    selected_index = DT_VALUES.index(selected_dt)
    plot_pair_distances_three(
        t_ref,
        ref_sun_earth,
        rows_simon[selected_index]["sun_earth"],
        rows_no_nn[selected_index]["sun_earth"],
        ref_earth_moon,
        rows_simon[selected_index]["earth_moon"],
        rows_no_nn[selected_index]["earth_moon"],
        os.path.join(OUT_DIR, "selected_dt016_distance_comparison.png"),
    )
    plot_xy_trajectories_three(
        pos_ref,
        pos_simon_all[selected_index],
        pos_no_nn_all[selected_index],
        os.path.join(OUT_DIR, "selected_dt016_xy_trajectories.png"),
    )

    print("\n[real_system_no_nn] Saved outputs:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print(f"  plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
