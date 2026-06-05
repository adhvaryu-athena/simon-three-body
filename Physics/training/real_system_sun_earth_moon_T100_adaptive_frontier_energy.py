"""
real_system_sun_earth_moon_T100_adaptive_frontier_energy.py

Long-horizon Track A real-system validation for SIMON using Sun-Earth-Moon
initial conditions from JPL Horizons.

This script runs the selected T=100 yr real-system timestep-frontier experiment:
    ias15 reference
    SIMON adaptive ON
    SIMON adaptive OFF
for selected dt values that were most informative in the T=1 refined sweep.

It keeps the same energy-drift diagnostic used in the corrected T=1 real-system
script and adds an early ejection guard for long-horizon unstable rollouts.

Main outputs:
    real_system_validation/sun_earth_moon_T100_selected_adaptive_frontier_energy/
        summary_T100_adaptive_frontier.txt
        T100_adaptive_frontier_results.csv
        T100_adaptive_frontier_results.npz
        timeavg_rms_vs_dt.png
        max_rms_vs_dt.png
        runtime_vs_dt.png
        speed_ratio_vs_dt.png
        earth_moon_range_vs_dt.png
        energy_drift_vs_dt.png
        selected_dt016_distance_comparison.png
        selected_dt016_xy_trajectories.png

Run:
    python real_system_sun_earth_moon_T100_adaptive_frontier_energy.py

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
# JPL Horizons vectors give position in AU and velocity in AU/day.
# SIMON uses AU, years, solar masses, so velocities are converted to AU/year.
DAYS_PER_YEAR = 365.25

# Physical gravitational constant in AU, years, solar masses.
# This gives G * M_sun = 4*pi^2 for a 1 AU, 1 year orbit.
G_REAL = 4.0 * np.pi**2

# Long-horizon boundedness/ejection guard. This matches the paper's boundedness
# convention: a rollout is no longer considered bounded once any body exceeds
# this distance from the centre-of-mass frame.
EJECTION_THRESHOLD_AU = 10.0


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

    vel = vel_au_per_day * DAYS_PER_YEAR
    return pos, vel


def move_to_center_of_mass(x0, v0, m):
    """
    Shift positions and velocities to the centre-of-mass frame.
    """
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

    x0, v0 = move_to_center_of_mass(x0, v0, m)
    return x0, v0, m


# -----------------------------------------------------------------------------
# SIMON model and configuration
# -----------------------------------------------------------------------------
class PairCorrectionNN(nn.Module):
    """
    Same scalar correction network used in the SIMON experiments.
    """
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
    """
    Convert trained PyTorch weights into NumPy arrays for fast SIMON inference.
    """
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


# -----------------------------------------------------------------------------
# Integrators
# -----------------------------------------------------------------------------
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    """
    Reference integration using REBOUND ias15.
    """
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


def simulate_simon_real_system(
    x0,
    v0,
    m,
    model,
    cfg,
    dt,
    T,
    n_samples,
    adaptive_enabled=True,
    ejection_threshold=EJECTION_THRESHOLD_AU,
    early_stop=True,
):
    """
    SIMON rollout for real Sun-Earth-Moon initial conditions.

    Uses:
      - leapfrog / velocity-Verlet backbone
      - analytic force direction
      - NN scalar correction for r < 0.15 AU
      - optional adaptive sub-stepping for r_min < 0.05 AU
      - optional early ejection stop for long-horizon failed rollouts
    """
    w = extract_weights_numpy(model)

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

            log_c = (h @ w["w3T"] + w["b3"]).ravel()
            c = np.exp(log_c).astype(np.float64)

            fallback = (
                (r_soft_close < r_soft_min)
                | (c < c_min)
                | (c > c_max)
                | ~np.isfinite(c)
            )

            F_corrected = np.where(
                fallback,
                F_scalar[close_mask],
                c * F_soft_close,
            )
            F_scalar[close_mask] = F_corrected

        F_vec = F_scalar[:, None] * rij

        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]

        return acc, n_close

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, n_close = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, n_close

    a, n_close = compute_acc(x)

    close_sum = n_close
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
    ejected = False
    ejection_time = np.nan
    ejection_max_com = np.nan
    steps_done = 0
    completed_time = 0.0

    for _ in range(n_steps):
        r_min = min_pair_dist(x)

        if adaptive_enabled and r_min < adapt_thresh:
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = float(dt) / n_sub
            for _ in range(n_sub):
                x, v, a, n_close = leapfrog_substep(x, v, a, sub_dt)
                close_sum += n_close
                pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * float(dt) * a
            x = x + float(dt) * vh
            a, n_close = compute_acc(x)
            v = vh + 0.5 * float(dt) * a
            close_sum += n_close
            pair_sum += P
            total_substeps += 1

        t_cur += float(dt)

        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]

        steps_done += 1
        completed_time = t_cur

        max_com_now = float(np.max(np.linalg.norm(x, axis=1)))
        if early_stop and (not np.isfinite(max_com_now) or max_com_now > ejection_threshold):
            ejected = True
            ejection_time = float(t_cur)
            ejection_max_com = max_com_now
            # Fill remaining samples with the final failed state so no unwritten
            # zeros contaminate distance or energy diagnostics.
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

    perf = {
        "steps": steps_done,
        "planned_steps": n_steps,
        "completed_time": completed_time,
        "ejected": ejected,
        "ejection_time": ejection_time,
        "ejection_max_com": ejection_max_com,
        "ejection_threshold_AU": ejection_threshold,
        "total_time_sec": time.perf_counter() - t0,
        "avg_close_pair_frac": close_sum / max(pair_sum, 1),
        "total_substeps": total_substeps,
    }

    return times, pos_out, vel_out, perf


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rms_sep(a, b):
    """
    Per-time RMS position separation across bodies.
    Shape expected: (n_samples, n_bodies, 3).
    """
    d = a - b
    per_body = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(per_body**2, axis=1))


def paper_time_avg_rms(delta):
    """
    Paper-consistent metric:
        RMS_{0:T} = sqrt(mean_t delta(t)^2)
    where delta(t) is the per-time RMS separation over bodies.
    """
    return float(np.sqrt(np.mean(delta**2)))


def pair_distance(pos_arr, i, j):
    return np.linalg.norm(pos_arr[:, i, :] - pos_arr[:, j, :], axis=1)


def max_distance_from_com(pos_arr):
    return np.max(np.linalg.norm(pos_arr, axis=2), axis=1)


def compute_energy_trajectory(pos_arr, vel_arr, m, G, eps):
    """
    Compute total mechanical energy E(t) = KE(t) + PE_softened(t).

    This follows the numerically robust convention used in
    energy_drift_scaling_new_float.py: potential energy is softened as
    PE = -G*m_i*m_j/sqrt(r^2 + eps^2). The softening matches SIMON's
    close-pair Hamiltonian and avoids pathological values when a failed
    non-adaptive rollout produces extremely small separations. For ordinary
    Solar-System separations where r >> eps, this is effectively Newtonian.
    
    All trajectory arrays are expected to be float64. The computation is
    vectorised over the time axis to avoid the older slow Python loop.
    """
    m = m.astype(np.float64)
    pos_arr = pos_arr.astype(np.float64, copy=False)
    vel_arr = vel_arr.astype(np.float64, copy=False)

    # KE(t) = 1/2 sum_i m_i |v_i(t)|^2
    KE = 0.5 * np.einsum("kij,i->k", vel_arr**2, m)

    # PE(t) = -G sum_{i<j} m_i*m_j / sqrt(r_ij(t)^2 + eps^2)
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
    Return paper-style energy drift diagnostics.

    dE(t) = (E(t) - E(0)) / |E(0)|
    max_energy_drift_pct is max_t |dE(t)| * 100.
    final_energy_drift_pct is signed dE(T) * 100.
    rms_energy_drift_pct is sqrt(mean_t dE(t)^2) * 100.
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


def classify_lunar_outcome(earth_moon_min, earth_moon_max, ref_em_min, ref_em_max, bounded, ejected):
    """
    Classify the real Sun-Earth-Moon rollout using both the paper's global
    boundedness criterion and a lunar-bound-state criterion.

    The paper's boundedness check uses max distance from COM < 10 AU.
    For Sun-Earth-Moon, that is not sufficient: the Moon can become unbound
    from Earth while the whole system remains within 10 AU. Therefore we also
    compare the Earth-Moon distance range against the ias15 reference range.

    Thresholds:
      - lunar orbit preserved: EM range remains within [0.5*ref_min, 2.0*ref_max]
      - degraded: preserved, but outside a tighter [0.8*ref_min, 1.2*ref_max] band
      - lunar orbit lost: outside the broader preservation band
      - ejected: global COM ejection threshold crossed
    """
    if ejected or (not bounded):
        return False, "Ejected"

    preserve_min = 0.5 * ref_em_min
    preserve_max = 2.0 * ref_em_max
    tight_min = 0.8 * ref_em_min
    tight_max = 1.2 * ref_em_max

    lunar_preserved = (
        np.isfinite(earth_moon_min)
        and np.isfinite(earth_moon_max)
        and earth_moon_min >= preserve_min
        and earth_moon_max <= preserve_max
    )

    if not lunar_preserved:
        return False, "Lunar orbit lost"

    if earth_moon_min < tight_min or earth_moon_max > tight_max:
        return True, "Bounded, degraded"

    return True, "Valid"


def summarize_rollout(
    pos_model,
    vel_model,
    pos_ref,
    perf,
    label,
    dt,
    m,
    G,
    eps,
    ref_em_min,
    ref_em_max,
    bounded_threshold=10.0,
):
    """
    Compute the metrics used in the real-system timestep sweep.
    """
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
        "planned_steps": int(perf.get("planned_steps", perf["steps"])),
        "completed_time": float(perf.get("completed_time", dt * perf["steps"])),
        "ejected": bool(perf.get("ejected", False)),
        "ejection_time": float(perf.get("ejection_time", np.nan)),
        "ejection_max_com": float(perf.get("ejection_max_com", np.nan)),
        "ejection_threshold_AU": float(perf.get("ejection_threshold_AU", bounded_threshold)),
        "total_substeps": int(perf["total_substeps"]),
        "avg_close_pair_frac": float(perf["avg_close_pair_frac"]),
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
def plot_metric_vs_dt(rows_on, rows_off, out_path, metric, ylabel, logy=True):
    dts = np.array([r["dt"] for r in rows_on], dtype=float)
    y_on = np.array([r[metric] for r in rows_on], dtype=float)
    y_off = np.array([r[metric] for r in rows_off], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_on, "o-", lw=1.8, label="SIMON adaptive ON")
    ax.plot(dts, y_off, "s--", lw=1.8, label="SIMON adaptive OFF")
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


def plot_energy_drift_vs_dt(rows_on, rows_off, ref_energy, out_path):
    dts = np.array([r["dt"] for r in rows_on], dtype=float)
    y_on = np.array([r["max_energy_drift_pct"] for r in rows_on], dtype=float)
    y_off = np.array([r["max_energy_drift_pct"] for r in rows_off], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, np.maximum(y_on, 1e-300), "o-", lw=1.8, label="SIMON adaptive ON")
    ax.plot(dts, np.maximum(y_off, 1e-300), "s--", lw=1.8, label="SIMON adaptive OFF")
    ax.axhline(
        max(ref_energy["max_energy_drift_pct"], 1e-300),
        lw=1.2,
        linestyle=":",
        label="ias15 reference",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Max energy drift |ΔE/E0| (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_runtime_vs_dt(rows_on, rows_off, ias15_time, out_path):
    dts = np.array([r["dt"] for r in rows_on], dtype=float)
    y_on = np.array([r["runtime_sec"] for r in rows_on], dtype=float)
    y_off = np.array([r["runtime_sec"] for r in rows_off], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_on, "o-", lw=1.8, label="SIMON adaptive ON")
    ax.plot(dts, y_off, "s--", lw=1.8, label="SIMON adaptive OFF")
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


def plot_speed_ratio_vs_dt(rows_on, rows_off, ias15_time, out_path):
    dts = np.array([r["dt"] for r in rows_on], dtype=float)
    y_on = np.array([ias15_time / max(r["runtime_sec"], 1e-30) for r in rows_on], dtype=float)
    y_off = np.array([ias15_time / max(r["runtime_sec"], 1e-30) for r in rows_off], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(dts, y_on, "o-", lw=1.8, label="SIMON adaptive ON")
    ax.plot(dts, y_off, "s--", lw=1.8, label="SIMON adaptive OFF")
    ax.axhline(1.0, lw=1.2, linestyle=":", label="same runtime as ias15")
    ax.set_xscale("log")
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Speed ratio: ias15 runtime / SIMON runtime")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_earth_moon_range_vs_dt(ref_earth_moon, rows_on, rows_off, out_path):
    dts = np.array([r["dt"] for r in rows_on], dtype=float)
    on_min = np.array([r["earth_moon_min"] for r in rows_on], dtype=float)
    on_max = np.array([r["earth_moon_max"] for r in rows_on], dtype=float)
    off_min = np.array([r["earth_moon_min"] for r in rows_off], dtype=float)
    off_max = np.array([r["earth_moon_max"] for r in rows_off], dtype=float)

    ref_min = float(np.min(ref_earth_moon))
    ref_max = float(np.max(ref_earth_moon))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(dts, on_min, on_max, alpha=0.20, label="SIMON ON range")
    ax.plot(dts, on_min, "o-", lw=1.2)
    ax.plot(dts, on_max, "o-", lw=1.2)
    ax.fill_between(dts, off_min, off_max, alpha=0.20, label="SIMON OFF range")
    ax.plot(dts, off_min, "s--", lw=1.2)
    ax.plot(dts, off_max, "s--", lw=1.2)
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


def plot_pair_distances_three(times, ref_se, on_se, off_se, ref_em, on_em, off_em, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    axes[0].plot(times, ref_se, lw=1.8, label="ias15")
    axes[0].plot(times, on_se, "--", lw=1.8, label="SIMON ON")
    axes[0].plot(times, off_se, ":", lw=1.8, label="SIMON OFF")
    axes[0].set_ylabel("Sun-Earth distance (AU)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    axes[1].plot(times, ref_em, lw=1.8, label="ias15")
    axes[1].plot(times, on_em, "--", lw=1.8, label="SIMON ON")
    axes[1].plot(times, off_em, ":", lw=1.8, label="SIMON OFF")
    axes[1].set_xlabel("Time (yr)")
    axes[1].set_ylabel("Earth-Moon distance (AU)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xy_trajectories_three(pos_ref, pos_on, pos_off, out_path):
    """
    Sun and Earth in barycentric frame; Moon relative to Earth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Sun barycentric motion
    axes[0].plot(pos_ref[:, 0, 0], pos_ref[:, 0, 1], lw=1.5, label="ias15")
    axes[0].plot(pos_on[:, 0, 0], pos_on[:, 0, 1], "--", lw=1.5, label="SIMON ON")
    axes[0].plot(pos_off[:, 0, 0], pos_off[:, 0, 1], ":", lw=1.5, label="SIMON OFF")
    axes[0].set_title("Sun")
    axes[0].set_xlabel("x (AU)")
    axes[0].set_ylabel("y (AU)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    # Earth barycentric orbit
    axes[1].plot(pos_ref[:, 1, 0], pos_ref[:, 1, 1], lw=1.5, label="ias15")
    axes[1].plot(pos_on[:, 1, 0], pos_on[:, 1, 1], "--", lw=1.5, label="SIMON ON")
    axes[1].plot(pos_off[:, 1, 0], pos_off[:, 1, 1], ":", lw=1.5, label="SIMON OFF")
    axes[1].set_title("Earth")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("y (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    # Moon relative to Earth
    moon_rel_ref = pos_ref[:, 2, :] - pos_ref[:, 1, :]
    moon_rel_on = pos_on[:, 2, :] - pos_on[:, 1, :]
    moon_rel_off = pos_off[:, 2, :] - pos_off[:, 1, :]

    axes[2].plot(moon_rel_ref[:, 0], moon_rel_ref[:, 1], lw=1.5, label="ias15")
    axes[2].plot(moon_rel_on[:, 0], moon_rel_on[:, 1], "--", lw=1.5, label="SIMON ON")
    axes[2].plot(moon_rel_off[:, 0], moon_rel_off[:, 1], ":", lw=1.5, label="SIMON OFF")
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
def write_csv(rows_on, rows_off, ias15_time, out_path):
    fieldnames = [
        "mode", "dt", "runtime_sec", "speed_ratio_ias15_over_simon",
        "steps", "planned_steps", "completed_time", "ejected", "ejection_time",
        "ejection_max_com", "ejection_threshold_AU",
        "total_substeps", "avg_close_pair_frac",
        "time_avg_rms", "final_rms", "max_rms",
        "max_energy_drift_pct", "final_energy_drift_pct", "rms_energy_drift_pct",
        "sun_earth_min", "sun_earth_max", "earth_moon_min", "earth_moon_max",
        "max_com", "energy_finite", "bounded", "finite",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, rows in [("adaptive_ON", rows_on), ("adaptive_OFF", rows_off)]:
            for r in rows:
                out = {k: r[k] for k in fieldnames if k in r}
                out["mode"] = mode
                out["speed_ratio_ias15_over_simon"] = ias15_time / max(r["runtime_sec"], 1e-30)
                writer.writerow(out)


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
    rows_on,
    rows_off,
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Real Sun-Earth-Moon T=100 selected timestep frontier using JPL Horizons initial conditions\n")
        f.write("=" * 82 + "\n\n")
        f.write(f"Epoch                   = {epoch}\n")
        f.write(f"T                       = {T:.2f} yr\n")
        f.write(f"n_samples               = {n_samples}\n")
        f.write(f"dt_values               = {', '.join(f'{d:.4f}' for d in dt_values)} yr\n")
        f.write(f"G_REAL                  = {G_REAL:.8f}\n")
        f.write(f"ejection_threshold      = {EJECTION_THRESHOLD_AU:.2f} AU from COM\n")
        f.write(f"ias15 runtime            = {perf_ref['total_time_sec']:.6f} s\n\n")

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
            f"{'substeps':>10} {'RMS0T(AU)':>13} {'Emax(%)':>12} "
            f"{'final(AU)':>12} {'max(AU)':>12} {'EM_min':>10} "
            f"{'EM_max':>10} {'bounded':>8} {'ejected':>8} {'t_ej':>10}\n"
        )
        f.write("-" * 162 + "\n")
        for mode, rows in [("adaptive_ON", rows_on), ("adaptive_OFF", rows_off)]:
            for r in rows:
                speed = perf_ref["total_time_sec"] / max(r["runtime_sec"], 1e-30)
                f.write(
                    f"{mode:<14} {r['dt']:>8.4f} {r['runtime_sec']:>12.6f} {speed:>9.3f} "
                    f"{r['total_substeps']:>10d} {r['time_avg_rms']:>13.6e} "
                    f"{r['max_energy_drift_pct']:>12.6e} "
                    f"{r['final_rms']:>12.6e} {r['max_rms']:>12.6e} "
                    f"{r['earth_moon_min']:>10.6f} {r['earth_moon_max']:>10.6f} "
                    f"{str(r['bounded']):>8} {str(r['ejected']):>8} "
                    f"{(r['ejection_time'] if np.isfinite(r['ejection_time']) else np.nan):>10.4f}\n"
                )
            f.write("\n")

        f.write("Interpretation guide:\n")
        f.write("- RMS0T is the paper-consistent time-averaged RMS: sqrt(mean_t delta(t)^2).\n")
        f.write("- Emax is max_t |(E(t)-E(0))/E(0)| in percent, using the same softened-energy convention as the corrected energy-drift scaling script.\n")
        f.write("- Speed is ias15 runtime divided by SIMON runtime; speed < 1 means SIMON is slower.\n")
        f.write("- This real Sun-Earth-Moon case is a physical-consistency and timestep-frontier test, not a guaranteed speedup case.\n")
        f.write("- Because the Earth-Moon pair is always below the adaptive threshold, adaptive ON can be slower even when stable.\n")
        f.write("- Long-horizon failed rollouts are stopped once any body exceeds the 10 AU boundedness threshold; remaining samples are filled with the final state to avoid unwritten zero-sample artifacts.\n")


def save_npz(
    out_path,
    times,
    x0,
    v0,
    m,
    pos_ref,
    vel_ref,
    rows_on,
    rows_off,
    pos_on_all,
    pos_off_all,
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
        pos_adaptive_on=pos_on_all,
        pos_adaptive_off=pos_off_all,
        dt_values=np.array(dt_values, dtype=np.float64),
        T=T,
        G_REAL=G,
        time_avg_rms_on=np.array([r["time_avg_rms"] for r in rows_on], dtype=np.float64),
        time_avg_rms_off=np.array([r["time_avg_rms"] for r in rows_off], dtype=np.float64),
        final_rms_on=np.array([r["final_rms"] for r in rows_on], dtype=np.float64),
        final_rms_off=np.array([r["final_rms"] for r in rows_off], dtype=np.float64),
        max_rms_on=np.array([r["max_rms"] for r in rows_on], dtype=np.float64),
        max_rms_off=np.array([r["max_rms"] for r in rows_off], dtype=np.float64),
        runtime_on=np.array([r["runtime_sec"] for r in rows_on], dtype=np.float64),
        runtime_off=np.array([r["runtime_sec"] for r in rows_off], dtype=np.float64),
        substeps_on=np.array([r["total_substeps"] for r in rows_on], dtype=np.int64),
        substeps_off=np.array([r["total_substeps"] for r in rows_off], dtype=np.int64),
        max_energy_drift_pct_ref=ref_energy["max_energy_drift_pct"],
        final_energy_drift_pct_ref=ref_energy["final_energy_drift_pct"],
        rms_energy_drift_pct_ref=ref_energy["rms_energy_drift_pct"],
        energy_drift_pct_ref=ref_energy["energy_drift_pct_series"],
        max_energy_drift_pct_on=np.array([r["max_energy_drift_pct"] for r in rows_on], dtype=np.float64),
        max_energy_drift_pct_off=np.array([r["max_energy_drift_pct"] for r in rows_off], dtype=np.float64),
        final_energy_drift_pct_on=np.array([r["final_energy_drift_pct"] for r in rows_on], dtype=np.float64),
        final_energy_drift_pct_off=np.array([r["final_energy_drift_pct"] for r in rows_off], dtype=np.float64),
        rms_energy_drift_pct_on=np.array([r["rms_energy_drift_pct"] for r in rows_on], dtype=np.float64),
        rms_energy_drift_pct_off=np.array([r["rms_energy_drift_pct"] for r in rows_off], dtype=np.float64),
        energy_drift_pct_on=np.stack([r["energy_drift_pct_series"] for r in rows_on], axis=0),
        energy_drift_pct_off=np.stack([r["energy_drift_pct_series"] for r in rows_off], axis=0),
        bounded_on=np.array([r["bounded"] for r in rows_on], dtype=bool),
        bounded_off=np.array([r["bounded"] for r in rows_off], dtype=bool),
        ejected_on=np.array([r["ejected"] for r in rows_on], dtype=bool),
        ejected_off=np.array([r["ejected"] for r in rows_off], dtype=bool),
        ejection_time_on=np.array([r["ejection_time"] for r in rows_on], dtype=np.float64),
        ejection_time_off=np.array([r["ejection_time"] for r in rows_off], dtype=np.float64),
        ejection_max_com_on=np.array([r["ejection_max_com"] for r in rows_on], dtype=np.float64),
        ejection_max_com_off=np.array([r["ejection_max_com"] for r in rows_off], dtype=np.float64),
        ejection_threshold_AU=EJECTION_THRESHOLD_AU,
    )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    MODEL_PATH = "pair_correction_nn.pt"
    EPOCH = "2026-01-01"

    # Selected long-horizon Track A settings.
    # These dt values were selected from the T=1 refined sweep:
    #   0.008 = stable for ON/OFF at T=1
    #   0.012 = OFF begins to distort Earth-Moon range at T=1
    #   0.016 = OFF fails at T=1
    T = 100.0
    N_SAMPLES = 5000
    DT_VALUES = [0.008, 0.012, 0.016]

    BASE_OUT_DIR = "real_system_validation"
    CASE_NAME = "sun_earth_moon_T100_selected_adaptive_frontier_energy"
    OUT_DIR = os.path.join(BASE_OUT_DIR, CASE_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[real_system_T100] Loading JPL Horizons Sun-Earth-Moon initial conditions.")
    x0, v0, m = load_sun_earth_moon_initial_conditions(epoch=EPOCH)

    print("\nSanity checks:")
    print(f"Sun-Earth distance  = {np.linalg.norm(x0[1] - x0[0]):.6f} AU")
    print(f"Earth-Moon distance = {np.linalg.norm(x0[2] - x0[1]):.6f} AU")
    print(f"G_REAL              = {G_REAL:.8f}")

    print("\n[real_system_T100] Loading SIMON model.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Put this script in the same folder as pair_correction_nn.pt."
        )
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig(G=G_REAL)

    print("\n[real_system_T100] Running ias15 reference once.")
    t_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(
        x0, v0, m, cfg.G, T, N_SAMPLES
    )
    print(f"  ias15 runtime = {perf_ref['total_time_sec']:.6f} s")

    ref_sun_earth = pair_distance(pos_ref, 0, 1)
    ref_earth_moon = pair_distance(pos_ref, 1, 2)
    ref_max_com = max_distance_from_com(pos_ref)
    ref_energy = energy_drift_metrics(pos_ref, vel_ref, m, G=cfg.G, eps=cfg.eps)
    print(f"  ias15 max energy drift = {ref_energy['max_energy_drift_pct']:.6e}%")

    rows_on = []
    rows_off = []
    pos_on_all = []
    pos_off_all = []
    vel_on_all = []
    vel_off_all = []

    print("\n[real_system_T100] Running selected long-horizon timestep sweep.")
    print(f"  T={T:.2f} yr | n_samples={N_SAMPLES} | dt_values={DT_VALUES}")
    print("  " + "-" * 136)
    print(
        f"  {'dt':>8} | {'mode':<12} | {'runtime(s)':>10} | {'speed':>7} | "
        f"{'RMS0T(AU)':>12} | {'Emax(%)':>10} | {'EM range (AU)':>25} | "
        f"{'bounded':>7} | {'ejected':>7} | {'t_ej':>8}"
    )
    print("  " + "-" * 136)

    for dt in DT_VALUES:
        # Adaptive ON
        _, pos_on, vel_on, perf_on = simulate_simon_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES, adaptive_enabled=True
        )
        row_on = summarize_rollout(pos_on, vel_on, pos_ref, perf_on, "adaptive_ON", dt, m, cfg.G, cfg.eps)
        rows_on.append(row_on)
        pos_on_all.append(pos_on)
        vel_on_all.append(vel_on)

        speed_on = perf_ref["total_time_sec"] / max(row_on["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'adaptive ON':<12} | {row_on['runtime_sec']:>10.4f} | "
            f"{speed_on:>7.3f} | {row_on['time_avg_rms']:>12.5e} | "
            f"{row_on['max_energy_drift_pct']:>10.3e} | "
            f"{row_on['earth_moon_min']:.6f} to {row_on['earth_moon_max']:.6f} | "
            f"{str(row_on['bounded']):>7} | {str(row_on['ejected']):>7} | "
            f"{(row_on['ejection_time'] if np.isfinite(row_on['ejection_time']) else np.nan):>8.3f}"
        )

        # Adaptive OFF
        _, pos_off, vel_off, perf_off = simulate_simon_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES, adaptive_enabled=False
        )
        row_off = summarize_rollout(pos_off, vel_off, pos_ref, perf_off, "adaptive_OFF", dt, m, cfg.G, cfg.eps)
        rows_off.append(row_off)
        pos_off_all.append(pos_off)
        vel_off_all.append(vel_off)

        speed_off = perf_ref["total_time_sec"] / max(row_off["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'adaptive OFF':<12} | {row_off['runtime_sec']:>10.4f} | "
            f"{speed_off:>7.3f} | {row_off['time_avg_rms']:>12.5e} | "
            f"{row_off['max_energy_drift_pct']:>10.3e} | "
            f"{row_off['earth_moon_min']:.6f} to {row_off['earth_moon_max']:.6f} | "
            f"{str(row_off['bounded']):>7} | {str(row_off['ejected']):>7} | "
            f"{(row_off['ejection_time'] if np.isfinite(row_off['ejection_time']) else np.nan):>8.3f}"
        )

    print("  " + "-" * 136)

    pos_on_all = np.stack(pos_on_all, axis=0)
    pos_off_all = np.stack(pos_off_all, axis=0)
    vel_on_all = np.stack(vel_on_all, axis=0)
    vel_off_all = np.stack(vel_off_all, axis=0)

    # Save numeric outputs.
    summary_path = os.path.join(OUT_DIR, "summary_T100_adaptive_frontier.txt")
    csv_path = os.path.join(OUT_DIR, "T100_adaptive_frontier_results.csv")
    npz_path = os.path.join(OUT_DIR, "T100_adaptive_frontier_results.npz")

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
        rows_on,
        rows_off,
    )
    write_csv(rows_on, rows_off, perf_ref["total_time_sec"], csv_path)
    save_npz(
        npz_path,
        t_ref,
        x0,
        v0,
        m,
        pos_ref,
        vel_ref,
        rows_on,
        rows_off,
        pos_on_all,
        pos_off_all,
        DT_VALUES,
        T,
        G_REAL,
        ref_energy,
    )

    # Save plots.
    plot_metric_vs_dt(
        rows_on, rows_off,
        os.path.join(OUT_DIR, "timeavg_rms_vs_dt.png"),
        metric="time_avg_rms",
        ylabel="Time-averaged RMS vs ias15 (AU)",
        logy=True,
    )
    plot_metric_vs_dt(
        rows_on, rows_off,
        os.path.join(OUT_DIR, "max_rms_vs_dt.png"),
        metric="max_rms",
        ylabel="Max RMS vs ias15 (AU)",
        logy=True,
    )
    plot_runtime_vs_dt(
        rows_on, rows_off, perf_ref["total_time_sec"],
        os.path.join(OUT_DIR, "runtime_vs_dt.png"),
    )
    plot_speed_ratio_vs_dt(
        rows_on, rows_off, perf_ref["total_time_sec"],
        os.path.join(OUT_DIR, "speed_ratio_vs_dt.png"),
    )
    plot_earth_moon_range_vs_dt(
        ref_earth_moon, rows_on, rows_off,
        os.path.join(OUT_DIR, "earth_moon_range_vs_dt.png"),
    )
    plot_energy_drift_vs_dt(
        rows_on, rows_off, ref_energy,
        os.path.join(OUT_DIR, "energy_drift_vs_dt.png"),
    )

    # Also save detailed plots for the coarsest selected timestep, dt=0.016.
    selected_idx = len(DT_VALUES) - 1
    plot_pair_distances_three(
        t_ref,
        ref_sun_earth,
        rows_on[selected_idx]["sun_earth"],
        rows_off[selected_idx]["sun_earth"],
        ref_earth_moon,
        rows_on[selected_idx]["earth_moon"],
        rows_off[selected_idx]["earth_moon"],
        os.path.join(OUT_DIR, "selected_dt016_distance_comparison.png"),
    )
    plot_xy_trajectories_three(
        pos_ref,
        pos_on_all[selected_idx],
        pos_off_all[selected_idx],
        os.path.join(OUT_DIR, "selected_dt016_xy_trajectories.png"),
    )

    print("\n[real_system_T100] Saved outputs:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print(f"  plots in {OUT_DIR}")

    print("\n[real_system_T100] Complete. Paste the printed timestep table back into ChatGPT for interpretation.")


if __name__ == "__main__":
    main()
