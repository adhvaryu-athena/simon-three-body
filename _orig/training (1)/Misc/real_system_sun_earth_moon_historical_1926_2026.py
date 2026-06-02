"""
real_system_sun_earth_moon_historical_1926_2026.py

Historical real-system validation for SIMON using Sun-Earth-Moon JPL Horizons
state vectors from 1926-01-01 to 2026-01-01.

Purpose:
    Compare three trajectories over the same historical interval:
        1. Horizons ephemeris samples ("actual" / reconstructed ephemeris)
        2. ias15 three-body rollout from the 1926 Horizons state
        3. SIMON three-body rollout from the same 1926 Horizons state

Interpretation:
    Horizons includes the full ephemeris model and observationally constrained Solar
    System dynamics. ias15 and SIMON here solve only the simplified Sun-Earth-Moon
    three-body system. Therefore:
        Horizons - ias15  = missing-physics gap of the simplified three-body model
        SIMON - ias15     = SIMON's error relative to the same simplified model
        Horizons - SIMON  = missing-physics gap + SIMON numerical/model error

Run:
    python real_system_sun_earth_moon_historical_1926_2026.py

Requirements in same folder:
    pair_correction_nn.pt

Outputs:
    real_system_validation/sun_earth_moon_historical_1926_2026/
        summary_historical_1926_2026.txt
        historical_1926_2026_metrics.csv
        historical_1926_2026_data.npz
        rms_error_budget_vs_time.png
        distance_comparison_historical.png
        xy_historical_comparison.png
        rms_timeavg_bar.png
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "pair_correction_nn.pt"
OUT_ROOT = "real_system_validation"
CASE_NAME = "sun_earth_moon_historical_1926_2026"
OUT_DIR = os.path.join(OUT_ROOT, CASE_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

START_EPOCH = "1926-01-01"
STOP_EPOCH = "2026-01-01"
# 14-day sampling gives about 2600 ephemeris samples: enough for clear plots
# without making the Horizons query unnecessarily heavy.
HORIZONS_STEP = "14d"

# SIMON timestep: chosen as the largest adaptive-ON timestep that remained valid
# in the final T=100 Sun-Earth-Moon adaptive-frontier experiment.
DT_SIMON = 0.016

DAYS_PER_YEAR = 365.25
G_REAL = 4.0 * np.pi**2
EJECTION_THRESHOLD_AU = 10.0

# JPL Horizons can be slow on first query.
Horizons.TIMEOUT = 180

# Masses in solar masses.
MASS_SEM = np.array([
    1.0,                 # Sun
    3.003489614915e-6,   # Earth
    3.694303349e-8,      # Moon
], dtype=np.float64)

BODY_IDS = ["10", "399", "301"]
BODY_NAMES = ["Sun", "Earth", "Moon"]


# -----------------------------------------------------------------------------
# Model and SIMON configuration
# -----------------------------------------------------------------------------
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
    G: float = G_REAL
    eps: float = 3e-4
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


# -----------------------------------------------------------------------------
# Horizons data loading
# -----------------------------------------------------------------------------
def query_horizons_body(body_id, start_epoch, stop_epoch, step):
    """Return jd, ISO date strings, positions AU, velocities AU/year for one body."""
    print(f"[Horizons] Fetching body {body_id} from {start_epoch} to {stop_epoch} step={step} ...")
    obj = Horizons(
        id=body_id,
        location="@0",  # Solar System barycentre
        epochs={"start": start_epoch, "stop": stop_epoch, "step": step},
    )
    vec = obj.vectors()

    jd = np.array(vec["datetime_jd"], dtype=np.float64)
    dates = np.array(vec["datetime_str"], dtype=str)
    pos = np.column_stack([
        np.array(vec["x"], dtype=np.float64),
        np.array(vec["y"], dtype=np.float64),
        np.array(vec["z"], dtype=np.float64),
    ])
    vel_au_day = np.column_stack([
        np.array(vec["vx"], dtype=np.float64),
        np.array(vec["vy"], dtype=np.float64),
        np.array(vec["vz"], dtype=np.float64),
    ])
    vel = vel_au_day * DAYS_PER_YEAR

    print(f"  body {body_id}: {len(jd)} samples | first={dates[0]} | last={dates[-1]}")
    return jd, dates, pos, vel


def move_trajectory_to_three_body_com(pos, vel, m):
    """
    Shift an entire trajectory to the COM of the three selected bodies.
    pos, vel shapes: (n_samples, 3, 3)
    """
    M = np.sum(m)
    x_com = np.einsum("i,kij->kj", m, pos) / M
    v_com = np.einsum("i,kij->kj", m, vel) / M
    return pos - x_com[:, None, :], vel - v_com[:, None, :]


def load_horizons_sem_trajectory():
    """Load sampled JPL Horizons Sun-Earth-Moon trajectory and COM-shift it."""
    jd_list = []
    dates_list = []
    pos_list = []
    vel_list = []

    for body_id in BODY_IDS:
        jd, dates, pos, vel = query_horizons_body(body_id, START_EPOCH, STOP_EPOCH, HORIZONS_STEP)
        jd_list.append(jd)
        dates_list.append(dates)
        pos_list.append(pos)
        vel_list.append(vel)

    # Check that all bodies returned the same sample times.
    jd0 = jd_list[0]
    for idx, jd in enumerate(jd_list[1:], start=1):
        if len(jd) != len(jd0) or not np.allclose(jd, jd0, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"Horizons sample-time mismatch between body {BODY_IDS[0]} and {BODY_IDS[idx]}."
            )

    if len(jd0) < 100:
        raise RuntimeError(f"Too few Horizons samples returned: {len(jd0)}")

    if not np.all(np.diff(jd0) > 0):
        raise RuntimeError("Horizons sample times are not strictly increasing.")

    pos = np.stack(pos_list, axis=1)  # (n_samples, 3 bodies, 3 coords)
    vel = np.stack(vel_list, axis=1)
    pos_com, vel_com = move_trajectory_to_three_body_com(pos, vel, MASS_SEM)

    times_years = (jd0 - jd0[0]) / DAYS_PER_YEAR
    dates = dates_list[0]

    print("\n[Horizons] DATA SUCCESS")
    print(f"  Samples downloaded       = {len(times_years)}")
    print(f"  First epoch              = {dates[0]}")
    print(f"  Last epoch               = {dates[-1]}")
    print(f"  Historical span          = {times_years[-1]:.6f} yr")
    print(f"  Initial Sun-Earth dist   = {np.linalg.norm(pos_com[0, 0] - pos_com[0, 1]):.6f} AU")
    print(f"  Initial Earth-Moon dist  = {np.linalg.norm(pos_com[0, 1] - pos_com[0, 2]):.6f} AU")
    print(f"  Final Horizons EM dist   = {np.linalg.norm(pos_com[-1, 1] - pos_com[-1, 2]):.6f} AU")

    return times_years, dates, pos_com, vel_com


# -----------------------------------------------------------------------------
# Integrators
# -----------------------------------------------------------------------------
def simulate_rebound_ias15_at_times(x0, v0, m, G, sample_times):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G

    for i in range(len(m)):
        sim.add(
            m=float(m[i]),
            x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
            vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]),
        )
    sim.move_to_com()

    pos = np.zeros((len(sample_times), len(m), 3), dtype=np.float64)
    vel = np.zeros((len(sample_times), len(m), 3), dtype=np.float64)

    print("\n[ias15] Running reference rollout at Horizons sample times ...")
    t0 = time.perf_counter()
    for k, t in enumerate(sample_times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    runtime = time.perf_counter() - t0
    print(f"[ias15] SUCCESS | runtime={runtime:.3f} s | samples={len(sample_times)}")
    return pos, vel, {"total_time_sec": runtime}


def simulate_simon_real_system_at_times(
    x0,
    v0,
    m,
    model,
    cfg,
    dt,
    sample_times,
    adaptive_enabled=True,
    ejection_threshold=EJECTION_THRESHOLD_AU,
    early_stop=True,
):
    """SIMON rollout with arbitrary output sample times."""
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

    T = float(sample_times[-1])
    times = np.asarray(sample_times, dtype=np.float64)
    n_steps = int(math.ceil(T / dt))

    pos_out = np.zeros((len(times), N, 3), dtype=np.float64)
    vel_out = np.zeros((len(times), N, 3), dtype=np.float64)

    nn_thresh = 500.0 * cfg.eps  # 0.15 AU
    adapt_thresh = 0.05
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
            h = h @ w["w0T"] + w["b0"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w["w1T"] + w["b1"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w["w2T"] + w["b2"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            log_c = (h @ w["w3T"] + w["b3"]).ravel()
            c = np.exp(log_c).astype(np.float64)

            fallback = (
                (r_soft_close < r_soft_min)
                | (c < c_min)
                | (c > c_max)
                | ~np.isfinite(c)
            )
            F_scalar[close_mask] = np.where(
                fallback,
                F_scalar[close_mask],
                c * F_soft_close,
            )

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
    while si < len(times) and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < len(times):
            nt = times[si]

    print("\n[SIMON] Running historical rollout ...")
    print(f"  dt={dt:.6f} yr | T={T:.6f} yr | adaptive_enabled={adaptive_enabled}")
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
        while si < len(times) and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < len(times):
                nt = times[si]

        steps_done += 1
        completed_time = t_cur

        max_com_now = float(np.max(np.linalg.norm(x, axis=1)))
        if early_stop and (not np.isfinite(max_com_now) or max_com_now > ejection_threshold):
            ejected = True
            ejection_time = float(t_cur)
            ejection_max_com = max_com_now
            while si < len(times):
                pos_out[si] = x
                vel_out[si] = v
                si += 1
            break

        if t_cur >= T - 1e-12:
            break

    while si < len(times):
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    runtime = time.perf_counter() - t0
    print(
        f"[SIMON] SUCCESS | runtime={runtime:.3f} s | steps={steps_done}/{n_steps} | "
        f"substeps={total_substeps} | ejected={ejected} | t_ej={ejection_time}"
    )

    perf = {
        "steps": steps_done,
        "planned_steps": n_steps,
        "completed_time": completed_time,
        "ejected": ejected,
        "ejection_time": ejection_time,
        "ejection_max_com": ejection_max_com,
        "ejection_threshold_AU": ejection_threshold,
        "total_time_sec": runtime,
        "avg_close_pair_frac": close_sum / max(pair_sum, 1),
        "total_substeps": total_substeps,
    }
    return pos_out, vel_out, perf


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rms_sep(a, b):
    d = a - b
    per_body = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(per_body**2, axis=1))


def paper_time_avg_rms(delta):
    return float(np.sqrt(np.mean(delta**2)))


def pair_distance(pos_arr, i, j):
    return np.linalg.norm(pos_arr[:, i, :] - pos_arr[:, j, :], axis=1)


def max_distance_from_com(pos_arr):
    return np.max(np.linalg.norm(pos_arr, axis=2), axis=1)


def compute_energy_trajectory(pos_arr, vel_arr, m, G, eps):
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
    dE_pct = ((E - E0) / denom) * 100.0
    return {
        "energy_E0": E0,
        "max_energy_drift_pct": float(np.max(np.abs(dE_pct))),
        "final_energy_drift_pct": float(dE_pct[-1]),
        "rms_energy_drift_pct": float(np.sqrt(np.mean(dE_pct**2))),
        "energy_drift_pct_series": dE_pct.astype(np.float64),
        "energy_finite": True,
    }


def metric_row(label, delta, pos, vel, ref_pos_for_ranges, cfg):
    sun_earth = pair_distance(pos, 0, 1)
    earth_moon = pair_distance(pos, 1, 2)
    max_com = max_distance_from_com(pos)
    energy = energy_drift_metrics(pos, vel, MASS_SEM, G=cfg.G, eps=cfg.eps)
    return {
        "label": label,
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
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_rms_error_budget(times, delta_hi, delta_hs, delta_si, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(times, np.maximum(delta_hi, 1e-12), lw=1.7, label="Horizons - ias15")
    ax.semilogy(times, np.maximum(delta_hs, 1e-12), lw=1.7, label="Horizons - SIMON")
    ax.semilogy(times, np.maximum(delta_si, 1e-12), lw=1.7, linestyle="--", label="SIMON - ias15")
    ax.set_xlabel("Time since 1926-01-01 (yr)")
    ax.set_ylabel("RMS position separation (AU)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_distance_comparison(times, pos_h, pos_i, pos_s, out_path):
    h_se = pair_distance(pos_h, 0, 1); i_se = pair_distance(pos_i, 0, 1); s_se = pair_distance(pos_s, 0, 1)
    h_em = pair_distance(pos_h, 1, 2); i_em = pair_distance(pos_i, 1, 2); s_em = pair_distance(pos_s, 1, 2)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    axes[0].plot(times, h_se, lw=1.5, label="Horizons")
    axes[0].plot(times, i_se, "--", lw=1.3, label="ias15 3-body")
    axes[0].plot(times, s_se, ":", lw=1.5, label="SIMON 3-body")
    axes[0].set_ylabel("Sun-Earth distance (AU)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    axes[1].plot(times, h_em, lw=1.5, label="Horizons")
    axes[1].plot(times, i_em, "--", lw=1.3, label="ias15 3-body")
    axes[1].plot(times, s_em, ":", lw=1.5, label="SIMON 3-body")
    axes[1].set_xlabel("Time since 1926-01-01 (yr)")
    axes[1].set_ylabel("Earth-Moon distance (AU)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xy_comparison(pos_h, pos_i, pos_s, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Sun barycentric/three-body COM motion
    axes[0].plot(pos_h[:, 0, 0], pos_h[:, 0, 1], lw=1.2, label="Horizons")
    axes[0].plot(pos_i[:, 0, 0], pos_i[:, 0, 1], "--", lw=1.2, label="ias15")
    axes[0].plot(pos_s[:, 0, 0], pos_s[:, 0, 1], ":", lw=1.4, label="SIMON")
    axes[0].set_title("Sun")
    axes[0].set_xlabel("x (AU)")
    axes[0].set_ylabel("y (AU)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85, fontsize=8)

    # Earth in COM frame
    axes[1].plot(pos_h[:, 1, 0], pos_h[:, 1, 1], lw=1.2, label="Horizons")
    axes[1].plot(pos_i[:, 1, 0], pos_i[:, 1, 1], "--", lw=1.2, label="ias15")
    axes[1].plot(pos_s[:, 1, 0], pos_s[:, 1, 1], ":", lw=1.4, label="SIMON")
    axes[1].set_title("Earth")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("y (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    # Moon relative to Earth
    h_rel = pos_h[:, 2, :] - pos_h[:, 1, :]
    i_rel = pos_i[:, 2, :] - pos_i[:, 1, :]
    s_rel = pos_s[:, 2, :] - pos_s[:, 1, :]
    axes[2].plot(h_rel[:, 0], h_rel[:, 1], lw=1.2, label="Horizons")
    axes[2].plot(i_rel[:, 0], i_rel[:, 1], "--", lw=1.2, label="ias15")
    axes[2].plot(s_rel[:, 0], s_rel[:, 1], ":", lw=1.4, label="SIMON")
    axes[2].set_title("Moon relative to Earth")
    axes[2].set_xlabel("x rel. Earth (AU)")
    axes[2].set_ylabel("y rel. Earth (AU)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_timeavg_bar(rows, out_path):
    labels = [r["label"] for r in rows]
    values = [r["time_avg_rms"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values)
    ax.set_ylabel(r"Time-averaged RMS$_{0:T}$ (AU)")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def write_metrics_csv(path, rows):
    fields = [
        "label", "time_avg_rms", "final_rms", "max_rms",
        "sun_earth_min", "sun_earth_max", "earth_moon_min", "earth_moon_max",
        "max_com", "max_energy_drift_pct", "final_energy_drift_pct",
        "rms_energy_drift_pct", "energy_E0",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_summary(path, config, runtime_info, rows, sanity):
    with open(path, "w", encoding="utf-8") as f:
        f.write("HISTORICAL SUN-EARTH-MOON VALIDATION: 1926 -> 2026\n")
        f.write("=" * 72 + "\n\n")
        f.write("Purpose:\n")
        f.write("Compare Horizons ephemeris samples, ias15 three-body rollout, and SIMON three-body rollout.\n\n")
        f.write("Interpretation:\n")
        f.write("Horizons-ias15 estimates the missing-physics gap of the simplified 3-body model.\n")
        f.write("SIMON-ias15 estimates SIMON's error under the same simplified 3-body equations.\n")
        f.write("Horizons-SIMON combines missing-physics gap plus SIMON error.\n\n")
        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nRuntime:\n")
        for k, v in runtime_info.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nSanity checks:\n")
        for k, v in sanity.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nMetrics:\n")
        f.write("label                    RMS0T(AU)       final(AU)       max(AU)      SE_min      SE_max      EM_min      EM_max       Emax(%)\n")
        f.write("-" * 128 + "\n")
        for r in rows:
            f.write(
                f"{r['label']:<22} {r['time_avg_rms']:>13.6e} {r['final_rms']:>15.6e} "
                f"{r['max_rms']:>13.6e} {r['sun_earth_min']:>11.6f} {r['sun_earth_max']:>11.6f} "
                f"{r['earth_moon_min']:>11.6f} {r['earth_moon_max']:>11.6f} "
                f"{r['max_energy_drift_pct']:>12.6e}\n"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("[historical_sem] Starting 1926 -> 2026 historical validation")
    print(f"  Output folder: {OUT_DIR}")
    print(f"  Horizons range: {START_EPOCH} to {STOP_EPOCH} step={HORIZONS_STEP}")
    print(f"  SIMON dt: {DT_SIMON} yr")

    times, dates, pos_h, vel_h = load_horizons_sem_trajectory()
    T_hist = float(times[-1])

    # Initial condition from the first Horizons sample, already COM-shifted.
    x0 = pos_h[0].copy()
    v0 = vel_h[0].copy()
    m = MASS_SEM.copy()

    cfg = HybridConfig()

    print("\n[model] Loading SIMON model ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find {MODEL_PATH}. Place it in the same folder as this script.")
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"[model] SUCCESS | {MODEL_PATH} | params={sum(p.numel() for p in model.parameters())}")

    # Run reference and SIMON.
    pos_i, vel_i, perf_i = simulate_rebound_ias15_at_times(x0, v0, m, cfg.G, times)
    pos_s, vel_s, perf_s = simulate_simon_real_system_at_times(
        x0, v0, m, model, cfg, DT_SIMON, times, adaptive_enabled=True
    )

    # Basic validity checks.
    if not np.all(np.isfinite(pos_i)) or not np.all(np.isfinite(pos_s)):
        raise RuntimeError("Non-finite positions found in ias15 or SIMON output.")
    if np.max(np.linalg.norm(pos_s, axis=2)) > EJECTION_THRESHOLD_AU:
        print("[WARNING] SIMON exceeds global ejection threshold in historical validation.")

    print("\n[metrics] Computing error budgets ...")
    delta_hi = rms_sep(pos_h, pos_i)  # missing-physics gap
    delta_hs = rms_sep(pos_h, pos_s)  # missing-physics + SIMON
    delta_si = rms_sep(pos_s, pos_i)  # SIMON under same 3-body model

    row_hi = metric_row("Horizons - ias15", delta_hi, pos_i, vel_i, pos_h, cfg)
    row_hs = metric_row("Horizons - SIMON", delta_hs, pos_s, vel_s, pos_h, cfg)
    row_si = metric_row("SIMON - ias15", delta_si, pos_s, vel_s, pos_i, cfg)
    rows = [row_hi, row_hs, row_si]

    print("\n[historical_sem] RESULT SUMMARY")
    print("  " + "-" * 110)
    print(f"  {'comparison':<22} | {'RMS0T(AU)':>12} | {'final(AU)':>12} | {'max(AU)':>12} | {'EM range of model (AU)':>25} | {'Emax(%)':>10}")
    print("  " + "-" * 110)
    for r in rows:
        print(
            f"  {r['label']:<22} | {r['time_avg_rms']:>12.5e} | {r['final_rms']:>12.5e} | "
            f"{r['max_rms']:>12.5e} | {r['earth_moon_min']:.6f} to {r['earth_moon_max']:.6f} | "
            f"{r['max_energy_drift_pct']:>10.3e}"
        )
    print("  " + "-" * 110)

    print("\n[plots] Saving visual diagnostics ...")
    plot_rms_error_budget(times, delta_hi, delta_hs, delta_si, os.path.join(OUT_DIR, "rms_error_budget_vs_time.png"))
    plot_distance_comparison(times, pos_h, pos_i, pos_s, os.path.join(OUT_DIR, "distance_comparison_historical.png"))
    plot_xy_comparison(pos_h, pos_i, pos_s, os.path.join(OUT_DIR, "xy_historical_comparison.png"))
    plot_timeavg_bar(rows, os.path.join(OUT_DIR, "rms_timeavg_bar.png"))

    # Save data and summary.
    csv_path = os.path.join(OUT_DIR, "historical_1926_2026_metrics.csv")
    write_metrics_csv(csv_path, rows)

    npz_path = os.path.join(OUT_DIR, "historical_1926_2026_data.npz")
    np.savez_compressed(
        npz_path,
        times=times,
        dates=dates,
        masses=m,
        pos_horizons=pos_h,
        vel_horizons=vel_h,
        pos_ias15=pos_i,
        vel_ias15=vel_i,
        pos_simon=pos_s,
        vel_simon=vel_s,
        delta_horizons_ias15=delta_hi,
        delta_horizons_simon=delta_hs,
        delta_simon_ias15=delta_si,
        dt_simon=DT_SIMON,
        T_hist=T_hist,
    )

    sanity = {
        "horizons_samples": len(times),
        "first_epoch": dates[0],
        "last_epoch": dates[-1],
        "T_hist_years": f"{T_hist:.6f}",
        "initial_sun_earth_AU": f"{pair_distance(pos_h[:1], 0, 1)[0]:.6f}",
        "initial_earth_moon_AU": f"{pair_distance(pos_h[:1], 1, 2)[0]:.6f}",
        "horizons_earth_moon_range_AU": f"{np.min(pair_distance(pos_h, 1, 2)):.6f} to {np.max(pair_distance(pos_h, 1, 2)):.6f}",
        "ias15_earth_moon_range_AU": f"{np.min(pair_distance(pos_i, 1, 2)):.6f} to {np.max(pair_distance(pos_i, 1, 2)):.6f}",
        "simon_earth_moon_range_AU": f"{np.min(pair_distance(pos_s, 1, 2)):.6f} to {np.max(pair_distance(pos_s, 1, 2)):.6f}",
        "simon_ejected": str(perf_s["ejected"]),
        "simon_total_substeps": perf_s["total_substeps"],
        "simon_avg_close_pair_frac": f"{perf_s['avg_close_pair_frac']:.6f}",
    }

    config = {
        "START_EPOCH": START_EPOCH,
        "STOP_EPOCH": STOP_EPOCH,
        "HORIZONS_STEP": HORIZONS_STEP,
        "DT_SIMON": DT_SIMON,
        "G_REAL": G_REAL,
        "EPS": cfg.eps,
        "EJECTION_THRESHOLD_AU": EJECTION_THRESHOLD_AU,
        "MODEL_PATH": MODEL_PATH,
    }
    runtime_info = {
        "ias15_runtime_sec": f"{perf_i['total_time_sec']:.6f}",
        "SIMON_runtime_sec": f"{perf_s['total_time_sec']:.6f}",
        "SIMON_speed_ratio_vs_ias15": f"{perf_i['total_time_sec'] / max(perf_s['total_time_sec'], 1e-30):.6f}",
    }

    summary_path = os.path.join(OUT_DIR, "summary_historical_1926_2026.txt")
    write_summary(summary_path, config, runtime_info, rows, sanity)

    print("\n[historical_sem] Saved outputs:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print(f"  plots in {OUT_DIR}")
    print("\n[historical_sem] COMPLETE")


if __name__ == "__main__":
    main()
