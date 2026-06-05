"""
real_system_sun_earth_moon_T100_nn_ablation.py

Clean trajectory-level NN ablation for the real Sun-Earth-Moon system using
JPL Horizons initial conditions.

Runs:
    1. ias15 reference
    2. SIMON with NN correction, adaptive sub-stepping ON
    3. No-NN baseline, adaptive sub-stepping ON

The only intended difference between SIMON and No-NN is the close-pair scalar
correction:
    SIMON: c is predicted by the trained NN.
    No-NN: c = 1, so close pairs use pure softened force.

Everything else is identical:
    - same JPL Horizons initial conditions
    - same G = 4*pi^2 real Solar-System units
    - same leapfrog backbone
    - same analytic force direction
    - same adaptive sub-stepping ON
    - same safety fallback
    - same metrics and ias15 reference

Validation check:
    If the previous Appendix-A CSV exists at
        real_system_validation/sun_earth_moon_T100_selected_adaptive_frontier_energy_final/
        T100_adaptive_frontier_results.csv
    this script compares its SIMON-with-NN adaptive_ON results against the newly
    computed SIMON-with-NN adaptive_ON results. This checks that the NN/adaptive
    branch remains consistent with the earlier real-system run.

Outputs:
    trajectory_level_nn_value_out/sun_earth_moon_T100_nn_ablation/
        summary_T100_nn_ablation.txt
        T100_nn_ablation_results.csv
        T100_nn_ablation_results.npz
        timeavg_rms_vs_dt.png / .pdf
        final_rms_vs_dt.png / .pdf
        max_rms_vs_dt.png / .pdf
        energy_drift_vs_dt.png / .pdf
        earth_moon_range_vs_dt.png / .pdf
        close_pair_and_nn_frac_vs_dt.png / .pdf
        selected_dtXXX_rms_timeseries.png / .pdf
        selected_dtXXX_distance_comparison.png / .pdf
        selected_dtXXX_xy_trajectories.png / .pdf

Run:
    python real_system_sun_earth_moon_T100_nn_ablation.py

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


# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
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
    "axes.unicode_minus": False,
})

Horizons.TIMEOUT = 120


# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
DAYS_PER_YEAR = 365.25
G_REAL = 4.0 * np.pi**2
EJECTION_THRESHOLD_AU = 10.0

MODEL_PATH = "pair_correction_nn.pt"
EPOCH = "2026-01-01"
T = 100.0
N_SAMPLES = 5000
DT_VALUES = [0.008, 0.012, 0.016]

BASE_OUT_DIR = "trajectory_level_nn_value_out"
CASE_NAME = "sun_earth_moon_T100_nn_ablation"
OUT_DIR = os.path.join(BASE_OUT_DIR, CASE_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

PREVIOUS_APPENDIX_A_CSV = os.path.join(
    "real_system_validation",
    "sun_earth_moon_T100_selected_adaptive_frontier_energy_final",
    "T100_adaptive_frontier_results.csv",
)

VALIDATION_METRICS = [
    "runtime_sec",
    "total_substeps",
    "avg_close_pair_frac",
    "time_avg_rms",
    "final_rms",
    "max_rms",
    "max_energy_drift_pct",
    "earth_moon_min",
    "earth_moon_max",
    "bounded",
    "lunar_orbit_preserved",
    "ejected",
    "outcome",
]


# -----------------------------------------------------------------------------
# JPL Horizons initial conditions
# -----------------------------------------------------------------------------
def get_horizons_vector(body_id, epoch=EPOCH):
    """Fetch Solar-System-barycentric state vector in AU and AU/year."""
    epochs = {"start": epoch, "stop": "2026-01-02", "step": "1d"}
    print(f"Fetching JPL Horizons vector for body {body_id} ...")
    obj = Horizons(id=body_id, location="@0", epochs=epochs)
    vec = obj.vectors()

    pos = np.array([float(vec["x"][0]), float(vec["y"][0]), float(vec["z"][0])], dtype=np.float64)
    vel_au_day = np.array([float(vec["vx"][0]), float(vec["vy"][0]), float(vec["vz"][0])], dtype=np.float64)
    vel = vel_au_day * DAYS_PER_YEAR
    return pos, vel


def move_to_center_of_mass(x0, v0, m):
    M = np.sum(m)
    x_com = np.sum(m[:, None] * x0, axis=0) / M
    v_com = np.sum(m[:, None] * v0, axis=0) / M
    return x0 - x_com, v0 - v_com


def load_sun_earth_moon_initial_conditions(epoch=EPOCH):
    """JPL Horizons body IDs: Sun=10, Earth=399, Moon=301."""
    sun_x, sun_v = get_horizons_vector("10", epoch)
    earth_x, earth_v = get_horizons_vector("399", epoch)
    moon_x, moon_v = get_horizons_vector("301", epoch)

    x0 = np.vstack([sun_x, earth_x, moon_x])
    v0 = np.vstack([sun_v, earth_v, moon_v])
    m = np.array([
        1.0,                 # Sun
        3.003489614915e-6,   # Earth
        3.694303349e-8,      # Moon
    ], dtype=np.float64)

    return move_to_center_of_mass(x0, v0, m) + (m,)


# -----------------------------------------------------------------------------
# SIMON model
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


def simon_forward_np(nn_in, w):
    h = (nn_in - w["mean"]) / w["std"]
    h = h @ w["w0T"] + w["b0"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
    h = h @ w["w1T"] + w["b1"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
    h = h @ w["w2T"] + w["b2"]; s = 1.0 / (1.0 + np.exp(-h)); h = h * s
    return (h @ w["w3T"] + w["b3"]).ravel()


# -----------------------------------------------------------------------------
# Integrators
# -----------------------------------------------------------------------------
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
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


def simulate_simon_real_system(
    x0,
    v0,
    m,
    model,
    cfg,
    dt,
    T,
    n_samples,
    model_type="simon",
    adaptive_enabled=True,
    ejection_threshold=EJECTION_THRESHOLD_AU,
    early_stop=True,
):
    """
    model_type='simon': c predicted by NN for close pairs.
    model_type='no_nn': c fixed to 1 for close pairs.

    Both modes keep analytic force direction, adaptive stepping, far-field exact
    Newtonian force, and safety fallback. Only c differs.
    """
    if model_type not in {"simon", "no_nn"}:
        raise ValueError("model_type must be 'simon' or 'no_nn'")

    w = extract_weights_numpy(model)
    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii); jj = np.array(jj); P = len(ii)

    G = cfg.G
    eps2 = cfg.eps * cfg.eps
    c_min = cfg.c_min
    c_max = cfg.c_max
    r_soft_min = cfg.r_soft_min
    nn_thresh = 500.0 * cfg.eps
    adapt_thresh = 0.05
    max_substeps = 16

    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]; mj_arr = m_f[jj]
    Gmimj = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr; inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    # Counters are counted per force-evaluation call and per pair.
    close_sum = 0
    nn_applied_sum = 0
    fallback_sum = 0
    pair_sum = 0
    c_sum = 0.0
    c_count = 0

    def compute_acc(pos):
        nonlocal close_sum, nn_applied_sum, fallback_sum, pair_sum, c_sum, c_count
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)

        # Default far-field: exact Newtonian.
        invr3 = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close_mask = r < nn_thresh
        n_close = int(np.sum(close_mask))
        pair_sum += P
        close_sum += n_close

        if n_close > 0:
            r2_c = r2[close_mask]
            r_soft_c = np.sqrt(r2_c + eps2)
            F_soft_c = Gmimj[close_mask] / ((r2_c + eps2) ** 1.5 + 1e-30)

            if model_type == "simon":
                nn_in = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close_mask]
                nn_in[:, 2] = log_mj[close_mask]

                log_c = simon_forward_np(nn_in, w)
                c = np.exp(log_c).astype(np.float64)
                fallback = (
                    (r_soft_c < r_soft_min)
                    | (c < c_min)
                    | (c > c_max)
                    | ~np.isfinite(c)
                )
                accepted = ~fallback
                F_scalar[close_mask] = np.where(fallback, F_scalar[close_mask], c * F_soft_c)

                nn_applied_sum += int(np.sum(accepted))
                fallback_sum += int(np.sum(fallback))
                if np.any(accepted):
                    c_sum += float(np.sum(c[accepted]))
                    c_count += int(np.sum(accepted))

            else:
                # No-NN ablation: pure softened gravity, c = 1. The same
                # extreme-distance fallback is retained for fairness.
                fallback = r_soft_c < r_soft_min
                F_scalar[close_mask] = np.where(fallback, F_scalar[close_mask], F_soft_c)
                fallback_sum += int(np.sum(fallback))
                # nn_applied_sum remains zero by construction.

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

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

    t0 = time.perf_counter()
    ejected = False
    ejection_time = np.nan
    ejection_max_com = np.nan
    steps_done = 0
    completed_time = 0.0
    total_substeps = 0

    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if adaptive_enabled and r_min < adapt_thresh:
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = float(dt) / n_sub
            for _ in range(n_sub):
                x, v, a = leapfrog_substep(x, v, a, sub_dt)
            total_substeps += n_sub
        else:
            vh = v + 0.5 * float(dt) * a
            x = x + float(dt) * vh
            a = compute_acc(x)
            v = vh + 0.5 * float(dt) * a
            total_substeps += 1

        t_cur += float(dt)
        steps_done += 1
        completed_time = t_cur

        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]

        max_com_now = float(np.max(np.linalg.norm(x, axis=1)))
        if early_stop and (not np.isfinite(max_com_now) or max_com_now > ejection_threshold):
            ejected = True
            ejection_time = float(t_cur)
            ejection_max_com = max_com_now
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
        "model_type": model_type,
        "adaptive_enabled": adaptive_enabled,
        "steps": steps_done,
        "planned_steps": n_steps,
        "completed_time": completed_time,
        "ejected": ejected,
        "ejection_time": ejection_time,
        "ejection_max_com": ejection_max_com,
        "ejection_threshold_AU": ejection_threshold,
        "total_time_sec": time.perf_counter() - t0,
        "total_substeps": total_substeps,
        "avg_close_pair_frac": close_sum / max(pair_sum, 1),
        "avg_nn_applied_frac": nn_applied_sum / max(pair_sum, 1),
        "avg_fallback_frac": fallback_sum / max(pair_sum, 1),
        "mean_c_accepted": c_sum / max(c_count, 1) if c_count > 0 else np.nan,
        "accepted_nn_count": c_count,
    }
    return times, pos_out, vel_out, perf


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
    dE = (E - E0) / max(abs(E0), 1e-30)
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


def summarize_rollout(pos_model, vel_model, pos_ref, perf, label, dt, m, G, eps, ref_em_min, ref_em_max):
    delta = rms_sep(pos_model, pos_ref)
    sun_earth = pair_distance(pos_model, 0, 1)
    earth_moon = pair_distance(pos_model, 1, 2)
    max_com = max_distance_from_com(pos_model)
    finite = bool(np.all(np.isfinite(pos_model)) and np.all(np.isfinite(vel_model)))
    bounded = bool(finite and np.max(max_com) < EJECTION_THRESHOLD_AU)
    energy = energy_drift_metrics(pos_model, vel_model, m, G=G, eps=eps)
    lunar_orbit_preserved, outcome = classify_lunar_outcome(
        earth_moon_min=float(np.min(earth_moon)),
        earth_moon_max=float(np.max(earth_moon)),
        ref_em_min=ref_em_min,
        ref_em_max=ref_em_max,
        bounded=bounded,
        ejected=bool(perf.get("ejected", False)),
    )
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
        "ejection_threshold_AU": EJECTION_THRESHOLD_AU,
        "total_substeps": int(perf["total_substeps"]),
        "avg_close_pair_frac": float(perf["avg_close_pair_frac"]),
        "avg_nn_applied_frac": float(perf["avg_nn_applied_frac"]),
        "avg_fallback_frac": float(perf["avg_fallback_frac"]),
        "mean_c_accepted": float(perf["mean_c_accepted"]),
        "accepted_nn_count": int(perf["accepted_nn_count"]),
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
        "lunar_orbit_preserved": lunar_orbit_preserved,
        "outcome": outcome,
        "finite": finite,
        "delta": delta,
        "sun_earth": sun_earth,
        "earth_moon": earth_moon,
        "max_com_series": max_com,
        "energy_drift_pct_series": energy["energy_drift_pct_series"],
    }


# -----------------------------------------------------------------------------
# Validation against previous Appendix A CSV
# -----------------------------------------------------------------------------
def _parse_bool(s):
    return str(s).strip().lower() in {"true", "1", "yes"}


def load_previous_adaptive_on_rows(path):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode", "")
            if mode == "adaptive_ON":
                dt = round(float(row["dt"]), 10)
                rows[dt] = row
    return rows


def compare_to_previous(rows_simon, previous_path):
    previous = load_previous_adaptive_on_rows(previous_path)
    validation_rows = []
    if previous is None:
        return validation_rows, False, f"Previous Appendix-A CSV not found: {previous_path}"

    all_pass = True
    for row in rows_simon:
        dt_key = round(float(row["dt"]), 10)
        prev = previous.get(dt_key)
        if prev is None:
            validation_rows.append({
                "dt": row["dt"], "metric": "ROW", "previous": "missing", "current": "present",
                "abs_diff": np.nan, "rel_diff": np.nan, "pass": False,
            })
            all_pass = False
            continue

        for metric in VALIDATION_METRICS:
            if metric not in prev or metric not in row:
                continue
            cur_val = row[metric]
            prev_str = prev[metric]
            if isinstance(cur_val, (bool, np.bool_)):
                cur = bool(cur_val)
                old = _parse_bool(prev_str)
                passed = (cur == old)
                abs_diff = 0.0 if passed else 1.0
                rel_diff = 0.0 if passed else 1.0
                prev_out = str(old)
                cur_out = str(cur)
            elif isinstance(cur_val, str):
                old = str(prev_str)
                cur = str(cur_val)
                passed = (cur == old)
                abs_diff = 0.0 if passed else 1.0
                rel_diff = 0.0 if passed else 1.0
                prev_out = old
                cur_out = cur
            else:
                old = float(prev_str)
                cur = float(cur_val)
                abs_diff = abs(cur - old)
                rel_diff = abs_diff / max(abs(old), 1e-30)
                # Runtime can vary; compare but do not fail on runtime.
                if metric == "runtime_sec":
                    passed = True
                elif metric in {"total_substeps"}:
                    passed = (abs_diff == 0)
                else:
                    passed = (abs_diff <= 1e-10 or rel_diff <= 1e-8)
                prev_out = f"{old:.12e}"
                cur_out = f"{cur:.12e}"

            if not passed:
                all_pass = False
            validation_rows.append({
                "dt": row["dt"], "metric": metric, "previous": prev_out, "current": cur_out,
                "abs_diff": abs_diff, "rel_diff": rel_diff, "pass": bool(passed),
            })
    return validation_rows, all_pass, f"Compared against previous Appendix-A CSV: {previous_path}"


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def savefig_both(fig, base_path_no_ext):
    fig.tight_layout()
    fig.savefig(base_path_no_ext + ".png", dpi=300)
    fig.savefig(base_path_no_ext + ".pdf")
    plt.close(fig)


def plot_metric_vs_dt(rows_simon, rows_nonn, metric, ylabel, out_base, logy=True):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    y_s = np.array([r[metric] for r in rows_simon], dtype=float)
    y_n = np.array([r[metric] for r in rows_nonn], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dts, np.maximum(y_s, 1e-300), "o-", lw=1.9, label="SIMON with NN")
    ax.plot(dts, np.maximum(y_n, 1e-300), "s--", lw=1.9, label="No-NN baseline")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{dt:.3f}" for dt in dts])
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


def plot_energy_drift_vs_dt(rows_simon, rows_nonn, ref_energy, out_base):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    y_s = np.array([r["max_energy_drift_pct"] for r in rows_simon], dtype=float)
    y_n = np.array([r["max_energy_drift_pct"] for r in rows_nonn], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dts, np.maximum(y_s, 1e-300), "o-", lw=1.9, label="SIMON with NN")
    ax.plot(dts, np.maximum(y_n, 1e-300), "s--", lw=1.9, label="No-NN baseline")
    ax.axhline(max(ref_energy["max_energy_drift_pct"], 1e-300), lw=1.2, linestyle=":", label="ias15 reference")
    ax.set_yscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{dt:.3f}" for dt in dts])
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Max energy drift |ΔE/E₀| (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


def plot_close_nn_frac(rows_simon, rows_nonn, out_base):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    close_s = np.array([r["avg_close_pair_frac"] for r in rows_simon], dtype=float)
    nn_s = np.array([r["avg_nn_applied_frac"] for r in rows_simon], dtype=float)
    fallback_s = np.array([r["avg_fallback_frac"] for r in rows_simon], dtype=float)
    close_n = np.array([r["avg_close_pair_frac"] for r in rows_nonn], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dts, close_s, "o-", lw=1.8, label="Close-pair fraction (SIMON)")
    ax.plot(dts, nn_s, "s-", lw=1.8, label="NN applied fraction (SIMON)")
    ax.plot(dts, fallback_s, "^-", lw=1.8, label="Fallback fraction (SIMON)")
    ax.plot(dts, close_n, "x--", lw=1.6, label="Close-pair fraction (No-NN)")
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{dt:.3f}" for dt in dts])
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Fraction of pair evaluations")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


def plot_earth_moon_range_vs_dt(ref_earth_moon, rows_simon, rows_nonn, out_base):
    dts = np.array([r["dt"] for r in rows_simon], dtype=float)
    s_min = np.array([r["earth_moon_min"] for r in rows_simon], dtype=float)
    s_max = np.array([r["earth_moon_max"] for r in rows_simon], dtype=float)
    n_min = np.array([r["earth_moon_min"] for r in rows_nonn], dtype=float)
    n_max = np.array([r["earth_moon_max"] for r in rows_nonn], dtype=float)
    ref_min = float(np.min(ref_earth_moon))
    ref_max = float(np.max(ref_earth_moon))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(dts, s_min, s_max, alpha=0.18, label="SIMON with NN range")
    ax.plot(dts, s_min, "o-", lw=1.2)
    ax.plot(dts, s_max, "o-", lw=1.2)
    ax.fill_between(dts, n_min, n_max, alpha=0.18, label="No-NN range")
    ax.plot(dts, n_min, "s--", lw=1.2)
    ax.plot(dts, n_max, "s--", lw=1.2)
    ax.axhline(ref_min, linestyle=":", lw=1.2, label="ias15 min/max")
    ax.axhline(ref_max, linestyle=":", lw=1.2)
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{dt:.3f}" for dt in dts])
    ax.set_xlabel("Timestep dt (yr)")
    ax.set_ylabel("Earth-Moon distance range (AU)")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


def plot_rms_timeseries(times, row_s, row_n, out_base):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(times, np.maximum(row_s["delta"], 1e-300), lw=1.8, label="SIMON with NN")
    ax.semilogy(times, np.maximum(row_n["delta"], 1e-300), lw=1.8, linestyle="--", label="No-NN baseline")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


def plot_pair_distances_three(times, ref_se, sim_se, nonn_se, ref_em, sim_em, nonn_em, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(times, ref_se, lw=1.6, label="ias15")
    axes[0].plot(times, sim_se, "--", lw=1.4, label="SIMON with NN")
    axes[0].plot(times, nonn_se, ":", lw=1.5, label="No-NN")
    axes[0].set_xlabel("Time (yr)")
    axes[0].set_ylabel("Sun-Earth distance (AU)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.88, loc="best")
    axes[1].plot(times, ref_em, lw=1.6, label="ias15")
    axes[1].plot(times, sim_em, "--", lw=1.4, label="SIMON with NN")
    axes[1].plot(times, nonn_em, ":", lw=1.5, label="No-NN")
    axes[1].set_xlabel("Time (yr)")
    axes[1].set_ylabel("Earth-Moon distance (AU)")
    axes[1].grid(True, alpha=0.25)
    savefig_both(fig, out_base)


def plot_xy_trajectories_three(pos_ref, pos_simon, pos_nonn, out_base):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    names = ["Sun", "Earth", "Moon relative to Earth"]
    # Sun barycentric
    datasets = [
        (pos_ref[:, 0, :], pos_simon[:, 0, :], pos_nonn[:, 0, :]),
        (pos_ref[:, 1, :], pos_simon[:, 1, :], pos_nonn[:, 1, :]),
        (pos_ref[:, 2, :] - pos_ref[:, 1, :], pos_simon[:, 2, :] - pos_simon[:, 1, :], pos_nonn[:, 2, :] - pos_nonn[:, 1, :]),
    ]
    for ax, name, data in zip(axes, names, datasets):
        ref, sim, nonn = data
        ax.plot(ref[:, 0], ref[:, 1], lw=1.5, label="ias15")
        ax.plot(sim[:, 0], sim[:, 1], "--", lw=1.2, label="SIMON with NN")
        ax.plot(nonn[:, 0], nonn[:, 1], ":", lw=1.4, label="No-NN")
        ax.set_title(name)
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.88, loc="best")
    savefig_both(fig, out_base)


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------
def write_results_csv(rows_simon, rows_nonn, ias15_time, out_path):
    fieldnames = [
        "mode", "dt", "runtime_sec", "speed_ratio_ias15_over_model",
        "steps", "planned_steps", "completed_time", "ejected", "ejection_time",
        "ejection_max_com", "ejection_threshold_AU", "total_substeps",
        "avg_close_pair_frac", "avg_nn_applied_frac", "avg_fallback_frac",
        "mean_c_accepted", "accepted_nn_count", "time_avg_rms", "final_rms",
        "max_rms", "max_energy_drift_pct", "final_energy_drift_pct",
        "rms_energy_drift_pct", "sun_earth_min", "sun_earth_max",
        "earth_moon_min", "earth_moon_max", "max_com", "energy_finite",
        "bounded", "lunar_orbit_preserved", "outcome", "finite",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, rows in [("SIMON_with_NN", rows_simon), ("No_NN_baseline", rows_nonn)]:
            for r in rows:
                out = {k: r[k] for k in fieldnames if k in r}
                out["mode"] = mode
                out["speed_ratio_ias15_over_model"] = ias15_time / max(r["runtime_sec"], 1e-30)
                writer.writerow(out)


def write_validation_csv(validation_rows, out_path):
    fieldnames = ["dt", "metric", "previous", "current", "abs_diff", "rel_diff", "pass"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in validation_rows:
            writer.writerow(row)


def write_summary(out_path, validation_message, validation_pass, rows_validation, T, n_samples, dt_values, epoch, x0, v0, m, perf_ref, ref_sun_earth, ref_earth_moon, ref_max_com, ref_energy, rows_simon, rows_nonn):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Real Sun-Earth-Moon T=100 trajectory-level NN ablation\n")
        f.write("=" * 78 + "\n\n")
        f.write("Purpose:\n")
        f.write("  Compare ias15, SIMON with NN correction, and a No-NN baseline on real\n")
        f.write("  Sun-Earth-Moon JPL Horizons initial conditions. Both SIMON variants keep\n")
        f.write("  adaptive sub-stepping ON and analytic force direction; the only intended\n")
        f.write("  difference is whether c is predicted by the NN or fixed at c=1.\n\n")
        f.write(f"Epoch                   = {epoch}\n")
        f.write(f"T                       = {T:.2f} yr\n")
        f.write(f"n_samples               = {n_samples}\n")
        f.write(f"dt_values               = {', '.join(f'{d:.4f}' for d in dt_values)} yr\n")
        f.write(f"G_REAL                  = {G_REAL:.8f}\n")
        f.write(f"eps                     = {HybridConfig().eps:.8e} AU\n")
        f.write(f"NN threshold            = {500.0 * HybridConfig().eps:.8e} AU\n")
        f.write(f"adaptive threshold      = 5.00000000e-02 AU\n")
        f.write(f"r_soft_min              = {HybridConfig().r_soft_min:.8e} AU\n")
        f.write(f"ejection_threshold      = {EJECTION_THRESHOLD_AU:.2f} AU from COM\n")
        f.write(f"ias15 runtime            = {perf_ref['total_time_sec']:.6f} s\n\n")

        f.write("Validation against previous Appendix-A adaptive-ON run:\n")
        f.write(f"  {validation_message}\n")
        f.write(f"  Overall validation pass: {validation_pass}\n")
        if rows_validation:
            failed = [r for r in rows_validation if not r["pass"]]
            f.write(f"  Validation checks: {len(rows_validation)} total, {len(failed)} failed.\n")
            if failed:
                f.write("  Failed checks:\n")
                for r in failed[:20]:
                    f.write(f"    dt={r['dt']} metric={r['metric']} previous={r['previous']} current={r['current']}\n")
        f.write("\n")

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
        header = (
            f"{'mode':<16} {'dt':>8} {'runtime(s)':>12} {'speed':>9} "
            f"{'close':>9} {'NN':>9} {'fallback':>9} {'c_mean':>9} "
            f"{'RMS0T(AU)':>13} {'Emax(%)':>12} {'final(AU)':>12} {'max(AU)':>12} "
            f"{'EM_min':>10} {'EM_max':>10} {'bounded':>8} {'lunar':>8} {'outcome':<18}\n"
        )
        f.write(header)
        f.write("-" * 194 + "\n")
        for mode, rows in [("SIMON_with_NN", rows_simon), ("No_NN_baseline", rows_nonn)]:
            for r in rows:
                speed = perf_ref["total_time_sec"] / max(r["runtime_sec"], 1e-30)
                c_mean = r["mean_c_accepted"] if np.isfinite(r["mean_c_accepted"]) else np.nan
                f.write(
                    f"{mode:<16} {r['dt']:>8.4f} {r['runtime_sec']:>12.6f} {speed:>9.3f} "
                    f"{r['avg_close_pair_frac']:>9.4f} {r['avg_nn_applied_frac']:>9.4f} "
                    f"{r['avg_fallback_frac']:>9.4f} {c_mean:>9.4f} "
                    f"{r['time_avg_rms']:>13.6e} {r['max_energy_drift_pct']:>12.6e} "
                    f"{r['final_rms']:>12.6e} {r['max_rms']:>12.6e} "
                    f"{r['earth_moon_min']:>10.6f} {r['earth_moon_max']:>10.6f} "
                    f"{str(r['bounded']):>8} {str(r['lunar_orbit_preserved']):>8} {r['outcome']:<18}\n"
                )
            f.write("\n")

        f.write("Direct NN-vs-NoNN comparison by timestep:\n")
        f.write(f"{'dt':>8} {'RMS ratio NoNN/SIMON':>24} {'RMS reduction':>16} {'Final ratio':>14} {'Final reduction':>18}\n")
        f.write("-" * 92 + "\n")
        for rs, rn in zip(rows_simon, rows_nonn):
            rms_ratio = rn["time_avg_rms"] / max(rs["time_avg_rms"], 1e-30)
            rms_reduction = 100.0 * (1.0 - rs["time_avg_rms"] / max(rn["time_avg_rms"], 1e-30))
            final_ratio = rn["final_rms"] / max(rs["final_rms"], 1e-30)
            final_reduction = 100.0 * (1.0 - rs["final_rms"] / max(rn["final_rms"], 1e-30))
            f.write(f"{rs['dt']:>8.4f} {rms_ratio:>24.6f} {rms_reduction:>15.3f}% {final_ratio:>14.6f} {final_reduction:>17.3f}%\n")

        f.write("\nInterpretation guide:\n")
        f.write("- RMS0T is sqrt(mean_t delta(t)^2), where delta(t) is RMS position error vs ias15.\n")
        f.write("- close is the fraction of pair evaluations with r < 0.15 AU.\n")
        f.write("- NN is the fraction of pair evaluations where the NN correction was accepted and applied.\n")
        f.write("- fallback is the fraction of pair evaluations using exact Newtonian fallback inside the close region.\n")
        f.write("- Both SIMON and No-NN retain adaptive stepping and analytic force direction.\n")
        f.write("- If SIMON is not better than No-NN here, the real Sun-Earth-Moon system should remain a real-data validation rather than an NN-value claim.\n")


def save_npz(out_path, times, x0, v0, m, pos_ref, vel_ref, rows_simon, rows_nonn, pos_simon_all, vel_simon_all, pos_nonn_all, vel_nonn_all, dt_values, ref_energy):
    np.savez_compressed(
        out_path,
        times=times,
        x0=x0,
        v0=v0,
        masses=m,
        pos_ias15=pos_ref,
        vel_ias15=vel_ref,
        pos_simon_nn=pos_simon_all,
        vel_simon_nn=vel_simon_all,
        pos_no_nn=pos_nonn_all,
        vel_no_nn=vel_nonn_all,
        dt_values=np.array(dt_values, dtype=np.float64),
        T=T,
        G_REAL=G_REAL,
        time_avg_rms_simon=np.array([r["time_avg_rms"] for r in rows_simon]),
        time_avg_rms_no_nn=np.array([r["time_avg_rms"] for r in rows_nonn]),
        final_rms_simon=np.array([r["final_rms"] for r in rows_simon]),
        final_rms_no_nn=np.array([r["final_rms"] for r in rows_nonn]),
        max_rms_simon=np.array([r["max_rms"] for r in rows_simon]),
        max_rms_no_nn=np.array([r["max_rms"] for r in rows_nonn]),
        max_energy_drift_pct_ref=ref_energy["max_energy_drift_pct"],
        energy_drift_pct_ref=ref_energy["energy_drift_pct_series"],
        max_energy_drift_pct_simon=np.array([r["max_energy_drift_pct"] for r in rows_simon]),
        max_energy_drift_pct_no_nn=np.array([r["max_energy_drift_pct"] for r in rows_nonn]),
        energy_drift_pct_simon=np.stack([r["energy_drift_pct_series"] for r in rows_simon], axis=0),
        energy_drift_pct_no_nn=np.stack([r["energy_drift_pct_series"] for r in rows_nonn], axis=0),
        close_pair_frac_simon=np.array([r["avg_close_pair_frac"] for r in rows_simon]),
        nn_applied_frac_simon=np.array([r["avg_nn_applied_frac"] for r in rows_simon]),
        fallback_frac_simon=np.array([r["avg_fallback_frac"] for r in rows_simon]),
        close_pair_frac_no_nn=np.array([r["avg_close_pair_frac"] for r in rows_nonn]),
        fallback_frac_no_nn=np.array([r["avg_fallback_frac"] for r in rows_nonn]),
        bounded_simon=np.array([r["bounded"] for r in rows_simon], dtype=bool),
        bounded_no_nn=np.array([r["bounded"] for r in rows_nonn], dtype=bool),
        outcome_simon=np.array([r["outcome"] for r in rows_simon]),
        outcome_no_nn=np.array([r["outcome"] for r in rows_nonn]),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n[Sun-Earth-Moon NN ablation] Loading JPL Horizons initial conditions.")
    x0, v0, m = load_sun_earth_moon_initial_conditions(epoch=EPOCH)
    print("\nSanity checks:")
    print(f"  Sun-Earth distance  = {np.linalg.norm(x0[1] - x0[0]):.6f} AU")
    print(f"  Earth-Moon distance = {np.linalg.norm(x0[2] - x0[1]):.6f} AU")
    print(f"  G_REAL              = {G_REAL:.8f}")

    print("\n[Sun-Earth-Moon NN ablation] Loading SIMON model.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find {MODEL_PATH}. Put this script in the same folder as pair_correction_nn.pt.")
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    cfg = HybridConfig(G=G_REAL)
    print(f"  Loaded {MODEL_PATH} with {sum(p.numel() for p in model.parameters())} parameters")

    print("\n[Sun-Earth-Moon NN ablation] Running ias15 reference once.")
    t_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(x0, v0, m, cfg.G, T, N_SAMPLES)
    print(f"  ias15 runtime = {perf_ref['total_time_sec']:.6f} s")

    ref_sun_earth = pair_distance(pos_ref, 0, 1)
    ref_earth_moon = pair_distance(pos_ref, 1, 2)
    ref_em_min = float(np.min(ref_earth_moon))
    ref_em_max = float(np.max(ref_earth_moon))
    ref_max_com = max_distance_from_com(pos_ref)
    ref_energy = energy_drift_metrics(pos_ref, vel_ref, m, G=cfg.G, eps=cfg.eps)
    print(f"  ias15 Earth-Moon range = {ref_em_min:.6f} to {ref_em_max:.6f} AU")
    print(f"  ias15 max energy drift = {ref_energy['max_energy_drift_pct']:.6e}%")

    rows_simon, rows_nonn = [], []
    pos_simon_all, vel_simon_all = [], []
    pos_nonn_all, vel_nonn_all = [], []

    print("\n[Sun-Earth-Moon NN ablation] Running T=100 trajectory-level NN ablation.")
    print(f"  Output folder: {OUT_DIR}")
    print(f"  T={T:.2f} yr | n_samples={N_SAMPLES} | dt_values={DT_VALUES}")
    print("  " + "-" * 170)
    print(
        f"  {'dt':>8} | {'mode':<15} | {'runtime(s)':>10} | {'speed':>7} | "
        f"{'close':>7} | {'NN':>7} | {'fallback':>8} | {'RMS0T(AU)':>12} | "
        f"{'Emax(%)':>10} | {'EM range (AU)':>25} | {'outcome':<18}"
    )
    print("  " + "-" * 170)

    for dt in DT_VALUES:
        # SIMON with NN, adaptive ON.
        _, pos_s, vel_s, perf_s = simulate_simon_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES,
            model_type="simon", adaptive_enabled=True,
        )
        row_s = summarize_rollout(pos_s, vel_s, pos_ref, perf_s, "SIMON_with_NN", dt, m, cfg.G, cfg.eps, ref_em_min, ref_em_max)
        rows_simon.append(row_s)
        pos_simon_all.append(pos_s)
        vel_simon_all.append(vel_s)
        speed_s = perf_ref["total_time_sec"] / max(row_s["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'SIMON with NN':<15} | {row_s['runtime_sec']:>10.4f} | {speed_s:>7.3f} | "
            f"{row_s['avg_close_pair_frac']:>7.4f} | {row_s['avg_nn_applied_frac']:>7.4f} | "
            f"{row_s['avg_fallback_frac']:>8.4f} | {row_s['time_avg_rms']:>12.5e} | "
            f"{row_s['max_energy_drift_pct']:>10.3e} | "
            f"{row_s['earth_moon_min']:.6f} to {row_s['earth_moon_max']:.6f} | {row_s['outcome']:<18}"
        )

        # No-NN baseline, adaptive ON.
        _, pos_n, vel_n, perf_n = simulate_simon_real_system(
            x0, v0, m, model, cfg, dt, T, N_SAMPLES,
            model_type="no_nn", adaptive_enabled=True,
        )
        row_n = summarize_rollout(pos_n, vel_n, pos_ref, perf_n, "No_NN_baseline", dt, m, cfg.G, cfg.eps, ref_em_min, ref_em_max)
        rows_nonn.append(row_n)
        pos_nonn_all.append(pos_n)
        vel_nonn_all.append(vel_n)
        speed_n = perf_ref["total_time_sec"] / max(row_n["runtime_sec"], 1e-30)
        print(
            f"  {dt:8.4f} | {'No-NN baseline':<15} | {row_n['runtime_sec']:>10.4f} | {speed_n:>7.3f} | "
            f"{row_n['avg_close_pair_frac']:>7.4f} | {row_n['avg_nn_applied_frac']:>7.4f} | "
            f"{row_n['avg_fallback_frac']:>8.4f} | {row_n['time_avg_rms']:>12.5e} | "
            f"{row_n['max_energy_drift_pct']:>10.3e} | "
            f"{row_n['earth_moon_min']:.6f} to {row_n['earth_moon_max']:.6f} | {row_n['outcome']:<18}"
        )

    print("  " + "-" * 170)

    pos_simon_all = np.stack(pos_simon_all, axis=0)
    vel_simon_all = np.stack(vel_simon_all, axis=0)
    pos_nonn_all = np.stack(pos_nonn_all, axis=0)
    vel_nonn_all = np.stack(vel_nonn_all, axis=0)

    # Validation against previous Appendix A adaptive-ON run.
    validation_rows, validation_pass, validation_message = compare_to_previous(rows_simon, PREVIOUS_APPENDIX_A_CSV)
    print("\n[validation check]")
    print(f"  {validation_message}")
    print(f"  Overall validation pass: {validation_pass}")
    if validation_rows:
        failed = [r for r in validation_rows if not r["pass"]]
        print(f"  Validation checks: {len(validation_rows)} total, {len(failed)} failed")
        if failed:
            for r in failed[:10]:
                print(f"    FAIL dt={r['dt']} metric={r['metric']} previous={r['previous']} current={r['current']}")

    # Save numeric outputs.
    summary_path = os.path.join(OUT_DIR, "summary_T100_nn_ablation.txt")
    csv_path = os.path.join(OUT_DIR, "T100_nn_ablation_results.csv")
    validation_csv_path = os.path.join(OUT_DIR, "validation_against_previous_appendix_A.csv")
    npz_path = os.path.join(OUT_DIR, "T100_nn_ablation_results.npz")

    write_results_csv(rows_simon, rows_nonn, perf_ref["total_time_sec"], csv_path)
    write_validation_csv(validation_rows, validation_csv_path)
    write_summary(
        summary_path,
        validation_message,
        validation_pass,
        validation_rows,
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
        rows_nonn,
    )
    save_npz(
        npz_path,
        t_ref,
        x0,
        v0,
        m,
        pos_ref,
        vel_ref,
        rows_simon,
        rows_nonn,
        pos_simon_all,
        vel_simon_all,
        pos_nonn_all,
        vel_nonn_all,
        DT_VALUES,
        ref_energy,
    )

    # Save paper-ready charts.
    plot_metric_vs_dt(rows_simon, rows_nonn, "time_avg_rms", "Time-averaged RMS vs ias15 (AU)", os.path.join(OUT_DIR, "timeavg_rms_vs_dt"), logy=True)
    plot_metric_vs_dt(rows_simon, rows_nonn, "final_rms", "Final RMS vs ias15 (AU)", os.path.join(OUT_DIR, "final_rms_vs_dt"), logy=True)
    plot_metric_vs_dt(rows_simon, rows_nonn, "max_rms", "Max RMS vs ias15 (AU)", os.path.join(OUT_DIR, "max_rms_vs_dt"), logy=True)
    plot_energy_drift_vs_dt(rows_simon, rows_nonn, ref_energy, os.path.join(OUT_DIR, "energy_drift_vs_dt"))
    plot_earth_moon_range_vs_dt(ref_earth_moon, rows_simon, rows_nonn, os.path.join(OUT_DIR, "earth_moon_range_vs_dt"))
    plot_close_nn_frac(rows_simon, rows_nonn, os.path.join(OUT_DIR, "close_pair_and_nn_frac_vs_dt"))

    # Detailed plots for the largest selected dt, usually the most visually informative.
    selected_idx = len(DT_VALUES) - 1
    selected_dt = DT_VALUES[selected_idx]
    tag = f"selected_dt{str(selected_dt).replace('.', 'p')}"
    plot_rms_timeseries(t_ref, rows_simon[selected_idx], rows_nonn[selected_idx], os.path.join(OUT_DIR, f"{tag}_rms_timeseries"))
    plot_pair_distances_three(
        t_ref,
        ref_sun_earth,
        rows_simon[selected_idx]["sun_earth"],
        rows_nonn[selected_idx]["sun_earth"],
        ref_earth_moon,
        rows_simon[selected_idx]["earth_moon"],
        rows_nonn[selected_idx]["earth_moon"],
        os.path.join(OUT_DIR, f"{tag}_distance_comparison"),
    )
    plot_xy_trajectories_three(
        pos_ref,
        pos_simon_all[selected_idx],
        pos_nonn_all[selected_idx],
        os.path.join(OUT_DIR, f"{tag}_xy_trajectories"),
    )

    print("\n[Sun-Earth-Moon NN ablation] Saved outputs:")
    print(f"  Summary: {summary_path}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Validation CSV: {validation_csv_path}")
    print(f"  NPZ: {npz_path}")
    print(f"  Figures saved in: {OUT_DIR}")

    print("\n[Sun-Earth-Moon NN ablation] Complete. Paste the printed table and summary back into ChatGPT for interpretation.")


if __name__ == "__main__":
    main()
