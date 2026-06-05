"""
compact_eccentric_triple_nn_value.py

Trajectory-level NN-value experiment for SIMON.

Purpose
-------
Test whether SIMON's scalar NN correction improves trajectory accuracy in a
three-body rollout where the NN is actively engaged many times.

Compares:
  1. ias15 reference
  2. Full SIMON with NN correction
  3. No-NN baseline with the same analytic direction and adaptive sub-stepping

Key design
----------
The compact eccentric hierarchical triple is deliberately chosen so that the
inner binary reaches periastron below SIMON's NN activation threshold:

    r_peri = a_in(1 - e_in) = 1.0 * (1 - 0.92) = 0.08 AU < 0.15 AU.

This makes the NN correction active at repeated periastron passages.
Adaptive sub-stepping and analytic force direction are kept ON in both SIMON
and No-NN. The only difference is whether the close-pair correction factor c is
predicted by the NN or fixed to c = 1.

Default run
-----------
    python compact_eccentric_triple_nn_value.py

Useful alternatives
-------------------
    python compact_eccentric_triple_nn_value.py --dt 0.005 --T 20
    python compact_eccentric_triple_nn_value.py --dt 0.02 --T 20
    python compact_eccentric_triple_nn_value.py --T 50 --n_samples 5000

Required file in the same folder
--------------------------------
    pair_correction_nn.pt

Outputs
-------
    trajectory_level_nn_value_out/compact_eccentric_triple_nn_value/
        summary_compact_eccentric_triple.txt
        compact_eccentric_triple_metrics.csv
        compact_eccentric_triple_timeseries.npz
        rms_error_early_0_10yr.png
        rms_error_full.png
        inner_separation.png
        energy_drift.png
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
# Global configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "pair_correction_nn.pt"
OUT_DIR = os.path.join(
    "trajectory_level_nn_value_out",
    "compact_eccentric_triple_nn_value",
)

G = 1.0
EPS = 3e-4
NN_THRESHOLD = 500.0 * EPS       # 0.15 AU
ADAPT_THRESH = 0.05
MAX_SUBSTEPS = 16
R_SOFT_MIN = 5e-4
C_MIN = 0.2
C_MAX = 5.0
EJECTION_AU = 50.0

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
# SIMON model
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


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run this script from the folder containing "
            "pair_correction_nn.pt, or update MODEL_PATH."
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
# Orbital-element initial conditions
# -----------------------------------------------------------------------------
def rotation_matrix(inc, Omega, omega):
    """Return Rz(Omega) Rx(inc) Rz(omega). Angles in radians."""
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(omega), np.sin(omega)

    RzO = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    Rxi = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    Rzo = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
    return RzO @ Rxi @ Rzo


def kepler_relative_state(a, e, inc, Omega, omega, f, mu):
    """
    Convert Keplerian orbital elements into relative Cartesian state.

    Returns r, v for the secondary relative to the primary/barycentre being orbited.
    Units are consistent with G=1, AU, solar masses, and years.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("Only bound elliptical orbits with 0 <= e < 1 are supported.")
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


def make_compact_eccentric_triple():
    """
    Construct a compact eccentric hierarchical triple using Jacobi coordinates.

    Body 0 + body 1 form the inner eccentric binary.
    Body 2 orbits the inner-binary centre of mass.

    The inner periastron is deliberately below 0.15 AU so that the NN correction
    is repeatedly activated.
    """
    # In-distribution stellar-scale masses.
    m0 = 1.0
    m1 = 0.5
    m2 = 0.2
    m = np.array([m0, m1, m2], dtype=np.float64)

    # Inner binary: high eccentricity, repeated close passages.
    a_in = 1.0
    e_in = 0.92
    inc_in = np.deg2rad(0.0)
    Omega_in = np.deg2rad(0.0)
    omega_in = np.deg2rad(0.0)
    f_in = np.deg2rad(0.0)      # start at periastron

    # Outer companion: wider, moderately inclined perturbing orbit.
    # a_out is deliberately wide enough to avoid immediate instability while still
    # making the configuration a genuine three-body hierarchy.
    a_out = 10.0
    e_out = 0.20
    inc_out = np.deg2rad(55.0)
    Omega_out = np.deg2rad(40.0)
    omega_out = np.deg2rad(90.0)
    f_out = np.deg2rad(180.0)   # start near outer apastron

    M_inner = m0 + m1
    M_total = M_inner + m2

    # Inner relative state: r_in = x1 - x0.
    r_in, v_in = kepler_relative_state(
        a=a_in,
        e=e_in,
        inc=inc_in,
        Omega=Omega_in,
        omega=omega_in,
        f=f_in,
        mu=G * M_inner,
    )

    # Outer relative state: r_out = x2 - x_inner_COM.
    r_out, v_out = kepler_relative_state(
        a=a_out,
        e=e_out,
        inc=inc_out,
        Omega=Omega_out,
        omega=omega_out,
        f=f_out,
        mu=G * M_total,
    )

    x_inner_com = -(m2 / M_total) * r_out
    v_inner_com = -(m2 / M_total) * v_out
    x2 = (M_inner / M_total) * r_out
    v2 = (M_inner / M_total) * v_out

    x0 = x_inner_com - (m1 / M_inner) * r_in
    x1 = x_inner_com + (m0 / M_inner) * r_in
    v0 = v_inner_com - (m1 / M_inner) * v_in
    v1 = v_inner_com + (m0 / M_inner) * v_in

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
        "period_in": 2.0 * np.pi * np.sqrt(a_in**3 / (G * M_inner)),
        "a_out": a_out,
        "e_out": e_out,
        "q_out": a_out * (1.0 - e_out),
        "Q_out": a_out * (1.0 + e_out),
        "period_out": 2.0 * np.pi * np.sqrt(a_out**3 / (G * M_total)),
        "mutual_inclination_deg": 55.0,
    }
    return x, v, m, params


# -----------------------------------------------------------------------------
# Integrators
# -----------------------------------------------------------------------------
def simulate_ias15(x0, v0, m, cfg, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = cfg.G
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
    perf = {"total_time_sec": time.perf_counter() - t0}
    return times, pos, vel, perf


def simulate_hybrid(x0, v0, m, model, cfg, dt, T, n_samples, model_type="simon"):
    """
    model_type='simon': NN predicts correction c for r < 0.15 AU.
    model_type='no_nn': c is fixed to 1, i.e. pure softened force in close regime.

    Both variants keep analytic force direction and adaptive sub-stepping.
    """
    if model_type not in {"simon", "no_nn"}:
        raise ValueError("model_type must be 'simon' or 'no_nn'")

    w = extract_weights_numpy(model)

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i)
            jj.append(j)
    ii = np.array(ii, dtype=int)
    jj = np.array(jj, dtype=int)
    P = len(ii)

    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)

    mi = m_f[ii]
    mj = m_f[jj]
    Gmimj = cfg.G * mi * mj
    inv_mi = 1.0 / mi
    inv_mj = 1.0 / mj
    log_mi = np.log(mi + 1e-30).astype(np.float32)
    log_mj = np.log(mj + 1e-30).astype(np.float32)

    eps2 = cfg.eps * cfg.eps
    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))

    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    close_pair_count = 0
    fallback_count = 0
    pair_eval_count = 0
    total_substeps = 0
    min_r_seen = np.inf
    ejected = False
    ejection_time = np.nan

    def compute_acc(pos):
        nonlocal close_pair_count, fallback_count, pair_eval_count, min_r_seen

        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        min_r_seen = min(min_r_seen, float(np.min(r)))

        invr3 = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close = r < cfg.nn_threshold
        n_close = int(np.sum(close))
        close_pair_count += n_close
        pair_eval_count += P

        if n_close > 0:
            r2_c = r2[close]
            r_soft_c = np.sqrt(r2_c + eps2)
            F_soft_c = Gmimj[close] / ((r2_c + eps2) ** 1.5 + 1e-30)

            if model_type == "simon":
                nn_in = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close]
                nn_in[:, 2] = log_mj[close]

                log_c = simon_forward_numpy(nn_in, w)
                c = np.exp(log_c).astype(np.float64)
                fallback = (
                    (r_soft_c < cfg.r_soft_min)
                    | (c < cfg.c_min)
                    | (c > cfg.c_max)
                    | ~np.isfinite(c)
                )
                F_scalar[close] = np.where(fallback, F_scalar[close], c * F_soft_c)
                fallback_count += int(np.sum(fallback))
            else:
                fallback = r_soft_c < cfg.r_soft_min
                F_scalar[close] = np.where(fallback, F_scalar[close], F_soft_c)
                fallback_count += int(np.sum(fallback))

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

    def substep(x_in, v_in, a_in, sub_dt):
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
    steps_done = 0
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < cfg.adapt_threshold:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_threshold / r_min))))
            sub_dt = float(dt) / n_sub
            for _ in range(n_sub):
                x, v, a = substep(x, v, a, sub_dt)
            total_substeps += n_sub
        else:
            vh = v + 0.5 * float(dt) * a
            x = x + float(dt) * vh
            a = compute_acc(x)
            v = vh + 0.5 * float(dt) * a
            total_substeps += 1

        t_cur += float(dt)
        steps_done += 1

        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]

        max_r = float(np.max(np.linalg.norm(x, axis=1)))
        if (not np.isfinite(max_r)) or max_r > EJECTION_AU:
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

    perf = {
        "model_type": model_type,
        "steps": steps_done,
        "planned_steps": n_steps,
        "dt": float(dt),
        "T": float(T),
        "total_time_sec": time.perf_counter() - t0,
        "time_per_step_sec": (time.perf_counter() - t0) / max(steps_done, 1),
        "close_pair_count": int(close_pair_count),
        "fallback_count": int(fallback_count),
        "pair_eval_count": int(pair_eval_count),
        "avg_close_pair_frac": close_pair_count / max(pair_eval_count, 1),
        "avg_fallback_frac": fallback_count / max(pair_eval_count, 1),
        "total_substeps": int(total_substeps),
        "min_pair_distance": float(min_r_seen),
        "ejected": bool(ejected),
        "ejection_time": float(ejection_time),
    }
    return times, pos_out, vel_out, perf


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rms_sep(pos_a, pos_b):
    d = pos_a - pos_b
    per_body = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(per_body * per_body, axis=1))


def time_avg_rms(delta, mask):
    return float(np.sqrt(np.mean(delta[mask] ** 2)))


def pair_distance(pos, i, j):
    return np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=1)


def max_distance_from_com(pos):
    return np.max(np.linalg.norm(pos, axis=2), axis=1)


def compute_energy_trajectory(pos, vel, m, G=1.0):
    """True Newtonian total energy. Periastron stays far above zero here."""
    m = m.astype(np.float64)
    KE = 0.5 * np.einsum("kij,i->k", vel * vel, m)
    PE = np.zeros(pos.shape[0], dtype=np.float64)
    N = len(m)
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=1)
            PE -= G * m[i] * m[j] / np.maximum(r, 1e-30)
    return KE + PE


def energy_drift_pct(pos, vel, m, G=1.0):
    E = compute_energy_trajectory(pos, vel, m, G=G)
    E0 = float(E[0])
    dE = (E - E0) / max(abs(E0), 1e-30)
    return dE * 100.0


def bounded_status(pos):
    finite = bool(np.all(np.isfinite(pos)))
    max_com = float(np.max(max_distance_from_com(pos))) if finite else np.inf
    return finite and max_com < EJECTION_AU, max_com


def summarize_model(label, times, pos, vel, pos_ref, m, perf, windows):
    delta = rms_sep(pos, pos_ref)
    e_drift = energy_drift_pct(pos, vel, m, G=G)
    bounded, max_com = bounded_status(pos)
    row = {
        "label": label,
        "runtime_sec": float(perf.get("total_time_sec", np.nan)),
        "steps": int(perf.get("steps", 0)),
        "total_substeps": int(perf.get("total_substeps", 0)),
        "close_pair_frac": float(perf.get("avg_close_pair_frac", np.nan)),
        "fallback_frac": float(perf.get("avg_fallback_frac", np.nan)),
        "min_pair_distance": float(perf.get("min_pair_distance", np.nan)),
        "ejected": bool(perf.get("ejected", False)),
        "ejection_time": float(perf.get("ejection_time", np.nan)),
        "bounded": bool(bounded),
        "max_com": float(max_com),
        "final_rms": float(delta[-1]),
        "max_rms": float(np.max(delta)),
        "full_time_avg_rms": float(np.sqrt(np.mean(delta**2))),
        "max_energy_drift_pct": float(np.max(np.abs(e_drift))),
        "final_energy_drift_pct": float(e_drift[-1]),
        "delta": delta,
        "energy_drift_pct": e_drift,
    }
    for name, t_end in windows.items():
        mask = times <= min(t_end, times[-1]) + 1e-12
        row[f"rms_{name}"] = time_avg_rms(delta, mask)
        row[f"final_rms_{name}"] = float(delta[mask][-1])
    return row


def improvement(no_nn_value, simon_value):
    ratio = float(no_nn_value / max(simon_value, 1e-30))
    reduction_pct = float(100.0 * (1.0 - simon_value / max(no_nn_value, 1e-30)))
    return ratio, reduction_pct


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_rms(times, delta_s, delta_n, out_path, t_max=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    mask = np.ones_like(times, dtype=bool)
    if t_max is not None:
        mask = times <= t_max + 1e-12
    ax.semilogy(times[mask], np.maximum(delta_s[mask], 1e-14), lw=1.8,
                label="SIMON with NN")
    ax.semilogy(times[mask], np.maximum(delta_n[mask], 1e-14), "--", lw=1.8,
                label="No-NN baseline")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_inner_sep(times, r_ref, r_s, r_n, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(times, r_ref, lw=1.7, label="ias15")
    ax.plot(times, r_s, "--", lw=1.4, label="SIMON with NN")
    ax.plot(times, r_n, ":", lw=1.7, label="No-NN baseline")
    ax.axhline(NN_THRESHOLD, color="0.35", linestyle="--", lw=1.1,
               label="NN threshold = 0.15 AU")
    ax.axhline(ADAPT_THRESH, color="0.55", linestyle=":", lw=1.1,
               label="adaptive threshold = 0.05 AU")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Inner-binary separation r01 (AU)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_energy(times, e_s, e_n, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(times, np.maximum(np.abs(e_s), 1e-14), lw=1.8,
                label="SIMON with NN")
    ax.semilogy(times, np.maximum(np.abs(e_n), 1e-14), "--", lw=1.8,
                label="No-NN baseline")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"$|\Delta E/E_0|$ (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xy(pos_ref, pos_s, pos_n, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    labels = ["Body 0", "Body 1", "Body 2"]
    for i, ax in enumerate(axes):
        ax.plot(pos_ref[:, i, 0], pos_ref[:, i, 1], lw=1.5, label="ias15")
        ax.plot(pos_s[:, i, 0], pos_s[:, i, 1], "--", lw=1.2, label="SIMON with NN")
        ax.plot(pos_n[:, i, 0], pos_n[:, i, 1], ":", lw=1.4, label="No-NN")
        ax.set_title(labels[i])
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_summary(out_path, params, args, ref_perf, simon_row, no_nn_row, windows):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("COMPACT ECCENTRIC TRIPLE NN-VALUE EXPERIMENT\n")
        f.write("=" * 78 + "\n\n")
        f.write("Purpose:\n")
        f.write(
            "  Test whether SIMON's NN correction improves trajectory accuracy, "
            "not stability, in a three-body rollout where the NN is repeatedly active.\n\n"
        )
        f.write("Design:\n")
        f.write("  ias15 reference vs SIMON with NN vs No-NN baseline.\n")
        f.write("  Adaptive sub-stepping and analytic force direction remain ON for both SIMON variants.\n")
        f.write("  The only model difference is c predicted by NN vs c fixed to 1.\n\n")

        f.write("Initial condition parameters:\n")
        for key, val in params.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"  NN threshold: {NN_THRESHOLD} AU\n")
        f.write(f"  adaptive threshold: {ADAPT_THRESH} AU\n")
        f.write(f"  safety threshold r_soft_min: {R_SOFT_MIN} AU\n\n")

        f.write("Run configuration:\n")
        f.write(f"  dt: {args.dt} yr\n")
        f.write(f"  T: {args.T} yr\n")
        f.write(f"  n_samples: {args.n_samples}\n")
        f.write(f"  ias15 runtime: {ref_perf['total_time_sec']:.6f} s\n\n")

        f.write("Core results:\n")
        f.write("  Model, bounded, ejected, close_pair_frac, min_pair_distance, full_time_avg_rms, final_rms, max_energy_drift_pct\n")
        for row in [simon_row, no_nn_row]:
            f.write(
                f"  {row['label']}, {row['bounded']}, {row['ejected']}, "
                f"{row['close_pair_frac']:.8e}, {row['min_pair_distance']:.8e}, "
                f"{row['full_time_avg_rms']:.8e}, {row['final_rms']:.8e}, "
                f"{row['max_energy_drift_pct']:.8e}\n"
            )
        f.write("\n")

        f.write("Early-window trajectory accuracy:\n")
        for name in windows:
            s_val = simon_row[f"rms_{name}"]
            n_val = no_nn_row[f"rms_{name}"]
            ratio, reduction = improvement(n_val, s_val)
            f.write(
                f"  {name}: SIMON={s_val:.8e}, No-NN={n_val:.8e}, "
                f"NoNN/SIMON={ratio:.4f}x, reduction={reduction:.2f}%\n"
            )
        f.write("\n")

        f.write("Interpretation guide:\n")
        f.write(
            "  If both SIMON and No-NN remain bounded but SIMON has lower early-window RMS, "
            "the experiment supports the claim that the NN correction improves accuracy rather "
            "than providing the primary stability mechanism.\n"
        )


def write_metrics_csv(out_path, simon_row, no_nn_row, windows):
    fieldnames = [
        "label", "runtime_sec", "steps", "total_substeps", "close_pair_frac",
        "fallback_frac", "min_pair_distance", "ejected", "ejection_time",
        "bounded", "max_com", "final_rms", "max_rms", "full_time_avg_rms",
        "max_energy_drift_pct", "final_energy_drift_pct",
    ]
    for name in windows:
        fieldnames += [f"rms_{name}", f"final_rms_{name}"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in [simon_row, no_nn_row]:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.01,
                        help="SIMON/No-NN macro timestep in years. Default: 0.01")
    parser.add_argument("--T", type=float, default=20.0,
                        help="Simulation horizon in years. Default: 20")
    parser.add_argument("--n_samples", type=int, default=4000,
                        help="Recorded samples. Default: 4000")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[compact_eccentric_triple_nn_value] Loading SIMON model ...")
    model = load_model(MODEL_PATH)
    cfg = HybridConfig()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {MODEL_PATH} with {n_params} parameters")

    x0, v0, m, params = make_compact_eccentric_triple()
    print("\n[initial condition]")
    print(f"  masses = {m}")
    print(f"  inner periastron q_in = {params['q_in']:.6f} AU")
    print(f"  inner period = {params['period_in']:.6f} yr")
    print(f"  outer periastron q_out = {params['q_out']:.6f} AU")
    print(f"  outer period = {params['period_out']:.6f} yr")
    print(f"  NN threshold = {NN_THRESHOLD:.6f} AU")
    print(f"  dt = {args.dt:.6f} yr, T = {args.T:.3f} yr, n_samples = {args.n_samples}")

    print("\n[running ias15 reference]")
    times, pos_ref, vel_ref, perf_ref = simulate_ias15(x0, v0, m, cfg, args.T, args.n_samples)
    print(f"  ias15 runtime: {perf_ref['total_time_sec']:.3f}s")

    print("\n[running SIMON with NN]")
    _, pos_simon, vel_simon, perf_simon = simulate_hybrid(
        x0, v0, m, model, cfg, args.dt, args.T, args.n_samples, model_type="simon"
    )
    print(
        f"  SIMON runtime: {perf_simon['total_time_sec']:.3f}s | "
        f"close_pair_frac={perf_simon['avg_close_pair_frac']:.6f} | "
        f"min_r={perf_simon['min_pair_distance']:.6e} AU | "
        f"ejected={perf_simon['ejected']}"
    )

    print("\n[running No-NN baseline]")
    _, pos_no_nn, vel_no_nn, perf_no_nn = simulate_hybrid(
        x0, v0, m, model, cfg, args.dt, args.T, args.n_samples, model_type="no_nn"
    )
    print(
        f"  No-NN runtime: {perf_no_nn['total_time_sec']:.3f}s | "
        f"close_pair_frac={perf_no_nn['avg_close_pair_frac']:.6f} | "
        f"min_r={perf_no_nn['min_pair_distance']:.6e} AU | "
        f"ejected={perf_no_nn['ejected']}"
    )

    windows = {
        "0_1yr": 1.0,
        "0_2yr": 2.0,
        "0_5yr": 5.0,
        "0_10yr": 10.0,
        "0_1inner_period": params["period_in"],
        "0_2inner_periods": 2.0 * params["period_in"],
    }

    simon_row = summarize_model("SIMON_with_NN", times, pos_simon, vel_simon,
                                pos_ref, m, perf_simon, windows)
    no_nn_row = summarize_model("No_NN_baseline", times, pos_no_nn, vel_no_nn,
                                pos_ref, m, perf_no_nn, windows)

    print("\n[accuracy summary]")
    for name in windows:
        s_val = simon_row[f"rms_{name}"]
        n_val = no_nn_row[f"rms_{name}"]
        ratio, reduction = improvement(n_val, s_val)
        print(
            f"  {name}: SIMON={s_val:.6e}, No-NN={n_val:.6e}, "
            f"NoNN/SIMON={ratio:.2f}x, reduction={reduction:.1f}%"
        )

    print("\n[boundedness and energy]")
    for row in [simon_row, no_nn_row]:
        print(
            f"  {row['label']}: bounded={row['bounded']}, ejected={row['ejected']}, "
            f"final_RMS={row['final_rms']:.6e}, "
            f"max_energy_drift={row['max_energy_drift_pct']:.6e}%"
        )

    # Pair separations and output data.
    r01_ref = pair_distance(pos_ref, 0, 1)
    r01_simon = pair_distance(pos_simon, 0, 1)
    r01_no_nn = pair_distance(pos_no_nn, 0, 1)

    delta_simon = simon_row["delta"]
    delta_no_nn = no_nn_row["delta"]
    e_simon = simon_row["energy_drift_pct"]
    e_no_nn = no_nn_row["energy_drift_pct"]

    # Plots.
    plot_rms(times, delta_simon, delta_no_nn,
             os.path.join(args.out_dir, "rms_error_early_0_10yr.png"), t_max=10.0)
    plot_rms(times, delta_simon, delta_no_nn,
             os.path.join(args.out_dir, "rms_error_full.png"), t_max=None)
    plot_inner_sep(times, r01_ref, r01_simon, r01_no_nn,
                   os.path.join(args.out_dir, "inner_separation.png"))
    plot_energy(times, e_simon, e_no_nn,
                os.path.join(args.out_dir, "energy_drift.png"))
    plot_xy(pos_ref, pos_simon, pos_no_nn,
            os.path.join(args.out_dir, "xy_trajectories.png"))

    # Raw time series.
    npz_path = os.path.join(args.out_dir, "compact_eccentric_triple_timeseries.npz")
    np.savez_compressed(
        npz_path,
        times=times,
        pos_ref=pos_ref,
        vel_ref=vel_ref,
        pos_simon=pos_simon,
        vel_simon=vel_simon,
        pos_no_nn=pos_no_nn,
        vel_no_nn=vel_no_nn,
        delta_simon=delta_simon,
        delta_no_nn=delta_no_nn,
        r01_ref=r01_ref,
        r01_simon=r01_simon,
        r01_no_nn=r01_no_nn,
        energy_drift_simon_pct=e_simon,
        energy_drift_no_nn_pct=e_no_nn,
    )

    metrics_path = os.path.join(args.out_dir, "compact_eccentric_triple_metrics.csv")
    write_metrics_csv(metrics_path, simon_row, no_nn_row, windows)

    summary_path = os.path.join(args.out_dir, "summary_compact_eccentric_triple.txt")
    write_summary(summary_path, params, args, perf_ref, simon_row, no_nn_row, windows)

    print("\n[outputs]")
    print(f"  Summary: {summary_path}")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Time series NPZ: {npz_path}")
    print(f"  Figures saved in: {args.out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
