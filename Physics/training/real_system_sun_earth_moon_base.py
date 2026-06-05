import os
import numpy as np
from astroquery.jplhorizons import Horizons

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JPL Horizons can be slow on first query, so increase timeout.
Horizons.TIMEOUT = 120

# -----------------------------
# Real-system validation setup
# -----------------------------
# Units:
# JPL Horizons vectors give:
#   position: AU
#   velocity: AU/day
#
# Your SIMON code uses:
#   position: AU
#   time: years
#   velocity: AU/year
#   G = 1.0
#
# So we convert velocity from AU/day to AU/year.
DAYS_PER_YEAR = 365.25

# In AU, years, and solar masses, the physical Solar System value is:
# G * M_sun = 4*pi^2 for a 1 AU, 1 year orbit.
G_REAL = 4.0 * np.pi**2

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
    This makes the real Sun-Earth-Moon setup consistent with your other experiments.
    """
    M = np.sum(m)
    x_com = np.sum(m[:, None] * x0, axis=0) / M
    v_com = np.sum(m[:, None] * v0, axis=0) / M

    return x0 - x_com, v0 - v_com

import torch
import torch.nn as nn
from dataclasses import dataclass
import math
import time
import rebound


class PairCorrectionNN(nn.Module):
    """
    Same scalar correction network used in your existing SIMON experiments.
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
    This follows the same approach as your existing pair_eval / multi-IC scripts.
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

    # Masses in solar masses
    m = np.array([
        1.0,                 # Sun
        3.003489614915e-6,   # Earth
        3.694303349e-8,      # Moon
    ], dtype=np.float64)

    # Shift to centre-of-mass frame, matching the convention used in your other scripts.
    x0, v0 = move_to_center_of_mass(x0, v0, m)

    return x0, v0, m

def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    """
    Reference integration using REBOUND ias15.
    This is the real-system ground truth comparison for Sun-Earth-Moon.
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

    perf = {
        "total_time_sec": time.perf_counter() - t0,
    }

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
):
    """
    SIMON rollout for real Sun-Earth-Moon initial conditions.

    Uses:
      - leapfrog / velocity-Verlet backbone
      - analytic force direction
      - NN scalar correction for r < 0.15 AU
      - adaptive sub-stepping for r_min < 0.05 AU
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
    adapt_thresh = 0.05              # same as your paper experiments
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

        if t_cur >= T - 1e-12:
            break

    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    perf = {
        "steps": n_steps,
        "total_time_sec": time.perf_counter() - t0,
        "avg_close_pair_frac": close_sum / max(pair_sum, 1),
        "total_substeps": total_substeps,
    }

    return times, pos_out, vel_out, perf


def rms_sep(a, b):
    """
    RMS position separation between two simulations.
    Shape expected: (n_samples, n_bodies, 3).
    """
    d = a - b
    per_body = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(per_body**2, axis=1))


def pair_distance(pos_arr, i, j):
    """
    Distance between body i and body j over time.
    """
    return np.linalg.norm(pos_arr[:, i, :] - pos_arr[:, j, :], axis=1)


def max_distance_from_com(pos_arr):
    """
    Maximum body distance from origin/COM at each sampled time.
    Useful for boundedness checks.
    """
    return np.max(np.linalg.norm(pos_arr, axis=2), axis=1)

def plot_rms_vs_time(times, delta, out_path):
    """
    Save RMS position error between SIMON and ias15 over time.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(times, delta, lw=1.8)

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_pair_distances(
    times,
    ref_sun_earth,
    sim_sun_earth,
    ref_earth_moon,
    sim_earth_moon,
    out_path,
):
    """
    Save Sun-Earth and Earth-Moon distance comparison over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    axes[0].plot(times, ref_sun_earth, lw=1.8, label="ias15")
    axes[0].plot(times, sim_sun_earth, "--", lw=1.8, label="SIMON")
    axes[0].set_ylabel("Sun-Earth distance (AU)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    axes[1].plot(times, ref_earth_moon, lw=1.8, label="ias15")
    axes[1].plot(times, sim_earth_moon, "--", lw=1.8, label="SIMON")
    axes[1].set_xlabel("Time (yr)")
    axes[1].set_ylabel("Earth-Moon distance (AU)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_xy_trajectories(pos_ref, pos_sim, out_path):
    """
    Save XY trajectory overlays for the real Sun-Earth-Moon validation.

    Panels:
      1. Sun in barycentric frame
      2. Earth in barycentric frame
      3. Moon relative to Earth, so the lunar orbit is visible
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel 1: Sun barycentric motion
    axes[0].plot(pos_ref[:, 0, 0], pos_ref[:, 0, 1], lw=1.5, label="ias15")
    axes[0].plot(pos_sim[:, 0, 0], pos_sim[:, 0, 1], "--", lw=1.5, label="SIMON")
    axes[0].set_title("Sun")
    axes[0].set_xlabel("x (AU)")
    axes[0].set_ylabel("y (AU)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(framealpha=0.85)

    # Panel 2: Earth barycentric orbit
    axes[1].plot(pos_ref[:, 1, 0], pos_ref[:, 1, 1], lw=1.5, label="ias15")
    axes[1].plot(pos_sim[:, 1, 0], pos_sim[:, 1, 1], "--", lw=1.5, label="SIMON")
    axes[1].set_title("Earth")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("y (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    # Panel 3: Moon relative to Earth
    moon_rel_ref = pos_ref[:, 2, :] - pos_ref[:, 1, :]
    moon_rel_sim = pos_sim[:, 2, :] - pos_sim[:, 1, :]

    axes[2].plot(moon_rel_ref[:, 0], moon_rel_ref[:, 1], lw=1.5, label="ias15")
    axes[2].plot(moon_rel_sim[:, 0], moon_rel_sim[:, 1], "--", lw=1.5, label="SIMON")
    axes[2].set_title("Moon relative to Earth")
    axes[2].set_xlabel("x relative to Earth (AU)")
    axes[2].set_ylabel("y relative to Earth (AU)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    MODEL_PATH = "pair_correction_nn.pt"

    # Keep this short for the first real-system test.
    # Later we can increase to 5 or 10 years after confirming it works.
    T = 1.0
    DT = 0.001
    N_SAMPLES = 1000

    # Clean output structure for real-system validation runs.
    BASE_OUT_DIR = "real_system_validation"
    CASE_NAME = "sun_earth_moon_1yr_dt0001"
    OUT_DIR = os.path.join(BASE_OUT_DIR, CASE_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    x0, v0, m = load_sun_earth_moon_initial_conditions()

    earth_moon_dist = np.linalg.norm(x0[2] - x0[1])
    sun_earth_dist = np.linalg.norm(x0[1] - x0[0])

    print("\nSanity checks:")
    print(f"Sun-Earth distance  = {sun_earth_dist:.6f} AU")
    print(f"Earth-Moon distance = {earth_moon_dist:.6f} AU")
    print(f"G_REAL              = {G_REAL:.8f}")

    print("\nLoading SIMON model...")
    model = PairCorrectionNN(hidden=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cfg = HybridConfig(G=G_REAL)

    print("\nRunning ias15 reference...")
    t_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(
        x0, v0, m, cfg.G, T, N_SAMPLES
    )

    print("Running SIMON real-system rollout (adaptive ON)...")
    t_sim, pos_sim, vel_sim, perf_sim = simulate_simon_real_system(
        x0, v0, m, model, cfg, DT, T, N_SAMPLES, adaptive_enabled=True
    )

    print("Running SIMON real-system rollout (adaptive OFF)...")
    t_no_adapt, pos_no_adapt, vel_no_adapt, perf_no_adapt = simulate_simon_real_system(
        x0, v0, m, model, cfg, DT, T, N_SAMPLES, adaptive_enabled=False
    )

    delta = rms_sep(pos_sim, pos_ref)
    delta_no_adapt = rms_sep(pos_no_adapt, pos_ref)

    # Paper-consistent accuracy metric:
    # RMS_{0:T} = sqrt(mean_t delta(t)^2), where delta(t) is the
    # per-time RMS separation over the three bodies.
    time_avg_rms = np.sqrt(np.mean(delta**2))
    final_rms = delta[-1]
    max_rms = np.max(delta)

    time_avg_rms_no_adapt = np.sqrt(np.mean(delta_no_adapt**2))
    final_rms_no_adapt = delta_no_adapt[-1]
    max_rms_no_adapt = np.max(delta_no_adapt)

    ref_sun_earth = pair_distance(pos_ref, 0, 1)
    sim_sun_earth = pair_distance(pos_sim, 0, 1)
    no_adapt_sun_earth = pair_distance(pos_no_adapt, 0, 1)

    ref_earth_moon = pair_distance(pos_ref, 1, 2)
    sim_earth_moon = pair_distance(pos_sim, 1, 2)
    no_adapt_earth_moon = pair_distance(pos_no_adapt, 1, 2)

    ref_max_com = max_distance_from_com(pos_ref)
    sim_max_com = max_distance_from_com(pos_sim)
    no_adapt_max_com = max_distance_from_com(pos_no_adapt)

    print("\nReal Sun-Earth-Moon validation summary:")
    print(f"T                       = {T:.2f} yr")
    print(f"dt                      = {DT:.4f} yr")
    print(f"ias15 runtime            = {perf_ref['total_time_sec']:.3f} s")

    print("\nAdaptive ON:")
    print(f"SIMON runtime            = {perf_sim['total_time_sec']:.3f} s")
    print(f"SIMON close-pair fraction= {perf_sim['avg_close_pair_frac']:.6f}")
    print(f"SIMON total substeps     = {perf_sim['total_substeps']}")
    print(f"Time-averaged RMS        = {time_avg_rms:.6e} AU")
    print(f"Final RMS                = {final_rms:.6e} AU")
    print(f"Max RMS                  = {max_rms:.6e} AU")

    print("\nAdaptive OFF:")
    print(f"SIMON runtime            = {perf_no_adapt['total_time_sec']:.3f} s")
    print(f"SIMON close-pair fraction= {perf_no_adapt['avg_close_pair_frac']:.6f}")
    print(f"SIMON total substeps     = {perf_no_adapt['total_substeps']}")
    print(f"Time-averaged RMS        = {time_avg_rms_no_adapt:.6e} AU")
    print(f"Final RMS                = {final_rms_no_adapt:.6e} AU")
    print(f"Max RMS                  = {max_rms_no_adapt:.6e} AU")

    print("\nDistance ranges over simulation:")
    print(
        f"ias15 Sun-Earth        : {np.min(ref_sun_earth):.6f} to {np.max(ref_sun_earth):.6f} AU"
    )
    print(
        f"SIMON ON Sun-Earth     : {np.min(sim_sun_earth):.6f} to {np.max(sim_sun_earth):.6f} AU"
    )
    print(
        f"SIMON OFF Sun-Earth    : {np.min(no_adapt_sun_earth):.6f} to {np.max(no_adapt_sun_earth):.6f} AU"
    )
    print(
        f"ias15 Earth-Moon       : {np.min(ref_earth_moon):.6f} to {np.max(ref_earth_moon):.6f} AU"
    )
    print(
        f"SIMON ON Earth-Moon    : {np.min(sim_earth_moon):.6f} to {np.max(sim_earth_moon):.6f} AU"
    )
    print(
        f"SIMON OFF Earth-Moon   : {np.min(no_adapt_earth_moon):.6f} to {np.max(no_adapt_earth_moon):.6f} AU"
    )

    print("\nBoundedness check:")
    print(f"ias15 max distance from COM     = {np.max(ref_max_com):.6f} AU")
    print(f"SIMON ON max distance from COM  = {np.max(sim_max_com):.6f} AU")
    print(f"SIMON OFF max distance from COM = {np.max(no_adapt_max_com):.6f} AU")

    summary_path = os.path.join(OUT_DIR, "summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Real Sun-Earth-Moon validation using JPL Horizons initial conditions\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"T                       = {T:.2f} yr\n")
        f.write(f"dt                      = {DT:.4f} yr\n")
        f.write(f"G_REAL                  = {G_REAL:.8f}\n")

        f.write(f"ias15 runtime            = {perf_ref['total_time_sec']:.6f} s\n\n")

        f.write("Adaptive ON:\n")
        f.write(f"SIMON runtime            = {perf_sim['total_time_sec']:.6f} s\n")
        f.write(f"SIMON close-pair fraction= {perf_sim['avg_close_pair_frac']:.6f}\n")
        f.write(f"SIMON total substeps     = {perf_sim['total_substeps']}\n")
        f.write(f"Time-averaged RMS        = {time_avg_rms:.6e} AU\n")
        f.write(f"Final RMS                = {final_rms:.6e} AU\n")
        f.write(f"Max RMS                  = {max_rms:.6e} AU\n\n")

        f.write("Adaptive OFF:\n")
        f.write(f"SIMON runtime            = {perf_no_adapt['total_time_sec']:.6f} s\n")
        f.write(f"SIMON close-pair fraction= {perf_no_adapt['avg_close_pair_frac']:.6f}\n")
        f.write(f"SIMON total substeps     = {perf_no_adapt['total_substeps']}\n")
        f.write(f"Time-averaged RMS        = {time_avg_rms_no_adapt:.6e} AU\n")
        f.write(f"Final RMS                = {final_rms_no_adapt:.6e} AU\n")
        f.write(f"Max RMS                  = {max_rms_no_adapt:.6e} AU\n\n")

        f.write("Distance ranges over simulation:\n")
        f.write(
            f"ias15 Sun-Earth   : {np.min(ref_sun_earth):.6f} to {np.max(ref_sun_earth):.6f} AU\n"
        )
        f.write(
            f"SIMON Sun-Earth   : {np.min(sim_sun_earth):.6f} to {np.max(sim_sun_earth):.6f} AU\n"
        )
        f.write(
            f"ias15 Earth-Moon  : {np.min(ref_earth_moon):.6f} to {np.max(ref_earth_moon):.6f} AU\n"
        )
        f.write(
            f"SIMON Earth-Moon  : {np.min(sim_earth_moon):.6f} to {np.max(sim_earth_moon):.6f} AU\n\n"
        )

        f.write("Boundedness check:\n")
        f.write(f"ias15 max distance from COM = {np.max(ref_max_com):.6f} AU\n")
        f.write(f"SIMON max distance from COM = {np.max(sim_max_com):.6f} AU\n")

    print(f"\nSaved summary to {summary_path}")

    data_path = os.path.join(OUT_DIR, "trajectory_data.npz")

    np.savez_compressed(
        data_path,
        times=t_ref,
        x0=x0,
        v0=v0,
        masses=m,
        pos_ias15=pos_ref,
        vel_ias15=vel_ref,
        pos_simon=pos_sim,
        vel_simon=vel_sim,

        rms_vs_ias15=delta,
        time_avg_rms=time_avg_rms,
        final_rms=final_rms,
        max_rms=max_rms,
        ref_sun_earth=ref_sun_earth,

        sim_sun_earth=sim_sun_earth,
        ref_earth_moon=ref_earth_moon,
        sim_earth_moon=sim_earth_moon,
        ref_max_com=ref_max_com,
        sim_max_com=sim_max_com,
        T=T,
        DT=DT,
        G_REAL=G_REAL,
    )

    print(f"Saved trajectory data to {data_path}")

    rms_plot_path = os.path.join(OUT_DIR, "rms_vs_time.png")
    plot_rms_vs_time(t_ref, delta, rms_plot_path)
    print(f"Saved RMS plot to {rms_plot_path}")

    distance_plot_path = os.path.join(OUT_DIR, "distance_comparison.png")
    plot_pair_distances(
        t_ref,
        ref_sun_earth,
        sim_sun_earth,
        ref_earth_moon,
        sim_earth_moon,
        distance_plot_path,
    )
    print(f"Saved distance comparison plot to {distance_plot_path}")

    xy_plot_path = os.path.join(OUT_DIR, "xy_trajectories.png")
    plot_xy_trajectories(pos_ref, pos_sim, xy_plot_path)
    print(f"Saved XY trajectory plot to {xy_plot_path}")