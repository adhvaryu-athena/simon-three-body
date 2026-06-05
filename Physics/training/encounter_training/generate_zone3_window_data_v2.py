"""
generate_zone3_window_data_v2.py  --  Zone 3 short-window trajectory data

Purpose
-------
Generates ONE batch of Zone 3 training data where the scalar target is not the
old one-step c_opt.  Instead, the target is a short-window optimum:

    c_window_opt = argmin_c mean_RMS_error(SIMON_window(c), IAS15_window)

The trial scalar c is applied only to the current Zone-3 pair (0,1) on the
first macro-step of the window.  The rest of the short window is then rolled
forward with the revised Phase-2 baseline rules:

    Zone 1: r < 4e-4 AU       -> direct Newtonian hard fallback
    Zone 2: r < 0.05 AU       -> direct Newtonian + adaptive sub-steps
    Zone 3: 0.05 <= r < 0.15  -> no-NN baseline c=1 times softened force
    Zone 4: r >= 0.15         -> direct Newtonian

This isolates the effect of the current Zone-3 correction decision while making
the target care about short future consequences rather than only the landing
position after one macro-step.

Compatibility with current pipeline
-----------------------------------
The .npz shard deliberately saves the SAME required fields as your current v3
merger/inspector/trainer:

    r_AU, r_soft, log_mi, log_mj, log_dt,
    v_rad_norm, v_tan_norm,
    c_opt, log_c_opt, c_ana, log_c_ana, improvement

In this file:
    c_opt       = c_window_opt
    log_c_opt   = log(c_window_opt)
    improvement = short-window improvement vs c=1 baseline

Extra metadata fields are also saved in the shard for traceability, but your
existing merge_encounter_shards.py will ignore them and still merge cleanly.

Recommended first test:
    python -B generate_zone3_window_data_v2.py --dt 0.08 --n 5 --batch 1 --mode mixed --window_years 0.5 --window_samples 21

Recommended real batches after the test:
    python -B generate_zone3_window_data_v2.py --dt 0.04 --n 25 --batch 101 --mode mixed --window_years 0.5 --window_samples 21
    python -B generate_zone3_window_data_v2.py --dt 0.08 --n 25 --batch 101 --mode mixed --window_years 0.5 --window_samples 21
    python -B generate_zone3_window_data_v2.py --dt 0.10 --n 25 --batch 101 --mode mixed --window_years 0.5 --window_samples 21
"""

import os
import sys
import time
import math
import argparse
import multiprocessing as mp

import numpy as np
import rebound
import faulthandler

faulthandler.enable(file=sys.stderr)

# =============================================================================
# Constants matched to Phase-2 evaluator/generator
# =============================================================================
G            = 1.0
EPS          = 3e-4
R_SOFT_MIN   = 5e-4
NN_THRESH    = 500.0 * EPS      # 0.15 AU
ADAPT_THRESH = 0.05             # Zone 2 / Zone 3 boundary
C_MIN, C_MAX = 0.25, 3.0        # v4 deployment-safe training bounds
R_GATE       = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))  # 4e-4 AU

# Interior Zone-3 generation range, safely away from boundaries.
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU

SEED = 42
IAS15_TIMEOUT_SEC = 90.0

# =============================================================================
# Small helpers
# =============================================================================
def dt_token(dt):
    """Safe token for filenames: 0.005 -> 0p005."""
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _finite3(x):
    try:
        return np.all(np.isfinite(x))
    except Exception:
        return False


def safe_max_norm_3x3(x):
    try:
        mx = 0.0
        for i in range(len(x)):
            n2 = 0.0
            for j in range(len(x[i])):
                val = float(x[i][j])
                if not math.isfinite(val):
                    return float("inf")
                n2 += val * val
            mx = max(mx, math.sqrt(n2))
        return mx
    except Exception:
        return float("inf")


def rms_traj_error(pos, ref):
    """Mean RMS position error over sampled times and bodies."""
    try:
        if pos is None or ref is None:
            return float("inf")
        if pos.shape != ref.shape:
            return float("inf")
        if not (_finite3(pos) and _finite3(ref)):
            return float("inf")
        d = pos - ref
        per_body = np.sqrt(np.sum(d * d, axis=-1))      # (samples, bodies)
        per_time = np.sqrt(np.mean(per_body * per_body, axis=1))
        val = float(np.mean(per_time))
        return val if math.isfinite(val) else float("inf")
    except Exception:
        return float("inf")


# =============================================================================
# IAS15 short-window reference, isolated in subprocess for Windows crash safety
# =============================================================================
def ias15_window(x0, v0, m_arr, window_years, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    sim.exit_min_distance = float(EPS)

    for i in range(3):
        sim.add(m=float(m_arr[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))

    sim.move_to_com()
    times = np.linspace(0.0, float(window_years), int(n_samples))
    pos = np.zeros((len(times), 3, 3), dtype=np.float64)

    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]

    if safe_max_norm_3x3(pos[-1]) == float("inf"):
        raise ValueError("ias15 returned non-finite positions")
    return times, pos


def _ias15_window_worker(conn, x0, v0, m_arr, window_years, n_samples):
    try:
        times, pos = ias15_window(x0, v0, m_arr, window_years, n_samples)
        conn.send(("ok", times, pos, ""))
    except BaseException as e:
        try:
            conn.send(("err", None, None, f"{type(e).__name__}: {e}"))
        except BaseException:
            pass
    finally:
        try:
            conn.close()
        except BaseException:
            pass


def ias15_window_isolated(x0, v0, m_arr, window_years, n_samples, timeout=IAS15_TIMEOUT_SEC):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_ias15_window_worker,
                    args=(child_conn, x0, v0, m_arr, float(window_years), int(n_samples)))
    p.start()
    child_conn.close()
    p.join(float(timeout))

    if p.is_alive():
        p.terminate()
        p.join(5.0)
        parent_conn.close()
        raise TimeoutError(f"ias15 window subprocess timed out after {timeout:.1f}s")

    if p.exitcode != 0:
        parent_conn.close()
        raise RuntimeError(f"ias15 window subprocess exited with code {p.exitcode}")

    if not parent_conn.poll():
        parent_conn.close()
        raise RuntimeError("ias15 window subprocess returned no result")

    status, times, pos, msg = parent_conn.recv()
    parent_conn.close()
    if status != "ok":
        raise RuntimeError(msg)
    return times, pos


def ias15_window_safe(x0, v0, m_arr, window_years, n_samples, timeout=IAS15_TIMEOUT_SEC, isolate=False):
    """
    Return the IAS15 short-window reference.

    v1 used a fresh spawned subprocess for every IAS15 reference. On Windows,
    repeated spawn calls can fail while the child is importing NumPy/platform,
    which produces noisy traceback messages and may stop long batches.

    v2 defaults to direct same-process IAS15 because the generated Zone-3
    states are already filtered and short-window runs are small. If you ever
    see a true REBOUND crash rather than spawn/import errors, rerun with
    --isolate_ias15 to restore the v1 subprocess behavior.
    """
    if isolate:
        return ias15_window_isolated(x0, v0, m_arr, window_years, n_samples, timeout=timeout)
    return ias15_window(x0, v0, m_arr, window_years, n_samples)


# =============================================================================
# Window rollout with one trial c applied to first Zone-3 pair step only
# =============================================================================
def pair_geometry(pos):
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = []
    for i, j in pairs:
        rij = pos[j] - pos[i]
        r2 = float(np.dot(rij, rij))
        r = float(math.sqrt(r2 + 1e-30))
        out.append((i, j, rij, r2, r))
    return out


def min_pair_distance(pos):
    return min(g[4] for g in pair_geometry(pos))


def acc_phase2_window(pos, m_arr, c_trial, apply_trial_this_step):
    """
    Revised Phase-2 acceleration for window-data generation.

    Pair (0,1) receives c_trial only if apply_trial_this_step=True and the pair
    is in Zone 3.  Otherwise Zone 3 uses the no-NN baseline c=1*F_soft.
    Zone 2 and hard-fallback use direct Newtonian.
    """
    acc = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS

    for idx, (i, j, rij, r2, r) in enumerate(pair_geometry(pos)):
        Gmimj = G * float(m_arr[i]) * float(m_arr[j])

        if r < ADAPT_THRESH:
            # Zone 1/2: direct Newtonian.  Time resolution is handled by substeps.
            denom = r2 * r + 1e-30
            F_scalar = Gmimj / denom
        elif r < NN_THRESH:
            # Zone 3: softened-force path.  Trial c only applies to current pair
            # on the first macro-step; all other Zone-3 uses are no-NN c=1.
            c_val = float(c_trial) if (idx == 0 and apply_trial_this_step) else 1.0
            denom = (r2 + eps2) ** 1.5 + 1e-30
            F_scalar = c_val * Gmimj / denom
        else:
            # Zone 4: direct Newtonian.
            denom = r2 * r + 1e-30
            F_scalar = Gmimj / denom

        F_vec = F_scalar * rij
        acc[i] +=  F_vec / float(m_arr[i])
        acc[j] -=  F_vec / float(m_arr[j])

    return acc


def verlet_step(pos, vel, m_arr, h, c_trial, apply_trial_this_step):
    a0 = acc_phase2_window(pos, m_arr, c_trial, apply_trial_this_step)
    v_half = vel + 0.5 * float(h) * a0
    x1 = pos + float(h) * v_half
    a1 = acc_phase2_window(x1, m_arr, c_trial, apply_trial_this_step)
    v1 = v_half + 0.5 * float(h) * a1
    return x1, v1


def macro_step_phase2_window(pos, vel, m_arr, dt_step, c_trial, apply_trial_this_step, max_substeps):
    rmin = min_pair_distance(pos)
    if rmin < ADAPT_THRESH:
        # Simple conservative adaptive subdivision for the short window.
        # This mirrors the Phase-2 principle: Zone 2 is solved by smaller
        # timesteps and direct Newtonian force, not NN correction.
        n_sub = int(max(1, min(int(max_substeps), math.ceil(float(dt_step) / max(float(dt_step) / int(max_substeps), 1e-30)))))
        # In practice this becomes max_substeps whenever Zone 2 is entered.
        h = float(dt_step) / n_sub
        x, v = pos, vel
        for _ in range(n_sub):
            # Never apply the Zone-3 trial c inside Zone-2 substeps.
            x, v = verlet_step(x, v, m_arr, h, c_trial, False)
        return x, v

    return verlet_step(pos, vel, m_arr, dt_step, c_trial, apply_trial_this_step)


def simulate_window_trial(x0, v0, m_arr, dt, window_years, sample_times, c_trial, max_substeps=16):
    """
    Roll forward a short window.  The trial scalar is applied only to the first
    macro-step if pair (0,1) starts in Zone 3.  Later Zone-3 firings use c=1.
    """
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    t = 0.0
    sample_times = np.asarray(sample_times, dtype=np.float64)
    pos_out = np.zeros((len(sample_times), 3, 3), dtype=np.float64)
    sample_idx = 0

    # Save t=0 if requested.
    while sample_idx < len(sample_times) and sample_times[sample_idx] <= 1e-14:
        pos_out[sample_idx] = x
        sample_idx += 1

    first_macro = True
    while t < float(window_years) - 1e-14:
        h = min(float(dt), float(window_years) - t)

        x_prev = x.copy()
        v_prev = v.copy()
        t_prev = t

        x, v = macro_step_phase2_window(x, v, m_arr, h, c_trial, first_macro, max_substeps)
        t = t + h
        first_macro = False

        if safe_max_norm_3x3(x) > 1e6:
            return None

        # Linear interpolation of positions between macro endpoints for the
        # sampled loss.  This is adequate for comparing candidate c values over
        # a short window with the same macro-step grid.
        while sample_idx < len(sample_times) and sample_times[sample_idx] <= t + 1e-14:
            alpha = 0.0 if t == t_prev else float((sample_times[sample_idx] - t_prev) / (t - t_prev))
            alpha = max(0.0, min(1.0, alpha))
            pos_out[sample_idx] = (1.0 - alpha) * x_prev + alpha * x
            sample_idx += 1

    while sample_idx < len(sample_times):
        pos_out[sample_idx] = x
        sample_idx += 1

    return pos_out


def find_c_window_opt(x0, v0, m_arr, dt, window_years, ref_times, ref_pos, c_grid, max_substeps):
    base_pos = simulate_window_trial(x0, v0, m_arr, dt, window_years, ref_times, 1.0, max_substeps=max_substeps)
    loss_c1 = rms_traj_error(base_pos, ref_pos)
    if not math.isfinite(loss_c1):
        return np.nan, float("inf"), float("inf")

    losses = []
    for c in c_grid:
        pos = simulate_window_trial(x0, v0, m_arr, dt, window_years, ref_times, float(c), max_substeps=max_substeps)
        losses.append(rms_traj_error(pos, ref_pos))

    finite_idx = [i for i, val in enumerate(losses) if math.isfinite(val)]
    if not finite_idx:
        return np.nan, float("inf"), loss_c1

    best_k = min(finite_idx, key=lambda i: losses[i])
    c_lo = float(c_grid[max(0, best_k - 1)])
    c_hi = float(c_grid[min(len(c_grid) - 1, best_k + 1)])

    # Refine with golden section on the local bracket.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(24):
        if c_hi - c_lo < 1e-6:
            break
        c1 = c_hi - phi * (c_hi - c_lo)
        c2 = c_lo + phi * (c_hi - c_lo)
        p1 = simulate_window_trial(x0, v0, m_arr, dt, window_years, ref_times, c1, max_substeps=max_substeps)
        p2 = simulate_window_trial(x0, v0, m_arr, dt, window_years, ref_times, c2, max_substeps=max_substeps)
        l1 = rms_traj_error(p1, ref_pos)
        l2 = rms_traj_error(p2, ref_pos)
        if not math.isfinite(l1):
            l1 = float("inf")
        if not math.isfinite(l2):
            l2 = float("inf")
        if l1 < l2:
            c_hi = c2
        else:
            c_lo = c1

    c_opt = 0.5 * (c_lo + c_hi)
    pos_opt = simulate_window_trial(x0, v0, m_arr, dt, window_years, ref_times, c_opt, max_substeps=max_substeps)
    loss_opt = rms_traj_error(pos_opt, ref_pos)
    return float(c_opt), float(loss_opt), float(loss_c1)


# =============================================================================
# Synthetic Zone-3 IC generator with broad / strong_approach / mixed modes
# =============================================================================
def compute_pair_velocity_features(x_all, v_all, m_arr):
    r_vec = x_all[1] - x_all[0]
    v_vec = v_all[1] - v_all[0]
    r = float(np.linalg.norm(r_vec))
    r_hat = r_vec / (r + 1e-30)
    v_rad = float(np.dot(v_vec, r_hat))
    v_tan_vec = v_vec - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))
    v_scale = float(np.sqrt(G * (float(m_arr[0]) + float(m_arr[1])) / (r + 1e-30)))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30))


def _sample_masses(rng, mode="broad"):
    if mode == "strong_approach" and rng.rand() < 0.50:
        m0 = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        m1 = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
        m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    else:
        m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
        m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
        m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    return m0, m1, m2


def _make_pair_velocity(rng, r_hat, m0, m1, r01, mode="broad"):
    v_circ = float(np.sqrt(G * (m0 + m1) / r01))
    if mode == "strong_approach":
        vr_norm = float(rng.uniform(-1.20, -0.60))
        vt_norm = float(rng.uniform(0.40, 0.90))
        t_hat = np.array([-r_hat[1], r_hat[0], 0.0], dtype=np.float64)
        return v_circ * (vr_norm * r_hat + vt_norm * t_hat)

    f = float(rng.uniform(0.4, 1.8))
    phi_v = float(rng.uniform(0.0, 2.0 * np.pi))
    v_hat = np.array([np.cos(phi_v), np.sin(phi_v), 0.0], dtype=np.float64)
    return f * v_circ * v_hat


def make_ic(rng, mode="broad"):
    if mode not in ("broad", "strong_approach"):
        raise ValueError(f"Unknown make_ic mode: {mode}")

    m0, m1, m2 = _sample_masses(rng, mode=mode)
    m = np.array([m0, m1, m2], dtype=np.float64)

    r01 = float(rng.uniform(0.080, 0.120)) if mode == "strong_approach" else float(rng.uniform(Z3_R_MIN, Z3_R_MAX))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    v_rel = _make_pair_velocity(rng, r_hat, m0, m1, r01, mode=mode)

    x_rel = r01 * r_hat
    x0p = -(m1 / (m0 + m1)) * x_rel
    x1p =  (m0 / (m0 + m1)) * x_rel
    v0p = -(m1 / (m0 + m1)) * v_rel
    v1p =  (m0 / (m0 + m1)) * v_rel

    r2 = float(rng.uniform(3.0, 8.0))
    th2 = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p = r2 * np.array([np.cos(th2), np.sin(th2), 0.0], dtype=np.float64)
    f2 = float(rng.uniform(0.5, 1.1))
    vc2 = float(np.sqrt(G * (m0 + m1) / r2))
    v2p = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0], dtype=np.float64)

    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)

    M_tot = float(m.sum())
    x_all = x_all - np.sum(m[:, None] * x_all, axis=0) / M_tot
    v_all = v_all - np.sum(m[:, None] * v_all, axis=0) / M_tot

    # Reject globally unbound systems.
    KE = 0.5 * float(np.sum(m * np.array([float(np.dot(v_all[i], v_all[i])) for i in range(3)], dtype=np.float64)))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            rij = float(np.linalg.norm(x_all[i] - x_all[j]))
            PE -= G * m[i] * m[j] / (rij + 1e-30)
    if KE + PE >= 0.0:
        return None

    r01_check = float(np.linalg.norm(x_all[1] - x_all[0]))
    if not (Z3_R_MIN < r01_check < Z3_R_MAX):
        return None

    # Reject very close expected two-body periapses.
    r_rel_pair = x_all[1] - x_all[0]
    v_rel_pair = v_all[1] - v_all[0]
    v_sq = float(np.dot(v_rel_pair, v_rel_pair))
    M_pair = m0 + m1
    eps_orb = v_sq / 2.0 - G * M_pair / r01_check
    h_z = float(r_rel_pair[0] * v_rel_pair[1] - r_rel_pair[1] * v_rel_pair[0])
    h_sq = h_z * h_z
    disc = max(0.0, 1.0 + 2.0 * eps_orb * h_sq / (G * M_pair) ** 2)
    e_ecc = float(np.sqrt(disc))
    r_peri = h_sq / (G * M_pair * (1.0 + e_ecc) + 1e-30)
    if r_peri < EPS:
        return None

    # Keep third body away from Zone 3 at the start.
    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    v_rad_norm, v_tan_norm = compute_pair_velocity_features(x_all, v_all, m)
    if not (math.isfinite(v_rad_norm) and math.isfinite(v_tan_norm)):
        return None

    return x_all, v_all, m, r01_check, v_rad_norm, v_tan_norm


# =============================================================================
# CLI and main
# =============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Generate one batch of Zone 3 short-window trajectory-correction data."
    )
    ap.add_argument("--dt", type=float, required=True,
                    help="Macro timestep for the window rollout, e.g. 0.08")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of accepted samples to generate")
    ap.add_argument("--batch", type=int, default=0,
                    help="Batch id used for seed and filename")
    ap.add_argument("--mode", choices=["broad", "strong_approach", "mixed"], default="mixed",
                    help="Sampling mode for starting Zone-3 states")
    ap.add_argument("--window_years", type=float, default=0.5,
                    help="Short future window length in years")
    ap.add_argument("--window_samples", type=int, default=21,
                    help="Number of IAS15/SIMON sample times inside the window")
    ap.add_argument("--grid_n", type=int, default=35,
                    help="Number of coarse c-grid values before golden-section refinement")
    ap.add_argument("--identity_tol", type=float, default=0.03,
                    help="If window improvement is below this fraction, store c=1 identity target")
    ap.add_argument("--max_substeps", type=int, default=16,
                    help="Substeps used if the short window enters Zone 2")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="encounter_shards_window",
                    help="Folder for shard .npz and summary .txt files")
    ap.add_argument("--prefix", default="encounter_data_zone3_window_v2",
                    help="Output filename prefix")
    ap.add_argument("--timeout", type=float, default=IAS15_TIMEOUT_SEC,
                    help="IAS15 subprocess timeout per candidate state; used only with --isolate_ias15")
    ap.add_argument("--isolate_ias15", action="store_true",
                    help="Use a fresh spawned subprocess for each IAS15 reference. Default is direct same-process IAS15 to avoid Windows spawn/import failures.")
    ap.add_argument("--max_tries", type=int, default=0,
                    help="Maximum attempts before saving partial output. 0 means n*1000")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Base random seed")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing shard file")
    ap.add_argument("--verbose_attempts", action="store_true",
                    help="Print every attempt; useful for debugging, noisy for real batches")
    return ap


def main():
    args = build_arg_parser().parse_args()

    dt = float(args.dt)
    n_target = int(args.n)
    batch_id = int(args.batch)
    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else max(1000 * n_target, 1000)
    c_grid = np.linspace(C_MIN, C_MAX, int(args.grid_n), dtype=np.float64)

    os.makedirs(args.out_dir, exist_ok=True)
    token = dt_token(dt)
    mode_suffix = "" if args.mode == "broad" else f"_{args.mode}"
    win_token = str(float(args.window_years)).replace(".", "p")
    out_file = os.path.join(args.out_dir, f"{args.prefix}{mode_suffix}_dt{token}_w{win_token}_batch{batch_id:03d}.npz")
    summary_file = os.path.join(args.out_dir, f"{args.prefix}{mode_suffix}_dt{token}_w{win_token}_batch{batch_id:03d}_summary.txt")

    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_file}\nUse --overwrite if replacing it intentionally.")

    mode_offset = {"broad": 0, "strong_approach": 37_000_000, "mixed": 73_000_000}[args.mode]
    seed = int(args.seed) + batch_id * 100000 + int(round(dt * 1_000_000)) + mode_offset + int(round(float(args.window_years) * 1000))
    rng = np.random.RandomState(seed)

    required_fields = [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "v_rad_norm", "v_tan_norm",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]
    arr = {k: [] for k in required_fields}
    extra = {k: [] for k in [
        "c_window_raw", "log_c_window_raw", "window_loss_c1", "window_loss_opt",
        "window_years", "window_samples", "target_type_code", "mode_code",
    ]}

    rej = dict(ic=0, ias15=0, ref_eject=0, bad_c=0, boundary=0, bad_window=0)
    c_buf, imp_buf, vr_buf, vt_buf, loss1_buf, lossopt_buf = [], [], [], [], [], []
    n_identity = 0
    n_strong = 0

    print("=" * 78)
    print("ZONE 3 SHORT-WINDOW TRAJECTORY DATA GENERATOR v2")
    print(f"  dt             : {dt:.6f} yr")
    print(f"  window_years   : {float(args.window_years):.4f} yr")
    print(f"  window_samples : {int(args.window_samples)}")
    print(f"  target n       : {n_target}")
    print(f"  batch          : {batch_id}")
    print(f"  mode           : {args.mode}")
    print(f"  c bounds/grid  : [{C_MIN}, {C_MAX}] grid_n={int(args.grid_n)}")
    print(f"  identity_tol   : {float(args.identity_tol):.2%}")
    print(f"  seed           : {seed}")
    print(f"  output         : {out_file}")
    print(f"  isolate_ias15  : {bool(args.isolate_ias15)}")
    print("  NOTE: c_opt in this shard means c_window_opt, not old one-step c_opt.")
    print("=" * 78)

    t0 = time.perf_counter()
    n_ok = 0
    n_try = 0

    while n_ok < n_target and n_try < max_tries:
        n_try += 1
        ic_mode = args.mode
        if ic_mode == "mixed":
            ic_mode = "strong_approach" if rng.rand() < 0.50 else "broad"
        if ic_mode == "strong_approach":
            n_strong += 1

        ic = make_ic(rng, mode=ic_mode)
        if ic is None:
            rej["ic"] += 1
            continue

        x0, v0, m, r01, v_rad_norm, v_tan_norm = ic

        if args.verbose_attempts:
            print(f"try={n_try} ok={n_ok} mode={ic_mode} r={r01:.4f} vr={v_rad_norm:+.3f} vt={v_tan_norm:.3f}", flush=True)

        try:
            ref_times, ref_pos = ias15_window_safe(
                x0, v0, m, float(args.window_years), int(args.window_samples),
                timeout=float(args.timeout), isolate=bool(args.isolate_ias15)
            )
        except Exception as e:
            if args.verbose_attempts:
                print(f"  IAS15 EXCP {type(e).__name__}: {e}", flush=True)
            rej["ias15"] += 1
            continue

        if safe_max_norm_3x3(ref_pos[-1]) > 100.0:
            rej["ref_eject"] += 1
            continue

        try:
            c_raw, loss_opt, loss_c1 = find_c_window_opt(
                x0, v0, m, dt, float(args.window_years), ref_times, ref_pos,
                c_grid, int(args.max_substeps)
            )
        except Exception as e:
            if args.verbose_attempts:
                print(f"  FIND WINDOW C EXCP {type(e).__name__}: {e}", flush=True)
            rej["bad_c"] += 1
            continue

        if not (math.isfinite(c_raw) and math.isfinite(loss_opt) and math.isfinite(loss_c1)):
            rej["bad_c"] += 1
            continue
        if loss_c1 <= 1e-20:
            rej["bad_window"] += 1
            continue
        if c_raw <= C_MIN + 0.01 or c_raw >= C_MAX - 0.01:
            # Reject boundary optima because they usually mean the target is not
            # well bracketed or the short-window problem is too noisy.
            rej["boundary"] += 1
            continue

        imp_raw = float((loss_c1 - loss_opt) / loss_c1)
        if imp_raw < float(args.identity_tol):
            c_store = 1.0
            log_c_store = 0.0
            imp_store = 0.0
            n_identity += 1
        else:
            c_store = float(c_raw)
            log_c_store = float(np.log(c_store))
            imp_store = imp_raw

        r_soft = float(np.sqrt(r01 * r01 + EPS * EPS))
        c_ana = float((r_soft / r01) ** 3)

        arr["r_AU"].append(r01)
        arr["r_soft"].append(r_soft)
        arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
        arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
        arr["log_dt"].append(float(np.log(dt)))
        arr["v_rad_norm"].append(v_rad_norm)
        arr["v_tan_norm"].append(v_tan_norm)
        arr["c_opt"].append(c_store)
        arr["log_c_opt"].append(log_c_store)
        arr["c_ana"].append(c_ana)
        arr["log_c_ana"].append(float(np.log(c_ana)))
        arr["improvement"].append(imp_store)

        extra["c_window_raw"].append(float(c_raw))
        extra["log_c_window_raw"].append(float(np.log(c_raw)))
        extra["window_loss_c1"].append(float(loss_c1))
        extra["window_loss_opt"].append(float(loss_opt))
        extra["window_years"].append(float(args.window_years))
        extra["window_samples"].append(float(args.window_samples))
        extra["target_type_code"].append(1.0)  # 1 = window target
        extra["mode_code"].append(1.0 if ic_mode == "strong_approach" else 0.0)

        c_buf.append(c_store)
        imp_buf.append(imp_store)
        vr_buf.append(v_rad_norm)
        vt_buf.append(v_tan_norm)
        loss1_buf.append(loss_c1)
        lossopt_buf.append(loss_opt)
        n_ok += 1

        if n_ok == 1 or n_ok == n_target or n_ok % max(1, min(25, n_target // 5)) == 0:
            el = time.perf_counter() - t0
            print(f"  accepted {n_ok:>5}/{n_target} | tried={n_try:>6} | "
                  f"c_med={np.median(c_buf):.4f} | impr_med={np.median(imp_buf):.2%} | "
                  f"vr_med={np.median(vr_buf):+.3f} | vt_med={np.median(vt_buf):.3f} | {el:.0f}s",
                  flush=True)

    elapsed = time.perf_counter() - t0
    save = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    for k, v in extra.items():
        save[k] = np.array(v, dtype=np.float32)

    np.savez_compressed(out_file, **save)
    size_kb = os.path.getsize(out_file) / 1024.0

    if n_ok > 0:
        line = (f"dt={dt:.6f}: n={n_ok} | tried={n_try} | pass={n_ok/max(n_try,1):.1%} | "
                f"window={float(args.window_years):.3f}yr | c_med={float(np.median(c_buf)):.5f} | "
                f"impr_med={float(np.median(imp_buf)):.2%} | identity={n_identity} | "
                f"loss_c1_med={float(np.median(loss1_buf)):.6e} | "
                f"loss_opt_med={float(np.median(lossopt_buf)):.6e} | "
                f"vr_med={float(np.median(vr_buf)):+.5f} | vt_med={float(np.median(vt_buf)):.5f} | {elapsed:.0f}s")
    else:
        line = f"dt={dt:.6f}: n=0 | tried={n_try} | pass=0.0% | NO ACCEPTED SAMPLES | {elapsed:.0f}s"

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("ZONE 3 SHORT-WINDOW TRAJECTORY DATASET SUMMARY v2\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"dt             : {dt:.6f}\n")
        fh.write(f"window_years   : {float(args.window_years):.6f}\n")
        fh.write(f"window_samples : {int(args.window_samples)}\n")
        fh.write(f"batch          : {batch_id}\n")
        fh.write(f"mode           : {args.mode}\n")
        fh.write(f"seed           : {seed}\n")
        fh.write(f"target         : {n_target}\n")
        fh.write(f"accepted       : {n_ok}\n")
        fh.write(f"tried          : {n_try}\n")
        fh.write(f"elapsed_sec    : {elapsed:.1f}\n")
        fh.write(f"output         : {out_file}\n")
        fh.write(f"isolate_ias15  : {bool(args.isolate_ias15)}\n")
        fh.write(f"size_kb        : {size_kb:.1f}\n")
        fh.write("zone           : r 0.052-0.148 AU\n")
        fh.write("target meaning : c_opt/log_c_opt are c_window_opt/log(c_window_opt)\n")
        fh.write("window method  : trial c applies only to pair (0,1) on the first macro-step; later Zone 3 uses c=1 baseline\n")
        fh.write("required fields: " + " ".join(required_fields) + "\n\n")
        fh.write(line + "\n")
        fh.write(f"Rejected       : {rej}\n")
        fh.write(f"Near-identity stored samples: {n_identity}\n")
        fh.write(f"Strong-attempt counter: {n_strong}\n\n")
        fh.write("Compatibility note:\n")
        fh.write("  This shard can be merged by merge_encounter_shards.py because all required v3 fields are present.\n")
        fh.write("  Your current merger will ignore extra fields such as window_loss_c1 and c_window_raw.\n")

    print("\n" + line)
    print(f"Rejected: {rej}")
    print(f"Near-identity stored samples: {n_identity}")
    print(f"Saved {out_file} ({n_ok} samples, {size_kb:.1f} KB)")
    print(f"Saved {summary_file}")
    if n_ok < n_target:
        print(f"WARNING: saved partial batch only: {n_ok}/{n_target} accepted before max_tries={max_tries}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
