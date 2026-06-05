"""
generate_encounter_data_OptionA_batch.py  --  Zone 3 trajectory data, batch-safe v3

Purpose
-------
Generates ONE small batch of Zone 3 training data for the trajectory-optimal
scalar correction c_opt. This version is designed for Windows machines that
crash or become unstable on large runs: run one dt and one batch at a time.

Example runs:
    python -B generate_encounter_data_OptionA_batch.py --dt 0.005 --n 100 --batch 2
    python -B generate_encounter_data_OptionA_batch.py --dt 0.010 --n 100 --batch 2
    python -B generate_encounter_data_OptionA_batch.py --dt 0.020 --n 100 --batch 2

What is new in v3
-----------------
The previous Zone 3 dataset saved only:
    [r_soft, log_mi, log_mj, log_dt] -> log_c_opt

But c_opt is obtained by matching one-step leapfrog position against ias15.
That one-step position depends on the initial velocity. Therefore this version
also saves two velocity-state features for pair (0,1):

    v_rad_norm = dot(v_rel, r_hat) / sqrt(G*(m0+m1)/r)
    v_tan_norm = ||v_rel - v_rad*r_hat|| / sqrt(G*(m0+m1)/r)

Interpretation:
    v_rad_norm < 0  : bodies are approaching
    v_rad_norm > 0  : bodies are receding
    v_tan_norm      : tangential/orbital component of the encounter

Output fields in each batch npz:
    r_AU, r_soft, log_mi, log_mj, log_dt,
    v_rad_norm, v_tan_norm,
    c_opt, log_c_opt, c_ana, log_c_ana, improvement

Method
------
For each accepted synthetic Zone 3 initial condition:
  1. Run ias15 from t to t+dt to obtain x_ref.
  2. Run one leapfrog step from the same x0, v0 for many scalar c values.
  3. Pick c_opt that minimises MSE(x_leapfrog(c), x_ref).

Zone 3 only:
    r in (0.052, 0.148) AU
    adaptive sub-stepping is not active in this region
    this is the macro-step close-encounter region
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
# Constants — keep matched with evaluator configuration
# =============================================================================
G            = 1.0
EPS          = 3e-4
R_SOFT_MIN   = 5e-4
NN_THRESH    = 500.0 * EPS      # 0.15 AU
ADAPT_THRESH = 0.05             # Zone 2 / Zone 3 boundary
C_MIN, C_MAX = 0.2, 5.0

R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))  # 4e-4 AU

# Zone 3 r boundaries — macro-step zone, NN active, no sub-stepping
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU

SEED = 42
N_GRID = 60
C_GRID = np.linspace(C_MIN, C_MAX, N_GRID)
IAS15_TIMEOUT_SEC = 60.0


# =============================================================================
# Utility helpers
# =============================================================================
def dt_token(dt):
    """Safe token for filenames: 0.005 -> 0p005."""
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _mse(a, b):
    """
    Ultra-safe mean squared error between two (3,3) position arrays.
    Avoids NumPy reductions inside the hot bad-state path.
    """
    try:
        if a is None or b is None:
            return float("inf")
        if len(a) != len(b):
            return float("inf")

        total = 0.0
        count = 0
        for i in range(len(a)):
            if len(a[i]) != len(b[i]):
                return float("inf")
            for j in range(len(a[i])):
                av = float(a[i][j])
                bv = float(b[i][j])
                if not math.isfinite(av) or not math.isfinite(bv):
                    return float("inf")
                if abs(av) > 1e6 or abs(bv) > 1e6:
                    return float("inf")
                d = av - bv
                total += d * d
                count += 1

        if count == 0:
            return float("inf")
        val = total / count
        return float(val) if math.isfinite(val) else float("inf")
    except Exception:
        return float("inf")


def safe_max_norm_3x3(x):
    """Small safe helper avoiding np.max on questionable arrays."""
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


# =============================================================================
# IAS15 one-step reference, isolated in subprocess for Windows crash safety
# =============================================================================
def ias15_one_step(x0, v0, m_arr, dt):
    """
    Advance a three-body system from t=0 to t=dt using ias15.
    Returns x_new, v_new.

    sim.exit_min_distance = EPS makes extremely close singular encounters raise
    a Python exception instead of risking a C-level crash.
    """
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    sim.exit_min_distance = float(EPS)

    for i in range(3):
        sim.add(m=float(m_arr[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))

    sim.move_to_com()
    sim.integrate(float(dt))

    x_new = np.array([[p.x,  p.y,  p.z]  for p in sim.particles], dtype=np.float64)
    v_new = np.array([[p.vx, p.vy, p.vz] for p in sim.particles], dtype=np.float64)

    if safe_max_norm_3x3(x_new) == float("inf"):
        raise ValueError("ias15 returned non-finite positions")

    return x_new, v_new


def _ias15_worker(conn, x0, v0, m_arr, dt):
    """Run one IAS15 step inside a fresh child process."""
    try:
        x_new, v_new = ias15_one_step(x0, v0, m_arr, dt)
        conn.send(("ok", x_new, v_new, ""))
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


def ias15_one_step_isolated(x0, v0, m_arr, dt, timeout=IAS15_TIMEOUT_SEC):
    """
    Option A crash fix: run every IAS15 one-step reference solve in a fresh
    subprocess. If the child crashes, this parent rejects the IC and continues.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_ias15_worker,
                    args=(child_conn, x0, v0, m_arr, float(dt)))
    p.start()
    child_conn.close()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join(5.0)
        parent_conn.close()
        raise TimeoutError(f"ias15 subprocess timed out after {timeout:.1f}s")

    if p.exitcode != 0:
        parent_conn.close()
        raise RuntimeError(f"ias15 subprocess exited with code {p.exitcode}")

    if not parent_conn.poll():
        parent_conn.close()
        raise RuntimeError("ias15 subprocess returned no result")

    status, x_new, v_new, msg = parent_conn.recv()
    parent_conn.close()

    if status != "ok":
        raise RuntimeError(msg)

    return x_new, v_new


# =============================================================================
# Leapfrog with fixed scalar c
# =============================================================================
def acc_fixed_c(pos, m_arr, c_val):
    """
    Accelerations for three bodies.
    Pair (0,1): softened force × c_val when r < NN_THRESH.
    Pairs (0,2), (1,2): exact Newtonian.
    """
    acc = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS

    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        rij = pos[j] - pos[i]
        r2 = float(np.dot(rij, rij))
        r = float((r2 + 1e-30) ** 0.5)
        Gmimj = G * float(m_arr[i]) * float(m_arr[j])

        if idx == 0 and r < NN_THRESH:
            denom = (r2 + eps2) ** 1.5 + 1e-30
            F_scalar = float(c_val) * Gmimj / denom
        else:
            F_scalar = Gmimj / (r2 * r + 1e-30)

        F_vec = F_scalar * rij
        acc[i] +=  F_vec / float(m_arr[i])
        acc[j] -=  F_vec / float(m_arr[j])

    return acc


def leapfrog_step(x0, v0, m_arr, dt, c_val):
    """One velocity-Verlet / leapfrog step with fixed scalar correction c_val."""
    a0 = acc_fixed_c(x0, m_arr, c_val)
    v_half = v0 + 0.5 * float(dt) * a0
    x1 = x0 + float(dt) * v_half
    a1 = acc_fixed_c(x1, m_arr, c_val)
    v1 = v_half + 0.5 * float(dt) * a1
    return x1, v1


# =============================================================================
# c_optimal search
# =============================================================================
def find_c_opt(x0, v0, m_arr, dt, x_ref):
    """
    Grid search + golden-section refinement for c in [C_MIN, C_MAX] that
    minimises MSE(x_leapfrog(c), x_ref) after one macro-step dt.
    Returns (c_opt, mse_opt, mse_at_c1).
    """
    x1, _ = leapfrog_step(x0, v0, m_arr, dt, 1.0)
    mse_1 = _mse(x1, x_ref)

    mse_g = []
    for c in C_GRID:
        xk, _ = leapfrog_step(x0, v0, m_arr, dt, float(c))
        mse_g.append(_mse(xk, x_ref))

    finite_idx = [i for i, val in enumerate(mse_g) if math.isfinite(val)]
    if not finite_idx:
        return np.nan, float("inf"), mse_1

    best_k = min(finite_idx, key=lambda i: mse_g[i])
    c_lo = float(C_GRID[max(0, best_k - 1)])
    c_hi = float(C_GRID[min(N_GRID - 1, best_k + 1)])

    phi = (float(np.sqrt(5.0)) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7:
            break

        c1 = c_hi - phi * (c_hi - c_lo)
        c2 = c_lo + phi * (c_hi - c_lo)

        xa, _ = leapfrog_step(x0, v0, m_arr, dt, c1)
        xb, _ = leapfrog_step(x0, v0, m_arr, dt, c2)

        m1 = _mse(xa, x_ref)
        m2 = _mse(xb, x_ref)
        if not math.isfinite(m1):
            m1 = float("inf")
        if not math.isfinite(m2):
            m2 = float("inf")

        if m1 < m2:
            c_hi = c2
        else:
            c_lo = c1

    c_opt = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step(x0, v0, m_arr, dt, c_opt)
    mse_opt = _mse(x_opt, x_ref)
    return c_opt, mse_opt, mse_1


# =============================================================================
# IC generator and velocity features
# =============================================================================
def compute_pair_velocity_features(x_all, v_all, m_arr):
    """
    Compute pair (0,1) velocity-state features for the NN.

    v_rad_norm is signed:
        negative = approaching
        positive = receding
    v_tan_norm is non-negative.
    """
    r_vec = x_all[1] - x_all[0]
    v_vec = v_all[1] - v_all[0]

    r = float(np.linalg.norm(r_vec))
    r_hat = r_vec / (r + 1e-30)

    v_rad = float(np.dot(v_vec, r_hat))
    v_tan_vec = v_vec - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))

    v_scale = float(np.sqrt(G * (float(m_arr[0]) + float(m_arr[1])) / (r + 1e-30)))

    v_rad_norm = v_rad / (v_scale + 1e-30)
    v_tan_norm = v_tan / (v_scale + 1e-30)

    return float(v_rad_norm), float(v_tan_norm)


def make_ic(rng):
    """
    Synthetic Zone 3 three-body starting state.
    Pair (0,1): r01 in (0.052, 0.148) AU.
    Body 2: far away to isolate the close-pair signal.

    Returns:
        x_all, v_all, m, r01_check, v_rad_norm, v_tan_norm
    or None if invalid.
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m = np.array([m0, m1, m2], dtype=np.float64)

    r01 = float(rng.uniform(Z3_R_MIN, Z3_R_MAX))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)

    # Relative velocity of the close pair. Direction is deliberately random;
    # the saved v_rad_norm/v_tan_norm features tell the NN which case it is.
    f = float(rng.uniform(0.4, 1.8))
    v_circ = float(np.sqrt(G * (m0 + m1) / r01))
    phi_v = float(rng.uniform(0.0, 2.0 * np.pi))
    v_hat = np.array([np.cos(phi_v), np.sin(phi_v), 0.0], dtype=np.float64)
    v_rel = f * v_circ * v_hat

    x_rel = r01 * r_hat
    x0p = -(m1 / (m0 + m1)) * x_rel
    x1p =  (m0 / (m0 + m1)) * x_rel
    v0p = -(m1 / (m0 + m1)) * v_rel
    v1p =  (m0 / (m0 + m1)) * v_rel

    # Distant third body
    r2 = float(rng.uniform(3.0, 8.0))
    th2 = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p = r2 * np.array([np.cos(th2), np.sin(th2), 0.0], dtype=np.float64)
    f2 = float(rng.uniform(0.5, 1.1))
    vc2 = float(np.sqrt(G * (m0 + m1) / r2))
    v2p = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0], dtype=np.float64)

    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)

    # Centre-of-mass frame. Relative pair position/velocity are unchanged.
    M_tot = float(m.sum())
    x_com = np.sum(m[:, None] * x_all, axis=0) / M_tot
    v_com = np.sum(m[:, None] * v_all, axis=0) / M_tot
    x_all = x_all - x_com
    v_all = v_all - v_com

    # Reject globally unbound systems.
    KE = 0.5 * float(np.sum(m * np.array([
        float(np.dot(v_all[i], v_all[i])) for i in range(3)
    ], dtype=np.float64)))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            rij = float(np.linalg.norm(x_all[i] - x_all[j]))
            PE -= G * m[i] * m[j] / (rij + 1e-30)
    if KE + PE >= 0.0:
        return None

    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not (Z3_R_MIN < r01_check < Z3_R_MAX):
        return None

    # Reject near-radial singular encounters likely to hit r < EPS in IAS15.
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

    # Keep the third body out of Zone 3; it should not create the training signal.
    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    v_rad_norm, v_tan_norm = compute_pair_velocity_features(x_all, v_all, m)
    if not (math.isfinite(v_rad_norm) and math.isfinite(v_tan_norm)):
        return None

    return x_all, v_all, m, r01_check, v_rad_norm, v_tan_norm


# =============================================================================
# Main batch generation
# =============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Generate one batch of Zone 3 trajectory-correction data with velocity features."
    )
    ap.add_argument("--dt", type=float, required=True,
                    help="One macro timestep to generate, e.g. 0.005")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of accepted samples to generate in this batch")
    ap.add_argument("--batch", type=int, default=0,
                    help="Batch id used for seed and filename")
    
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir",
                    default="encounter_shards",
                    help="Folder where shard .npz and summary .txt files are saved")
    
    ap.add_argument("--prefix", default="encounter_data_zone3_v3",
                    help="Output filename prefix")
    ap.add_argument("--timeout", type=float, default=IAS15_TIMEOUT_SEC,
                    help="IAS15 subprocess timeout per trial, seconds")
    ap.add_argument("--max_tries", type=int, default=0,
                    help="Maximum attempts before saving partial output. 0 means n*1000.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Base random seed")
    ap.add_argument("--verbose_attempts", action="store_true",
                    help="Print every IAS15 and c_opt attempt. Useful for debugging crashes, noisy for large runs.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing shard file if present.")
    return ap


def main():
    args = build_arg_parser().parse_args()

    dt = float(args.dt)
    n_target = int(args.n)
    batch_id = int(args.batch)
    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else max(1000 * n_target, 1000)

    os.makedirs(args.out_dir, exist_ok=True)
    token = dt_token(dt)
    out_file = os.path.join(args.out_dir, f"{args.prefix}_dt{token}_batch{batch_id:03d}.npz")
    summary_file = os.path.join(args.out_dir, f"{args.prefix}_dt{token}_batch{batch_id:03d}_summary.txt")

    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_file}\n"
            f"Use --overwrite if you intentionally want to replace it."
        )

    
    # Deterministic but different seed for every dt/batch.
    seed = int(args.seed) + batch_id * 100000 + int(round(dt * 1_000_000))
    rng = np.random.RandomState(seed)

    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "v_rad_norm", "v_tan_norm",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]}

    rej = dict(ic=0, ias15=0, boundary=0, eject=0, no_improve=0, bad_c=0)
    c_buf, imp_buf, c_ana_buf = [], [], []
    vr_buf, vt_buf = [], []
    n_identity = 0

    print("=" * 72)
    print("ZONE 3 TRAJECTORY DATA GENERATOR — BATCH v3")
    print(f"  dt       : {dt:.6f} yr")
    print(f"  target n : {n_target}")
    print(f"  batch    : {batch_id}")
    print(f"  seed     : {seed}")
    print(f"  r zone   : ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU")
    print(f"  output   : {out_file}")
    print("  fields   : r_AU r_soft log_mi log_mj log_dt v_rad_norm v_tan_norm")
    print("             c_opt log_c_opt c_ana log_c_ana improvement")
    print("=" * 72)

    t0 = time.perf_counter()
    n_ok = 0
    n_try = 0

    while n_ok < n_target and n_try < max_tries:
        n_try += 1

        ic = make_ic(rng)
        if ic is None:
            rej["ic"] += 1
            continue

        x0, v0, m, r01, v_rad_norm, v_tan_norm = ic

        if args.verbose_attempts:
            print(f"IAS15 START try={n_try} ok={n_ok} r01={r01:.4f} "
                  f"vr={v_rad_norm:+.3f} vt={v_tan_norm:.3f} "
                  f"m0={m[0]:.4f} m1={m[1]:.4f} dt={dt:.4f}", flush=True)

        try:
            x_ref, _ = ias15_one_step_isolated(x0, v0, m, dt, timeout=float(args.timeout))
            if args.verbose_attempts:
                print(f"IAS15 DONE  try={n_try}", flush=True)
        except Exception as e:
            if args.verbose_attempts:
                print(f"IAS15 EXCP  try={n_try} {type(e).__name__}: {e}", flush=True)
            rej["ias15"] += 1
            continue

        if safe_max_norm_3x3(x_ref) > 50.0:
            rej["eject"] += 1
            continue

        if args.verbose_attempts:
            print(f"FINDCOPT START try={n_try}", flush=True)

        try:
            result = find_c_opt(x0, v0, m, dt, x_ref)
            if not isinstance(result, tuple) or len(result) != 3:
                rej["bad_c"] += 1
                continue
            c_opt, mse_opt, mse_1 = result
            c_opt = float(c_opt)
            mse_opt = float(mse_opt)
            mse_1 = float(mse_1)
        except Exception as e:
            if args.verbose_attempts:
                print(f"FIND_C EXCP try={n_try} {type(e).__name__}: {e}", flush=True)
            rej["bad_c"] += 1
            continue

        if not (math.isfinite(c_opt) and math.isfinite(mse_opt) and math.isfinite(mse_1)):
            rej["bad_c"] += 1
            continue

        if args.verbose_attempts:
            print(f"FINDCOPT DONE try={n_try} c_opt={c_opt:.4f}", flush=True)

        if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
            rej["boundary"] += 1
            continue

        x_best, _ = leapfrog_step(x0, v0, m, dt, c_opt)
        if safe_max_norm_3x3(x_best) > 50.0:
            rej["eject"] += 1
            continue

        if mse_1 < 1e-20:
            rej["no_improve"] += 1
            continue

        # Improvement from the best searched c relative to c=1.
        # If the improvement is less than 5%, keep the sample but store
        # a near-identity target. This teaches the NN that some Zone 3
        # states should be left essentially unchanged.
        imp_raw = float((mse_1 - mse_opt) / mse_1)

        if imp_raw < 0.05:
            c_store = 1.0
            log_c_store = 0.0
            imp = 0.0
            n_identity += 1
        else:
            c_store = c_opt
            log_c_store = float(np.log(c_opt))
            imp = imp_raw

        # Accept sample.
        r_soft = float(np.sqrt(r01**2 + EPS**2))
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
        arr["improvement"].append(imp)

        c_buf.append(c_store)

        imp_buf.append(imp)
        c_ana_buf.append(c_ana)
        vr_buf.append(v_rad_norm)
        vt_buf.append(v_tan_norm)
        n_ok += 1

        if n_ok == 1 or n_ok == n_target or n_ok % max(1, min(50, n_target // 10)) == 0:
            el = time.perf_counter() - t0
            print(f"  accepted {n_ok:>5}/{n_target} | tried={n_try:>6} | "
                  f"c_med={np.median(c_buf):.4f} | "
                  f"vr_med={np.median(vr_buf):+.3f} | "
                  f"vt_med={np.median(vt_buf):.3f} | {el:.0f}s",
                  flush=True)

    elapsed = time.perf_counter() - t0

    save = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(out_file, **save)
    size_kb = os.path.getsize(out_file) / 1024.0

    if n_ok > 0:
        line = (f"dt={dt:.6f}: n={n_ok} | tried={n_try} | pass={n_ok/max(n_try,1):.1%} | "
                f"c_opt_med={float(np.median(c_buf)):.5f} | "
                f"c_ana_med={float(np.median(c_ana_buf)):.6f} | "
                
                f"impr_med={float(np.median(imp_buf)):.2%} | "
                f"identity={n_identity} | "
                f"v_rad_norm_med={float(np.median(vr_buf)):+.5f} | "
                f"v_tan_norm_med={float(np.median(vt_buf)):.5f} | {elapsed:.0f}s")
        
    else:
        line = (f"dt={dt:.6f}: n=0 | tried={n_try} | pass=0.0% | "
                f"NO ACCEPTED SAMPLES | {elapsed:.0f}s")

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("ZONE 3 TRAJECTORY DATASET SUMMARY — BATCH v3\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"dt      : {dt:.6f}\n")
        fh.write(f"batch   : {batch_id}\n")
        fh.write(f"seed    : {seed}\n")
        fh.write(f"target  : {n_target}\n")
        fh.write(f"accepted: {n_ok}\n")
        fh.write(f"tried   : {n_try}\n")
        fh.write(f"elapsed : {elapsed:.1f} sec\n")
        fh.write(f"output  : {out_file}\n")
        fh.write(f"size_kb : {size_kb:.1f}\n")
        fh.write(f"zone    : r {Z3_R_MIN:.3f}-{Z3_R_MAX:.3f} AU\n")
        fh.write("fields  : r_AU r_soft log_mi log_mj log_dt v_rad_norm v_tan_norm "
                 "c_opt log_c_opt c_ana log_c_ana improvement\n\n")
        fh.write(line + "\n")
        
        fh.write(f"Rejected: {rej}\n")
        fh.write(f"Near-identity stored samples: {n_identity}\n\n")
        fh.write("Velocity feature definitions:\n")

        fh.write("  v_rad_norm = dot(v_rel, r_hat) / sqrt(G*(m0+m1)/r)\n")
        fh.write("  v_tan_norm = ||v_rel - v_rad*r_hat|| / sqrt(G*(m0+m1)/r)\n")
        fh.write("  v_rad_norm < 0 means approaching; > 0 means receding.\n")

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
