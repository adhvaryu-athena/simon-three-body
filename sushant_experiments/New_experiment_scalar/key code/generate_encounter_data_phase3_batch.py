"""
generate_encounter_data_phase3_batch.py  --  Zone 3 trajectory data, Phase 3

Phase 3 improvements over Phase 2 generator:
-----------------------------------------------
1. dt=0.005 and dt=0.01 REMOVED. No training signal there; NN fires near-identity
   and adds chaos noise. Valid dt: 0.02, 0.04, 0.05, 0.06, 0.08, 0.10.

2. Improved make_ic (broad mode) using Keplerian velocity sampling.
   Phase 2 sampled velocity direction uniformly in [0, 2pi] with magnitude
   f * v_circ. This produced unrealistic (v_rad, v_tan) pairs with no orbital
   physics. Phase 3 samples eccentricity e and true anomaly f and derives
   (v_rad, v_tan) from Keplerian orbit mechanics. This matches the actual
   velocity correlations seen during live 3-body deployment.

3. Body 2 placed at r=1.0-6.0 AU (Phase 2 used 3-8 AU). In real deployment
   the third body can be at intermediate distances; sampling it closer makes
   the training distribution more representative.

4. New modes:
   - broad        : Keplerian sampling (default, replaces Phase 2 broad)
   - strong_recede: targets v_rad_norm in [+0.4, +1.2], r in [0.052, 0.085 AU]
                    Phase 2 under-represented recede at large dt.
   - probe_fix    : targets r=[0.088, 0.112 AU], v_rad_norm in [-1.2, -0.3],
                    v_tan_norm in [0.40, 0.90]. Fixes the diagnostic OFF_MANIFOLD
                    failure at approach dt=0.05-0.10.

Usage examples:
    # Broad batches (1 dt, 1 batch at a time)
    python -B generate_encounter_data_phase3_batch.py --dt 0.04 --n 100 --batch 1 --mode broad
    python -B generate_encounter_data_phase3_batch.py --dt 0.08 --n 100 --batch 1 --mode broad

    # Strong recede batches
    python -B generate_encounter_data_phase3_batch.py --dt 0.04 --n 100 --batch 1 --mode strong_recede
    python -B generate_encounter_data_phase3_batch.py --dt 0.08 --n 100 --batch 1 --mode strong_recede

    # Probe fix batches (only valid for dt >= 0.05)
    python -B generate_encounter_data_phase3_batch.py --dt 0.05 --n 100 --batch 1 --mode probe_fix
    python -B generate_encounter_data_phase3_batch.py --dt 0.08 --n 100 --batch 1 --mode probe_fix

Output fields (same format as Phase 2 for compatibility with merge/inspect/trainer):
    r_AU, r_soft, log_mi, log_mj, log_dt,
    v_rad_norm, v_tan_norm,
    c_opt, log_c_opt, c_ana, log_c_ana, improvement
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
# Constants -- keep matched with evaluator configuration
# =============================================================================
G            = 1.0
EPS          = 3e-4
R_SOFT_MIN   = 5e-4
NN_THRESH    = 500.0 * EPS      # 0.15 AU
ADAPT_THRESH = 0.05             # Zone 2 / Zone 3 boundary
C_MIN, C_MAX = 0.2, 5.0

R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))  # 4e-4 AU

# Zone 3 r boundaries
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU

# Phase 3: only these dts are valid -- 0.005 and 0.01 removed
VALID_DTS = {0.020, 0.040, 0.050, 0.060, 0.080, 0.100}
# probe_fix only makes sense at dts where the probe failure was observed
PROBE_FIX_DTS = {0.040, 0.050, 0.060, 0.080, 0.100}

SEED = 42
N_GRID = 60
C_GRID = np.linspace(C_MIN, C_MAX, N_GRID)
IAS15_TIMEOUT_SEC = 60.0


# =============================================================================
# Utility helpers (unchanged from Phase 2)
# =============================================================================
def dt_token(dt):
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _mse(a, b):
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
# IAS15 one-step reference in isolated subprocess (unchanged from Phase 2)
# =============================================================================
def ias15_one_step(x0, v0, m_arr, dt):
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
# Leapfrog and c_opt search (unchanged from Phase 2)
# =============================================================================
def acc_fixed_c(pos, m_arr, c_val):
    """Pair (0,1): softened force * c_val in Zone 3. Others: exact Newtonian."""
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
    a0 = acc_fixed_c(x0, m_arr, c_val)
    v_half = v0 + 0.5 * float(dt) * a0
    x1 = x0 + float(dt) * v_half
    a1 = acc_fixed_c(x1, m_arr, c_val)
    v1 = v_half + 0.5 * float(dt) * a1
    return x1, v1


def find_c_opt(x0, v0, m_arr, dt, x_ref):
    """Grid + golden-section search for c minimising MSE(x_leapfrog(c), x_ref)."""
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
        if not math.isfinite(m1): m1 = float("inf")
        if not math.isfinite(m2): m2 = float("inf")
        if m1 < m2:
            c_hi = c2
        else:
            c_lo = c1
    c_opt = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step(x0, v0, m_arr, dt, c_opt)
    mse_opt = _mse(x_opt, x_ref)
    return c_opt, mse_opt, mse_1


# =============================================================================
# Velocity features (unchanged from Phase 2)
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


# =============================================================================
# Shared validation checks (used by all IC modes)
# =============================================================================
def _validate_ic(x_all, v_all, m, M_pair, r01_check):
    """Run all standard IC checks. Returns True if valid."""
    # Reject globally unbound systems
    KE = 0.5 * sum(float(m[i]) * float(np.dot(v_all[i], v_all[i])) for i in range(3))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            rij = float(np.linalg.norm(x_all[i] - x_all[j]))
            PE -= G * float(m[i]) * float(m[j]) / (rij + 1e-30)
    if KE + PE >= 0.0:
        return False

    if not (Z3_R_MIN < r01_check < Z3_R_MAX):
        return False

    # Reject near-radial singular encounters (periapsis too small)
    r_rel = x_all[1] - x_all[0]
    v_rel = v_all[1] - v_all[0]
    v_sq = float(np.dot(v_rel, v_rel))
    eps_orb = v_sq / 2.0 - G * M_pair / r01_check
    h_z = float(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0])
    h_sq = h_z * h_z
    disc = max(0.0, 1.0 + 2.0 * eps_orb * h_sq / (G * M_pair) ** 2)
    e_ecc = float(np.sqrt(disc))
    r_peri = h_sq / (G * M_pair * (1.0 + e_ecc) + 1e-30)
    if r_peri < EPS:
        return False

    # Third body must stay out of Zone 3 to avoid polluting the training signal
    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return False

    return True


def _com_center(x_all, v_all, m):
    M = m.sum()
    x_com = (m[:, None] * x_all).sum(0) / M
    v_com = (m[:, None] * v_all).sum(0) / M
    return x_all - x_com, v_all - v_com


# =============================================================================
# IC generators -- Phase 3
# =============================================================================

def make_ic_broad(rng):
    """
    Keplerian velocity sampling for broad coverage.

    Phase 2 sampled velocity direction uniformly in [0, 2pi]. This produces
    (v_rad, v_tan) pairs with no physical correlation. Phase 3 samples orbital
    eccentricity e and true anomaly f and derives v_rad, v_tan from:

        p     = r * (1 + e * cos(f))      semi-latus rectum
        h     = sqrt(G * M * p)            specific angular momentum
        v_rad = (G*M/h) * e * sin(f)      radial velocity
        v_tan = (G*M/h) * (1 + e*cos(f))  tangential speed

    This matches the actual velocity structure seen during orbital close encounters.
    Body 2 is placed at r=1.0-6.0 AU (Phase 2 used 3-8 AU).
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    M_pair = m0 + m1

    r01 = float(rng.uniform(Z3_R_MIN, Z3_R_MAX))

    # Orbital orientation in the xy-plane
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])
    t_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])

    # Keplerian orbit parameters
    e  = float(rng.uniform(0.0, 0.85))
    f  = float(rng.uniform(0.0, 2.0 * np.pi))

    # Semi-latus rectum from current position on orbit
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    p = r01 * (1.0 + e * cos_f)
    if p < 1e-6:
        return None

    # Specific angular momentum and velocity components
    h     = float(np.sqrt(G * M_pair * p + 1e-30))
    GMh   = G * M_pair / h
    v_rad = GMh * e * sin_f           # positive = receding
    v_tan = GMh * (1.0 + e * cos_f)   # always positive for e < 1, bound orbit

    if v_tan < 0.0:
        return None  # shouldn't happen; guard against numerics

    # Random prograde/retrograde
    t_sign = 1.0 if rng.rand() < 0.5 else -1.0
    v_rel  = v_rad * r_hat + v_tan * t_sign * t_hat

    x_rel = r01 * r_hat
    x0p   = -(m1 / M_pair) * x_rel
    x1p   =  (m0 / M_pair) * x_rel
    v0p   = -(m1 / M_pair) * v_rel
    v1p   =  (m0 / M_pair) * v_rel

    # Third body: 1.0-6.0 AU (closer than Phase 2's 3-8 AU)
    r2   = float(rng.uniform(1.0, 6.0))
    th2  = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p  = r2 * np.array([np.cos(th2), np.sin(th2), 0.0])
    f2   = float(rng.uniform(0.5, 1.1))
    vc2  = float(np.sqrt(G * M_pair / r2))
    v2p  = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0])

    m     = np.array([m0, m1, m2], dtype=np.float64)
    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)
    x_all, v_all = _com_center(x_all, v_all, m)

    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not _validate_ic(x_all, v_all, m, M_pair, r01_check):
        return None

    vr, vt = compute_pair_velocity_features(x_all, v_all, m)
    if not (math.isfinite(vr) and math.isfinite(vt)):
        return None

    return x_all, v_all, m, r01_check, vr, vt


def make_ic_strong_recede(rng):
    """
    Targeted sampling: receding pairs at large-dt range.

    Phase 2 had systematic under-prediction of the recede correction:
    model predicted c~0.83 when true was ~0.79. This mode generates
    receding encounters in the deep Zone 3 range to balance the dataset.

    Targets:
        v_rad_norm in [+0.40, +1.20]   (receding)
        v_tan_norm in [0.30,  1.00]
        r          in [0.052, 0.085 AU] (deep Zone 3)
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    M_pair = m0 + m1

    # Deep Zone 3, receding
    r01 = float(rng.uniform(Z3_R_MIN, 0.085))
    v_scale = float(np.sqrt(G * M_pair / (r01 + 1e-30)))

    # Sample target velocity directly
    target_vr_norm = float(rng.uniform(0.80, 1.60))
    target_vt_norm = float(rng.uniform(0.30, 1.00))
    v_rad_raw = target_vr_norm * v_scale   # positive = receding
    v_tan_raw = target_vt_norm * v_scale

    theta  = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat  = np.array([np.cos(theta), np.sin(theta), 0.0])
    t_hat  = np.array([-np.sin(theta), np.cos(theta), 0.0])
    t_sign = 1.0 if rng.rand() < 0.5 else -1.0
    v_rel  = v_rad_raw * r_hat + v_tan_raw * t_sign * t_hat

    x_rel = r01 * r_hat
    x0p   = -(m1 / M_pair) * x_rel
    x1p   =  (m0 / M_pair) * x_rel
    v0p   = -(m1 / M_pair) * v_rel
    v1p   =  (m0 / M_pair) * v_rel

    r2   = float(rng.uniform(1.0, 6.0))
    th2  = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p  = r2 * np.array([np.cos(th2), np.sin(th2), 0.0])
    f2   = float(rng.uniform(0.5, 1.1))
    vc2  = float(np.sqrt(G * M_pair / r2))
    v2p  = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0])

    m     = np.array([m0, m1, m2], dtype=np.float64)
    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)
    x_all, v_all = _com_center(x_all, v_all, m)

    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not _validate_ic(x_all, v_all, m, M_pair, r01_check):
        return None

    vr, vt = compute_pair_velocity_features(x_all, v_all, m)
    if not (math.isfinite(vr) and math.isfinite(vt)):
        return None

    # Verify we got a genuinely receding sample after COM centering
    if vr < 0.15:
        return None

    return x_all, v_all, m, r01_check, vr, vt


def make_ic_probe_fix(rng):
    """
    Targeted sampling: approaching pairs in the probe-failure region.

    The Phase 2 diagnostic showed c=0.388 for approach at dt=0.06/0.08
    (should be ~1.10). The NN was OFF_MANIFOLD at r=0.10 AU, v_rad=-0.8,
    v_tan=0.65, dt>=0.05. This mode fills that exact gap.

    Targets:
        r          in [0.088, 0.112 AU]  (around 0.10 AU)
        v_rad_norm in [-1.20, -0.30]     (approaching)
        v_tan_norm in [0.40,   0.90]
    Only valid for dt >= 0.05 (where the failure was observed).
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    M_pair = m0 + m1

    # Probe failure region
    r01 = float(rng.uniform(0.088, 0.112))
    v_scale = float(np.sqrt(G * M_pair / (r01 + 1e-30)))

    # Approaching with moderate tangential velocity
    target_vr_norm = float(rng.uniform(-1.20, -0.30))
    target_vt_norm = float(rng.uniform( 0.40,  0.90))
    v_rad_raw = target_vr_norm * v_scale   # negative = approaching
    v_tan_raw = target_vt_norm * v_scale

    theta  = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat  = np.array([np.cos(theta), np.sin(theta), 0.0])
    t_hat  = np.array([-np.sin(theta), np.cos(theta), 0.0])
    t_sign = 1.0 if rng.rand() < 0.5 else -1.0
    v_rel  = v_rad_raw * r_hat + v_tan_raw * t_sign * t_hat

    x_rel = r01 * r_hat
    x0p   = -(m1 / M_pair) * x_rel
    x1p   =  (m0 / M_pair) * x_rel
    v0p   = -(m1 / M_pair) * v_rel
    v1p   =  (m0 / M_pair) * v_rel

    r2   = float(rng.uniform(1.0, 6.0))
    th2  = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p  = r2 * np.array([np.cos(th2), np.sin(th2), 0.0])
    f2   = float(rng.uniform(0.5, 1.1))
    vc2  = float(np.sqrt(G * M_pair / r2))
    v2p  = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0])

    m     = np.array([m0, m1, m2], dtype=np.float64)
    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)
    x_all, v_all = _com_center(x_all, v_all, m)

    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not _validate_ic(x_all, v_all, m, M_pair, r01_check):
        return None

    vr, vt = compute_pair_velocity_features(x_all, v_all, m)
    if not (math.isfinite(vr) and math.isfinite(vt)):
        return None

    # Verify genuinely approaching after COM centering
    if vr > -0.10:
        return None

    return x_all, v_all, m, r01_check, vr, vt


def make_ic(rng, mode="broad"):
    if mode == "broad":
        return make_ic_broad(rng)
    elif mode == "strong_recede":
        return make_ic_strong_recede(rng)
    elif mode == "probe_fix":
        return make_ic_probe_fix(rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Argument parser
# =============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Phase 3 Zone 3 trajectory data generator."
    )
    ap.add_argument("--dt", type=float, required=True,
                    help="Macro timestep. Valid values: 0.02, 0.04, 0.05, 0.06, 0.08, 0.10")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of accepted samples")
    ap.add_argument("--batch", type=int, default=0,
                    help="Batch ID (used for seed and filename)")
    ap.add_argument("--mode", default="broad",
                    choices=["broad", "strong_recede", "probe_fix"],
                    help="Sampling mode")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir",
                    default="encounter_shards",
                    help="Output folder for shard .npz files")
    ap.add_argument("--prefix", default="encounter_data_zone3_p3",
                    help="Output filename prefix")
    ap.add_argument("--timeout", type=float, default=IAS15_TIMEOUT_SEC)
    ap.add_argument("--max_tries", type=int, default=0,
                    help="Max attempts. 0 = n*2000.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose_attempts", action="store_true")
    return ap


# =============================================================================
# Main
# =============================================================================
def main():
    args = build_arg_parser().parse_args()
    dt = float(args.dt)

    # --- Phase 3 dt validation ---
    dt_rounded = round(dt, 4)
    if dt_rounded not in VALID_DTS:
        raise ValueError(
            f"dt={dt} is not a valid Phase 3 dt.\n"
            f"Valid values: {sorted(VALID_DTS)}\n"
            f"dt=0.005 and dt=0.01 are intentionally excluded -- "
            f"no meaningful training signal at fine timesteps."
        )
    if args.mode == "probe_fix" and dt_rounded not in PROBE_FIX_DTS:
        raise ValueError(
            f"probe_fix mode is only valid for dt in {sorted(PROBE_FIX_DTS)}. "
            f"Got dt={dt}."
        )

    n_target  = int(args.n)
    batch_id  = int(args.batch)
    max_tries = int(args.max_tries) if args.max_tries > 0 else max(2000 * n_target, 2000)

    os.makedirs(args.out_dir, exist_ok=True)
    token = dt_token(dt)

    # Include mode in filename so broad/strong_recede/probe_fix shards
    # coexist in the same encounter_shards folder
    mode_tag = "" if args.mode == "broad" else f"_{args.mode}"
    out_file     = os.path.join(args.out_dir,
                                f"{args.prefix}{mode_tag}_dt{token}_batch{batch_id:03d}.npz")
    summary_file = out_file.replace(".npz", "_summary.txt")

    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_file}\nUse --overwrite to replace.")

    # Mode-specific seed offset so same batch_id + dt gives different samples per mode
    mode_offsets = {"broad": 0, "strong_recede": 37_000_000, "probe_fix": 91_000_000}
    seed = int(args.seed) + batch_id * 100000 + int(round(dt * 1_000_000)) + mode_offsets[args.mode]
    rng  = np.random.RandomState(seed)

    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "v_rad_norm", "v_tan_norm",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]}

    rej = dict(ic=0, ias15=0, boundary=0, eject=0, no_improve=0, bad_c=0)
    c_buf, imp_buf, vr_buf, vt_buf = [], [], [], []
    n_identity = 0
    n_strong06 = 0
    n_strong08 = 0

    print("=" * 72)
    print("ZONE 3 TRAJECTORY DATA GENERATOR -- PHASE 3")
    print(f"  dt       : {dt:.6f} yr")
    print(f"  mode     : {args.mode}")
    print(f"  target n : {n_target}")
    print(f"  batch    : {batch_id}")
    print(f"  seed     : {seed}")
    print(f"  r zone   : ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU")
    print(f"  output   : {out_file}")
    print("=" * 72)

    t0   = time.perf_counter()
    n_ok = 0
    n_try = 0

    while n_ok < n_target and n_try < max_tries:
        n_try += 1

        ic = make_ic(rng, mode=args.mode)
        if ic is None:
            rej["ic"] += 1
            continue

        x0, v0, m, r01, v_rad_norm, v_tan_norm = ic

        if args.verbose_attempts:
            print(f"IAS15 START try={n_try} ok={n_ok} r01={r01:.4f} "
                  f"vr={v_rad_norm:+.3f} vt={v_tan_norm:.3f} dt={dt:.4f}", flush=True)

        try:
            x_ref, _ = ias15_one_step_isolated(x0, v0, m, dt, timeout=float(args.timeout))
        except Exception as e:
            if args.verbose_attempts:
                print(f"IAS15 EXCP try={n_try} {type(e).__name__}: {e}", flush=True)
            rej["ias15"] += 1
            continue

        if safe_max_norm_3x3(x_ref) > 50.0:
            rej["eject"] += 1
            continue

        try:
            result = find_c_opt(x0, v0, m, dt, x_ref)
            if not isinstance(result, tuple) or len(result) != 3:
                rej["bad_c"] += 1
                continue
            c_opt, mse_opt, mse_1 = result
            c_opt  = float(c_opt)
            mse_opt = float(mse_opt)
            mse_1  = float(mse_1)
        except Exception:
            rej["bad_c"] += 1
            continue

        if not (math.isfinite(c_opt) and math.isfinite(mse_opt) and math.isfinite(mse_1)):
            rej["bad_c"] += 1
            continue

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

        imp_raw = float((mse_1 - mse_opt) / mse_1)

        # Phase 3 improvement threshold: 0.10 (Phase 2 used 0.05).
        # Samples below threshold are stored as near-identity (c=1) so the
        # NN learns to leave borderline cases alone. We still accept them
        # but give them low training weight in the trainer (--low-impr-weight).
        if imp_raw < 0.10:
            c_store     = 1.0
            log_c_store = 0.0
            imp         = 0.0
            n_identity  += 1
        else:
            c_store     = c_opt
            log_c_store = float(np.log(c_opt))
            imp         = imp_raw

        r_soft = float(np.sqrt(r01**2 + EPS**2))
        c_ana  = float((r_soft / r01) ** 3)

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
        vr_buf.append(v_rad_norm)
        vt_buf.append(v_tan_norm)
        if v_rad_norm < -0.6: n_strong06 += 1
        if v_rad_norm < -0.8: n_strong08 += 1
        n_ok += 1

        if n_ok == 1 or n_ok == n_target or n_ok % max(1, min(50, n_target // 10)) == 0:
            el = time.perf_counter() - t0
            print(f"  accepted {n_ok:>5}/{n_target} | tried={n_try:>6} | "
                  f"c_med={np.median(c_buf):.4f} | "
                  f"vr_med={np.median(vr_buf):+.3f} | "
                  f"vt_med={np.median(vt_buf):.3f} | {el:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    save = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(out_file, **save)
    size_kb = os.path.getsize(out_file) / 1024.0

    summary = (f"dt={dt:.6f} mode={args.mode}: n={n_ok} | tried={n_try} | "
               f"pass={n_ok/max(n_try,1):.1%} | c_med={float(np.median(c_buf)) if c_buf else float('nan'):.5f} | "
               f"impr_med={float(np.median(imp_buf)) if imp_buf else float('nan'):.2%} | "
               f"identity={n_identity} | "
               f"vr_med={float(np.median(vr_buf)) if vr_buf else float('nan'):+.3f} | "
               f"vr<-0.6={n_strong06} | vr<-0.8={n_strong08} | {elapsed:.0f}s")

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("ZONE 3 TRAJECTORY DATASET SUMMARY -- PHASE 3\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"dt      : {dt:.6f}\n")
        fh.write(f"mode    : {args.mode}\n")
        fh.write(f"batch   : {batch_id}\n")
        fh.write(f"seed    : {seed}\n")
        fh.write(f"target  : {n_target}\n")
        fh.write(f"accepted: {n_ok}\n")
        fh.write(f"tried   : {n_try}\n")
        fh.write(f"elapsed : {elapsed:.1f} sec\n")
        fh.write(f"output  : {out_file}\n")
        fh.write(f"size_kb : {size_kb:.1f}\n")
        fh.write(f"zone    : r {Z3_R_MIN:.3f}-{Z3_R_MAX:.3f} AU\n")
        fh.write(f"improvement_threshold: 0.10 (Phase 3; Phase 2 used 0.05)\n\n")
        fh.write(summary + "\n")
        fh.write(f"Rejected: {rej}\n")
        fh.write(f"Near-identity (c=1) stored: {n_identity}\n")

    print("\n" + summary)
    print(f"Rejected: {rej}")
    print(f"Near-identity stored: {n_identity}")
    print(f"Saved {out_file} ({n_ok} samples, {size_kb:.1f} KB)")
    if n_ok < n_target:
        print(f"WARNING: partial batch {n_ok}/{n_target}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
