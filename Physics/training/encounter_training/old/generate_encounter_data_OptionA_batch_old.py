"""
generate_encounter_data.py  --  Steps 1 and 2  (FINAL)

Generates training data for the trajectory-optimal scalar correction c in
Zone 3 (macro-step zone, r in 0.052-0.148 AU).

═══════════════════════════════════════════════════════════════
DESIGN DECISION: Zone 3 only — Zone 2 removed
═══════════════════════════════════════════════════════════════
The new c_opt is mass-dependent:
    step_fraction = v_circ × dt / r = sqrt(G·M/r) × dt / r
    c_opt depends on M = m_i + m_j through v_circ.

Zone 2 in the simulation is the Earth-Moon bound orbit
(r = 0.00257 AU, permanent). Earth-Moon masses are 3.7e-8 to
3.0e-6 M☉ — 4 to 7 orders of magnitude below training masses
(0.001 to 2.0 M☉). The NN cannot extrapolate this far.

Zone 2 is therefore handled by the analytic formula c=(r_soft/r)³
in the simulation, unchanged from the original model. This is:
  - Correct: Earth-Moon has 188 sub-steps/orbit at sub_dt=0.0025 yr;
    the leapfrog resolves the circular orbit accurately;
    only the 2% softening correction is needed (c_analytic=1.020)
  - Safe: no risk of corrupting the Earth-Moon orbit with wrong c

Zone 3 (r 0.05-0.15 AU, macro steps):
  - c_analytic ≈ 1.000 (softening negligible at these r values)
  - c_opt is dt-dependent: 0.999 at dt=0.005, 0.764 at dt=0.100
  - This dependence has NO analytic form — it requires a NN
  - Training masses (0.001-2.0 M☉) match Zone 3 use (Sun ≈ 1 M☉)
═══════════════════════════════════════════════════════════════

Fixes carried forward from previous version:
  Bug1  _mse() uses np.sum() FUNCTION, not .sum() array METHOD.
        Fixes TypeError on some Windows/NumPy versions.
  Bug2  Summary c_ana_med computed from full buffer, not last r01.
  Bug3  make_ic() validity check uses Zone 3 bounds explicitly.

New vs original generate_encounter_data.py (pre-revamp):
  - dt=0.06 added to DT_VALUES (all paper energy-drift sweep values)
  - log_dt stored as 10th field — the new 4th NN input

Output (10 float32 arrays in encounter_data.npz):
  r_AU, r_soft, log_mi, log_mj, log_dt,
  c_opt, log_c_opt, c_ana, log_c_ana, improvement
"""

import os
import time
import math
import argparse
import multiprocessing as mp
import numpy as np
import rebound
import faulthandler, sys
faulthandler.enable(file=sys.stderr)   # print C stack trace to stderr on crash

# ── Constants — must match pair_eval_after_adaptive.py exactly ─────────────────
G            = 1.0
EPS          = 3e-4         # cfg.eps  (softening length)
R_SOFT_MIN   = 5e-4         # cfg.r_soft_min  (safety gate on r_soft)
NN_THRESH    = 500.0 * EPS  # nn_thresh = 0.15 AU
ADAPT_THRESH = 0.05         # adapt_thresh = 0.05 AU
C_MIN, C_MAX = 0.2, 5.0

# R_GATE: r where r_soft = R_SOFT_MIN — lower boundary of Zone 2 / Zone 3 split
# r_gate = sqrt(R_SOFT_MIN² - EPS²) = 4e-4 AU.  Zone 2 (r < ADAPT_THRESH)
# uses c = (r_soft/r)³ analytically; Zone 3 (above ADAPT_THRESH) uses this NN.
R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))   # = 4e-4 AU

# ── Dataset parameters ─────────────────────────────────────────────────────────
N_PER_DT  = 10000  # accepted samples per dt value

# All dt values used in the paper (energy-drift sweep + speed/accuracy frontier)
DT_VALUES = [0.005, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]

SEED     = 42
N_GRID   = 60
C_GRID   = np.linspace(C_MIN, C_MAX, N_GRID)
OUT_FILE = "encounter_data.npz"
SUMMARY  = "encounter_summary.txt"
IAS15_TIMEOUT_SEC = 60.0  # subprocess safety timeout per one-step IAS15 call

# Zone 3 r boundaries — macro-step zone, NN active, no sub-stepping
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU


# ── MSE helper ────────────────────────────────────────────────────────────────
def _mse(a, b):
    """
    Ultra-safe mean squared error between two (3,3) position arrays.

    Avoids NumPy reductions because repeated bad trial-c states on Windows
    previously triggered C-level access violations inside np.sum / np.all.
    Returns +inf for non-finite or exploded trial states.
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
        if not math.isfinite(val):
            return float("inf")

        return float(val)

    except Exception:
        return float("inf")


# ── ias15 one-step ground truth ───────────────────────────────────────────────
def ias15_one_step(x0, v0, m_arr, dt):
    """
    Advance a three-body system from t=0 to t=dt using ias15 (G=1).
    Returns x_new (3,3), v_new (3,3).

    CRASH FIX: sim.exit_min_distance = EPS
      Without this, unbound Zone 3 pairs (v_rel > v_escape for all mass/r
      combinations when f=1.8) can reach r=0 during ias15 integration.
      ias15 uses exact Newtonian (no softening), so force -> infinity causes
      a C-level segfault that except Exception cannot catch.
      Setting exit_min_distance = EPS makes rebound raise a Python exception
      instead, which is safely caught in the main loop. Samples where the
      pair reaches r < EPS are also physically meaningless for training
      (the simulation uses softening at these scales anyway).
    """
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    sim.exit_min_distance = float(EPS)   # raise Python exception, not C crash
    for i in range(3):
        sim.add(m=float(m_arr[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    sim.integrate(float(dt))
    x_new = np.array([[p.x,  p.y,  p.z]  for p in sim.particles])
    v_new = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    if not np.all(np.isfinite(x_new)):
        raise ValueError("ias15 returned non-finite positions")
    return x_new, v_new


# ── IAS15 subprocess isolation (Option A) ─────────────────────────────────────
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
    subprocess. If REBOUND/IAS15 hits a Windows access violation, only the
    child process dies; the main data generator rejects that IC and continues.

    This does not change the training methodology: successful samples use the
    same ias15_one_step() ground truth as before.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_ias15_worker, args=(child_conn, x0, v0, m_arr, float(dt)))
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


# ── Leapfrog with fixed scalar c ──────────────────────────────────────────────
def acc_fixed_c(pos, m_arr, c_val):
    """
    Accelerations for three bodies.
    Pair (0,1): softened force × c_val when r < NN_THRESH.
    Pairs (0,2), (1,2): exact Newtonian (far apart in all training ICs).
    m_arr not m to avoid shadowing outer mass variables.
    """
    acc  = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS
    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        rij    = pos[j] - pos[i]
        r2     = float(np.dot(rij, rij))
        r      = float((r2 + 1e-30) ** 0.5)
        Gmimj  = G * float(m_arr[i]) * float(m_arr[j])
        if idx == 0 and r < NN_THRESH:
            denom    = (r2 + eps2) ** 1.5 + 1e-30
            F_scalar = c_val * Gmimj / denom
        else:
            F_scalar = Gmimj / (r2 * r + 1e-30)
        F_vec   = F_scalar * rij
        acc[i] +=  F_vec / float(m_arr[i])
        acc[j] -= F_vec / float(m_arr[j])
    return acc


def leapfrog_step(x0, v0, m_arr, dt, c_val):
    """One velocity-Verlet step with fixed scalar correction c_val."""
    a0     = acc_fixed_c(x0, m_arr, c_val)
    v_half = v0 + 0.5 * dt * a0
    x1     = x0 + dt * v_half
    a1     = acc_fixed_c(x1, m_arr, c_val)
    v1     = v_half + 0.5 * dt * a1
    return x1, v1


# ── c_optimal search ──────────────────────────────────────────────────────────
def find_c_opt(x0, v0, m_arr, dt, x_ref):
    """
    Grid search + golden-section refinement for c in [C_MIN, C_MAX] that
    minimises MSE(x_leapfrog(c), x_ref) after one step of size dt = macro_dt.

    All MSE calls use _mse(), which avoids NumPy reductions for crash safety.
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
    c_lo   = float(C_GRID[max(0, best_k - 1)])
    c_hi   = float(C_GRID[min(N_GRID - 1, best_k + 1)])

    phi = (float(np.sqrt(5.0)) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7:
            break
        c1    = c_hi - phi * (c_hi - c_lo)
        c2    = c_lo + phi * (c_hi - c_lo)
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

    c_opt    = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step(x0, v0, m_arr, dt, c_opt)
    mse_opt  = _mse(x_opt, x_ref)
    return c_opt, mse_opt, mse_1


# ── IC generator ──────────────────────────────────────────────────────────────
def make_ic(rng):
    """
    Synthetic Zone 3 three-body starting state.
    Pair (0,1): r01 in (Z3_R_MIN, Z3_R_MAX) = (0.052, 0.148) AU.
    Body 2: far away (3-8 AU) to isolate the close-pair signal.

    Returns (x_all, v_all, m, r01) or None if IC is invalid.
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m  = np.array([m0, m1, m2], dtype=np.float64)

    r01   = float(rng.uniform(Z3_R_MIN, Z3_R_MAX))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)

    f      = float(rng.uniform(0.4, 1.8))
    v_circ = float(np.sqrt(G * (m0 + m1) / r01))
    phi_v  = float(rng.uniform(0.0, 2.0 * np.pi))
    v_hat  = np.array([np.cos(phi_v), np.sin(phi_v), 0.0], dtype=np.float64)
    v_rel  = f * v_circ * v_hat

    x_rel = r01 * r_hat
    x0p   = -(m1 / (m0 + m1)) * x_rel
    x1p   =  (m0 / (m0 + m1)) * x_rel
    v0p   = -(m1 / (m0 + m1)) * v_rel
    v1p   =  (m0 / (m0 + m1)) * v_rel

    r2   = float(rng.uniform(3.0, 8.0))
    th2  = float(rng.uniform(0.0, 2.0 * np.pi))
    x2p  = r2 * np.array([np.cos(th2), np.sin(th2), 0.0], dtype=np.float64)
    f2   = float(rng.uniform(0.5, 1.1))
    vc2  = float(np.sqrt(G * (m0 + m1) / r2))
    v2p  = f2 * vc2 * np.array([-np.sin(th2), np.cos(th2), 0.0], dtype=np.float64)

    x_all = np.array([x0p, x1p, x2p], dtype=np.float64)
    v_all = np.array([v0p, v1p, v2p], dtype=np.float64)

    # CoM frame
    M_tot = float(m.sum())
    x_com = np.sum(m[:, None] * x_all, axis=0) / M_tot
    v_com = np.sum(m[:, None] * v_all, axis=0) / M_tot
    x_all = x_all - x_com
    v_all = v_all - v_com

    # Reject unbound systems
    KE = 0.5 * float(np.sum(m * np.array(
        [float(np.dot(v_all[i], v_all[i])) for i in range(3)])))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            PE -= G * m[i] * m[j] / float(np.linalg.norm(x_all[i] - x_all[j]))
    if KE + PE >= 0.0:
        return None

    # Bug3 fix: validity check uses Zone 3 bounds explicitly
    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not (Z3_R_MIN < r01_check < Z3_R_MAX):
        return None

    # CRASH FIX: reject ICs where pair (0,1) reaches r < EPS during ias15.
    # ias15 uses exact Newtonian (no softening). At r -> 0, force -> infinity
    # causing a C-level segfault that except Exception cannot catch.
    # Even bound orbits crash if the approach angle is nearly radial
    # (periapsis = 0 for exactly radial at any f; < EPS for angle within ~2deg).
    # The CoM shift does not change relative position or velocity.
    # Compute 2-body Keplerian periapsis and reject if r_peri < EPS.
    r_rel_pair = x_all[1] - x_all[0]
    v_rel_pair = v_all[1] - v_all[0]
    v_sq       = float(np.dot(v_rel_pair, v_rel_pair))
    M_pair     = m0 + m1
    eps_orb    = v_sq / 2.0 - G * M_pair / r01_check
    # Specific angular momentum z-component (orbit is in x-y plane)
    h_z        = float(r_rel_pair[0] * v_rel_pair[1]
                       - r_rel_pair[1] * v_rel_pair[0])
    h_sq       = h_z * h_z
    disc       = max(0.0, 1.0 + 2.0 * eps_orb * h_sq / (G * M_pair) ** 2)
    e_ecc      = float(np.sqrt(disc))
    r_peri     = h_sq / (G * M_pair * (1.0 + e_ecc) + 1e-30)
    if r_peri < EPS:
        return None   # near-radial: periapsis too close, ias15 would crash

    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    return x_all, v_all, m, r01_check


# ── Main: one small shard per run ─────────────────────────────────────────────
def _dt_tag(dt):
    """Filename-safe dt tag, e.g. 0.04 -> 0p040."""
    return f"{float(dt):.3f}".replace(".", "p")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Generate one small Zone-3 encounter-data shard. "
            "Methodology is unchanged from OptionA: IAS15 one-step reference, "
            "grid + golden-section search for c_opt, same stored fields."
        )
    )
    ap.add_argument("--dt", type=float, required=True,
                    help="Single macro timestep for this shard, e.g. 0.04")
    ap.add_argument("--n", type=int, default=500,
                    help="Accepted samples to generate in this shard. Start with 100, then 250/500.")
    ap.add_argument("--batch", type=int, default=0,
                    help="Batch/shard index used in filename and default seed.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional explicit random seed. If omitted, seed is derived from dt and batch.")
    ap.add_argument("--out-dir", default="encounter_shards",
                    help="Folder where shard .npz and summary .txt are saved.")
    ap.add_argument("--prefix", default="z3",
                    help="Filename prefix for shard outputs.")
    ap.add_argument("--max-tries", type=int, default=0,
                    help="Maximum candidate IC attempts. 0 means max(5000, 20*n).")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print progress every this many accepted samples.")
    ap.add_argument("--timeout", type=float, default=IAS15_TIMEOUT_SEC,
                    help="IAS15 subprocess timeout in seconds.")
    ap.add_argument("--debug", action="store_true",
                    help="Print IAS15/FINDCOPT start/done lines for every candidate.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing shard file if present.")
    return ap.parse_args()


def main():
    args = parse_args()
    dt = float(args.dt)
    n_target = int(args.n)

    valid_dts = [round(float(x), 6) for x in DT_VALUES]
    if round(dt, 6) not in valid_dts:
        raise ValueError(f"dt={dt} is not in DT_VALUES={DT_VALUES}")
    if n_target <= 0:
        raise ValueError("--n must be positive")

    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else max(5000, 20 * n_target)

    if args.seed is None:
        # Deterministic but distinct per dt and batch.
        seed = int(SEED + 100000 * int(args.batch) + round(dt * 1_000_000))
    else:
        seed = int(args.seed)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = _dt_tag(dt)
    out_file = os.path.join(out_dir, f"{args.prefix}_dt{tag}_batch{int(args.batch):03d}.npz")
    summary_file = os.path.join(out_dir, f"{args.prefix}_dt{tag}_batch{int(args.batch):03d}_summary.txt")

    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_file}\n"
            f"Use --overwrite if you intentionally want to replace it."
        )

    rng = np.random.RandomState(seed)
    t0_all = time.perf_counter()

    # 10 stored fields: original 9 + log_dt (new 4th NN input)
    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj",
        "log_dt",      # log(macro_dt) — 4th NN input
        "c_opt", "log_c_opt",
        "c_ana", "log_c_ana",
        "improvement",
    ]}

    print("=" * 72)
    print("CLOSE-ENCOUNTER DATA GENERATOR  (Zone 3 only, SHARD MODE)")
    print(f"  r       : ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU  (macro-step zone)")
    print(f"  dt      : {dt:.4f}")
    print(f"  target  : {n_target} accepted samples")
    print(f"  batch   : {int(args.batch):03d}")
    print(f"  seed    : {seed}")
    print(f"  max_try : {max_tries}")
    print(f"  output  : {out_file}")
    print(f"  Zone 2  : not generated here; handled analytically in simulation")
    print("=" * 72)

    n_ok = 0
    n_try = 0
    rej = dict(ic=0, ias15=0, boundary=0, eject=0, no_improve=0, find_c=0)
    c_buf, imp_buf, c_ana_buf = [], [], []

    while n_ok < n_target and n_try < max_tries:
        n_try += 1

        ic = make_ic(rng)
        if ic is None:
            rej["ic"] += 1
            continue
        x0, v0, m, r01 = ic

        if args.debug:
            print(f"IAS15 START try={n_try} ok={n_ok}"
                  f" r01={r01:.4f} m0={m[0]:.4f} m1={m[1]:.4f}"
                  f" m2={m[2]:.4f} dt={dt:.4f}", flush=True)

        try:
            x_ref, _ = ias15_one_step_isolated(x0, v0, m, dt, timeout=float(args.timeout))
            if args.debug:
                print(f"IAS15 DONE  try={n_try}", flush=True)
        except Exception as e:
            if args.debug:
                print(f"IAS15 EXCP  try={n_try} {type(e).__name__}: {e}", flush=True)
            rej["ias15"] += 1
            continue

        if float(np.max(np.linalg.norm(x_ref, axis=1))) > 50.0:
            rej["eject"] += 1
            continue

        if args.debug:
            print(f"FINDCOPT START try={n_try}", flush=True)
        try:
            result = find_c_opt(x0, v0, m, dt, x_ref)

            if not isinstance(result, tuple) or len(result) != 3:
                if args.debug:
                    print(f"FIND_C BADRET try={n_try} result_type={type(result).__name__}", flush=True)
                rej["find_c"] += 1
                continue

            c_opt, mse_opt, mse_1 = result
            c_opt = float(c_opt)
            mse_opt = float(mse_opt)
            mse_1 = float(mse_1)

        except Exception as e:
            if args.debug:
                print(f"FIND_C EXCP try={n_try} {type(e).__name__}: {e}", flush=True)
            rej["find_c"] += 1
            continue

        if not math.isfinite(c_opt) or not math.isfinite(mse_opt):
            rej["find_c"] += 1
            continue
        if args.debug:
            print(f"FINDCOPT DONE  try={n_try} c_opt={c_opt:.4f}", flush=True)

        if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
            rej["boundary"] += 1
            continue
        x_best, _ = leapfrog_step(x0, v0, m, dt, c_opt)
        if float(np.max(np.linalg.norm(x_best, axis=1))) > 50.0:
            rej["eject"] += 1
            continue
        if mse_1 < 1e-20 or (mse_opt / mse_1) > 0.95:
            rej["no_improve"] += 1
            continue

        # Accept
        r_soft = float(np.sqrt(r01**2 + EPS**2))
        c_ana = float((r_soft / r01) ** 3)
        imp = float((mse_1 - mse_opt) / mse_1)

        arr["r_AU"].append(r01)
        arr["r_soft"].append(r_soft)
        arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
        arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
        arr["log_dt"].append(float(np.log(dt)))
        arr["c_opt"].append(c_opt)
        arr["log_c_opt"].append(float(np.log(c_opt)))
        arr["c_ana"].append(c_ana)
        arr["log_c_ana"].append(float(np.log(c_ana)))
        arr["improvement"].append(imp)

        c_buf.append(c_opt)
        imp_buf.append(imp)
        c_ana_buf.append(c_ana)
        n_ok += 1

        if n_ok % max(1, int(args.progress_every)) == 0 or n_ok == n_target:
            el = time.perf_counter() - t0_all
            pass_rate = n_ok / max(n_try, 1)
            print(f"  dt={dt:.3f} batch={int(args.batch):03d} "
                  f"{n_ok}/{n_target} accepted | tried={n_try} | "
                  f"pass={pass_rate:.1%} | {el:.0f}s", flush=True)

    if n_ok < n_target:
        print("WARNING: shard ended before reaching target accepted samples.")
        print(f"  accepted={n_ok}, target={n_target}, tried={n_try}, max_tries={max_tries}")

    save = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(out_file, **save)
    total_n = len(arr["r_AU"])
    size_kb = os.path.getsize(out_file) / 1024 if os.path.exists(out_file) else 0.0
    elapsed = time.perf_counter() - t0_all

    if total_n > 0:
        line = (f"dt={dt:.3f}: n={total_n} | tried={n_try} | "
                f"pass={total_n/max(n_try,1):.1%} | "
                f"c_opt_med={np.median(c_buf):.4f} | "
                f"c_ana_med={np.median(c_ana_buf):.6f} | "
                f"impr_med={np.median(imp_buf):.2%} | {elapsed:.0f}s")
    else:
        line = (f"dt={dt:.3f}: n=0 | tried={n_try} | pass=0.0% | "
                f"no accepted samples | {elapsed:.0f}s")

    print(f"\nSaved {out_file}  ({total_n} samples, {size_kb:.0f} KB)")
    print(f"Summary: {line}")
    print(f"Rejected: {rej}")
    print(f"Wall time: {elapsed:.0f}s")

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("CLOSE-ENCOUNTER DATA SHARD SUMMARY  (Zone 3 only)\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"File   : {os.path.basename(out_file)}\n")
        fh.write(f"Total  : {total_n}\n")
        fh.write(f"Target : {n_target}\n")
        fh.write(f"Tried  : {n_try}\n")
        fh.write(f"Batch  : {int(args.batch):03d}\n")
        fh.write(f"Seed   : {seed}\n")
        fh.write(f"Zone   : 3 only (r {Z3_R_MIN:.3f}-{Z3_R_MAX:.3f} AU, macro-step)\n")
        fh.write(f"DT     : {dt}\n")
        fh.write(f"Fields : r_AU r_soft log_mi log_mj log_dt "
                 f"c_opt log_c_opt c_ana log_c_ana improvement\n\n")
        fh.write(line + "\n")
        fh.write(f"Rejected: {rej}\n")
    print(f"Saved {summary_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
