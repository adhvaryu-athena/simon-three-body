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

# Zone 3 r boundaries — macro-step zone, NN active, no sub-stepping
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU


# ── MSE helper ────────────────────────────────────────────────────────────────
def _mse(a, b):
    """
    Safe mean squared error between two (3,3) position arrays.

    Returns +inf if either input is non-finite or numerically extreme.
    This prevents bad trial c values from crashing the optimiser.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        return float("inf")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("inf")

    # Reject numerically exploded trial states.
    if np.max(np.abs(a)) > 1e6 or np.max(np.abs(b)) > 1e6:
        return float("inf")

    d = a - b

    if not np.all(np.isfinite(d)):
        return float("inf")

    val = np.einsum("ij,ij->", d, d) / d.size

    if not np.isfinite(val):
        return float("inf")

    return float(val)


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

    All 5 MSE calls use _mse() with np.sum() (not .sum() method).
    Returns (c_opt, mse_opt, mse_at_c1).
    """
    x1, _ = leapfrog_step(x0, v0, m_arr, dt, 1.0)
    mse_1 = _mse(x1, x_ref)

    mse_g = np.empty(N_GRID)
    for k, c in enumerate(C_GRID):
        xk, _    = leapfrog_step(x0, v0, m_arr, dt, float(c))
        mse_g[k] = _mse(xk, x_ref)

    if not np.any(np.isfinite(mse_g)):
        return np.nan, float("inf"), mse_1

    best_k = int(np.argmin(mse_g))
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

        if not np.isfinite(m1):
            m1 = float("inf")
        if not np.isfinite(m2):
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng    = np.random.RandomState(SEED)
    t0_all = time.perf_counter()

    # 10 stored fields: original 9 + log_dt (new 4th NN input)
    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj",
        "log_dt",      # NEW: log(macro_dt) — 4th NN input
        "c_opt", "log_c_opt",
        "c_ana", "log_c_ana",
        "improvement",
    ]}

    print("=" * 65)
    print("CLOSE-ENCOUNTER DATA GENERATOR  (Zone 3 only, FINAL)")
    print(f"  r    : ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU  (macro-step zone)")
    print(f"  DT   : {DT_VALUES}")
    print(f"  N/dt : {N_PER_DT}   Total: {N_PER_DT * len(DT_VALUES)}")
    print(f"  Zone 2 (r < {ADAPT_THRESH}) uses analytic c=(r_soft/r)^3 in simulation")
    print("=" * 65)

    summary_lines = []

    for dt in DT_VALUES:
        n_ok  = 0
        n_try = 0
        rej   = dict(ic=0, boundary=0, eject=0, no_improve=0)
        t0dt  = time.perf_counter()
        c_buf, imp_buf, c_ana_buf = [], [], []

        while n_ok < N_PER_DT:
            n_try += 1

            import sys

            ic = make_ic(rng)
            if ic is None:
                rej["ic"] += 1
                continue
            x0, v0, m, r01 = ic

            # ── DIAGNOSTIC: print before/after EVERY ias15 call ───────────
            print(f"IAS15 START try={n_try} ok={n_ok}"
                  f" r01={r01:.4f} m0={m[0]:.4f} m1={m[1]:.4f}"
                  f" m2={m[2]:.4f} dt={dt:.4f}", flush=True)

            try:
                x_ref, _ = ias15_one_step(x0, v0, m, dt)
                print(f"IAS15 DONE  try={n_try}", flush=True)
            except Exception as e:
                print(f"IAS15 EXCP  try={n_try} {type(e).__name__}", flush=True)
                rej["ic"] += 1
                continue
            # ─────────────────────────────────────────────────────────────

            if float(np.max(np.linalg.norm(x_ref, axis=1))) > 50.0:
                rej["eject"] += 1
                continue

            print(f"FINDCOPT START try={n_try}", flush=True)
            try:
                c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, dt, x_ref)
            except Exception as e:
                print(f"FIND_C EXCP try={n_try} {type(e).__name__}", flush=True)
                rej["ic"] += 1
                continue
            if not np.isfinite(c_opt) or not np.isfinite(mse_opt):
                rej["ic"] += 1
                continue
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
            c_ana  = float((r_soft / r01) ** 3)
            imp    = float((mse_1 - mse_opt) / mse_1)

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

            if n_ok % 2000 == 0:
                el = time.perf_counter() - t0dt
                print(f"  dt={dt:.3f}  {n_ok}/{N_PER_DT}"
                      f"  tried={n_try}  {el:.0f}s")

        el = time.perf_counter() - t0dt
        # Bug2 fix: c_ana_med from full buffer, not last r01
        line = (f"dt={dt:.3f}: n={n_ok} | tried={n_try} | "
                f"pass={n_ok/n_try:.1%} | "
                f"c_opt_med={np.median(c_buf):.4f} | "
                f"c_ana_med={np.median(c_ana_buf):.6f} | "
                f"impr_med={np.median(imp_buf):.2%} | {el:.0f}s")
        print(f"  {line}")
        print(f"  Rejected: {rej}")
        summary_lines.append(line)

    # ── Save ──────────────────────────────────────────────────────────────────
    save    = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(OUT_FILE, **save)
    total_n = len(arr["r_AU"])
    size_kb = os.path.getsize(OUT_FILE) / 1024
    elapsed = time.perf_counter() - t0_all

    print(f"\nSaved {OUT_FILE}  ({total_n} samples, {size_kb:.0f} KB)")
    print(f"Wall time: {elapsed:.0f}s")

    with open(SUMMARY, "w") as fh:
        fh.write("CLOSE-ENCOUNTER DATASET SUMMARY  (Zone 3 only)\n")
        fh.write("=" * 65 + "\n")
        fh.write(f"Total  : {total_n}\n")
        fh.write(f"Zone   : 3 only (r {Z3_R_MIN:.3f}-{Z3_R_MAX:.3f} AU, macro-step)\n")
        fh.write(f"DT     : {DT_VALUES}\n")
        fh.write(f"Fields : r_AU r_soft log_mi log_mj log_dt "
                 f"c_opt log_c_opt c_ana log_c_ana improvement\n\n")
        for ln in summary_lines:
            fh.write(ln + "\n")
        fh.write("\nKEY CHECKS (what to look for in inspect_encounter_data.py):\n")
        fh.write("  1. c_opt_med decreases monotonically as dt increases\n")
        fh.write("     dt=0.005: c_opt~1.0  (leapfrog resolves encounter)\n")
        fh.write("     dt=0.080: c_opt~0.81 (leapfrog misses encounter)\n")
        fh.write("     This dt-dependence has no analytic form -> NN is needed.\n")
        fh.write("  2. c_ana_med ~= 1.000 for all dt (softening negligible in Z3)\n")
        fh.write("  3. impr_med 75-85% (c_opt consistently better than c=1)\n")
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
