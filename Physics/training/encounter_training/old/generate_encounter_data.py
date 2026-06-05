"""
generate_encounter_data.py  --  Steps 1 and 2 (REVISED)

Fixes from generate_encounter_data_latest.py:
  Bug1  _mse() used (d*d).sum() array METHOD which raises
        TypeError: 'numpy.ndarray' object is not callable on some Windows/NumPy
        versions. Fixed: replaced with np.sum(d*d) FUNCTION call. All 5 MSE
        computations now use _mse() with np.sum().
  Bug2  Summary line used last r01/r_soft from the loop for c_ana median.
        Fixed: compute median from the full c_ana buffer.
  Bug3  make_ic() validity check was hardcoded for Zone 3 bounds. Would always
        reject Zone 2 ICs (r01 < ADAPT_THRESH). Fixed: r_min/r_max parametrised.
  Z2    Z2_R_MIN was 0.01 AU -- too high, inconsistent with diagram Zone 2.
        Fixed: Z2_R_MIN = R_GATE + 1e-5 = 4.1e-4 AU, matching Zone 2 lower
        boundary from the diagram (where r_soft = R_SOFT_MIN).
        Note: at r < ~0.005 AU most samples are filtered (encounters unresolvable
        even after sub-stepping). MAX_TRIES_Z2 prevents infinite loops there.

New features vs original generate_encounter_data.py:
  - dt=0.06 added to DT_VALUES (covers all paper energy-drift sweep values).
  - Zone 2 (sub-step zone): r in (R_GATE+1e-5, 0.048) AU.
      Effective sub_dt = macro_dt/n_sub computed per sample from r01.
      log_dt stored = log(sub_dt) -- the NN input at inference time.
  - Zone 3 (macro-step zone): r in (0.052, 0.148) AU. Unchanged from original.
      log_dt stored = log(macro_dt).
  - Two new stored fields: zone (2 or 3), log_macro_dt (for grouping in inspect).
  - MAX_TRIES_Z2 prevents infinite loops if Zone 2 acceptance rate is low.

Physics:
  Zone 3 (0.052-0.148 AU, no sub-stepping):
    c_analytic ~= 1.0 (softening negligible here).
    c_optimal < 1 due to leapfrog discretisation error at large macro_dt.
    Primary value-add of trajectory-trained c.
  Zone 2 (4.1e-4 to 0.048 AU, inside adaptive sub-steps):
    c_analytic: 1.0 to ~1.95 (softening matters at small r).
    sub_dt is small; c_optimal ~= c_analytic ~= 1.0 for r > 0.002.
    Samples teach the NN boundary condition: small effective_dt -> c ~= 1.0.

Output:
  encounter_data.npz    -- dataset (12 float32 arrays)
  encounter_summary.txt -- per-zone per-dt statistics
"""

import os
import time
import numpy as np
import rebound

# ── Constants (must match pair_eval_after_adaptive.py exactly) ─────────────────
G            = 1.0
EPS          = 3e-4            # cfg.eps  (softening length)
R_SOFT_MIN   = 5e-4            # cfg.r_soft_min  (safety gate on r_soft)
NN_THRESH    = 500.0 * EPS     # nn_thresh = 0.15 AU
ADAPT_THRESH = 0.05            # adapt_thresh = 0.05 AU
MAX_SUBSTEPS = 16              # max_substeps
C_MIN, C_MAX = 0.2, 5.0

# R_GATE: r where r_soft = R_SOFT_MIN.  Below this, NN output is rejected.
# r_soft = sqrt(r^2 + eps^2) = R_SOFT_MIN  =>  r = sqrt(R_SOFT_MIN^2 - EPS^2)
R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))   # = 4e-4 AU exactly

# ── Dataset parameters ─────────────────────────────────────────────────────────
N_PER_DT     = 10000      # accepted samples per (macro_dt, zone=3)
N_PER_DT_Z2  = 3000       # accepted samples per (macro_dt, zone=2)
MAX_TRIES_Z2 = 80000      # abort Zone 2 for this macro_dt if limit hit

# All dt values used in the paper (energy-drift sweep + speed/accuracy frontier)
DT_VALUES = [0.005, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]

SEED     = 42
N_GRID   = 60
C_GRID   = np.linspace(C_MIN, C_MAX, N_GRID)
OUT_FILE = "encounter_data.npz"
SUMMARY  = "encounter_summary.txt"

# ── Zone r boundaries ──────────────────────────────────────────────────────────
Z3_R_MIN = ADAPT_THRESH + 0.002    # 0.052 AU  (macro-step zone)
Z3_R_MAX = NN_THRESH    - 0.002    # 0.148 AU

Z2_R_MIN = R_GATE + 1e-5           # ~4.1e-4 AU (matches diagram Zone 2 boundary)
Z2_R_MAX = ADAPT_THRESH - 0.002    # 0.048 AU


# ── MSE helper ────────────────────────────────────────────────────────────────
def _mse(a, b):
    """
    Mean squared error between two position arrays.

    IMPORTANT: uses np.sum() FUNCTION, not the .sum() array METHOD.
    The array method raises TypeError on some Windows/NumPy versions.
    np.sum() is portable across all supported NumPy versions.
    """
    d = a - b
    return float(np.sum(d * d)) / int(d.size)


# ── ias15 one-step ground truth ───────────────────────────────────────────────
def ias15_one_step(x0, v0, m_arr, dt):
    """
    Advance a three-body system from t=0 to t=dt using ias15 (G=1).
    dt may be macro_dt (Zone 3) or effective sub_dt (Zone 2).
    Returns x_new (3,3), v_new (3,3).
    """
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(3):
        sim.add(m=float(m_arr[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    sim.integrate(float(dt))
    x_new = np.array([[p.x,  p.y,  p.z]  for p in sim.particles])
    v_new = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    return x_new, v_new


# ── One-step leapfrog with fixed scalar c ─────────────────────────────────────
def acc_fixed_c(pos, m_arr, c_val):
    """
    Accelerations for three bodies.
    Pair (0,1): softened force * c_val when r < NN_THRESH.
    Pairs (0,2), (1,2): exact Newtonian (far apart in all ICs).

    Parameter named m_arr (not m) to avoid any risk of shadowing the outer
    masses variable passed in from find_c_opt / leapfrog_step callers.
    """
    acc  = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS
    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        rij      = pos[j] - pos[i]
        r2       = float(np.dot(rij, rij))
        r        = float((r2 + 1e-30) ** 0.5)
        Gmimj    = G * float(m_arr[i]) * float(m_arr[j])
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
    Grid search + golden-section refinement to find c in [C_MIN, C_MAX] that
    minimises MSE(x_leapfrog(c), x_ref) after one step of size dt.

    dt = effective_dt: macro_dt for Zone 3, sub_dt for Zone 2.
    All 5 MSE calls use _mse() with np.sum() (not .sum() method).
    Returns (c_opt, mse_opt, mse_at_c1).
    """
    # Baseline: c = 1.0 (softened Newtonian, no correction)
    x1, _ = leapfrog_step(x0, v0, m_arr, dt, 1.0)
    mse_1 = _mse(x1, x_ref)

    # Coarse grid across [C_MIN, C_MAX]
    mse_g = np.empty(N_GRID)
    for k, c in enumerate(C_GRID):
        xk, _    = leapfrog_step(x0, v0, m_arr, dt, float(c))
        mse_g[k] = _mse(xk, x_ref)

    best_k = int(np.argmin(mse_g))
    c_lo   = float(C_GRID[max(0, best_k - 1)])
    c_hi   = float(C_GRID[min(N_GRID - 1, best_k + 1)])

    # Golden-section refinement within best interval
    phi = (float(np.sqrt(5.0)) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7:
            break
        c1    = c_hi - phi * (c_hi - c_lo)
        c2    = c_lo + phi * (c_hi - c_lo)
        xa, _ = leapfrog_step(x0, v0, m_arr, dt, c1)
        xb, _ = leapfrog_step(x0, v0, m_arr, dt, c2)
        m1    = _mse(xa, x_ref)
        m2    = _mse(xb, x_ref)
        if m1 < m2:
            c_hi = c2
        else:
            c_lo = c1

    c_opt    = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step(x0, v0, m_arr, dt, c_opt)
    mse_opt  = _mse(x_opt, x_ref)
    return c_opt, mse_opt, mse_1


# ── Effective sub_dt for Zone 2 ───────────────────────────────────────────────
def get_effective_dt(macro_dt, r01):
    """
    Compute the sub_dt SIMON uses for a pair at r01 < ADAPT_THRESH.
    Mirrors the simulation exactly:
        n_sub  = min(MAX_SUBSTEPS, max(2, ceil(ADAPT_THRESH / r01)))
        sub_dt = macro_dt / n_sub
    Returns (sub_dt, n_sub).
    """
    n_sub  = min(MAX_SUBSTEPS, max(2, int(np.ceil(ADAPT_THRESH / r01))))
    sub_dt = macro_dt / n_sub
    return sub_dt, n_sub


# ── IC generator ──────────────────────────────────────────────────────────────
def make_ic(rng, r_min, r_max):
    """
    Synthetic three-body starting state.
    Pair (0,1): separation r01 in (r_min, r_max).
    Body 2: far away (3-8 AU) to isolate the close-pair physics.

    Bug3 fix: validity check uses r_min/r_max (not hardcoded Zone 3 bounds),
    so this works correctly for both Zone 2 and Zone 3 ICs.

    Returns (x_all, v_all, m, r01) or None if IC is invalid.
    """
    m0 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m1 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m2 = float(np.exp(rng.uniform(np.log(0.001), np.log(2.0))))
    m  = np.array([m0, m1, m2], dtype=np.float64)

    r01   = float(rng.uniform(r_min, r_max))
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

    # Shift to system CoM frame
    M_tot = float(m.sum())
    x_com = np.array([float(m[i]) * x_all[i] for i in range(3)],
                     dtype=np.float64).sum(axis=0) / M_tot
    v_com = np.array([float(m[i]) * v_all[i] for i in range(3)],
                     dtype=np.float64).sum(axis=0) / M_tot
    x_all = x_all - x_com
    v_all = v_all - v_com

    # Reject unbound systems
    KE = 0.5 * float(np.sum(m * np.array([float(np.dot(v_all[i], v_all[i]))
                                           for i in range(3)])))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            PE -= G * m[i] * m[j] / float(np.linalg.norm(x_all[i] - x_all[j]))
    if KE + PE >= 0.0:
        return None

    # Reject if pair (0,1) drifted outside target r range after CoM shift
    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not (r_min < r01_check < r_max):
        return None

    # Reject if other pairs are too close
    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    return x_all, v_all, m, r01_check


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng    = np.random.RandomState(SEED)
    t0_all = time.perf_counter()

    arr = {k: [] for k in [
        "r_AU",
        "r_soft",
        "log_mi",
        "log_mj",
        "log_dt",        # log(effective_dt): NN input [sub_dt Z2, macro_dt Z3]
        "log_macro_dt",  # log(macro_dt): for grouping in inspect script
        "c_opt",
        "log_c_opt",
        "c_ana",
        "log_c_ana",
        "improvement",
        "zone",          # 2.0 or 3.0
    ]}

    print("=" * 70)
    print("CLOSE-ENCOUNTER DATA GENERATOR  (REVISED)")
    print(f"  Zone 3: r in ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU  "
          f"{N_PER_DT} per dt")
    print(f"  Zone 2: r in ({Z2_R_MIN:.2e}, {Z2_R_MAX:.4f}) AU  "
          f"{N_PER_DT_Z2} per dt  max_tries={MAX_TRIES_Z2}")
    print(f"  DT values: {DT_VALUES}")
    print(f"  R_GATE = {R_GATE:.2e} AU  EPS = {EPS:.1e} AU")
    print("=" * 70)

    summary_lines = []

    for macro_dt in DT_VALUES:

        # ── ZONE 3 ────────────────────────────────────────────────────────────
        n_ok  = 0
        n_try = 0
        rej   = dict(ic=0, boundary=0, eject=0, no_improve=0)
        t0z3  = time.perf_counter()
        c_buf3, imp_buf3, c_ana_buf3 = [], [], []

        while n_ok < N_PER_DT:
            n_try += 1
            ic = make_ic(rng, Z3_R_MIN, Z3_R_MAX)
            if ic is None:
                rej["ic"] += 1
                continue
            x0, v0, m, r01 = ic

            try:
                x_ref, _ = ias15_one_step(x0, v0, m, macro_dt)
            except Exception:
                rej["ic"] += 1
                continue
            if float(np.max(np.linalg.norm(x_ref, axis=1))) > 50.0:
                rej["eject"] += 1
                continue

            c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, macro_dt, x_ref)

            if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
                rej["boundary"] += 1
                continue
            x_best, _ = leapfrog_step(x0, v0, m, macro_dt, c_opt)
            if float(np.max(np.linalg.norm(x_best, axis=1))) > 50.0:
                rej["eject"] += 1
                continue
            if mse_1 < 1e-20 or (mse_opt / mse_1) > 0.95:
                rej["no_improve"] += 1
                continue

            r_soft = float(np.sqrt(r01**2 + EPS**2))
            c_ana  = float((r_soft / r01) ** 3)
            imp    = float((mse_1 - mse_opt) / mse_1)

            arr["r_AU"].append(r01)
            arr["r_soft"].append(r_soft)
            arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
            arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
            arr["log_dt"].append(float(np.log(macro_dt)))
            arr["log_macro_dt"].append(float(np.log(macro_dt)))
            arr["c_opt"].append(c_opt)
            arr["log_c_opt"].append(float(np.log(c_opt)))
            arr["c_ana"].append(c_ana)
            arr["log_c_ana"].append(float(np.log(c_ana)))
            arr["improvement"].append(imp)
            arr["zone"].append(3.0)

            c_buf3.append(c_opt)
            imp_buf3.append(imp)
            c_ana_buf3.append(c_ana)
            n_ok += 1

            if n_ok % 2000 == 0:
                el = time.perf_counter() - t0z3
                print(f"    Z3 dt={macro_dt:.3f}  {n_ok}/{N_PER_DT}"
                      f"  tried={n_try}  {el:.0f}s")

        el3   = time.perf_counter() - t0z3
        line3 = (f"Z3 dt={macro_dt:.3f}: n={n_ok} | tried={n_try} | "
                 f"pass={n_ok/n_try:.1%} | "
                 f"c_opt_med={np.median(c_buf3):.4f} | "
                 f"c_ana_med={np.median(c_ana_buf3):.6f} | "
                 f"impr_med={np.median(imp_buf3):.2%} | {el3:.0f}s")
        print(f"  {line3}")
        print(f"  Rej Z3: {rej}")
        summary_lines.append(line3)

        # ── ZONE 2 ────────────────────────────────────────────────────────────
        n_ok2  = 0
        n_try2 = 0
        rej2   = dict(ic=0, boundary=0, eject=0, no_improve=0)
        t0z2   = time.perf_counter()
        c_buf2, imp_buf2, c_ana_buf2 = [], [], []

        while n_ok2 < N_PER_DT_Z2 and n_try2 < MAX_TRIES_Z2:
            n_try2 += 1
            ic = make_ic(rng, Z2_R_MIN, Z2_R_MAX)
            if ic is None:
                rej2["ic"] += 1
                continue
            x0, v0, m, r01 = ic

            sub_dt, _n_sub = get_effective_dt(macro_dt, r01)

            try:
                x_ref, _ = ias15_one_step(x0, v0, m, sub_dt)
            except Exception:
                rej2["ic"] += 1
                continue
            if float(np.max(np.linalg.norm(x_ref, axis=1))) > 50.0:
                rej2["eject"] += 1
                continue

            c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, sub_dt, x_ref)

            if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
                rej2["boundary"] += 1
                continue
            x_best, _ = leapfrog_step(x0, v0, m, sub_dt, c_opt)
            if float(np.max(np.linalg.norm(x_best, axis=1))) > 50.0:
                rej2["eject"] += 1
                continue
            if mse_1 < 1e-20 or (mse_opt / mse_1) > 0.95:
                rej2["no_improve"] += 1
                continue

            r_soft = float(np.sqrt(r01**2 + EPS**2))
            c_ana  = float((r_soft / r01) ** 3)
            imp    = float((mse_1 - mse_opt) / mse_1)

            arr["r_AU"].append(r01)
            arr["r_soft"].append(r_soft)
            arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
            arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
            arr["log_dt"].append(float(np.log(sub_dt)))
            arr["log_macro_dt"].append(float(np.log(macro_dt)))
            arr["c_opt"].append(c_opt)
            arr["log_c_opt"].append(float(np.log(c_opt)))
            arr["c_ana"].append(c_ana)
            arr["log_c_ana"].append(float(np.log(c_ana)))
            arr["improvement"].append(imp)
            arr["zone"].append(2.0)

            c_buf2.append(c_opt)
            imp_buf2.append(imp)
            c_ana_buf2.append(c_ana)
            n_ok2 += 1

            if n_ok2 % 500 == 0 and n_ok2 > 0:
                el = time.perf_counter() - t0z2
                print(f"    Z2 dt={macro_dt:.3f}  {n_ok2}/{N_PER_DT_Z2}"
                      f"  tried={n_try2}  {el:.0f}s")

        el2      = time.perf_counter() - t0z2
        at_limit = (n_try2 >= MAX_TRIES_Z2 and n_ok2 < N_PER_DT_Z2)
        suffix   = f"  [HIT MAX_TRIES: {n_ok2}/{N_PER_DT_Z2}]" if at_limit else ""
        if n_ok2 > 0:
            line2 = (f"Z2 dt={macro_dt:.3f}: n={n_ok2} | tried={n_try2} | "
                     f"pass={n_ok2/max(n_try2, 1):.1%} | "
                     f"c_opt_med={np.median(c_buf2):.4f} | "
                     f"c_ana_med={np.median(c_ana_buf2):.6f} | "
                     f"impr_med={np.median(imp_buf2):.2%} | {el2:.0f}s{suffix}")
        else:
            line2 = (f"Z2 dt={macro_dt:.3f}: n=0 | tried={n_try2} | "
                     f"NO SAMPLES COLLECTED{suffix}")
        print(f"  {line2}")
        print(f"  Rej Z2: {rej2}")
        summary_lines.append(line2)

    # ── Save ──────────────────────────────────────────────────────────────────
    save    = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(OUT_FILE, **save)
    total_n = len(arr["r_AU"])
    n_z3    = int(np.sum(np.array(arr["zone"]) == 3.0))
    n_z2    = int(np.sum(np.array(arr["zone"]) == 2.0))
    size_kb = os.path.getsize(OUT_FILE) / 1024
    elapsed = time.perf_counter() - t0_all

    print(f"\nSaved {OUT_FILE}")
    print(f"  Total={total_n}  Zone3={n_z3}  Zone2={n_z2}  ({size_kb:.0f} KB)")
    print(f"  Wall time: {elapsed:.0f}s")

    with open(SUMMARY, "w") as fh:
        fh.write("CLOSE-ENCOUNTER DATASET SUMMARY\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Total  : {total_n}  (Zone3={n_z3}, Zone2={n_z2})\n")
        fh.write(f"DT     : {DT_VALUES}\n")
        fh.write(f"Zone 3 : r in ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU\n")
        fh.write(f"Zone 2 : r in ({Z2_R_MIN:.2e}, {Z2_R_MAX:.4f}) AU\n")
        fh.write(f"R_GATE : {R_GATE:.2e} AU\n\n")
        for ln in summary_lines:
            fh.write(ln + "\n")
        fh.write("\nKEY CHECKS:\n")
        fh.write("  Zone 3: c_opt_med < 1 and dt-dependent at large dt? YES -> proceed.\n")
        fh.write("  Zone 2: c_opt_med ~= c_ana_med ~= 1.0? YES -> boundary OK.\n")
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
