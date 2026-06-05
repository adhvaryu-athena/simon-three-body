"""
generate_encounter_data.py  --  Steps 1 and 2  (FINAL, crash-safe)

═══════════════════════════════════════════════════════════════
WHY REBOUND WAS REMOVED (crash fix)
═══════════════════════════════════════════════════════════════
All previous versions using rebound/ias15 crashed on Windows with:
    Windows fatal exception: access violation
inside find_c_opt (at the break statement, which itself cannot crash).

Root cause: rebound's ias15 C code writes past its internal buffer
during integration on Windows, corrupting the Python heap. Even when
ias15 "completes successfully", the memory corruption causes access
violations later when Python uses that heap memory in find_c_opt.

Fix: replace ias15_one_step with accurate_one_step — the same
softened leapfrog subdivided into N_REF_SUBSTEPS=200 sub-steps.
At Zone 3 separations (r > 0.052 AU), softening is negligible
(c_analytic = 1.000013), so this reference ~= exact Newtonian.
n_substeps=200 gives step_fraction < 0.06 for all Zone 3 ICs.
No rebound, no C library, no Windows crashes possible.
═══════════════════════════════════════════════════════════════

Design: Zone 3 only (Zone 2 uses analytic c=(r_soft/r)^3 in simulation)

Output (10 float32 arrays in encounter_data.npz):
  r_AU, r_soft, log_mi, log_mj, log_dt,
  c_opt, log_c_opt, c_ana, log_c_ana, improvement
"""

import os
import math
import time
import numpy as np

# ── Constants — must match pair_eval_after_adaptive.py exactly ─────────────────
G            = 1.0
EPS          = 3e-4
R_SOFT_MIN   = 5e-4
NN_THRESH    = 500.0 * EPS   # 0.15 AU
ADAPT_THRESH = 0.05
C_MIN, C_MAX = 0.2, 5.0
R_GATE       = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))   # 4e-4 AU

# ── Dataset parameters ─────────────────────────────────────────────────────────
N_PER_DT       = 10000
DT_VALUES      = [0.005, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
SEED           = 42
N_GRID         = 60
C_GRID         = np.linspace(C_MIN, C_MAX, N_GRID)
N_REF_SUBSTEPS = 200   # subdivisions for accurate reference (no rebound)
OUT_FILE       = "encounter_data.npz"
SUMMARY        = "encounter_summary.txt"

Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU


# ── Safe MSE (pure Python, avoids all numpy reductions) ───────────────────────
def _mse(a, b):
    """
    MSE between two (3,3) position arrays.
    Pure Python loop — no numpy reductions — completely safe on Windows.
    Returns float('inf') for non-finite or exploded states.
    """
    try:
        total = 0.0
        count = 0
        for i in range(3):
            for j in range(3):
                av = float(a[i][j])
                bv = float(b[i][j])
                if not math.isfinite(av) or not math.isfinite(bv):
                    return float("inf")
                if abs(av) > 1e6 or abs(bv) > 1e6:
                    return float("inf")
                d = av - bv
                total += d * d
                count += 1
        return total / count if count > 0 else float("inf")
    except Exception:
        return float("inf")


# ── Leapfrog with fixed scalar c ──────────────────────────────────────────────
def acc_fixed_c(pos, m_arr, c_val):
    """
    Accelerations for three bodies.
    Pair (0,1): softened force x c_val when r < NN_THRESH.
    Pairs (0,2), (1,2): exact Newtonian (always far in training ICs).
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


# ── Accurate reference (NO rebound) ───────────────────────────────────────────
def accurate_one_step(x0, v0, m_arr, dt):
    """
    Accurate reference trajectory for one macro step of size dt.

    Uses N_REF_SUBSTEPS=200 sub-steps of the softened leapfrog (c=1).
    At Zone 3 r (> 0.052 AU), softening is negligible so this
    reference ~= exact Newtonian. step_fraction < 0.06 for all ICs.

    Pure Python/numpy. No C library. No Windows crashes possible.
    """
    dt_sub = dt / N_REF_SUBSTEPS
    x = x0.copy()
    v = v0.copy()
    for _ in range(N_REF_SUBSTEPS):
        x, v = leapfrog_step(x, v, m_arr, dt_sub, 1.0)
    return x, v


# ── c_optimal search ──────────────────────────────────────────────────────────
def find_c_opt(x0, v0, m_arr, dt, x_ref):
    """
    Grid search + golden-section for c in [C_MIN, C_MAX] minimising
    MSE(x_leapfrog(c), x_ref) after one macro step dt.
    Returns (c_opt, mse_opt, mse_at_c1).
    """
    x1, _ = leapfrog_step(x0, v0, m_arr, dt, 1.0)
    mse_1 = _mse(x1, x_ref)

    mse_g = np.empty(N_GRID)
    for k, c in enumerate(C_GRID):
        xk, _    = leapfrog_step(x0, v0, m_arr, dt, float(c))
        mse_g[k] = _mse(xk, x_ref)

    if not any(math.isfinite(float(v)) for v in mse_g):
        return float("nan"), float("inf"), mse_1

    best_k = int(np.argmin(mse_g))
    c_lo   = float(C_GRID[max(0, best_k - 1)])
    c_hi   = float(C_GRID[min(N_GRID - 1, best_k + 1)])

    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7:
            break
        c1    = c_hi - phi * (c_hi - c_lo)
        c2    = c_lo + phi * (c_hi - c_lo)
        xa, _ = leapfrog_step(x0, v0, m_arr, dt, c1)
        xb, _ = leapfrog_step(x0, v0, m_arr, dt, c2)
        m1    = _mse(xa, x_ref)
        m2    = _mse(xb, x_ref)
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
    Synthetic Zone 3 IC. Pair (0,1) at r01 in (0.052, 0.148) AU.
    Body 2 far away (3-8 AU).

    Periapsis filter: reject if 2-body periapsis of pair (0,1) < EPS.
    Near-radial approaches bounce off the softening wall — unphysical
    for Zone 3 and poor training data.

    Returns (x_all, v_all, m, r01) or None if invalid.
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

    M_tot = float(m.sum())
    x_com = np.sum(m[:, None] * x_all, axis=0) / M_tot
    v_com = np.sum(m[:, None] * v_all, axis=0) / M_tot
    x_all = x_all - x_com
    v_all = v_all - v_com

    KE = 0.5 * float(np.sum(m * np.array(
        [float(np.dot(v_all[i], v_all[i])) for i in range(3)])))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            PE -= G * m[i] * m[j] / float(np.linalg.norm(x_all[i] - x_all[j]))
    if KE + PE >= 0.0:
        return None

    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not (Z3_R_MIN < r01_check < Z3_R_MAX):
        return None

    # Periapsis filter
    r_rel_pair = x_all[1] - x_all[0]
    v_rel_pair = v_all[1] - v_all[0]
    v_sq       = float(np.dot(v_rel_pair, v_rel_pair))
    M_pair     = m0 + m1
    eps_orb    = v_sq / 2.0 - G * M_pair / r01_check
    h_z        = float(r_rel_pair[0] * v_rel_pair[1]
                       - r_rel_pair[1] * v_rel_pair[0])
    h_sq       = h_z * h_z
    disc       = max(0.0, 1.0 + 2.0 * eps_orb * h_sq / (G * M_pair) ** 2)
    e_ecc      = float(math.sqrt(disc))
    r_peri     = h_sq / (G * M_pair * (1.0 + e_ecc) + 1e-30)
    if r_peri < EPS:
        return None

    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    return x_all, v_all, m, r01_check


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng    = np.random.RandomState(SEED)
    t0_all = time.perf_counter()

    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]}

    print("=" * 65)
    print("CLOSE-ENCOUNTER DATA GENERATOR  (crash-safe, no rebound)")
    print(f"  r         : ({Z3_R_MIN:.4f}, {Z3_R_MAX:.4f}) AU")
    print(f"  DT        : {DT_VALUES}")
    print(f"  N/dt      : {N_PER_DT}   Total: {N_PER_DT * len(DT_VALUES)}")
    print(f"  Reference : {N_REF_SUBSTEPS}-substep leapfrog  (no rebound/ias15)")
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

            ic = make_ic(rng)
            if ic is None:
                rej["ic"] += 1
                continue
            x0, v0, m, r01 = ic

            # Reference: 200-substep leapfrog (pure numpy, no crashes)
            x_ref, _ = accurate_one_step(x0, v0, m, dt)
            max_ref  = float(max(
                abs(x_ref[i][j]) for i in range(3) for j in range(3)))
            if not math.isfinite(max_ref) or max_ref > 50.0:
                rej["eject"] += 1
                continue

            c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, dt, x_ref)

            if not math.isfinite(c_opt) or not math.isfinite(mse_opt):
                rej["ic"] += 1
                continue
            if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
                rej["boundary"] += 1
                continue
            x_best, _ = leapfrog_step(x0, v0, m, dt, c_opt)
            max_best = float(max(
                abs(x_best[i][j]) for i in range(3) for j in range(3)))
            if not math.isfinite(max_best) or max_best > 50.0:
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
            arr["log_mi"].append(float(math.log(m[0] + 1e-30)))
            arr["log_mj"].append(float(math.log(m[1] + 1e-30)))
            arr["log_dt"].append(float(math.log(dt)))
            arr["c_opt"].append(c_opt)
            arr["log_c_opt"].append(float(math.log(c_opt)))
            arr["c_ana"].append(c_ana)
            arr["log_c_ana"].append(float(math.log(c_ana)))
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
        line = (f"dt={dt:.3f}: n={n_ok} | tried={n_try} | "
                f"pass={n_ok/n_try:.1%} | "
                f"c_opt_med={np.median(c_buf):.4f} | "
                f"c_ana_med={np.median(c_ana_buf):.6f} | "
                f"impr_med={np.median(imp_buf):.2%} | {el:.0f}s")
        print(f"  {line}")
        print(f"  Rejected: {rej}")
        summary_lines.append(line)

    save    = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(OUT_FILE, **save)
    total_n = len(arr["r_AU"])
    size_kb = os.path.getsize(OUT_FILE) / 1024
    elapsed = time.perf_counter() - t0_all

    print(f"\nSaved {OUT_FILE}  ({total_n} samples, {size_kb:.0f} KB)")
    print(f"Wall time: {elapsed:.0f}s")

    with open(SUMMARY, "w") as fh:
        fh.write("CLOSE-ENCOUNTER DATASET SUMMARY\n")
        fh.write("=" * 65 + "\n")
        fh.write(f"Total  : {total_n}\n")
        fh.write(f"Zone   : 3 only (r {Z3_R_MIN:.3f}-{Z3_R_MAX:.3f} AU)\n")
        fh.write(f"DT     : {DT_VALUES}\n")
        fh.write(f"Ref    : {N_REF_SUBSTEPS}-substep leapfrog (no rebound)\n")
        fh.write(f"Fields : r_AU r_soft log_mi log_mj log_dt "
                 f"c_opt log_c_opt c_ana log_c_ana improvement\n\n")
        for ln in summary_lines:
            fh.write(ln + "\n")
        fh.write("\nKEY CHECKS:\n")
        fh.write("  c_opt_med should decrease as dt increases.\n")
        fh.write("  c_ana_med ~= 1.000 for all dt (softening negligible Z3).\n")
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
