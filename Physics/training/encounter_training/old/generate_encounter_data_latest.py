"""
generate_encounter_data.py  --  Steps 1 and 2 (REVISED)

Fixes from original:
  Bug1  Lines 138-139: np.mean in golden-section refinement replaced with _mse()
        (NumPy version compatibility fix -- all 5 MSE calls now use _mse)
  Bug2  Line 314: summary c_ana was using last r01/r_soft from loop only.
        Now computes median from the full c_ana buffer.
  Bug3  make_ic validity check was hardcoded for Zone 3 (ADAPT_THRESH<r<NN_THRESH).
        Would always reject Zone 2 ICs. Now parametrised via r_min/r_max.

New features:
  - dt=0.06 added to DT_VALUES (covers all paper energy-drift sweep values)
  - Zone 2 (sub-step zone): r in (Z2_R_MIN, Z2_R_MAX) = (0.010, 0.048) AU
      For each sample, effective sub_dt = macro_dt / n_sub is computed from r01.
      log_dt stored = log(sub_dt)  -- this is the NN input at inference time.
  - Zone 3 (macro-step zone): r in (Z3_R_MIN, Z3_R_MAX) = (0.052, 0.148) AU
      log_dt stored = log(macro_dt)  -- unchanged from original.
  - Two new stored fields: zone (2 or 3), log_macro_dt (for grouping in inspect).
  - MAX_TRIES_Z2 prevents infinite loops if Zone 2 acceptance rate is very low.

Physics:
  Zone 3 (0.052-0.148 AU, no sub-stepping): c_analytic ≈ 1.0 (softening
    negligible); c_optimal < 1 due to leapfrog discretisation at large macro_dt.
    This is the primary value-add of trajectory-trained c.
  Zone 2 (0.010-0.048 AU, inside sub-steps): c_analytic ≈ 1.0-1.014;
    effective sub_dt is small so encounters better resolved; c_optimal ≈ 1.0.
    These samples teach the NN the correct boundary: small effective_dt -> c≈1.

Output:
  encounter_data.npz    -- dataset (numpy)
  encounter_summary.txt -- per-zone per-dt statistics
"""

import os, time
import numpy as np
import rebound

# ── Constants (must match pair_eval_after_adaptive.py exactly) ─────────────────
G            = 1.0
EPS          = 3e-4            # cfg.eps  (gravitational softening length)
R_SOFT_MIN   = 5e-4            # cfg.r_soft_min  (safety gate threshold on r_soft)
NN_THRESH    = 500.0 * EPS     # nn_thresh  = 0.15 AU
ADAPT_THRESH = 0.05            # adapt_thresh = 0.05 AU
MAX_SUBSTEPS = 16              # max_substeps
C_MIN, C_MAX = 0.2, 5.0

# R_GATE: the r value where r_soft = R_SOFT_MIN (actual lower NN-active boundary)
# Derived: r_soft = sqrt(r^2 + eps^2) = R_SOFT_MIN => r = sqrt(R_SOFT_MIN^2 - EPS^2)
R_GATE = np.sqrt(R_SOFT_MIN**2 - EPS**2)   # = 4e-4 AU exactly

# ── Dataset parameters ─────────────────────────────────────────────────────────
N_PER_DT     = 10000       # accepted samples per (macro_dt, zone=3)
N_PER_DT_Z2  = 3000        # accepted samples per (macro_dt, zone=2)
MAX_TRIES_Z2 = 80000       # give up Zone 2 for this macro_dt if hit this limit

# All macro dt values used in the paper (energy-drift sweep + frontier sweep)
DT_VALUES = [0.005, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]

SEED     = 42
N_GRID   = 60
C_GRID   = np.linspace(C_MIN, C_MAX, N_GRID)
OUT_FILE = "encounter_data.npz"
SUMMARY  = "encounter_summary.txt"

# Zone r boundaries (small margin inside true limits to avoid edge effects)
Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU  macro-step zone
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU
Z2_R_MIN = 0.010                  # 0.010 AU  sub-step zone
# Lower bound chosen at 0.01 rather than R_GATE (4e-4) to keep ias15 runtime
# manageable. Below 0.01 AU the encounter is completely unresolvable at any
# sub_dt (step_fraction >> 1) so samples would be rejected by the filter anyway.
Z2_R_MAX = ADAPT_THRESH - 0.002   # 0.048 AU


# ── MSE helper (avoids np.mean keyword issue on older NumPy) ──────────────────
def _mse(a, b):
    """Mean squared error between two (3,3) position arrays. Uses .sum() only."""
    d = a - b
    return float((d * d).sum()) / float(d.size)


# ── ias15 one-step ground truth ───────────────────────────────────────────────
def ias15_one_step(x0, v0, m, dt):
    """
    Advance three-body system from t=0 to t=dt using ias15 (G=1).
    Returns x_new (3,3), v_new (3,3).
    dt may be a macro_dt (Zone 3) or an effective sub_dt (Zone 2).
    """
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(3):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    sim.integrate(float(dt))
    x_new = np.array([[p.x,  p.y,  p.z]  for p in sim.particles])
    v_new = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    return x_new, v_new


# ── One-step leapfrog with fixed scalar c ─────────────────────────────────────
def acc_fixed_c(pos, m, c_val):
    """
    Accelerations for three bodies.
    Pair (0,1): uses softened force × c_val when r < NN_THRESH.
    Pairs (0,2), (1,2): exact Newtonian (they are far apart in all our ICs).
    """
    acc   = np.zeros((3, 3), dtype=np.float64)
    pairs = [(0, 1), (0, 2), (1, 2)]
    eps2  = EPS * EPS
    for idx, (i, j) in enumerate(pairs):
        rij     = pos[j] - pos[i]
        r2      = float(np.dot(rij, rij))
        r       = (r2 + 1e-30) ** 0.5
        Gmimj   = G * m[i] * m[j]
        if idx == 0 and r < NN_THRESH:
            # Close pair: softened force scaled by c_val
            denom    = (r2 + eps2) ** 1.5 + 1e-30
            F_scalar = c_val * Gmimj / denom
        else:
            # Exact Newtonian
            F_scalar = Gmimj / (r2 * r + 1e-30)
        F_vec    = F_scalar * rij
        acc[i]  +=  F_vec / m[i]
        acc[j]  -= F_vec / m[j]
    return acc


def leapfrog_step(x0, v0, m, dt, c_val):
    """One velocity-Verlet step with fixed scalar correction c_val."""
    a0     = acc_fixed_c(x0, m, c_val)
    v_half = v0 + 0.5 * dt * a0
    x1     = x0 + dt * v_half
    a1     = acc_fixed_c(x1, m, c_val)
    v1     = v_half + 0.5 * dt * a1
    return x1, v1


# ── c_optimal search ──────────────────────────────────────────────────────────
def find_c_opt(x0, v0, m, dt, x_ref):
    """
    Grid search + golden-section refinement for c in [C_MIN, C_MAX] that
    minimises MSE(x_leapfrog(c), x_ref) after one step of size dt.

    dt is the effective step size (macro_dt for Zone 3, sub_dt for Zone 2).
    Returns (c_opt, mse_opt, mse_at_c1).
    All MSE calls use _mse() to avoid np.mean compatibility issues.
    """
    # Baseline: c = 1 (softened Newtonian, no correction)
    x1, _ = leapfrog_step(x0, v0, m, dt, 1.0)
    mse_1 = _mse(x1, x_ref)

    # Coarse grid search across [C_MIN, C_MAX]
    mse_g = np.empty(N_GRID)
    for k, c in enumerate(C_GRID):
        xk, _    = leapfrog_step(x0, v0, m, dt, c)
        mse_g[k] = _mse(xk, x_ref)

    best_k = int(np.argmin(mse_g))
    c_lo   = C_GRID[max(0, best_k - 1)]
    c_hi   = C_GRID[min(N_GRID - 1, best_k + 1)]

    # Golden-section refinement within the best grid interval
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7:
            break
        c1     = c_hi - phi * (c_hi - c_lo)
        c2     = c_lo + phi * (c_hi - c_lo)
        xa, _  = leapfrog_step(x0, v0, m, dt, c1)
        xb, _  = leapfrog_step(x0, v0, m, dt, c2)
        m1     = _mse(xa, x_ref)   # Bug1 fix: was np.mean()
        m2     = _mse(xb, x_ref)   # Bug1 fix: was np.mean()
        if m1 < m2:
            c_hi = c2
        else:
            c_lo = c1

    c_opt    = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step(x0, v0, m, dt, c_opt)
    mse_opt  = _mse(x_opt, x_ref)
    return c_opt, mse_opt, mse_1


# ── Effective sub_dt for Zone 2 ───────────────────────────────────────────────
def get_effective_dt(macro_dt, r01):
    """
    Compute the sub_dt that SIMON actually uses for a close pair at separation
    r01 < ADAPT_THRESH. Mirrors the sub-stepping formula in the simulation:
        n_sub = min(MAX_SUBSTEPS, max(2, ceil(ADAPT_THRESH / r01)))
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
    Pair (0,1) is at separation r01 in (r_min, r_max).
    Body 2 is far away (3-8 AU) to isolate the close pair interaction.

    Bug3 fix: r validity check now uses r_min/r_max instead of hardcoded
    ADAPT_THRESH/NN_THRESH, so this works for both Zone 2 and Zone 3 ICs.

    Returns (x_all, v_all, m, r01) or None if invalid.
    """
    # Log-uniform masses in SIMON training range
    m0 = np.exp(rng.uniform(np.log(0.001), np.log(2.0)))
    m1 = np.exp(rng.uniform(np.log(0.001), np.log(2.0)))
    m2 = np.exp(rng.uniform(np.log(0.001), np.log(2.0)))
    m  = np.array([m0, m1, m2])

    # Close pair at r01 in (r_min, r_max)
    r01    = rng.uniform(r_min, r_max)
    theta  = rng.uniform(0.0, 2.0 * np.pi)
    r_hat  = np.array([np.cos(theta), np.sin(theta), 0.0])

    # Relative velocity: random direction, magnitude f × v_circ
    f      = rng.uniform(0.4, 1.8)          # covers approaching and receding
    v_circ = np.sqrt(G * (m0 + m1) / r01)
    phi_v  = rng.uniform(0.0, 2.0 * np.pi)
    v_hat  = np.array([np.cos(phi_v), np.sin(phi_v), 0.0])
    v_rel  = f * v_circ * v_hat

    # Body positions from pair CoM
    x_rel = r01 * r_hat
    x0p   = -(m1 / (m0 + m1)) * x_rel
    x1p   =  (m0 / (m0 + m1)) * x_rel
    v0p   = -(m1 / (m0 + m1)) * v_rel
    v1p   =  (m0 / (m0 + m1)) * v_rel

    # Body 2: far, roughly circular orbit around pair CoM
    r2    = rng.uniform(3.0, 8.0)
    th2   = rng.uniform(0.0, 2.0 * np.pi)
    x2p   = r2 * np.array([np.cos(th2), np.sin(th2), 0.0])
    f2    = rng.uniform(0.5, 1.1)
    v_c2  = np.sqrt(G * (m0 + m1) / r2)
    v2p   = f2 * v_c2 * np.array([-np.sin(th2), np.cos(th2), 0.0])

    x_all = np.array([x0p, x1p, x2p])
    v_all = np.array([v0p, v1p, v2p])

    # Shift to CoM frame
    M     = m.sum()
    x_all -= (m[:, None] * x_all).sum(0) / M
    v_all -= (m[:, None] * v_all).sum(0) / M

    # Reject if system is unbound
    KE = 0.5 * float(np.sum(m * np.sum(v_all**2, axis=1)))
    PE = sum(-G * m[i] * m[j] / float(np.linalg.norm(x_all[i] - x_all[j]))
             for i in range(3) for j in range(i + 1, 3))
    if KE + PE >= 0.0:
        return None

    # Reject if close pair has drifted outside the target r range (after CoM shift)
    r01_check = float(np.linalg.norm(x_all[0] - x_all[1]))
    if not (r_min < r01_check < r_max):   # Bug3 fix: parametrised bounds
        return None

    # Reject if the other pairs are too close (would confound the close-pair signal)
    for i, j in [(0, 2), (1, 2)]:
        if float(np.linalg.norm(x_all[i] - x_all[j])) < NN_THRESH:
            return None

    return x_all, v_all, m, r01_check


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng    = np.random.RandomState(SEED)
    t0_all = time.perf_counter()

    # Storage: two new fields -- zone and log_macro_dt
    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj",
        "log_dt",        # log(effective_dt): NN input (sub_dt for Z2, macro_dt for Z3)
        "log_macro_dt",  # log(macro_dt): used for grouping in inspect script
        "c_opt", "log_c_opt",
        "c_ana", "log_c_ana",
        "improvement",
        "zone",          # 2 or 3
    ]}

    print("=" * 70)
    print("CLOSE-ENCOUNTER DATA GENERATOR  (REVISED)")
    print(f"  Zone 3 (macro-step): r in ({Z3_R_MIN:.3f}, {Z3_R_MAX:.3f}) AU"
          f"  target {N_PER_DT} samples per dt")
    print(f"  Zone 2 (sub-step):   r in ({Z2_R_MIN:.3f}, {Z2_R_MAX:.3f}) AU"
          f"  target {N_PER_DT_Z2} samples per dt")
    print(f"  DT values: {DT_VALUES}")
    print(f"  R_GATE = {R_GATE:.2e} AU  (lower NN-active boundary)")
    print("=" * 70)

    summary_lines = []

    for macro_dt in DT_VALUES:

        # ──────────────────────────────────────────────────────────────────────
        # ZONE 3: macro-step zone, effective_dt = macro_dt
        # ──────────────────────────────────────────────────────────────────────
        n_ok  = 0
        n_try = 0
        rej   = dict(ic=0, boundary=0, eject=0, no_improve=0)
        t_dt  = time.perf_counter()
        c_buf, imp_buf, c_ana_buf = [], [], []

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
            if np.max(np.linalg.norm(x_ref, axis=1)) > 50.0:
                rej["eject"] += 1
                continue

            c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, macro_dt, x_ref)

            if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
                rej["boundary"] += 1
                continue
            x_best, _ = leapfrog_step(x0, v0, m, macro_dt, c_opt)
            if np.max(np.linalg.norm(x_best, axis=1)) > 50.0:
                rej["eject"] += 1
                continue
            if mse_1 < 1e-20 or (mse_opt / mse_1) > 0.95:
                rej["no_improve"] += 1
                continue

            # Accept
            r_soft = float(np.sqrt(r01**2 + EPS**2))
            c_ana  = (r_soft / r01) ** 3
            imp    = (mse_1 - mse_opt) / mse_1

            arr["r_AU"].append(float(r01))
            arr["r_soft"].append(r_soft)
            arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
            arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
            arr["log_dt"].append(float(np.log(macro_dt)))        # = log(effective_dt)
            arr["log_macro_dt"].append(float(np.log(macro_dt)))  # same for Zone 3
            arr["c_opt"].append(float(c_opt))
            arr["log_c_opt"].append(float(np.log(c_opt)))
            arr["c_ana"].append(float(c_ana))
            arr["log_c_ana"].append(float(np.log(c_ana)))
            arr["improvement"].append(float(imp))
            arr["zone"].append(3.0)

            c_buf.append(c_opt)
            imp_buf.append(imp)
            c_ana_buf.append(c_ana)
            n_ok += 1

            if n_ok % 2000 == 0:
                el = time.perf_counter() - t_dt
                print(f"  Z3 dt={macro_dt:.3f}  {n_ok}/{N_PER_DT}"
                      f"  tried={n_try}  {el:.0f}s")

        el_dt = time.perf_counter() - t_dt
        # Bug2 fix: compute median c_ana from buffer, not from last r01
        line3 = (f"Z3 dt={macro_dt:.3f}: n={n_ok} | tried={n_try} | "
                 f"pass={n_ok/n_try:.1%} | "
                 f"c_opt med={np.median(c_buf):.4f} | "
                 f"c_ana med={np.median(c_ana_buf):.6f} | "
                 f"impr med={np.median(imp_buf):.2%} | {el_dt:.0f}s")
        print(f"  {line3}")
        print(f"  Rejected Z3: {rej}")
        summary_lines.append(line3)

        # ──────────────────────────────────────────────────────────────────────
        # ZONE 2: sub-step zone, effective_dt = sub_dt = macro_dt / n_sub
        # ──────────────────────────────────────────────────────────────────────
        n_ok2  = 0
        n_try2 = 0
        rej2   = dict(ic=0, boundary=0, eject=0, no_improve=0)
        t_dt2  = time.perf_counter()
        c_buf2, imp_buf2, c_ana_buf2 = [], [], []

        while n_ok2 < N_PER_DT_Z2 and n_try2 < MAX_TRIES_Z2:
            n_try2 += 1
            ic = make_ic(rng, Z2_R_MIN, Z2_R_MAX)
            if ic is None:
                rej2["ic"] += 1
                continue
            x0, v0, m, r01 = ic

            # Effective dt for this sample: the sub_dt SIMON would use
            sub_dt, n_sub = get_effective_dt(macro_dt, r01)

            try:
                x_ref, _ = ias15_one_step(x0, v0, m, sub_dt)
            except Exception:
                rej2["ic"] += 1
                continue
            if np.max(np.linalg.norm(x_ref, axis=1)) > 50.0:
                rej2["eject"] += 1
                continue

            c_opt, mse_opt, mse_1 = find_c_opt(x0, v0, m, sub_dt, x_ref)

            if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
                rej2["boundary"] += 1
                continue
            x_best, _ = leapfrog_step(x0, v0, m, sub_dt, c_opt)
            if np.max(np.linalg.norm(x_best, axis=1)) > 50.0:
                rej2["eject"] += 1
                continue
            if mse_1 < 1e-20 or (mse_opt / mse_1) > 0.95:
                rej2["no_improve"] += 1
                continue

            # Accept
            r_soft = float(np.sqrt(r01**2 + EPS**2))
            c_ana  = (r_soft / r01) ** 3
            imp    = (mse_1 - mse_opt) / mse_1

            arr["r_AU"].append(float(r01))
            arr["r_soft"].append(r_soft)
            arr["log_mi"].append(float(np.log(m[0] + 1e-30)))
            arr["log_mj"].append(float(np.log(m[1] + 1e-30)))
            arr["log_dt"].append(float(np.log(sub_dt)))          # NN input: log(sub_dt)
            arr["log_macro_dt"].append(float(np.log(macro_dt)))  # for grouping
            arr["c_opt"].append(float(c_opt))
            arr["log_c_opt"].append(float(np.log(c_opt)))
            arr["c_ana"].append(float(c_ana))
            arr["log_c_ana"].append(float(np.log(c_ana)))
            arr["improvement"].append(float(imp))
            arr["zone"].append(2.0)

            c_buf2.append(c_opt)
            imp_buf2.append(imp)
            c_ana_buf2.append(c_ana)
            n_ok2 += 1

            if n_ok2 % 500 == 0:
                el = time.perf_counter() - t_dt2
                print(f"  Z2 dt={macro_dt:.3f}  {n_ok2}/{N_PER_DT_Z2}"
                      f"  tried={n_try2}  {el:.0f}s")

        el_dt2   = time.perf_counter() - t_dt2
        hit_limit = n_try2 >= MAX_TRIES_Z2 and n_ok2 < N_PER_DT_Z2
        suffix    = f" [HIT MAX_TRIES: only {n_ok2}/{N_PER_DT_Z2} collected]" \
                    if hit_limit else ""
        if n_ok2 > 0:
            line2 = (f"Z2 dt={macro_dt:.3f}: n={n_ok2} | tried={n_try2} | "
                     f"pass={n_ok2/max(n_try2,1):.1%} | "
                     f"c_opt med={np.median(c_buf2):.4f} | "
                     f"c_ana med={np.median(c_ana_buf2):.6f} | "
                     f"impr med={np.median(imp_buf2):.2%} | {el_dt2:.0f}s{suffix}")
        else:
            line2 = (f"Z2 dt={macro_dt:.3f}: n=0 | tried={n_try2} | "
                     f"NO SAMPLES COLLECTED{suffix}")
        print(f"  {line2}")
        print(f"  Rejected Z2: {rej2}")
        summary_lines.append(line2)

    # ── Save ──────────────────────────────────────────────────────────────────
    save    = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(OUT_FILE, **save)
    total_n = len(arr["r_AU"])
    n_z3    = int(np.sum(np.array(arr["zone"]) == 3.0))
    n_z2    = int(np.sum(np.array(arr["zone"]) == 2.0))
    size_kb = os.path.getsize(OUT_FILE) / 1024

    print(f"\nSaved {OUT_FILE}  ({total_n} total samples:"
          f" {n_z3} Zone-3, {n_z2} Zone-2, {size_kb:.0f} KB)")
    print(f"Total wall time: {time.perf_counter() - t0_all:.0f}s")

    # ── Summary file ──────────────────────────────────────────────────────────
    with open(SUMMARY, "w") as f:
        f.write("CLOSE-ENCOUNTER DATASET SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples : {total_n}  (Zone-3: {n_z3}, Zone-2: {n_z2})\n")
        f.write(f"DT values     : {DT_VALUES}\n")
        f.write(f"Zone 3 r range: ({Z3_R_MIN:.3f}, {Z3_R_MAX:.3f}) AU\n")
        f.write(f"Zone 2 r range: ({Z2_R_MIN:.3f}, {Z2_R_MAX:.3f}) AU\n")
        f.write(f"R_GATE        : {R_GATE:.2e} AU (lower NN-active boundary)\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\nKEY CHECKS:\n")
        f.write("  Zone 3: c_opt should be < 1.0 and dt-dependent for large dt.\n")
        f.write("    c_opt med << c_ana med (~1.0) at dt=0.04, 0.08 -> value-add confirmed.\n")
        f.write("  Zone 2: c_opt should be ≈ c_analytic (both ≈ 1.0 for r>0.005).\n")
        f.write("    This validates the NN boundary condition at small effective_dt.\n")
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
