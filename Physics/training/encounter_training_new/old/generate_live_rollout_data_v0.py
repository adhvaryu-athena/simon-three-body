"""
generate_live_rollout_data.py  --  Zone 3 live-rollout state harvester, Phase 3

Purpose
-------
The Phase 2 generator used synthetic ICs (isolated pair + distant third body).
This produced a distribution shift: at deployment the NN sees velocity states
shaped by actual 3-body orbital history, not isolated-pair sampling.

This script fixes that by:
  1. Running actual SIMON leapfrog trajectories on known stable ICs (IC1, IC3, IC4).
  2. Recording every Zone 3 trigger event: the exact (x0, v0, m) state just
     before a macro-step where pair (0,1) is in Zone 3 and NOT in Zone 2.
  3. Running find_c_opt on each recorded state to get the trajectory-optimal c.
  4. Saving results as shard .npz files in the same format as the batch generator.

These are "on-manifold" samples -- they come from the actual deployment trajectory.
Even ~100-150 such samples per IC per dt substantially reduces the distribution shift.

Zone 3 condition (consistent with evaluator):
  adapt_thresh <= r_01 < nn_thresh   (0.05 <= r < 0.15 AU)
  AND r_01 >= adapt_thresh            (NOT in Zone 2 / substepping)

Usage:
    python -B generate_live_rollout_data.py --ic IC1 --dt 0.04 --T 50 --batch 1
    python -B generate_live_rollout_data.py --ic IC3 --dt 0.08 --T 50 --batch 1
    python -B generate_live_rollout_data.py --ic IC4 --dt 0.04 --T 50 --batch 1

    Valid --ic choices: IC1, IC3, IC4  (IC2/IC5/IC7 eject; IC6/IC4 have few Zone 3 events)
    Valid --dt values:  0.02, 0.04, 0.05, 0.06, 0.08, 0.10

Output: encounter_shards/encounter_data_zone3_p3_live_{IC}_{dt}_batch{N:03d}.npz
        Same fields as generate_encounter_data_phase3_batch.py.
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
# Constants (must match evaluator and batch generator)
# =============================================================================
G            = 1.0
EPS          = 3e-4
R_SOFT_MIN   = 5e-4
NN_THRESH    = 500.0 * EPS      # 0.15 AU
ADAPT_THRESH = 0.05             # Zone 2/3 boundary
C_MIN, C_MAX = 0.2, 5.0

Z3_R_MIN = ADAPT_THRESH + 0.002   # 0.052 AU
Z3_R_MAX = NN_THRESH    - 0.002   # 0.148 AU

VALID_DTS = {0.020, 0.040, 0.050, 0.060, 0.080, 0.100}

N_GRID = 60
C_GRID = np.linspace(C_MIN, C_MAX, N_GRID)
IAS15_TIMEOUT_SEC = 60.0


# =============================================================================
# Initial conditions (from multi_ic_eval_v3.py -- verified against Phase 1 paper)
# IC2/IC5/IC7 eject. IC6 near-circular (NN never invoked). IC4 hierarchical
# (Zone 3 rarely triggered). Use IC1, IC3 as primary sources; IC4 as supplement.
# =============================================================================
def get_ic(name: str):
    """Return (x0, v0, m) in CoM frame for the requested IC."""
    ICS_RAW = {
        "IC1": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[1,0,0],[0,1.2,0]],    dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1,0],[-0.9,0,0]],   dtype=np.float64),
        },
        "IC3": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[0.5,0,0],[0,2.5,0]], dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1.3,0],[-0.4,0,0]], dtype=np.float64),
        },
        "IC4": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[1,0,0],[0,5.0,0]],    dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1.0,0],[-0.12,0,0]], dtype=np.float64),
        },
    }
    if name not in ICS_RAW:
        raise ValueError(f"Unknown IC: {name}. Valid: {list(ICS_RAW.keys())}")
    ic = ICS_RAW[name]
    m  = ic["m"]
    x0 = ic["x0"].copy()
    v0 = ic["v0"].copy()
    M  = m.sum()
    x0 -= (m[:, None] * x0).sum(0) / M
    v0 -= (m[:, None] * v0).sum(0) / M
    return x0, v0, m


# =============================================================================
# Utility helpers
# =============================================================================
def dt_token(dt):
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _mse(a, b):
    try:
        if a is None or b is None: return float("inf")
        total = 0.0; count = 0
        for i in range(len(a)):
            for j in range(len(a[i])):
                av = float(a[i][j]); bv = float(b[i][j])
                if not math.isfinite(av) or not math.isfinite(bv): return float("inf")
                if abs(av) > 1e6 or abs(bv) > 1e6: return float("inf")
                total += (av - bv) ** 2; count += 1
        if count == 0: return float("inf")
        val = total / count
        return float(val) if math.isfinite(val) else float("inf")
    except Exception:
        return float("inf")


def safe_max_norm(x):
    try:
        mx = 0.0
        for row in x:
            n2 = sum(float(v)**2 for v in row)
            mx = max(mx, math.sqrt(n2))
        return mx
    except Exception:
        return float("inf")


# =============================================================================
# IAS15 one-step isolated subprocess
# =============================================================================
def ias15_one_step(x0, v0, m_arr, dt):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    sim.exit_min_distance = float(EPS)
    for i in range(3):
        sim.add(m=float(m_arr[i]),
                x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    sim.integrate(float(dt))
    xn = np.array([[p.x, p.y, p.z]  for p in sim.particles], dtype=np.float64)
    vn = np.array([[p.vx,p.vy,p.vz] for p in sim.particles], dtype=np.float64)
    return xn, vn


def _ias15_worker(conn, x0, v0, m_arr, dt):
    try:
        xn, vn = ias15_one_step(x0, v0, m_arr, dt)
        conn.send(("ok", xn, vn, ""))
    except BaseException as e:
        try: conn.send(("err", None, None, f"{type(e).__name__}: {e}"))
        except BaseException: pass
    finally:
        try: conn.close()
        except BaseException: pass


def ias15_isolated(x0, v0, m_arr, dt, timeout=IAS15_TIMEOUT_SEC):
    ctx = mp.get_context("spawn")
    pc, cc = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_ias15_worker, args=(cc, x0, v0, m_arr, float(dt)))
    p.start(); cc.close(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(5.0); pc.close()
        raise TimeoutError(f"IAS15 timed out after {timeout}s")
    if p.exitcode != 0:
        pc.close()
        raise RuntimeError(f"IAS15 exited {p.exitcode}")
    if not pc.poll():
        pc.close()
        raise RuntimeError("IAS15 no result")
    status, xn, vn, msg = pc.recv(); pc.close()
    if status != "ok": raise RuntimeError(msg)
    return xn, vn


# =============================================================================
# Leapfrog with fixed c (pair 0-1 only)
# =============================================================================
def acc_leapfrog(pos, m_arr, c_val=1.0, apply_pair=(0, 1)):
    """
    Three-body acceleration.
    apply_pair: pair that receives the c-scaled softened force.
    All other pairs use exact Newtonian.
    """
    acc  = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS
    pairs = [(0,1), (0,2), (1,2)]
    for (i, j) in pairs:
        rij = pos[j] - pos[i]
        r2  = float(np.dot(rij, rij))
        r   = float((r2 + 1e-30) ** 0.5)
        Gm  = G * float(m_arr[i]) * float(m_arr[j])
        if (i, j) == apply_pair and r < NN_THRESH:
            F = float(c_val) * Gm / ((r2 + eps2) ** 1.5 + 1e-30)
        else:
            F = Gm / (r2 * r + 1e-30)
        Fv = F * rij
        acc[i] +=  Fv / float(m_arr[i])
        acc[j] -=  Fv / float(m_arr[j])
    return acc


def leapfrog_step_c(x0, v0, m_arr, dt, c_val=1.0):
    a0 = acc_leapfrog(x0, m_arr, c_val)
    vh = v0 + 0.5 * dt * a0
    x1 = x0 + dt * vh
    a1 = acc_leapfrog(x1, m_arr, c_val)
    v1 = vh + 0.5 * dt * a1
    return x1, v1


# =============================================================================
# c_opt search
# =============================================================================
def find_c_opt(x0, v0, m_arr, dt, x_ref):
    x1, _ = leapfrog_step_c(x0, v0, m_arr, dt, 1.0)
    mse_1 = _mse(x1, x_ref)
    mse_g = []
    for c in C_GRID:
        xk, _ = leapfrog_step_c(x0, v0, m_arr, dt, float(c))
        mse_g.append(_mse(xk, x_ref))
    finite_idx = [i for i, v in enumerate(mse_g) if math.isfinite(v)]
    if not finite_idx:
        return np.nan, float("inf"), mse_1
    best_k = min(finite_idx, key=lambda i: mse_g[i])
    c_lo = float(C_GRID[max(0, best_k - 1)])
    c_hi = float(C_GRID[min(N_GRID - 1, best_k + 1)])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(30):
        if c_hi - c_lo < 1e-7: break
        c1 = c_hi - phi * (c_hi - c_lo)
        c2 = c_lo + phi * (c_hi - c_lo)
        m1 = _mse(leapfrog_step_c(x0, v0, m_arr, dt, c1)[0], x_ref)
        m2 = _mse(leapfrog_step_c(x0, v0, m_arr, dt, c2)[0], x_ref)
        if not math.isfinite(m1): m1 = float("inf")
        if not math.isfinite(m2): m2 = float("inf")
        if m1 < m2: c_hi = c2
        else:        c_lo = c1
    c_opt = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step_c(x0, v0, m_arr, dt, c_opt)
    return c_opt, _mse(x_opt, x_ref), mse_1


# =============================================================================
# Velocity features
# =============================================================================
def compute_pair_vfeatures(x_all, v_all, m_arr):
    r_vec   = x_all[1] - x_all[0]
    v_vec   = v_all[1] - v_all[0]
    r       = float(np.linalg.norm(r_vec))
    r_hat   = r_vec / (r + 1e-30)
    v_rad   = float(np.dot(v_vec, r_hat))
    v_tan   = float(np.linalg.norm(v_vec - v_rad * r_hat))
    v_scale = float(np.sqrt(G * (float(m_arr[0]) + float(m_arr[1])) / (r + 1e-30)))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30))


# =============================================================================
# No-NN SIMON leapfrog for trajectory generation
# Uses Zone 2 direct Newtonian + adaptive substeps (no Zone 3 NN correction).
# This gives physically realistic Zone 3 states without needing a pre-trained model.
# =============================================================================
def run_no_nn_leapfrog(x0, v0, m_arr, dt, T, max_substeps=16):
    """
    Run SIMON leapfrog without Zone 3 NN. Returns generator of
    (step_idx, t_cur, x, v) at every point where pair (0,1) is in Zone 3
    and NOT in Zone 2 (i.e., adapt_thresh <= r01 < nn_thresh).

    Zone 2 adaptive substepping is active for stability. Zone 3 uses c=1
    (pure softened force, same as no-NN baseline in evaluator).
    """
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    n_steps = int(math.ceil(T / dt))
    dt_f = float(dt)
    t_cur = 0.0
    states = []   # list of (step_idx, t_cur, x_snapshot, v_snapshot)

    def pair_r(xi):
        return float(np.linalg.norm(xi[1] - xi[0]))

    def acc_newtonian(pos):
        """Pure Newtonian for Zone 2 direct-force path."""
        acc = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(i+1, 3):
                rij = pos[j] - pos[i]
                r2  = float(np.dot(rij, rij))
                r   = float((r2 + 1e-30) ** 0.5)
                Gm  = G * float(m_arr[i]) * float(m_arr[j])
                F   = Gm / (r2 * r + 1e-30)
                Fv  = F * rij
                acc[i] +=  Fv / float(m_arr[i])
                acc[j] -=  Fv / float(m_arr[j])
        return acc

    def step_newtonian(xi, vi, dts):
        a0 = acc_newtonian(xi)
        vh = vi + 0.5 * dts * a0
        xn = xi + dts * vh
        a1 = acc_newtonian(xn)
        vn = vh + 0.5 * dts * a1
        return xn, vn

    def step_zone3_no_nn(xi, vi, dts):
        """Zone 3 step: c=1 (softened force, no NN). Same as evaluator no-NN path."""
        return leapfrog_step_c(xi, vi, m_arr, dts, c_val=1.0)

    for step_idx in range(n_steps):
        r01 = pair_r(x)

        if r01 < ADAPT_THRESH:
            # Zone 2: adaptive substepping with direct Newtonian
            n_sub = min(max_substeps, max(2, int(math.ceil(ADAPT_THRESH / r01))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v = step_newtonian(x, v, sub_dt)
        elif r01 < NN_THRESH:
            # Zone 3: record state BEFORE the step (this is the training input)
            # Only record pair (0,1) which is in Zone 3
            states.append((step_idx, t_cur, x.copy(), v.copy()))
            # Take Zone 3 step (c=1, no NN)
            x, v = step_zone3_no_nn(x, v, dt_f)
        else:
            # Zone 4: pure Newtonian
            x, v = step_newtonian(x, v, dt_f)

        t_cur += dt_f
        if t_cur >= T - 1e-12:
            break

    return states


# =============================================================================
# Main
# =============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Live rollout Zone 3 data harvester for Phase 3."
    )
    ap.add_argument("--ic", required=True, choices=["IC1", "IC3", "IC4"],
                    help="Which IC to run. IC1=default, IC3=tight pair, IC4=hierarchical.")
    ap.add_argument("--dt", type=float, required=True,
                    help="Leapfrog macro timestep. Valid: 0.02, 0.04, 0.05, 0.06, 0.08, 0.10")
    ap.add_argument("--T", type=float, default=50.0,
                    help="Simulation horizon in years. Default 50.")
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch ID for filename.")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir",
                    default="encounter_shards")
    ap.add_argument("--prefix", default="encounter_data_zone3_p3_live")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout", type=float, default=IAS15_TIMEOUT_SEC)
    ap.add_argument("--max_substeps", type=int, default=16)
    return ap


def main():
    args = build_arg_parser().parse_args()
    dt = float(args.dt)

    if round(dt, 4) not in VALID_DTS:
        raise ValueError(f"dt={dt} not in {sorted(VALID_DTS)}")

    os.makedirs(args.out_dir, exist_ok=True)
    token    = dt_token(dt)
    ic_tag   = args.ic.lower()
    out_file = os.path.join(args.out_dir,
                            f"{args.prefix}_{ic_tag}_dt{token}_batch{args.batch:03d}.npz")
    summary_file = out_file.replace(".npz", "_summary.txt")

    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_file}\nUse --overwrite.")

    x0, v0, m = get_ic(args.ic)
    T = float(args.T)

    print("=" * 72)
    print(f"LIVE ROLLOUT DATA HARVESTER -- Phase 3")
    print(f"  IC       : {args.ic}")
    print(f"  dt       : {dt:.6f} yr")
    print(f"  T        : {T:.1f} yr")
    print(f"  output   : {out_file}")
    print("=" * 72)

    t_run0 = time.perf_counter()
    print(f"  Running no-NN leapfrog trajectory for {T} yr...", flush=True)
    states = run_no_nn_leapfrog(x0, v0, m, dt, T, max_substeps=args.max_substeps)
    t_run = time.perf_counter() - t_run0
    print(f"  Trajectory done in {t_run:.1f}s. Harvested {len(states)} Zone 3 states.", flush=True)

    if not states:
        print("WARNING: No Zone 3 states found. Try a different IC or smaller dt.")
        return

    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "v_rad_norm", "v_tan_norm",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]}

    n_ok = 0
    n_skip = 0
    n_identity = 0
    c_buf, imp_buf, vr_buf, vt_buf = [], [], [], []

    t_copt0 = time.perf_counter()
    for k, (step_idx, t_yr, x_state, v_state) in enumerate(states):
        r01 = float(np.linalg.norm(x_state[1] - x_state[0]))

        # Double-check Zone 3 bounds (substepping might have shifted it)
        if not (Z3_R_MIN < r01 < Z3_R_MAX):
            n_skip += 1
            continue

        v_rad_norm, v_tan_norm = compute_pair_vfeatures(x_state, v_state, m)
        if not (math.isfinite(v_rad_norm) and math.isfinite(v_tan_norm)):
            n_skip += 1
            continue

        # Run IAS15 one step
        try:
            x_ref, _ = ias15_isolated(x_state, v_state, m, dt, timeout=float(args.timeout))
        except Exception as e:
            n_skip += 1
            continue

        if safe_max_norm(x_ref) > 50.0:
            n_skip += 1
            continue

        # Find c_opt
        try:
            c_opt, mse_opt, mse_1 = find_c_opt(x_state, v_state, m, dt, x_ref)
        except Exception:
            n_skip += 1
            continue

        if not (math.isfinite(c_opt) and math.isfinite(mse_opt) and math.isfinite(mse_1)):
            n_skip += 1
            continue
        if c_opt <= C_MIN + 0.05 or c_opt >= C_MAX - 0.05:
            n_skip += 1
            continue
        if mse_1 < 1e-20:
            n_skip += 1
            continue

        imp_raw = float((mse_1 - mse_opt) / mse_1)
        if imp_raw < 0.10:
            c_store = 1.0; log_c_store = 0.0; imp = 0.0; n_identity += 1
        else:
            c_store = c_opt; log_c_store = float(np.log(c_opt)); imp = imp_raw

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

        c_buf.append(c_store); imp_buf.append(imp)
        vr_buf.append(v_rad_norm); vt_buf.append(v_tan_norm)
        n_ok += 1

        if n_ok % 20 == 0 or n_ok == 1:
            el = time.perf_counter() - t_copt0
            print(f"  processed {n_ok}/{len(states)} states | "
                  f"c_med={np.median(c_buf):.4f} | "
                  f"vr_med={np.median(vr_buf):+.3f} | {el:.0f}s", flush=True)

    t_total = time.perf_counter() - t_run0

    if n_ok == 0:
        print("WARNING: 0 valid samples after c_opt processing.")
        return

    save = {k: np.array(v, dtype=np.float32) for k, v in arr.items()}
    np.savez_compressed(out_file, **save)
    size_kb = os.path.getsize(out_file) / 1024.0

    approach_n = sum(1 for v in vr_buf if v < 0)
    recede_n   = sum(1 for v in vr_buf if v >= 0)
    summary = (f"IC={args.ic} dt={dt:.6f}: "
               f"harvested={len(states)} zone3_states | "
               f"accepted={n_ok} | skipped={n_skip} | "
               f"c_med={np.median(c_buf):.5f} | "
               f"impr_med={np.median(imp_buf):.2%} | "
               f"identity={n_identity} | "
               f"approach={approach_n} recede={recede_n} | "
               f"total_time={t_total:.0f}s")

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("LIVE ROLLOUT ZONE 3 DATASET SUMMARY -- PHASE 3\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"IC      : {args.ic}\n")
        fh.write(f"dt      : {dt:.6f}\n")
        fh.write(f"T       : {T:.1f} yr\n")
        fh.write(f"batch   : {args.batch}\n")
        fh.write(f"output  : {out_file}\n")
        fh.write(f"size_kb : {size_kb:.1f}\n\n")
        fh.write(summary + "\n")

    print("\n" + summary)
    print(f"Saved {out_file} ({n_ok} samples, {size_kb:.1f} KB)")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
