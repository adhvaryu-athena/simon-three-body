"""
generate_live_rollout_data.py  --  Zone 3 live-rollout state harvester, Phase 3

BUG FIX v2:
  v1 only checked pair (0,1) for Zone 3. For IC1, pair (0,1) starts at r~0.99 AU
  in a nearly circular orbit and may never reach Zone 3 in 50 yr. The evaluator
  checks ALL three pairs and uses r_min to determine the zone. This version
  correctly checks all pairs and records the state for whichever pair enters Zone 3.

Purpose
-------
Run actual SIMON leapfrog trajectories on IC1, IC3, IC4.
For every step where any pair is in Zone 3 (0.05 <= r < 0.15 AU), record the
exact (x, v, m) state and the pair indices. Then run find_c_opt for that pair
to get the trajectory-optimal c. Save as training shards.

Usage:
    python -B generate_live_rollout_data.py --ic IC1 --dt 0.04 --T 100 --batch 1
    python -B generate_live_rollout_data.py --ic IC3 --dt 0.08 --T 100 --batch 1
    python -B generate_live_rollout_data.py --ic IC4 --dt 0.04 --T 100 --batch 1

Recommended: use --T 100 (not 50) to ensure enough Zone 3 encounters.
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

# All three pairs in a 3-body system
ALL_PAIRS = [(0, 1), (0, 2), (1, 2)]

N_GRID = 60
C_GRID = np.linspace(C_MIN, C_MAX, N_GRID)
IAS15_TIMEOUT_SEC = 60.0


# =============================================================================
# Initial conditions
# =============================================================================
def get_ic(name: str):
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


def pair_separation(x, pi, pj):
    return float(np.linalg.norm(x[pj] - x[pi]))


def r_min_all_pairs(x):
    """Minimum separation across all three pairs. Matches evaluator zone logic."""
    return min(pair_separation(x, pi, pj) for (pi, pj) in ALL_PAIRS)


def zone3_pairs(x):
    """Return list of (pi,pj) pairs currently in Zone 3 (ADAPT_THRESH <= r < NN_THRESH)."""
    result = []
    for (pi, pj) in ALL_PAIRS:
        r = pair_separation(x, pi, pj)
        if ADAPT_THRESH <= r < NN_THRESH:
            result.append((pi, pj))
    return result


# =============================================================================
# IAS15 subprocess
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
# Force computation and leapfrog -- generalised to any pair (pi, pj)
# =============================================================================
def acc_for_pair(pos, m_arr, c_val, pi, pj):
    """
    Three-body acceleration.
    Pair (pi, pj): softened force * c_val when r < NN_THRESH.
    All other pairs: exact Newtonian.
    """
    acc  = np.zeros((3, 3), dtype=np.float64)
    eps2 = EPS * EPS
    for (i, j) in ALL_PAIRS:
        rij = pos[j] - pos[i]
        r2  = float(np.dot(rij, rij))
        r   = float((r2 + 1e-30) ** 0.5)
        Gm  = G * float(m_arr[i]) * float(m_arr[j])
        if (i, j) == (pi, pj) and r < NN_THRESH:
            F = float(c_val) * Gm / ((r2 + eps2) ** 1.5 + 1e-30)
        else:
            F = Gm / (r2 * r + 1e-30)
        Fv = F * rij
        acc[i] +=  Fv / float(m_arr[i])
        acc[j] -=  Fv / float(m_arr[j])
    return acc


def acc_newtonian_all(pos, m_arr):
    """Pure Newtonian for all pairs. Used during Zone 2 substepping."""
    acc = np.zeros((3, 3), dtype=np.float64)
    for (i, j) in ALL_PAIRS:
        rij = pos[j] - pos[i]
        r2  = float(np.dot(rij, rij))
        r   = float((r2 + 1e-30) ** 0.5)
        Gm  = G * float(m_arr[i]) * float(m_arr[j])
        F   = Gm / (r2 * r + 1e-30)
        Fv  = F * rij
        acc[i] +=  Fv / float(m_arr[i])
        acc[j] -=  Fv / float(m_arr[j])
    return acc


def leapfrog_step_pair(x0, v0, m_arr, dt, c_val, pi, pj):
    """One leapfrog step with softened c correction on pair (pi, pj)."""
    a0 = acc_for_pair(x0, m_arr, c_val, pi, pj)
    vh = v0 + 0.5 * float(dt) * a0
    x1 = x0 + float(dt) * vh
    a1 = acc_for_pair(x1, m_arr, c_val, pi, pj)
    v1 = vh + 0.5 * float(dt) * a1
    return x1, v1


def leapfrog_step_newtonian(x0, v0, m_arr, dt):
    """One leapfrog step with pure Newtonian for all pairs."""
    a0 = acc_newtonian_all(x0, m_arr)
    vh = v0 + 0.5 * float(dt) * a0
    x1 = x0 + float(dt) * vh
    a1 = acc_newtonian_all(x1, m_arr)
    v1 = vh + 0.5 * float(dt) * a1
    return x1, v1


# =============================================================================
# c_opt search -- generalised to pair (pi, pj)
# =============================================================================
def find_c_opt_pair(x0, v0, m_arr, dt, x_ref, pi, pj):
    """
    Grid + golden-section search for c minimising MSE(x_leapfrog(c), x_ref)
    where c is applied to pair (pi, pj).
    """
    x1, _ = leapfrog_step_pair(x0, v0, m_arr, dt, 1.0, pi, pj)
    mse_1 = _mse(x1, x_ref)
    mse_g = []
    for c in C_GRID:
        xk, _ = leapfrog_step_pair(x0, v0, m_arr, dt, float(c), pi, pj)
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
        m1 = _mse(leapfrog_step_pair(x0, v0, m_arr, dt, c1, pi, pj)[0], x_ref)
        m2 = _mse(leapfrog_step_pair(x0, v0, m_arr, dt, c2, pi, pj)[0], x_ref)
        if not math.isfinite(m1): m1 = float("inf")
        if not math.isfinite(m2): m2 = float("inf")
        if m1 < m2: c_hi = c2
        else:        c_lo = c1
    c_opt = 0.5 * (c_lo + c_hi)
    x_opt, _ = leapfrog_step_pair(x0, v0, m_arr, dt, c_opt, pi, pj)
    return c_opt, _mse(x_opt, x_ref), mse_1


# =============================================================================
# Velocity features -- generalised to pair (pi, pj)
# =============================================================================
def compute_pair_vfeatures(x_all, v_all, m_arr, pi, pj):
    r_vec   = x_all[pj] - x_all[pi]
    v_vec   = v_all[pj] - v_all[pi]
    r       = float(np.linalg.norm(r_vec))
    r_hat   = r_vec / (r + 1e-30)
    v_rad   = float(np.dot(v_vec, r_hat))
    v_tan   = float(np.linalg.norm(v_vec - v_rad * r_hat))
    v_scale = float(np.sqrt(G * (float(m_arr[pi]) + float(m_arr[pj])) / (r + 1e-30)))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30))


# =============================================================================
# No-NN SIMON leapfrog for trajectory generation
# Checks ALL pairs to determine zone. Records (pi,pj) with each Zone 3 state.
# =============================================================================
def run_no_nn_leapfrog(x0, v0, m_arr, dt, T, max_substeps=16):
    """
    Run SIMON leapfrog (no Zone 3 NN) and collect Zone 3 states for ALL pairs.

    Zone logic (mirrors the evaluator exactly):
      - r_min across ALL pairs < ADAPT_THRESH: Zone 2 substepping (Newtonian)
      - ANY pair in [ADAPT_THRESH, NN_THRESH): Zone 3 -- record each such pair
      - All pairs >= NN_THRESH: Zone 4 (Newtonian)

    Returns list of (step_idx, t_cur, x_snapshot, v_snapshot, pi, pj).
    """
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    n_steps = int(math.ceil(T / dt))
    dt_f    = float(dt)
    t_cur   = 0.0
    states  = []

    for step_idx in range(n_steps):
        # Determine zone from minimum pair separation (matches evaluator)
        r_min = r_min_all_pairs(x)

        if r_min < ADAPT_THRESH:
            # Zone 2: adaptive substepping with pure Newtonian
            n_sub   = min(max_substeps, max(2, int(math.ceil(ADAPT_THRESH / r_min))))
            sub_dt  = dt_f / n_sub
            for _ in range(n_sub):
                x, v = leapfrog_step_newtonian(x, v, m_arr, sub_dt)

        else:
            # Check if any pair is in Zone 3
            z3 = zone3_pairs(x)
            if z3:
                # Record each Zone 3 pair before the step
                for (pi, pj) in z3:
                    r_pair = pair_separation(x, pi, pj)
                    if Z3_R_MIN < r_pair < Z3_R_MAX:
                        states.append((step_idx, t_cur, x.copy(), v.copy(), pi, pj))

                # Take one Zone 3 step (c=1, softened force on the closest pair)
                # Use the closest Zone 3 pair for the step force correction
                closest = min(z3, key=lambda ij: pair_separation(x, ij[0], ij[1]))
                x, v = leapfrog_step_pair(x, v, m_arr, dt_f, 1.0, closest[0], closest[1])
            else:
                # Zone 4: pure Newtonian
                x, v = leapfrog_step_newtonian(x, v, m_arr, dt_f)

        t_cur += dt_f
        if t_cur >= T - 1e-12:
            break

    return states


# =============================================================================
# Main
# =============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Live rollout Zone 3 data harvester for Phase 3 (v2 fix)."
    )
    ap.add_argument("--ic", required=True, choices=["IC1", "IC3", "IC4"])
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=100.0,
                    help="Simulation horizon in years. Default 100 (was 50 in v1; "
                         "use 100 to ensure enough Zone 3 encounters).")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="encounter_shards")
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
    print(f"LIVE ROLLOUT DATA HARVESTER -- Phase 3 (v2: all-pair zone check)")
    print(f"  IC       : {args.ic}")
    print(f"  dt       : {dt:.6f} yr")
    print(f"  T        : {T:.1f} yr")
    print(f"  output   : {out_file}")
    print("=" * 72)

    # Print initial pair separations so user can see which pairs might enter Zone 3
    print("  Initial pair separations:")
    for (pi, pj) in ALL_PAIRS:
        r = pair_separation(x0, pi, pj)
        print(f"    pair ({pi},{pj}): r={r:.4f} AU  "
              f"m_i={m[pi]:.4f} m_j={m[pj]:.4f}")

    t_run0 = time.perf_counter()
    print(f"\n  Running no-NN leapfrog for {T} yr (checking all 3 pairs)...", flush=True)
    states = run_no_nn_leapfrog(x0, v0, m, dt, T, max_substeps=args.max_substeps)
    t_run  = time.perf_counter() - t_run0

    # Count per-pair
    pair_counts = {(0,1): 0, (0,2): 0, (1,2): 0}
    for s in states:
        pair_counts[(s[4], s[5])] += 1

    print(f"  Trajectory done in {t_run:.1f}s.")
    print(f"  Harvested {len(states)} Zone 3 states:")
    for (pi, pj), cnt in pair_counts.items():
        print(f"    pair ({pi},{pj}): {cnt} states")

    if not states:
        print("\nWARNING: No Zone 3 states found.")
        print("Try --T 200 or a different IC.")
        return

    arr = {k: [] for k in [
        "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
        "v_rad_norm", "v_tan_norm",
        "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
    ]}

    n_ok = 0; n_skip = 0; n_identity = 0
    c_buf, imp_buf, vr_buf, vt_buf = [], [], [], []

    t_copt0 = time.perf_counter()
    for k, (step_idx, t_yr, x_state, v_state, pi, pj) in enumerate(states):
        r_pair = pair_separation(x_state, pi, pj)

        if not (Z3_R_MIN < r_pair < Z3_R_MAX):
            n_skip += 1
            continue

        vr, vt = compute_pair_vfeatures(x_state, v_state, m, pi, pj)
        if not (math.isfinite(vr) and math.isfinite(vt)):
            n_skip += 1
            continue

        # IAS15 one step from this exact 3-body state
        try:
            x_ref, _ = ias15_isolated(x_state, v_state, m, dt, timeout=float(args.timeout))
        except Exception:
            n_skip += 1
            continue

        if safe_max_norm(x_ref) > 50.0:
            n_skip += 1
            continue

        # find c_opt for pair (pi, pj)
        try:
            c_opt, mse_opt, mse_1 = find_c_opt_pair(x_state, v_state, m, dt, x_ref, pi, pj)
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

        r_soft = float(np.sqrt(r_pair**2 + EPS**2))
        c_ana  = float((r_soft / r_pair) ** 3)

        # Use masses of the active pair (pi, pj)
        arr["r_AU"].append(r_pair)
        arr["r_soft"].append(r_soft)
        arr["log_mi"].append(float(np.log(m[pi] + 1e-30)))
        arr["log_mj"].append(float(np.log(m[pj] + 1e-30)))
        arr["log_dt"].append(float(np.log(dt)))
        arr["v_rad_norm"].append(vr)
        arr["v_tan_norm"].append(vt)
        arr["c_opt"].append(c_store)
        arr["log_c_opt"].append(log_c_store)
        arr["c_ana"].append(c_ana)
        arr["log_c_ana"].append(float(np.log(c_ana)))
        arr["improvement"].append(imp)

        c_buf.append(c_store); imp_buf.append(imp)
        vr_buf.append(vr); vt_buf.append(vt)
        n_ok += 1

        if n_ok % 20 == 0 or n_ok == 1:
            el = time.perf_counter() - t_copt0
            print(f"  processed {k+1}/{len(states)} states | "
                  f"accepted={n_ok} | c_med={np.median(c_buf):.4f} | "
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
               f"zone3_states_harvested={len(states)} | "
               f"accepted={n_ok} | skipped={n_skip} | "
               f"c_med={np.median(c_buf):.5f} | "
               f"impr_med={np.median(imp_buf):.2%} | "
               f"identity={n_identity} | "
               f"approach={approach_n} recede={recede_n} | "
               f"pairs: (0,1)={pair_counts[(0,1)]} (0,2)={pair_counts[(0,2)]} "
               f"(1,2)={pair_counts[(1,2)]} | "
               f"total_time={t_total:.0f}s")

    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write("LIVE ROLLOUT ZONE 3 DATASET SUMMARY -- PHASE 3 v2\n")
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


if __name__ == "__main__":
    mp.freeze_support()
    main()
