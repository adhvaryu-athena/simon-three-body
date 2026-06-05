"""
generate_encounter_surrogate_data_v1.py

Encounter-level surrogate data generator for SIMON (prototype, dt=0.08).

Purpose
-------
Generate full-state encounter-window training data for a future neural
encounter surrogate.  This is NOT the old scalar-c correction dataset.

For each accepted Zone-3 encounter state:
  1. Build a full 3-body state at Zone-3 start.
  2. Run IAS15 over a fixed window (default 0.5 yr) as the reference.
  3. Run revised no-Zone-3-NN SIMON/leapfrog over the same exact endpoint.
  4. Save the residual target:
         residual = IAS15_final_state - noNN_final_state

The dataset is designed for a residual encounter-exit model:
    input  = full encounter-start state
    target = residual correction at the encounter-window exit

Why this file is separate
-------------------------
The old 7100-row file (encounter_data_zone3_v3_augmented.npz) contains
pair-level scalar-c features, not full 18D/21D state-transition samples.
Here we use that file only as an empirical proposal distribution for r,
log masses, v_rad_norm, and v_tan_norm.

Recommended smoke test:
    python -B generate_encounter_surrogate_data_v1.py --n 10 --batch 1 --dt 0.08 --mode mixed --window-years 0.5

Recommended first real shards:
    python -B generate_encounter_surrogate_data_v1.py --n 50 --batch 2 --dt 0.08 --mode mixed --window-years 0.5

Notes
-----
- Default active pair is bodies 1-2 to mirror your real rollout audits.
- Body 0 is the third body in the prototype.
- The output contains raw states and rich diagnostics; a later trainer can
  choose the most appropriate feature representation.
"""

import os
import sys
import time
import math
import argparse
import faulthandler
from typing import Dict, Tuple, Optional

import numpy as np
import rebound

faulthandler.enable(file=sys.stderr)

# =============================================================================
# Constants matched to revised SIMON methodology
# =============================================================================
G = 1.0
EPS = 3e-4
R_SOFT_MIN = 5e-4
ADAPT_THRESH = 0.05
NN_THRESH = 500.0 * EPS  # 0.15 AU
Z3_R_MIN = ADAPT_THRESH + 0.002  # 0.052 AU
Z3_R_MAX = NN_THRESH - 0.002     # 0.148 AU
ZONE1_R_GATE = float(np.sqrt(R_SOFT_MIN**2 - EPS**2))  # ~4e-4 AU
MAX_SUBSTEPS = 16
SEED = 42

PAIR_CHOICES = {
    "0-1": (0, 1, 2),
    "0-2": (0, 2, 1),
    "1-2": (1, 2, 0),
}

# =============================================================================
# Utility helpers
# =============================================================================
def dt_token(dt: float) -> str:
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def safe_norm(x) -> float:
    try:
        v = float(np.linalg.norm(x))
        return v if math.isfinite(v) else float("inf")
    except Exception:
        return float("inf")


def all_finite(*arrays) -> bool:
    for a in arrays:
        if not np.all(np.isfinite(a)):
            return False
    return True


def center_to_com(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x_com = np.sum(m[:, None] * x, axis=0) / M
    v_com = np.sum(m[:, None] * v, axis=0) / M
    return x - x_com, v - v_com


def pair_distances(x: np.ndarray) -> np.ndarray:
    return np.array([
        safe_norm(x[1] - x[0]),
        safe_norm(x[2] - x[0]),
        safe_norm(x[2] - x[1]),
    ], dtype=np.float64)


def min_pair_distance(x: np.ndarray) -> float:
    return float(np.min(pair_distances(x)))


def max_radius(x: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(x, axis=1)))


def compute_energy_newtonian(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> float:
    """Physical Newtonian mechanical energy."""
    KE = 0.5 * float(np.sum(m * np.sum(v * v, axis=1)))
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = safe_norm(x[i] - x[j])
            PE -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(KE + PE)


def rel_energy_drift(E0: float, E1: float) -> float:
    denom = max(abs(float(E0)), 1e-12)
    return float(abs(float(E1) - float(E0)) / denom)


def angular_momentum_z(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> float:
    Lz = 0.0
    for i in range(3):
        Lz += float(m[i]) * float(x[i, 0] * v[i, 1] - x[i, 1] * v[i, 0])
    return float(Lz)


def pair_velocity_features(x: np.ndarray, v: np.ndarray, m: np.ndarray, i: int, j: int) -> Tuple[float, float, float]:
    r_vec = x[j] - x[i]
    v_vec = v[j] - v[i]
    r = safe_norm(r_vec)
    r_hat = r_vec / (r + 1e-30)
    v_rad = float(np.dot(v_vec, r_hat))
    v_tan_vec = v_vec - v_rad * r_hat
    v_tan = safe_norm(v_tan_vec)
    v_scale = float(np.sqrt(G * (float(m[i]) + float(m[j])) / (r + 1e-30)))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r


def residual_rms(dx: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))


# =============================================================================
# Empirical proposal distribution from old scalar-c dataset
# =============================================================================
def load_scalar_distribution(path: str, dt: float) -> Optional[Dict[str, np.ndarray]]:
    if not path or not os.path.exists(path):
        print(f"[sampler] scalar-data file not found; using fallback random sampler: {path}")
        return None

    data = np.load(path)
    required = ["r_AU", "log_mi", "log_mj", "log_dt", "v_rad_norm", "v_tan_norm"]
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f"[sampler] scalar-data file missing {missing}; using fallback random sampler")
        return None

    log_dt = data["log_dt"].astype(np.float64)
    dt_vals = np.exp(log_dt)
    mask = np.isclose(dt_vals, float(dt), rtol=0.0, atol=5e-5)
    if int(np.sum(mask)) < 50:
        print(f"[sampler] too few rows for dt={dt}; using all scalar rows as fallback distribution")
        mask = np.ones_like(dt_vals, dtype=bool)

    out = {k: data[k][mask].astype(np.float64) for k in required if k != "log_dt"}
    out["row_indices"] = np.nonzero(mask)[0].astype(np.int64)
    print(f"[sampler] loaded empirical scalar proposal from {path}: rows={len(out['r_AU'])} for dt≈{dt}")
    return out


def choose_empirical_row(rng: np.random.RandomState, dist: Optional[Dict[str, np.ndarray]], mode: str) -> Tuple[float, float, float, float, float, int]:
    """
    Return r, log_mi, log_mj, v_rad_norm, v_tan_norm, source_row_idx.
    Uses current scalar dataset as proposal distribution when available.
    """
    if dist is None:
        # Fallback: similar to old generator ranges.
        r = float(rng.uniform(Z3_R_MIN, Z3_R_MAX))
        log_mi = float(rng.uniform(np.log(0.001), np.log(2.0)))
        log_mj = float(rng.uniform(np.log(0.001), np.log(2.0)))
        if mode == "strong_approach":
            vr = float(rng.uniform(-1.2, -0.6))
            vt = float(rng.uniform(0.4, 0.9))
        elif mode == "recede":
            vr = float(rng.uniform(0.2, 1.2))
            vt = float(rng.uniform(0.3, 1.2))
        else:
            vr = float(rng.uniform(-1.2, 1.2))
            vt = float(rng.uniform(0.2, 1.4))
        return r, log_mi, log_mj, vr, vt, -1

    vr_all = dist["v_rad_norm"]
    vt_all = dist["v_tan_norm"]
    r_all = dist["r_AU"]
    row_idx_all = dist["row_indices"]

    if mode == "strong_approach":
        #mask = (vr_all < -0.6) & (vt_all >= 0.35) & (vt_all <= 1.0) & (r_all >= 0.075) & (r_all <= 0.125)
        mask = (vr_all < -0.6) & (r_all >= 0.052) & (r_all <= 0.148)
    elif mode == "recede":
        mask = (vr_all > 0.2)
    elif mode == "weak_side":
        mask = (vr_all >= -0.6) & (vr_all <= 0.2)
    else:
        mask = np.ones_like(vr_all, dtype=bool)

    idxs = np.nonzero(mask)[0]
    if len(idxs) < 10:
        idxs = np.arange(len(r_all))
    k = int(rng.choice(idxs))
    return (
        float(dist["r_AU"][k]),
        float(dist["log_mi"][k]),
        float(dist["log_mj"][k]),
        float(dist["v_rad_norm"][k]),
        float(dist["v_tan_norm"][k]),
        int(row_idx_all[k]),
    )


def mode_for_attempt(rng: np.random.RandomState, mode: str) -> str:
    if mode != "mixed":
        return mode
    # Balanced prototype: ensure approach, side/weak, and recede are all represented.
    u = float(rng.rand())
    if u < 0.40:
        return "strong_approach"
    if u < 0.70:
        return "weak_side"
    return "recede"


# =============================================================================
# Full-state construction
# =============================================================================
def construct_state(
    rng: np.random.RandomState,
    dist: Optional[Dict[str, np.ndarray]],
    dt: float,
    mode: str,
    active_pair: str,
    third_r_min: float,
    third_r_max: float,
) -> Optional[Dict[str, object]]:
    i, j, k_third = PAIR_CHOICES[active_pair]
    submode = mode_for_attempt(rng, mode)
    r_pair, log_mi, log_mj, vr_norm, vt_norm, source_row_idx = choose_empirical_row(rng, dist, submode)

    # For the prototype, sample the third-body mass from the same empirical mass pool
    # or fallback range. This avoids inventing a separate mass distribution.
    if dist is not None:
        kk = int(rng.randint(0, len(dist["r_AU"])))
        log_mk = float(rng.choice([dist["log_mi"][kk], dist["log_mj"][kk]]))
    else:
        log_mk = float(rng.uniform(np.log(0.001), np.log(2.0)))

    m = np.zeros(3, dtype=np.float64)
    m[i] = float(np.exp(log_mi))
    m[j] = float(np.exp(log_mj))
    m[k_third] = float(np.exp(log_mk))

    # Random 2D orientation for the active pair.
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    # Tangential direction sign is random because old vt is a magnitude.
    t_sign = -1.0 if rng.rand() < 0.5 else 1.0
    t_hat = t_sign * np.array([-r_hat[1], r_hat[0], 0.0], dtype=np.float64)

    v_scale = float(np.sqrt(G * (m[i] + m[j]) / (r_pair + 1e-30)))
    v_rel = v_scale * (vr_norm * r_hat + vt_norm * t_hat)
    x_rel = r_pair * r_hat

    x = np.zeros((3, 3), dtype=np.float64)
    v = np.zeros((3, 3), dtype=np.float64)

    # Pair COM initially at origin.
    mi, mj = float(m[i]), float(m[j])
    Mpair = mi + mj
    x[i] = -(mj / Mpair) * x_rel
    x[j] = +(mi / Mpair) * x_rel
    v[i] = -(mj / Mpair) * v_rel
    v[j] = +(mi / Mpair) * v_rel

    # Third body relative to pair COM, kept outside Zone 3 at start.
    r3 = float(rng.uniform(third_r_min, third_r_max))
    th3 = float(rng.uniform(0.0, 2.0 * np.pi))
    r3_hat = np.array([np.cos(th3), np.sin(th3), 0.0], dtype=np.float64)
    t3_hat = np.array([-np.sin(th3), np.cos(th3), 0.0], dtype=np.float64)
    x[k_third] = r3 * r3_hat

    # Bound-ish third-body velocity around the pair COM.
    # Small radial noise prevents all samples from being perfectly circular.
    v_circ3 = float(np.sqrt(G * Mpair / (r3 + 1e-30)))
    f_t = float(rng.uniform(0.45, 1.05))
    f_r = float(rng.uniform(-0.15, 0.15))
    v[k_third] = v_circ3 * (f_r * r3_hat + f_t * t3_hat)

    x, v = center_to_com(x, v, m)

    vr_check, vt_check, r_check = pair_velocity_features(x, v, m, i, j)
    if not (Z3_R_MIN < r_check < Z3_R_MAX):
        return None
    # Keep non-active pairs safely away from Zone 3 for the first prototype.
    dists = pair_distances(x)
    pair_index_map = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    active_idx = pair_index_map[tuple(sorted((i, j)))]
    for pidx, d in enumerate(dists):
        if pidx != active_idx and d < NN_THRESH:
            return None

    E0 = compute_energy_newtonian(x, v, m)
    if not math.isfinite(E0) or E0 >= 0.0:
        return None

    Lz0 = angular_momentum_z(x, v, m)
    category_id = {"strong_approach": 0, "weak_side": 1, "recede": 2, "broad": 3}.get(submode, 4)
    return dict(
        x0=x, v0=v, m=m, active_i=i, active_j=j, third_k=k_third,
        r_pair=float(r_check), v_rad_norm=float(vr_check), v_tan_norm=float(vt_check),
        E0=float(E0), Lz0=float(Lz0), source_row_idx=int(source_row_idx),
        category_id=int(category_id), submode=submode,
    )


# =============================================================================
# IAS15 and revised no-Zone-3-NN baseline window solvers
# =============================================================================
def simulate_ias15_window(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, window_years: float, n_samples: int) -> Dict[str, object]:
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    sim.exit_min_distance = float(EPS)
    for q in range(3):
        sim.add(m=float(m[q]),
                x=float(x0[q, 0]), y=float(x0[q, 1]), z=float(x0[q, 2]),
                vx=float(v0[q, 0]), vy=float(v0[q, 1]), vz=float(v0[q, 2]))
    sim.move_to_com()
    times = np.linspace(0.0, float(window_years), int(n_samples))
    pos = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel = np.zeros((len(times), 3, 3), dtype=np.float64)
    min_r = float("inf")
    max_rad = 0.0
    for idx, t in enumerate(times):
        sim.integrate(float(t))
        for q, p in enumerate(sim.particles):
            pos[idx, q] = [p.x, p.y, p.z]
            vel[idx, q] = [p.vx, p.vy, p.vz]
        min_r = min(min_r, min_pair_distance(pos[idx]))
        max_rad = max(max_rad, max_radius(pos[idx]))
    xf = pos[-1].copy()
    vf = vel[-1].copy()
    if not all_finite(pos, vel):
        raise ValueError("IAS15 returned non-finite state")
    return dict(times=times, pos=pos, vel=vel, xf=xf, vf=vf, min_r=float(min_r), max_radius=float(max_rad))


def revised_no_nn_acc(pos: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, float, int, int, int, int]:
    """
    Revised no-Zone-3-NN acceleration:
      Zone 1/2 r < 0.05: direct Newtonian
      Zone 3 0.05 <= r < 0.15: softened force with c=1
      Zone 4 r >= 0.15: direct Newtonian
    Returns acceleration and zone counts for the 3 pair evaluations.
    """
    acc = np.zeros((3, 3), dtype=np.float64)
    min_r = float("inf")
    z1 = z2 = z3 = z4 = 0
    for a in range(3):
        for b in range(a + 1, 3):
            rij = pos[b] - pos[a]
            r2 = float(np.dot(rij, rij))
            r = float(np.sqrt(r2 + 1e-30))
            min_r = min(min_r, r)
            Gmimj = G * float(m[a]) * float(m[b])
            if r < ZONE1_R_GATE:
                scalar = Gmimj / (r2 * r + 1e-30)
                z1 += 1
            elif r < ADAPT_THRESH:
                scalar = Gmimj / (r2 * r + 1e-30)
                z2 += 1
            elif r < NN_THRESH:
                scalar = Gmimj / ((r2 + EPS * EPS) ** 1.5 + 1e-30)
                z3 += 1
            else:
                scalar = Gmimj / (r2 * r + 1e-30)
                z4 += 1
            F = scalar * rij
            acc[a] += F / float(m[a])
            acc[b] -= F / float(m[b])
    return acc, min_r, z1, z2, z3, z4


def simulate_nonn_window(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, dt: float, window_years: float) -> Dict[str, object]:
    """Velocity-Verlet over an exact window endpoint with final short step."""
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    t = 0.0
    steps = 0
    total_substeps = 0
    zone_counts = dict(zone1=0, zone2=0, zone3=0, zone4=0)
    min_r_seen = min_pair_distance(x)
    max_rad_seen = max_radius(x)

    a, rmin, z1, z2, z3, z4 = revised_no_nn_acc(x, m)
    for key, val in zip(["zone1", "zone2", "zone3", "zone4"], [z1, z2, z3, z4]):
        zone_counts[key] += int(val)

    while t < float(window_years) - 1e-14:
        step_dt = min(float(dt), float(window_years) - t)
        # Adaptive substeps only if current min distance is Zone 2.
        r_now = min_pair_distance(x)
        if r_now < ADAPT_THRESH:
            n_sub = min(MAX_SUBSTEPS, max(2, int(math.ceil(ADAPT_THRESH / max(r_now, 1e-30)))))
        else:
            n_sub = 1
        sub_dt = step_dt / n_sub
        for _ in range(n_sub):
            vh = v + 0.5 * sub_dt * a
            x = x + sub_dt * vh
            a, rmin, z1, z2, z3, z4 = revised_no_nn_acc(x, m)
            v = vh + 0.5 * sub_dt * a
            min_r_seen = min(min_r_seen, rmin)
            max_rad_seen = max(max_rad_seen, max_radius(x))
            for key, val in zip(["zone1", "zone2", "zone3", "zone4"], [z1, z2, z3, z4]):
                zone_counts[key] += int(val)
            total_substeps += 1
        t += step_dt
        steps += 1
        if not all_finite(x, v):
            raise ValueError("noNN produced non-finite state")
    return dict(xf=x, vf=v, min_r=float(min_r_seen), max_radius=float(max_rad_seen),
                steps=int(steps), substeps=int(total_substeps), **zone_counts)


# =============================================================================
# Acceptance and saving
# =============================================================================
def make_feature_vectors(sample: Dict[str, object], dt: float, window_years: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two useful feature vectors:
      X_raw21: [log m0..2, x0 flat 9, v0 flat 9]
      X_rel18: canonical relative features for later model prototyping
    """
    x = sample["x0"]
    v = sample["v0"]
    m = sample["m"]
    i = int(sample["active_i"]); j = int(sample["active_j"]); k = int(sample["third_k"])
    mi, mj = float(m[i]), float(m[j])
    pair_com_x = (mi * x[i] + mj * x[j]) / (mi + mj)
    pair_com_v = (mi * v[i] + mj * v[j]) / (mi + mj)
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r3 = x[k] - pair_com_x
    v3 = v[k] - pair_com_v
    # 3 log masses + 3 rij + 3 vij + 3 r3 + 3 v3 + 3 scalars = 18
    X_rel18 = np.concatenate([
        np.log(m + 1e-30),
        rij, vij, r3, v3,
        np.array([math.log(float(dt) + 1e-30), float(sample["v_rad_norm"]), float(sample["v_tan_norm"])], dtype=np.float64),
    ]).astype(np.float64)
    X_raw21 = np.concatenate([np.log(m + 1e-30), x.reshape(-1), v.reshape(-1)]).astype(np.float64)
    return X_raw21, X_rel18


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate encounter-level surrogate full-state residual data.")
    ap.add_argument("--dt", type=float, default=0.08, help="Macro timestep for noNN baseline; prototype uses 0.08.")
    ap.add_argument("--window-years", "--window_years", dest="window_years", type=float, default=0.5,
                    help="Fixed encounter window length in years.")
    ap.add_argument("--window-samples", "--window_samples", dest="window_samples", type=int, default=21,
                    help="IAS15 samples inside the window for min-distance/ejection checks.")
    ap.add_argument("--n", type=int, default=50, help="Accepted samples to generate.")
    ap.add_argument("--batch", type=int, default=1, help="Batch id for seed and filename.")
    ap.add_argument("--mode", choices=["broad", "strong_approach", "weak_side", "recede", "mixed"], default="mixed",
                    help="Sampling mode. mixed = 40%% strong approach, 30%% weak/side, 30%% recede.")
    ap.add_argument("--active-pair", "--active_pair", dest="active_pair", choices=sorted(PAIR_CHOICES), default="1-2",
                    help="Active Zone-3 pair. Default 1-2 to mirror real rollout audits.")
    ap.add_argument("--scalar-data", "--scalar_data", dest="scalar_data", default="encounter_data_zone3_v3_augmented.npz",
                    help="Old scalar-c dataset used only as empirical proposal distribution.")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="encounter_surrogate_shards",
                    help="Output shard folder.")
    ap.add_argument("--prefix", default="encounter_surrogate_v1",
                    help="Output filename prefix.")
    ap.add_argument("--seed", type=int, default=SEED, help="Base seed.")
    ap.add_argument("--max-tries", "--max_tries", dest="max_tries", type=int, default=0,
                    help="Maximum attempts. 0 means n*1000.")
    ap.add_argument("--third-r-min", "--third_r_min", dest="third_r_min", type=float, default=2.5)
    ap.add_argument("--third-r-max", "--third_r_max", dest="third_r_max", type=float, default=8.0)
    ap.add_argument("--min-ias15-r", "--min_ias15_r", dest="min_ias15_r", type=float, default=ADAPT_THRESH,
                    help="Reject if IAS15 min pair distance in window falls below this. Default keeps prototype Zone-3-only.")
    ap.add_argument("--max-ias15-energy-drift", "--max_ias15_energy_drift", dest="max_ias15_energy_drift", type=float, default=1e-8)
    ap.add_argument("--max-nonn-energy-drift", "--max_nonn_energy_drift", dest="max_nonn_energy_drift", type=float, default=5e-2)
    ap.add_argument("--ejection-au", "--ejection_au", dest="ejection_au", type=float, default=50.0)
    ap.add_argument("--max-residual-rms", "--max_residual_rms", dest="max_residual_rms", type=float, default=5.0,
                    help="Reject extreme residual position RMS outliers.")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    token = dt_token(args.dt)
    wtok = dt_token(args.window_years)
    mode_suffix = args.mode
    pair_suffix = args.active_pair.replace("-", "")
    out_file = os.path.join(args.out_dir, f"{args.prefix}_{mode_suffix}_pair{pair_suffix}_dt{token}_w{wtok}_batch{int(args.batch):03d}.npz")
    summary_file = out_file.replace(".npz", "_summary.txt")
    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_file}\nUse --overwrite to replace it.")

    seed = int(args.seed) + int(args.batch) * 100000 + int(round(float(args.dt) * 1_000_000)) + 91_000_000
    rng = np.random.RandomState(seed)
    dist = load_scalar_distribution(args.scalar_data, args.dt)
    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else max(1000 * int(args.n), 1000)

    fields = {
        # Raw state and target arrays.
        "m": [], "x0": [], "v0": [],
        "x_ias_final": [], "v_ias_final": [],
        "x_nonn_final": [], "v_nonn_final": [],
        "residual_x": [], "residual_v": [],
        "X_raw21": [], "X_rel18": [],
        # Metadata and diagnostics.
        "dt": [], "window_years": [], "active_i": [], "active_j": [], "third_k": [],
        "r_pair": [], "r_soft_pair": [], "v_rad_norm": [], "v_tan_norm": [],
        "source_row_idx": [], "category_id": [],
        "E0": [], "E_ias_final": [], "E_nonn_final": [],
        "relE_ias": [], "relE_nonn": [], "Lz0": [],
        "min_r_ias": [], "min_r_nonn": [], "max_radius_ias": [], "max_radius_nonn": [],
        "nonn_steps": [], "nonn_substeps": [],
        "nonn_zone1": [], "nonn_zone2": [], "nonn_zone3": [], "nonn_zone4": [],
        "pos_residual_rms": [], "vel_residual_rms": [], "nonn_pos_error_rms": [],
    }
    rejections = dict(
        ic=0, ias15=0, ias15_eject=0, ias15_min_r=0, ias15_energy=0,
        nonn=0, nonn_eject=0, nonn_energy=0, residual_outlier=0, nonfinite=0,
    )

    print("=" * 88)
    print("ENCOUNTER-LEVEL SURROGATE DATA GENERATOR v1")
    print(f"  dt/window       : dt={args.dt:.6f} yr, window={args.window_years:.6f} yr")
    print(f"  target n        : {args.n}")
    print(f"  batch/seed      : {args.batch} / {seed}")
    print(f"  mode            : {args.mode}")
    print(f"  active_pair     : {args.active_pair}")
    print(f"  scalar proposal : {args.scalar_data}")
    print(f"  Zone-3 start    : ({Z3_R_MIN:.3f}, {Z3_R_MAX:.3f}) AU")
    print(f"  min IAS15 r     : {args.min_ias15_r:.6f} AU")
    print(f"  energy filters  : IAS15<{args.max_ias15_energy_drift:.1e}, noNN<{args.max_nonn_energy_drift:.1e}")
    print(f"  output          : {out_file}")
    print("  target          : residual = IAS15_final - revised_noNN_final")
    print("=" * 88)

    t_start = time.perf_counter()
    n_ok = 0
    n_try = 0
    while n_ok < int(args.n) and n_try < max_tries:
        n_try += 1
        sample = construct_state(rng, dist, float(args.dt), args.mode, args.active_pair, float(args.third_r_min), float(args.third_r_max))
        if sample is None:
            rejections["ic"] += 1
            continue
        x0 = sample["x0"]; v0 = sample["v0"]; m = sample["m"]
        E0 = float(sample["E0"])

        try:
            ref = simulate_ias15_window(x0, v0, m, float(args.window_years), int(args.window_samples))
        except BaseException:
            rejections["ias15"] += 1
            continue

        if ref["max_radius"] > float(args.ejection_au):
            rejections["ias15_eject"] += 1
            continue
        if ref["min_r"] < float(args.min_ias15_r):
            rejections["ias15_min_r"] += 1
            continue
        E_ias = compute_energy_newtonian(ref["xf"], ref["vf"], m)
        relE_ias = rel_energy_drift(E0, E_ias)
        if not math.isfinite(relE_ias) or relE_ias > float(args.max_ias15_energy_drift):
            rejections["ias15_energy"] += 1
            continue

        try:
            base = simulate_nonn_window(x0, v0, m, float(args.dt), float(args.window_years))
        except BaseException:
            rejections["nonn"] += 1
            continue

        if base["max_radius"] > float(args.ejection_au):
            rejections["nonn_eject"] += 1
            continue
        E_nonn = compute_energy_newtonian(base["xf"], base["vf"], m)
        relE_nonn = rel_energy_drift(E0, E_nonn)
        if not math.isfinite(relE_nonn) or relE_nonn > float(args.max_nonn_energy_drift):
            rejections["nonn_energy"] += 1
            continue

        rx = ref["xf"] - base["xf"]
        rv = ref["vf"] - base["vf"]
        pos_rms = residual_rms(rx)
        vel_rms = residual_rms(rv)
        if not all_finite(rx, rv) or not math.isfinite(pos_rms) or not math.isfinite(vel_rms):
            rejections["nonfinite"] += 1
            continue
        if pos_rms > float(args.max_residual_rms):
            rejections["residual_outlier"] += 1
            continue

        X_raw21, X_rel18 = make_feature_vectors(sample, float(args.dt), float(args.window_years))
        if not all_finite(X_raw21, X_rel18):
            rejections["nonfinite"] += 1
            continue

        # Store accepted sample.
        fields["m"].append(m.astype(np.float64))
        fields["x0"].append(x0.astype(np.float64))
        fields["v0"].append(v0.astype(np.float64))
        fields["x_ias_final"].append(ref["xf"].astype(np.float64))
        fields["v_ias_final"].append(ref["vf"].astype(np.float64))
        fields["x_nonn_final"].append(base["xf"].astype(np.float64))
        fields["v_nonn_final"].append(base["vf"].astype(np.float64))
        fields["residual_x"].append(rx.astype(np.float64))
        fields["residual_v"].append(rv.astype(np.float64))
        fields["X_raw21"].append(X_raw21.astype(np.float64))
        fields["X_rel18"].append(X_rel18.astype(np.float64))
        for kfield, value in [
            ("dt", args.dt), ("window_years", args.window_years),
            ("active_i", sample["active_i"]), ("active_j", sample["active_j"]), ("third_k", sample["third_k"]),
            ("r_pair", sample["r_pair"]), ("r_soft_pair", math.sqrt(float(sample["r_pair"])**2 + EPS**2)),
            ("v_rad_norm", sample["v_rad_norm"]), ("v_tan_norm", sample["v_tan_norm"]),
            ("source_row_idx", sample["source_row_idx"]), ("category_id", sample["category_id"]),
            ("E0", E0), ("E_ias_final", E_ias), ("E_nonn_final", E_nonn),
            ("relE_ias", relE_ias), ("relE_nonn", relE_nonn), ("Lz0", sample["Lz0"]),
            ("min_r_ias", ref["min_r"]), ("min_r_nonn", base["min_r"]),
            ("max_radius_ias", ref["max_radius"]), ("max_radius_nonn", base["max_radius"]),
            ("nonn_steps", base["steps"]), ("nonn_substeps", base["substeps"]),
            ("nonn_zone1", base["zone1"]), ("nonn_zone2", base["zone2"]),
            ("nonn_zone3", base["zone3"]), ("nonn_zone4", base["zone4"]),
            ("pos_residual_rms", pos_rms), ("vel_residual_rms", vel_rms),
            ("nonn_pos_error_rms", residual_rms(base["xf"] - ref["xf"])),
        ]:
            fields[kfield].append(value)

        n_ok += 1
        if n_ok == 1 or n_ok % max(1, int(args.n) // 5) == 0:
            elapsed = time.perf_counter() - t_start
            vr_med = float(np.median(fields["v_rad_norm"]))
            vt_med = float(np.median(fields["v_tan_norm"]))
            pr_med = float(np.median(fields["pos_residual_rms"]))
            e_med = float(np.median(fields["relE_nonn"]))
            print(f"  accepted {n_ok:5d}/{int(args.n)} | tried={n_try:6d} | "
                  f"vr_med={vr_med:+.3f} vt_med={vt_med:.3f} | "
                  f"pos_resid_med={pr_med:.3e} noNN_E_med={e_med:.2e} | {elapsed:.0f}s")

    if n_ok == 0:
        raise RuntimeError(f"No samples accepted after {n_try} tries. Rejections: {rejections}")

    # Convert fields to arrays with compact dtypes where safe.
    out = {}
    float64_fields = {"m", "x0", "v0", "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final", "residual_x", "residual_v", "X_raw21", "X_rel18"}
    int_fields = {"active_i", "active_j", "third_k", "source_row_idx", "category_id", "nonn_steps", "nonn_substeps", "nonn_zone1", "nonn_zone2", "nonn_zone3", "nonn_zone4"}
    for k, vals in fields.items():
        arr = np.asarray(vals)
        if k in int_fields:
            out[k] = arr.astype(np.int32)
        elif k in float64_fields:
            out[k] = arr.astype(np.float64)
        else:
            out[k] = arr.astype(np.float32)

    np.savez_compressed(out_file, **out)
    elapsed = time.perf_counter() - t_start

    # Summary text.
    vr = out["v_rad_norm"].astype(np.float64)
    vt = out["v_tan_norm"].astype(np.float64)
    cat = out["category_id"].astype(np.int32)
    summary_lines = []
    def add(s=""):
        summary_lines.append(str(s))
        print(s)

    add("=" * 88)
    add("ENCOUNTER SURROGATE SHARD SUMMARY")
    add(f"file: {out_file}")
    add(f"n={n_ok} tried={n_try} pass_rate={n_ok/max(n_try,1):.1%} elapsed={elapsed:.1f}s")
    add(f"dt={args.dt:.6f} window={args.window_years:.6f} active_pair={args.active_pair} mode={args.mode}")
    add(f"r_pair: min={out['r_pair'].min():.6f} med={np.median(out['r_pair']):.6f} max={out['r_pair'].max():.6f}")
    add(f"v_rad_norm: min={vr.min():+.3f} med={np.median(vr):+.3f} max={vr.max():+.3f}")
    add(f"v_tan_norm: min={vt.min():.3f} med={np.median(vt):.3f} max={vt.max():.3f}")
    add(f"approach frac={np.mean(vr < 0):.1%}; strong approach vr<-0.6={np.sum(vr < -0.6)}; recede vr>0.2={np.sum(vr > 0.2)}")
    add(f"category counts: strong={int(np.sum(cat==0))}, weak_side={int(np.sum(cat==1))}, recede={int(np.sum(cat==2))}, broad={int(np.sum(cat==3))}")
    add(f"min_r_ias: min={out['min_r_ias'].min():.6f} med={np.median(out['min_r_ias']):.6f} max={out['min_r_ias'].max():.6f}")
    add(f"min_r_nonn: min={out['min_r_nonn'].min():.6f} med={np.median(out['min_r_nonn']):.6f} max={out['min_r_nonn'].max():.6f}")
    add(f"relE_ias: max={out['relE_ias'].max():.3e} med={np.median(out['relE_ias']):.3e}")
    add(f"relE_nonn: max={out['relE_nonn'].max():.3e} med={np.median(out['relE_nonn']):.3e}")
    add(f"pos_residual_rms: min={out['pos_residual_rms'].min():.3e} med={np.median(out['pos_residual_rms']):.3e} max={out['pos_residual_rms'].max():.3e}")
    add(f"vel_residual_rms: min={out['vel_residual_rms'].min():.3e} med={np.median(out['vel_residual_rms']):.3e} max={out['vel_residual_rms'].max():.3e}")
    add(f"noNN zone counts total: z1={int(np.sum(out['nonn_zone1']))}, z2={int(np.sum(out['nonn_zone2']))}, z3={int(np.sum(out['nonn_zone3']))}, z4={int(np.sum(out['nonn_zone4']))}")
    add(f"rejections: {rejections}")
    add("=" * 88)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved {out_file} ({n_ok} samples, {os.path.getsize(out_file)/1024:.1f} KB)")
    print(f"Saved {summary_file}")


if __name__ == "__main__":
    main()
