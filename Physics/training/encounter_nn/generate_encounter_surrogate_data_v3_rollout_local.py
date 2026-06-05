"""
generate_encounter_surrogate_data_v3_rollout_local.py

Rollout-local encounter-window data generator for the IC1 dt=0.08 event.

Why this v3 generator exists
----------------------------
The synthetic event-targeted generator matched the failed rollout event in simple
pair features (r_pair, v_rad_norm, v_tan_norm, min_r), but the single-event
replay diagnostics showed that both MLP and kNN residual methods still failed on
that exact rollout event.  The nearest-neighbor diagnosis indicated that the
real event was not close to the training set in the full X_rel18 state space.

This generator therefore samples directly from the revised noNN IC1 rollout
trajectory around the real encounter time, then adds small local perturbations
around those actual full states.  The goal is to create training windows that
are close to the real rollout in full three-body geometry, not merely in scalar
pair-level features.

For each accepted window:
  1. Take an anchor state from the revised noNN IC1 rollout near t ~ 82.56 yr.
  2. Add small COM-projected perturbations to the full 3-body position/velocity.
  3. Require active pair 1-2 to be in Zone 3 and approaching.
  4. Run IAS15 over a fixed 0.5 yr window.
  5. Run revised noNN over the same exact endpoint.
  6. Save residual = IAS15_exit - noNN_exit in the same format as the v1/v2
     encounter-surrogate shards, so existing merge/inspect/train scripts work.

Typical smoke test:
  python -B generate_encounter_surrogate_data_v3_rollout_local.py ^
      --n 30 --batch 201 --dt 0.08 --window-years 0.5 ^
      --t-lo 82.20 --t-hi 82.80 --pos-sigma 0.0005 --vel-sigma 0.002 ^
      --prefix encounter_surrogate_v3_rollout_local

Then merge with a pattern including v1/v2/v3 shards, inspect, and retrain.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import faulthandler
from typing import Dict, Tuple, List

import numpy as np

# Reuse the already-tested rollout/noNN/IAS15 implementation from the replay diagnostic.
from diagnose_single_event_replay_v1 import (
    G, EPS, R_SOFT_MIN, ADAPT_THRESH, NN_THRESH, ZONE1_R_GATE, MAX_SUBSTEPS,
    get_ic, integrate_nonn, simulate_ias15_at_times, compute_pair_velocity_features,
    min_pair_distance, max_radius, total_energy, rel_energy, all_finite,
    make_X_rel18,
)

faulthandler.enable(file=sys.stderr)

ACTIVE_I = 1
ACTIVE_J = 2
THIRD_K = 0
Z3_R_MIN = ADAPT_THRESH + 0.002
Z3_R_MAX = NN_THRESH - 0.002
SEED = 42


def dt_token(x: float) -> str:
    return f"{float(x):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def state_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def residual_rms(dx: np.ndarray) -> float:
    return state_rms(dx, np.zeros_like(dx))


def center_to_com(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x_com = np.sum(m[:, None] * x, axis=0) / M
    v_com = np.sum(m[:, None] * v, axis=0) / M
    return (x - x_com).astype(np.float64), (v - v_com).astype(np.float64)


def build_X_raw21(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([np.log(m + 1e-30), x.reshape(-1), v.reshape(-1)]).astype(np.float64)


def category_from_vr(vr: float) -> int:
    if vr < -0.6:
        return 0  # strong_approach
    if vr <= 0.2:
        return 1  # weak_side
    if vr > 0.2:
        return 2  # recede
    return 3


def soft_r(r: float) -> float:
    return float(math.sqrt(float(r) * float(r) + EPS * EPS))


def make_anchor_pool(dt: float, t_lo: float, t_hi: float, anchor_spacing: float, include_event_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run revised noNN once and return anchor states sampled around the event."""
    x0, v0, m = get_ic("IC1")
    if t_hi <= t_lo:
        raise ValueError("t_hi must be greater than t_lo")
    times = list(np.arange(float(t_lo), float(t_hi) + 0.5 * float(anchor_spacing), float(anchor_spacing)))
    if float(include_event_time) >= float(t_lo) - 1e-12 and float(include_event_time) <= float(t_hi) + 1e-12:
        times.append(float(include_event_time))
    times = np.array(sorted(set([round(float(t), 10) for t in times])), dtype=np.float64)
    # integrate_nonn samples at absolute times from t=0.
    pos, vel, stats = integrate_nonn(x0, v0, m, dt, float(t_hi), times)
    return times, pos, vel, m


def perturb_state(rng: np.random.RandomState,
                  x_anchor: np.ndarray,
                  v_anchor: np.ndarray,
                  m: np.ndarray,
                  pos_sigma: float,
                  vel_sigma: float,
                  radial_sigma: float,
                  tangential_sigma: float,
                  exact_prob: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Local perturbation around an actual noNN rollout state.

    The perturbation is deliberately small and COM-projected.  It combines:
      - full-state Gaussian perturbation, to create local full-state coverage;
      - optional pair radial/tangential perturbation, to diversify close approach
        depth without losing the actual third-body geometry.
    """
    x = x_anchor.astype(np.float64, copy=True)
    v = v_anchor.astype(np.float64, copy=True)

    if rng.rand() < float(exact_prob):
        return *center_to_com(x, v, m), {"pos_pert_norm": 0.0, "vel_pert_norm": 0.0, "radial_pert": 0.0, "tangential_pert": 0.0}

    dx = rng.normal(0.0, float(pos_sigma), size=x.shape)
    dv = rng.normal(0.0, float(vel_sigma), size=v.shape)
    # keep z exactly zero for the first prototype, matching existing planar IC1/data.
    dx[:, 2] = 0.0
    dv[:, 2] = 0.0
    x = x + dx
    v = v + dv

    # Pair-specific small perturbations in the active pair relative coordinates.
    rij = x[ACTIVE_J] - x[ACTIVE_I]
    r = float(np.linalg.norm(rij))
    if r > 1e-12:
        rhat = rij / r
        that = np.array([-rhat[1], rhat[0], 0.0], dtype=np.float64)
        d_rad = float(rng.normal(0.0, float(radial_sigma)))
        d_tan = float(rng.normal(0.0, float(tangential_sigma)))
        # Apply equal-and-opposite relative perturbations while approximately preserving pair COM.
        x[ACTIVE_I] -= 0.5 * d_rad * rhat
        x[ACTIVE_J] += 0.5 * d_rad * rhat
        v[ACTIVE_I] -= 0.5 * d_tan * that
        v[ACTIVE_J] += 0.5 * d_tan * that
    else:
        d_rad = 0.0
        d_tan = 0.0

    x, v = center_to_com(x, v, m)
    return x, v, {
        "pos_pert_norm": float(np.sqrt(np.mean(np.sum(dx * dx, axis=1)))),
        "vel_pert_norm": float(np.sqrt(np.mean(np.sum(dv * dv, axis=1)))),
        "radial_pert": float(d_rad),
        "tangential_pert": float(d_tan),
    }


def add_sample(fields: Dict[str, List], key_values: Dict[str, object]) -> None:
    for k in fields:
        fields[k].append(key_values[k])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate rollout-local encounter surrogate data around the real IC1 dt=0.08 event.")
    ap.add_argument("--n", type=int, default=50, help="Number of accepted samples to generate.")
    ap.add_argument("--batch", type=int, default=201, help="Batch id for seed and filename.")
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--window-years", "--window_years", dest="window_years", type=float, default=0.5)
    ap.add_argument("--window-samples", "--window_samples", dest="window_samples", type=int, default=41)
    ap.add_argument("--t-lo", "--t_lo", dest="t_lo", type=float, default=82.20)
    ap.add_argument("--t-hi", "--t_hi", dest="t_hi", type=float, default=82.90)
    ap.add_argument("--event-time", "--event_time", dest="event_time", type=float, default=82.56)
    ap.add_argument("--anchor-spacing", "--anchor_spacing", dest="anchor_spacing", type=float, default=0.02)
    ap.add_argument("--time-sigma", "--time_sigma", dest="time_sigma", type=float, default=0.16,
                    help="Used only for biased anchor selection around event_time.")
    ap.add_argument("--pos-sigma", "--pos_sigma", dest="pos_sigma", type=float, default=5e-4)
    ap.add_argument("--vel-sigma", "--vel_sigma", dest="vel_sigma", type=float, default=2e-3)
    ap.add_argument("--radial-sigma", "--radial_sigma", dest="radial_sigma", type=float, default=5e-4)
    ap.add_argument("--tangential-sigma", "--tangential_sigma", dest="tangential_sigma", type=float, default=2e-3)
    ap.add_argument("--exact-prob", "--exact_prob", dest="exact_prob", type=float, default=0.05,
                    help="Probability of using the anchor state with no perturbation.")
    ap.add_argument("--vr-min", "--vr_min", dest="vr_min", type=float, default=-1.45)
    ap.add_argument("--vr-max", "--vr_max", dest="vr_max", type=float, default=-0.40)
    ap.add_argument("--vt-min", "--vt_min", dest="vt_min", type=float, default=0.50)
    ap.add_argument("--vt-max", "--vt_max", dest="vt_max", type=float, default=1.25)
    ap.add_argument("--min-ias15-r", "--min_ias15_r", dest="min_ias15_r", type=float, default=ADAPT_THRESH)
    ap.add_argument("--target-min-r-lo", "--target_min_r_lo", dest="target_min_r_lo", type=float, default=0.052)
    ap.add_argument("--target-min-r-hi", "--target_min_r_hi", dest="target_min_r_hi", type=float, default=0.085)
    ap.add_argument("--max-ias15-energy-drift", "--max_ias15_energy_drift", dest="max_ias15_energy_drift", type=float, default=1e-8)
    ap.add_argument("--max-nonn-energy-drift", "--max_nonn_energy_drift", dest="max_nonn_energy_drift", type=float, default=5e-2)
    ap.add_argument("--ejection-au", "--ejection_au", dest="ejection_au", type=float, default=50.0)
    ap.add_argument("--max-residual-rms", "--max_residual_rms", dest="max_residual_rms", type=float, default=1.0)
    ap.add_argument("--max-tries", "--max_tries", dest="max_tries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="encounter_surrogate_shards")
    ap.add_argument("--prefix", default="encounter_surrogate_v3_rollout_local")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    token = dt_token(args.dt)
    wtok = dt_token(args.window_years)
    out_file = os.path.join(args.out_dir, f"{args.prefix}_pair12_dt{token}_w{wtok}_batch{int(args.batch):03d}.npz")
    summary_file = out_file.replace(".npz", "_summary.txt")
    if os.path.exists(out_file) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_file}\nUse --overwrite to replace it.")

    seed = int(args.seed) + int(args.batch) * 100000 + int(round(float(args.dt) * 1_000_000)) + 93_000_000
    rng = np.random.RandomState(seed)
    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else max(1500 * int(args.n), 3000)

    print("=" * 96)
    print("ROLLOUT-LOCAL ENCOUNTER SURROGATE DATA GENERATOR v3")
    print(f"  target n        : {args.n}")
    print(f"  batch/seed      : {args.batch} / {seed}")
    print(f"  dt/window       : {args.dt:.6f} yr / {args.window_years:.6f} yr")
    print(f"  rollout window  : t in [{args.t_lo:.4f}, {args.t_hi:.4f}] yr, event_time={args.event_time:.4f}")
    print(f"  anchor spacing  : {args.anchor_spacing:.4f} yr")
    print(f"  perturb sigmas  : pos={args.pos_sigma:g}, vel={args.vel_sigma:g}, radial={args.radial_sigma:g}, tangential={args.tangential_sigma:g}")
    print(f"  required vr/vt  : vr in [{args.vr_min:.2f}, {args.vr_max:.2f}], vt in [{args.vt_min:.2f}, {args.vt_max:.2f}]")
    print(f"  target min_r IAS: [{args.target_min_r_lo:.4f}, {args.target_min_r_hi:.4f}] AU")
    print(f"  output          : {out_file}")
    print("=" * 96)

    print("[anchor] running revised noNN once to build local anchor pool...")
    anchor_times, anchor_pos, anchor_vel, m = make_anchor_pool(args.dt, args.t_lo, args.t_hi, args.anchor_spacing, args.event_time)
    anchor_info = []
    for idx, t in enumerate(anchor_times):
        x = anchor_pos[idx]; v = anchor_vel[idx]
        vr, vt, r = compute_pair_velocity_features(x, v, m, ACTIVE_I, ACTIVE_J)
        if Z3_R_MIN <= r <= Z3_R_MAX and args.vr_min <= vr <= args.vr_max and args.vt_min <= vt <= args.vt_max:
            anchor_info.append((idx, float(t), float(r), float(vr), float(vt)))
    if not anchor_info:
        raise RuntimeError("No usable anchors found. Broaden --t-lo/--t-hi or vr/vt filters.")
    anchor_info_arr = np.array(anchor_info, dtype=np.float64)
    print(f"[anchor] total anchors={len(anchor_times)}, usable={len(anchor_info)}")
    print(f"[anchor] usable t range={anchor_info_arr[:,1].min():.4f}..{anchor_info_arr[:,1].max():.4f}, "
          f"r med={np.median(anchor_info_arr[:,2]):.6f}, vr med={np.median(anchor_info_arr[:,3]):+.4f}, vt med={np.median(anchor_info_arr[:,4]):.4f}")

    fields = {
        "m": [], "x0": [], "v0": [],
        "x_ias_final": [], "v_ias_final": [],
        "x_nonn_final": [], "v_nonn_final": [],
        "residual_x": [], "residual_v": [],
        "X_raw21": [], "X_rel18": [],
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

    # Extra diagnostics are written to a sidecar CSV-style txt, not to the merge-required npz fields.
    accepted_diag = []
    rejections = dict(
        anchor_filter=0, perturb_bad=0, z3=0, vr_vt=0, ias15=0, ias15_eject=0,
        ias15_min_r=0, target_min_r_low=0, target_min_r_high=0, ias15_energy=0,
        nonn=0, nonn_eject=0, nonn_energy=0, residual_outlier=0, nonfinite=0,
    )

    # Prefer anchors near event_time but keep some broader local coverage.
    usable_indices = np.array([int(row[0]) for row in anchor_info], dtype=np.int64)
    usable_times = np.array([float(row[1]) for row in anchor_info], dtype=np.float64)
    time_weights = np.exp(-0.5 * ((usable_times - float(args.event_time)) / max(float(args.time_sigma), 1e-6)) ** 2)
    time_weights = time_weights / np.sum(time_weights)

    t_start = time.perf_counter()
    n_ok = 0
    n_try = 0
    while n_ok < int(args.n) and n_try < max_tries:
        n_try += 1
        if rng.rand() < 0.75:
            local_pos = int(rng.choice(np.arange(len(usable_indices)), p=time_weights))
        else:
            local_pos = int(rng.randint(0, len(usable_indices)))
        aidx = int(usable_indices[local_pos])
        anchor_t = float(anchor_times[aidx])
        x_anchor = anchor_pos[aidx]
        v_anchor = anchor_vel[aidx]

        x0, v0, pmeta = perturb_state(
            rng, x_anchor, v_anchor, m,
            pos_sigma=float(args.pos_sigma),
            vel_sigma=float(args.vel_sigma),
            radial_sigma=float(args.radial_sigma),
            tangential_sigma=float(args.tangential_sigma),
            exact_prob=float(args.exact_prob),
        )
        if not all_finite(x0, v0):
            rejections["perturb_bad"] += 1
            continue

        X_rel18, vr, vt, r_pair = make_X_rel18(x0, v0, m, args.dt)
        if not (Z3_R_MIN <= r_pair <= Z3_R_MAX):
            rejections["z3"] += 1
            continue
        if not (float(args.vr_min) <= vr <= float(args.vr_max) and float(args.vt_min) <= vt <= float(args.vt_max)):
            rejections["vr_vt"] += 1
            continue

        E0 = total_energy(x0, v0, m, softened_pe=False)
        Lz0 = float(np.sum(m * (x0[:, 0] * v0[:, 1] - x0[:, 1] * v0[:, 0])))

        try:
            sample_times = np.linspace(0.0, float(args.window_years), int(args.window_samples))
            pos_ias, vel_ias, _ = simulate_ias15_at_times(x0, v0, m, sample_times)
        except BaseException:
            rejections["ias15"] += 1
            continue
        x_ias_f = pos_ias[-1].copy(); v_ias_f = vel_ias[-1].copy()
        min_r_ias = float(np.min([min_pair_distance(px) for px in pos_ias]))
        max_r_ias = float(np.max([max_radius(px) for px in pos_ias]))
        if max_r_ias > float(args.ejection_au):
            rejections["ias15_eject"] += 1
            continue
        if min_r_ias < float(args.min_ias15_r):
            rejections["ias15_min_r"] += 1
            continue
        if min_r_ias < float(args.target_min_r_lo):
            rejections["target_min_r_low"] += 1
            continue
        if min_r_ias > float(args.target_min_r_hi):
            rejections["target_min_r_high"] += 1
            continue
        E_ias = total_energy(x_ias_f, v_ias_f, m, softened_pe=False)
        relE_ias = rel_energy(E_ias, E0)
        if (not math.isfinite(relE_ias)) or relE_ias > float(args.max_ias15_energy_drift):
            rejections["ias15_energy"] += 1
            continue

        try:
            pos_no, vel_no, stats_no = integrate_nonn(x0, v0, m, args.dt, args.window_years, np.array([0.0, float(args.window_years)], dtype=np.float64))
        except BaseException:
            rejections["nonn"] += 1
            continue
        x_no_f = pos_no[-1].copy(); v_no_f = vel_no[-1].copy()
        max_r_no = max(max_radius(pos_no[0]), max_radius(x_no_f))
        if max_r_no > float(args.ejection_au):
            rejections["nonn_eject"] += 1
            continue
        E_no = total_energy(x_no_f, v_no_f, m, softened_pe=False)
        relE_no = rel_energy(E_no, E0)
        if (not math.isfinite(relE_no)) or relE_no > float(args.max_nonn_energy_drift):
            rejections["nonn_energy"] += 1
            continue

        rx = x_ias_f - x_no_f
        rv = v_ias_f - v_no_f
        pos_resid = residual_rms(rx)
        vel_resid = residual_rms(rv)
        if not all_finite(rx, rv, X_rel18):
            rejections["nonfinite"] += 1
            continue
        if pos_resid > float(args.max_residual_rms):
            rejections["residual_outlier"] += 1
            continue

        X_raw21 = build_X_raw21(x0, v0, m)
        sample_values = {
            "m": m.astype(np.float64),
            "x0": x0.astype(np.float64),
            "v0": v0.astype(np.float64),
            "x_ias_final": x_ias_f.astype(np.float64),
            "v_ias_final": v_ias_f.astype(np.float64),
            "x_nonn_final": x_no_f.astype(np.float64),
            "v_nonn_final": v_no_f.astype(np.float64),
            "residual_x": rx.astype(np.float64),
            "residual_v": rv.astype(np.float64),
            "X_raw21": X_raw21.astype(np.float64),
            "X_rel18": X_rel18.astype(np.float64),
            "dt": float(args.dt),
            "window_years": float(args.window_years),
            "active_i": ACTIVE_I,
            "active_j": ACTIVE_J,
            "third_k": THIRD_K,
            "r_pair": float(r_pair),
            "r_soft_pair": soft_r(r_pair),
            "v_rad_norm": float(vr),
            "v_tan_norm": float(vt),
            "source_row_idx": int(-500000 - n_ok),
            "category_id": int(category_from_vr(vr)),
            "E0": float(E0),
            "E_ias_final": float(E_ias),
            "E_nonn_final": float(E_no),
            "relE_ias": float(relE_ias),
            "relE_nonn": float(relE_no),
            "Lz0": float(Lz0),
            "min_r_ias": float(min_r_ias),
            "min_r_nonn": float(stats_no["min_r"]),
            "max_radius_ias": float(max_r_ias),
            "max_radius_nonn": float(max_r_no),
            "nonn_steps": int(stats_no["steps"]),
            "nonn_substeps": int(stats_no["substeps"]),
            "nonn_zone1": int(stats_no["zone1"]),
            "nonn_zone2": int(stats_no["zone2"]),
            "nonn_zone3": int(stats_no["zone3"]),
            "nonn_zone4": int(stats_no["zone4"]),
            "pos_residual_rms": float(pos_resid),
            "vel_residual_rms": float(vel_resid),
            "nonn_pos_error_rms": float(state_rms(x_no_f, x_ias_f)),
        }
        add_sample(fields, sample_values)
        accepted_diag.append({
            "anchor_time": anchor_t,
            "r_pair": float(r_pair),
            "v_rad_norm": float(vr),
            "v_tan_norm": float(vt),
            "min_r_ias": float(min_r_ias),
            "min_r_nonn": float(stats_no["min_r"]),
            "pos_residual_rms": float(pos_resid),
            "vel_residual_rms": float(vel_resid),
            **pmeta,
        })
        n_ok += 1
        if n_ok == 1 or n_ok % max(1, int(args.n) // 5) == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  accepted {n_ok:5d}/{int(args.n)} | tried={n_try:6d} | "
                  f"t_med={np.median([d['anchor_time'] for d in accepted_diag]):.3f} "
                  f"r_med={np.median(fields['r_pair']):.6f} vr_med={np.median(fields['v_rad_norm']):+.3f} "
                  f"minr_med={np.median(fields['min_r_ias']):.6f} posR_med={np.median(fields['pos_residual_rms']):.3e} | {elapsed:.1f}s")

    if n_ok == 0:
        raise RuntimeError(f"No accepted samples after {n_try} tries. Rejections: {rejections}")

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

    # Sidecar CSV for local-source diagnostics.
    diag_file = out_file.replace(".npz", "_local_diag.csv")
    with open(diag_file, "w", encoding="utf-8") as f:
        keys = list(accepted_diag[0].keys()) if accepted_diag else []
        f.write(",".join(keys) + "\n")
        for row in accepted_diag:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    vr_arr = out["v_rad_norm"].astype(np.float64)
    vt_arr = out["v_tan_norm"].astype(np.float64)
    cat = out["category_id"].astype(np.int32)
    anchor_times_ok = np.array([d["anchor_time"] for d in accepted_diag], dtype=np.float64)

    lines = []
    def add(s: str = ""):
        lines.append(str(s))
        print(s)

    add("=" * 96)
    add("ROLLOUT-LOCAL ENCOUNTER SURROGATE SHARD SUMMARY")
    add(f"file: {out_file}")
    add(f"n={n_ok} tried={n_try} pass_rate={n_ok/max(n_try,1):.1%} elapsed={elapsed:.1f}s")
    add(f"dt={args.dt:.6f} window={args.window_years:.6f} active_pair=1-2 source=IC1 revised noNN rollout")
    add(f"anchor_time: min={anchor_times_ok.min():.6f} med={np.median(anchor_times_ok):.6f} max={anchor_times_ok.max():.6f}")
    add(f"r_pair: min={out['r_pair'].min():.6f} med={np.median(out['r_pair']):.6f} max={out['r_pair'].max():.6f}")
    add(f"v_rad_norm: min={vr_arr.min():+.3f} med={np.median(vr_arr):+.3f} max={vr_arr.max():+.3f}")
    add(f"v_tan_norm: min={vt_arr.min():.3f} med={np.median(vt_arr):.3f} max={vt_arr.max():.3f}")
    add(f"category counts: strong={int(np.sum(cat==0))}, weak_side={int(np.sum(cat==1))}, recede={int(np.sum(cat==2))}, broad={int(np.sum(cat==3))}")
    add(f"min_r_ias: min={out['min_r_ias'].min():.6f} med={np.median(out['min_r_ias']):.6f} max={out['min_r_ias'].max():.6f}")
    add(f"min_r_nonn: min={out['min_r_nonn'].min():.6f} med={np.median(out['min_r_nonn']):.6f} max={out['min_r_nonn'].max():.6f}")
    add(f"relE_ias: max={out['relE_ias'].max():.3e} med={np.median(out['relE_ias']):.3e}")
    add(f"relE_nonn: max={out['relE_nonn'].max():.3e} med={np.median(out['relE_nonn']):.3e}")
    add(f"pos_residual_rms: min={out['pos_residual_rms'].min():.3e} med={np.median(out['pos_residual_rms']):.3e} max={out['pos_residual_rms'].max():.3e}")
    add(f"vel_residual_rms: min={out['vel_residual_rms'].min():.3e} med={np.median(out['vel_residual_rms']):.3e} max={out['vel_residual_rms'].max():.3e}")
    add(f"noNN zone counts total: z1={int(np.sum(out['nonn_zone1']))}, z2={int(np.sum(out['nonn_zone2']))}, z3={int(np.sum(out['nonn_zone3']))}, z4={int(np.sum(out['nonn_zone4']))}")
    add(f"rejections: {rejections}")
    add(f"sidecar diagnostics: {diag_file}")
    add("=" * 96)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out_file} ({n_ok} samples, {os.path.getsize(out_file)/1024:.1f} KB)")
    print(f"Saved {summary_file}")
    print(f"Saved {diag_file}")


if __name__ == "__main__":
    main()
