"""
diagnose_single_event_replay_knn_v1.py

Single-event replay diagnostic for the local kNN residual surrogate.

Purpose
-------
The neural encounter-residual models over-corrected the real IC1 rollout event
near t=82.56 yr.  The kNN diagnostic showed that local neighbor averaging can
be safe on held-out encounter windows.  This script tests the exact event:

  1. Reconstruct the revised noNN IC1 state at event_time.
  2. Run a local IAS15 window and revised noNN window from that exact state.
  3. Estimate an 18D residual using k nearest training windows.
  4. Apply alpha * residual only if v_rad_norm < gate_vr.
  5. Compare noNN-exit vs kNN-corrected exit against IAS15.
  6. Optionally continue both branches to T to check branch sensitivity.

Typical run from C:\\Aarush\\Physics\\training\\encounter_nn:

  python -B diagnose_single_event_replay_knn_v1.py ^
      --data encounter_surrogate_v2_event_enriched.npz ^
      --event-time 82.56 --dt 0.08 --window-years 0.5 --T 100 ^
      --feature-mode xrel18 --k 10 --alpha 0.75 --gate-vr -0.40 ^
      --out-dir single_event_replay_knn_v1_t82p56

Outputs
-------
  <out-dir>/single_event_replay_knn_summary.txt
  <out-dir>/single_event_replay_knn_metrics.csv
  <out-dir>/single_event_replay_knn_arrays.npz
  <out-dir>/single_event_replay_knn_neighbors.csv
  <out-dir>/local_window_rms_knn.png       (if matplotlib available)
  <out-dir>/post_window_rms_knn.png        (if matplotlib available)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

# Reuse the already-tested physics/replay code from the neural event diagnostic.
from diagnose_single_event_replay_v1 import (
    G, EPS,
    get_ic, advance_nonn_to_time, make_X_rel18,
    simulate_ias15_at_times, integrate_nonn,
    compute_pair_velocity_features, min_pair_distance,
    total_energy, rel_energy, state_rms, traj_rms,
    project_com_to_reference, max_radius, all_finite,
)

# Reuse the already-tested kNN dataset/features/split utilities.
from diagnose_knn_residual_surrogate_v1 import (
    load_dataset, make_stratified_splits, build_feature_space,
    knn_predict_residual, rms_pos_vel_from_resid,
)


def _safe_std(x: np.ndarray, axis=0) -> np.ndarray:
    s = np.nanstd(x, axis=axis).astype(np.float64)
    return np.where(s < 1e-10, 1.0, s)


def build_event_feature(mode: str,
                        x_evt: np.ndarray,
                        v_evt: np.ndarray,
                        m: np.ndarray,
                        dt: float,
                        r_pair: float,
                        vr: float,
                        vt: float,
                        min_r_nonn: float,
                        relE_nonn: float) -> np.ndarray:
    """Build one event feature row matching diagnose_knn_residual_surrogate_v1 modes."""
    xrel18, _, _, _ = make_X_rel18(x_evt, v_evt, m, dt)
    logm = np.log(np.clip(m.astype(np.float64), 1e-30, None))
    compact3 = np.array([r_pair, vr, vt], dtype=np.float64)
    compact5 = np.array([r_pair, vr, vt, min_r_nonn, relE_nonn], dtype=np.float64)
    compact8 = np.concatenate([compact5, logm]).astype(np.float64)

    if mode == "compact3":
        return compact3.reshape(1, -1)
    if mode == "compact5":
        return compact5.reshape(1, -1)
    if mode == "compact8":
        return compact8.reshape(1, -1)
    if mode == "xrel18":
        return xrel18.astype(np.float64).reshape(1, -1)
    if mode == "hybrid26":
        return np.concatenate([xrel18.astype(np.float64), compact8]).reshape(1, -1)
    raise ValueError(f"Unknown feature mode: {mode}")


def find_knn_for_event(data_path: str,
                       feature_mode: str,
                       k: int,
                       alpha: float,
                       seed: int,
                       x_evt: np.ndarray,
                       v_evt: np.ndarray,
                       m: np.ndarray,
                       dt: float,
                       r_pair: float,
                       vr: float,
                       vt: float,
                       min_r_nonn: float,
                       relE_nonn: float,
                       out_dir: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return alpha*kNN residual18 and diagnostic metadata for the event."""
    ds = load_dataset(data_path)
    raw = ds.raw
    Y = ds.Y.astype(np.float64)
    train_idx, val_idx, test_idx = make_stratified_splits(raw, seed=seed)

    X_all = build_feature_space(raw, feature_mode)
    X_train_raw = X_all[train_idx].astype(np.float64)
    Y_train = Y[train_idx].astype(np.float64)
    mean = X_train_raw.mean(axis=0)
    std = _safe_std(X_train_raw, axis=0)
    X_train = (X_train_raw - mean) / std

    X_evt_raw = build_event_feature(feature_mode, x_evt, v_evt, m, dt, r_pair, vr, vt, min_r_nonn, relE_nonn)
    if X_evt_raw.shape[1] != X_train_raw.shape[1]:
        raise ValueError(f"event feature width {X_evt_raw.shape[1]} != training feature width {X_train_raw.shape[1]}")
    X_evt = (X_evt_raw - mean) / std

    pred, mean_dist = knn_predict_residual(X_train, Y_train, X_evt, k=int(k), weight_power=2.0, batch=1)
    pred18_unscaled = pred.reshape(18)
    pred18 = float(alpha) * pred18_unscaled

    # Also identify and log the neighbor rows for inspection.
    diff = X_train - X_evt[0]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    kk = min(int(k), len(d))
    order_local = np.argsort(d)[:kk]
    neighbor_idx = train_idx[order_local]
    dist = d[order_local]
    w = 1.0 / np.power(dist + 1e-8, 2.0)
    w = w / np.sum(w)

    rx = np.asarray(raw["residual_x"], dtype=np.float64).reshape(len(Y), 9)
    rv = np.asarray(raw["residual_v"], dtype=np.float64).reshape(len(Y), 9)
    pos_rms, vel_rms = rms_pos_vel_from_resid(Y[neighbor_idx])

    neigh_csv = os.path.join(out_dir, "single_event_replay_knn_neighbors.csv")
    with open(neigh_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "dataset_index", "distance", "weight",
            "r_pair", "v_rad_norm", "v_tan_norm", "min_r_nonn", "relE_nonn",
            "resid_pos_rms", "resid_vel_rms", "category_id",
        ])
        for rank, idx in enumerate(neighbor_idx, start=1):
            writer.writerow([
                rank, int(idx), f"{dist[rank-1]:.10e}", f"{w[rank-1]:.10e}",
                f"{float(raw['r_pair'][idx]):.10e}",
                f"{float(raw['v_rad_norm'][idx]):.10e}",
                f"{float(raw['v_tan_norm'][idx]):.10e}",
                f"{float(raw['min_r_nonn'][idx]):.10e}",
                f"{float(raw['relE_nonn'][idx]):.10e}",
                f"{float(pos_rms[rank-1]):.10e}",
                f"{float(vel_rms[rank-1]):.10e}",
                int(raw['category_id'][idx]),
            ])

    meta = {
        "n_samples": int(len(Y)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "feature_dim": int(X_train_raw.shape[1]),
        "neighbor_distance_min": float(np.min(dist)),
        "neighbor_distance_med": float(np.median(dist)),
        "neighbor_distance_max": float(np.max(dist)),
        "neighbor_weight_max": float(np.max(w)),
        "neighbor_pos_rms_med": float(np.median(pos_rms)),
        "neighbor_vel_rms_med": float(np.median(vel_rms)),
        "neighbor_csv": neigh_csv,
        "pred_unscaled_pos_rms": float(np.sqrt(np.mean(np.sum(pred18_unscaled[:9].reshape(3,3)**2, axis=1)))),
        "pred_unscaled_vel_rms": float(np.sqrt(np.mean(np.sum(pred18_unscaled[9:].reshape(3,3)**2, axis=1)))),
    }
    return pred18, meta


def write_metrics_csv(path: str, rows: list[Dict[str, float | str]]) -> None:
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(out_dir: str,
               local_times: np.ndarray,
               pos_ias_w: np.ndarray,
               pos_no_w: np.ndarray,
               x_knn_exit: np.ndarray,
               post_times_abs: np.ndarray,
               pos_ias_post_local: np.ndarray,
               pos_no_post: np.ndarray,
               pos_knn_post: np.ndarray) -> None:
    if not HAS_MPL:
        return
    no_pos_rms_w = traj_rms(pos_no_w, pos_ias_w)
    knn_pos_exit = state_rms(x_knn_exit, pos_ias_w[-1])

    plt.figure(figsize=(9, 5))
    plt.plot(local_times, no_pos_rms_w, label="noNN window vs local IAS15")
    plt.scatter([local_times[-1]], [knn_pos_exit], marker="x", s=90, label="kNN-corrected exit")
    plt.xlabel("time from event start (yr)")
    plt.ylabel("RMS position error (AU)")
    plt.title("Single-event local 0.5-year replay: kNN residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "local_window_rms_knn.png"), dpi=200)
    plt.close()

    no_post = traj_rms(pos_no_post, pos_ias_post_local)
    knn_post = traj_rms(pos_knn_post, pos_ias_post_local)
    plt.figure(figsize=(9, 5))
    plt.plot(post_times_abs, no_post, label="noNN branch vs local IAS15")
    plt.plot(post_times_abs, knn_post, label="kNN branch vs local IAS15")
    plt.xlabel("absolute time (yr)")
    plt.ylabel("RMS position error (AU)")
    plt.title("Post-window branch replay: kNN residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "post_window_rms_knn.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-event replay diagnostic for kNN residual surrogate.")
    ap.add_argument("--data", default="encounter_surrogate_v2_event_enriched.npz")
    ap.add_argument("--ic", default="IC1", choices=["IC1"])
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--event-time", type=float, default=82.56)
    ap.add_argument("--window-years", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--feature-mode", default="xrel18", choices=["compact3", "compact5", "compact8", "hybrid26", "xrel18"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--gate-vr", type=float, default=-0.40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--energy-gate", type=float, default=0.20)
    ap.add_argument("--max-radius-gate", type=float, default=100.0)
    ap.add_argument("--n-window-samples", type=int, default=201)
    ap.add_argument("--n-post-samples", type=int, default=700)
    ap.add_argument("--out-dir", default="single_event_replay_knn_v1_t82p56")
    ap.add_argument("--no-com-project", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    x0, v0, m = get_ic(args.ic)

    print("=" * 96)
    print("SINGLE-EVENT REPLAY DIAGNOSTIC -- kNN RESIDUAL")
    print(f"  data         : {args.data}")
    print(f"  event_time   : {args.event_time:.6f} yr")
    print(f"  dt/window/T  : {args.dt:.6f} / {args.window_years:.6f} / {args.T:.6f} yr")
    print(f"  kNN          : mode={args.feature_mode}, k={args.k}, alpha={args.alpha:.3f}, gate vr<{args.gate_vr:.3f}")
    print(f"  out_dir      : {args.out_dir}")
    print("=" * 96)

    # 1. Reconstruct event state from revised noNN rollout.
    print("[1/5] advancing revised noNN to event time...")
    x_evt, v_evt, stats_pre = advance_nonn_to_time(x0, v0, m, args.dt, args.event_time)
    E_evt = total_energy(x_evt, v_evt, m, softened_pe=False)
    X_rel18, vr, vt, r_pair = make_X_rel18(x_evt, v_evt, m, args.dt)
    print(f"      event features: r={r_pair:.8f} AU, vr={vr:+.6f}, vt={vt:.6f}, min_r={min_pair_distance(x_evt):.8f}")

    # 2. Local IAS15 and noNN window.
    print("[2/5] local 0.5-year IAS15 and noNN replay...")
    local_times = np.linspace(0.0, float(args.window_years), int(args.n_window_samples))
    pos_ias_w, vel_ias_w, t_ias_w = simulate_ias15_at_times(x_evt, v_evt, m, local_times)
    pos_no_w, vel_no_w, stats_win = integrate_nonn(x_evt, v_evt, m, args.dt, args.window_years, local_times)
    x_no_exit, v_no_exit = pos_no_w[-1].copy(), vel_no_w[-1].copy()
    x_ias_exit, v_ias_exit = pos_ias_w[-1].copy(), vel_ias_w[-1].copy()
    E_no_exit = total_energy(x_no_exit, v_no_exit, m, softened_pe=False)
    relE_nonn = rel_energy(E_no_exit, E_evt)
    min_r_nonn = float(stats_win["min_r"])
    min_r_ias = float(np.min([min_pair_distance(p) for p in pos_ias_w]))

    # 3. kNN residual correction at window exit.
    print("[3/5] estimating kNN residual and applying correction...")
    pred18, knn_meta = find_knn_for_event(
        data_path=args.data,
        feature_mode=args.feature_mode,
        k=args.k,
        alpha=args.alpha,
        seed=args.seed,
        x_evt=x_evt,
        v_evt=v_evt,
        m=m,
        dt=args.dt,
        r_pair=r_pair,
        vr=vr,
        vt=vt,
        min_r_nonn=min_r_nonn,
        relE_nonn=relE_nonn,
        out_dir=args.out_dir,
    )
    pred_rx = pred18[:9].reshape(3, 3)
    pred_rv = pred18[9:].reshape(3, 3)

    gate_fires = bool(vr < float(args.gate_vr))
    if gate_fires:
        x_knn_exit = x_no_exit + pred_rx
        v_knn_exit = v_no_exit + pred_rv
        if not args.no_com_project:
            x_knn_exit, v_knn_exit = project_com_to_reference(x_knn_exit, v_knn_exit, x_no_exit, v_no_exit, m)
    else:
        x_knn_exit = x_no_exit.copy()
        v_knn_exit = v_no_exit.copy()

    E_knn_exit = total_energy(x_knn_exit, v_knn_exit, m, softened_pe=False)
    relE_knn = rel_energy(E_knn_exit, E_evt)
    pred_pos_norm = state_rms(pred_rx, np.zeros_like(pred_rx))
    pred_vel_norm = state_rms(pred_rv, np.zeros_like(pred_rv))

    accepted = gate_fires
    reason = "used" if gate_fires else "gate_vr_not_fired"
    if accepted and not all_finite(x_knn_exit, v_knn_exit):
        accepted = False; reason = "nonfinite"
    elif accepted and max_radius(x_knn_exit) > float(args.max_radius_gate):
        accepted = False; reason = "max_radius_gate"
    elif accepted and relE_knn > float(args.energy_gate):
        accepted = False; reason = "energy_gate"

    # For branch continuation, use fallback if rejected.
    x_branch_exit = x_knn_exit if accepted else x_no_exit
    v_branch_exit = v_knn_exit if accepted else v_no_exit

    # 4. Continue noNN-exit and kNN/fallback branches.
    print("[4/5] continuing post-window branches...")
    t_exit_abs = float(args.event_time) + float(args.window_years)
    post_duration = max(0.0, float(args.T) - t_exit_abs)
    post_times = np.linspace(0.0, post_duration, int(args.n_post_samples)) if post_duration > 0 else np.array([0.0])
    post_times_abs = t_exit_abs + post_times

    ias_query_times = float(args.window_years) + post_times
    pos_ias_post_local, vel_ias_post_local, t_ias_post = simulate_ias15_at_times(x_evt, v_evt, m, ias_query_times)
    pos_no_post, vel_no_post, stats_no_post = integrate_nonn(x_no_exit, v_no_exit, m, args.dt, post_duration, post_times)
    pos_knn_post, vel_knn_post, stats_knn_post = integrate_nonn(x_branch_exit, v_branch_exit, m, args.dt, post_duration, post_times)

    # 5. Metrics and output.
    print("[5/5] writing diagnostics...")
    no_pos_exit_err = state_rms(x_no_exit, x_ias_exit)
    knn_pos_exit_err = state_rms(x_knn_exit, x_ias_exit)
    branch_pos_exit_err = state_rms(x_branch_exit, x_ias_exit)
    no_vel_exit_err = state_rms(v_no_exit, v_ias_exit)
    knn_vel_exit_err = state_rms(v_knn_exit, v_ias_exit)
    branch_vel_exit_err = state_rms(v_branch_exit, v_ias_exit)

    true_rx = x_ias_exit - x_no_exit
    true_rv = v_ias_exit - v_no_exit
    true_pos_norm = state_rms(true_rx, np.zeros_like(true_rx))
    true_vel_norm = state_rms(true_rv, np.zeros_like(true_rv))
    pred_resid_pos_err = state_rms(pred_rx, true_rx)
    pred_resid_vel_err = state_rms(pred_rv, true_rv)

    no_post_pos = traj_rms(pos_no_post, pos_ias_post_local)
    knn_post_pos = traj_rms(pos_knn_post, pos_ias_post_local)
    no_post_vel = traj_rms(vel_no_post, vel_ias_post_local)
    knn_post_vel = traj_rms(vel_knn_post, vel_ias_post_local)

    metrics_rows = [
        {
            "section": "local_exit",
            "method": "noNN",
            "pos_err": no_pos_exit_err,
            "vel_err": no_vel_exit_err,
            "relE": relE_nonn,
            "accepted": "baseline",
        },
        {
            "section": "local_exit",
            "method": "kNN_raw_candidate",
            "pos_err": knn_pos_exit_err,
            "vel_err": knn_vel_exit_err,
            "relE": relE_knn,
            "accepted": str(accepted),
        },
        {
            "section": "local_exit",
            "method": "kNN_or_fallback_branch",
            "pos_err": branch_pos_exit_err,
            "vel_err": branch_vel_exit_err,
            "relE": relE_knn if accepted else relE_nonn,
            "accepted": str(accepted),
        },
        {
            "section": "post_window_timeavg",
            "method": "noNN_branch",
            "pos_err": float(np.mean(no_post_pos)),
            "vel_err": float(np.mean(no_post_vel)),
            "relE": "",
            "accepted": "baseline",
        },
        {
            "section": "post_window_timeavg",
            "method": "kNN_or_fallback_branch",
            "pos_err": float(np.mean(knn_post_pos)),
            "vel_err": float(np.mean(knn_post_vel)),
            "relE": "",
            "accepted": str(accepted),
        },
        {
            "section": "post_window_final",
            "method": "noNN_branch",
            "pos_err": float(no_post_pos[-1]),
            "vel_err": float(no_post_vel[-1]),
            "relE": "",
            "accepted": "baseline",
        },
        {
            "section": "post_window_final",
            "method": "kNN_or_fallback_branch",
            "pos_err": float(knn_post_pos[-1]),
            "vel_err": float(knn_post_vel[-1]),
            "relE": "",
            "accepted": str(accepted),
        },
    ]
    metrics_csv = os.path.join(args.out_dir, "single_event_replay_knn_metrics.csv")
    write_metrics_csv(metrics_csv, metrics_rows)

    np.savez_compressed(
        os.path.join(args.out_dir, "single_event_replay_knn_arrays.npz"),
        local_times=local_times,
        post_times_abs=post_times_abs,
        x_evt=x_evt,
        v_evt=v_evt,
        pos_ias_w=pos_ias_w,
        vel_ias_w=vel_ias_w,
        pos_no_w=pos_no_w,
        vel_no_w=vel_no_w,
        pred_rx=pred_rx,
        pred_rv=pred_rv,
        x_no_exit=x_no_exit,
        v_no_exit=v_no_exit,
        x_knn_exit=x_knn_exit,
        v_knn_exit=v_knn_exit,
        x_branch_exit=x_branch_exit,
        v_branch_exit=v_branch_exit,
        pos_ias_post_local=pos_ias_post_local,
        vel_ias_post_local=vel_ias_post_local,
        pos_no_post=pos_no_post,
        vel_no_post=vel_no_post,
        pos_knn_post=pos_knn_post,
        vel_knn_post=vel_knn_post,
        event_features=np.array([r_pair, vr, vt, min_r_nonn, relE_nonn], dtype=np.float64),
        accepted=np.array([accepted]),
        reason=np.array([reason]),
    )

    maybe_plot(args.out_dir, local_times, pos_ias_w, pos_no_w, x_branch_exit, post_times_abs,
               pos_ias_post_local, pos_no_post, pos_knn_post)

    local_pos_impr = 100.0 * (1.0 - branch_pos_exit_err / max(no_pos_exit_err, 1e-12))
    local_vel_impr = 100.0 * (1.0 - branch_vel_exit_err / max(no_vel_exit_err, 1e-12))
    post_pos_gain = 100.0 * (1.0 - float(np.mean(knn_post_pos)) / max(float(np.mean(no_post_pos)), 1e-12))
    post_vel_gain = 100.0 * (1.0 - float(np.mean(knn_post_vel)) / max(float(np.mean(no_post_vel)), 1e-12))

    verdict = ""
    if not gate_fires:
        verdict = "NO EVENT: kNN gate did not fire at this event."
    elif not accepted:
        verdict = f"FALLBACK: kNN candidate rejected by safety gate ({reason})."
    elif branch_pos_exit_err < no_pos_exit_err and branch_vel_exit_err < no_vel_exit_err:
        verdict = "LOCAL PASS: kNN improves both position and velocity at the 0.5-year encounter exit."
    elif branch_pos_exit_err < no_pos_exit_err or branch_vel_exit_err < no_vel_exit_err:
        verdict = "PARTIAL LOCAL PASS: kNN improves one local metric but not both."
    else:
        verdict = "LOCAL FAIL: kNN is worse already at the 0.5-year encounter exit."

    summary_path = os.path.join(args.out_dir, "single_event_replay_knn_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        def w(line: str = ""):
            print(line, file=f)

        w("Single-event replay diagnostic -- kNN residual surrogate")
        w("=" * 96)
        w(f"data                       : {args.data}")
        w(f"ic                         : {args.ic}")
        w(f"event_time                 : {args.event_time:.8f} yr")
        w(f"dt/window/T                : {args.dt:.8f} / {args.window_years:.8f} / {args.T:.8f} yr")
        w(f"kNN mode/k/alpha/gate       : {args.feature_mode} / {args.k} / {args.alpha:.4f} / vr<{args.gate_vr:.4f}")
        w(f"seed/split train-val-test   : {args.seed} / {knn_meta['n_train']} / {knn_meta['n_val']} / {knn_meta['n_test']}")
        w(f"feature_dim                 : {knn_meta['feature_dim']}")
        w(f"com_project                 : {not args.no_com_project}")
        w("")
        w("Event features")
        w("-" * 96)
        w(f"r_pair                     : {r_pair:.8f} AU")
        w(f"v_rad_norm                 : {vr:+.8f}")
        w(f"v_tan_norm                 : {vt:.8f}")
        w(f"window min_r IAS15         : {min_r_ias:.8f} AU")
        w(f"window min_r noNN          : {min_r_nonn:.8f} AU")
        w(f"relE noNN window           : {relE_nonn:.8e}")
        w("")
        w("kNN neighbor diagnostics")
        w("-" * 96)
        w(f"neighbor distance min/med/max: {knn_meta['neighbor_distance_min']:.6e} / {knn_meta['neighbor_distance_med']:.6e} / {knn_meta['neighbor_distance_max']:.6e}")
        w(f"neighbor max weight          : {knn_meta['neighbor_weight_max']:.6f}")
        w(f"neighbor residual pos med    : {knn_meta['neighbor_pos_rms_med']:.8e}")
        w(f"neighbor residual vel med    : {knn_meta['neighbor_vel_rms_med']:.8e}")
        w(f"neighbor csv                 : {knn_meta['neighbor_csv']}")
        w("")
        w("Residual size")
        w("-" * 96)
        w(f"true residual pos RMS        : {true_pos_norm:.8e}")
        w(f"true residual vel RMS        : {true_vel_norm:.8e}")
        w(f"kNN residual pos RMS         : {pred_pos_norm:.8e}  ({pred_pos_norm/max(true_pos_norm,1e-12):.2f}x true)")
        w(f"kNN residual vel RMS         : {pred_vel_norm:.8e}  ({pred_vel_norm/max(true_vel_norm,1e-12):.2f}x true)")
        w(f"unscaled kNN pos RMS         : {knn_meta['pred_unscaled_pos_rms']:.8e}")
        w(f"unscaled kNN vel RMS         : {knn_meta['pred_unscaled_vel_rms']:.8e}")
        w(f"pred-vs-true pos err RMS     : {pred_resid_pos_err:.8e}")
        w(f"pred-vs-true vel err RMS     : {pred_resid_vel_err:.8e}")
        w("")
        w("Local 0.5-year encounter exit")
        w("-" * 96)
        w(f"gate_fires                  : {gate_fires}")
        w(f"accepted                    : {accepted}")
        w(f"reason                      : {reason}")
        w(f"relE kNN candidate          : {relE_knn:.8e}")
        w(f"noNN pos exit err           : {no_pos_exit_err:.8e}")
        w(f"kNN candidate pos exit err  : {knn_pos_exit_err:.8e}")
        w(f"branch pos exit err         : {branch_pos_exit_err:.8e}")
        w(f"local pos improvement       : {local_pos_impr:+.2f}%")
        w(f"noNN vel exit err           : {no_vel_exit_err:.8e}")
        w(f"kNN candidate vel exit err  : {knn_vel_exit_err:.8e}")
        w(f"branch vel exit err         : {branch_vel_exit_err:.8e}")
        w(f"local vel improvement       : {local_vel_impr:+.2f}%")
        w("")
        w("Post-window continuation to T")
        w("-" * 96)
        w(f"noNN branch pos timeavg      : {float(np.mean(no_post_pos)):.8e}")
        w(f"kNN branch pos timeavg       : {float(np.mean(knn_post_pos)):.8e}")
        w(f"post pos gain vs noNN        : {post_pos_gain:+.2f}%")
        w(f"noNN branch vel timeavg      : {float(np.mean(no_post_vel)):.8e}")
        w(f"kNN branch vel timeavg       : {float(np.mean(knn_post_vel)):.8e}")
        w(f"post vel gain vs noNN        : {post_vel_gain:+.2f}%")
        w(f"noNN branch pos final        : {float(no_post_pos[-1]):.8e}")
        w(f"kNN branch pos final         : {float(knn_post_pos[-1]):.8e}")
        w(f"noNN branch vel final        : {float(no_post_vel[-1]):.8e}")
        w(f"kNN branch vel final         : {float(knn_post_vel[-1]):.8e}")
        w("")
        w("Decision")
        w("-" * 96)
        w(verdict)
        if accepted and branch_pos_exit_err < no_pos_exit_err and branch_vel_exit_err < no_vel_exit_err:
            if post_pos_gain < 0 or post_vel_gain < 0:
                w("NOTE: kNN improves local exit but worsens at least one long continuation metric; this indicates branch sensitivity after a locally better correction.")
            else:
                w("NOTE: kNN improves the local exit and the post-window continuation metrics in this replay.")
        w("")
        w("Files written")
        w("-" * 96)
        w(summary_path)
        w(metrics_csv)
        w(os.path.join(args.out_dir, "single_event_replay_knn_arrays.npz"))
        w(os.path.join(args.out_dir, "single_event_replay_knn_neighbors.csv"))

    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
