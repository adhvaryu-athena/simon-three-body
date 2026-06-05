"""
diagnose_event_nearest_neighbors_v1.py

Nearest-neighbor diagnostic for the encounter-level surrogate failure at the
real rollout event t≈82.56 yr.

Purpose
-------
The single-event replay showed that the enriched surrogate over-corrected the
exact dt=0.08 rollout event. This script asks why by comparing that exact event
state against the enriched training/prototype dataset.

It reconstructs the exact revised-noNN event state, computes the true local
0.5-year residual (IAS15_exit - noNN_exit), optionally computes the model's
predicted residual, then finds nearest dataset samples in:

  1. full model input space X_rel18,
  2. compact encounter scalar space [r_pair, v_rad_norm, v_tan_norm], and
  3. diagnostic scalar space [r_pair, v_rad_norm, v_tan_norm, min_r_ias, min_r_nonn].

Outputs
-------
  <out_dir>/<prefix>_summary.txt
  <out_dir>/<prefix>_nearest_Xrel18.csv
  <out_dir>/<prefix>_nearest_r_vr_vt.csv
  <out_dir>/<prefix>_nearest_with_minr.csv

Typical run
-----------
python -B diagnose_event_nearest_neighbors_v1.py ^
  --data encounter_surrogate_v2_event_enriched.npz ^
  --model encounter_surrogate_v2_event_enriched_velocitysafe.pt ^
  --event-time 82.56 --dt 0.08 --window-years 0.5 ^
  --out-dir event_nn_diag_t82p56
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None

# Reuse the already-tested single-event replay functions so that the event state
# and local window definition exactly match the previous diagnostic.
try:
    from diagnose_single_event_replay_v1 import (
        get_ic,
        advance_nonn_to_time,
        make_X_rel18,
        simulate_ias15_at_times,
        integrate_nonn,
        state_rms,
        min_pair_distance,
        load_surrogate_model,
        predict_residual,
    )
except Exception as e:
    raise ImportError(
        "Could not import helper functions from diagnose_single_event_replay_v1.py. "
        "Put this file in the same folder as diagnose_single_event_replay_v1.py.\n"
        f"Original import error: {e}"
    )


CATEGORY_NAMES = {
    0: "strong_approach",
    1: "weak_side",
    2: "recede",
    3: "broad",
    4: "event_strong_approach",
}


def safe_std(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x, axis=0).astype(np.float64)
    s[~np.isfinite(s)] = 1.0
    s[s < 1e-8] = 1.0
    return s


def rms_state(arr: np.ndarray) -> np.ndarray:
    """Per-row RMS over bodies of 3D vector field: (n,3,3) -> (n,)."""
    return np.sqrt(np.mean(np.sum(arr.astype(np.float64) ** 2, axis=2), axis=1))


def flatten_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a.astype(np.float64).ravel() ** 2)))


def cosine_to_event(rows: np.ndarray, event_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity between flattened rows and event_vec."""
    A = rows.reshape(rows.shape[0], -1).astype(np.float64)
    e = event_vec.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(A, axis=1) * (np.linalg.norm(e) + 1e-30)
    return (A @ e) / (denom + 1e-30)


def percentile_rank(values: np.ndarray, x: float) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(100.0 * np.mean(values <= x))


def zdist(A: np.ndarray, a: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
    A = A.astype(np.float64)
    a = a.astype(np.float64)
    if mean is None:
        mean = np.nanmean(A, axis=0)
    if std is None:
        std = safe_std(A)
    Z = (A - mean) / (std + 1e-12)
    z = (a - mean) / (std + 1e-12)
    return np.sqrt(np.mean((Z - z) ** 2, axis=1))


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path)
    required = [
        "X_rel18", "r_pair", "v_rad_norm", "v_tan_norm",
        "min_r_ias", "min_r_nonn", "residual_x", "residual_v",
        "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final",
    ]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"Dataset is missing required fields: {missing}\nAvailable: {d.files}")
    out = {k: d[k] for k in d.files}
    return out


def reconstruct_event(args) -> Dict[str, np.ndarray]:
    x0, v0, m = get_ic(args.ic)
    x_evt, v_evt, _ = advance_nonn_to_time(x0, v0, m, args.dt, args.event_time)
    X_rel18, vr, vt, r_pair = make_X_rel18(x_evt, v_evt, m, args.dt)

    local_times = np.linspace(0.0, float(args.window_years), int(args.n_window_samples))
    pos_ias_w, vel_ias_w, _ = simulate_ias15_at_times(x_evt, v_evt, m, local_times)
    pos_no_w, vel_no_w, _ = integrate_nonn(x_evt, v_evt, m, args.dt, args.window_years, local_times)

    x_ias_exit = pos_ias_w[-1].copy()
    v_ias_exit = vel_ias_w[-1].copy()
    x_no_exit = pos_no_w[-1].copy()
    v_no_exit = vel_no_w[-1].copy()
    true_rx = x_ias_exit - x_no_exit
    true_rv = v_ias_exit - v_no_exit

    return {
        "m": m,
        "x_evt": x_evt,
        "v_evt": v_evt,
        "X_rel18": X_rel18,
        "r_pair": np.array(r_pair),
        "v_rad_norm": np.array(vr),
        "v_tan_norm": np.array(vt),
        "min_r_event": np.array(min_pair_distance(x_evt)),
        "min_r_ias": np.array(float(np.min([min_pair_distance(p) for p in pos_ias_w]))),
        "min_r_nonn": np.array(float(np.min([min_pair_distance(p) for p in pos_no_w]))),
        "x_ias_exit": x_ias_exit,
        "v_ias_exit": v_ias_exit,
        "x_nonn_exit": x_no_exit,
        "v_nonn_exit": v_no_exit,
        "true_residual_x": true_rx,
        "true_residual_v": true_rv,
        "true_pos_rms": np.array(state_rms(x_ias_exit, x_no_exit)),
        "true_vel_rms": np.array(state_rms(v_ias_exit, v_no_exit)),
    }


def maybe_predict_model(args, event: Dict[str, np.ndarray]):
    if args.no_model or not args.model:
        return None, None
    if torch is None:
        print("[warn] torch unavailable; skipping model prediction")
        return None, None
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = load_surrogate_model(args.model, device)
    pred_rx, pred_rv = predict_residual(model, event["X_rel18"], device)
    return pred_rx.astype(np.float64), pred_rv.astype(np.float64)


def make_neighbor_rows(data: Dict[str, np.ndarray], idx: np.ndarray, dist: np.ndarray,
                       event_rx: np.ndarray, event_rv: np.ndarray) -> List[Dict[str, object]]:
    rx = data["residual_x"].astype(np.float64)
    rv = data["residual_v"].astype(np.float64)
    pos_rms = data.get("pos_residual_rms", rms_state(rx)).astype(np.float64)
    vel_rms = data.get("vel_residual_rms", rms_state(rv)).astype(np.float64)
    cos_x = cosine_to_event(rx[idx], event_rx)
    cos_v = cosine_to_event(rv[idx], event_rv)

    rows = []
    for rank, j in enumerate(idx, start=1):
        cat_id = int(data["category_id"][j]) if "category_id" in data else -1
        row = {
            "rank": rank,
            "index": int(j),
            "distance": float(dist[j]),
            "category_id": cat_id,
            "category": CATEGORY_NAMES.get(cat_id, str(cat_id)),
            "source_row_idx": int(data["source_row_idx"][j]) if "source_row_idx" in data else -1,
            "r_pair": float(data["r_pair"][j]),
            "v_rad_norm": float(data["v_rad_norm"][j]),
            "v_tan_norm": float(data["v_tan_norm"][j]),
            "min_r_ias": float(data["min_r_ias"][j]),
            "min_r_nonn": float(data["min_r_nonn"][j]),
            "relE_nonn": float(data["relE_nonn"][j]) if "relE_nonn" in data else float("nan"),
            "pos_residual_rms": float(pos_rms[j]),
            "vel_residual_rms": float(vel_rms[j]),
            "pos_resid_cos_to_event": float(cos_x[rank-1]),
            "vel_resid_cos_to_event": float(cos_v[rank-1]),
        }
        rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_neighbors(name: str, rows: List[Dict[str, object]], event_pos: float, event_vel: float) -> List[str]:
    pos = np.array([float(r["pos_residual_rms"]) for r in rows], dtype=np.float64)
    vel = np.array([float(r["vel_residual_rms"]) for r in rows], dtype=np.float64)
    dx = np.array([float(r["distance"]) for r in rows], dtype=np.float64)
    cosx = np.array([float(r["pos_resid_cos_to_event"]) for r in rows], dtype=np.float64)
    cosv = np.array([float(r["vel_resid_cos_to_event"]) for r in rows], dtype=np.float64)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    lines = []
    lines.append(f"{name}")
    lines.append("-" * 88)
    lines.append(f"nearest distance min/med/max : {np.min(dx):.6e} / {np.median(dx):.6e} / {np.max(dx):.6e}")
    lines.append(f"neighbor pos residual med    : {np.median(pos):.8e}  (event true={event_pos:.8e}, ratio={np.median(pos)/(event_pos+1e-30):.2f}x)")
    lines.append(f"neighbor pos residual p10-p90: {np.percentile(pos,10):.8e} .. {np.percentile(pos,90):.8e}")
    lines.append(f"neighbor vel residual med    : {np.median(vel):.8e}  (event true={event_vel:.8e}, ratio={np.median(vel)/(event_vel+1e-30):.2f}x)")
    lines.append(f"neighbor vel residual p10-p90: {np.percentile(vel,10):.8e} .. {np.percentile(vel,90):.8e}")
    lines.append(f"residual direction cos pos   : med={np.median(cosx):+.3f}, p10={np.percentile(cosx,10):+.3f}, p90={np.percentile(cosx,90):+.3f}")
    lines.append(f"residual direction cos vel   : med={np.median(cosv):+.3f}, p10={np.percentile(cosv,10):+.3f}, p90={np.percentile(cosv,90):+.3f}")
    lines.append(f"category counts              : {cats}")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description="Nearest-neighbor diagnostic for exact encounter event.")
    ap.add_argument("--data", default="encounter_surrogate_v2_event_enriched.npz")
    ap.add_argument("--model", default="encounter_surrogate_v2_event_enriched_velocitysafe.pt")
    ap.add_argument("--ic", default="IC1", choices=["IC1"])
    ap.add_argument("--event-time", type=float, default=82.56)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--window-years", type=float, default=0.5)
    ap.add_argument("--n-window-samples", type=int, default=201)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--out-dir", default="event_nn_diag_t82p56")
    ap.add_argument("--prefix", default="nearest_neighbors_event_t82p56")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 92)
    print("EVENT NEAREST-NEIGHBOR DIAGNOSTIC")
    print(f"  data       : {args.data}")
    print(f"  model      : {args.model if not args.no_model else '(skipped)'}")
    print(f"  event time : {args.event_time:.6f} yr")
    print(f"  dt/window  : {args.dt:.6f} / {args.window_years:.6f} yr")
    print(f"  out_dir    : {args.out_dir}")
    print("=" * 92)

    data = load_dataset(args.data)
    n = len(data["r_pair"])
    event = reconstruct_event(args)
    pred_rx, pred_rv = maybe_predict_model(args, event)

    rx = data["residual_x"].astype(np.float64)
    rv = data["residual_v"].astype(np.float64)
    pos_rms = data.get("pos_residual_rms", rms_state(rx)).astype(np.float64)
    vel_rms = data.get("vel_residual_rms", rms_state(rv)).astype(np.float64)

    X = data["X_rel18"].astype(np.float64)
    X_event = event["X_rel18"].astype(np.float64)
    dist_x = zdist(X, X_event)

    scal3 = np.stack([data["r_pair"], data["v_rad_norm"], data["v_tan_norm"]], axis=1).astype(np.float64)
    scal3_event = np.array([event["r_pair"], event["v_rad_norm"], event["v_tan_norm"]], dtype=np.float64).reshape(3)
    dist_scal3 = zdist(scal3, scal3_event)

    scal5 = np.stack([data["r_pair"], data["v_rad_norm"], data["v_tan_norm"], data["min_r_ias"], data["min_r_nonn"]], axis=1).astype(np.float64)
    scal5_event = np.array([event["r_pair"], event["v_rad_norm"], event["v_tan_norm"], event["min_r_ias"], event["min_r_nonn"]], dtype=np.float64).reshape(5)
    dist_scal5 = zdist(scal5, scal5_event)

    k = min(int(args.k), n)
    idx_x = np.argsort(dist_x)[:k]
    idx_s3 = np.argsort(dist_scal3)[:k]
    idx_s5 = np.argsort(dist_scal5)[:k]

    rows_x = make_neighbor_rows(data, idx_x, dist_x, event["true_residual_x"], event["true_residual_v"])
    rows_s3 = make_neighbor_rows(data, idx_s3, dist_scal3, event["true_residual_x"], event["true_residual_v"])
    rows_s5 = make_neighbor_rows(data, idx_s5, dist_scal5, event["true_residual_x"], event["true_residual_v"])

    csv_x = os.path.join(args.out_dir, f"{args.prefix}_nearest_Xrel18.csv")
    csv_s3 = os.path.join(args.out_dir, f"{args.prefix}_nearest_r_vr_vt.csv")
    csv_s5 = os.path.join(args.out_dir, f"{args.prefix}_nearest_with_minr.csv")
    write_csv(csv_x, rows_x)
    write_csv(csv_s3, rows_s3)
    write_csv(csv_s5, rows_s5)

    event_pos = float(event["true_pos_rms"])
    event_vel = float(event["true_vel_rms"])
    lines: List[str] = []
    add = lines.append
    add("Event nearest-neighbor diagnostic summary")
    add("=" * 88)
    add(f"data                      : {args.data}")
    add(f"samples                   : {n}")
    add(f"event_time                : {args.event_time:.8f} yr")
    add(f"dt/window                 : {args.dt:.8f} / {args.window_years:.8f} yr")
    add(f"k                         : {k}")
    add("")
    add("Exact event features")
    add("-" * 88)
    add(f"r_pair                   : {float(event['r_pair']):.8f} AU")
    add(f"v_rad_norm               : {float(event['v_rad_norm']):+.8f}")
    add(f"v_tan_norm               : {float(event['v_tan_norm']):.8f}")
    add(f"min_r_event              : {float(event['min_r_event']):.8f} AU")
    add(f"window min_r IAS15       : {float(event['min_r_ias']):.8f} AU")
    add(f"window min_r noNN        : {float(event['min_r_nonn']):.8f} AU")
    add(f"true residual pos RMS    : {event_pos:.8e}")
    add(f"true residual vel RMS    : {event_vel:.8e}")
    add(f"event pos residual percentile in dataset: {percentile_rank(pos_rms, event_pos):.2f}%")
    add(f"event vel residual percentile in dataset: {percentile_rank(vel_rms, event_vel):.2f}%")
    if pred_rx is not None:
        pred_pos = state_rms(pred_rx, np.zeros_like(pred_rx))
        pred_vel = state_rms(pred_rv, np.zeros_like(pred_rv))
        pred_pos_err = state_rms(pred_rx, event["true_residual_x"])
        pred_vel_err = state_rms(pred_rv, event["true_residual_v"])
        add("")
        add("Model prediction on exact event")
        add("-" * 88)
        add(f"pred residual pos RMS    : {pred_pos:.8e}  ({pred_pos/(event_pos+1e-30):.2f}x true)")
        add(f"pred residual vel RMS    : {pred_vel:.8e}  ({pred_vel/(event_vel+1e-30):.2f}x true)")
        add(f"pred-vs-true pos err RMS : {pred_pos_err:.8e}")
        add(f"pred-vs-true vel err RMS : {pred_vel_err:.8e}")
    add("")
    add("Dataset residual distribution")
    add("-" * 88)
    add(f"pos residual RMS p05/med/p95 : {np.percentile(pos_rms,5):.8e} / {np.median(pos_rms):.8e} / {np.percentile(pos_rms,95):.8e}")
    add(f"vel residual RMS p05/med/p95 : {np.percentile(vel_rms,5):.8e} / {np.median(vel_rms):.8e} / {np.percentile(vel_rms,95):.8e}")
    add("")
    lines += summarize_neighbors("Nearest neighbors in full X_rel18 model-input space", rows_x, event_pos, event_vel)
    lines += summarize_neighbors("Nearest neighbors in compact [r_pair, v_rad_norm, v_tan_norm] space", rows_s3, event_pos, event_vel)
    lines += summarize_neighbors("Nearest neighbors in diagnostic [r, vr, vt, min_r_ias, min_r_nonn] space", rows_s5, event_pos, event_vel)

    # Simple diagnosis flags.
    med_x_pos = np.median([float(r["pos_residual_rms"]) for r in rows_x])
    med_x_vel = np.median([float(r["vel_residual_rms"]) for r in rows_x])
    med_s5_pos = np.median([float(r["pos_residual_rms"]) for r in rows_s5])
    med_s5_vel = np.median([float(r["vel_residual_rms"]) for r in rows_s5])
    min_x_dist = float(np.min(dist_x))
    pos_cos_med = np.median([float(r["pos_resid_cos_to_event"]) for r in rows_x])
    vel_cos_med = np.median([float(r["vel_resid_cos_to_event"]) for r in rows_x])

    add("Interpretation flags")
    add("-" * 88)
    add(f"closest X_rel18 distance     : {min_x_dist:.6e}")
    add(f"X-neighbor pos median / event: {med_x_pos/(event_pos+1e-30):.2f}x")
    add(f"X-neighbor vel median / event: {med_x_vel/(event_vel+1e-30):.2f}x")
    add(f"diagnostic-minr pos median / event: {med_s5_pos/(event_pos+1e-30):.2f}x")
    add(f"diagnostic-minr vel median / event: {med_s5_vel/(event_vel+1e-30):.2f}x")
    add(f"X-neighbor residual cosine med pos/vel: {pos_cos_med:+.3f} / {vel_cos_med:+.3f}")

    if min_x_dist > 1.5:
        diagnosis = "COVERAGE ISSUE: nearest full-state neighbors are not very close; generate states closer in full X_rel18, not just r/vr/vt."
    elif (med_x_pos > 8 * event_pos) or (med_x_vel > 8 * event_vel):
        diagnosis = "FEATURE/TARGET AMBIGUITY: nearest full-state neighbors have much larger residuals than this event; current inputs may not distinguish small-residual cases."
    elif (pos_cos_med < 0.2) or (vel_cos_med < 0.2):
        diagnosis = "MULTI-VALUED/NOISY TARGET: nearby samples do not agree on residual direction; model may average or overfit incompatible corrections."
    else:
        diagnosis = "MODEL/TRAINING ISSUE LIKELY: nearby samples look compatible with the event; investigate loss, normalization, regularization, and residual magnitude caps."
    add(f"DIAGNOSIS                  : {diagnosis}")
    add("")
    add("Files written")
    add("-" * 88)
    add(csv_x)
    add(csv_s3)
    add(csv_s5)

    summary_path = os.path.join(args.out_dir, f"{args.prefix}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
