"""
diagnose_knn_residual_surrogate_v1.py

Prototype diagnostic for an encounter-level residual surrogate that DOES NOT train a new NN.

Purpose
-------
The previous full-vector MLP residual surrogates could over-correct a real rollout event.
This script tests whether the DATA ITSELF contains useful local 18D residual directions
when queried by compact physical encounter features.

It uses k-nearest-neighbor residual averaging:

    R_knn = weighted average of true training residuals from local neighbors
    corrected = noNN_exit + alpha * R_knn

where alpha is a scalar blend in [0, 1].

This answers a key question before training another model:

    Can local compact-neighbor residual averaging beat noNN safely on held-out windows?

If yes, the next direction is a learned/gated blend model.
If no, the current data/features do not provide a reliable local 18D correction direction.

Example
-------
python -B diagnose_knn_residual_surrogate_v1.py --data encounter_surrogate_v2_event_enriched.npz --out-dir knn_residual_diag_v1

Outputs
-------
  knn_residual_diag_v1/knn_residual_summary.txt
  knn_residual_diag_v1/knn_residual_results.csv
  knn_residual_diag_v1/knn_best_predictions_test.npz
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


REQUIRED_FIELDS = [
    "residual_x", "residual_v",
    "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final",
    "m",
    "X_rel18",
    "r_pair", "v_rad_norm", "v_tan_norm",
    "min_r_nonn", "relE_nonn",
    "category_id",
]

CATEGORY_NAMES = {
    0: "strong_approach",
    1: "weak_side",
    2: "recede",
    3: "broad",
}


@dataclass
class Dataset:
    raw: Dict[str, np.ndarray]
    Y: np.ndarray  # (n,18) true residual [x9,v9]


def _safe_std(x: np.ndarray, axis=0) -> np.ndarray:
    s = np.nanstd(x, axis=axis).astype(np.float64)
    s = np.where(s < 1e-10, 1.0, s)
    return s


def load_dataset(path: str) -> Dataset:
    data = np.load(path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(f"{path} is missing fields: {missing}\nFound: {data.files}")
    raw = {k: data[k] for k in data.files}

    rx = np.asarray(raw["residual_x"], dtype=np.float64)
    rv = np.asarray(raw["residual_v"], dtype=np.float64)
    if rx.ndim != 3 or rx.shape[1:] != (3, 3):
        raise ValueError(f"residual_x must be (n,3,3), got {rx.shape}")
    if rv.ndim != 3 or rv.shape[1:] != (3, 3):
        raise ValueError(f"residual_v must be (n,3,3), got {rv.shape}")
    n = rx.shape[0]
    for k in REQUIRED_FIELDS:
        if len(raw[k]) != n:
            raise ValueError(f"Field {k} length mismatch: {len(raw[k])} vs {n}")
        if not np.all(np.isfinite(raw[k])):
            raise ValueError(f"Field {k} contains non-finite values")

    x_ias = np.asarray(raw["x_ias_final"], dtype=np.float64)
    x_no = np.asarray(raw["x_nonn_final"], dtype=np.float64)
    v_ias = np.asarray(raw["v_ias_final"], dtype=np.float64)
    v_no = np.asarray(raw["v_nonn_final"], dtype=np.float64)
    if not np.allclose(rx, x_ias - x_no, rtol=2e-5, atol=2e-8):
        raise ValueError("residual_x does not match x_ias_final - x_nonn_final")
    if not np.allclose(rv, v_ias - v_no, rtol=2e-5, atol=2e-8):
        raise ValueError("residual_v does not match v_ias_final - v_nonn_final")

    Y = np.concatenate([rx.reshape(n, 9), rv.reshape(n, 9)], axis=1).astype(np.float64)
    return Dataset(raw=raw, Y=Y)


def make_stratified_splits(raw: Dict[str, np.ndarray], seed: int,
                           train_frac: float = 0.70,
                           val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat = np.asarray(raw["category_id"], dtype=np.int64)
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for c in sorted(set(cat.tolist())):
        idx = np.where(cat == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])
    train = np.concatenate(train_parts); val = np.concatenate(val_parts); test = np.concatenate(test_parts)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def build_feature_space(raw: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    r = np.asarray(raw["r_pair"], dtype=np.float64)
    vr = np.asarray(raw["v_rad_norm"], dtype=np.float64)
    vt = np.asarray(raw["v_tan_norm"], dtype=np.float64)
    minr = np.asarray(raw["min_r_nonn"], dtype=np.float64)
    relE = np.asarray(raw["relE_nonn"], dtype=np.float64)
    m = np.asarray(raw["m"], dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"m must be (n,3), got {m.shape}")
    logm = np.log(np.clip(m, 1e-30, None))
    xrel = np.asarray(raw["X_rel18"], dtype=np.float64)
    if xrel.ndim != 2 or xrel.shape[1] != 18:
        raise ValueError(f"X_rel18 must be (n,18), got {xrel.shape}")

    if mode == "compact3":
        X = np.column_stack([r, vr, vt])
    elif mode == "compact5":
        X = np.column_stack([r, vr, vt, minr, relE])
    elif mode == "compact8":
        X = np.column_stack([r, vr, vt, minr, relE, logm[:, 0], logm[:, 1], logm[:, 2]])
    elif mode == "hybrid26":
        X = np.column_stack([xrel, r, vr, vt, minr, relE, logm[:, 0], logm[:, 1], logm[:, 2]])
    elif mode == "xrel18":
        X = xrel
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Feature mode {mode} produced non-finite values")
    return X.astype(np.float64)


def rms_pos_vel_from_resid(resid18: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = resid18.shape[0]
    rx = resid18[:, :9].reshape(n, 3, 3)
    rv = resid18[:, 9:].reshape(n, 3, 3)
    pos = np.sqrt(np.mean(np.sum(rx ** 2, axis=2), axis=1))
    vel = np.sqrt(np.mean(np.sum(rv ** 2, axis=2), axis=1))
    return pos, vel


def evaluate_prediction(raw: Dict[str, np.ndarray], idx: np.ndarray, pred_resid: np.ndarray,
                        unsafe_factor: float = 2.0) -> Dict[str, float]:
    x_no = np.asarray(raw["x_nonn_final"], dtype=np.float64)[idx]
    v_no = np.asarray(raw["v_nonn_final"], dtype=np.float64)[idx]
    x_ias = np.asarray(raw["x_ias_final"], dtype=np.float64)[idx]
    v_ias = np.asarray(raw["v_ias_final"], dtype=np.float64)[idx]

    n = len(idx)
    prx = pred_resid[:, :9].reshape(n, 3, 3)
    prv = pred_resid[:, 9:].reshape(n, 3, 3)
    x_corr = x_no + prx
    v_corr = v_no + prv

    base_pos = np.sqrt(np.mean(np.sum((x_no - x_ias) ** 2, axis=2), axis=1))
    corr_pos = np.sqrt(np.mean(np.sum((x_corr - x_ias) ** 2, axis=2), axis=1))
    base_vel = np.sqrt(np.mean(np.sum((v_no - v_ias) ** 2, axis=2), axis=1))
    corr_vel = np.sqrt(np.mean(np.sum((v_corr - v_ias) ** 2, axis=2), axis=1))

    pos_ratio = corr_pos / np.clip(base_pos, 1e-12, None)
    vel_ratio = corr_vel / np.clip(base_vel, 1e-12, None)
    unsafe_pos = pos_ratio > unsafe_factor
    unsafe_vel = vel_ratio > unsafe_factor
    both_success = (corr_pos < base_pos) & (corr_vel < base_vel)

    def med(a): return float(np.median(a))
    def mean(a): return float(np.mean(a))
    def pct(a, p): return float(np.percentile(a, p))

    out = {
        "base_pos_mean": mean(base_pos),
        "method_pos_mean": mean(corr_pos),
        "base_pos_med": med(base_pos),
        "method_pos_med": med(corr_pos),
        "base_pos_p95": pct(base_pos, 95),
        "method_pos_p95": pct(corr_pos, 95),
        "pos_impr_mean_pct": float(100.0 * (1.0 - mean(corr_pos) / max(mean(base_pos), 1e-12))),
        "pos_impr_med_pct": float(100.0 * (1.0 - med(corr_pos) / max(med(base_pos), 1e-12))),
        "pos_success_frac": float(np.mean(corr_pos < base_pos)),
        "pos_unsafe_frac": float(np.mean(unsafe_pos)),
        "base_vel_mean": mean(base_vel),
        "method_vel_mean": mean(corr_vel),
        "base_vel_med": med(base_vel),
        "method_vel_med": med(corr_vel),
        "base_vel_p95": pct(base_vel, 95),
        "method_vel_p95": pct(corr_vel, 95),
        "vel_impr_mean_pct": float(100.0 * (1.0 - mean(corr_vel) / max(mean(base_vel), 1e-12))),
        "vel_impr_med_pct": float(100.0 * (1.0 - med(corr_vel) / max(med(base_vel), 1e-12))),
        "vel_success_frac": float(np.mean(corr_vel < base_vel)),
        "vel_unsafe_frac": float(np.mean(unsafe_vel)),
        "any_unsafe_frac": float(np.mean(unsafe_pos | unsafe_vel)),
        "both_success_frac": float(np.mean(both_success)),
        "pos_ratio_p95": pct(pos_ratio, 95),
        "vel_ratio_p95": pct(vel_ratio, 95),
    }
    return out


def knn_predict_residual(X_train: np.ndarray, Y_train: np.ndarray, X_query: np.ndarray,
                         k: int, weight_power: float = 2.0, batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Return weighted kNN residual and mean neighbor distance for each query."""
    n_query = X_query.shape[0]
    pred = np.zeros((n_query, Y_train.shape[1]), dtype=np.float64)
    mean_dist = np.zeros(n_query, dtype=np.float64)

    # Squared distance computed in batches to keep memory stable.
    for start in range(0, n_query, batch):
        end = min(start + batch, n_query)
        Q = X_query[start:end]
        # d2: (b, n_train)
        q2 = np.sum(Q ** 2, axis=1, keepdims=True)
        t2 = np.sum(X_train ** 2, axis=1)[None, :]
        d2 = np.maximum(q2 + t2 - 2.0 * (Q @ X_train.T), 0.0)
        kk = min(k, X_train.shape[0])
        part = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        for bi in range(end - start):
            ids = part[bi]
            dist = np.sqrt(d2[bi, ids] + 1e-30)
            order = np.argsort(dist)
            ids = ids[order]
            dist = dist[order]
            # inverse-distance weighting; exact duplicate gets dominant weight safely.
            w = 1.0 / np.power(dist + 1e-8, weight_power)
            w = w / np.sum(w)
            pred[start + bi] = np.sum(Y_train[ids] * w[:, None], axis=0)
            mean_dist[start + bi] = float(np.mean(dist))
    return pred, mean_dist


def format_pct(x: float) -> str:
    return f"{100.0*x:6.2f}%"


def main():
    ap = argparse.ArgumentParser(description="Diagnose compact-kNN residual surrogate before training another NN.")
    ap.add_argument("--data", required=True, help="Encounter surrogate .npz dataset")
    ap.add_argument("--out-dir", default="knn_residual_diag_v1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--k-list", default="5,10,20,40,80", help="Comma-separated k values")
    ap.add_argument("--alpha-list", default="0.25,0.50,0.75,1.00", help="Comma-separated blend strengths")
    ap.add_argument("--feature-modes", default="compact3,compact5,compact8,hybrid26,xrel18")
    ap.add_argument("--gate-vr", type=float, default=-0.40, help="Apply correction only when v_rad_norm < gate-vr for gated metrics")
    ap.add_argument("--unsafe-factor", type=float, default=2.0)
    ap.add_argument("--neighbor-batch", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.data)
    raw, Y = ds.raw, ds.Y
    n = Y.shape[0]
    train_idx, val_idx, test_idx = make_stratified_splits(raw, seed=args.seed)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    alpha_list = [float(x.strip()) for x in args.alpha_list.split(",") if x.strip()]
    modes = [x.strip() for x in args.feature_modes.split(",") if x.strip()]

    # Baseline noNN = zero residual prediction.
    zero_pred = np.zeros((len(test_idx), 18), dtype=np.float64)
    baseline = evaluate_prediction(raw, test_idx, zero_pred, unsafe_factor=args.unsafe_factor)

    rows: List[Dict[str, object]] = []
    best = None
    best_pred = None
    best_neighbor_dist = None

    for mode in modes:
        X = build_feature_space(raw, mode)
        mean = X[train_idx].mean(axis=0)
        std = _safe_std(X[train_idx], axis=0)
        Xn = (X - mean) / (std + 1e-12)
        Xtr = Xn[train_idx]
        Ytr = Y[train_idx]
        Xte = Xn[test_idx]

        for k in k_list:
            pred_full, neigh_dist = knn_predict_residual(
                Xtr, Ytr, Xte, k=k, batch=args.neighbor_batch
            )
            for alpha in alpha_list:
                pred_alpha = alpha * pred_full

                # Full correction metrics.
                m_full = evaluate_prediction(raw, test_idx, pred_alpha, unsafe_factor=args.unsafe_factor)
                row_full: Dict[str, object] = {
                    "mode": mode, "k": k, "alpha": alpha, "gate": "full", "used": len(test_idx),
                    "use_frac": 1.0, "neighbor_dist_med": float(np.median(neigh_dist)),
                    **m_full,
                }
                rows.append(row_full)

                # Gated correction metrics: only apply if v_rad_norm < gate_vr.
                vr_test = np.asarray(raw["v_rad_norm"], dtype=np.float64)[test_idx]
                gate = vr_test < args.gate_vr
                pred_gate = np.zeros_like(pred_alpha)
                pred_gate[gate] = pred_alpha[gate]
                m_gate = evaluate_prediction(raw, test_idx, pred_gate, unsafe_factor=args.unsafe_factor)
                row_gate = {
                    "mode": mode, "k": k, "alpha": alpha, "gate": f"vr<{args.gate_vr:.2f}",
                    "used": int(np.sum(gate)), "use_frac": float(np.mean(gate)),
                    "neighbor_dist_med": float(np.median(neigh_dist)),
                    **m_gate,
                }
                rows.append(row_gate)

                # Score: prioritise low median errors and unsafe control.
                # Must beat baseline on both medians to be considered promising.
                score = (
                    m_gate["method_pos_med"]
                    + 0.5 * m_gate["method_vel_med"]
                    + 0.05 * m_gate["method_pos_p95"]
                    + 0.05 * m_gate["method_vel_p95"]
                    + 0.10 * m_gate["any_unsafe_frac"]
                )
                improves_both = (
                    m_gate["method_pos_med"] < baseline["base_pos_med"]
                    and m_gate["method_vel_med"] < baseline["base_vel_med"]
                )
                safe = m_gate["any_unsafe_frac"] <= 0.10
                candidate_key = (not improves_both, not safe, score)
                if best is None or candidate_key < best["key"]:
                    best = {"key": candidate_key, "row": row_gate, "score": score, "mode": mode, "k": k, "alpha": alpha}
                    best_pred = pred_gate.copy()
                    best_neighbor_dist = neigh_dist.copy()

    # Sort rows by a useful score for display.
    def row_score(r: Dict[str, object]) -> float:
        return float(r["method_pos_med"]) + 0.5 * float(r["method_vel_med"]) + 0.05 * float(r["any_unsafe_frac"])

    rows_sorted = sorted(rows, key=row_score)

    # Save CSV.
    csv_path = os.path.join(args.out_dir, "knn_residual_results.csv")
    fieldnames = list(rows_sorted[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    # Save best predictions for follow-up diagnostics.
    if best_pred is not None:
        np.savez_compressed(
            os.path.join(args.out_dir, "knn_best_predictions_test.npz"),
            test_idx=test_idx.astype(np.int64),
            train_idx=train_idx.astype(np.int64),
            val_idx=val_idx.astype(np.int64),
            pred_residual=best_pred.astype(np.float32),
            true_residual=Y[test_idx].astype(np.float32),
            neighbor_dist=best_neighbor_dist.astype(np.float32),
            mode=str(best["mode"]),
            k=int(best["k"]),
            alpha=float(best["alpha"]),
            gate_vr=float(args.gate_vr),
        )

    # Category breakdown for best.
    cat_test = np.asarray(raw["category_id"], dtype=np.int64)[test_idx]
    vr_test = np.asarray(raw["v_rad_norm"], dtype=np.float64)[test_idx]
    event_like = (
        (vr_test > -1.30) & (vr_test < -0.90)
        & (np.asarray(raw["v_tan_norm"], dtype=np.float64)[test_idx] > 0.70)
        & (np.asarray(raw["v_tan_norm"], dtype=np.float64)[test_idx] < 1.10)
        & (np.asarray(raw["r_pair"], dtype=np.float64)[test_idx] > 0.10)
        & (np.asarray(raw["r_pair"], dtype=np.float64)[test_idx] < 0.145)
    )

    summary_path = os.path.join(args.out_dir, "knn_residual_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        def w(s=""):
            f.write(s + "\n")
            print(s)

        w("Compact-kNN residual surrogate diagnostic v1")
        w("=" * 96)
        w(f"data              : {args.data}")
        w(f"samples           : {n}")
        w(f"train/val/test    : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
        w(f"feature modes     : {', '.join(modes)}")
        w(f"k values          : {k_list}")
        w(f"alpha values      : {alpha_list}")
        w(f"gate              : v_rad_norm < {args.gate_vr:.4f}")
        w(f"unsafe factor     : {args.unsafe_factor:.2f}x baseline error")
        w("")
        w("Dataset/test distribution")
        w("-" * 96)
        w(f"r_pair test min/med/max     : {np.min(np.asarray(raw['r_pair'])[test_idx]):.6e} / {np.median(np.asarray(raw['r_pair'])[test_idx]):.6e} / {np.max(np.asarray(raw['r_pair'])[test_idx]):.6e}")
        w(f"v_rad_norm test min/med/max : {np.min(vr_test):+.4f} / {np.median(vr_test):+.4f} / {np.max(vr_test):+.4f}")
        for c in sorted(set(cat_test.tolist())):
            w(f"{CATEGORY_NAMES.get(int(c), 'category_'+str(int(c))):16s}: {int(np.sum(cat_test == c))}")
        w(f"event_like test samples     : {int(np.sum(event_like))}")
        w("")
        w("noNN baseline on test")
        w("-" * 96)
        w(f"pos_med={baseline['base_pos_med']:.6e} pos_p95={baseline['base_pos_p95']:.6e} | vel_med={baseline['base_vel_med']:.6e} vel_p95={baseline['base_vel_p95']:.6e}")
        w("")
        w("Top 15 kNN configurations by score")
        w("-" * 96)
        w(f"{'rank':>4s} {'mode':>10s} {'k':>4s} {'alpha':>6s} {'gate':>10s} {'used':>5s} {'pos_med':>10s} {'vel_med':>10s} {'posUns':>8s} {'velUns':>8s} {'anyUns':>8s}")
        w("  " + "-" * 94)
        for rank, r in enumerate(rows_sorted[:15], 1):
            w(f"{rank:4d} {str(r['mode']):>10s} {int(r['k']):4d} {float(r['alpha']):6.2f} {str(r['gate']):>10s} {int(r['used']):5d} "
              f"{float(r['method_pos_med']):10.3e} {float(r['method_vel_med']):10.3e} "
              f"{format_pct(float(r['pos_unsafe_frac'])):>8s} {format_pct(float(r['vel_unsafe_frac'])):>8s} {format_pct(float(r['any_unsafe_frac'])):>8s}")
        w("")
        if best is not None:
            br = best["row"]
            w("Selected best gated candidate")
            w("-" * 96)
            w(f"mode/k/alpha/gate : {best['mode']} / {best['k']} / {best['alpha']:.2f} / {br['gate']}")
            w(f"used              : {int(br['used'])}/{len(test_idx)} ({100*float(br['use_frac']):.2f}%)")
            w(f"pos_med           : {baseline['base_pos_med']:.6e} -> {float(br['method_pos_med']):.6e} ({float(br['pos_impr_med_pct']):+.2f}%)")
            w(f"vel_med           : {baseline['base_vel_med']:.6e} -> {float(br['method_vel_med']):.6e} ({float(br['vel_impr_med_pct']):+.2f}%)")
            w(f"pos_p95           : {baseline['base_pos_p95']:.6e} -> {float(br['method_pos_p95']):.6e}")
            w(f"vel_p95           : {baseline['base_vel_p95']:.6e} -> {float(br['method_vel_p95']):.6e}")
            w(f"unsafe pos/vel/any: {format_pct(float(br['pos_unsafe_frac']))} / {format_pct(float(br['vel_unsafe_frac']))} / {format_pct(float(br['any_unsafe_frac']))}")
            w(f"both_success      : {format_pct(float(br['both_success_frac']))}")
            w("")
            w("Best candidate category breakdown")
            w("-" * 96)
            for c in sorted(set(cat_test.tolist())):
                mask = cat_test == c
                if not np.any(mask):
                    continue
                sub_idx = test_idx[mask]
                sub_pred = best_pred[mask]
                mm = evaluate_prediction(raw, sub_idx, sub_pred, unsafe_factor=args.unsafe_factor)
                w(f"{CATEGORY_NAMES.get(int(c), 'category_'+str(int(c))):16s} n={int(mask.sum()):4d} | "
                  f"pos {mm['base_pos_med']:.3e}->{mm['method_pos_med']:.3e} ({mm['pos_impr_med_pct']:+6.2f}%) | "
                  f"vel {mm['base_vel_med']:.3e}->{mm['method_vel_med']:.3e} ({mm['vel_impr_med_pct']:+6.2f}%) | "
                  f"anyUns={format_pct(mm['any_unsafe_frac'])}")
            if np.any(event_like):
                sub_idx = test_idx[event_like]
                sub_pred = best_pred[event_like]
                mm = evaluate_prediction(raw, sub_idx, sub_pred, unsafe_factor=args.unsafe_factor)
                w(f"event_like       n={int(np.sum(event_like)):4d} | "
                  f"pos {mm['base_pos_med']:.3e}->{mm['method_pos_med']:.3e} ({mm['pos_impr_med_pct']:+6.2f}%) | "
                  f"vel {mm['base_vel_med']:.3e}->{mm['method_vel_med']:.3e} ({mm['vel_impr_med_pct']:+6.2f}%) | "
                  f"anyUns={format_pct(mm['any_unsafe_frac'])}")
            w("")
            improves = float(br["method_pos_med"]) < baseline["base_pos_med"] and float(br["method_vel_med"]) < baseline["base_vel_med"]
            safe = float(br["any_unsafe_frac"]) <= 0.10
            w("Decision")
            w("-" * 96)
            w(f"Best kNN candidate improves both medians? : {improves}")
            w(f"Best kNN candidate any unsafe <= 10%?     : {safe}")
            if improves and safe:
                w("VERDICT: PROCEED TO EVENT REPLAY: local kNN residual direction is promising enough to test on the exact rollout event.")
            elif improves:
                w("VERDICT: PARTIAL: kNN improves medians but is not safe enough. Tighten alpha/gate before event replay.")
            else:
                w("VERDICT: STOP/REASSESS: compact-neighbor residual averaging does not beat noNN reliably on held-out windows.")
        w("")
        w("Files written")
        w("-" * 96)
        w(csv_path)
        w(os.path.join(args.out_dir, "knn_best_predictions_test.npz"))


if __name__ == "__main__":
    main()
