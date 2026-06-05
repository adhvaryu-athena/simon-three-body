"""
evaluate_encounter_surrogate_gate_v1.py

Evaluate a simple deployment gate for the encounter-level surrogate.

Purpose
-------
The first v1 encounter surrogate improved strong-approach held-out cases, but
hurt weak/side and receding cases. This script evaluates a deployment rule:

    if v_rad_norm < threshold: use NN-corrected encounter-exit state
    else:                      use revised noNN encounter-exit state

This is an offline held-out evaluation script. It does not retrain the model.
It uses:
  1. the original encounter surrogate dataset .npz
  2. the saved held-out predictions .npz from train_encounter_surrogate_v1.py

Typical run:
    python -B evaluate_encounter_surrogate_gate_v1.py ^
        --data encounter_surrogate_v1_proto.npz ^
        --pred encounter_surrogate_v1_proto_predictions_test.npz ^
        --out encounter_surrogate_v1_gated_eval.txt ^
        --vr-thresh -0.6

Outputs:
    encounter_surrogate_v1_gated_eval.txt
    encounter_surrogate_v1_gated_eval_predictions.npz

Author: generated for Aarush's SIMON encounter-surrogate prototype.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np


CATEGORY_NAMES = {
    0: "strong_approach",
    1: "weak_side",
    2: "recede",
    3: "broad",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate gated deployment of encounter surrogate predictions.")
    ap.add_argument("--data", default="encounter_surrogate_v1_proto.npz",
                    help="Original encounter-surrogate dataset used for training.")
    ap.add_argument("--pred", default="encounter_surrogate_v1_proto_predictions_test.npz",
                    help="Saved held-out prediction file from train_encounter_surrogate_v1.py.")
    ap.add_argument("--out", default="encounter_surrogate_v1_gated_eval.txt",
                    help="Output text report.")
    ap.add_argument("--vr-thresh", type=float, default=-0.6,
                    help="Use NN correction only when v_rad_norm < this threshold.")
    ap.add_argument("--unsafe-factor", type=float, default=2.0,
                    help="Unsafe if corrected/gated error is more than this factor times noNN error.")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Small epsilon for ratio calculations.")
    return ap.parse_args()


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _rms_state(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample RMS over bodies and coordinates for arrays (n,3,3)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.mean(d * d, axis=(1, 2)))


def _metrics(base_err: np.ndarray, method_err: np.ndarray, eps: float, unsafe_factor: float) -> Dict[str, float]:
    base_err = np.asarray(base_err, dtype=np.float64)
    method_err = np.asarray(method_err, dtype=np.float64)
    ratio = method_err / (base_err + eps)
    improvement = 100.0 * (base_err - method_err) / (base_err + eps)
    return {
        "base_mean": float(np.mean(base_err)),
        "method_mean": float(np.mean(method_err)),
        "base_median": float(np.median(base_err)),
        "method_median": float(np.median(method_err)),
        "base_p95": float(np.percentile(base_err, 95)),
        "method_p95": float(np.percentile(method_err, 95)),
        "improvement_mean_pct": float(np.mean(improvement)),
        "improvement_median_pct": float(np.median(improvement)),
        "success_frac": float(np.mean(method_err < base_err)),
        "unsafe_frac": float(np.mean(ratio > unsafe_factor)),
        "ratio_median": float(np.median(ratio)),
        "ratio_p95": float(np.percentile(ratio, 95)),
    }


def _write_metrics(w, title: str, m: Dict[str, float]):
    w(title)
    w("-" * 80)
    for k, v in m.items():
        if "frac" in k:
            w(f"  {k:28s}: {v:10.2%}")
        elif "pct" in k:
            w(f"  {k:28s}: {v:+10.2f}%")
        else:
            w(f"  {k:28s}: {v: .8e}")
    w()


def _category_table(w, name: str, base_pos, full_pos, gated_pos, base_vel, full_vel, gated_vel, category_id, vr, gate_mask, eps, unsafe_factor):
    w(name)
    w("-" * 80)
    header = (
        f"  {'category':16s} {'n':>5s} {'gate%':>8s} | "
        f"{'base_pos_med':>12s} {'full_pos_med':>12s} {'gated_pos_med':>13s} "
        f"{'gated_impr':>11s} {'gated_succ':>11s} {'gated_unsafe':>13s}"
    )
    w(header)
    w("  " + "-" * (len(header) - 2))
    for cid in sorted(set(int(x) for x in category_id)):
        mask = category_id == cid
        if not np.any(mask):
            continue
        gm = _metrics(base_pos[mask], gated_pos[mask], eps, unsafe_factor)
        label = CATEGORY_NAMES.get(cid, f"cat_{cid}")
        w(
            f"  {label:16s} {int(mask.sum()):5d} {np.mean(gate_mask[mask]):8.1%} | "
            f"{np.median(base_pos[mask]):12.4e} {np.median(full_pos[mask]):12.4e} {np.median(gated_pos[mask]):13.4e} "
            f"{gm['improvement_median_pct']:+10.2f}% {gm['success_frac']:10.1%} {gm['unsafe_frac']:12.1%}"
        )
    # Also add threshold-derived groups, independent of category_id.
    w()
    w("  Threshold-derived groups")
    for label, mask in [
        (f"vr < {np.min([0,0]):.0f} dummy", np.zeros_like(vr, dtype=bool)),
    ]:
        pass
    groups = [
        (f"strong vr<{args_global_vr_thresh:.2f}", vr < args_global_vr_thresh),
        (f"not strong", ~(vr < args_global_vr_thresh)),
        ("approach vr<0", vr < 0.0),
        ("recede/side vr>=0", vr >= 0.0),
    ]
    for label, mask in groups:
        if not np.any(mask):
            continue
        gm = _metrics(base_pos[mask], gated_pos[mask], eps, unsafe_factor)
        w(
            f"  {label:16s} {int(mask.sum()):5d} {np.mean(gate_mask[mask]):8.1%} | "
            f"{np.median(base_pos[mask]):12.4e} {np.median(full_pos[mask]):12.4e} {np.median(gated_pos[mask]):13.4e} "
            f"{gm['improvement_median_pct']:+10.2f}% {gm['success_frac']:10.1%} {gm['unsafe_frac']:12.1%}"
        )
    w()


# Global used only for printing threshold-group labels inside _category_table.
args_global_vr_thresh = -0.6


def main():
    global args_global_vr_thresh
    args = parse_args()
    args_global_vr_thresh = float(args.vr_thresh)

    data = _load_npz(args.data)
    pred = _load_npz(args.pred)

    required_data = ["v_rad_norm", "v_tan_norm", "category_id", "r_pair", "relE_nonn"]
    required_pred = [
        "test_idx", "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final",
        "x_corrected_final", "v_corrected_final", "pred_residual_x", "pred_residual_v",
    ]
    missing_data = [k for k in required_data if k not in data]
    missing_pred = [k for k in required_pred if k not in pred]
    if missing_data:
        raise KeyError(f"Data file missing fields: {missing_data}")
    if missing_pred:
        raise KeyError(f"Prediction file missing fields: {missing_pred}")

    test_idx = pred["test_idx"].astype(np.int64)
    n = len(test_idx)
    if n == 0:
        raise ValueError("Prediction file has empty test_idx")

    vr = np.asarray(data["v_rad_norm"], dtype=np.float64)[test_idx]
    vt = np.asarray(data["v_tan_norm"], dtype=np.float64)[test_idx]
    r_pair = np.asarray(data["r_pair"], dtype=np.float64)[test_idx]
    relE_nonn = np.asarray(data["relE_nonn"], dtype=np.float64)[test_idx]
    category_id = np.asarray(data["category_id"], dtype=np.int32)[test_idx]

    x_ias = np.asarray(pred["x_ias_final"], dtype=np.float64)
    v_ias = np.asarray(pred["v_ias_final"], dtype=np.float64)
    x_nonn = np.asarray(pred["x_nonn_final"], dtype=np.float64)
    v_nonn = np.asarray(pred["v_nonn_final"], dtype=np.float64)
    x_full = np.asarray(pred["x_corrected_final"], dtype=np.float64)
    v_full = np.asarray(pred["v_corrected_final"], dtype=np.float64)

    if x_ias.shape[0] != n:
        raise ValueError(f"Prediction arrays length {x_ias.shape[0]} does not match test_idx length {n}")

    # Deployment gate: only use NN correction for strong approach.
    gate_mask = vr < float(args.vr_thresh)
    x_gated = x_nonn.copy()
    v_gated = v_nonn.copy()
    x_gated[gate_mask] = x_full[gate_mask]
    v_gated[gate_mask] = v_full[gate_mask]

    # Errors relative to IAS15.
    base_pos = _rms_state(x_nonn, x_ias)
    full_pos = _rms_state(x_full, x_ias)
    gated_pos = _rms_state(x_gated, x_ias)
    base_vel = _rms_state(v_nonn, v_ias)
    full_vel = _rms_state(v_full, v_ias)
    gated_vel = _rms_state(v_gated, v_ias)

    m_full_pos = _metrics(base_pos, full_pos, args.eps, args.unsafe_factor)
    m_gated_pos = _metrics(base_pos, gated_pos, args.eps, args.unsafe_factor)
    m_full_vel = _metrics(base_vel, full_vel, args.eps, args.unsafe_factor)
    m_gated_vel = _metrics(base_vel, gated_vel, args.eps, args.unsafe_factor)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        def w(line: str = ""):
            print(line)
            f.write(line + "\n")

        w("Encounter-level surrogate v1 gated evaluation")
        w("=" * 80)
        w(f"data              : {args.data}")
        w(f"predictions       : {args.pred}")
        w(f"test samples      : {n}")
        w(f"gate rule         : use NN iff v_rad_norm < {args.vr_thresh:.4f}")
        w(f"gate used         : {int(gate_mask.sum())}/{n} ({np.mean(gate_mask):.2%})")
        w(f"unsafe factor     : {args.unsafe_factor:.3g}x noNN error")
        w()

        w("Held-out test distribution")
        w("-" * 80)
        w(f"  r_pair      : min={np.min(r_pair):.6e} med={np.median(r_pair):.6e} max={np.max(r_pair):.6e}")
        w(f"  v_rad_norm  : min={np.min(vr):+.4f} med={np.median(vr):+.4f} max={np.max(vr):+.4f}")
        w(f"  v_tan_norm  : min={np.min(vt):.4f} med={np.median(vt):.4f} max={np.max(vt):.4f}")
        w(f"  relE_nonn   : max={np.max(relE_nonn):.3e} med={np.median(relE_nonn):.3e}")
        for cid in sorted(set(int(x) for x in category_id)):
            mask = category_id == cid
            w(f"  {CATEGORY_NAMES.get(cid, f'cat_{cid}'):16s}: {int(mask.sum()):5d}")
        w()

        _write_metrics(w, "FULL NN correction -- position error", m_full_pos)
        _write_metrics(w, "GATED correction -- position error", m_gated_pos)
        _write_metrics(w, "FULL NN correction -- velocity error", m_full_vel)
        _write_metrics(w, "GATED correction -- velocity error", m_gated_vel)

        _category_table(w, "Position category breakdown", base_pos, full_pos, gated_pos,
                        base_vel, full_vel, gated_vel, category_id, vr, gate_mask, args.eps, args.unsafe_factor)

        w("Decision check")
        w("-" * 80)
        full_med_ok = m_full_pos["method_median"] < m_full_pos["base_median"]
        gated_med_ok = m_gated_pos["method_median"] < m_gated_pos["base_median"]
        gated_success_ok = m_gated_pos["success_frac"] > 0.50
        gated_unsafe_ok = m_gated_pos["unsafe_frac"] < 0.20
        w(f"  Full NN improves median position error?   {full_med_ok}")
        w(f"  Gated improves median position error?     {gated_med_ok}")
        w(f"  Gated success fraction > 50%?             {gated_success_ok}")
        w(f"  Gated unsafe fraction < 20%?              {gated_unsafe_ok}")
        w()
        if gated_med_ok and gated_success_ok and gated_unsafe_ok:
            w("VERDICT: PASS as a prototype gated encounter surrogate on held-out windows.")
            w("Use only as an offline window-level result until integrated into a rollout evaluator.")
        elif gated_med_ok:
            w("VERDICT: PARTIAL PASS. The gate improves median error, but safety/success is not yet strong enough.")
            w("Next step: tighten gate or train only on strong-approach cases.")
        else:
            w("VERDICT: STOP/REASSESS. Simple v_rad gate is not sufficient on this held-out split.")
            w("Next step: try stricter gate or strong-approach-only training/evaluation.")

    # Save arrays for future inspection/plotting.
    stem = os.path.splitext(args.out)[0]
    np.savez_compressed(
        stem + "_predictions.npz",
        test_idx=test_idx,
        gate_mask=gate_mask.astype(np.bool_),
        v_rad_norm=vr.astype(np.float32),
        v_tan_norm=vt.astype(np.float32),
        category_id=category_id.astype(np.int32),
        base_pos_error=base_pos.astype(np.float32),
        full_pos_error=full_pos.astype(np.float32),
        gated_pos_error=gated_pos.astype(np.float32),
        base_vel_error=base_vel.astype(np.float32),
        full_vel_error=full_vel.astype(np.float32),
        gated_vel_error=gated_vel.astype(np.float32),
        x_gated_final=x_gated.astype(np.float32),
        v_gated_final=v_gated.astype(np.float32),
    )
    print(f"Saved report: {args.out}")
    print(f"Saved arrays: {stem}_predictions.npz")


if __name__ == "__main__":
    main()
