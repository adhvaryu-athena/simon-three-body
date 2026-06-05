"""
evaluate_encounter_surrogate_fixed_gate_v2.py

Fixed-gate evaluator for the v2 velocity-safe encounter surrogate.

Purpose
-------
After the gate sweep, the best practical deployment rule was approximately:

    use encounter surrogate iff v_rad_norm < -0.40

This script does not retrain and does not sweep. It produces a clean final
comparison report for exactly one chosen gate:

    noNN baseline vs full NN vs gated NN

It reports position and velocity errors, unsafe fractions, and category
breakdowns. This is the report to use before deciding whether the gated
encounter surrogate is ready to test inside a full rollout.

Typical run from C:\\Aarush\\Physics\\training\\encounter_nn:

    python -B evaluate_encounter_surrogate_fixed_gate_v2.py \
      --data encounter_surrogate_v1_large1.npz \
      --pred encounter_surrogate_v2_velocitysafe_large1_predictions_test.npz \
      --vr-thresh -0.40 \
      --out encounter_surrogate_v2_velocitysafe_fixed_gate_vr_m0p40.txt
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# Helpers
# =============================================================================
def rms_state(a: np.ndarray) -> np.ndarray:
    """RMS over bodies and xyz for arrays shaped (n, 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"Expected residual array with shape (n,3,3), got {a.shape}")
    return np.sqrt(np.mean(a * a, axis=(1, 2)))


def safe_ratio(method: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.asarray(method, dtype=np.float64) / (np.asarray(base, dtype=np.float64) + 1e-30)


def metric_block(base: np.ndarray, method: np.ndarray, unsafe_factor: float) -> Dict[str, float]:
    base = np.asarray(base, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    ratio = safe_ratio(method, base)
    return {
        "base_mean": float(np.mean(base)),
        "method_mean": float(np.mean(method)),
        "base_median": float(np.median(base)),
        "method_median": float(np.median(method)),
        "base_p90": float(np.percentile(base, 90)),
        "method_p90": float(np.percentile(method, 90)),
        "base_p95": float(np.percentile(base, 95)),
        "method_p95": float(np.percentile(method, 95)),
        "base_max": float(np.max(base)),
        "method_max": float(np.max(method)),
        "success_frac": float(np.mean(method < base)),
        "unsafe_frac": float(np.mean(method > unsafe_factor * base)),
        "ratio_median": float(np.median(ratio)),
        "ratio_p90": float(np.percentile(ratio, 90)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "improvement_mean_pct": float(np.mean((1.0 - ratio) * 100.0)),
        "improvement_median_pct": float(np.median((1.0 - ratio) * 100.0)),
    }


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "   nan"
    return f"{100.0 * x:6.2f}%"


def metric_lines(title: str, m: Dict[str, float]) -> List[str]:
    keys = [
        "base_mean", "method_mean", "base_median", "method_median",
        "base_p90", "method_p90", "base_p95", "method_p95",
        "base_max", "method_max", "improvement_mean_pct",
        "improvement_median_pct", "success_frac", "unsafe_frac",
        "ratio_median", "ratio_p90", "ratio_p95",
    ]
    lines = [title, "-" * 96]
    for k in keys:
        v = m[k]
        if "frac" in k:
            s = fmt_pct(v)
        elif "pct" in k:
            s = f"{v:+9.2f}%"
        else:
            s = f"{v: .8e}"
        lines.append(f"  {k:30s}: {s}")
    lines.append("")
    return lines


def get_field(npz, names: List[str], default=None):
    for name in names:
        if name in npz.files:
            return npz[name]
    return default


def category_names(data, test_idx: np.ndarray, vr: np.ndarray) -> np.ndarray:
    cat = get_field(data, ["category_id", "category", "mode_id"], default=None)
    if cat is not None:
        cat = np.asarray(cat)[test_idx]
        out = []
        for i, ci in enumerate(cat):
            ci = int(ci)
            if ci == 0:
                out.append("strong_approach")
            elif ci == 1:
                out.append("weak_side")
            elif ci == 2:
                out.append("recede")
            else:
                # Fallback to velocity-based labels for unknown category IDs.
                if vr[i] < -0.6:
                    out.append("strong_approach")
                elif vr[i] < 0.2:
                    out.append("weak_side")
                else:
                    out.append("recede")
        return np.array(out, dtype=object)

    # Fallback when category_id is not saved.
    out = []
    for x in vr:
        if x < -0.6:
            out.append("strong_approach")
        elif x < 0.2:
            out.append("weak_side")
        else:
            out.append("recede")
    return np.array(out, dtype=object)


def apply_gate(base_err: np.ndarray, corr_err: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return np.where(gate, corr_err, base_err)


def summarize_method(name: str, base_pos, method_pos, base_vel, method_vel, unsafe_factor: float) -> Dict[str, object]:
    mp = metric_block(base_pos, method_pos, unsafe_factor)
    mv = metric_block(base_vel, method_vel, unsafe_factor)
    any_unsafe = float(np.mean((method_pos > unsafe_factor * base_pos) | (method_vel > unsafe_factor * base_vel)))
    both_success = float(np.mean((method_pos < base_pos) & (method_vel < base_vel)))
    either_success = float(np.mean((method_pos < base_pos) | (method_vel < base_vel)))
    return {
        "name": name,
        "pos": mp,
        "vel": mv,
        "any_unsafe": any_unsafe,
        "both_success": both_success,
        "either_success": either_success,
    }


def category_breakdown(cat_names, gate, base_pos, corr_pos, base_vel, corr_vel, unsafe_factor):
    rows = []
    for cname in ["strong_approach", "weak_side", "recede"]:
        cmask = np.asarray(cat_names == cname, dtype=bool)
        if not np.any(cmask):
            continue
        local_gate = gate & cmask
        method_pos = apply_gate(base_pos[cmask], corr_pos[cmask], local_gate[cmask])
        method_vel = apply_gate(base_vel[cmask], corr_vel[cmask], local_gate[cmask])
        mp = metric_block(base_pos[cmask], method_pos, unsafe_factor)
        mv = metric_block(base_vel[cmask], method_vel, unsafe_factor)
        any_unsafe = float(np.mean(
            (method_pos > unsafe_factor * base_pos[cmask]) |
            (method_vel > unsafe_factor * base_vel[cmask])
        ))
        rows.append({
            "category": cname,
            "n": int(np.sum(cmask)),
            "gate_frac": float(np.mean(gate[cmask])),
            "pos_base_med": mp["base_median"],
            "pos_method_med": mp["method_median"],
            "pos_success": mp["success_frac"],
            "pos_unsafe": mp["unsafe_frac"],
            "pos_impr_med_pct": mp["improvement_median_pct"],
            "vel_base_med": mv["base_median"],
            "vel_method_med": mv["method_median"],
            "vel_success": mv["success_frac"],
            "vel_unsafe": mv["unsafe_frac"],
            "vel_impr_med_pct": mv["improvement_median_pct"],
            "any_unsafe": any_unsafe,
        })
    return rows


def token_from_thresh(x: float) -> str:
    # -0.40 -> m0p40, +0.10 -> p0p10
    prefix = "m" if x < 0 else "p"
    return prefix + f"{abs(float(x)):.2f}".replace(".", "p")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="encounter_surrogate_v1_large1.npz")
    ap.add_argument("--pred", default="encounter_surrogate_v2_velocitysafe_large1_predictions_test.npz")
    ap.add_argument("--vr-thresh", type=float, default=-0.40)
    ap.add_argument("--unsafe-factor", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"encounter_surrogate_v2_velocitysafe_fixed_gate_vr_{token_from_thresh(args.vr_thresh)}.txt"

    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)
    if not os.path.exists(args.pred):
        raise FileNotFoundError(args.pred)

    data = np.load(args.data)
    pred = np.load(args.pred)

    required_pred = [
        "test_idx", "base_pos_error", "corr_pos_error", "base_vel_error", "corr_vel_error",
        "pred_residual_x", "pred_residual_v",
    ]
    missing = [k for k in required_pred if k not in pred.files]
    if missing:
        raise KeyError(f"Predictions file missing fields {missing}. Found: {pred.files}")

    test_idx = pred["test_idx"].astype(np.int64)
    n = len(test_idx)

    vr_all = get_field(data, ["v_rad_norm", "vr_norm", "v_rad"], default=None)
    vt_all = get_field(data, ["v_tan_norm", "vt_norm", "v_tan"], default=None)
    if vr_all is None or vt_all is None:
        raise KeyError("Data file must contain v_rad_norm and v_tan_norm")
    vr = np.asarray(vr_all, dtype=np.float64)[test_idx]
    vt = np.asarray(vt_all, dtype=np.float64)[test_idx]
    cats = category_names(data, test_idx, vr)

    base_pos = np.asarray(pred["base_pos_error"], dtype=np.float64)
    corr_pos = np.asarray(pred["corr_pos_error"], dtype=np.float64)
    base_vel = np.asarray(pred["base_vel_error"], dtype=np.float64)
    corr_vel = np.asarray(pred["corr_vel_error"], dtype=np.float64)

    pred_pos_norm = rms_state(pred["pred_residual_x"])
    pred_vel_norm = rms_state(pred["pred_residual_v"])

    gate = vr < float(args.vr_thresh)
    no_gate = np.zeros(n, dtype=bool)
    full_gate = np.ones(n, dtype=bool)

    method_pos_gated = apply_gate(base_pos, corr_pos, gate)
    method_vel_gated = apply_gate(base_vel, corr_vel, gate)

    summaries = [
        summarize_method("noNN baseline", base_pos, apply_gate(base_pos, corr_pos, no_gate), base_vel, apply_gate(base_vel, corr_vel, no_gate), args.unsafe_factor),
        summarize_method("full NN", base_pos, apply_gate(base_pos, corr_pos, full_gate), base_vel, apply_gate(base_vel, corr_vel, full_gate), args.unsafe_factor),
        summarize_method(f"gated NN: v_rad_norm < {args.vr_thresh:+.2f}", base_pos, method_pos_gated, base_vel, method_vel_gated, args.unsafe_factor),
    ]

    lines: List[str] = []
    lines.append("Encounter-level surrogate v2 fixed-gate evaluation")
    lines.append("=" * 96)
    lines.append(f"data              : {args.data}")
    lines.append(f"predictions       : {args.pred}")
    lines.append(f"test samples      : {n}")
    lines.append(f"gate rule         : use NN iff v_rad_norm < {args.vr_thresh:+.4f}")
    lines.append(f"gate used         : {int(np.sum(gate))}/{n} ({100*np.mean(gate):.2f}%)")
    lines.append(f"unsafe factor     : {args.unsafe_factor:g}x baseline error")
    lines.append("")

    lines.append("Held-out test distribution")
    lines.append("-" * 96)
    lines.append(f"  v_rad_norm      : min={vr.min():+.4f} med={np.median(vr):+.4f} max={vr.max():+.4f}")
    lines.append(f"  v_tan_norm      : min={vt.min():.4f} med={np.median(vt):.4f} max={vt.max():.4f}")
    lines.append(f"  pred_pos_norm   : med={np.median(pred_pos_norm):.4e} p90={np.percentile(pred_pos_norm,90):.4e} p95={np.percentile(pred_pos_norm,95):.4e}")
    lines.append(f"  pred_vel_norm   : med={np.median(pred_vel_norm):.4e} p90={np.percentile(pred_vel_norm,90):.4e} p95={np.percentile(pred_vel_norm,95):.4e}")
    for cname in ["strong_approach", "weak_side", "recede"]:
        cm = cats == cname
        lines.append(f"  {cname:16s}: {int(np.sum(cm)):5d} | gate used {int(np.sum(gate & cm)):5d}/{int(np.sum(cm)):5d} ({100*np.mean(gate[cm]) if np.any(cm) else 0:.1f}%)")
    lines.append("")

    lines.append("Compact comparison")
    lines.append("-" * 96)
    lines.append(
        f"{'method':32s} {'pos_med':>10s} {'pos_p95':>10s} {'posUns':>8s} "
        f"{'vel_med':>10s} {'vel_p95':>10s} {'velUns':>8s} {'anyUns':>8s} {'bothSucc':>9s}"
    )
    lines.append("  " + "-" * 94)
    for s in summaries:
        lines.append(
            f"{s['name'][:32]:32s} "
            f"{s['pos']['method_median']:10.3e} {s['pos']['method_p95']:10.3e} {fmt_pct(s['pos']['unsafe_frac']):>8s} "
            f"{s['vel']['method_median']:10.3e} {s['vel']['method_p95']:10.3e} {fmt_pct(s['vel']['unsafe_frac']):>8s} "
            f"{fmt_pct(s['any_unsafe']):>8s} {fmt_pct(s['both_success']):>9s}"
        )
    lines.append("")

    # Full metric blocks for gated method only, plus full/noNN compact blocks are enough.
    lines += metric_lines("GATED correction -- position error", summaries[2]["pos"])
    lines += metric_lines("GATED correction -- velocity error", summaries[2]["vel"])

    lines.append("Gated category breakdown")
    lines.append("-" * 96)
    lines.append(
        f"{'category':16s} {'n':>5s} {'gate%':>8s} | "
        f"{'pos_base':>10s} {'pos_gate':>10s} {'pos_impr':>9s} {'posUns':>8s} | "
        f"{'vel_base':>10s} {'vel_gate':>10s} {'vel_impr':>9s} {'velUns':>8s} {'anyUns':>8s}"
    )
    lines.append("  " + "-" * 94)
    for r in category_breakdown(cats, gate, base_pos, corr_pos, base_vel, corr_vel, args.unsafe_factor):
        lines.append(
            f"{r['category']:16s} {r['n']:5d} {fmt_pct(r['gate_frac']):>8s} | "
            f"{r['pos_base_med']:10.3e} {r['pos_method_med']:10.3e} {r['pos_impr_med_pct']:+8.2f}% {fmt_pct(r['pos_unsafe']):>8s} | "
            f"{r['vel_base_med']:10.3e} {r['vel_method_med']:10.3e} {r['vel_impr_med_pct']:+8.2f}% {fmt_pct(r['vel_unsafe']):>8s} {fmt_pct(r['any_unsafe']):>8s}"
        )
    lines.append("")

    # Decision check.
    no = summaries[0]
    full = summaries[1]
    gated = summaries[2]
    lines.append("Decision check")
    lines.append("-" * 96)
    lines.append(f"  Full NN improves position median vs noNN?    {full['pos']['method_median'] < no['pos']['method_median']}")
    lines.append(f"  Full NN improves velocity median vs noNN?    {full['vel']['method_median'] < no['vel']['method_median']}")
    lines.append(f"  Gate improves position median vs noNN?       {gated['pos']['method_median'] < no['pos']['method_median']}")
    lines.append(f"  Gate improves velocity median vs noNN?       {gated['vel']['method_median'] < no['vel']['method_median']}")
    lines.append(f"  Gate position unsafe <= 5%?                  {gated['pos']['unsafe_frac'] <= 0.05}")
    lines.append(f"  Gate velocity unsafe <= 10%?                 {gated['vel']['unsafe_frac'] <= 0.10}")
    lines.append(f"  Gate any unsafe <= 10%?                      {gated['any_unsafe'] <= 0.10}")
    lines.append("")

    if (
        gated['pos']['method_median'] < no['pos']['method_median'] and
        gated['vel']['method_median'] < no['vel']['method_median'] and
        gated['pos']['unsafe_frac'] <= 0.05 and
        gated['vel']['unsafe_frac'] <= 0.10
    ):
        verdict = "PROCEED: fixed gate improves both medians while keeping unsafe fractions low enough for the next rollout prototype."
    elif gated['pos']['method_median'] < no['pos']['method_median'] and gated['vel']['method_median'] < no['vel']['method_median']:
        verdict = "PARTIAL PASS: fixed gate improves both medians, but safety thresholds still need tightening."
    else:
        verdict = "STOP/REASSESS: fixed gate does not improve both position and velocity medians."
    lines.append("VERDICT: " + verdict)

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
