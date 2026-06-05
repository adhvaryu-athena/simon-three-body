"""
evaluate_encounter_surrogate_gate_sweep_v2.py

Gate-sweep evaluator for the encounter-level surrogate predictions.

Purpose
-------
The v2 velocity-safe model improved median velocity error, but full deployment
still had unsafe velocity cases, especially for receding encounters. This script
uses the saved held-out predictions and sweeps simple deployment gates to find
whether a rule can preserve the NN benefit while reducing position/velocity
unsafe fractions.

It does NOT retrain. It only compares:
    noNN baseline
    full NN correction
    gated NN correction

Typical run:
    python -B evaluate_encounter_surrogate_gate_sweep_v2.py \
      --data encounter_surrogate_v1_large1.npz \
      --pred encounter_surrogate_v2_velocitysafe_large1_predictions_test.npz \
      --out encounter_surrogate_v2_velocitysafe_gate_sweep.txt

Expected inputs
---------------
--data: the full encounter-surrogate dataset npz, containing at least:
    v_rad_norm, v_tan_norm, and optionally r_pair/category_id/relE_nonn

--pred: the predictions npz created by train_encounter_surrogate_v2_velocitysafe,
containing at least:
    test_idx, base_pos_error, corr_pos_error, base_vel_error, corr_vel_error,
    pred_residual_x, pred_residual_v

Gate families tested
--------------------
1. vr_only:
       use NN iff v_rad_norm < threshold
2. vr_plus_residual_cap:
       use NN iff v_rad_norm < threshold AND predicted residual norm <= cap
3. approach_plus_velocity_cap:
       use NN iff v_rad_norm < 0 AND predicted velocity residual norm <= cap
4. strong_plus_weaksafe:
       use NN for strong approach always, and for weaker approach only if
       predicted velocity residual norm <= cap

The residual caps are computed from percentiles of the model's predicted
residual norms on the held-out test set, so the sweep is self-contained.
"""

import argparse
import os
import math
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def rms_state(a: np.ndarray) -> np.ndarray:
    """RMS over bodies and xyz for shape (n,3,3)."""
    return np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2, axis=(1, 2)))


def safe_ratio(method: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.asarray(method, dtype=np.float64) / (np.asarray(base, dtype=np.float64) + 1e-30)


def metrics(base: np.ndarray, method: np.ndarray, unsafe_factor: float) -> Dict[str, float]:
    base = np.asarray(base, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    ratio = safe_ratio(method, base)
    return {
        "base_mean": float(np.mean(base)),
        "method_mean": float(np.mean(method)),
        "base_median": float(np.median(base)),
        "method_median": float(np.median(method)),
        "base_p95": float(np.percentile(base, 95)),
        "method_p95": float(np.percentile(method, 95)),
        "success_frac": float(np.mean(method < base)),
        "unsafe_frac": float(np.mean(method > unsafe_factor * base)),
        "ratio_median": float(np.median(ratio)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        # Median of per-sample percentage improvement, not improvement of medians.
        "improvement_median_pct": float(np.median((1.0 - ratio) * 100.0)),
        "improvement_mean_pct": float(np.mean((1.0 - ratio) * 100.0)),
    }


def used_metrics(base: np.ndarray, method: np.ndarray, mask: np.ndarray, unsafe_factor: float) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "used_success_frac": float("nan"),
            "used_unsafe_frac": float("nan"),
            "used_pos_or_vel_med_ratio": float("nan"),
        }
    ratio = safe_ratio(method[mask], base[mask])
    return {
        "used_success_frac": float(np.mean(method[mask] < base[mask])),
        "used_unsafe_frac": float(np.mean(method[mask] > unsafe_factor * base[mask])),
        "used_med_ratio": float(np.median(ratio)),
    }


def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "   nan"
    return f"{100.0*x:6.1f}%"


def fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6e}"


def line_metric_block(title: str, m: Dict[str, float]) -> List[str]:
    keys = [
        "base_mean", "method_mean", "base_median", "method_median",
        "base_p95", "method_p95", "improvement_mean_pct",
        "improvement_median_pct", "success_frac", "unsafe_frac",
        "ratio_median", "ratio_p95",
    ]
    out = [title, "-" * 88]
    for k in keys:
        v = m[k]
        if "frac" in k:
            s = fmt_pct(v)
        elif "pct" in k:
            s = f"{v:+9.2f}%"
        else:
            s = f"{v: .8e}"
        out.append(f"  {k:30s}: {s}")
    out.append("")
    return out


def get_field(data, names: List[str], default=None):
    for name in names:
        if name in data.files:
            return data[name]
    return default


def category_names_from_data(data, test_idx, vr):
    cat = get_field(data, ["category_id", "category", "mode_id"], default=None)
    if cat is not None:
        cat = np.asarray(cat)[test_idx]
    names = []
    for i in range(len(vr)):
        if cat is not None:
            ci = int(cat[i])
            # Generator convention used in v1 files: 0 strong, 1 weak/side, 2 recede, 3 broad if present.
            if ci == 0:
                names.append("strong_approach")
            elif ci == 1:
                names.append("weak_side")
            elif ci == 2:
                names.append("recede")
            else:
                # fall back to velocity-derived label
                names.append("approach" if vr[i] < 0 else "recede_side")
        else:
            if vr[i] < -0.6:
                names.append("strong_approach")
            elif vr[i] < 0.2:
                names.append("weak_side")
            else:
                names.append("recede")
    return np.array(names, dtype=object)


# -----------------------------------------------------------------------------
# Gate construction
# -----------------------------------------------------------------------------
def build_gate_rows(vr, vt, pred_pos_norm, pred_vel_norm, pred_state_norm) -> List[Tuple[str, np.ndarray]]:
    rows: List[Tuple[str, np.ndarray]] = []
    n = len(vr)
    all_true = np.ones(n, dtype=bool)
    rows.append(("full_nn_all", all_true))

    vr_thresholds = [-0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20]
    for th in vr_thresholds:
        rows.append((f"vr<{th:+.2f}", vr < th))

    # Residual caps from percentiles. These are deployment-feasible because they use only predicted residual size.
    cap_percentiles = [50, 60, 70, 80, 85, 90, 95]
    vel_caps = [(p, float(np.percentile(pred_vel_norm, p))) for p in cap_percentiles]
    pos_caps = [(p, float(np.percentile(pred_pos_norm, p))) for p in cap_percentiles]
    state_caps = [(p, float(np.percentile(pred_state_norm, p))) for p in cap_percentiles]

    # Approach + residual cap variants.
    for th in [-0.60, -0.40, -0.30, -0.20, 0.00]:
        for p, cap in vel_caps:
            rows.append((f"vr<{th:+.2f} & pred_vel<=p{p}", (vr < th) & (pred_vel_norm <= cap)))

    for th in [-0.60, -0.40, -0.30, -0.20, 0.00]:
        for p, cap in state_caps:
            rows.append((f"vr<{th:+.2f} & pred_state<=p{p}", (vr < th) & (pred_state_norm <= cap)))

    # Strong always + weaker approach if predicted residual is modest.
    strong = vr < -0.60
    weak_approach = (vr >= -0.60) & (vr < 0.00)
    for p, cap in vel_caps:
        rows.append((f"strong OR (weak_app & pred_vel<=p{p})", strong | (weak_approach & (pred_vel_norm <= cap))))

    # Exclude receding but allow all approaching if residual looks small.
    for p, cap in vel_caps:
        rows.append((f"approach_only & pred_vel<=p{p}", (vr < 0.00) & (pred_vel_norm <= cap)))

    # Very conservative: strong and not high tangential extremes.
    for vt_hi in [0.9, 1.1, 1.3, 1.6]:
        rows.append((f"vr<-0.60 & vt<={vt_hi:.1f}", (vr < -0.60) & (vt <= vt_hi)))

    # Remove duplicate masks by byte representation.
    seen = set()
    uniq = []
    for name, mask in rows:
        key = mask.astype(np.uint8).tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append((name, mask))
    return uniq


def evaluate_gate(name, mask, base_pos, corr_pos, base_vel, corr_vel, unsafe_factor):
    method_pos = np.where(mask, corr_pos, base_pos)
    method_vel = np.where(mask, corr_vel, base_vel)
    mp = metrics(base_pos, method_pos, unsafe_factor)
    mv = metrics(base_vel, method_vel, unsafe_factor)
    up = used_metrics(base_pos, method_pos, mask, unsafe_factor)
    uv = used_metrics(base_vel, method_vel, mask, unsafe_factor)
    unsafe_any = float(np.mean((method_pos > unsafe_factor * base_pos) | (method_vel > unsafe_factor * base_vel)))

    # Lower score is better. The absolute values are not physical; used only for ranking gates.
    score = (
        mp["method_median"]
        + 0.5 * mv["method_median"]
        + 0.05 * mp["method_p95"]
        + 0.05 * mv["method_p95"]
        + 0.02 * mp["unsafe_frac"]
        + 0.05 * mv["unsafe_frac"]
        + 0.08 * unsafe_any
    )
    return {
        "name": name,
        "used": int(np.sum(mask)),
        "used_frac": float(np.mean(mask)),
        "pos_med": mp["method_median"],
        "pos_p95": mp["method_p95"],
        "pos_success": mp["success_frac"],
        "pos_unsafe": mp["unsafe_frac"],
        "pos_impr_med_pct": mp["improvement_median_pct"],
        "vel_med": mv["method_median"],
        "vel_p95": mv["method_p95"],
        "vel_success": mv["success_frac"],
        "vel_unsafe": mv["unsafe_frac"],
        "vel_impr_med_pct": mv["improvement_median_pct"],
        "unsafe_any": unsafe_any,
        "score": float(score),
        "mask": mask,
        "pos_metrics": mp,
        "vel_metrics": mv,
        "used_pos": up,
        "used_vel": uv,
    }


def category_breakdown(names, mask, base_pos, corr_pos, base_vel, corr_vel, unsafe_factor):
    out = []
    for cname in ["strong_approach", "weak_side", "recede", "approach", "recede_side"]:
        if cname == "approach":
            cmask = np.array([str(x) in ("strong_approach", "weak_side") for x in names], dtype=bool)
        elif cname == "recede_side":
            cmask = np.array([str(x) == "recede" for x in names], dtype=bool)
        else:
            cmask = np.array([str(x) == cname for x in names], dtype=bool)
        if not np.any(cmask):
            continue
        g = mask & cmask
        mp = metrics(base_pos[cmask], np.where(g[cmask], corr_pos[cmask], base_pos[cmask]), unsafe_factor)
        mv = metrics(base_vel[cmask], np.where(g[cmask], corr_vel[cmask], base_vel[cmask]), unsafe_factor)
        out.append((cname, int(cmask.sum()), float(np.mean(mask[cmask])), mp, mv))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="encounter_surrogate_v1_large1.npz")
    ap.add_argument("--pred", default="encounter_surrogate_v2_velocitysafe_large1_predictions_test.npz")
    ap.add_argument("--out", default="encounter_surrogate_v2_velocitysafe_gate_sweep.txt")
    ap.add_argument("--unsafe-factor", type=float, default=2.0)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)
    if not os.path.exists(args.pred):
        raise FileNotFoundError(args.pred)

    data = np.load(args.data)
    pred = np.load(args.pred)

    required_pred = ["test_idx", "base_pos_error", "corr_pos_error", "base_vel_error", "corr_vel_error", "pred_residual_x", "pred_residual_v"]
    missing = [k for k in required_pred if k not in pred.files]
    if missing:
        raise KeyError(f"Predictions file missing required fields: {missing}. Found: {pred.files}")

    test_idx = pred["test_idx"].astype(np.int64)
    n = len(test_idx)

    vr_all = get_field(data, ["v_rad_norm", "vr_norm", "v_rad"], default=None)
    vt_all = get_field(data, ["v_tan_norm", "vt_norm", "v_tan"], default=None)
    if vr_all is None or vt_all is None:
        raise KeyError("Data file must contain v_rad_norm and v_tan_norm")
    vr = np.asarray(vr_all, dtype=np.float64)[test_idx]
    vt = np.asarray(vt_all, dtype=np.float64)[test_idx]
    cat_names = category_names_from_data(data, test_idx, vr)

    base_pos = np.asarray(pred["base_pos_error"], dtype=np.float64)
    corr_pos = np.asarray(pred["corr_pos_error"], dtype=np.float64)
    base_vel = np.asarray(pred["base_vel_error"], dtype=np.float64)
    corr_vel = np.asarray(pred["corr_vel_error"], dtype=np.float64)

    pred_rx = np.asarray(pred["pred_residual_x"], dtype=np.float64)
    pred_rv = np.asarray(pred["pred_residual_v"], dtype=np.float64)
    pred_pos_norm = rms_state(pred_rx)
    pred_vel_norm = rms_state(pred_rv)
    pred_state_norm = np.sqrt(pred_pos_norm ** 2 + pred_vel_norm ** 2)

    gates = build_gate_rows(vr, vt, pred_pos_norm, pred_vel_norm, pred_state_norm)
    rows = [evaluate_gate(name, mask, base_pos, corr_pos, base_vel, corr_vel, args.unsafe_factor) for name, mask in gates]
    rows_sorted = sorted(rows, key=lambda r: r["score"])

    # Useful constrained rankings.
    safe_rows = [r for r in rows_sorted if r["pos_unsafe"] <= 0.05 and r["vel_unsafe"] <= 0.10]
    balanced_rows = [r for r in rows_sorted if r["pos_unsafe"] <= 0.10 and r["vel_unsafe"] <= 0.20 and r["used_frac"] >= 0.20]

    lines = []
    lines.append("Encounter-level surrogate v2 velocity-safe gate sweep")
    lines.append("=" * 96)
    lines.append(f"data              : {args.data}")
    lines.append(f"predictions       : {args.pred}")
    lines.append(f"test samples      : {n}")
    lines.append(f"unsafe factor     : {args.unsafe_factor:g}x baseline error")
    lines.append("")
    lines.append("Held-out test distribution")
    lines.append("-" * 96)
    lines.append(f"  r?              : data file may contain r_pair/r_AU; not required for sweep")
    lines.append(f"  v_rad_norm      : min={vr.min():+.4f} med={np.median(vr):+.4f} max={vr.max():+.4f}")
    lines.append(f"  v_tan_norm      : min={vt.min():.4f} med={np.median(vt):.4f} max={vt.max():.4f}")
    for cname in ["strong_approach", "weak_side", "recede"]:
        lines.append(f"  {cname:16s}: {int(np.sum(cat_names == cname)):5d}")
    lines.append(f"  pred_pos_norm   : med={np.median(pred_pos_norm):.4e} p90={np.percentile(pred_pos_norm,90):.4e} p95={np.percentile(pred_pos_norm,95):.4e}")
    lines.append(f"  pred_vel_norm   : med={np.median(pred_vel_norm):.4e} p90={np.percentile(pred_vel_norm,90):.4e} p95={np.percentile(pred_vel_norm,95):.4e}")
    lines.append("")

    no_mask = np.zeros(n, dtype=bool)
    full_mask = np.ones(n, dtype=bool)
    no_row = evaluate_gate("noNN_baseline", no_mask, base_pos, corr_pos, base_vel, corr_vel, args.unsafe_factor)
    full_row = evaluate_gate("full_NN_all", full_mask, base_pos, corr_pos, base_vel, corr_vel, args.unsafe_factor)

    lines += line_metric_block("noNN baseline -- position error", no_row["pos_metrics"])
    lines += line_metric_block("full NN -- position error", full_row["pos_metrics"])
    lines += line_metric_block("noNN baseline -- velocity error", no_row["vel_metrics"])
    lines += line_metric_block("full NN -- velocity error", full_row["vel_metrics"])

    def add_table(title, table_rows):
        lines.append(title)
        lines.append("-" * 96)
        lines.append(
            f"{'rank':>4s}  {'gate':45s} {'used':>5s} {'use%':>7s} "
            f"{'pos_med':>10s} {'vel_med':>10s} {'posUns':>7s} {'velUns':>7s} {'anyUns':>7s} {'score':>10s}"
        )
        lines.append("  " + "-" * 94)
        for rank, r in enumerate(table_rows[:args.top_k], 1):
            lines.append(
                f"{rank:4d}  {r['name'][:45]:45s} {r['used']:5d} {fmt_pct(r['used_frac']):>7s} "
                f"{r['pos_med']:10.3e} {r['vel_med']:10.3e} {fmt_pct(r['pos_unsafe']):>7s} "
                f"{fmt_pct(r['vel_unsafe']):>7s} {fmt_pct(r['unsafe_any']):>7s} {r['score']:10.4e}"
            )
        lines.append("")

    add_table("Top gates by combined score", rows_sorted)
    add_table("Top gates with pos_unsafe<=5% and vel_unsafe<=10%", safe_rows)
    add_table("Top balanced gates with pos_unsafe<=10%, vel_unsafe<=20%, used>=20%", balanced_rows)

    # Pick recommendation: prefer balanced if available, otherwise safe, otherwise best overall.
    if balanced_rows:
        rec = balanced_rows[0]
        rec_reason = "best balanced gate under pos<=10%, vel<=20%, used>=20%"
    elif safe_rows:
        rec = safe_rows[0]
        rec_reason = "best strict-safe gate under pos<=5%, vel<=10%"
    else:
        rec = rows_sorted[0]
        rec_reason = "best score, but safety constraints were not satisfied"

    lines.append("Recommended gate for next experiment")
    lines.append("-" * 96)
    lines.append(f"  gate       : {rec['name']}")
    lines.append(f"  reason     : {rec_reason}")
    lines.append(f"  used       : {rec['used']}/{n} ({100*rec['used_frac']:.2f}%)")
    lines.append(f"  pos median : {rec['pos_med']:.6e}, unsafe={100*rec['pos_unsafe']:.2f}%")
    lines.append(f"  vel median : {rec['vel_med']:.6e}, unsafe={100*rec['vel_unsafe']:.2f}%")
    lines.append(f"  any unsafe : {100*rec['unsafe_any']:.2f}%")
    lines.append("")

    # Category breakdown for recommended gate.
    lines.append("Recommended gate category breakdown")
    lines.append("-" * 96)
    lines.append(
        f"{'category':16s} {'n':>5s} {'gate%':>7s} | "
        f"{'pos_base_med':>12s} {'pos_gate_med':>12s} {'posUns':>7s} | "
        f"{'vel_base_med':>12s} {'vel_gate_med':>12s} {'velUns':>7s}"
    )
    lines.append("  " + "-" * 94)
    for cname, count, gfrac, mp, mv in category_breakdown(cat_names, rec["mask"], base_pos, corr_pos, base_vel, corr_vel, args.unsafe_factor):
        lines.append(
            f"{cname:16s} {count:5d} {fmt_pct(gfrac):>7s} | "
            f"{mp['base_median']:12.4e} {mp['method_median']:12.4e} {fmt_pct(mp['unsafe_frac']):>7s} | "
            f"{mv['base_median']:12.4e} {mv['method_median']:12.4e} {fmt_pct(mv['unsafe_frac']):>7s}"
        )
    lines.append("")

    # Decision check.
    lines.append("Decision check")
    lines.append("-" * 96)
    lines.append(f"  Full NN has lower position median than noNN?      {full_row['pos_med'] < no_row['pos_med']}")
    lines.append(f"  Full NN has lower velocity median than noNN?      {full_row['vel_med'] < no_row['vel_med']}")
    lines.append(f"  Recommended gate lower position median than noNN? {rec['pos_med'] < no_row['pos_med']}")
    lines.append(f"  Recommended gate lower velocity median than noNN? {rec['vel_med'] < no_row['vel_med']}")
    lines.append(f"  Recommended gate any-unsafe <= 15%?              {rec['unsafe_any'] <= 0.15}")
    lines.append("")

    if rec["pos_med"] < no_row["pos_med"] and rec["vel_med"] < no_row["vel_med"] and rec["unsafe_any"] <= 0.15:
        verdict = "PROCEED: a deployment gate improves both median position and velocity with acceptable unsafe rate."
    elif rec["pos_med"] < no_row["pos_med"] and rec["vel_med"] < no_row["vel_med"]:
        verdict = "PARTIAL PASS: gate improves both medians, but unsafe rate may still need tightening."
    else:
        verdict = "STOP/REASSESS: no simple gate gives a clean position+velocity improvement."
    lines.append("VERDICT: " + verdict)

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
