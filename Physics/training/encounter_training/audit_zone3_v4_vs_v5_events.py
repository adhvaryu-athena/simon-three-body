"""
audit_zone3_v4_vs_v5_events.py

Compare OLD v4 and NEW v5/blended model predictions on the actual Zone-3
firings produced during SIMON rollouts at selected dt values.

Purpose
-------
Before generating more data/training again, this audit checks whether the new
window/blended model changed the real deployed Zone-3 correction pattern:
  - approach c values
  - recede c values
  - min/mean/max correction
  - event timing
  - pair sequence

It imports the existing logged evaluator file:
    pair_eval_v4_zone3_frozen_timeavg_v3_with log.py
because that file already implements the intended revised Zone 2/Zone 3 physics
and supports nn_log_rows output.

Typical run from C:\\Aarush\\Physics\\training\\encounter_training:
    python -B audit_zone3_v4_vs_v5_events.py --old_model pair_correction_nn_v4_bounded.pt --new_model pair_correction_nn_v5_blended_weighted.pt

Outputs
-------
zone3_v4_vs_v5_event_audit_out/
    audit_summary.txt
    events_old_v4_dt0p04.csv
    events_new_v5_dt0p04.csv
    matched_events_dt0p04.csv
    ... same for dt0p08 and dt0p1
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


def dt_token(dt: float) -> str:
    return f"{float(dt):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def load_logged_evaluator(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find logged evaluator file: {path}\n"
            "Run this script from encounter_training, or pass --base_file with the full path."
        )
    spec = importlib.util.spec_from_file_location("logged_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["logged_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def ic1_default():
    """Same IC1 setup used by the 3-way evaluator."""
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M
    return x0, v0, m


def rms_time_avg(eval_mod, pos, ref_pos) -> Tuple[float, float]:
    err = eval_mod.rms_sep(pos, ref_pos)
    return float(err[-1]), float(np.sqrt(np.mean(err ** 2)))


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def pair_label(row: Dict) -> str:
    return f"{int(row['pair_i'])}-{int(row['pair_j'])}"


def summarize_events(rows: List[Dict], model_label: str) -> Dict[str, object]:
    out: Dict[str, object] = {"model": model_label, "n_events": len(rows)}
    if not rows:
        out.update({
            "time_span": "none", "pair_sequence": "none", "pairs": "none",
            "approach_n": 0, "recede_n": 0,
            "c_min": float("nan"), "c_mean": float("nan"), "c_max": float("nan"),
            "approach_c_mean": float("nan"), "approach_c_min": float("nan"), "approach_c_max": float("nan"),
            "recede_c_mean": float("nan"), "recede_c_min": float("nan"), "recede_c_max": float("nan"),
            "fallback_n": 0, "strong_n": 0,
        })
        return out

    times = np.array([float(r["t_yr"]) for r in rows], dtype=float)
    c = np.array([float(r["c_pred"]) for r in rows], dtype=float)
    vr = np.array([float(r["v_rad_norm"]) for r in rows], dtype=float)
    fb = np.array([str(r.get("gate_fallback", False)).lower() == "true" for r in rows])
    strong = np.array([str(r.get("strong_approach", False)).lower() == "true" for r in rows])

    app = vr < 0
    rec = ~app
    pairs = [pair_label(r) for r in rows]
    pair_counts = Counter(pairs)
    # Keep sequence compact but informative.
    seq = ", ".join([f"{p}@{float(r['t_yr']):.3f}" for p, r in zip(pairs, rows[:20])])
    if len(rows) > 20:
        seq += ", ..."

    def stats(mask):
        if not np.any(mask):
            return (float("nan"), float("nan"), float("nan"))
        vals = c[mask]
        return (float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)))

    app_min, app_mean, app_max = stats(app)
    rec_min, rec_mean, rec_max = stats(rec)

    out.update({
        "time_span": f"{times.min():.6f} to {times.max():.6f}",
        "pair_sequence": seq,
        "pairs": ", ".join([f"{k}:{v}" for k, v in sorted(pair_counts.items())]),
        "approach_n": int(np.sum(app)),
        "recede_n": int(np.sum(rec)),
        "c_min": float(np.nanmin(c)),
        "c_mean": float(np.nanmean(c)),
        "c_max": float(np.nanmax(c)),
        "approach_c_min": app_min,
        "approach_c_mean": app_mean,
        "approach_c_max": app_max,
        "recede_c_min": rec_min,
        "recede_c_mean": rec_mean,
        "recede_c_max": rec_max,
        "fallback_n": int(np.sum(fb)),
        "strong_n": int(np.sum(strong)),
    })
    return out


def match_events(old_rows: List[Dict], new_rows: List[Dict], max_time_gap: float) -> List[Dict]:
    """
    Greedy pair/time matching. This is not used for physics; it simply makes it
    easy to see whether c changed on the nearest corresponding event.
    """
    unused = set(range(len(new_rows)))
    matched = []
    for i, old in enumerate(old_rows):
        oi, oj = int(old["pair_i"]), int(old["pair_j"])
        ot = float(old["t_yr"])
        best_j: Optional[int] = None
        best_gap = float("inf")
        for j in list(unused):
            new = new_rows[j]
            if int(new["pair_i"]) != oi or int(new["pair_j"]) != oj:
                continue
            gap = abs(float(new["t_yr"]) - ot)
            if gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j is None or best_gap > max_time_gap:
            matched.append({
                "old_event_index": i,
                "new_event_index": "",
                "pair": f"{oi}-{oj}",
                "old_t_yr": ot,
                "new_t_yr": "",
                "time_gap": "",
                "old_r_au": float(old["r_au"]),
                "new_r_au": "",
                "old_v_rad_norm": float(old["v_rad_norm"]),
                "new_v_rad_norm": "",
                "old_v_tan_norm": float(old["v_tan_norm"]),
                "new_v_tan_norm": "",
                "old_c_pred": float(old["c_pred"]),
                "new_c_pred": "",
                "delta_c_new_minus_old": "",
                "old_phase": "approach" if float(old["v_rad_norm"]) < 0 else "recede/side",
                "new_phase": "",
            })
            continue
        unused.remove(best_j)
        new = new_rows[best_j]
        old_c = float(old["c_pred"])
        new_c = float(new["c_pred"])
        matched.append({
            "old_event_index": i,
            "new_event_index": best_j,
            "pair": f"{oi}-{oj}",
            "old_t_yr": ot,
            "new_t_yr": float(new["t_yr"]),
            "time_gap": best_gap,
            "old_r_au": float(old["r_au"]),
            "new_r_au": float(new["r_au"]),
            "old_v_rad_norm": float(old["v_rad_norm"]),
            "new_v_rad_norm": float(new["v_rad_norm"]),
            "old_v_tan_norm": float(old["v_tan_norm"]),
            "new_v_tan_norm": float(new["v_tan_norm"]),
            "old_c_pred": old_c,
            "new_c_pred": new_c,
            "delta_c_new_minus_old": new_c - old_c,
            "old_phase": "approach" if float(old["v_rad_norm"]) < 0 else "recede/side",
            "new_phase": "approach" if float(new["v_rad_norm"]) < 0 else "recede/side",
        })
    return matched


def fmt_float(x, nd=6):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def add_summary_block(lines: List[str], dt: float, perf: Dict[str, Dict], old_sum: Dict, new_sum: Dict, matched: List[Dict]):
    lines.append("=" * 88)
    lines.append(f"dt={dt:.6f}")
    lines.append("=" * 88)
    lines.append("Rollout metrics vs IAS15:")
    for name in ["no_zone3_nn", "old_v4", "new_v5"]:
        p = perf[name]
        lines.append(
            f"  {name:12s}: final_err={p['final_err']:.6e}  "
            f"time_avg_rms={p['time_avg_rms']:.6e}  "
            f"zone3_frac={p.get('zone3_nn_frac', p.get('zone3_no_nn_frac', float('nan'))):.6f}  "
            f"speedup={p.get('speedup_vs_ias15', float('nan')):.2f}x"
        )
    lines.append("")
    lines.append("Event summary:")
    for s in [old_sum, new_sum]:
        lines.append(f"  {s['model']}: n_events={s['n_events']}  pairs={s['pairs']}  time_span={s['time_span']}")
        lines.append(
            f"    c all      : min/mean/max = {fmt_float(s['c_min'])} / {fmt_float(s['c_mean'])} / {fmt_float(s['c_max'])}"
        )
        lines.append(
            f"    approach  : n={s['approach_n']}  c min/mean/max = "
            f"{fmt_float(s['approach_c_min'])} / {fmt_float(s['approach_c_mean'])} / {fmt_float(s['approach_c_max'])}"
        )
        lines.append(
            f"    recede    : n={s['recede_n']}  c min/mean/max = "
            f"{fmt_float(s['recede_c_min'])} / {fmt_float(s['recede_c_mean'])} / {fmt_float(s['recede_c_max'])}"
        )
        lines.append(f"    fallback_n={s['fallback_n']}  strong_approach_n={s['strong_n']}")
        lines.append(f"    sequence: {s['pair_sequence']}")
    lines.append("")

    # Matched delta summary.
    deltas = []
    app_deltas = []
    rec_deltas = []
    for r in matched:
        if r["delta_c_new_minus_old"] == "":
            continue
        d = float(r["delta_c_new_minus_old"])
        deltas.append(d)
        if r["old_phase"] == "approach":
            app_deltas.append(d)
        else:
            rec_deltas.append(d)
    lines.append(f"Matched events: {len(deltas)} / {len(matched)} old events matched")
    if deltas:
        lines.append(
            f"  Δc(new-old) all     : mean={np.mean(deltas):+.6f}, min={np.min(deltas):+.6f}, max={np.max(deltas):+.6f}"
        )
    if app_deltas:
        lines.append(
            f"  Δc(new-old) approach: mean={np.mean(app_deltas):+.6f}, min={np.min(app_deltas):+.6f}, max={np.max(app_deltas):+.6f}"
        )
    if rec_deltas:
        lines.append(
            f"  Δc(new-old) recede  : mean={np.mean(rec_deltas):+.6f}, min={np.min(rec_deltas):+.6f}, max={np.max(rec_deltas):+.6f}"
        )
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description="Audit old v4 vs new v5 Zone-3 event predictions on same IC/dt values.")
    ap.add_argument("--old_model", default="pair_correction_nn_v4_bounded.pt")
    ap.add_argument("--new_model", default="pair_correction_nn_v5_blended_weighted.pt")
    ap.add_argument("--base_file", default="pair_eval_v4_zone3_frozen_timeavg_v3_with log.py",
                    help="Logged evaluator file to import.")
    ap.add_argument("--dts", default="0.04,0.08,0.10")
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--out_dir", default="zone3_v4_vs_v5_event_audit_out")
    ap.add_argument("--match_frac_dt", type=float, default=0.60,
                    help="Max time gap for event matching = match_frac_dt * dt")
    args = ap.parse_args()

    eval_mod = load_logged_evaluator(args.base_file)
    cfg = eval_mod.HybridConfig()
    x0, v0, m = ic1_default()
    dts = [float(x.strip()) for x in args.dts.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 88)
    print("ZONE 3 OLD-v4 vs NEW-v5 EVENT AUDIT")
    print(f"  old_model : {args.old_model}")
    print(f"  new_model : {args.new_model}")
    print(f"  base_file : {args.base_file}")
    print(f"  dts       : {dts}")
    print(f"  out_dir   : {args.out_dir}")
    print("=" * 88)

    old_model = eval_mod.load_v4_model(args.old_model)
    new_model = eval_mod.load_v4_model(args.new_model)

    print("[audit] Running IAS15 reference once...")
    t_ref, p_ref, v_ref, perf_ref = eval_mod.simulate_rebound_ias15(x0, v0, m, cfg.G, args.T, args.n_samples)
    ias15_time = float(perf_ref["total_time_sec"])
    print(f"[audit] IAS15 time={ias15_time:.6f}s")

    summary_lines: List[str] = []
    summary_lines.append("ZONE 3 OLD-v4 vs NEW-v5 EVENT AUDIT")
    summary_lines.append(f"old_model: {args.old_model}")
    summary_lines.append(f"new_model: {args.new_model}")
    summary_lines.append(f"base_file: {args.base_file}")
    summary_lines.append(f"T_years: {args.T}")
    summary_lines.append(f"n_samples: {args.n_samples}")
    summary_lines.append(f"IAS15_time_sec: {ias15_time:.6f}")
    summary_lines.append("")

    for dt in dts:
        tok = dt_token(dt)
        print(f"[audit] dt={dt}: no-Zone-3-NN baseline...")
        _, p_no, _, perf_no = eval_mod.simulate_leapfrog_v4_zone3(
            x0, v0, m, old_model, cfg, dt, args.T, args.n_samples,
            use_zone3_nn=False, label="no_zone3_nn", nn_log_rows=None,
        )
        no_final, no_tavg = rms_time_avg(eval_mod, p_no, p_ref)
        perf_no["final_err"] = no_final
        perf_no["time_avg_rms"] = no_tavg
        perf_no["speedup_vs_ias15"] = ias15_time / max(float(perf_no.get("total_time_sec", np.nan)), 1e-30)

        print(f"[audit] dt={dt}: old v4 model...")
        old_rows: List[Dict] = []
        _, p_old, _, perf_old = eval_mod.simulate_leapfrog_v4_zone3(
            x0, v0, m, old_model, cfg, dt, args.T, args.n_samples,
            use_zone3_nn=True, label="old_v4", nn_log_rows=old_rows,
        )
        old_final, old_tavg = rms_time_avg(eval_mod, p_old, p_ref)
        perf_old["final_err"] = old_final
        perf_old["time_avg_rms"] = old_tavg
        perf_old["speedup_vs_ias15"] = ias15_time / max(float(perf_old.get("total_time_sec", np.nan)), 1e-30)

        print(f"[audit] dt={dt}: new v5 blended model...")
        new_rows: List[Dict] = []
        _, p_new, _, perf_new = eval_mod.simulate_leapfrog_v4_zone3(
            x0, v0, m, new_model, cfg, dt, args.T, args.n_samples,
            use_zone3_nn=True, label="new_v5", nn_log_rows=new_rows,
        )
        new_final, new_tavg = rms_time_avg(eval_mod, p_new, p_ref)
        perf_new["final_err"] = new_final
        perf_new["time_avg_rms"] = new_tavg
        perf_new["speedup_vs_ias15"] = ias15_time / max(float(perf_new.get("total_time_sec", np.nan)), 1e-30)

        write_csv(os.path.join(args.out_dir, f"events_old_v4_dt{tok}.csv"), old_rows)
        write_csv(os.path.join(args.out_dir, f"events_new_v5_dt{tok}.csv"), new_rows)

        max_gap = float(args.match_frac_dt) * float(dt)
        matched = match_events(old_rows, new_rows, max_time_gap=max_gap)
        write_csv(os.path.join(args.out_dir, f"matched_events_dt{tok}.csv"), matched)

        old_sum = summarize_events(old_rows, "old_v4")
        new_sum = summarize_events(new_rows, "new_v5")
        perf = {"no_zone3_nn": perf_no, "old_v4": perf_old, "new_v5": perf_new}
        add_summary_block(summary_lines, dt, perf, old_sum, new_sum, matched)

        print(
            f"[audit] dt={dt}: old_events={len(old_rows)} new_events={len(new_rows)} "
            f"old_rms={old_tavg:.6e} new_rms={new_tavg:.6e} noNN_rms={no_tavg:.6e}"
        )

    summary_path = os.path.join(args.out_dir, "audit_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[audit] wrote {summary_path}")
    print("[audit] done.")


if __name__ == "__main__":
    main()
