"""
pair_eval_v4_zone3_encounter_timing_audit.py

Purpose
-------
Small follow-up diagnostic for Phase 2 Zone 3 results.

It uses the current optimized frozen-c evaluator as the source of truth:
    pair_eval_v4_zone3_frozen_timeavg_3way.py

Question answered:
    Why does dt=0.04 get worse with v4 Zone 3 NN while dt=0.08 improves?

For dt=0.04 and dt=0.08, it compares:
    1. IAS15 reference
    2. revised no-Zone-3-NN baseline
    3. frozen v4 Zone 3 NN

It reports:
    - global minimum pair distance and timing for each method
    - closest pair involved
    - Zone 3 sampled-time counts
    - timing shifts of closest encounter
    - RMS error at the closest encounter and at +1 yr / +5 yr
    - plots of minimum distance and RMS error vs time

Run from:
    C:\Aarush\Physics\training\encounter_training

Command:
    python -B pair_eval_v4_zone3_encounter_timing_audit.py --model_path pair_correction_nn_v4_bounded.pt
"""

import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact current optimized frozen evaluator implementation.
try:
    import pair_eval_v4_zone3_frozen_timeavg_3way as base
except ImportError as e:
    raise ImportError(
        "Could not import pair_eval_v4_zone3_frozen_timeavg_3way.py. "
        "Save this audit file in the same folder as that evaluator."
    ) from e


PAIR_NAMES = ["0-1", "0-2", "1-2"]
PAIR_I = np.array([0, 0, 1], dtype=np.int64)
PAIR_J = np.array([1, 2, 2], dtype=np.int64)


def ic1_state() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IC1 default, matching the evaluator/paper code."""
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(axis=0) / M
    v0 = v0 - (m[:, None] * v0).sum(axis=0) / M
    return x0, v0, m


def pair_distance_series(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        dists: shape (n_times, 3), pair distances [0-1,0-2,1-2]
        r_min: shape (n_times,), minimum pair distance
        p_min: shape (n_times,), pair index achieving minimum
    """
    rij = pos[:, PAIR_J, :] - pos[:, PAIR_I, :]
    dists = np.sqrt(np.sum(rij * rij, axis=-1) + 1e-30)
    p_min = np.argmin(dists, axis=1)
    r_min = dists[np.arange(len(pos)), p_min]
    return dists, r_min, p_min


def nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - float(t))))


def summarize_min_distance(times: np.ndarray, pos: np.ndarray) -> Dict[str, float]:
    dists, r_min, p_min = pair_distance_series(pos)
    k = int(np.argmin(r_min))
    z3_mask = (r_min >= 0.05) & (r_min < 0.15)
    z2_mask = r_min < 0.05
    return {
        "min_time": float(times[k]),
        "min_r": float(r_min[k]),
        "min_pair_idx": int(p_min[k]),
        "zone3_sample_count": int(np.sum(z3_mask)),
        "zone2_sample_count": int(np.sum(z2_mask)),
        "min_pair_name": PAIR_NAMES[int(p_min[k])],
    }


def value_at_offsets(times: np.ndarray, series: np.ndarray, t0: float, offsets: List[float]) -> Dict[str, float]:
    out = {}
    for off in offsets:
        target = float(t0) + float(off)
        if target < times[0] or target > times[-1]:
            out[f"plus_{off:g}yr"] = float("nan")
        else:
            out[f"plus_{off:g}yr"] = float(series[nearest_index(times, target)])
    return out


def first_zone3_interval(times: np.ndarray, r_min: np.ndarray) -> Tuple[float, float, int]:
    """Return first continuous sampled interval where r_min is in Zone 3."""
    z = (r_min >= 0.05) & (r_min < 0.15)
    idx = np.where(z)[0]
    if len(idx) == 0:
        return float("nan"), float("nan"), 0
    # First continuous block.
    start = int(idx[0])
    end = start
    for k in idx[1:]:
        if int(k) == end + 1:
            end = int(k)
        else:
            break
    return float(times[start]), float(times[end]), int(end - start + 1)


def plot_min_distance(times, r_ias, r_no, r_v4, dt, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(times, r_ias, lw=1.0, label="IAS15")
    ax.plot(times, r_no, lw=1.0, label="revised no Zone 3 NN")
    ax.plot(times, r_v4, lw=1.0, label="v4 Zone 3 NN")
    ax.axhspan(0.05, 0.15, alpha=0.12, label="Zone 3 band")
    ax.axhline(0.05, ls="--", lw=0.8)
    ax.axhline(0.15, ls="--", lw=0.8)
    for r, name in [(r_no, "noNN min"), (r_v4, "v4 min")]:
        k = int(np.argmin(r))
        ax.axvline(times[k], lw=0.8, alpha=0.6)
        ax.annotate(name, xy=(times[k], r[k]), xytext=(5, 8), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Minimum pair distance r_min (AU, log)")
    ax.set_title(f"Encounter timing audit: minimum distance, dt={dt:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_error(times, err_no, err_v4, r_no, r_v4, dt, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(times, err_no, lw=1.0, label="no Zone 3 NN vs IAS15")
    ax.plot(times, err_v4, lw=1.0, label="v4 Zone 3 NN vs IAS15")
    for r, name in [(r_no, "noNN min"), (r_v4, "v4 min")]:
        k = int(np.argmin(r))
        ax.axvline(times[k], lw=0.8, alpha=0.6)
        ax.annotate(name, xy=(times[k], max(err_no[k], err_v4[k])), xytext=(5, 8), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS separation from IAS15 (AU, log)")
    ax.set_title(f"Encounter timing audit: RMS error, dt={dt:g}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_one_dt(dt: float, args, model, cfg, times_ref, pos_ref, vel_ref, x0, v0, m, out_dir) -> Dict[str, object]:
    print(f"[timing-audit] Running no-Zone-3-NN dt={dt:g}...")
    t_no, p_no, v_no, perf_no = base.simulate_leapfrog_v4_zone3(
        x0, v0, m, model, cfg, dt, args.T, args.n_samples,
        use_zone3_nn=False,
        label="zone3_no_nn",
    )

    print(f"[timing-audit] Running v4 Zone 3 NN dt={dt:g}...")
    t_v4, p_v4, v_v4, perf_v4 = base.simulate_leapfrog_v4_zone3(
        x0, v0, m, model, cfg, dt, args.T, args.n_samples,
        use_zone3_nn=True,
        label="v4_zone3_nn",
    )

    # All methods use the same sampled time grid by construction.
    times = times_ref
    d_ias, r_ias, pair_ias = pair_distance_series(pos_ref)
    d_no, r_no, pair_no = pair_distance_series(p_no)
    d_v4, r_v4, pair_v4 = pair_distance_series(p_v4)

    err_no = base.rms_sep(p_no, pos_ref)
    err_v4 = base.rms_sep(p_v4, pos_ref)

    s_ias = summarize_min_distance(times, pos_ref)
    s_no = summarize_min_distance(times, p_no)
    s_v4 = summarize_min_distance(times, p_v4)

    z3_ias = first_zone3_interval(times, r_ias)
    z3_no = first_zone3_interval(times, r_no)
    z3_v4 = first_zone3_interval(times, r_v4)

    # Use the v4 closest-encounter time as one anchor and noNN closest time as another.
    err_offsets_from_v4_min = {
        "noNN": value_at_offsets(times, err_no, s_v4["min_time"], [0.0, 0.2, 1.0, 5.0]),
        "v4": value_at_offsets(times, err_v4, s_v4["min_time"], [0.0, 0.2, 1.0, 5.0]),
    }
    err_offsets_from_no_min = {
        "noNN": value_at_offsets(times, err_no, s_no["min_time"], [0.0, 0.2, 1.0, 5.0]),
        "v4": value_at_offsets(times, err_v4, s_no["min_time"], [0.0, 0.2, 1.0, 5.0]),
    }

    dt_token = f"{dt:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    plot_min_distance(times, r_ias, r_no, r_v4, dt, os.path.join(out_dir, f"min_distance_dt{dt_token}.png"))
    plot_error(times, err_no, err_v4, r_no, r_v4, dt, os.path.join(out_dir, f"error_vs_time_dt{dt_token}.png"))

    # Save compact sampled data for later if needed.
    np.savetxt(
        os.path.join(out_dir, f"min_distance_error_dt{dt_token}.csv"),
        np.column_stack([times, r_ias, r_no, r_v4, pair_ias, pair_no, pair_v4, err_no, err_v4]),
        delimiter=",",
        header="time,rmin_ias15,rmin_noNN,rmin_v4,pair_ias15,pair_noNN,pair_v4,err_noNN,err_v4",
        comments="",
    )

    return {
        "dt": dt,
        "perf_no": perf_no,
        "perf_v4": perf_v4,
        "err_no": err_no,
        "err_v4": err_v4,
        "time_avg_no": float(np.sqrt(np.mean(err_no ** 2))),
        "time_avg_v4": float(np.sqrt(np.mean(err_v4 ** 2))),
        "final_no": float(err_no[-1]),
        "final_v4": float(err_v4[-1]),
        "ias_min": s_ias,
        "no_min": s_no,
        "v4_min": s_v4,
        "z3_ias": z3_ias,
        "z3_no": z3_no,
        "z3_v4": z3_v4,
        "err_offsets_from_v4_min": err_offsets_from_v4_min,
        "err_offsets_from_no_min": err_offsets_from_no_min,
    }


def _fmt_float(x: float, fmt: str = ".6e") -> str:
    try:
        if not np.isfinite(x):
            return "nan"
        return format(float(x), fmt)
    except Exception:
        return "nan"


def write_summary(path: str, rows: List[Dict[str, object]]):
    lines = []
    lines.append("Encounter timing audit: dt=0.04 failure vs dt=0.08 improvement")
    lines.append("=" * 86)
    lines.append("Method: revised Zone 2 direct Newtonian + sub-stepping; compare no-Zone-3-NN vs frozen v4 Zone 3 NN")
    lines.append("")
    lines.append("Main performance summary")
    lines.append("dt\tnoNN_time_avg\tv4_time_avg\tv4_minus_noNN\tnoNN_final\tv4_final\tnoNN_min_t\tv4_min_t\tnoNN_min_r\tv4_min_r\tnoNN_pair\tv4_pair")
    for r in rows:
        no_min = r["no_min"]
        v4_min = r["v4_min"]
        lines.append(
            f"{r['dt']:g}\t{r['time_avg_no']:.6e}\t{r['time_avg_v4']:.6e}\t{(r['time_avg_v4']-r['time_avg_no']):+.6e}\t"
            f"{r['final_no']:.6e}\t{r['final_v4']:.6e}\t"
            f"{no_min['min_time']:.6f}\t{v4_min['min_time']:.6f}\t"
            f"{no_min['min_r']:.6e}\t{v4_min['min_r']:.6e}\t"
            f"{no_min['min_pair_name']}\t{v4_min['min_pair_name']}"
        )
    lines.append("")

    for r in rows:
        dt = r["dt"]
        lines.append("-" * 86)
        lines.append(f"dt = {dt:g}")
        lines.append("-" * 86)
        lines.append(f"noNN: time_avg_rms={r['time_avg_no']:.6e}, final_err={r['final_no']:.6e}")
        lines.append(f"v4  : time_avg_rms={r['time_avg_v4']:.6e}, final_err={r['final_v4']:.6e}")
        lines.append(f"v4_minus_noNN_time_avg = {(r['time_avg_v4'] - r['time_avg_no']):+.6e}")
        lines.append("")
        for name, s in [("IAS15", r["ias_min"]), ("noNN", r["no_min"]), ("v4", r["v4_min"] )]:
            lines.append(
                f"{name:5s} global min: t={s['min_time']:.6f} yr, r={s['min_r']:.6e} AU, "
                f"pair={s['min_pair_name']}, sampled Zone2 count={s['zone2_sample_count']}, sampled Zone3 count={s['zone3_sample_count']}"
            )
        lines.append("")
        lines.append(
            f"Timing shifts: v4_min - noNN_min = {(r['v4_min']['min_time'] - r['no_min']['min_time']):+.6f} yr; "
            f"noNN_min - IAS15_min = {(r['no_min']['min_time'] - r['ias_min']['min_time']):+.6f} yr; "
            f"v4_min - IAS15_min = {(r['v4_min']['min_time'] - r['ias_min']['min_time']):+.6f} yr"
        )
        lines.append("")
        lines.append("First sampled Zone 3 interval [start, end, count]:")
        lines.append(f"  IAS15: {r['z3_ias'][0]:.6f} to {r['z3_ias'][1]:.6f}, count={r['z3_ias'][2]}")
        lines.append(f"  noNN : {r['z3_no'][0]:.6f} to {r['z3_no'][1]:.6f}, count={r['z3_no'][2]}")
        lines.append(f"  v4   : {r['z3_v4'][0]:.6f} to {r['z3_v4'][1]:.6f}, count={r['z3_v4'][2]}")
        lines.append("")
        lines.append("RMS errors around v4 global closest-encounter time:")
        lines.append("offset\tnoNN_err\tv4_err\tv4_minus_noNN")
        no_offsets = r["err_offsets_from_v4_min"]["noNN"]
        v4_offsets = r["err_offsets_from_v4_min"]["v4"]
        for key in ["plus_0yr", "plus_0.2yr", "plus_1yr", "plus_5yr"]:
            lines.append(f"{key}\t{_fmt_float(no_offsets[key])}\t{_fmt_float(v4_offsets[key])}\t{_fmt_float(v4_offsets[key]-no_offsets[key], '+.6e')}")
        lines.append("")
        lines.append("RMS errors around noNN global closest-encounter time:")
        lines.append("offset\tnoNN_err\tv4_err\tv4_minus_noNN")
        no_offsets = r["err_offsets_from_no_min"]["noNN"]
        v4_offsets = r["err_offsets_from_no_min"]["v4"]
        for key in ["plus_0yr", "plus_0.2yr", "plus_1yr", "plus_5yr"]:
            lines.append(f"{key}\t{_fmt_float(no_offsets[key])}\t{_fmt_float(v4_offsets[key])}\t{_fmt_float(v4_offsets[key]-no_offsets[key], '+.6e')}")
        lines.append("")
        lines.append("Performance counters:")
        pv4 = r["perf_v4"]
        pno = r["perf_no"]
        lines.append(f"  v4 zone3_nn_frac={pv4.get('zone3_nn_frac', float('nan')):.6f}, gate_frac={pv4.get('zone3_gate_fallback_frac', float('nan')):.6f}, strong_frac={pv4.get('strong_approach_zone3_frac', float('nan')):.6f}")
        lines.append(f"  noNN zone3_no_nn_frac={pno.get('zone3_no_nn_frac', float('nan')):.6f}")
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("- If v4 and noNN have similar closest-encounter timing but different post-encounter error, the NN changes the branch after the same encounter.")
    lines.append("- If v4 shifts closest-encounter timing, the NN changes the phase/geometry of the encounter itself.")
    lines.append("- Use the PNG plots to see whether the error gap appears suddenly near the encounter or grows later.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pair_correction_nn_v4_bounded.pt")
    parser.add_argument("--out_dir", default="encounter_timing_audit_out")
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--dts", default="0.04,0.08")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = base.HybridConfig()

    model_path = args.model_path
    if not model_path.endswith(".pt") and os.path.exists(model_path + ".pt"):
        model_path += ".pt"
    model = base.load_v4_model(model_path)

    x0, v0, m = ic1_state()
    dts = [float(x.strip()) for x in args.dts.split(",") if x.strip()]

    print("[timing-audit] Running IAS15 reference...")
    times_ref, pos_ref, vel_ref, perf_ref = base.simulate_rebound_ias15(
        x0, v0, m, cfg.G, float(args.T), int(args.n_samples)
    )
    print(f"[timing-audit] IAS15 time={perf_ref['total_time_sec']:.3f}s")

    rows = []
    for dt in dts:
        rows.append(run_one_dt(dt, args, model, cfg, times_ref, pos_ref, vel_ref, x0, v0, m, args.out_dir))

    summary_path = os.path.join(args.out_dir, "encounter_timing_summary.txt")
    write_summary(summary_path, rows)
    print(f"[timing-audit] wrote {args.out_dir}")
    print(f"[timing-audit] summary: {summary_path}")
    print("[timing-audit] done.")


if __name__ == "__main__":
    main()
