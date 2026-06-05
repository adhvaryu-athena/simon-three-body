# pair_eval_adaptive_threshold_sweep.py
#
# Sweep the adaptive sub-stepping threshold for SIMON using direct Newtonian
# gravity everywhere. This tests whether the timestep-control idea is better
# than applying a Zone 3 neural force multiplier.
#
# Method tested:
#   - Force law: exact Newtonian pairwise force for all separations.
#   - Adaptive sub-stepping: triggered when min pair distance < adapt_thresh.
#   - No neural network is loaded or used.
#
# Default experiment:
#   IC1 default, T=100 yr, n_samples=5000
#   dt frontier = [0.005, 0.01, 0.02, 0.04, 0.08]
#   threshold sweep = [0.05, 0.08, 0.10, 0.12, 0.15] AU
#
# Main question:
#   Can increasing the adaptive threshold improve long-horizon accuracy enough
#   to justify the extra sub-step cost?
#
# Run:
#   python -B pair_eval_adaptive_threshold_sweep.py
#
# Optional:
#   python -B pair_eval_adaptive_threshold_sweep.py --dt_rep 0.04 --thresholds 0.05 0.08 0.10 0.12 0.15

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found. Install it in your Python environment.")
    raise


# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})


@dataclass
class Config:
    G: float = 1.0
    T: float = 100.0
    n_samples: int = 5000
    max_substeps: int = 16
    ejection_au: float = 50.0


# ── Initial condition: IC1 default, matching the paper scripts ────────────────
def com_center(m: np.ndarray, x0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x = x0 - (m[:, None] * x0).sum(axis=0) / M
    v = v0 - (m[:, None] * v0).sum(axis=0) / M
    return x.astype(np.float64), v.astype(np.float64)


def make_ic1() -> Dict[str, np.ndarray]:
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    x0, v0 = com_center(m, x0, v0)
    return {"name": "IC1_default", "label": "IC1: Default", "m": m, "x0": x0, "v0": v0}


# ── Simulation helpers ────────────────────────────────────────────────────────
def pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ii.append(i)
            jj.append(j)
    return np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64)


def simulate_ias15(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = cfg.G
    for i in range(len(m)):
        sim.add(
            m=float(m[i]),
            x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
            vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]),
        )
    sim.move_to_com()

    times = np.linspace(0.0, cfg.T, cfg.n_samples)
    pos = np.zeros((cfg.n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((cfg.n_samples, len(m), 3), dtype=np.float64)

    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    elapsed = time.perf_counter() - t0
    return times, pos, vel, {"total_time_sec": elapsed}


def simulate_newtonian_adaptive(
    x0: np.ndarray,
    v0: np.ndarray,
    m: np.ndarray,
    cfg: Config,
    dt: float,
    adapt_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Velocity-Verlet/leapfrog with exact Newtonian forces everywhere.
    If min pair distance < adapt_thresh, split the macro step into substeps.
    """
    N = x0.shape[0]
    ii, jj = pair_indices(N)
    P = len(ii)

    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)

    mi = m_f[ii]
    mj = m_f[jj]
    Gmimj = cfg.G * mi * mj
    inv_mi = 1.0 / mi
    inv_mj = 1.0 / mj

    times = np.linspace(0.0, cfg.T, cfg.n_samples)
    pos_out = np.zeros((cfg.n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((cfg.n_samples, N, 3), dtype=np.float64)
    n_steps = int(math.ceil(cfg.T / float(dt)))

    def compute_acc(pos: np.ndarray) -> np.ndarray:
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        F_scalar = Gmimj / (r2 * r + 1e-30)
        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        # Generic accumulation, safe for any N.
        np.add.at(acc, ii, F_vec * inv_mi[:, None])
        np.add.at(acc, jj, -F_vec * inv_mj[:, None])
        return acc

    def min_pair_distance(pos: np.ndarray) -> float:
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return float(np.sqrt(np.min(r2) + 1e-30))

    def substep(x_in: np.ndarray, v_in: np.ndarray, a_in: np.ndarray, sub_dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vh = v_in + 0.5 * sub_dt * a_in
        xn = x_in + sub_dt * vh
        an = compute_acc(xn)
        vn = vh + 0.5 * sub_dt * an
        return xn, vn, an

    a = compute_acc(x)
    si = 0
    nt = times[0]
    t_cur = 0.0
    while si < cfg.n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < cfg.n_samples:
            nt = times[si]

    n_adaptive_macro = 0
    total_substeps = 0
    max_substeps_used = 1
    min_r_seen = float("inf")
    ejected = False

    t0 = time.perf_counter()
    dt_f = float(dt)
    for _ in range(n_steps):
        r_min = min_pair_distance(x)
        min_r_seen = min(min_r_seen, r_min)

        if r_min < adapt_thresh:
            n_adaptive_macro += 1
            n_sub = min(cfg.max_substeps, max(2, int(math.ceil(adapt_thresh / max(r_min, 1e-30)))))
            max_substeps_used = max(max_substeps_used, n_sub)
            sub_dt = dt_f / n_sub
            for _sub in range(n_sub):
                x, v, a = substep(x, v, a, sub_dt)
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x = x + dt_f * vh
            a = compute_acc(x)
            v = vh + 0.5 * dt_f * a
            total_substeps += 1

        t_cur += dt_f
        while si < cfg.n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < cfg.n_samples:
                nt = times[si]

        if np.max(np.linalg.norm(x, axis=1)) > cfg.ejection_au:
            ejected = True
            while si < cfg.n_samples:
                pos_out[si] = x
                vel_out[si] = v
                si += 1
            break

        if t_cur >= cfg.T - 1e-12:
            break

    while si < cfg.n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    elapsed = time.perf_counter() - t0
    steps_done = max(1, int(round(t_cur / dt_f)))
    metrics = {
        "dt": float(dt),
        "adapt_thresh": float(adapt_thresh),
        "steps": int(steps_done),
        "total_time_sec": float(elapsed),
        "time_per_step_sec": float(elapsed / max(steps_done, 1)),
        "macro_adaptive_frac": float(n_adaptive_macro / max(steps_done, 1)),
        "total_substeps": int(total_substeps),
        "avg_substeps_per_macro": float(total_substeps / max(steps_done, 1)),
        "max_substeps_used": int(max_substeps_used),
        "min_r_seen": float(min_r_seen),
        "ejected": bool(ejected),
    }
    return times, pos_out, vel_out, metrics


# ── Metrics and parsing ───────────────────────────────────────────────────────
def rms_sep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    per_body = np.sqrt(np.sum(d ** 2, axis=-1))
    return np.sqrt(np.mean(per_body ** 2, axis=1))


def fit_log_slope(times: np.ndarray, delta: np.ndarray, t0_frac: float = 0.10, t1_frac: float = 0.50) -> Tuple[float, Tuple[float, float]]:
    T = float(times[-1])
    t0 = t0_frac * T
    t1 = t1_frac * T
    mask = (times >= t0) & (times <= t1)
    x = times[mask]
    y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = float(x.mean())
    y0 = float(y.mean())
    slope = float(np.sum((x - x0) * (y - y0)) / (np.sum((x - x0) ** 2) + 1e-30))
    return slope, (t0, t1)


def evaluate_run(times_ref: np.ndarray, pos_ref: np.ndarray, pos_model: np.ndarray, sim_metrics: Dict[str, float], ias_time: float) -> Dict[str, float]:
    delta = rms_sep(pos_model, pos_ref)
    slope, window = fit_log_slope(times_ref, delta)
    out = dict(sim_metrics)
    out.update({
        "final_err": float(delta[-1]),
        "time_avg_rms": float(np.sqrt(np.mean(delta ** 2))),
        "divergence_slope_1_per_yr": float(slope),
        "divergence_fit_window_start": float(window[0]),
        "divergence_fit_window_end": float(window[1]),
        "throughput_samp_per_sec": float(len(times_ref) / max(sim_metrics["total_time_sec"], 1e-30)),
        "speedup_vs_ias15": float(ias_time / max(sim_metrics["total_time_sec"], 1e-30)),
    })
    return out


def parse_old_summary(path: str) -> Dict[float, Dict[str, float]]:
    """Parse perf_summary_T100_timeavg.txt-style frontier table if present."""
    if not path or not os.path.exists(path):
        return {}
    rows: Dict[float, Dict[str, float]] = {}
    in_table = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("dt\t") or s.startswith("dt final_err"):
                in_table = True
                continue
            if not in_table or not s:
                continue
            parts = re.split(r"\s+", s)
            if len(parts) < 4:
                continue
            try:
                dt = float(parts[0])
                final_err = float(parts[1])
                time_avg = float(parts[2])
                speedup_txt = parts[-1].replace("x", "")
                speedup = float(speedup_txt)
            except Exception:
                continue
            rows[dt] = {"final_err": final_err, "time_avg_rms": time_avg, "speedup": speedup}
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_threshold_tradeoff(rep_rows: List[Dict[str, float]], out_path: str) -> None:
    thr = [r["adapt_thresh"] for r in rep_rows]
    err = [r["time_avg_rms"] for r in rep_rows]
    speed = [r["speedup_vs_ias15"] for r in rep_rows]
    sub = [r["avg_substeps_per_macro"] for r in rep_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(thr, err, "o-", label="Time-averaged RMS")
    ax1.set_xlabel("Adaptive threshold (AU)")
    ax1.set_ylabel("Time-averaged RMS vs IAS15 (AU)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thr, speed, "s--", label="Speedup")
    ax2.plot(thr, sub, "^:", label="Avg substeps / macro")
    ax2.set_ylabel("Speedup / avg substeps")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_final_err_threshold(rep_rows: List[Dict[str, float]], out_path: str) -> None:
    thr = [r["adapt_thresh"] for r in rep_rows]
    final_err = [r["final_err"] for r in rep_rows]
    time_avg = [r["time_avg_rms"] for r in rep_rows]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(thr, final_err, "o-", label="Final RMS")
    plt.plot(thr, time_avg, "s--", label="Time-avg RMS")
    plt.xlabel("Adaptive threshold (AU)")
    plt.ylabel("RMS deviation vs IAS15 (AU)")
    plt.grid(True, alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_frontier_by_threshold(all_rows: List[Dict[str, float]], out_path: str) -> None:
    thresholds = sorted(set(float(r["adapt_thresh"]) for r in all_rows))
    plt.figure(figsize=(8, 5.5))
    for th in thresholds:
        rows = sorted([r for r in all_rows if abs(r["adapt_thresh"] - th) < 1e-12], key=lambda z: z["dt"])
        xs = [r["throughput_samp_per_sec"] for r in rows]
        ys = [r["time_avg_rms"] for r in rows]
        plt.plot(xs, ys, "o-", label=f"r_adapt={th:.2f}")
        for r in rows:
            plt.annotate(f"dt={r['dt']:.3g}", (r["throughput_samp_per_sec"], r["time_avg_rms"]),
                         textcoords="offset points", xytext=(4, 5), fontsize=7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Throughput (samples/sec, log)")
    plt.ylabel("Time-averaged RMS vs IAS15 (AU, log)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=7, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_divergence(times: np.ndarray, delta: np.ndarray, title: str, out_path: str) -> None:
    slope, window = fit_log_slope(times, delta)
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(times, delta, label="RMS(model - IAS15)")
    plt.axvspan(window[0], window[1], alpha=0.15, label="fit window")
    plt.xlabel("Time (yr)")
    plt.ylabel("RMS position deviation (AU, log)")
    plt.title(f"{title} | slope={slope:.4f} yr$^{{-1}}$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ── Summary writing ───────────────────────────────────────────────────────────
def write_summary(
    out_dir: str,
    cfg: Config,
    dt_values: List[float],
    thresholds: List[float],
    ias_metrics: Dict[str, float],
    rep_rows: List[Dict[str, float]],
    all_rows: List[Dict[str, float]],
    old_rows: Dict[float, Dict[str, float]],
    old_summary_path: str,
) -> str:
    path = os.path.join(out_dir, "adaptive_threshold_sweep_summary.txt")
    best_time = min(rep_rows, key=lambda r: r["time_avg_rms"])
    best_final = min(rep_rows, key=lambda r: r["final_err"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("Adaptive threshold sweep: direct Newtonian force everywhere\n")
        f.write("=" * 78 + "\n")
        f.write(f"T_years: {cfg.T}\n")
        f.write(f"n_samples: {cfg.n_samples}\n")
        f.write(f"dt_values: {dt_values}\n")
        f.write(f"thresholds_AU: {thresholds}\n")
        f.write(f"max_substeps: {cfg.max_substeps}\n")
        f.write("method: exact Newtonian pairwise force; adaptive sub-stepping only changes dt\n")
        f.write("neural_network: none\n\n")

        f.write("Baseline IAS15:\n")
        f.write(f"  total_time_sec: {ias_metrics['total_time_sec']:.6f}\n\n")

        f.write("Representative dt threshold sweep:\n")
        f.write("adapt_thresh\tfinal_err\ttime_avg_rms\tdiv_slope\ttotal_time\tspeedup\tmacro_adapt_frac\tavg_substeps\tmax_substeps\tmin_r\tejected\n")
        for r in rep_rows:
            f.write(
                f"{r['adapt_thresh']:.6g}\t{r['final_err']:.6e}\t{r['time_avg_rms']:.6e}\t"
                f"{r['divergence_slope_1_per_yr']:.6e}\t{r['total_time_sec']:.6f}\t{r['speedup_vs_ias15']:.3f}x\t"
                f"{r['macro_adaptive_frac']:.6f}\t{r['avg_substeps_per_macro']:.6f}\t{r['max_substeps_used']}\t"
                f"{r['min_r_seen']:.6e}\t{r['ejected']}\n"
            )
        f.write("\n")

        f.write("Best representative threshold by time_avg_rms:\n")
        f.write(f"  adapt_thresh={best_time['adapt_thresh']:.6g} AU\n")
        f.write(f"  time_avg_rms={best_time['time_avg_rms']:.6e}\n")
        f.write(f"  final_err={best_time['final_err']:.6e}\n")
        f.write(f"  speedup={best_time['speedup_vs_ias15']:.3f}x\n")
        f.write(f"  avg_substeps_per_macro={best_time['avg_substeps_per_macro']:.6f}\n\n")

        f.write("Best representative threshold by final_err:\n")
        f.write(f"  adapt_thresh={best_final['adapt_thresh']:.6g} AU\n")
        f.write(f"  final_err={best_final['final_err']:.6e}\n")
        f.write(f"  time_avg_rms={best_final['time_avg_rms']:.6e}\n")
        f.write(f"  speedup={best_final['speedup_vs_ias15']:.3f}x\n\n")

        if old_rows:
            f.write("Comparison against previous Run D summary:\n")
            f.write(f"  old_summary_path: {old_summary_path}\n")
            f.write("dt\told_final\told_timeavg\told_speedup\tbest_new_thresh\tnew_final\tnew_timeavg\tnew_speedup\ttimeavg_delta_pct\n")
            for dt in sorted(set(r["dt"] for r in all_rows)):
                if dt not in old_rows:
                    continue
                candidates = [r for r in all_rows if abs(r["dt"] - dt) < 1e-12]
                best = min(candidates, key=lambda r: r["time_avg_rms"])
                old = old_rows[dt]
                delta_pct = 100.0 * (best["time_avg_rms"] - old["time_avg_rms"]) / max(old["time_avg_rms"], 1e-30)
                f.write(
                    f"{dt:.6g}\t{old['final_err']:.6e}\t{old['time_avg_rms']:.6e}\t{old['speedup']:.3f}x\t"
                    f"{best['adapt_thresh']:.6g}\t{best['final_err']:.6e}\t{best['time_avg_rms']:.6e}\t"
                    f"{best['speedup_vs_ias15']:.3f}x\t{delta_pct:+.2f}%\n"
                )
        else:
            f.write("Comparison against previous Run D summary:\n")
            f.write(f"  previous summary not found or could not be parsed: {old_summary_path}\n")
        f.write("\n")

        f.write("Full frontier sweep:\n")
        f.write("adapt_thresh\tdt\tfinal_err\ttime_avg_rms\tthroughput\ttime_per_step\ttotal_time\tsteps\tspeedup\tmacro_adapt_frac\tavg_substeps\tmax_substeps\tmin_r\tejected\n")
        for r in sorted(all_rows, key=lambda z: (z["adapt_thresh"], z["dt"])):
            f.write(
                f"{r['adapt_thresh']:.6g}\t{r['dt']:.6g}\t{r['final_err']:.6e}\t{r['time_avg_rms']:.6e}\t"
                f"{r['throughput_samp_per_sec']:.3f}\t{r['time_per_step_sec']:.6e}\t{r['total_time_sec']:.6f}\t"
                f"{int(r['steps'])}\t{r['speedup_vs_ias15']:.3f}x\t{r['macro_adaptive_frac']:.6f}\t"
                f"{r['avg_substeps_per_macro']:.6f}\t{int(r['max_substeps_used'])}\t{r['min_r_seen']:.6e}\t{r['ejected']}\n"
            )

    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="adaptive_threshold_sweep_out")
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--dt_rep", type=float, default=0.04)
    ap.add_argument("--dt_values", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.08, 0.10, 0.12, 0.15])
    ap.add_argument("--max_substeps", type=int, default=16)
    ap.add_argument("--old_summary", default="perf_summary_T100_timeavg.txt")
    ap.add_argument("--skip_frontier", action="store_true", help="Only run representative dt threshold sweep, not full dt frontier.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(T=float(args.T), n_samples=int(args.n_samples), max_substeps=int(args.max_substeps))
    ic = make_ic1()

    print("[adapt-sweep] Method: direct Newtonian force everywhere; adaptive dt threshold sweep")
    print(f"[adapt-sweep] Output folder: {args.out_dir}")
    print(f"[adapt-sweep] T={cfg.T} yr, n_samples={cfg.n_samples}, max_substeps={cfg.max_substeps}")
    print(f"[adapt-sweep] thresholds={args.thresholds}")
    print(f"[adapt-sweep] dt_values={args.dt_values}")

    print("[adapt-sweep] Running IAS15 baseline...")
    times_ref, pos_ref, vel_ref, ias_metrics = simulate_ias15(ic["x0"], ic["v0"], ic["m"], cfg)
    print(f"[adapt-sweep] IAS15 time={ias_metrics['total_time_sec']:.3f}s")

    # Representative threshold sweep at dt_rep.
    print(f"[adapt-sweep] Representative threshold sweep at dt={args.dt_rep}...")
    rep_rows: List[Dict[str, float]] = []
    rep_deltas: Dict[float, np.ndarray] = {}
    for th in args.thresholds:
        t, pos, vel, sim_m = simulate_newtonian_adaptive(ic["x0"], ic["v0"], ic["m"], cfg, args.dt_rep, th)
        row = evaluate_run(times_ref, pos_ref, pos, sim_m, ias_metrics["total_time_sec"])
        rep_rows.append(row)
        rep_deltas[float(th)] = rms_sep(pos, pos_ref)
        print(
            f"  thresh={th:0.3f} AU | final_err={row['final_err']:.3e} | "
            f"time_avg={row['time_avg_rms']:.3e} | time={row['total_time_sec']:.3f}s | "
            f"speedup={row['speedup_vs_ias15']:.2f}x | adapt_frac={row['macro_adaptive_frac']:.4f} | "
            f"avg_substeps={row['avg_substeps_per_macro']:.3f} | min_r={row['min_r_seen']:.3e}"
        )

    # Full frontier across dt and thresholds.
    all_rows: List[Dict[str, float]] = []
    if args.skip_frontier:
        all_rows = list(rep_rows)
    else:
        print("[adapt-sweep] Full dt x threshold frontier sweep...")
        for th in args.thresholds:
            for dt in args.dt_values:
                t, pos, vel, sim_m = simulate_newtonian_adaptive(ic["x0"], ic["v0"], ic["m"], cfg, dt, th)
                row = evaluate_run(times_ref, pos_ref, pos, sim_m, ias_metrics["total_time_sec"])
                all_rows.append(row)
                print(
                    f"  th={th:0.3f} dt={dt:0.3g} | final={row['final_err']:.3e} | "
                    f"time_avg={row['time_avg_rms']:.3e} | total={row['total_time_sec']:.3f}s | "
                    f"speedup={row['speedup_vs_ias15']:.2f}x | adapt={row['macro_adaptive_frac']:.4f} | "
                    f"sub={row['avg_substeps_per_macro']:.3f}"
                )

    # Ensure representative rows are included in all_rows if dt_rep not in dt_values or skip behavior.
    key_pairs = {(round(r["adapt_thresh"], 10), round(r["dt"], 10)) for r in all_rows}
    for r in rep_rows:
        k = (round(r["adapt_thresh"], 10), round(r["dt"], 10))
        if k not in key_pairs:
            all_rows.append(r)

    old_rows = parse_old_summary(args.old_summary)

    # Plots.
    plot_threshold_tradeoff(rep_rows, os.path.join(args.out_dir, "01_threshold_tradeoff_dt_rep.png"))
    plot_final_err_threshold(rep_rows, os.path.join(args.out_dir, "02_threshold_error_dt_rep.png"))
    plot_frontier_by_threshold(all_rows, os.path.join(args.out_dir, "03_frontier_by_threshold.png"))

    best_rep = min(rep_rows, key=lambda r: r["time_avg_rms"])
    best_th = float(best_rep["adapt_thresh"])
    plot_divergence(
        times_ref,
        rep_deltas[best_th],
        f"Best threshold at dt={args.dt_rep}: r_adapt={best_th:.3f} AU",
        os.path.join(args.out_dir, "04_divergence_best_threshold_dt_rep.png"),
    )

    summary_path = write_summary(
        args.out_dir, cfg, list(args.dt_values), list(args.thresholds),
        ias_metrics, rep_rows, all_rows, old_rows, args.old_summary,
    )

    print(f"[adapt-sweep] wrote {args.out_dir}")
    print(f"[adapt-sweep] summary: {summary_path}")
    if old_rows:
        print(f"[adapt-sweep] parsed old summary: {args.old_summary}")
    else:
        print(f"[adapt-sweep] old summary not found/parsed: {args.old_summary}")
    print("[adapt-sweep] done.")


if __name__ == "__main__":
    main()
