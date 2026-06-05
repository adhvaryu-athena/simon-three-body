# eval_hybrid_forcemag_v3.py
# ------------------------------------------------------------
# Eval-only script (NO retraining). Uses a saved model:
#   pair_correction_nn(.pt)
#
# Outputs (ALL with T=100 years):
#   - traj_overlay_xy_subplots_all_bodies_T100.png  (clean: one subplot per body, XY)
#   - traj_overlay_timeseries_all_bodies_T100.png   (x(t), y(t) per body)
#   - model_vs_rebound_divergence_T100.png          (RMS(model - IAS15) + fitted log-slope)
#   - speed_accuracy_frontier_T100.png              (final error vs throughput)
#   - comp_cost_frontier_T100.png                   (dt vs time/step and total time)
#   - perf_summary_T100.txt                         (includes computational cost table)
#
# “Lyapunov divergence proxy” here means:
#   run IAS15 baseline and one model sim (same ICs),
#   compute δ(t)=RMS(model - IAS15),
#   fit slope of log δ(t) over a window => divergence rate relative to IAS15.
# ------------------------------------------------------------

import os
import time
import math
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# --- Matplotlib styling to match Overleaf-ish paper fonts ---
plt.rcParams.update({
    
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],

    # Font sizes (good starting point for 10pt/11pt paper text)
    "font.size": 15,          # base text
    "axes.titlesize": 20,     # subplot title
    "axes.labelsize": 20,     # axis labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "figure.titlesize": 20,   # suptitle

    
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",

    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

import rebound


# ----------------------------
# Model (must match training)
# ----------------------------
class PairCorrectionNN(nn.Module):
    def __init__(self, hidden=128, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4

    # Uncertainty / hybrid fallback
    mc_samples: int = 8
    unc_rel_thresh: float = 0.25
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


# ----------------------------
# Physics helpers
# ----------------------------
def force_newton_vec(rij: torch.Tensor, mi: float, mj: float, G: float = 1.0) -> torch.Tensor:
    r2 = torch.dot(rij, rij) + 1e-30
    invr3 = torch.rsqrt(r2) / r2  # 1/r^3
    return G * (mi * mj) * rij * invr3


def force_soft_vec(rij: torch.Tensor, mi: float, mj: float, eps: float, G: float = 1.0) -> torch.Tensor:
    r2 = torch.dot(rij, rij)
    denom = (r2 + eps * eps) ** 1.5 + 1e-30
    return G * (mi * mj) * rij / denom


@torch.no_grad()
def hybrid_forces(pos, m, model, cfg: HybridConfig, enable_dropout_uncertainty=True, device="cpu"):
    """
    Pairwise hybrid rule:
      - If NN looks unreliable (uncertainty / bounds / nonfinite / close encounter),
        replace neural prediction with Newtonian (analytic) force for that pair.
      - Else use corrected softened force: F = c * F_soft.

    Returns:
      acc: (N,3)
      stats: dict including fallback fraction
    """
    N = pos.shape[0]
    acc = torch.zeros((N, 3), dtype=pos.dtype, device=device)

    fallback_pairs = 0
    total_pairs = 0

    # MC Dropout: set train() to activate dropout layers
    if enable_dropout_uncertainty:
        model.train()
    else:
        model.eval()

    for i in range(N):
        for j in range(i + 1, N):
            total_pairs += 1
            rij = pos[j] - pos[i]

            r2 = torch.dot(rij, rij)
            r_soft = torch.sqrt(r2 + cfg.eps * cfg.eps + 1e-30)

            # baseline softened vector force
            F_soft = force_soft_vec(rij, float(m[i]), float(m[j]), cfg.eps, cfg.G)

            # decide fallback
            use_fallback = False

            # always fallback for very close approaches (numerical safety)
            if float(r_soft) < cfg.r_soft_min:
                use_fallback = True

            c_mean = 1.0
            if not use_fallback:
                x = torch.tensor(
                    [float(torch.log(r_soft)),
                     float(torch.log(m[i] + 1e-30)),
                     float(torch.log(m[j] + 1e-30))],
                    dtype=pos.dtype,
                    device=device
                ).unsqueeze(0)  # (1,3)

                if enable_dropout_uncertainty:
                    cs = []
                    for _ in range(cfg.mc_samples):
                        logc = model(x)
                        c = torch.exp(logc).clamp(1e-6, 1e6)
                        cs.append(float(c.item()))
                    cs = np.array(cs, dtype=np.float64)
                    c_mean = float(cs.mean())
                    c_std = float(cs.std(ddof=0))
                    rel_unc = c_std / (abs(c_mean) + 1e-12)

                    if (not np.isfinite(c_mean)) or (not np.isfinite(rel_unc)):
                        use_fallback = True
                    elif rel_unc > cfg.unc_rel_thresh:
                        use_fallback = True
                    elif (c_mean < cfg.c_min) or (c_mean > cfg.c_max):
                        use_fallback = True
                else:
                    model.eval()
                    logc = model(x)
                    c_mean = float(torch.exp(logc).clamp(1e-6, 1e6))
                    if (c_mean < cfg.c_min) or (c_mean > cfg.c_max) or (not np.isfinite(c_mean)):
                        use_fallback = True

            if use_fallback:
                fallback_pairs += 1
                F_ij = force_newton_vec(rij, float(m[i]), float(m[j]), cfg.G)
            else:
                F_ij = c_mean * F_soft

            # Apply equal and opposite forces => exact momentum conservation
            acc[i] += F_ij / (m[i] + 1e-30)
            acc[j] -= F_ij / (m[j] + 1e-30)

    return acc, {
        "fallback_pairs": fallback_pairs,
        "total_pairs": total_pairs,
        "fallback_frac": fallback_pairs / max(total_pairs, 1),
    }


# ----------------------------
# Hybrid integrator (leapfrog)
# ----------------------------
@torch.no_grad()
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples, device="cpu", dtype=torch.float32):
    """
    Velocity-Verlet / leapfrog with hybrid forces.
    Returns:
      times, pos(T,N,3), vel(T,N,3), perf dict with computational cost.
    """
    x = torch.tensor(x0, dtype=dtype, device=device)
    v = torch.tensor(v0, dtype=dtype, device=device)
    mt = torch.tensor(m, dtype=dtype, device=device)

    times = np.linspace(0.0, T, n_samples)
    pos_out = np.zeros((n_samples, x0.shape[0], 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, x0.shape[0], 3), dtype=np.float64)

    # initial accel
    a, st = hybrid_forces(x, mt, model, cfg, enable_dropout_uncertainty=True, device=device)
    fallback_sum = st["fallback_pairs"]
    pair_sum = st["total_pairs"]

    # recording schedule
    sample_idx = 0
    next_t = times[sample_idx]

    t = 0.0
    n_steps = int(math.ceil(T / dt))
    steps = 0

    # record at t=0
    while sample_idx < n_samples and t >= next_t - 1e-12:
        pos_out[sample_idx] = x.detach().cpu().numpy()
        vel_out[sample_idx] = v.detach().cpu().numpy()
        sample_idx += 1
        if sample_idx < n_samples:
            next_t = times[sample_idx]

    t_start = time.perf_counter()
    for _ in range(n_steps):
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        t += dt

        a, st = hybrid_forces(x, mt, model, cfg, enable_dropout_uncertainty=True, device=device)
        fallback_sum += st["fallback_pairs"]
        pair_sum += st["total_pairs"]

        v = v_half + 0.5 * dt * a
        steps += 1

        while sample_idx < n_samples and t >= next_t - 1e-12:
            pos_out[sample_idx] = x.detach().cpu().numpy()
            vel_out[sample_idx] = v.detach().cpu().numpy()
            sample_idx += 1
            if sample_idx < n_samples:
                next_t = times[sample_idx]

        if t >= T - 1e-12:
            break

    total_time = time.perf_counter() - t_start

    perf = {
        "steps": steps,
        "dt": dt,
        "T_years": T,
        "n_samples": n_samples,
        "total_time_sec": total_time,
        "time_per_step_sec": total_time / max(steps, 1),
        "avg_fallback_frac": fallback_sum / max(pair_sum, 1),
        "avg_pairs_per_step": pair_sum / max(steps, 1),  # diagnostics
    }
    return times, pos_out, vel_out, perf


# ----------------------------
# REBOUND IAS15 baseline
# ----------------------------
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()

    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((n_samples, len(m), 3), dtype=np.float64)

    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


# ----------------------------
# Metrics
# ----------------------------
def rms_sep(posA, posB):
    d = posA - posB
    per_body = np.sqrt(np.sum(d ** 2, axis=-1))  # (T,N)
    return np.sqrt(np.mean(per_body ** 2, axis=1))  # (T,)


def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    """
    Fit slope of log(delta(t)) between [t0, t1] where:
      t0 = t0_frac * T, t1 = t1_frac * T
    This is the "rate of trajectory divergence relative to REBOUND IAS15".
    """
    T = times[-1]
    t0 = t0_frac * T
    t1 = t1_frac * T
    mask = (times >= t0) & (times <= t1)
    x = times[mask]
    y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean()
    y0 = y.mean()
    slope = np.sum((x - x0) * (y - y0)) / (np.sum((x - x0) ** 2) + 1e-30)
    return float(slope), (t0, t1)


# ----------------------------
# Plots
# ----------------------------
def plot_overlay_xy_subplots_all_bodies(pos_ref, pos_mod, outpath):
    """
    Clean XY overlay: one subplot per body.
    Baseline IAS15 is solid, hybrid is dashed.
    """
    N = pos_ref.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.5 * N, 4.5), sharex=False, sharey=False)

    if N == 1:
        axes = [axes]

    for i in range(N):
        ax = axes[i]
        ax.plot(pos_ref[:, i, 0], pos_ref[:, i, 1], linestyle="-", label="IAS15")
        ax.plot(pos_mod[:, i, 0], pos_mod[:, i, 1], linestyle="--", label="Hybrid")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Body {i} XY (T=100 yr)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc = 'upper left')

    fig.suptitle("Trajectory overlay (XY), one panel per body")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_overlay_timeseries_all_bodies(times, pos_ref, pos_mod, outpath):
    """
    x(t), y(t) overlays for each body.
    Figure: N rows (bodies) × 2 cols (x(t), y(t)).
    """
    N = pos_ref.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(11, 3.1 * N), sharex=True)

    if N == 1:
        axes = np.array([axes])

    for i in range(N):
        axx = axes[i, 0]
        axy = axes[i, 1]

        axx.plot(times, pos_ref[:, i, 0], label="IAS15")
        axx.plot(times, pos_mod[:, i, 0], linestyle="--", label="Hybrid")
        axx.set_ylabel(f"body {i} x(t)")
        axx.grid(True, alpha=0.3)
        if i == 0:
            axx.legend(loc = "upper left")

        axy.plot(times, pos_ref[:, i, 1], label="IAS15")
        axy.plot(times, pos_mod[:, i, 1], linestyle="--", label="Hybrid")
        axy.set_ylabel(f"body {i} y(t)")
        axy.grid(True, alpha=0.3)
        if i == 0:
            axy.legend()

    axes[-1, 0].set_xlabel("time (yr)")
    axes[-1, 1].set_xlabel("time (yr)")
    fig.suptitle("Trajectory Time-Series Overlay")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_divergence(times, delta, slope, window, outpath):
    plt.figure()
    plt.semilogy(times, delta, label="δ(t) = RMS(Hybrid − IAS15)")
    plt.axvspan(window[0], window[1], alpha=0.15, label="fit window")
    plt.xlabel("Time (yr)")
    plt.ylabel("Root Mean Square Position")
    plt.title(f"SIMON-vs-REBOUND Divergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_speed_accuracy_frontier(points, outpath):
    """
    Scatter: throughput vs final error at T=100
    """
    plt.figure()
    xs = [p["throughput_samp_per_sec"] for p in points]
    ys = [p["final_err"] for p in points]
    plt.scatter(xs, ys)
    for p in points:
        plt.annotate(f"dt={p['dt']}", (p["throughput_samp_per_sec"], p["final_err"]))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Throughput (recorded samples / sec, log)")
    plt.ylabel("Final RMS deviation vs IAS15 at T=100 yr (log)")
    plt.title("Speed–accuracy frontier (hybrid model, T=100 yr)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_comp_cost_frontier(points, outpath):
    """
    Computational cost plots:
      - time per step vs dt
      - total sim time vs dt
    """
    dts = [p["dt"] for p in points]
    tps = [p["time_per_step_sec"] for p in points]
    tots = [p["total_time_sec"] for p in points]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(dts, tps, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dt (yr)")
    axes[0].set_ylabel("time per integration step (sec)")
    axes[0].set_title("Time Per Step vs dt")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(dts, tots, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("dt (years, log)")
    axes[1].set_ylabel("total simulation time (sec, log)")
    axes[1].set_title("Total sim time vs dt")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Computational cost (hybrid model), T=100 yr")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main eval
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pair_correction_nn")
    parser.add_argument("--out_dir", type=str, default="hybrid_eval_out_v3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dt_rep", type=float, default=0.02)
    parser.add_argument("--n_samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = HybridConfig()
    device = args.device
    dtype = torch.float32

    # allow both "pair_correction_nn" and "pair_correction_nn.pt"
    model_path = args.model_path
    if not model_path.endswith(".pt"):
        if os.path.exists(model_path + ".pt"):
            model_path = model_path + ".pt"

    # load model
    model = PairCorrectionNN(hidden=128, p_drop=0.1).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # Standard horizon for ALL analyses
    T = 100.0
    n_samples = args.n_samples

    # ICs (same as prior)
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.2, 0.0]], dtype=np.float64)
    v0 = np.array([[0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [-0.9, 0.0, 0.0]], dtype=np.float64)

    # center of mass shift (match REBOUND move_to_com)
    M = m.sum()
    x_cm = (m[:, None] * x0).sum(axis=0) / M
    v_cm = (m[:, None] * v0).sum(axis=0) / M
    x0 = x0 - x_cm[None, :]
    v0 = v0 - v_cm[None, :]

    # Baseline IAS15 (REBOUND)
    print("[eval] Running REBOUND IAS15 baseline (T=100 yr)...")
    t_ref, pos_ref, vel_ref, perf_ref = simulate_rebound_ias15(x0, v0, m, cfg.G, T, n_samples)
    print(f"[eval] IAS15 total_time_sec={perf_ref['total_time_sec']:.3f}")

    # Representative model run (dt_rep)
    dt_rep = args.dt_rep
    print(f"[eval] Running hybrid model (leapfrog) dt={dt_rep} (T=100 yr)...")
    t_mod, pos_mod, vel_mod, perf_mod = simulate_leapfrog_hybrid(
        x0, v0, m, model, cfg, dt_rep, T, n_samples, device=device, dtype=dtype
    )
    print(f"[eval] hybrid(dt={dt_rep}) total_time_sec={perf_mod['total_time_sec']:.3f}  "
          f"time_per_step_sec={perf_mod['time_per_step_sec']:.3e}  "
          f"avg_fallback_frac={perf_mod['avg_fallback_frac']:.3f}")

    # Trajectory overlays (UPDATED: clean subplots for all bodies)
    overlay_xy = os.path.join(args.out_dir, "traj_overlay_xy_subplots_all_bodies_T100.png")
    plot_overlay_xy_subplots_all_bodies(pos_ref, pos_mod, overlay_xy)
    print(f"[eval] wrote {overlay_xy}")

    overlay_ts = os.path.join(args.out_dir, "traj_overlay_timeseries_all_bodies_T100.png")
    plot_overlay_timeseries_all_bodies(t_ref, pos_ref, pos_mod, overlay_ts)
    print(f"[eval] wrote {overlay_ts}")

    # Divergence proxy: model vs REBOUND (trajectory divergence rate relative to IAS15)
    delta = rms_sep(pos_mod, pos_ref)
    slope, win = fit_log_slope(t_ref, delta, t0_frac=0.10, t1_frac=0.50)
    div_path = os.path.join(args.out_dir, "model_vs_rebound_divergence_T100.png")
    plot_divergence(t_ref, delta, slope, win, div_path)
    print(f"[eval] divergence slope≈{slope:.3e} 1/yr, window={win}")
    print(f"[eval] wrote {div_path}")

    # Speed–accuracy frontier + computational cost (T=100 yr)
    dts = [0.005, 0.01, 0.02, 0.04, 0.08]
    frontier = []
    print("[eval] Frontier sweep (T=100 yr)...")
    for dt in dts:
        _, pos_dt, _, perf = simulate_leapfrog_hybrid(
            x0, v0, m, model, cfg, dt, T, n_samples, device=device, dtype=dtype
        )
        final_err = float(rms_sep(pos_dt[-1:, :, :], pos_ref[-1:, :, :])[0])
        throughput = n_samples / max(perf["total_time_sec"], 1e-12)

        row = {
            "dt": dt,
            "final_err": final_err,
            "throughput_samp_per_sec": throughput,
            "time_per_step_sec": perf["time_per_step_sec"],
            "total_time_sec": perf["total_time_sec"],
            "avg_fallback_frac": perf["avg_fallback_frac"],
            "steps": perf["steps"],
        }
        frontier.append(row)

        print(f"  dt={dt:>6} | final_err={final_err:10.3e} | "
              f"throughput={throughput:10.1f} samp/s | "
              f"time/step={perf['time_per_step_sec']:.3e}s | "
              f"total={perf['total_time_sec']:.3f}s | "
              f"fallback={perf['avg_fallback_frac']:.3f}")

    speed_acc_path = os.path.join(args.out_dir, "speed_accuracy_frontier_T100.png")
    plot_speed_accuracy_frontier(frontier, speed_acc_path)
    print(f"[eval] wrote {speed_acc_path}")

    comp_cost_path = os.path.join(args.out_dir, "comp_cost_frontier_T100.png")
    plot_comp_cost_frontier(frontier, comp_cost_path)
    print(f"[eval] wrote {comp_cost_path}")

    # Summary file (includes computational cost)
    summary_path = os.path.join(args.out_dir, "perf_summary_T100.txt")
    with open(summary_path, "w") as f:
        f.write(f"T_years: {T}\n")
        f.write(f"n_samples: {n_samples}\n\n")

        f.write("Baseline (REBOUND IAS15):\n")
        f.write(f"  total_time_sec: {perf_ref['total_time_sec']:.6f}\n\n")

        f.write(f"Representative hybrid run (dt_rep={dt_rep}):\n")
        f.write(f"  steps: {perf_mod['steps']}\n")
        f.write(f"  total_time_sec: {perf_mod['total_time_sec']:.6f}\n")
        f.write(f"  time_per_step_sec: {perf_mod['time_per_step_sec']:.6e}\n")
        f.write(f"  avg_fallback_frac: {perf_mod['avg_fallback_frac']:.6f}\n")
        f.write(f"  divergence_slope_1_per_yr: {slope:.6e}\n")
        f.write(f"  divergence_fit_window_years: {win}\n\n")

        f.write("Frontier sweep (hybrid):\n")
        f.write("dt\tfinal_err\tthroughput(samp/s)\ttime_per_step(s)\ttotal_time(s)\tsteps\tfallback_frac\n")
        for p in frontier:
            f.write(f"{p['dt']}\t{p['final_err']:.6e}\t{p['throughput_samp_per_sec']:.3f}\t"
                    f"{p['time_per_step_sec']:.6e}\t{p['total_time_sec']:.6f}\t"
                    f"{p['steps']}\t{p['avg_fallback_frac']:.3f}\n")

        f.write("\nNotes:\n")
        f.write("- final_err is RMS(model - IAS15) at T=100 years.\n")
        f.write("- divergence_slope is the fitted slope of log δ(t) where δ(t)=RMS(model - IAS15).\n")
        f.write("- time_per_step and total_time are measured for the hybrid leapfrog simulation.\n")

    print(f"[eval] wrote {summary_path}")
    print("[eval] done.")


if __name__ == "__main__":
    main()
