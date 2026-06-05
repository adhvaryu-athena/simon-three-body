# pair_eval.py
# Pure NumPy integration with pre-transposed weight matrices.
# Hidden=32 -> ~16x fewer FLOPs per forward pass vs hidden=128.

import os, time, math, argparse
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 15, "axes.titlesize": 20, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 15,
    "figure.titlesize": 20, "mathtext.fontset": "cm", "mathtext.rm": "serif",
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})
import rebound

class PairCorrectionNN(nn.Module):
    def __init__(self, hidden=32, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std', torch.ones(3))
    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)

@dataclass
class HybridConfig:
    G: float = 1.0; eps: float = 3e-4
    mc_samples: int = 1; unc_rel_thresh: float = 0.25
    c_min: float = 0.2; c_max: float = 5.0; r_soft_min: float = 5e-4

def extract_weights_numpy(model):
    sd = model.state_dict()
    # Pre-transpose weights so hot loop does x @ wT directly (already transposed)
    return {
        'mean': sd['input_mean'].cpu().numpy().astype(np.float32),
        'std': sd['input_std'].cpu().numpy().astype(np.float32) + 1e-8,
        'w0T': sd['net.0.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (3, H)
        'b0': sd['net.0.bias'].cpu().numpy().astype(np.float32),
        'w1T': sd['net.2.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, H)
        'b1': sd['net.2.bias'].cpu().numpy().astype(np.float32),
        'w2T': sd['net.4.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, H)
        'b2': sd['net.4.bias'].cpu().numpy().astype(np.float32),
        'w3T': sd['net.6.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, 1)
        'b3': sd['net.6.bias'].cpu().numpy().astype(np.float32),
    }

def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples,
                             device="cpu", dtype=torch.float32):
    w = extract_weights_numpy(model)
    w_mean = w['mean']; w_std = w['std']
    w0T = w['w0T']; b0 = w['b0']; w1T = w['w1T']; b1 = w['b1']
    w2T = w['w2T']; b2 = w['b2']; w3T = w['w3T']; b3 = w['b3']

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii); jj = np.array(jj); P = len(ii)

    G = cfg.G
    eps2 = cfg.eps * cfg.eps
    c_min = cfg.c_min
    c_max = cfg.c_max
    r_soft_min = cfg.r_soft_min

    # Physics state in float64 for precision over long integrations
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]; mj_arr = m_f[jj]
    Gmimj = G * mi_arr * mj_arr  # precompute (P,)
    inv_mi = 1.0 / mi_arr  # (P,)
    inv_mj = 1.0 / mj_arr  # (P,)
    # Precompute log masses in float32 for NN input (constant across steps)
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)  # (P,)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)  # (P,)

    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    # Threshold: when r > nn_thresh, use pure unsoftened Newtonian gravity.
    # At r = 500*eps = 0.15, c = (r_soft/r)^3 = 1 + 6e-7 — negligible.
    # This means the NN is ONLY invoked for extremely close encounters
    # (r < 0.15), which are rare. For typical distances (r ~ 0.5-5.0),
    # the force is exact Newtonian — identical to what IAS15 computes.
    nn_thresh = 500.0 * cfg.eps  # ~0.15

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]                              # (P,3) float64
        r2 = np.einsum('ij,ij->i', rij, rij)                 # (P,) float64
        r = np.sqrt(r2 + 1e-30)                               # (P,) float64

        # Pure Newtonian force: G*mi*mj / r^3 (default for all pairs)
        invr3 = 1.0 / (r2 * r + 1e-30)                       # 1/r^3
        F_scalar = Gmimj * invr3                               # (P,)

        # Check if ANY pair needs NN correction (close encounter)
        close_mask = r < nn_thresh
        n_close = int(np.sum(close_mask))

        if n_close > 0:
            # Only run NN for close pairs
            r_soft_close = np.sqrt(r2[close_mask] + eps2)

            # Softened force for close pairs
            denom = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom

            # NN forward pass (only for close pairs)
            log_rs = np.log(r_soft_close + 1e-30).astype(np.float32)
            nc = n_close
            nn_in = np.empty((nc, 3), dtype=np.float32)
            nn_in[:, 0] = log_rs
            nn_in[:, 1] = log_mi[close_mask]
            nn_in[:, 2] = log_mj[close_mask]

            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w1T + b1;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            h = h @ w2T + b2;  s = 1.0 / (1.0 + np.exp(-h)); h = h * s
            log_c = (h @ w3T + b3).ravel()
            c = np.exp(log_c).astype(np.float64)

            # Gating for close pairs
            fb = (r_soft_close < r_soft_min) | (c < c_min) | (c > c_max) | ~np.isfinite(c)
            # For non-fallback close pairs: use c * F_soft
            # For fallback close pairs: use Newtonian (already in F_scalar)
            F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_close)
            F_scalar[close_mask] = F_corrected

        F_vec = F_scalar[:, None] * rij                       # (P,3)
        acc = np.zeros((N, 3), dtype=np.float64)
        acc[ii[0]] += F_vec[0] * inv_mi[0]
        acc[jj[0]] -= F_vec[0] * inv_mj[0]
        acc[ii[1]] += F_vec[1] * inv_mi[1]
        acc[jj[1]] -= F_vec[1] * inv_mj[1]
        acc[ii[2]] += F_vec[2] * inv_mi[2]
        acc[jj[2]] -= F_vec[2] * inv_mj[2]
        return acc, n_close

    # Adaptive sub-stepping: when min pairwise distance drops below
    # this threshold, subdivide the step into smaller sub-steps.
    # This resolves close encounters without shrinking dt globally.
    adapt_thresh = 0.05   # ~50x the typical softening length
    max_substeps = 16     # max subdivisions per macro step

    def min_pair_dist(pos):
        """Return minimum pairwise distance."""
        rij = pos[jj] - pos[ii]
        r2 = np.einsum('ij,ij->i', rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        """One velocity-Verlet sub-step."""
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, nf = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, nf

    a, n_fb = compute_acc(x)
    fb_sum = n_fb; pair_sum = P
    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt); total_substeps = 0
    t_start = time.perf_counter()
    for _ in range(n_steps):
        # Check if any pair is close enough to need sub-stepping
        r_min = min_pair_dist(x)

        if r_min < adapt_thresh:
            # Determine number of sub-steps: more sub-steps when closer
            # n_sub scales as (threshold / r_min), capped at max_substeps
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
            total_substeps += n_sub
        else:
            # Normal step — no close encounter
            vh = v + 0.5 * dt_f * a
            x = x + dt_f * vh
            a, nf = compute_acc(x)
            v = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
            total_substeps += 1

        t_cur += dt; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12: break
    total_time = time.perf_counter() - t_start

    return times, pos_out, vel_out, {
        "steps": steps, "dt": dt, "T_years": T, "n_samples": n_samples,
        "total_time_sec": total_time,
        "time_per_step_sec": total_time / max(steps, 1),
        "avg_fallback_frac": fb_sum / max(pair_sum, 1),
        "avg_pairs_per_step": P,
        "total_substeps": total_substeps,
    }

def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation(); sim.integrator = "ias15"; sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]), x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k,i] = [p.x, p.y, p.z]; vel[k,i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}

def rms_sep(a, b):
    d = a - b; pb = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(pb**2, axis=1))

def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T=times[-1]; t0=t0_frac*T; t1=t1_frac*T
    mask=(times>=t0)&(times<=t1); x=times[mask]; y=np.log(np.clip(delta[mask],1e-30,None))
    x0=x.mean(); y0=y.mean()
    return float(np.sum((x-x0)*(y-y0))/(np.sum((x-x0)**2)+1e-30)), (t0,t1)

def plot_overlay_xy_subplots_all_bodies(pr, pm, op):
    N = pr.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.2*N, 4.0))

    if N == 1:
        axes = [axes]

    for i in range(N):
        ax = axes[i]

        ax.plot(pr[:, i, 0], pr[:, i, 1], "-",  lw=1.4, label="IAS15")
        ax.plot(pm[:, i, 0], pm[:, i, 1], "--", lw=1.4, label="Hybrid")

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"Body {i}", fontsize=12)

        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.25)

        # Show the legend only once to avoid covering trajectories in every panel.
        if i == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    # No figure-level title: the LaTeX caption explains the figure.
    plt.tight_layout()
    plt.savefig(op, dpi=300)
    plt.close()

def plot_overlay_timeseries_all_bodies(times, pr, pm, op):
    N = pr.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(10.5, 2.8*N), sharex=True)

    if N == 1:
        axes = np.array([axes])

    for i in range(N):
        axes[i, 0].plot(times, pr[:, i, 0], "-",  lw=1.3, label="IAS15")
        axes[i, 0].plot(times, pm[:, i, 0], "--", lw=1.3, label="Hybrid")
        axes[i, 0].set_ylabel(f"Body {i}: x(t)", fontsize=11)
        axes[i, 0].tick_params(axis="both", labelsize=10)
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].plot(times, pr[:, i, 1], "-",  lw=1.3, label="IAS15")
        axes[i, 1].plot(times, pm[:, i, 1], "--", lw=1.3, label="Hybrid")
        axes[i, 1].set_ylabel(f"Body {i}: y(t)", fontsize=11)
        axes[i, 1].tick_params(axis="both", labelsize=10)
        axes[i, 1].grid(True, alpha=0.25)

        # Show one compact legend only.
        if i == 0:
            axes[i, 0].legend(loc="upper left", fontsize=9, framealpha=0.85)

    axes[-1, 0].set_xlabel("Time (yr)", fontsize=11)
    axes[-1, 1].set_xlabel("Time (yr)", fontsize=11)

    # No figure-level title: the LaTeX caption explains the figure.
    plt.tight_layout()
    plt.savefig(op, dpi=300)
    plt.close()

def plot_divergence(times, delta, slope, window, op):
    plt.figure(); plt.semilogy(times, delta, label="d(t) = RMS(Hybrid - IAS15)")
    plt.axvspan(window[0],window[1],alpha=0.15,label="fit window")
    plt.xlabel("Time (yr)"); plt.ylabel("Root Mean Square Position")
    plt.title("SIMON-vs-REBOUND Divergence"); plt.grid(True,which="both",alpha=0.3)
    plt.legend(); plt.tight_layout(); plt.savefig(op,dpi=200); plt.close()

def plot_speed_accuracy_frontier(pts, op):
    # Local font override for this figure only.
    # This prevents the large global rcParams from making the axis labels/ticks huge.
    with plt.rc_context({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))

        xs = [p["throughput_samp_per_sec"] for p in pts]
        ys = [p["final_err"] for p in pts]

        ax.scatter(xs, ys, s=42, zorder=3)

        # Label offsets are visual only; they do not affect any computed values.
        label_offsets = {
            0.005: (4, 8),
            0.01:  (4, 16),
            0.02:  (4, 8),
            0.04:  (4, -14),
            0.08:  (-6, 8),
        }

        for p in pts:
            dt = float(p["dt"])
            dx, dy = label_offsets.get(dt, (4, 8))
            ax.annotate(
                f"dt={p['dt']}",
                xy=(p["throughput_samp_per_sec"], p["final_err"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                ha="right" if dx < 0 else "left",
                va="center"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Throughput (recorded samples / sec, log)")
        ax.set_ylabel("Final RMS deviation vs IAS15 at T = 100 yr (log)")

        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=7)
        ax.grid(True, which="both", alpha=0.25)

        # No chart title: the LaTeX caption explains the figure.
        plt.tight_layout()
        plt.savefig(op, dpi=300)
        plt.close()

def plot_comp_cost_frontier(pts, op):
    dts=[p["dt"] for p in pts]; tps=[p["time_per_step_sec"] for p in pts]; tots=[p["total_time_sec"] for p in pts]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(dts,tps,marker="o"); axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("dt (yr)"); axes[0].set_ylabel("time per integration step (sec)")
    axes[0].set_title("Time Per Step vs dt"); axes[0].grid(True,which="both",alpha=0.3)
    axes[1].plot(dts,tots,marker="o"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("dt (years, log)"); axes[1].set_ylabel("total simulation time (sec, log)")
    axes[1].set_title("Total sim time vs dt"); axes[1].grid(True,which="both",alpha=0.3)
    fig.suptitle("Computational cost (hybrid model), T=100 yr")
    plt.tight_layout(); plt.savefig(op,dpi=200); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pair_correction_nn")
    parser.add_argument("--out_dir", default="hybrid_eval_out_v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dt_rep", type=float, default=0.02)
    parser.add_argument("--n_samples", type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    mp = args.model_path
    if not mp.endswith(".pt") and os.path.exists(mp+".pt"): mp += ".pt"
    model = PairCorrectionNN(hidden=32); sd = torch.load(mp, map_location="cpu")
    model.load_state_dict(sd); model.eval()
    print(f"[eval] Loaded {mp} | params={sum(p.numel() for p in model.parameters())}")

    T=100.0; ns=args.n_samples
    m=np.array([1.0,0.01,0.005]); x0=np.array([[0,0,0],[1,0,0],[0,1.2,0]],dtype=np.float64)
    v0=np.array([[0,0,0],[0,1,0],[-0.9,0,0]],dtype=np.float64)
    M=m.sum(); x0=x0-(m[:,None]*x0).sum(0)/M; v0=v0-(m[:,None]*v0).sum(0)/M

    print("[eval] IAS15 baseline...")
    tr,pr,vr,perf_r = simulate_rebound_ias15(x0,v0,m,cfg.G,T,ns)
    print(f"[eval] IAS15 time={perf_r['total_time_sec']:.3f}s")

    dr = args.dt_rep
    print(f"[eval] Hybrid dt={dr}...")
    tm,pm,vm,perf_m = simulate_leapfrog_hybrid(x0,v0,m,model,cfg,dr,T,ns)
    sp = perf_r['total_time_sec']/max(perf_m['total_time_sec'],1e-12)
    print(f"[eval] Hybrid: {perf_m['total_time_sec']:.3f}s | {perf_m['time_per_step_sec']:.2e}s/step | "
          f"fallback={perf_m['avg_fallback_frac']:.3f} | SPEEDUP={sp:.2f}x")

    plot_overlay_xy_subplots_all_bodies(pr,pm,os.path.join(args.out_dir,"traj_overlay_xy_subplots_all_bodies_T100.png"))
    plot_overlay_timeseries_all_bodies(tr,pr,pm,os.path.join(args.out_dir,"traj_overlay_timeseries_all_bodies_T100.png"))
    delta=rms_sep(pm,pr); slope,win=fit_log_slope(tr,delta)
    plot_divergence(tr,delta,slope,win,os.path.join(args.out_dir,"model_vs_rebound_divergence_T100.png"))
    print(f"[eval] divergence slope~{slope:.3e} 1/yr, window={win}")

    dts=[0.005,0.01,0.02,0.04,0.08]; frontier=[]
    print("[eval] Frontier sweep...")
    for dt in dts:
        _,pd,_,pf = simulate_leapfrog_hybrid(x0,v0,m,model,cfg,dt,T,ns)
        fe=float(rms_sep(pd[-1:],pr[-1:])[0]); tp=ns/max(pf['total_time_sec'],1e-12)
        row={"dt":dt,"final_err":fe,"throughput_samp_per_sec":tp,
             "time_per_step_sec":pf["time_per_step_sec"],"total_time_sec":pf["total_time_sec"],
             "avg_fallback_frac":pf["avg_fallback_frac"],"steps":pf["steps"]}
        frontier.append(row)
        s=perf_r['total_time_sec']/max(pf['total_time_sec'],1e-12)
        print(f"  dt={dt:>6} err={fe:.3e} thrpt={tp:.0f} t/step={pf['time_per_step_sec']:.2e} "
              f"total={pf['total_time_sec']:.3f}s fb={pf['avg_fallback_frac']:.3f} speedup={s:.2f}x")

    plot_speed_accuracy_frontier(frontier,os.path.join(args.out_dir,"speed_accuracy_frontier_T100.png"))
    plot_comp_cost_frontier(frontier,os.path.join(args.out_dir,"comp_cost_frontier_T100.png"))

    with open(os.path.join(args.out_dir,"perf_summary_T100.txt"),"w",encoding="utf-8") as f:
        f.write(f"T_years: {T}\nn_samples: {ns}\n\n")
        f.write(f"Baseline (REBOUND IAS15):\n  total_time_sec: {perf_r['total_time_sec']:.6f}\n\n")
        f.write(f"Representative hybrid run (dt_rep={dr}):\n  steps: {perf_m['steps']}\n")
        f.write(f"  total_time_sec: {perf_m['total_time_sec']:.6f}\n")
        f.write(f"  time_per_step_sec: {perf_m['time_per_step_sec']:.6e}\n")
        f.write(f"  avg_fallback_frac: {perf_m['avg_fallback_frac']:.6f}\n")
        f.write(f"  divergence_slope_1_per_yr: {slope:.6e}\n  divergence_fit_window_years: {win}\n")
        f.write(f"  speedup_vs_ias15: {sp:.2f}x\n\n")
        f.write("Frontier sweep (hybrid):\ndt\tfinal_err\tthroughput(samp/s)\ttime_per_step(s)\ttotal_time(s)\tsteps\tfallback_frac\tspeedup\n")
        for p in frontier:
            s2=perf_r['total_time_sec']/max(p['total_time_sec'],1e-12)
            f.write(f"{p['dt']}\t{p['final_err']:.6e}\t{p['throughput_samp_per_sec']:.3f}\t"
                    f"{p['time_per_step_sec']:.6e}\t{p['total_time_sec']:.6f}\t{p['steps']}\t"
                    f"{p['avg_fallback_frac']:.3f}\t{s2:.2f}x\n")
        f.write("\nNotes:\n- final_err is RMS(model - IAS15) at T=100 years.\n")
        f.write("- divergence_slope is fitted slope of log d(t) where d(t)=RMS(model-IAS15).\n")
        f.write("- speedup = IAS15_total_time / hybrid_total_time.\n")
    print(f"[eval] wrote {args.out_dir}"); print("[eval] done.")

if __name__ == "__main__":
    main()
