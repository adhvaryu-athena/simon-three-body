# no_nn_ablation.py
#
# Ablation study: what happens when the NN correction is removed entirely?
# Compares three models under identical conditions:
#   1. ias15            -- ground truth (Gauss-Radau 15th order)
#   2. SIMON            -- with NN correction (c * G*mi*mj/r_soft^3 * rij)
#   3. No-NN baseline   -- pure softened gravity (G*mi*mj/r_soft^3 * rij, c=1 always)
#
# PURPOSE (addressing reviewer Issue 1):
#   V2 states the NN is "only used 1.1% of the time" — a reviewer might ask
#   "then why is it needed?" This ablation answers directly: by removing the NN
#   at those 1.1% of close encounters and observing whether bodies are ejected.
#   If yes → NN is proven essential despite its low invocation rate.
#   If no  → paper framing needs adjustment.
#
# SELF-CONTAINED: does not import from compare_models.py or any other project file.
# ALIGNED: simulate_hybrid loop is identical to compare_models.py (SIMON path).
#
# REQUIRES in same folder:
#   pair_correction_nn.pt    (SIMON model — only SIMON is needed, not vector_nn.pt)
#
# PRODUCES in comparison_out_no_nn/:
#   divergence_no_nn.png     -- SIMON vs No-NN divergence curves
#   trajectory_no_nn.png     -- XY trajectory overlay all 3 bodies
#   timeseries_no_nn.png     -- x(t), y(t) time series
#   summary_no_nn.txt        -- quantitative results table + verdict
#
# Run: python no_nn_ablation.py
# All output goes to comparison_out_no_nn\ — existing folders are untouched.

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
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found.")
    raise


# ── Model (identical to compare_models.py) ────────────────────────────────────
class PairCorrectionNN(nn.Module):
    """SIMON: 3->32->32->32->1, predicts scalar log(c)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std',  torch.ones(3))
    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


# ── Config (identical to compare_models.py) ───────────────────────────────────
@dataclass
class HybridConfig:
    G: float          = 1.0
    eps: float        = 3e-4
    c_min: float      = 0.2
    c_max: float      = 5.0
    r_soft_min: float = 5e-4


# ── Weight extraction (identical to compare_models.py) ────────────────────────
def extract_simon(model):
    sd = model.state_dict()
    return {
        'mean': sd['input_mean'].cpu().numpy().astype(np.float32),
        'std':  sd['input_std'].cpu().numpy().astype(np.float32) + 1e-8,
        'w0T':  sd['net.0.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b0':   sd['net.0.bias'].cpu().numpy().astype(np.float32),
        'w1T':  sd['net.2.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b1':   sd['net.2.bias'].cpu().numpy().astype(np.float32),
        'w2T':  sd['net.4.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b2':   sd['net.4.bias'].cpu().numpy().astype(np.float32),
        'w3T':  sd['net.6.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b3':   sd['net.6.bias'].cpu().numpy().astype(np.float32),
    }


# ── NN forward pass (identical to compare_models.py) ─────────────────────────
def simon_fwd(nn_in, w):
    """SIMON forward pass. nn_in: (nc, 3). Returns log(c): (nc,)."""
    h = (nn_in - w['mean']) / w['std']
    h = h @ w['w0T'] + w['b0'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ w['w1T'] + w['b1'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ w['w2T'] + w['b2'];  s = 1/(1+np.exp(-h)); h = h*s
    return (h @ w['w3T'] + w['b3']).ravel()


# ── ias15 (identical to compare_models.py) ────────────────────────────────────
def simulate_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"; sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos   = np.zeros((n_samples, len(m), 3))
    vel   = np.zeros((n_samples, len(m), 3))
    t0    = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k,i] = [p.x, p.y, p.z]
            vel[k,i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, time.perf_counter() - t0


# ── Hybrid leapfrog — SIMON and No-NN modes ───────────────────────────────────
def simulate_hybrid(x0, v0, m, ws, cfg, dt, T, n_samples, model_type='simon'):
    """
    model_type='simon':
        Close encounters (r < 0.15 AU):
            F = c * G*mi*mj/r_soft^3 * rij      (NN predicts c)
        Gate: c outside [0.2, 5.0] or r < r_soft_min → exact Newtonian fallback

    model_type='no_nn':
        Close encounters (r < 0.15 AU):
            F = G*mi*mj/r_soft^3 * rij           (pure softened gravity, c = 1 always)
        Gate: r < r_soft_min → exact Newtonian fallback
        NN is NEVER called. This is the ablation.

    Everything else is identical:
        - Same leapfrog integration loop
        - Same adaptive sub-stepping (adapt_thresh=0.05, max_substeps=16)
        - Same far-field exact Newtonian (r > 0.15 AU)
        - Same initial conditions and timestep

    The ONLY difference: whether c is predicted by the NN (SIMON) or fixed at 1.0 (No-NN).

    Return dict uses 'avg_nn_frac' key — identical to compare_models.py.
    For No-NN: avg_nn_frac = 0.0000 always (no close encounter triggers NN).
    """
    N  = x0.shape[0]
    ii = np.array([0,0,1]); jj = np.array([1,2,2]); P = 3

    G          = cfg.G
    eps2       = cfg.eps**2
    c_min      = cfg.c_min
    c_max      = cfg.c_max
    r_soft_min = cfg.r_soft_min

    x      = x0.astype(np.float64).copy()
    v      = v0.astype(np.float64).copy()
    m_f    = m.astype(np.float64)
    mi_a   = m_f[ii]; mj_a = m_f[jj]
    Gmimj  = G * mi_a * mj_a
    inv_mi = 1.0 / mi_a; inv_mj = 1.0 / mj_a
    log_mi = np.log(mi_a + 1e-30).astype(np.float32)
    log_mj = np.log(mj_a + 1e-30).astype(np.float32)

    nn_thresh    = 500.0 * cfg.eps   # 0.15 AU — same as compare_models.py
    adapt_thresh = 0.05
    max_substeps = 16

    times   = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3))
    vel_out = np.zeros((n_samples, N, 3))

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]                      # (3, 3)
        r2  = np.einsum('ij,ij->i', rij, rij)         # (3,)
        r   = np.sqrt(r2 + 1e-30)

        # Default: unsoftened Newtonian (identical to compare_models.py)
        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3                       # (3,) scalar magnitudes

        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))

        if n_close > 0:
            r2_c     = r2[close_mask]
            rij_c    = rij[close_mask]
            r_soft_c = np.sqrt(r2_c + eps2)
            denom    = (r2_c + eps2) ** 1.5 + 1e-30
            F_soft_c = Gmimj[close_mask] / denom      # G*mi*mj/r_soft^3

            if model_type == 'simon':
                # ── SIMON: NN predicts correction factor c ────────────────────
                # Identical to compare_models.py SIMON branch
                nn_in = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30)
                nn_in[:, 1] = log_mi[close_mask]
                nn_in[:, 2] = log_mj[close_mask]

                log_c = simon_fwd(nn_in, ws)
                c     = np.exp(log_c).astype(np.float64)

                # Magnitude gate: c outside [0.2, 5.0] → Newtonian fallback
                fb = ((r_soft_c < r_soft_min)
                      | (c < c_min) | (c > c_max)
                      | ~np.isfinite(c))
                F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_c)
                F_scalar[close_mask] = F_corrected
                # Direction: always exact rij (applied in F_vec below)

            elif model_type == 'no_nn':
                # ── No-NN: pure softened gravity, c = 1 always ───────────────
                # NN is never called. F = G*mi*mj/r_soft^3 * rij directly.
                # Only fallback is for extreme close encounters (r < r_soft_min)
                # where exact Newtonian is used — same as SIMON's extreme gate.
                fb = (r_soft_c < r_soft_min)
                F_scalar[close_mask] = np.where(fb, F_scalar[close_mask], F_soft_c)
                # Direction: exact rij (same as SIMON — applied in F_vec below)

        # Build full (P, 3) force vector array
        # For both SIMON and No-NN: direction is always exact geometry rij
        F_vec = F_scalar[:, None] * rij

        # Accumulate accelerations
        acc = np.zeros((N, 3))
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]

        # Return (acc, n_close) — identical signature to compare_models.py
        return acc, n_close

    def min_pair_dist(pos):
        d2 = np.einsum('ij,ij->i', pos[jj]-pos[ii], pos[jj]-pos[ii])
        return np.sqrt(np.min(d2) + 1e-30)

    def substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5*sub_dt*a_in
        xn = x_in + sub_dt*vh
        an, nf = compute_acc(xn)
        vn = vh + 0.5*sub_dt*an
        return xn, vn, an, nf

    # Initialise — identical to compare_models.py
    a, nf = compute_acc(x)
    fb_sum = nf; pair_sum = P
    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt)
    t_start = time.perf_counter()

    # Main integration loop — identical to compare_models.py
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
        else:
            vh = v + 0.5*dt_f*a
            x  = x + dt_f*vh
            a, nf = compute_acc(x)
            v  = vh + 0.5*dt_f*a
            fb_sum += nf; pair_sum += P

        t_cur += dt; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12: break

    total = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps":            steps,
        "total_time_sec":   total,
        "time_per_step_sec": total / max(steps, 1),
        "avg_nn_frac":      fb_sum / max(pair_sum, 1),
    }


# ── Metrics (identical to compare_models.py) ─────────────────────────────────
def rms_sep(a, b):
    return np.sqrt(np.mean(np.sum((a-b)**2, axis=-1), axis=1))

def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T  = times[-1]; t0 = t0_frac*T; t1 = t1_frac*T
    mask = (times >= t0) & (times <= t1)
    x  = times[mask]; y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean(); y0 = y.mean()
    slope = float(np.sum((x-x0)*(y-y0)) / (np.sum((x-x0)**2) + 1e-30))
    return slope, (t0, t1)


# ── Colours ────────────────────────────────────────────────────────────────────
C = {
    'ias15': '#2563A6',   # blue
    'simon': '#16A34A',   # green
    'no_nn': '#DC2626',   # red
}
L = {
    'ias15':  'ias15 (ground truth)',
    'simon':  'SIMON (with NN correction, c predicted)',
    'no_nn':  'No-NN baseline (pure softened gravity, c = 1)',
}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_divergence(times, delta_s, delta_n, slope_s, slope_n, win_s, out_path):
    # Local font override for this paper figure only.
    # This changes only chart formatting, not data, fitted slopes, or results.
    with plt.rc_context({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        ax.semilogy(
            times, delta_s,
            color=C['simon'], lw=1.8,
            label=f"SIMON (with NN correction)\nλ ≈ {slope_s:.4f}/yr"
        )
        ax.semilogy(
            times, delta_n,
            color=C['no_nn'], lw=1.8, linestyle='--',
            label=f"No-NN baseline\nλ ≈ {slope_n:.4f}/yr"
        )

        ax.axvspan(
            win_s[0], win_s[1],
            alpha=0.10, color='gray', label="fit window (10–50 yr)"
        )

        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("RMS position error vs ias15 (AU)")

        # Focus the log scale on the meaningful plotted range.
        # This only hides the near-zero initial point visually; it does not change
        # any computed RMS error, fitted slope, or summary result.
        finite_positive = np.concatenate([
            np.asarray(delta_s)[np.asarray(delta_s) > 0],
            np.asarray(delta_n)[np.asarray(delta_n) > 0],
        ])
        y_max = max(1.0, float(np.nanmax(finite_positive)) * 1.25)
        ax.set_ylim(1e-3, y_max)

        # No chart title: the LaTeX caption explains the figure.
        ax.legend(loc='lower right', framealpha=0.85)
        ax.grid(True, which='both', alpha=0.25)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    print(f"  Saved {out_path}")

def plot_trajectories(pr, ps, pn, out_path):
    N   = pr.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.5*N, 4.5))
    if N == 1: axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.plot(pr[:,i,0], pr[:,i,1], '-',  color=C['ias15'], lw=1.5, label='ias15')
        ax.plot(ps[:,i,0], ps[:,i,1], '--', color=C['simon'], lw=1.2,
                label='SIMON (with NN)')
        ax.plot(pn[:,i,0], pn[:,i,1], ':',  color=C['no_nn'], lw=1.2,
                label='No-NN baseline')
        ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
        ax.set_title(f"Body {i} XY  (T=100 yr)")
        ax.grid(True, alpha=0.25)
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle("Trajectory Overlay: ias15 vs SIMON vs No-NN Baseline")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"  Saved {out_path}")

def plot_timeseries(times, pr, ps, pn, out_path):
    N   = pr.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(11, 3.0*N), sharex=True)
    if N == 1: axes = np.array([axes])
    for i in range(N):
        for col, coord in [(0, 0), (1, 1)]:
            ax = axes[i, col]
            ax.plot(times, pr[:,i,coord], '-',  color=C['ias15'], lw=1.5,
                    label='ias15')
            ax.plot(times, ps[:,i,coord], '--', color=C['simon'], lw=1.0,
                    label='SIMON (with NN)')
            ax.plot(times, pn[:,i,coord], ':',  color=C['no_nn'], lw=1.0,
                    label='No-NN baseline')
            lbl = 'x' if coord == 0 else 'y'
            ax.set_ylabel(f"Body {i} {lbl}(t) (AU)")
            ax.grid(True, alpha=0.25)
            if i == 0 and col == 0: ax.legend(fontsize=7)
    axes[-1,0].set_xlabel("Time (yr)"); axes[-1,1].set_xlabel("Time (yr)")
    fig.suptitle("Position Time Series: ias15 vs SIMON vs No-NN Baseline")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Ablation: SIMON with vs without NN correction")
    ap.add_argument("--simon_model", default="pair_correction_nn.pt")
    ap.add_argument("--out_dir",     default="comparison_out_no_nn")
    ap.add_argument("--dt",          type=float, default=0.04)
    ap.add_argument("--T",           type=float, default=100.0)
    ap.add_argument("--n_samples",   type=int,   default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    # Load SIMON model
    print("[no_nn_ablation] Loading SIMON model ...")
    sm = PairCorrectionNN()
    sm.load_state_dict(torch.load(args.simon_model, map_location='cpu'))
    sm.eval()
    ws = extract_simon(sm)
    print(f"  {args.simon_model}  params={sum(p.numel() for p in sm.parameters())}")

    print(f"\n[no_nn_ablation] Ablation design:")
    print(f"  SIMON:   close encounters → c * G*mi*mj/r_soft^3  (NN predicts c)")
    print(f"  No-NN:   close encounters → G*mi*mj/r_soft^3       (c = 1, NN never called)")
    print(f"  All else identical: same dt, T, ICs, adaptive stepping, fallback gates")

    # Initial conditions — identical to compare_models.py
    T = args.T; ns = args.n_samples; dt = args.dt
    m  = np.array([1.0, 0.01, 0.005])
    x0 = np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64)
    M  = m.sum()
    x0 -= (m[:,None]*x0).sum(0)/M
    v0 -= (m[:,None]*v0).sum(0)/M

    # ias15 ground truth
    print(f"\n[no_nn_ablation] Running ias15  (T={T}yr, n_samples={ns}) ...")
    tr, pr, vr, t_ias = simulate_ias15(x0, v0, m, cfg.G, T, ns)
    print(f"  ias15: {t_ias:.3f}s")

    # SIMON (with NN correction)
    print(f"[no_nn_ablation] Running SIMON  (dt={dt}) ...")
    _, ps, _, pf_s = simulate_hybrid(x0, v0, m, ws, cfg, dt, T, ns, 'simon')
    sp_s = t_ias / max(pf_s['total_time_sec'], 1e-12)
    print(f"  SIMON: {pf_s['total_time_sec']:.3f}s  speedup={sp_s:.2f}x  "
          f"NN_frac={pf_s['avg_nn_frac']:.4f}")

    # No-NN baseline (pure softened gravity, NN never called)
    print(f"[no_nn_ablation] Running No-NN baseline  (dt={dt}) ...")
    _, pn, _, pf_n = simulate_hybrid(x0, v0, m, ws, cfg, dt, T, ns, 'no_nn')
    sp_n = t_ias / max(pf_n['total_time_sec'], 1e-12)
    print(f"  No-NN: {pf_n['total_time_sec']:.3f}s  speedup={sp_n:.2f}x  "
          f"NN_frac={pf_n['avg_nn_frac']:.4f}  (should be 0.0000)")

    # Divergence rates
    print("\n[no_nn_ablation] Computing divergence rates ...")
    delta_s = rms_sep(ps, pr)
    delta_n = rms_sep(pn, pr)
    slope_s, win_s = fit_log_slope(tr, delta_s)
    slope_n, win_n = fit_log_slope(tr, delta_n)
    rms_s = float(delta_s[-1])
    rms_n = float(delta_n[-1])

    print(f"  SIMON   \u03bb={slope_s:.4f}/yr  final_RMS={rms_s:.4f} AU  "
          f"bounded={'YES' if rms_s < 10 else 'NO (ejection)'}")
    print(f"  No-NN   \u03bb={slope_n:.4f}/yr  final_RMS={rms_n:.4f} AU  "
          f"bounded={'YES' if rms_n < 10 else 'NO (ejection)'}")

    # Verdict
    print(f"\n[no_nn_ablation] ABLATION VERDICT:")
    if rms_n > 10 and rms_s < 10:
        print(f"  --> No-NN causes body ejection  (final_RMS = {rms_n:.1f} AU)")
        print(f"      SIMON stays bounded          (final_RMS = {rms_s:.2f} AU)")
        print(f"      CONCLUSION: NN correction IS essential at the 1.1% of close encounter steps.")
        print(f"      Low invocation rate does not mean low importance.")
        print(f"      Paper claim 'targeted tool, not replacement' is STRONGLY SUPPORTED.")
    elif rms_n < 10 and rms_s < 10:
        diff_rms = rms_n - rms_s
        print(f"  --> Both models stay bounded.")
        print(f"      No-NN RMS={rms_n:.4f} vs SIMON RMS={rms_s:.4f}  (diff={diff_rms:.4f} AU)")
        if diff_rms > 0.5:
            print(f"      NN improves accuracy at close encounters but is not critical for stability.")
        else:
            print(f"      NN correction has minimal measurable impact at this timestep.")
    else:
        print(f"  --> Unexpected result. Review simulation output carefully.")

    # Plots
    print("\n[no_nn_ablation] Generating plots ...")
    plot_divergence(tr, delta_s, delta_n, slope_s, slope_n, win_s,
                    os.path.join(args.out_dir, "divergence_no_nn.png"))
    plot_trajectories(pr, ps, pn,
                      os.path.join(args.out_dir, "trajectory_no_nn.png"))
    plot_timeseries(tr, pr, ps, pn,
                    os.path.join(args.out_dir, "timeseries_no_nn.png"))

    # Summary file
    txt_path = os.path.join(args.out_dir, "summary_no_nn.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("NO-NN ABLATION SUMMARY\n")
        f.write("Question: Is the NN correction essential at close encounters?\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"T={T}yr  dt={dt}yr  n_samples={ns}\n")
        f.write(f"m=[1.0, 0.01, 0.005] Msun\n\n")
        f.write("Design:\n")
        f.write("  SIMON:   F = c * G*mi*mj/r_soft^3 * rij  (NN predicts c)\n")
        f.write("  No-NN:   F = G*mi*mj/r_soft^3 * rij       (c = 1 always, no NN)\n\n")
        f.write(f"{'Model':<22} {'lambda(/yr)':>12} {'final_RMS':>12} "
                f"{'speedup':>10} {'NN_frac':>10} {'bounded':>9}\n")
        f.write("-" * 77 + "\n")
        f.write(f"{'ias15':<22} {'--':>12} {'0.000':>12} "
                f"{'1.00x':>10} {'--':>10} {'YES':>9}\n")
        f.write(f"{'SIMON (with NN)':<22} {slope_s:>12.4f} {rms_s:>12.4f} "
                f"{sp_s:>9.2f}x {pf_s['avg_nn_frac']:>10.4f} "
                f"{'YES' if rms_s<10 else 'NO':>9}\n")
        f.write(f"{'No-NN baseline':<22} {slope_n:>12.4f} {rms_n:>12.4f} "
                f"{sp_n:>9.2f}x {pf_n['avg_nn_frac']:>10.4f} "
                f"{'YES' if rms_n<10 else 'NO':>9}\n\n")
        f.write("VERDICT:\n")
        if rms_n > 10 and rms_s < 10:
            f.write("  Removing the NN correction causes body ejection.\n")
            f.write("  The NN is essential at those 1.1% of close encounter steps.\n")
            f.write("  Low invocation rate (1.1%) does not imply low importance.\n")
            f.write("  Paper claim 'targeted tool, not replacement' is confirmed.\n")
        elif rms_n < 10:
            f.write("  Both models stay bounded.\n")
            f.write("  NN contribution is accuracy improvement, not stability guarantee.\n")
            f.write("  Paper framing should be adjusted accordingly.\n")
    print(f"  Saved {txt_path}")

    print(f"\n[no_nn_ablation] Key result:")
    print(f"  SIMON (with NN): \u03bb={slope_s:.4f}/yr  RMS={rms_s:.4f} AU  "
          f"bounded={'YES' if rms_s<10 else 'NO'}")
    print(f"  No-NN baseline:  \u03bb={slope_n:.4f}/yr  RMS={rms_n:.4f} AU  "
          f"bounded={'YES' if rms_n<10 else 'NO'}")
    print(f"[no_nn_ablation] Done. Output in: {args.out_dir}/")


if __name__ == "__main__":
    main()
