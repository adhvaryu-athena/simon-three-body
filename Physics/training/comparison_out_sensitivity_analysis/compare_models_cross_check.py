# compare_models.py  (final — with correct direction gate for Vector NN)
#
# Runs ias15, SIMON, and Vector NN through identical 100-year simulations.
#
# GATING PHILOSOPHY (both models use same principle):
#   SIMON:     if c (magnitude correction) outside [0.2, 5.0] → exact Newtonian fallback
#   Vector NN: if cos(angle between geometric dir and predicted dir) < 0.866
#              (i.e. direction deviates > 30°) → exact Newtonian fallback
#
#   Why NOT a force-magnitude gate for Vector NN:
#   Vector NN magnitude is ALWAYS analytic (G*mi*mj/r_soft^3 * r), never from the NN.
#   A ratio check (predicted/analytic) would always be ~1.0 and never trigger.
#   The meaningful quantity to gate is direction quality, not magnitude.
#
# REQUIRES in the same folder:
#   pair_correction_nn.pt  (SIMON)
#   vector_nn.pt           (Vector NN)
#
# PRODUCES in comparison_out/:
#   divergence_3model.png
#   trajectory_3model.png
#   timeseries_3model.png
#   summary_comparison.txt
#
# Run: python compare_models.py

import os, time, math, argparse
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
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


# ── Models ────────────────────────────────────────────────────────────────────
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


class VectorNN(nn.Module):
    """Vector NN: 7->32->32->32->3, predicts unit direction vector."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer('input_mean', torch.zeros(7))
        self.register_buffer('input_std',  torch.ones(7))
    def forward(self, x):
        return F.normalize(
            self.net((x - self.input_mean) / (self.input_std + 1e-8)),
            dim=-1, eps=1e-8)


@dataclass
class HybridConfig:
    G: float          = 1.0
    eps: float        = 3e-4
    c_min: float      = 0.2     # SIMON magnitude gate lower bound
    c_max: float      = 5.0     # SIMON magnitude gate upper bound
    r_soft_min: float = 5e-4    # extreme-close fallback radius
    dir_cos_min: float = 0.9999985   # cos(0.1°) — tighter than NN's 0.04° accuracy
                                # if predicted direction deviates > 30° from
                                # geometric direction → fallback to exact Newtonian


# ── Weight extraction ─────────────────────────────────────────────────────────
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

def extract_vector(model):
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


# ── Numpy inference ───────────────────────────────────────────────────────────
def simon_fwd(nn_in, w):
    """SIMON forward pass. nn_in: (nc, 3). Returns log(c): (nc,)."""
    h = (nn_in - w['mean']) / w['std']
    h = h @ w['w0T'] + w['b0'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ w['w1T'] + w['b1'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ w['w2T'] + w['b2'];  s = 1/(1+np.exp(-h)); h = h*s
    return (h @ w['w3T'] + w['b3']).ravel()

def vector_fwd(nn_in, wv):
    """Vector NN forward pass. nn_in: (nc, 7). Returns unit dirs: (nc, 3)."""
    h = (nn_in - wv['mean']) / wv['std']
    h = h @ wv['w0T'] + wv['b0'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ wv['w1T'] + wv['b1'];  s = 1/(1+np.exp(-h)); h = h*s
    h = h @ wv['w2T'] + wv['b2'];  s = 1/(1+np.exp(-h)); h = h*s
    out   = h @ wv['w3T'] + wv['b3']              # (nc, 3)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-8)).astype(np.float64)


# ── ias15 ground truth ────────────────────────────────────────────────────────
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


# ── Hybrid leapfrog (SIMON or Vector NN) ─────────────────────────────────────
def simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, n_samples, model_type='simon'):
    """
    model_type='simon':
        Close pairs: F = c * G*mi*mj/r_soft^3 * rij
        Direction: always exact geometry (rij/|rij|)
        Magnitude gate: c outside [c_min, c_max] → Newtonian fallback

    model_type='vector':
        Close pairs: F = G*mi*mj/r_soft^3 * r * dirs_pred
        Direction: from Vector NN (may have small residual error ~0.04°)
        Magnitude: always analytic (no NN involvement)
        Direction gate: cos(geometric_dir, predicted_dir) < 0.866
                        (> 30° deviation) → Newtonian fallback
        This gate is structurally equivalent to SIMON's magnitude gate:
        both ask "is the NN output physically reasonable?" and fall back
        to exact Newtonian if not.
    """
    N  = x0.shape[0]
    ii = np.array([0,0,1]); jj = np.array([1,2,2]); P = 3

    G          = cfg.G
    eps2       = cfg.eps**2
    c_min      = cfg.c_min
    c_max      = cfg.c_max
    r_soft_min = cfg.r_soft_min
    dir_cos_min= cfg.dir_cos_min   # = 0.866, i.e. cos(30°)

    x      = x0.astype(np.float64).copy()
    v      = v0.astype(np.float64).copy()
    m_f    = m.astype(np.float64)
    mi_a   = m_f[ii]; mj_a = m_f[jj]
    Gmimj  = G * mi_a * mj_a
    inv_mi = 1.0 / mi_a; inv_mj = 1.0 / mj_a
    log_mi = np.log(mi_a + 1e-30).astype(np.float32)
    log_mj = np.log(mj_a + 1e-30).astype(np.float32)

    nn_thresh    = 500.0 * cfg.eps   # ~0.15 AU
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

        # Default: unsoftened Newtonian
        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3                       # (3,) scalar magnitudes

        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))

        if n_close > 0:
            r2_c     = r2[close_mask]
            r_c      = r[close_mask]
            rij_c    = rij[close_mask]
            r_soft_c = np.sqrt(r2_c + eps2)
            denom    = (r2_c + eps2) ** 1.5 + 1e-30
            F_soft_c = Gmimj[close_mask] / denom      # G*mi*mj/r_soft^3

            if model_type == 'simon':
                # ── SIMON: scalar correction factor ───────────────────────────
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

            elif model_type == 'vector':
                # ── Vector NN: unit direction prediction ─────────────────────
                # Geometric unit direction (exact, used as gate reference)
                dirs_c = rij_c / (r_c[:, None] + 1e-30)  # (nc, 3) exact dirs

                nn_in_v = np.empty((n_close, 7), dtype=np.float32)
                nn_in_v[:, 0] = dirs_c[:, 0]
                nn_in_v[:, 1] = dirs_c[:, 1]
                nn_in_v[:, 2] = dirs_c[:, 2]
                nn_in_v[:, 3] = np.log(r_soft_c + 1e-30)
                nn_in_v[:, 4] = np.log(r_c + 1e-30)
                nn_in_v[:, 5] = log_mi[close_mask]
                nn_in_v[:, 6] = log_mj[close_mask]

                dirs_pred = vector_fwd(nn_in_v, wv)    # (nc, 3) predicted unit dirs

                # Analytic force magnitude (no NN involvement in magnitude)
                F_mag_c = F_soft_c * r_c               # G*mi*mj*r/r_soft^3

                # Direction gate — equivalent to SIMON's c_min/c_max gate.
                # Checks whether NN direction is physically reasonable.
                # cos(geometric, predicted) < 0.866 means > 30° deviation → fallback.
                cos_sim = np.sum(dirs_c * dirs_pred, axis=1)  # dot product
                bad_dir = cos_sim < dir_cos_min

                fb = ((r_soft_c < r_soft_min)
                      | ~np.all(np.isfinite(dirs_pred), axis=1)
                      | bad_dir)

                # Force vectors for close pairs
                F_vec_vnn = np.where(
                    fb[:, None],
                    F_scalar[close_mask, None] * rij_c,  # Newtonian fallback
                    F_mag_c[:, None] * dirs_pred           # VectorNN direction
                )

        # Build full (P, 3) force vector array
        F_vec = F_scalar[:, None] * rij   # correct for far-field + SIMON close pairs

        if n_close > 0 and model_type == 'vector':
            F_vec[close_mask] = F_vec_vnn  # override close pairs with VectorNN

        # Accumulate accelerations
        acc = np.zeros((N, 3))
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
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

    # Initialise
    a, nf = compute_acc(x)
    fb_sum = nf; pair_sum = P
    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt)
    t_start = time.perf_counter()

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
        "steps": steps, "total_time_sec": total,
        "time_per_step_sec": total / max(steps, 1),
        "avg_nn_frac": fb_sum / max(pair_sum, 1),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def rms_sep(a, b):
    return np.sqrt(np.mean(np.sum((a-b)**2, axis=-1), axis=1))

def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T  = times[-1]; t0 = t0_frac*T; t1 = t1_frac*T
    mask = (times >= t0) & (times <= t1)
    x  = times[mask]; y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean(); y0 = y.mean()
    slope = float(np.sum((x-x0)*(y-y0)) / (np.sum((x-x0)**2) + 1e-30))
    return slope, (t0, t1)


# ── Plots ─────────────────────────────────────────────────────────────────────
C = {'ias15': '#2563A6', 'simon': '#16A34A', 'vector': '#DC2626'}
L = {
    'ias15':  'ias15 (ground truth)',
    'simon':  'SIMON (magnitude NN, analytic direction)',
    'vector': 'Vector NN (analytic magnitude, NN direction)',
}

def plot_divergence(times, deltas, slopes, windows, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in ('simon', 'vector'):
        ax.semilogy(times, deltas[k], color=C[k], lw=1.8,
                    label=f"{L[k]}\n  λ ≈ {slopes[k]:.4f}/yr")
    ax.axvspan(windows['simon'][0], windows['simon'][1],
               alpha=0.10, color='gray', label="fit window (10–50 yr)")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    ax.set_title("Divergence Rate: SIMON vs Vector NN")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  Saved {path}")

def plot_trajectories(pr, ps, pv, path):
    N   = pr.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.5*N, 4.5))
    if N == 1: axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.plot(pr[:,i,0], pr[:,i,1], '-',  color=C['ias15'],  lw=1.5, label='ias15')
        ax.plot(ps[:,i,0], ps[:,i,1], '--', color=C['simon'],  lw=1.2, label='SIMON')
        ax.plot(pv[:,i,0], pv[:,i,1], ':',  color=C['vector'], lw=1.2, label='Vector NN')
        ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
        ax.set_title(f"Body {i} XY  (T=100 yr)")
        ax.grid(True, alpha=0.25)
        if i == 0: ax.legend(fontsize=8)
    fig.suptitle("Trajectory Overlay: ias15 vs SIMON vs Vector NN")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  Saved {path}")

def plot_timeseries(times, pr, ps, pv, path):
    N   = pr.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(11, 3.0*N), sharex=True)
    if N == 1: axes = np.array([axes])
    for i in range(N):
        for col, coord in [(0, 0), (1, 1)]:
            ax = axes[i, col]
            ax.plot(times, pr[:,i,coord], '-',  color=C['ias15'],  lw=1.5, label='ias15')
            ax.plot(times, ps[:,i,coord], '--', color=C['simon'],  lw=1.0, label='SIMON')
            ax.plot(times, pv[:,i,coord], ':',  color=C['vector'], lw=1.0, label='Vector NN')
            lbl = 'x' if coord == 0 else 'y'
            ax.set_ylabel(f"Body {i} {lbl}(t) (AU)")
            ax.grid(True, alpha=0.25)
            if i == 0 and col == 0: ax.legend(fontsize=7)
    axes[-1,0].set_xlabel("Time (yr)"); axes[-1,1].set_xlabel("Time (yr)")
    fig.suptitle("Position Time Series: ias15 vs SIMON vs Vector NN")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simon_model",  default="pair_correction_nn.pt")
    ap.add_argument("--vector_model", default="vector_nn.pt")
    ap.add_argument("--out_dir",      default="comparison_out")
    ap.add_argument("--dt",           type=float, default=0.04)
    ap.add_argument("--T",            type=float, default=100.0)
    ap.add_argument("--n_samples",    type=int,   default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    # Load models
    print("[compare] Loading SIMON ...")
    sm = PairCorrectionNN(); sm.load_state_dict(torch.load(args.simon_model, map_location='cpu')); sm.eval()
    ws = extract_simon(sm)
    print(f"  {args.simon_model}  params={sum(p.numel() for p in sm.parameters())}")

    print("[compare] Loading Vector NN ...")
    vm = VectorNN(); vm.load_state_dict(torch.load(args.vector_model, map_location='cpu')); vm.eval()
    wv = extract_vector(vm)
    print(f"  {args.vector_model}  params={sum(p.numel() for p in vm.parameters())}")

    print(f"\n[compare] Gates:")
    print(f"  SIMON:     c outside [{cfg.c_min}, {cfg.c_max}] → Newtonian fallback")
    print(f"  Vector NN: cos(geometric, predicted) < {cfg.dir_cos_min} "
          f"(> {int(np.degrees(np.arccos(cfg.dir_cos_min)))}°) → Newtonian fallback")

    # Initial conditions — identical to pair_eval_after_adaptive.py
    T = args.T; ns = args.n_samples; dt = args.dt
    m  = np.array([1.0, 0.01, 0.005])
    x0 = np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64)
    M  = m.sum(); x0 -= (m[:,None]*x0).sum(0)/M; v0 -= (m[:,None]*v0).sum(0)/M

    # Run simulations
    print(f"\n[compare] ias15 ...")
    tr, pr, vr, t_ias = simulate_ias15(x0, v0, m, cfg.G, T, ns)
    print(f"  {t_ias:.3f}s")

    print(f"[compare] SIMON (dt={dt}) ...")
    _, ps, _, pf_s = simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, ns, 'simon')
    sp_s = t_ias / max(pf_s['total_time_sec'], 1e-12)
    print(f"  {pf_s['total_time_sec']:.3f}s  speedup={sp_s:.2f}x  "
          f"NN_frac={pf_s['avg_nn_frac']:.4f}")

    print(f"[compare] Vector NN (dt={dt}) ...")
    _, pv, _, pf_v = simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, ns, 'vector')
    sp_v = t_ias / max(pf_v['total_time_sec'], 1e-12)
    print(f"  {pf_v['total_time_sec']:.3f}s  speedup={sp_v:.2f}x  "
          f"NN_frac={pf_v['avg_nn_frac']:.4f}")

    # Divergence rates
    print("\n[compare] Divergence rates ...")
    delta_s = rms_sep(ps, pr); delta_v = rms_sep(pv, pr)
    slope_s, win_s = fit_log_slope(tr, delta_s)
    slope_v, win_v = fit_log_slope(tr, delta_v)
    rms_s = float(delta_s[-1]); rms_v = float(delta_v[-1])

    print(f"  SIMON     λ={slope_s:.4f}/yr  final_RMS={rms_s:.4f} AU  "
          f"bounded={'YES' if rms_s < 10 else 'NO (ejection)'}")
    print(f"  Vector NN λ={slope_v:.4f}/yr  final_RMS={rms_v:.4f} AU  "
          f"bounded={'YES' if rms_v < 10 else 'NO (ejection)'}")

    diff = slope_v - slope_s
    if rms_v > 10 and rms_s < 10:
        print(f"\n  --> Vector NN ejected a body (final_RMS={rms_v:.1f} AU)")
        print(f"      Direction gate did not prevent ejection.")
        print(f"      Consider tightening dir_cos_min (e.g. 0.95 = 18°).")
    elif diff > 0.005:
        print(f"\n  --> λ(VectorNN) > λ(SIMON) by {diff:.4f}/yr")
        print(f"      Analytic direction reduces error-growth rate.")
    else:
        print(f"\n  --> λ values nearly identical (diff={diff:.4f}/yr)")
        print(f"      Both models exhibit the same chaotic divergence rate.")
        print(f"      Stability difference is in final_RMS, not in λ.")

    # Plots
    print("\n[compare] Generating plots ...")
    plot_divergence(tr,
        {'simon': delta_s, 'vector': delta_v},
        {'simon': slope_s, 'vector': slope_v},
        {'simon': win_s,   'vector': win_v},
        os.path.join(args.out_dir, "divergence_3model.png"))
    plot_trajectories(pr, ps, pv, os.path.join(args.out_dir, "trajectory_3model.png"))
    plot_timeseries(tr, pr, ps, pv, os.path.join(args.out_dir, "timeseries_3model.png"))

    # Summary
    path_txt = os.path.join(args.out_dir, "summary_comparison.txt")
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n3-MODEL COMPARISON SUMMARY\n" + "=" * 65 + "\n\n")
        f.write(f"T={T}yr  dt={dt}yr  n_samples={ns}\n")
        f.write(f"m=[1.0, 0.01, 0.005] Msun\n\n")
        f.write(f"Gates:\n")
        f.write(f"  SIMON:     c outside [{cfg.c_min}, {cfg.c_max}] → Newtonian\n")
        f.write(f"  Vector NN: cos(dir) < {cfg.dir_cos_min} (>{int(np.degrees(np.arccos(cfg.dir_cos_min)))}°) → Newtonian\n\n")
        f.write(f"{'Model':<22} {'lambda(/yr)':>12} {'final_RMS':>12} "
                f"{'speedup':>10} {'NN_frac':>10} {'bounded':>9}\n")
        f.write("-" * 77 + "\n")
        f.write(f"{'ias15':<22} {'--':>12} {'0.000':>12} {'1.00x':>10} {'--':>10} {'YES':>9}\n")
        f.write(f"{'SIMON':<22} {slope_s:>12.4f} {rms_s:>12.4f} "
                f"{sp_s:>9.2f}x {pf_s['avg_nn_frac']:>10.4f} "
                f"{'YES' if rms_s<10 else 'NO':>9}\n")
        f.write(f"{'Vector NN':<22} {slope_v:>12.4f} {rms_v:>12.4f} "
                f"{sp_v:>9.2f}x {pf_v['avg_nn_frac']:>10.4f} "
                f"{'YES' if rms_v<10 else 'NO':>9}\n")
    print(f"  Saved {path_txt}")
    print(f"\n[compare] Key result: λ_SIMON={slope_s:.4f}/yr  λ_VectorNN={slope_v:.4f}/yr")
    print(f"          final_RMS_SIMON={rms_s:.4f} AU  final_RMS_VectorNN={rms_v:.4f} AU")


if __name__ == "__main__":
    main()
