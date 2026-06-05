# compare_models.py
# Runs ias15, SIMON, and Vector NN through identical 100-year simulations
# and compares divergence rates, trajectories, and speed-accuracy frontier.
#
# REQUIRES in the same folder:
#   pair_correction_nn.pt   -- SIMON model (from train_pair_correction_new.py)
#   vector_nn.pt            -- Vector NN model (from train_vector_nn.py)
#
# PRODUCES:
#   divergence_3model.png          -- main comparison: lambda for each model
#   trajectory_3model.png          -- XY trajectories for all 3 models
#   timeseries_3model.png          -- x(t), y(t) for all 3 models
#   summary_comparison.txt         -- lambda values, RMS, speedups
#
# SCIENTIFIC QUESTION BEING ANSWERED:
#   Does enforcing analytic force direction (SIMON) give lower divergence rate
#   than learning direction from data (Vector NN)?
#   If lambda(VectorNN) > lambda(SIMON): analytic direction matters.
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
    print("[ERROR] rebound not found. Install it or ensure it is in your environment.")
    raise


# ── SIMON model (PairCorrectionNN) ────────────────────────────────────────────
class PairCorrectionNN(nn.Module):
    """Scalar correction factor: 3->32->32->32->1 with SiLU."""
    def __init__(self, hidden=32, p_drop=0.0):
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


# ── Vector NN model ───────────────────────────────────────────────────────────
class VectorNN(nn.Module):
    """Unit direction predictor: 7->32->32->32->3 with SiLU + F.normalize."""
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
        raw = self.net((x - self.input_mean) / (self.input_std + 1e-8))
        return F.normalize(raw, dim=-1, eps=1e-8)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


# ── Weight extraction: SIMON ──────────────────────────────────────────────────
def extract_weights_simon(model):
    """Pre-transpose SIMON weights for fast numpy matmul."""
    sd = model.state_dict()
    return {
        'mean': sd['input_mean'].cpu().numpy().astype(np.float32),
        'std':  sd['input_std'].cpu().numpy().astype(np.float32) + 1e-8,
        'w0T':  sd['net.0.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (3, H)
        'b0':   sd['net.0.bias'].cpu().numpy().astype(np.float32),
        'w1T':  sd['net.2.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, H)
        'b1':   sd['net.2.bias'].cpu().numpy().astype(np.float32),
        'w2T':  sd['net.4.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b2':   sd['net.4.bias'].cpu().numpy().astype(np.float32),
        'w3T':  sd['net.6.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, 1)
        'b3':   sd['net.6.bias'].cpu().numpy().astype(np.float32),
    }


# ── Weight extraction: Vector NN ──────────────────────────────────────────────
def extract_weights_vector(model):
    """Pre-transpose VectorNN weights for fast numpy matmul."""
    sd = model.state_dict()
    return {
        'mean': sd['input_mean'].cpu().numpy().astype(np.float32),
        'std':  sd['input_std'].cpu().numpy().astype(np.float32) + 1e-8,
        'w0T':  sd['net.0.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (7, H)
        'b0':   sd['net.0.bias'].cpu().numpy().astype(np.float32),
        'w1T':  sd['net.2.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, H)
        'b1':   sd['net.2.bias'].cpu().numpy().astype(np.float32),
        'w2T':  sd['net.4.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b2':   sd['net.4.bias'].cpu().numpy().astype(np.float32),
        'w3T':  sd['net.6.weight'].cpu().numpy().T.astype(np.float32).copy(),  # (H, 3)
        'b3':   sd['net.6.bias'].cpu().numpy().astype(np.float32),
    }


# ── Numpy inference: SIMON ────────────────────────────────────────────────────
def simon_forward(nn_in, w):
    """Run SIMON NN forward pass in numpy. Returns log(c). nn_in: (nc,3) float32."""
    h = (nn_in - w['mean']) / w['std']
    h = h @ w['w0T'] + w['b0'];  s = 1.0/(1.0+np.exp(-h)); h = h*s  # SiLU
    h = h @ w['w1T'] + w['b1'];  s = 1.0/(1.0+np.exp(-h)); h = h*s
    h = h @ w['w2T'] + w['b2'];  s = 1.0/(1.0+np.exp(-h)); h = h*s
    return (h @ w['w3T'] + w['b3']).ravel()  # log(c)


# ── Numpy inference: Vector NN ────────────────────────────────────────────────
def vector_nn_forward(nn_in, wv):
    """
    Run Vector NN forward pass in numpy. Returns unit direction vectors.
    nn_in: (nc, 7) float32 — [rx/r, ry/r, rz/r, log_r_soft, log_r, log_mi, log_mj]
    Returns: (nc, 3) float64 unit direction vectors
    """
    h = (nn_in - wv['mean']) / wv['std']
    h = h @ wv['w0T'] + wv['b0'];  s = 1.0/(1.0+np.exp(-h)); h = h*s
    h = h @ wv['w1T'] + wv['b1'];  s = 1.0/(1.0+np.exp(-h)); h = h*s
    h = h @ wv['w2T'] + wv['b2'];  s = 1.0/(1.0+np.exp(-h)); h = h*s
    out = (h @ wv['w3T'] + wv['b3'])  # (nc, 3) unnormalized
    # F.normalize: divide by L2 norm (eps=1e-8 for safety)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    dirs  = out / np.maximum(norms, 1e-8)
    return dirs.astype(np.float64)


# ── Simulation: ias15 (ground truth) ─────────────────────────────────────────
def simulate_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0    = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k,i] = [p.x, p.y, p.z]
            vel[k,i] = [p.vx, p.vy, p.vz]
    total = time.perf_counter() - t0
    return times, pos, vel, total


# ── Simulation: leapfrog hybrid (SIMON or Vector NN) ─────────────────────────
def simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, n_samples, model_type='simon'):
    """
    model_type: 'simon'  — uses SIMON scalar correction c, analytic direction
                'vector' — uses Vector NN unit direction, analytic magnitude

    At inference:
        SIMON:     F = c * G*mi*mj/r_soft^3 * rij       (direction: exact rij/|rij|)
        Vector NN: F = G*mi*mj/r_soft^3 * r * dirs_pred  (direction: NN predicted)
    """
    N    = x0.shape[0]
    ii   = np.array([0, 0, 1]); jj = np.array([1, 2, 2])
    P    = len(ii)

    G        = cfg.G
    eps2     = cfg.eps * cfg.eps
    c_min    = cfg.c_min
    c_max    = cfg.c_max
    r_soft_min = cfg.r_soft_min

    x     = x0.astype(np.float64).copy()
    v     = v0.astype(np.float64).copy()
    m_f   = m.astype(np.float64)
    mi_a  = m_f[ii]; mj_a = m_f[jj]
    Gmimj = G * mi_a * mj_a
    inv_mi = 1.0 / mi_a; inv_mj = 1.0 / mj_a
    log_mi = np.log(mi_a + 1e-30).astype(np.float32)
    log_mj = np.log(mj_a + 1e-30).astype(np.float32)

    nn_thresh  = 500.0 * cfg.eps   # ~0.15 AU — only invoke NN for close pairs
    adapt_thresh = 0.05            # sub-step when r_min < this (AU)
    max_substeps = 16

    times    = np.linspace(0.0, T, n_samples)
    n_steps  = int(math.ceil(T / dt))
    pos_out  = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out  = np.zeros((n_samples, N, 3), dtype=np.float64)

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]                         # (P, 3)
        r2  = np.einsum('ij,ij->i', rij, rij)           # (P,)
        r   = np.sqrt(r2 + 1e-30)                       # (P,)

        # Default: unsoftened Newtonian G*mi*mj/r^3
        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3                         # (P,) scalar force magnitude

        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))

        if n_close > 0:
            r2_c  = r2[close_mask]
            r_c   = r[close_mask]
            rij_c = rij[close_mask]
            denom = (r2_c + eps2) ** 1.5 + 1e-30
            F_soft_c = Gmimj[close_mask] / denom         # G*mi*mj/r_soft^3

            if model_type == 'simon':
                # ── SIMON: predict scalar log(c) ──────────────────────────────
                r_soft_c = np.sqrt(r2_c + eps2)
                nn_in    = np.empty((n_close, 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close_mask]
                nn_in[:, 2] = log_mj[close_mask]

                log_c  = simon_forward(nn_in, ws)
                c      = np.exp(log_c).astype(np.float64)
                fb     = (r_soft_c < r_soft_min) | (c < c_min) | (c > c_max) | ~np.isfinite(c)
                F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_c)
                F_scalar[close_mask] = F_corrected
                # Direction comes from rij (exact geometry) in F_vec below

            elif model_type == 'vector':
                # ── Vector NN: predict unit direction ─────────────────────────
                r_soft_c = np.sqrt(r2_c + eps2)
                dirs_c   = rij_c / (r_c[:, None] + 1e-30)  # (nc,3) exact unit dirs (input)

                nn_in_v  = np.empty((n_close, 7), dtype=np.float32)
                nn_in_v[:, 0] = dirs_c[:, 0].astype(np.float32)
                nn_in_v[:, 1] = dirs_c[:, 1].astype(np.float32)
                nn_in_v[:, 2] = dirs_c[:, 2].astype(np.float32)
                nn_in_v[:, 3] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in_v[:, 4] = np.log(r_c + 1e-30).astype(np.float32)
                nn_in_v[:, 5] = log_mi[close_mask]
                nn_in_v[:, 6] = log_mj[close_mask]

                dirs_pred = vector_nn_forward(nn_in_v, wv)  # (nc,3) predicted unit dirs

                # Force magnitude: exact analytic (no NN involvement)
                F_mag_c = F_soft_c * r_c               # G*mi*mj*r/r_soft^3

                # Gating: fallback to Newtonian for extreme close encounters
                fb = (r_soft_c < r_soft_min) | ~np.all(np.isfinite(dirs_pred), axis=1)

                # For SIMON: F_vec = F_scalar[close]*rij later; here we pre-compute
                # For Vector NN: F_vec[close] = F_mag_c * dirs_pred (NN direction)
                F_vec_vnn_close = np.where(
                    fb[:, None],
                    F_scalar[close_mask, None] * rij_c,  # Newtonian fallback (exact)
                    F_mag_c[:, None] * dirs_pred           # VectorNN: analytic mag, NN dir
                )

        # Build full force vector array
        F_vec = F_scalar[:, None] * rij   # (P, 3) — correct for far-field and SIMON close

        if n_close > 0 and model_type == 'vector':
            F_vec[close_mask] = F_vec_vnn_close  # override close-pair forces with VectorNN

        # Accumulate accelerations
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc, n_close

    def min_pair_dist(pos):
        rij_ = pos[jj] - pos[ii]
        r2_  = np.einsum('ij,ij->i', rij_, rij_)
        return np.sqrt(np.min(r2_) + 1e-30)

    def substep(x_in, v_in, a_in, sub_dt):
        vh    = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, nf = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, nf

    # Initialise
    a, n_fb = compute_acc(x)
    fb_sum  = n_fb; pair_sum = P

    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt); total_sub = 0
    t_start = time.perf_counter()

    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
            total_sub += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
            total_sub += 1

        t_cur += dt; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12: break

    total_time = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps": steps, "dt": dt, "T_years": T,
        "total_time_sec": total_time,
        "time_per_step_sec": total_time / max(steps, 1),
        "avg_nn_frac": fb_sum / max(pair_sum, 1),
    }


# ── Metric helpers ────────────────────────────────────────────────────────────
def rms_sep(a, b):
    d = a - b
    return np.sqrt(np.mean(np.sum(d**2, axis=-1), axis=1))

def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T = times[-1]; t0 = t0_frac * T; t1 = t1_frac * T
    mask = (times >= t0) & (times <= t1)
    x = times[mask]; y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean(); y0 = y.mean()
    slope = float(np.sum((x-x0)*(y-y0)) / (np.sum((x-x0)**2) + 1e-30))
    return slope, (t0, t1)


# ── Plotting helpers ──────────────────────────────────────────────────────────
COLORS = {
    'ias15':  '#2563A6',   # blue
    'simon':  '#16A34A',   # green
    'vector': '#DC2626',   # red
}
LABELS = {
    'ias15':  'ias15 (ground truth)',
    'simon':  'SIMON (magnitude NN, analytic direction)',
    'vector': 'Vector NN (analytic magnitude, NN direction)',
}

def plot_divergence_3model(times, deltas, slopes, windows, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in ('simon', 'vector'):
        d = deltas[key]; sl = slopes[key]; win = windows[key]
        ax.semilogy(times, d, color=COLORS[key], lw=1.8,
                    label=f"{LABELS[key]}\n  λ ≈ {sl:.3f}/yr")
    # Shade common fit window (use SIMON's window)
    ax.axvspan(windows['simon'][0], windows['simon'][1], alpha=0.10,
               color='gray', label="fit window (10–50 yr)")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error vs ias15 (AU)")
    ax.set_title("Divergence Rate: SIMON vs Vector NN")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

def plot_trajectories_3model(pos_ias, pos_simon, pos_vector, out_path):
    N   = pos_ias.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.5 * N, 4.5))
    if N == 1: axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.plot(pos_ias[:,i,0],    pos_ias[:,i,1],    '-',  color=COLORS['ias15'],
                lw=1.5, label='ias15', alpha=0.9)
        ax.plot(pos_simon[:,i,0],  pos_simon[:,i,1],  '--', color=COLORS['simon'],
                lw=1.2, label='SIMON', alpha=0.9)
        ax.plot(pos_vector[:,i,0], pos_vector[:,i,1], ':',  color=COLORS['vector'],
                lw=1.2, label='Vector NN', alpha=0.9)
        ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
        ax.set_title(f"Body {i} XY  (T=100 yr)")
        ax.grid(True, alpha=0.25)
        if i == 0: ax.legend(loc='upper left', fontsize=8)
    fig.suptitle("Trajectory Overlay: ias15 vs SIMON vs Vector NN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

def plot_timeseries_3model(times, pos_ias, pos_simon, pos_vector, out_path):
    N   = pos_ias.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(11, 3.0 * N), sharex=True)
    if N == 1: axes = np.array([axes])
    for i in range(N):
        for col, coord, clabel in [(0, 0, 'x'), (1, 1, 'y')]:
            ax = axes[i, col]
            ax.plot(times, pos_ias[:,i,coord],    '-',  color=COLORS['ias15'],
                    lw=1.5, alpha=0.9, label='ias15')
            ax.plot(times, pos_simon[:,i,coord],  '--', color=COLORS['simon'],
                    lw=1.0, alpha=0.9, label='SIMON')
            ax.plot(times, pos_vector[:,i,coord], ':',  color=COLORS['vector'],
                    lw=1.0, alpha=0.9, label='Vector NN')
            ax.set_ylabel(f"Body {i} {clabel}(t) (AU)")
            ax.grid(True, alpha=0.25)
            if i == 0 and col == 0: ax.legend(loc='upper left', fontsize=7)
    axes[-1, 0].set_xlabel("Time (yr)")
    axes[-1, 1].set_xlabel("Time (yr)")
    fig.suptitle("Position Time Series: ias15 vs SIMON vs Vector NN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="3-model comparison: ias15, SIMON, Vector NN")
    ap.add_argument("--simon_model",  default="pair_correction_nn.pt")
    ap.add_argument("--vector_model", default="vector_nn.pt")
    ap.add_argument("--out_dir",      default="comparison_out")
    ap.add_argument("--dt",  type=float, default=0.04)   # reference timestep
    ap.add_argument("--T",   type=float, default=100.0)
    ap.add_argument("--n_samples", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    # ── Load models ───────────────────────────────────────────────────────────
    print("[compare] Loading SIMON model ...")
    simon_model = PairCorrectionNN(hidden=32)
    simon_model.load_state_dict(torch.load(args.simon_model, map_location='cpu'))
    simon_model.eval()
    ws = extract_weights_simon(simon_model)
    n_simon = sum(p.numel() for p in simon_model.parameters())
    print(f"  Loaded {args.simon_model}  params={n_simon}")

    print("[compare] Loading Vector NN model ...")
    vector_model = VectorNN(hidden=32)
    vector_model.load_state_dict(torch.load(args.vector_model, map_location='cpu'))
    vector_model.eval()
    wv = extract_weights_vector(vector_model)
    n_vector = sum(p.numel() for p in vector_model.parameters())
    print(f"  Loaded {args.vector_model}  params={n_vector}")

    # ── Initial conditions (same as pair_eval_after_adaptive.py) ─────────────
    T  = args.T; ns = args.n_samples; dt = args.dt
    m  = np.array([1.0, 0.01, 0.005])
    x0 = np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64)
    M  = m.sum()
    x0 = x0 - (m[:,None]*x0).sum(0)/M
    v0 = v0 - (m[:,None]*v0).sum(0)/M

    # ── ias15 ground truth ────────────────────────────────────────────────────
    print(f"\n[compare] Running ias15  (T={T}yr, n_samples={ns}) ...")
    tr, pr, vr, t_ias = simulate_ias15(x0, v0, m, cfg.G, T, ns)
    print(f"  ias15: {t_ias:.3f}s")

    # ── SIMON ─────────────────────────────────────────────────────────────────
    print(f"[compare] Running SIMON  (dt={dt}) ...")
    _, ps, vs, perf_s = simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, ns,
                                         model_type='simon')
    sp_s = t_ias / max(perf_s['total_time_sec'], 1e-12)
    print(f"  SIMON: {perf_s['total_time_sec']:.3f}s  speedup={sp_s:.2f}x  "
          f"NN_frac={perf_s['avg_nn_frac']:.4f}")

    # ── Vector NN ─────────────────────────────────────────────────────────────
    print(f"[compare] Running Vector NN  (dt={dt}) ...")
    _, pv, vv, perf_v = simulate_hybrid(x0, v0, m, ws, wv, cfg, dt, T, ns,
                                         model_type='vector')
    sp_v = t_ias / max(perf_v['total_time_sec'], 1e-12)
    print(f"  VectorNN: {perf_v['total_time_sec']:.3f}s  speedup={sp_v:.2f}x  "
          f"NN_frac={perf_v['avg_nn_frac']:.4f}")

    # ── Divergence rates ──────────────────────────────────────────────────────
    print("\n[compare] Computing divergence rates ...")
    delta_s = rms_sep(ps, pr)
    delta_v = rms_sep(pv, pr)

    slope_s, win_s = fit_log_slope(tr, delta_s)
    slope_v, win_v = fit_log_slope(tr, delta_v)

    rms_final_s = float(delta_s[-1])
    rms_final_v = float(delta_v[-1])

    print(f"  SIMON   λ ≈ {slope_s:.4f}/yr  final_RMS = {rms_final_s:.4f}")
    print(f"  VectorNN λ ≈ {slope_v:.4f}/yr  final_RMS = {rms_final_v:.4f}")

    if slope_v > slope_s:
        print(f"  --> Vector NN diverges FASTER (higher λ by {slope_v-slope_s:.4f}/yr)")
        print(f"      Supports claim: analytic direction reduces error-growth rate.")
    else:
        print(f"  --> SIMON diverges faster or equal (unexpected).")
        print(f"      Magnitude errors may dominate over directional errors.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[compare] Generating plots ...")
    plot_divergence_3model(
        tr,
        {'simon': delta_s, 'vector': delta_v},
        {'simon': slope_s, 'vector': slope_v},
        {'simon': win_s,   'vector': win_v},
        os.path.join(args.out_dir, "divergence_3model.png")
    )
    plot_trajectories_3model(pr, ps, pv,
        os.path.join(args.out_dir, "trajectory_3model.png"))
    plot_timeseries_3model(tr, pr, ps, pv,
        os.path.join(args.out_dir, "timeseries_3model.png"))

    # ── Summary text ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "summary_comparison.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("3-MODEL COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"T = {T} years  |  dt = {dt} yr  |  n_samples = {ns}\n")
        f.write(f"Initial conditions: m=[1.0, 0.01, 0.005] Msun\n\n")
        f.write(f"{'Model':<20} {'lambda(/yr)':>12} {'final_RMS':>12} "
                f"{'speedup':>10} {'NN_frac':>10}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'ias15 (reference)':<20} {'--':>12} {'0.000':>12} "
                f"{'1.00x':>10} {'--':>10}\n")
        f.write(f"{'SIMON':<20} {slope_s:>12.4f} {rms_final_s:>12.4f} "
                f"{sp_s:>9.2f}x {perf_s['avg_nn_frac']:>10.4f}\n")
        f.write(f"{'Vector NN':<20} {slope_v:>12.4f} {rms_final_v:>12.4f} "
                f"{sp_v:>9.2f}x {perf_v['avg_nn_frac']:>10.4f}\n\n")
        f.write("INTERPRETATION:\n")
        if slope_v > slope_s:
            f.write(f"  lambda(VectorNN) - lambda(SIMON) = "
                    f"{slope_v-slope_s:.4f}/yr\n")
            f.write("  Vector NN diverges faster. Enforcing analytic force direction\n")
            f.write("  (SIMON) reduces the error-growth rate in this chaotic system.\n")
            f.write("  Small NN directional errors accumulate faster than SIMON's\n")
            f.write("  scalar magnitude errors over 100-year chaotic orbits.\n")
        else:
            f.write("  SIMON and Vector NN have similar divergence rates.\n")
            f.write("  Consider running with more epochs or different initial conds.\n")
    print(f"  Saved {summary_path}")

    print(f"\n[compare] Done. Output in: {args.out_dir}/")
    print(f"  Key result: lambda_SIMON={slope_s:.4f}/yr  "
          f"lambda_VectorNN={slope_v:.4f}/yr")


if __name__ == "__main__":
    main()
