"""
simon_core.py — clean, bug-fixed SIMON harness (general N), shared by all tracks.

Design goals:
  * ONE integrator core, general over N bodies (no hardcoded 3-body unrolling).
  * Sampling bug fixed at the source: t_cur = (step+1)*dt (index-derived, no
    accumulation drift); post-loop fill of any unwritten slot; hard assertion
    that no output sample is the all-zero vector for all bodies.
  * Faithful reproduction of the published force model (softened + scalar NN
    correction + gating + adaptive sub-stepping) so baselines reproduce.
  * Metric panel discipline: energy |dE/E0|max (softened PE), short-horizon
    accuracy (within ~1-2 Lyapunov times), force-eval cost, bounded/ejected.
    100-yr global RMS only as a coarse sanity check.

Conventions:
  * Toy ICs IC1-IC6 use G=1.0; the real Sun-Earth-Moon system uses G=4*pi^2.
"""
import math
import numpy as np
import torch
import torch.nn as nn

# ---- physical / model constants (match published SIMON) ----
EPS         = 3e-4          # Plummer softening length
NN_THRESH   = 500.0 * EPS   # 0.15 AU: NN/softening only for closer pairs
C_MIN, C_MAX = 0.2, 5.0     # NN correction-factor gate
R_SOFT_MIN  = 5e-4          # gate: below this softened sep -> Newtonian fallback
ADAPT_THRESH = 0.05         # default adaptive sub-step trigger
MAX_SUBSTEPS = 16           # default adaptive cap
G_TOY = 1.0
G_REAL = 4.0 * np.pi**2

EJECT_AU = 10.0             # bounded := max dist from COM < this

# ---------------------------------------------------------------------------
# Initial conditions (IC1-IC6), verbatim from multi_ic_eval_v3.py
# ---------------------------------------------------------------------------
_ICS_RAW = {
    "IC1": dict(m=[1.0, 0.01, 0.005], x0=[[0,0,0],[1,0,0],[0,1.2,0]],
                v0=[[0,0,0],[0,1,0],[-0.9,0,0]], exp_lambda=0.1655, stable=True),
    "IC2": dict(m=[1.0, 0.5, 0.25], x0=[[0,0,0],[1,0,0],[-0.5,0.8,0]],
                v0=[[0,0,0],[0,0.6,0],[-0.4,-0.3,0]], exp_lambda=0.0430, stable=False),
    "IC3": dict(m=[1.0, 0.01, 0.005], x0=[[0,0,0],[0.5,0,0],[0,2.5,0]],
                v0=[[0,0,0],[0,1.3,0],[-0.4,0,0]], exp_lambda=0.1207, stable=True),
    "IC4": dict(m=[1.0, 0.01, 0.005], x0=[[0,0,0],[1,0,0],[0,5.0,0]],
                v0=[[0,0,0],[0,1.0,0],[-0.12,0,0]], exp_lambda=0.0990, stable=True),
    "IC5": dict(m=[1.0, 0.01, 0.005], x0=[[0,0,0],[0.3,0,0],[0,2.0,0]],
                v0=[[0,0,0],[0,2.2,0],[-0.3,0,0]], exp_lambda=0.0648, stable=False),
    "IC6": dict(m=[1.0, 0.01, 0.005], x0=[[0,0,0],[1.5,0,0],[-3.0,0,0]],
                v0=[[0,0,0],[0,0.816,0],[0,-0.577,0]], exp_lambda=-0.0189, stable=True),
}

def get_ic(name):
    """Return (m, x0, v0) COM-centred, float64."""
    d = _ICS_RAW[name]
    m  = np.array(d["m"], dtype=np.float64)
    x0 = np.array(d["x0"], dtype=np.float64)
    v0 = np.array(d["v0"], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M
    return m, x0, v0

def ic_meta(name):
    return dict(_ICS_RAW[name])

# ---------------------------------------------------------------------------
# Scalar NN force-magnitude correction (the published model). Kept ONLY to
# reproduce SIMON faithfully -- finding #1 says it is ~an identity.
# ---------------------------------------------------------------------------
class PairCorrectionNN(nn.Module):
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

def load_scalar_nn_weights(path):
    m = PairCorrectionNN(32)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    sd = m.state_dict()
    return {
        'mean': sd['input_mean'].numpy().astype(np.float32),
        'std':  sd['input_std'].numpy().astype(np.float32) + 1e-8,
        'w0T': sd['net.0.weight'].numpy().T.astype(np.float32).copy(), 'b0': sd['net.0.bias'].numpy().astype(np.float32),
        'w1T': sd['net.2.weight'].numpy().T.astype(np.float32).copy(), 'b1': sd['net.2.bias'].numpy().astype(np.float32),
        'w2T': sd['net.4.weight'].numpy().T.astype(np.float32).copy(), 'b2': sd['net.4.bias'].numpy().astype(np.float32),
        'w3T': sd['net.6.weight'].numpy().T.astype(np.float32).copy(), 'b3': sd['net.6.bias'].numpy().astype(np.float32),
    }

def _nn_logc(w, nn_in):
    h = (nn_in - w['mean']) / w['std']
    h = h @ w['w0T'] + w['b0']; s = 1.0/(1.0+np.exp(-h)); h = h*s
    h = h @ w['w1T'] + w['b1']; s = 1.0/(1.0+np.exp(-h)); h = h*s
    h = h @ w['w2T'] + w['b2']; s = 1.0/(1.0+np.exp(-h)); h = h*s
    return (h @ w['w3T'] + w['b3']).ravel()

# ---------------------------------------------------------------------------
# Force model. mode in {'newton','soft','nn'} controls close-pair handling:
#   newton : pure unsoftened Newtonian for all pairs
#   soft   : softened (Plummer) for close pairs (c=1), Newtonian far  (Run C "no NN")
#   nn     : softened * NN correction c (gated) for close pairs, Newtonian far (SIMON)
# Returns acc (N,3). force-eval accounting handled by caller (1 per call).
# ---------------------------------------------------------------------------
def make_pairs(N):
    ii, jj = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)

def compute_acc(x, ii, jj, Gmimj, inv_mi, inv_mj, log_mi, log_mj, w, eps2, mode):
    rij = x[jj] - x[ii]
    r2  = np.einsum('ij,ij->i', rij, rij)
    r   = np.sqrt(r2 + 1e-30)
    invr3 = 1.0 / (r2 * r + 1e-30)
    F = Gmimj * invr3                       # Newtonian default (all pairs)
    if mode != 'newton':
        close = r < NN_THRESH
        if np.any(close):
            r2c = r2[close]
            r_soft_c = np.sqrt(r2c + eps2)
            denom = (r2c + eps2) ** 1.5 + 1e-30
            F_soft = Gmimj[close] / denom
            if mode == 'nn':
                nn_in = np.empty((int(close.sum()), 3), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft_c + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[close]; nn_in[:, 2] = log_mj[close]
                c = np.exp(_nn_logc(w, nn_in)).astype(np.float64)
                fb = (r_soft_c < R_SOFT_MIN) | (c < C_MIN) | (c > C_MAX) | ~np.isfinite(c)
                F[close] = np.where(fb, F[close], c * F_soft)
            else:  # 'soft'
                fb = (r_soft_c < R_SOFT_MIN)
                F[close] = np.where(fb, F[close], F_soft)
    Fvec = F[:, None] * rij
    N = x.shape[0]
    acc = np.zeros((N, 3))
    np.add.at(acc, ii,  Fvec * inv_mi[:, None])
    np.add.at(acc, jj, -Fvec * inv_mj[:, None])
    return acc

# ---------------------------------------------------------------------------
# Leapfrog integrator (general N). Fixed-step or heuristic-adaptive sub-stepping.
# Bug-fixed sampling. Returns times, pos, vel, info(force_evals, n_sub_history,...)
# ---------------------------------------------------------------------------
def _prep(m, G):
    ii, jj = make_pairs(len(m))
    mi, mj = m[ii], m[jj]
    return dict(ii=ii, jj=jj, Gmimj=G*mi*mj, inv_mi=1.0/mi, inv_mj=1.0/mj,
                log_mi=np.log(mi+1e-30).astype(np.float32),
                log_mj=np.log(mj+1e-30).astype(np.float32))

def simulate_leapfrog(m, x0, v0, dt, T, n_samples, G, mode='nn', w=None,
                      adaptive=False, adapt_thresh=ADAPT_THRESH, max_substeps=MAX_SUBSTEPS):
    P = _prep(m, G); ii, jj = P['ii'], P['jj']
    eps2 = EPS*EPS
    x = x0.astype(np.float64).copy(); v = v0.astype(np.float64).copy()
    N = len(m)
    times = np.linspace(0.0, T, n_samples)
    n_steps = int(round(T/dt))
    pos = np.full((n_samples, N, 3), np.nan)   # NaN init: catch any unwritten slot
    vel = np.full((n_samples, N, 3), np.nan)
    fe = [0]
    def acc_of(xx):
        fe[0] += 1
        return compute_acc(xx, ii, jj, P['Gmimj'], P['inv_mi'], P['inv_mj'],
                           P['log_mi'], P['log_mj'], w, eps2, mode)
    def min_sep(xx):
        d = xx[jj]-xx[ii]; return math.sqrt(float(np.min(np.einsum('ij,ij->i', d, d))) + 1e-30)
    a = acc_of(x)
    pos[0] = x; vel[0] = v
    si = 0
    n_sub_hist = []
    for step in range(n_steps):
        if adaptive and min_sep(x) < adapt_thresh:
            rmin = min_sep(x)
            n_sub = min(max_substeps, max(2, int(np.ceil(adapt_thresh/rmin))))
            h = dt/n_sub
            for _ in range(n_sub):
                vh = v + 0.5*h*a; x = x + h*vh; a = acc_of(x); v = vh + 0.5*h*a
            n_sub_hist.append(n_sub)
        else:
            vh = v + 0.5*dt*a; x = x + dt*vh; a = acc_of(x); v = vh + 0.5*dt*a
            n_sub_hist.append(1)
        t_cur = (step+1)*dt                       # index-derived: no accumulation drift
        while si < n_samples-1 and times[si+1] <= t_cur + 1e-9:
            si += 1; pos[si] = x; vel[si] = v
    while si < n_samples-1:                        # fill any trailing slot with final state
        si += 1; pos[si] = x; vel[si] = v
    # --- sampling-bug guards ---
    assert not np.isnan(pos).any(), "unwritten sample slot remained (NaN)"
    assert not np.all(pos == 0.0, axis=2).all(axis=1).any(), "all-zero sample for all bodies"
    info = dict(force_evals=fe[0], n_sub_history=n_sub_hist,
                total_substeps=int(np.sum(n_sub_hist)), n_steps=n_steps,
                min_sep_run=float(min(min_sep(pos[k]) for k in range(0, n_samples, max(1, n_samples//200)))))
    return times, pos, vel, info

# ---------------------------------------------------------------------------
# REBOUND IAS15 reference
# ---------------------------------------------------------------------------
def ias15_reference(m, x0, v0, T, n_samples, G):
    import rebound
    sim = rebound.Simulation(); sim.integrator = "ias15"; sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]), x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3)); vel = np.zeros((n_samples, len(m), 3))
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k,i] = [p.x, p.y, p.z]; vel[k,i] = [p.vx, p.vy, p.vz]
    return times, pos, vel

# ---------------------------------------------------------------------------
# Diagnostics / metric panel
# ---------------------------------------------------------------------------
def energy_series(pos, vel, m, G, eps=EPS):
    KE = 0.5 * np.einsum("kij,i->k", vel**2, m)
    PE = np.zeros(pos.shape[0]); e2 = eps*eps; N = len(m)
    for i in range(N):
        for j in range(i+1, N):
            d = pos[:, i, :] - pos[:, j, :]; r2 = np.einsum("ki,ki->k", d, d)
            PE -= G * m[i]*m[j] / np.sqrt(r2 + e2)
    return KE + PE

def max_dE(pos, vel, m, G, eps=EPS):
    E = energy_series(pos, vel, m, G, eps)
    if not np.all(np.isfinite(E)): return float("inf")
    E0 = E[0]
    return float(np.max(np.abs((E - E0)/max(abs(E0), 1e-30))) * 100.0)

def rms_sep(a, b):
    d = a - b; pb = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(pb**2, axis=1))

def bounded(pos, m):
    if not np.all(np.isfinite(pos)): return False
    com = (pos * m[None,:,None]).sum(1, keepdims=True) / m.sum()
    return bool(np.max(np.linalg.norm(pos - com, axis=2)) < EJECT_AU)

def metric_panel(times, pos, vel, pos_ref, m, G, force_evals, lyap=0.15, eps=EPS):
    """The disciplined panel: energy, short-horizon accuracy, cost, bounded."""
    d = rms_sep(pos, pos_ref)
    T = times[-1]
    t_lyap = min(2.0/max(lyap, 1e-3), T)          # ~2 Lyapunov times
    short = d[times <= t_lyap]
    return dict(
        max_dE_pct = max_dE(pos, vel, m, G, eps),
        rms_short  = float(np.sqrt(np.mean(short**2))) if short.size else float("nan"),
        t_short    = float(t_lyap),
        rms_final  = float(d[-1]),                # coarse 100-yr sanity ONLY
        rms_timeavg= float(np.sqrt(np.mean(d**2))),
        force_evals= int(force_evals),
        bounded    = bounded(pos, m),
    )
