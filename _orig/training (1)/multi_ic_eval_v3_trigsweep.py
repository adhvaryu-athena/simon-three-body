# multi_ic_eval_v3.py
#
# Runs SIMON across 7 distinct three-body initial conditions.
# Extends multi_ic_eval_v2.py (6 ICs) by adding IC7 (high-eccentricity,
# redesigned from IC5 to be dynamically stable).
#
# IC history:
#   IC1-IC4: original multi_ic_eval.py ICs (unchanged)
#   IC5:     high-eccentricity attempt (v=2.2 at r=0.3 AU) -- physically
#             unstable (confirmed by check_ic5.py: ias15 also ejects)
#   IC6:     near-circular coplanar -- stable, low-chaos
#   IC7:     high-eccentricity redesigned (v=1.6 at r=0.6 AU, Body2 at 4 AU)
#             -- e~0.52, 1.24x v_circ, safer periapsis than IC5
#
# SELF-CONTAINED: no imports from pair_eval_after_adaptive.py
#   All simulation logic copied verbatim from pair_eval_after_adaptive.py.
#
# VERIFICATION:
#   IC1-IC4 verified against multi_ic_eval.py (lambda, RMS, bounded status).
#   IC5, IC6 verified against multi_ic_eval_v2.py results.
#   IC7 is new -- no prior expected value.
#
# PHYSICALLY UNSTABLE ICs (confirmed by separate check scripts):
#   IC2: check_ic2.py -- ias15 ejects to 58.94 AU
#   IC5: check_ic5.py -- ias15 ejects to 47.08 AU
#   Both are hatched in figures and honestly framed in summary.
#
# OUTPUT:  multi_ic_out_v3\  (v2 folder untouched)
#   multi_ic_divergence_v3.png     -- divergence curves (2x4 panel)
#   multi_ic_results_v3.png        -- bar charts (lambda, RMS, speedup)
#   multi_ic_trajectories_v3.png   -- trajectory overlays (2x4 panel)
#   multi_ic_summary_v3.txt        -- full quantitative summary
#
# Run: python multi_ic_eval_v3.py
# Expected runtime: ~3-4 minutes (7 ICs x ~30s each)

import os, time, math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found. Install it in your environment.")
    raise


# ── Model (verbatim from pair_eval_after_adaptive.py) ────────────────────────
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
        self.register_buffer('input_std',  torch.ones(3))
    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


# ── Config (verbatim from pair_eval_after_adaptive.py) ───────────────────────
@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    mc_samples: int = 1
    unc_rel_thresh: float = 0.25
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


# ── Weight extraction (verbatim from pair_eval_after_adaptive.py) ─────────────
def extract_weights_numpy(model):
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


# ── simulate_leapfrog_hybrid (verbatim from pair_eval_after_adaptive.py) ──────
# Key 'avg_fallback_frac' preserved to match original exactly.
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples,
                             device="cpu", dtype=torch.float32, adapt_thresh=0.05):
    w = extract_weights_numpy(model)
    w_mean = w['mean']; w_std  = w['std']
    w0T = w['w0T'];     b0 = w['b0']
    w1T = w['w1T'];     b1 = w['b1']
    w2T = w['w2T'];     b2 = w['b2']
    w3T = w['w3T'];     b3 = w['b3']

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii); jj = np.array(jj); P = len(ii)

    G          = cfg.G
    eps2       = cfg.eps * cfg.eps
    c_min      = cfg.c_min
    c_max      = cfg.c_max
    r_soft_min = cfg.r_soft_min

    x      = x0.astype(np.float64).copy()
    v      = v0.astype(np.float64).copy()
    m_f    = m.astype(np.float64)
    mi_arr = m_f[ii]; mj_arr = m_f[jj]
    Gmimj  = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times   = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    nn_thresh = 500.0 * cfg.eps   # 0.15 AU

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum('ij,ij->i', rij, rij)
        r   = np.sqrt(r2 + 1e-30)
        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3
        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))
        if n_close > 0:
            r_soft_close = np.sqrt(r2[close_mask] + eps2)
            denom        = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom
            log_rs = np.log(r_soft_close + 1e-30).astype(np.float32)
            nc     = n_close
            nn_in  = np.empty((nc, 3), dtype=np.float32)
            nn_in[:, 0] = log_rs
            nn_in[:, 1] = log_mi[close_mask]
            nn_in[:, 2] = log_mj[close_mask]
            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            h = h @ w1T + b1;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            h = h @ w2T + b2;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            log_c = (h @ w3T + b3).ravel()
            c     = np.exp(log_c).astype(np.float64)
            fb = ((r_soft_close < r_soft_min) |
                  (c < c_min) | (c > c_max) | ~np.isfinite(c))
            F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_close)
            F_scalar[close_mask] = F_corrected
        F_vec = F_scalar[:, None] * rij
        acc   = np.zeros((N, 3), dtype=np.float64)
        acc[ii[0]] += F_vec[0] * inv_mi[0]; acc[jj[0]] -= F_vec[0] * inv_mj[0]
        acc[ii[1]] += F_vec[1] * inv_mi[1]; acc[jj[1]] -= F_vec[1] * inv_mj[1]
        acc[ii[2]] += F_vec[2] * inv_mi[2]; acc[jj[2]] -= F_vec[2] * inv_mj[2]
        return acc, n_close

    max_substeps = 16

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum('ij,ij->i', rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh    = v_in + 0.5 * sub_dt * a_in
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
    n_sub_history = []; r_min_min = np.inf
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < r_min_min: r_min_min = r_min
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
            total_substeps += n_sub
            n_sub_history.append(n_sub)
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
            total_substeps += 1
            n_sub_history.append(1)
        t_cur += dt; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12: break

    total_time = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps":              steps,
        "dt":                 dt,
        "T_years":            T,
        "n_samples":          n_samples,
        "total_time_sec":     total_time,
        "time_per_step_sec":  total_time / max(steps, 1),
        "avg_fallback_frac":  fb_sum / max(pair_sum, 1),
        "avg_pairs_per_step": P,
        "total_substeps":     total_substeps,
        "n_sub_history":      n_sub_history,
        "r_min_min":          float(r_min_min),
    }


# ── simulate_rebound_ias15 (verbatim from pair_eval_after_adaptive.py) ────────
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"; sim.G = G
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
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


# ── rms_sep (verbatim from pair_eval_after_adaptive.py) ──────────────────────
def rms_sep(a, b):
    d  = a - b
    pb = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(pb**2, axis=1))


# ── fit_log_slope (verbatim from pair_eval_after_adaptive.py) ─────────────────
def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T_   = times[-1]; t0 = t0_frac * T_; t1 = t1_frac * T_
    mask = (times >= t0) & (times <= t1)
    x    = times[mask]
    y    = np.log(np.clip(delta[mask], 1e-30, None))
    x0_  = x.mean(); y0_ = y.mean()
    slope = float(np.sum((x-x0_)*(y-y0_)) / (np.sum((x-x0_)**2) + 1e-30))
    return slope, (t0, t1)


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH         = "pair_correction_nn.pt"
OUT_DIR            = "multi_ic_out_v3"          # new folder -- v2 untouched
DT_REF             = 0.04
T                  = 100.0
N_SAMPLES          = 5000
EJECTION_THRESHOLD = 10.0   # AU

# ICs confirmed physically unstable by separate check scripts:
#   IC2: check_ic2.py -- ias15 ejects to 58.94 AU at t~?
#   IC5: check_ic5.py -- ias15 ejects to 47.08 AU at t=34.8yr
KNOWN_UNSTABLE = {"IC2_near_equal", "IC5_high_ecc"}

# Verification targets (from previously validated runs):
#   IC1-IC4: multi_ic_eval.py
#   IC5-IC6: multi_ic_eval_v2.py
EXPECTED = {
    "IC1_default":      {"lambda": 0.1655, "rms": 1.54,    "bounded": True},
    "IC2_near_equal":   {"lambda": 0.0430, "rms": 116.87,  "bounded": False},
    "IC3_tight":        {"lambda": 0.1207, "rms": 3.45,    "bounded": True},
    "IC4_hierarchical": {"lambda": 0.0990, "rms": 1.60,    "bounded": True},
    "IC5_high_ecc":     {"lambda": 0.0648, "rms": 19.97,   "bounded": False},
    "IC6_near_circular":{"lambda":-0.0189, "rms": 0.0079,  "bounded": True},
    # IC7 is new -- no prior expected values
}
LAMBDA_TOL   = 0.005    # +/- tolerance for lambda match
RMS_TOL_FRAC = 0.10     # 10% tolerance for RMS match

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────
print("[multi_ic_v3] Loading SIMON model ...")
model = PairCorrectionNN(hidden=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
cfg = HybridConfig()
print(f"  {MODEL_PATH}  params={sum(p.numel() for p in model.parameters())}")


# ── Initial conditions ────────────────────────────────────────────────────────
# IC1-IC6: IDENTICAL to multi_ic_eval_v2.py (do not change)
# IC7:     new -- high-eccentricity redesigned from IC5
ICS = {
    # ── IC1-IC4: original (must reproduce multi_ic_eval.py results) ──────────
    "IC1_default": {
        "label": "IC1: Default (V2)",
        "desc":  "m=[1.0,0.01,0.005]  |  V2 baseline",
        "regime":"Moderate three-body scattering",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,1.2,0]],    dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1,0],[-0.9,0,0]],   dtype=np.float64),
        "color": "#2563A6", "new": False,
    },
    "IC2_near_equal": {
        "label": "IC2: Near-Equal Mass",
        "desc":  "m=[1.0,0.5,0.25]  |  Strongly interacting",
        "regime":"Strongly interacting -- physically unstable (check_ic2.py)",
        "m":  np.array([1.0, 0.5, 0.25]),
        "x0": np.array([[0,0,0],[1,0,0],[-0.5,0.8,0]],   dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.6,0],[-0.4,-0.3,0]], dtype=np.float64),
        "color": "#16A34A", "new": False,
    },
    "IC3_tight": {
        "label": "IC3: Tight Inner Pair",
        "desc":  "m=[1.0,0.01,0.005]  |  Body 1 at 0.5 AU",
        "regime":"Tight inner pair -- continuous close encounters",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.5,0,0],[0,2.5,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.3,0],[-0.4,0,0]], dtype=np.float64),
        "color": "#DC2626", "new": False,
    },
    "IC4_hierarchical": {
        "label": "IC4: Hierarchical",
        "desc":  "m=[1.0,0.01,0.005]  |  Body 2 at 5 AU",
        "regime":"Hierarchical -- weakly coupled outer body",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,5.0,0]],    dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.0,0],[-0.12,0,0]], dtype=np.float64),
        "color": "#9333EA", "new": False,
    },
    # ── IC5-IC6: from multi_ic_eval_v2.py (must reproduce v2 results) ────────
    "IC5_high_ecc": {
        "label": "IC5: High Ecc (unstable)",
        "desc":  "m=[1.0,0.01,0.005]  |  Body 1 at 0.3 AU, v=2.2 (85% v_esc)",
        "regime":"High eccentricity -- physically unstable (check_ic5.py)",
        # Physically unstable: ias15 ejects Body 2 to 47.08 AU at t=34.8yr.
        # Kept for completeness; not used in bounded-IC statistics.
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.3,0,0],[0,2.0,0]],  dtype=np.float64),
        "v0": np.array([[0,0,0],[0,2.2,0],[-0.3,0,0]], dtype=np.float64),
        "color": "#EA580C", "new": False,
    },
    "IC6_near_circular": {
        "label": "IC6: Near-Circular",
        "desc":  "m=[1.0,0.01,0.005]  |  Both bodies on Keplerian circular orbits",
        "regime":"Near-circular coplanar -- low chaos, NN never invoked",
        # v_circ at r=1.5: sqrt(1/1.5)=0.816. v_circ at r=3.0: sqrt(1/3.0)=0.577.
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1.5,0,0],[-3.0,0,0]],   dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.816,0],[0,-0.577,0]], dtype=np.float64),
        "color": "#0891B2", "new": False,
    },
    # ── IC7: new -- high-eccentricity redesigned ──────────────────────────────
    "IC7_high_ecc_stable": {
        "label": "IC7: High Ecc (redesign)",
        "desc":  "m=[1.0,0.01,0.005]  |  Body 1 at 0.6 AU, v=1.6 (1.24x v_circ, e~0.52)",
        "regime":"High eccentricity -- redesigned from IC5 for stability",
        # IC5 used v=2.2 at r=0.3 AU -- too aggressive, both integrators ejected.
        # IC7 uses v=1.6 at r=0.6 AU: same eccentricity ratio (1.24x v_circ) but
        # gentler periapsis and Body 2 at 4 AU (further from Body 1's orbit).
        # Estimated eccentricity e~0.52 (semi-major axis ~1.25 AU).
        # Energy E=-0.00520 (bound). v_esc at r=0.6 = 1.826, so v=1.6 is 88% v_esc.
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.6,0,0],[0,4.0,0]],  dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.6,0],[-0.15,0,0]], dtype=np.float64),
        "color": "#B45309", "new": True,
    },
}

# ── Apply CoM centering (identical to all previous scripts) ──────────────────
for name, ic in ICS.items():
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]
    M  = m.sum()
    ic["x0"] = x0 - (m[:, None] * x0).sum(0) / M
    ic["v0"] = v0 - (m[:, None] * v0).sum(0) / M

# ── Energy verification for all ICs ──────────────────────────────────────────
print(f"\n[multi_ic_v3] Energy check (all ICs must have E < 0):")
for name, ic in ICS.items():
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]
    KE = 0.5 * np.sum(m[:, None] * v0**2)
    PE = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(x0[i] - x0[j])
            PE -= cfg.G * m[i] * m[j] / r
    E_total = KE + PE
    ic["E_total"] = E_total
    unstable_tag = " [KNOWN UNSTABLE]" if name in KNOWN_UNSTABLE else ""
    new_tag      = " [NEW]"            if ic["new"]               else ""
    status = "BOUND" if E_total < 0 else "UNBOUND -- ABORTING"
    print(f"  {ic['label']:<28}{new_tag:<7}{unstable_tag:<18}  "
          f"E={E_total:.5f}  {status}")
    if E_total >= 0:
        raise ValueError(f"IC {name} is not bound! E={E_total:.5f}")


# ============================================================================
# TRIGGER SWEEP DRIVER  (replaces the original 7-IC run/figure section).
# Only adapt_thresh varies. Physics, forces, ias15, masses, dt(=DT_REF),
# T, sampling, and the model are exactly as defined above (verbatim from the
# original). Energy diagnostic = softened PE (eps=cfg.eps), the project convention.
# ============================================================================
PHASEA_OUT = "weekend_phaseA"
os.makedirs(PHASEA_OUT, exist_ok=True)

SWEEP_ICS    = ["IC1_default", "IC3_tight", "IC4_hierarchical", "IC6_near_circular"]
SWEEP_THRESH = [0.05, 0.10, 0.20, 0.30, 0.50]
EPS = cfg.eps  # softened-PE softening, unchanged (3e-4)

# user-provided baseline drift at adapt_thresh=0.05, for the verification guard
BASE_DRIFT = {"IC1_default": 1.39, "IC3_tight": 0.67,
              "IC4_hierarchical": 13.77, "IC6_near_circular": 0.0006}

def energy_softened(pos, vel, mm, G, eps):
    KE = 0.5 * np.einsum("kij,i->k", vel**2, mm)
    PE = np.zeros(pos.shape[0]); e2 = eps*eps; N = len(mm)
    for i in range(N):
        for j in range(i+1, N):
            d = pos[:, i, :] - pos[:, j, :]
            r2 = np.einsum("ki,ki->k", d, d)
            PE -= G * mm[i] * mm[j] / np.sqrt(r2 + e2)
    return KE + PE

def max_dE_pct(pos, vel, mm, G, eps):
    E = energy_softened(pos, vel, mm, G, eps)
    if not np.all(np.isfinite(E)): return float("inf")
    E0 = E[0]
    return float(np.max(np.abs((E - E0) / max(abs(E0), 1e-30))) * 100.0)

times_arr = np.linspace(0.0, T, N_SAMPLES)

print("\n[trigsweep] ias15 references (per IC, adapt_thresh-independent) ...")
REF = {}
for key in SWEEP_ICS:
    ic = ICS[key]
    _, pr, vr, _ = simulate_rebound_ias15(ic["x0"], ic["v0"], ic["m"], cfg.G, T, N_SAMPLES)
    REF[key] = pr

print("[trigsweep] sweeping adapt_thresh x ICs ...")
R = {}
for key in SWEEP_ICS:
    ic = ICS[key]; pr = REF[key]; mm = ic["m"]
    for th in SWEEP_THRESH:
        _, pm, vm, perf = simulate_leapfrog_hybrid(
            ic["x0"], ic["v0"], mm, model, cfg, DT_REF, T, N_SAMPLES, adapt_thresh=th)
        nsh = np.asarray(perf["n_sub_history"], dtype=np.int64)
        delta = rms_sep(pm, pr)
        com = (pm * mm[None, :, None]).sum(1, keepdims=True) / mm.sum()
        maxcom = float(np.max(np.linalg.norm(pm - com, axis=2)))
        R[(key, th)] = dict(
            r_min_min = perf["r_min_min"],
            nsub_avg  = float(nsh.mean()),
            nsub_max  = int(nsh.max()),
            nsub_jump = int(np.max(np.abs(np.diff(nsh)))) if nsh.size > 1 else 0,
            total     = int(perf["total_substeps"]),
            rms       = float(np.sqrt(np.mean(delta**2))),
            final_rms = float(delta[-1]),
            lam       = float(fit_log_slope(times_arr, delta)[0]),
            maxdE     = max_dE_pct(pm, vm, mm, cfg.G, EPS),
            bounded   = bool(np.all(np.isfinite(pm)) and maxcom < 10.0),
        )
        r = R[(key, th)]
        print(f"  {key:<18} th={th:.2f}: minr={r['r_min_min']:.4f} "
              f"nsub(avg/max/jump)={r['nsub_avg']:.2f}/{r['nsub_max']}/{r['nsub_jump']} "
              f"RMS={r['rms']:.3e} dE={r['maxdE']:.4f}% bnd={r['bounded']}")

# ---------------- verification guard at adapt_thresh = 0.05 ----------------
print("\n[trigsweep] VERIFICATION GUARD (adapt_thresh=0.05 must reproduce baseline):")
guard_ok = True; guard_lines = []
for key in SWEEP_ICS:
    r = R[(key, 0.05)]
    exp = EXPECTED.get(key, {})
    exp_lam = exp.get("lambda"); exp_rms = exp.get("rms")
    dE_exp = BASE_DRIFT[key]
    dE_ok  = abs(r["maxdE"] - dE_exp) <= max(0.05 * dE_exp, 0.05)
    lam_ok = (exp_lam is None) or abs(r["lam"] - exp_lam) < 0.005
    rms_ok = (exp_rms is None) or abs(r["final_rms"] - exp_rms) / max(abs(exp_rms), 1e-6) < 0.10
    ok = dE_ok and lam_ok and rms_ok
    guard_ok = guard_ok and ok
    line = (f"  {key:<18} dE={r['maxdE']:.4f}% (exp~{dE_exp}) "
            f"lam={r['lam']:+.4f} (exp {exp_lam:+.4f}) "
            f"finRMS={r['final_rms']:.4f} (exp {exp_rms}) -> {'OK' if ok else 'MISMATCH'}")
    guard_lines.append(line); print(line)

if not guard_ok:
    msg = "VERIFICATION GUARD FAILED - baseline not reproduced at adapt_thresh=0.05. STOPPING."
    print("\n" + msg)
    with open(os.path.join(PHASEA_OUT, "ic4_trigger_sweep.txt"), "w", encoding="utf-8") as f:
        f.write("IC4 TRIGGER SWEEP - ABORTED\n" + msg + "\n\n" + "\n".join(guard_lines) + "\n")
    raise SystemExit(msg)

# ---------------- table ----------------
L = []
L.append("IC4 TRIGGER SWEEP - does lowering the sub-stepping trigger fix IC4's energy drift?")
L.append("=" * 118)
L.append(f"T={T}yr  dt={DT_REF}yr  n_samples={N_SAMPLES}  max_substeps=16  energy=softened PE (eps={EPS}, G={cfg.G})")
L.append("Integrator/forces/ias15/masses/dt/sampling identical to multi_ic_eval_v3.py; ONLY adapt_thresh varies.")
L.append("min r_min = closest macro-step pair distance over the run (engages only if < adapt_thresh).")
L.append("nsub_jump = max |n_sub[k]-n_sub[k-1]| over the run (symplecticity-kick proxy).")
L.append("")
hdr = (f"{'IC':<18} {'adapt':>6} {'min r_min':>10} {'nsub_avg':>9} {'nsub_max':>9} "
       f"{'nsub_jump':>10} {'total_sub':>10} {'RMS_0:T':>11} {'|dE/E0|max':>12} {'bounded':>8}")
L.append(hdr); L.append("-" * len(hdr))
for key in SWEEP_ICS:
    for th in SWEEP_THRESH:
        r = R[(key, th)]
        L.append(f"{key:<18} {th:>6.2f} {r['r_min_min']:>10.4f} {r['nsub_avg']:>9.2f} "
                 f"{r['nsub_max']:>9d} {r['nsub_jump']:>10d} {r['total']:>10d} "
                 f"{r['rms']:>11.4e} {r['maxdE']:>11.4f}% {str(r['bounded']):>8}")
    L.append("")

# ---------------- verdict ----------------
ic4_base = R[("IC4_hierarchical", 0.05)]
def regressed(th):
    bad = []
    for key in ["IC1_default", "IC3_tight", "IC6_near_circular"]:
        rb = R[(key, 0.05)]; rt = R[(key, th)]
        if rt["maxdE"] > rb["maxdE"] * 1.05 + 0.01:
            bad.append(f"{key}:dE {rb['maxdE']:.3f}->{rt['maxdE']:.3f}%")
        if rt["rms"] > rb["rms"] * 1.05 + 1e-6:
            bad.append(f"{key}:RMS {rb['rms']:.3e}->{rt['rms']:.3e}")
    return bad

ic4_below3  = [th for th in SWEEP_THRESH if th != 0.05
               and R[("IC4_hierarchical", th)]["maxdE"] < 3.0
               and R[("IC4_hierarchical", th)]["rms"] <= ic4_base["rms"] * 1.05]
ic4_dropped = [th for th in SWEEP_THRESH if th != 0.05
               and R[("IC4_hierarchical", th)]["maxdE"] < ic4_base["maxdE"] * 0.9]
clean3 = [th for th in ic4_below3 if not regressed(th)]

if clean3:
    verdict = "PASS"; best = min(clean3, key=lambda t: R[("IC4_hierarchical", t)]["maxdE"])
elif ic4_below3:
    verdict = "PARTIAL"; best = min(ic4_below3, key=lambda t: R[("IC4_hierarchical", t)]["maxdE"])
elif ic4_dropped:
    verdict = "PARTIAL"; best = min(ic4_dropped, key=lambda t: R[("IC4_hierarchical", t)]["maxdE"])
else:
    verdict = "FAIL"; best = None

L.append("=" * 118)
L.append(f"VERDICT: {verdict}")
if best is not None:
    r4 = R[("IC4_hierarchical", best)]; reg = regressed(best)
    L.append(f"  Best IC4 trigger: adapt_thresh={best}  |dE/E0|max {ic4_base['maxdE']:.4f}% -> {r4['maxdE']:.4f}%"
             f"  (RMS_0:T {ic4_base['rms']:.4e} -> {r4['rms']:.4e})")
    L.append(f"  IC4 n_sub at best: avg={r4['nsub_avg']:.2f} max={r4['nsub_max']} jump={r4['nsub_jump']}  (min r_min={r4['r_min_min']:.4f})")
    L.append(f"  Collateral regressions at adapt_thresh={best}: {reg if reg else 'none'}")
else:
    L.append(f"  IC4 |dE/E0|max never drops below 3% / never materially improves. Baseline {ic4_base['maxdE']:.4f}%.")
    for th in SWEEP_THRESH:
        r4 = R[("IC4_hierarchical", th)]
        eng = "ENGAGES" if r4["r_min_min"] < th else "never fires"
        L.append(f"    adapt_thresh={th}: min r_min={r4['r_min_min']:.4f} -> trigger {eng}; nsub_avg={r4['nsub_avg']:.2f}, dE={r4['maxdE']:.4f}%")
L.append("")
L.append("Verification guard (adapt_thresh=0.05) reproduced the known baseline:")
L.extend(guard_lines)

txt = "\n".join(L)
with open(os.path.join(PHASEA_OUT, "ic4_trigger_sweep.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print("\n" + txt)

# ---------------- PNG: IC4 (target) + IC1 (regression check) max|dE/E0| vs adapt_thresh ----------------
ths = np.array(SWEEP_THRESH, float)
ic4_dE = np.array([R[("IC4_hierarchical", th)]["maxdE"] for th in SWEEP_THRESH], float)
ic1_dE = np.array([R[("IC1_default", th)]["maxdE"] for th in SWEEP_THRESH], float)
ic4_dE = np.where(np.isfinite(ic4_dE), ic4_dE, np.nan)
ic1_dE = np.where(np.isfinite(ic1_dE), ic1_dE, np.nan)
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.plot(ths, ic4_dE, "o-", color="#9333EA", lw=1.9, label="IC4 hierarchical (target)")
ax.plot(ths, ic1_dE, "s--", color="#2563A6", lw=1.7, label="IC1 default (regression check)")
ax.axhline(3.0, color="gray", ls=":", lw=1.0, label="3% target")
ax.set_xlabel("adapt_thresh (AU)"); ax.set_ylabel("max |dE/E0|  (%, softened PE)")
ax.set_title("IC4 energy drift vs sub-stepping trigger threshold")
ax.set_yscale("log"); ax.set_xticks(SWEEP_THRESH)
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PHASEA_OUT, "ic4_trigger_sweep.png"), dpi=200); plt.close()

print(f"\n[trigsweep] VERDICT: {verdict}")
print(f"[trigsweep] wrote {PHASEA_OUT}/ic4_trigger_sweep.txt + ic4_trigger_sweep.png")
