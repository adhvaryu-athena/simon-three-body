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
# DT-REFINEMENT DRIVER (replaces the trigger-sweep driver).
# Pure fixed-step leapfrog (adapt_thresh=0.0 => trigger never fires) at several
# dt, vs the published adaptive baseline (adapt_thresh=0.05, dt=0.04).
# Force model / NN / ias15 / masses / T / energy diagnostic are byte-identical;
# only the trigger (off) and macro dt change.
# force_evals = total_substeps + 1  (1 pre-loop compute_acc; every in-loop force
# call increments total_substeps -> exact, no instrumentation needed).
# ============================================================================
PHASEA_OUT = "weekend_phaseA"
os.makedirs(PHASEA_OUT, exist_ok=True)
EPS = cfg.eps  # 3e-4, unchanged

REFINE_ICS = ["IC1_default", "IC4_hierarchical"]
DT_LIST    = [0.04, 0.02, 0.01, 0.005]
BASE_DRIFT = {"IC1_default": 1.39, "IC4_hierarchical": 13.77}

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

print("\n[dtrefine] ias15 references (IC1, IC4) ...")
REF = {}
for key in REFINE_ICS:
    ic = ICS[key]
    _, pr, vr, _ = simulate_rebound_ias15(ic["x0"], ic["v0"], ic["m"], cfg.G, T, N_SAMPLES)
    REF[key] = pr

def run_one(key, mode, ath, dt):
    ic = ICS[key]; mm = ic["m"]; pr = REF[key]
    _, pm, vm, perf = simulate_leapfrog_hybrid(
        ic["x0"], ic["v0"], mm, model, cfg, dt, T, N_SAMPLES, adapt_thresh=ath)
    valid = ~np.all(pm == 0.0, axis=(1, 2))   # drop trailing UNWRITTEN zero-samples
    n_unwritten = int((~valid).sum())
    pm, vm, prv = pm[valid], vm[valid], pr[valid]
    delta = rms_sep(pm, prv)
    finite = bool(np.all(np.isfinite(pm)))
    if finite:
        com = (pm * mm[None, :, None]).sum(1, keepdims=True) / mm.sum()
        maxcom = float(np.max(np.linalg.norm(pm - com, axis=2)))
    else:
        maxcom = float("inf")
    r = dict(key=key, mode=mode, ath=ath, dt=dt,
             maxdE=max_dE_pct(pm, vm, mm, cfg.G, EPS),
             rms=(float(np.sqrt(np.mean(delta**2))) if finite else float("inf")),
             bounded=bool(finite and maxcom < 10.0),
             fevals=int(perf["total_substeps"]) + 1,
             n_unwritten=n_unwritten)
    print(f"  {key:<18} {mode:<8} ath={ath:.2f} dt={dt:<6}: "
          f"dE={r['maxdE']:.4f}% RMS={r['rms']:.3e} bnd={r['bounded']} fevals={r['fevals']} unwritten={n_unwritten}")
    return r

print("[dtrefine] running adaptive baseline + fixed-step dt sweep ...")
rows = []
for key in REFINE_ICS:
    rows.append(run_one(key, "adaptive", 0.05, 0.04))   # published baseline
    for dt in DT_LIST:
        rows.append(run_one(key, "fixed", 0.0, dt))      # pure fixed-step

def get(key, mode, dt):
    return next(r for r in rows if r["key"] == key and r["mode"] == mode and r["dt"] == dt)

def dE_str(key, dt):
    v = get(key, "fixed", dt)["maxdE"]
    return "EJECT" if not np.isfinite(v) else f"{v:.3f}%"

# ---------------- verification guard ----------------
print("\n[dtrefine] VERIFICATION GUARD (adaptive baseline must reproduce):")
guard_ok = True; guard_lines = []
for key in REFINE_ICS:
    r = get(key, "adaptive", 0.04)
    exp = BASE_DRIFT[key]
    ok = abs(r["maxdE"] - exp) <= max(0.05*exp, 0.05) and r["bounded"]
    guard_ok = guard_ok and ok
    line = f"  {key:<18} adaptive dt=0.04: dE={r['maxdE']:.4f}% (exp~{exp}) bounded={r['bounded']} -> {'OK' if ok else 'MISMATCH'}"
    guard_lines.append(line); print(line)
if not guard_ok:
    msg = "VERIFICATION GUARD FAILED - adaptive baseline not reproduced. STOPPING."
    print("\n" + msg)
    with open(os.path.join(PHASEA_OUT, "ic4_dt_refinement.txt"), "w", encoding="utf-8") as f:
        f.write("IC4 DT REFINEMENT - ABORTED\n" + msg + "\n\n" + "\n".join(guard_lines) + "\n")
    raise SystemExit(msg)

# ---------------- answer the three questions ----------------
def coarsest_clean(key):
    ds = sorted([r["dt"] for r in rows if r["key"] == key and r["mode"] == "fixed"
                 and r["maxdE"] < 1.0 and r["bounded"]], reverse=True)
    return ds[0] if ds else None

ic4_clean_dt = coarsest_clean("IC4_hierarchical")
ic1_clean_dt = coarsest_clean("IC1_default")
both_dt = None
for dt in sorted(DT_LIST, reverse=True):
    r1 = get("IC1_default", "fixed", dt); r4 = get("IC4_hierarchical", "fixed", dt)
    if r1["maxdE"] < 1.0 and r1["bounded"] and r4["maxdE"] < 1.0 and r4["bounded"]:
        both_dt = dt; break
adapt_fe = {k: get(k, "adaptive", 0.04)["fevals"] for k in REFINE_ICS}

# ---------------- table ----------------
L = []
L.append("IC4 DT REFINEMENT - is the drift macro-step under-resolution (fixable by smaller UNIFORM dt)?")
L.append("=" * 104)
L.append(f"T={T}yr  n_samples={N_SAMPLES}  energy=softened PE (eps={EPS}, G={cfg.G})  force model+NN unchanged.")
L.append("Pure fixed-step leapfrog = adapt_thresh=0.0 (symplectic). Adaptive baseline = adapt_thresh=0.05, dt=0.04.")
L.append("force_evals = total_substeps + 1 (cost metric).")
L.append("NOTE: trailing UNWRITTEN zero-samples (harness float-accumulation artifact) masked before all metrics.")
L.append("")
hdr = f"{'IC':<18} {'mode':<9} {'adapt':>6} {'dt':>7} {'|dE/E0|max':>13} {'RMS_0:T':>11} {'bounded':>8} {'force_evals':>12}"
L.append(hdr); L.append("-" * len(hdr))
for key in REFINE_ICS:
    for r in [rr for rr in rows if rr["key"] == key]:
        dE = "inf" if not np.isfinite(r["maxdE"]) else f"{r['maxdE']:.4f}%"
        rms = "inf" if not np.isfinite(r["rms"]) else f"{r['rms']:.4e}"
        L.append(f"{key:<18} {r['mode']:<9} {r['ath']:>6.2f} {r['dt']:>7} {dE:>13} {rms:>11} "
                 f"{str(r['bounded']):>8} {r['fevals']:>12d}")
    L.append("")

# ---------------- decisions ----------------
L.append("=" * 104)
L.append("DECISIONS:")
if ic4_clean_dt is not None:
    r = get("IC4_hierarchical", "fixed", ic4_clean_dt)
    L.append(f"  Q1  IC4 13.77% collapses with smaller UNIFORM dt. Coarsest fixed dt with <1% & bounded "
             f"= {ic4_clean_dt} yr (dE={r['maxdE']:.4f}%, RMS={r['rms']:.3e}).")
else:
    L.append(f"  Q1  IC4 never reaches <1% & bounded at any tested fixed dt (smallest tried = {min(DT_LIST)}).")
L.append("      IC4 fixed-step trend: " + ", ".join(f"dt={d}:{dE_str('IC4_hierarchical', d)}" for d in DT_LIST))
if ic1_clean_dt is not None:
    r = get("IC1_default", "fixed", ic1_clean_dt)
    L.append(f"  Q2  IC1 (ejects with sub-stepping off at dt=0.04): largest UNIFORM dt that is bounded & <1% "
             f"= {ic1_clean_dt} yr (dE={r['maxdE']:.4f}%, RMS={r['rms']:.3e}).")
else:
    L.append(f"  Q2  IC1 is NOT bounded-and-<1% at ANY tested uniform dt down to {min(DT_LIST)} yr "
             f"-> uniform dt does NOT robustly bound IC1 in this range.")
L.append("      IC1 fixed-step trend: " + ", ".join(f"dt={d}:{dE_str('IC1_default', d)}" for d in DT_LIST))
if both_dt is not None:
    fe = get("IC1_default", "fixed", both_dt)["fevals"]
    L.append(f"  Q3  Coarsest UNIFORM dt making BOTH clean = {both_dt} yr, force_evals={fe}.")
    for key in REFINE_ICS:
        L.append(f"        vs {key} adaptive baseline ({adapt_fe[key]} fevals): ratio = {fe/adapt_fe[key]:.2f}x")
else:
    L.append(f"  Q3  No single tested uniform dt makes BOTH clean -> no both-clean cost to quote; IC1 is the blocker (see Q2).")
L.append("")
L.append("Verification guard:")
L.extend(guard_lines)

txt = "\n".join(L)
with open(os.path.join(PHASEA_OUT, "ic4_dt_refinement.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print("\n" + txt)

# ---------------- PNG: max|dE/E0| vs dt (log-log), IC1 & IC4, mark adaptive baseline ----------------
def fixed_curve(key):
    xs, ys = [], []
    for dt in DT_LIST:
        r = get(key, "fixed", dt)
        xs.append(dt); ys.append(r["maxdE"] if np.isfinite(r["maxdE"]) else np.nan)
    return np.array(xs), np.array(ys)

fig, ax = plt.subplots(figsize=(7.4, 5.0))
x1, y1 = fixed_curve("IC1_default")
x4, y4 = fixed_curve("IC4_hierarchical")
ax.plot(x4, y4, "o-", color="#9333EA", lw=1.9, label="IC4 fixed-step")
ax.plot(x1, y1, "s-", color="#2563A6", lw=1.9, label="IC1 fixed-step")
b4 = get("IC4_hierarchical", "adaptive", 0.04); b1 = get("IC1_default", "adaptive", 0.04)
ax.scatter([0.04], [b4["maxdE"]], marker="*", s=190, color="#9333EA", edgecolor="k", zorder=5,
           label=f"IC4 adaptive baseline ({b4['maxdE']:.1f}%)")
if np.isfinite(b1["maxdE"]):
    ax.scatter([0.04], [b1["maxdE"]], marker="*", s=190, color="#2563A6", edgecolor="k", zorder=5,
               label=f"IC1 adaptive baseline ({b1['maxdE']:.2f}%)")
ytop = np.nanmax([np.nanmax(y4), b4["maxdE"]]) * 2.0
for dt in DT_LIST:
    r = get("IC1_default", "fixed", dt)
    if not r["bounded"]:
        ax.annotate("IC1\neject", (dt, ytop), color="#2563A6", fontsize=7, ha="center", va="top")
ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="1% target")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("uniform dt (yr, log)"); ax.set_ylabel("max |dE/E0|  (%, softened PE, log)")
ax.set_title("Fixed-step energy drift vs dt  (stars = adaptive baseline @ dt=0.04)")
ax.set_xticks(DT_LIST); ax.set_xticklabels([str(d) for d in DT_LIST])
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7.5, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(PHASEA_OUT, "ic4_dt_refinement.png"), dpi=200); plt.close()

print(f"\n[dtrefine] wrote {PHASEA_OUT}/ic4_dt_refinement.txt + ic4_dt_refinement.png")
