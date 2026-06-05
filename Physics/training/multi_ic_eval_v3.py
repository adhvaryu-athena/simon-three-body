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
                             device="cpu", dtype=torch.float32):
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

    adapt_thresh = 0.05
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
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
            total_substeps += 1
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

# ── Run simulations ───────────────────────────────────────────────────────────
print(f"\n[multi_ic_v3] Running 7 ICs  |  dt={DT_REF}  |  T={T}yr")
print(f"{'='*70}")

results = {}

for name, ic in ICS.items():
    new_tag      = " [NEW]"            if ic["new"]               else ""
    unstable_tag = " [KNOWN UNSTABLE]" if name in KNOWN_UNSTABLE  else ""
    print(f"\n[{ic['label']}]{new_tag}{unstable_tag}")
    print(f"  {ic['desc']}")
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]

    # ias15 ground truth
    t0 = time.perf_counter()
    tr, pr, vr, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, T, N_SAMPLES)
    t_ias = time.perf_counter() - t0
    print(f"  ias15:  {t_ias:.3f}s")

    # SIMON
    t0 = time.perf_counter()
    tm, pm, vm, perf_m = simulate_leapfrog_hybrid(
        x0, v0, m, model, cfg, DT_REF, T, N_SAMPLES)
    t_simon = time.perf_counter() - t0
    speedup = perf_r["total_time_sec"] / max(perf_m["total_time_sec"], 1e-12)
    print(f"  SIMON:  {t_simon:.3f}s  speedup={speedup:.2f}x  "
          f"NN_frac={perf_m['avg_fallback_frac']:.4f}")

    # Metrics
    delta     = rms_sep(pm, pr)
    slope, win= fit_log_slope(tr, delta)
    final_rms = float(delta[-1])
    bounded   = final_rms < EJECTION_THRESHOLD

    print(f"  lambda: {slope:.4f}/yr  |  final_RMS: {final_rms:.4f} AU  |  "
          f"bounded: {'YES' if bounded else 'NO (ejection)'}")

    # Verification check (IC1-IC6 only; IC7 is new)
    if name in EXPECTED:
        exp    = EXPECTED[name]
        lam_ok = abs(slope - exp["lambda"]) < LAMBDA_TOL
        rms_ok = abs(final_rms - exp["rms"]) / max(abs(exp["rms"]), 1e-6) < RMS_TOL_FRAC
        bnd_ok = bounded == exp["bounded"]
        if lam_ok and rms_ok and bnd_ok:
            print(f"  VERIFICATION: PASS")
        else:
            print(f"  VERIFICATION: MISMATCH  "
                  f"lambda_ok={lam_ok}  rms_ok={rms_ok}  bounded_ok={bnd_ok}")
            print(f"    Expected: lambda={exp['lambda']:.4f}  "
                  f"rms={exp['rms']:.4f}  bounded={exp['bounded']}")
    else:
        print(f"  VERIFICATION: N/A (new IC -- no prior expected value)")

    results[name] = {
        "ic":        ic,
        "tr":        tr,
        "pr":        pr,
        "pm":        pm,
        "delta":     delta,
        "slope":     slope,
        "win":       win,
        "final_rms": final_rms,
        "bounded":   bounded,
        "speedup":   speedup,
        "nn_frac":   perf_m["avg_fallback_frac"],
        "E_total":   ic["E_total"],
        "unstable":  name in KNOWN_UNSTABLE,
    }

# ── Statistics (bounded ICs only; known unstable excluded) ───────────────────
all_names   = list(results.keys())
all_slopes  = [results[n]["slope"]     for n in all_names]
all_rms     = [results[n]["final_rms"] for n in all_names]
all_bounded = [results[n]["bounded"]   for n in all_names]
all_speedup = [results[n]["speedup"]   for n in all_names]

bounded_slopes = [s for s, b in zip(all_slopes, all_bounded) if b]
bounded_rms    = [r for r, b in zip(all_rms,    all_bounded) if b]

mean_slope = np.mean(bounded_slopes) if bounded_slopes else float("nan")
std_slope  = np.std(bounded_slopes)  if bounded_slopes else float("nan")
mean_rms   = np.mean(bounded_rms)    if bounded_rms    else float("nan")
std_rms    = np.std(bounded_rms)     if bounded_rms    else float("nan")

print(f"\n{'='*70}")
print("MULTI-IC SUMMARY (v3 -- 7 ICs)")
print(f"{'='*70}")
print(f"  {'IC':<28} {'lambda(/yr)':>11} {'RMS(AU)':>10} "
      f"{'bounded':>9} {'speedup':>9} {'note':>16}")
print(f"  {'-'*85}")
for name, r in results.items():
    b      = "YES" if r["bounded"] else "NO"
    note   = "[UNSTABLE]" if r["unstable"] else ("[NEW]" if r["ic"]["new"] else "")
    print(f"  {r['ic']['label']:<28} {r['slope']:>11.4f} "
          f"{r['final_rms']:>10.4f} {b:>9} {r['speedup']:>8.2f}x {note:>16}")
print(f"  {'-'*85}")
print(f"  {'Mean (bounded only)':<28} {mean_slope:>11.4f} {mean_rms:>10.4f}")
print(f"  {'Std  (bounded only)':<28} {std_slope:>11.4f} {std_rms:>10.4f}")
print(f"  Bounded: {sum(all_bounded)}/{len(all_bounded)} ICs  "
      f"| Known unstable: {len(KNOWN_UNSTABLE)}/7 (IC2, IC5)")


# ── Helper: bar label placement (handles negative lambda e.g. IC6) ────────────
def bar_label_y(bar, offset_pos=0.003, offset_neg=-0.003):
    """Place label above bar for positive height, below for negative."""
    h = bar.get_height()
    if h >= 0:
        return h + offset_pos, "bottom"
    else:
        return h + offset_neg, "top"


# ── Helper: apply hatching to known-unstable IC bars ─────────────────────────
def hatch_unstable(bars, names_list):
    for bar, nm in zip(bars, names_list):
        if nm in KNOWN_UNSTABLE:
            bar.set_hatch("//")
            bar.set_edgecolor("#555555")
            bar.set_linewidth(0.8)


# ── Figure 1: Divergence curves (2x4 panel for 7 ICs) ───────────────────────
n_ics  = len(results)
n_cols = 4
n_rows = math.ceil(n_ics / n_cols)   # = 2

with plt.rc_context({
    "font.size": 9,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
}):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7.2), sharex=False)
    axes = axes.flatten()

    for idx, (name, r) in enumerate(results.items()):
        ax  = axes[idx]
        col = r["ic"]["color"]

        # Figure-only display flags. This changes only chart annotations/titles,
        # not simulation results, statistics, tables, or saved numeric output.
        chart_phys_unstable = name in {
            "IC2_near_equal",
            "IC5_high_ecc",
            "IC7_high_ecc_stable",
        }

        ax.semilogy(r["tr"], r["delta"], color=col, lw=1.6, alpha=0.9)

        # Fit-window shading retained, but no repeated legend in every panel.
        ax.axvspan(r["win"][0], r["win"][1], alpha=0.10, color="gray")

        status     = "bounded" if r["bounded"] else "ejection"
        status_col = "#166534" if r["bounded"] else "#DC2626"

        # Remove visible version wording such as "(V2)" from figure titles only.
        # Also avoid duplicate "unstable" wording in IC5 and simplify IC7 title.
        clean_label = r["ic"]["label"].replace(" (V2)", "")
        if name == "IC7_high_ecc_stable":
            clean_label = "IC7: High Ecc (unstable)"

        ax.set_title(
            f"{clean_label}\n"
            f"λ={r['slope']:.4f}/yr, RMS={r['final_rms']:.3f} AU ({status})",
            color=status_col
        )

        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("RMS error (AU)")
        ax.grid(True, which="both", alpha=0.25)

        # Remove visually unhelpful near-zero initial range while preserving
        # all computed data/results. This affects display only.
        positive_delta = np.asarray(r["delta"])[np.asarray(r["delta"]) > 0]
        if len(positive_delta) > 0:
            ymin = max(1e-3, float(np.nanmin(positive_delta)) * 0.8)
            ymax = max(float(np.nanmax(positive_delta)) * 1.25, ymin * 10)
            ax.set_ylim(ymin, ymax)

        if chart_phys_unstable:
            ax.text(
                0.97, 0.03, "physically\nunstable",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=6.5,
                color="#DC2626",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85)
            )

    # Hide unused panel (index 7 for 7 ICs in 2x4 grid)
    for idx in range(n_ics, n_rows * n_cols):
        axes[idx].set_visible(False)

    # No figure-level title: the LaTeX caption explains the figure.
    plt.tight_layout()
    div_path = os.path.join(OUT_DIR, "multi_ic_divergence_v3.png")
    plt.savefig(div_path, dpi=300)
    plt.close()

print(f"\n[multi_ic_v3] Saved {div_path}")


# ── Figure 2: Summary bar charts (3 panels, 7 bars each) ─────────────────────
with plt.rc_context({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
}):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    names_list = list(results.keys())

    # Short labels prevent overlap in the paper figure and remove visible V2 wording.
    disp_names = [
        "IC1\nDefault",
        "IC2\nNear-equal",
        "IC3\nTight pair",
        "IC4\nHierarchical",
        "IC5\nHigh-ecc",
        "IC6\nNear-circ.",
        "IC7\nHigh-ecc",
    ]

    colors = [r["ic"]["color"] for r in results.values()]
    slopes = [r["slope"]       for r in results.values()]
    rmss   = [r["final_rms"]   for r in results.values()]
    speeds = [r["speedup"]     for r in results.values()]
    bar_w  = 0.55

    # Panel 1: lambda
    ax = axes[0]
    bars = ax.bar(
        range(n_ics), slopes,
        color=colors, edgecolor="white", linewidth=1.0, width=bar_w
    )
    hatch_unstable(bars, names_list)

    ax.axhline(
        mean_slope,
        color="black", lw=1.3, linestyle="--",
        label=f"Mean bounded = {mean_slope:.4f}/yr"
    )

    for bar, val in zip(bars, slopes):
        ly, va = bar_label_y(bar, offset_pos=0.0025, offset_neg=-0.0025)
        ax.text(
            bar.get_x() + bar.get_width()/2, ly,
            f"{val:.4f}",
            ha="center", va=va,
            fontsize=6.8, fontweight="bold"
        )

    ax.set_xticks(range(n_ics))
    ax.set_xticklabels(disp_names)
    ax.set_ylabel("Divergence rate λ (1/yr)")
    ax.set_title("Divergence rate")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: final RMS (log scale)
    ax = axes[1]
    bars2 = ax.bar(
        range(n_ics), rmss,
        color=colors, edgecolor="white", linewidth=1.0, width=bar_w
    )
    hatch_unstable(bars2, names_list)

    ax.set_yscale("log")

    for bar, val, b in zip(bars2, rmss, all_bounded):
        col_ = "#166534" if b else "#DC2626"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.25,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=col_
        )

    ax.axhline(
        EJECTION_THRESHOLD,
        color="#DC2626", lw=1.1, linestyle=":",
        label=f"Ejection threshold = {EJECTION_THRESHOLD:.0f} AU"
    )

    ax.set_xticks(range(n_ics))
    ax.set_xticklabels(disp_names)
    ax.set_ylabel("Final RMS error (AU, log scale)")
    ax.set_title(f"Final RMS at T = 100 yr ({sum(all_bounded)}/{len(all_bounded)} bounded)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: speedup
    ax = axes[2]
    bars3 = ax.bar(
        range(n_ics), speeds,
        color=colors, edgecolor="white", linewidth=1.0, width=bar_w
    )
    hatch_unstable(bars3, names_list)

    ax.axhline(
        1.0,
        color="black", lw=1.1, linestyle="--",
        label="ias15 = 1.00×"
    )

    for bar, val in zip(bars3, speeds):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            f"{val:.2f}×",
            ha="center", va="bottom",
            fontsize=6.8,
            fontweight="bold"
        )

    ax.set_xticks(range(n_ics))
    ax.set_xticklabels(disp_names)
    ax.set_ylabel("Speedup vs ias15")
    ax.set_title(f"Computational speedup (dt = {DT_REF} yr)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # No figure-level title: the LaTeX caption explains the figure.
    plt.tight_layout()
    res_path = os.path.join(OUT_DIR, "multi_ic_results_v3.png")
    plt.savefig(res_path, dpi=300)
    plt.close()

print(f"[multi_ic_v3] Saved {res_path}")


# ── Figure 3: Trajectory overlays (2x4 panel, body 1) ────────────────────────
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9))
axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    ax      = axes[idx]
    col     = r["ic"]["color"]
    new_tag = " [NEW]" if r["ic"]["new"] else ""
    pr      = r["pr"]; pm = r["pm"]
    ax.plot(pr[:,1,0], pr[:,1,1], "-",  color="#2563A6",
            lw=1.5, alpha=0.9, label="ias15")
    ax.plot(pm[:,1,0], pm[:,1,1], "--", color=col,
            lw=1.2, alpha=0.9, label="SIMON")
    status     = "BOUNDED" if r["bounded"] else "EJECTION"
    status_col = "#166534"  if r["bounded"] else "#DC2626"
    ax.set_title(f"{r['ic']['label']}{new_tag}  [{status}]",
                 color=status_col, fontsize=9)
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

for idx in range(n_ics, n_rows * n_cols):
    axes[idx].set_visible(False)

fig.suptitle(
    "Body 1 Trajectory Overlay (ias15 vs SIMON)  --  7 Initial Conditions\n"
    "IC7 [NEW] = high-eccentricity redesigned",
    fontsize=12, fontweight="bold")
plt.tight_layout()
traj_path = os.path.join(OUT_DIR, "multi_ic_trajectories_v3.png")
plt.savefig(traj_path, dpi=300); plt.close()
print(f"[multi_ic_v3] Saved {traj_path}")


# ── Summary text file ─────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "multi_ic_summary_v3.txt")
r2  = results["IC2_near_equal"]
r5  = results["IC5_high_ecc"]

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("MULTI-INITIAL-CONDITION EVALUATION SUMMARY (v3 -- 7 ICs)\n")
    f.write(f"SIMON  |  T={T}yr  |  dt={DT_REF}yr  |  n_samples={N_SAMPLES}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write("=" * 72 + "\n\n")

    f.write("INITIAL CONDITIONS:\n")
    for name, ic in ICS.items():
        r       = results[name]
        new_tag = " [NEW]"      if ic["new"]             else ""
        ust_tag = " [UNSTABLE]" if name in KNOWN_UNSTABLE else ""
        f.write(f"\n  {ic['label']}{new_tag}{ust_tag}\n")
        f.write(f"    {ic['desc']}\n")
        f.write(f"    Regime: {ic['regime']}\n")
        f.write(f"    m  = {ic['m'].tolist()}\n")
        f.write(f"    E_total = {r['E_total']:.5f}  (negative = bound)\n")

    f.write("\n\nRESULTS:\n")
    f.write(f"\n  {'IC':<28} {'lambda(/yr)':>12} {'RMS(AU)':>10} "
            f"{'bounded':>9} {'speedup':>9} {'NN_frac':>9} {'note':>11}\n")
    f.write(f"  {'-'*90}\n")
    for name, r in results.items():
        b    = "YES" if r["bounded"] else "NO"
        note = "[UNSTABLE]" if r["unstable"] else ("[NEW]" if r["ic"]["new"] else "")
        f.write(f"  {r['ic']['label']:<28} {r['slope']:>12.4f} "
                f"{r['final_rms']:>10.4f} {b:>9} "
                f"{r['speedup']:>8.2f}x {r['nn_frac']:>9.4f} {note:>11}\n")
    f.write(f"  {'-'*90}\n")
    f.write(f"  {'Mean (bounded only)':<28} {mean_slope:>12.4f} {mean_rms:>10.4f}\n")
    f.write(f"  {'Std  (bounded only)':<28} {std_slope:>12.4f} {std_rms:>10.4f}\n\n")
    f.write(f"  Bounded: {sum(all_bounded)}/{len(all_bounded)} ICs\n")
    f.write(f"  Known unstable: IC2, IC5 (confirmed by check_ic2.py, check_ic5.py)\n\n")

    f.write("=" * 72 + "\n")
    f.write("IC2 HONEST FRAMING (per mentor feedback):\n")
    f.write("=" * 72 + "\n")
    f.write(f"\n  SIMON final separation:  {r2['final_rms']:.2f} AU\n")
    f.write(f"  ias15 final separation:  58.94 AU  (from check_ic2.py)\n")
    f.write(f"  Ratio SIMON/ias15:       {r2['final_rms']/58.94:.1f}x\n\n")
    f.write("  CORRECT FRAMING: IC2 is dynamically unstable; both integrators\n")
    f.write("  predict ejection. Post-ejection separations differ by ~2x,\n")
    f.write("  consistent with chaotic divergence amplifying integrator\n")
    f.write("  differences after the system becomes dynamically unbound.\n\n")

    f.write("=" * 72 + "\n")
    f.write("IC5 HONEST FRAMING (confirmed by check_ic5.py):\n")
    f.write("=" * 72 + "\n")
    f.write(f"\n  SIMON final separation:  {r5['final_rms']:.2f} AU\n")
    f.write(f"  ias15 final separation:  47.08 AU  (from check_ic5.py)\n")
    f.write(f"  ias15 ejection time:     34.8 yr\n")
    f.write(f"  Ratio SIMON/ias15:       {r5['final_rms']/47.08:.2f}x\n\n")
    f.write("  CORRECT FRAMING: IC5 is dynamically unstable; both integrators\n")
    f.write("  predict ejection. IC5 was redesigned as IC7 (gentler periapsis,\n")
    f.write("  Body 2 further away) to obtain a bounded high-eccentricity case.\n\n")

    f.write("=" * 72 + "\n")
    f.write("IC7 DESIGN RATIONALE (high-eccentricity redesign):\n")
    f.write("=" * 72 + "\n")
    r7 = results["IC7_high_ecc_stable"]
    f.write(f"\n  IC5 failure: v=2.2 at r=0.3 AU. Body 2 at 2.0 AU got kicked out.\n")
    f.write(f"  IC7 fix: v=1.6 at r=0.6 AU. Body 2 at 4.0 AU.\n")
    f.write(f"  Same eccentricity character: 1.24x v_circ (IC5 was 1.20x v_circ).\n")
    f.write(f"  Safer: periapsis at 0.6 vs 0.3 AU, Body 2 separation 4.0 vs 2.0 AU.\n")
    f.write(f"  Estimated eccentricity e~0.52, semi-major axis a~1.25 AU.\n")
    f.write(f"  Result: lambda={r7['slope']:.4f}/yr  RMS={r7['final_rms']:.4f} AU  "
            f"bounded={'YES' if r7['bounded'] else 'NO'}\n\n")

    f.write("=" * 72 + "\n")
    f.write("KEY NUMBERS FOR PAPER:\n")
    f.write("=" * 72 + "\n")
    if bounded_slopes:
        f.write(f"  lambda range (bounded ICs): "
                f"[{min(bounded_slopes):.4f}, {max(bounded_slopes):.4f}] /yr\n")
    f.write(f"  lambda mean +/- std (bounded): "
            f"{mean_slope:.4f} +/- {std_slope:.4f} /yr\n")
    if bounded_rms:
        f.write(f"  RMS range (bounded ICs):    "
                f"[{min(bounded_rms):.4f}, {max(bounded_rms):.4f}] AU\n")
    f.write(f"  Bounded:         {sum(all_bounded)}/{len(all_bounded)} ICs\n")
    f.write(f"  Unstable:        2/7 ICs (IC2 and IC5 -- physically confirmed)\n")
    f.write(f"  Speedup range:   [{min(all_speedup):.2f}x, {max(all_speedup):.2f}x]\n")
    f.write(f"\n  IC1-IC6 verification: see VERIFICATION lines in terminal output.\n")
    f.write(f"  IC7 is new -- result above.\n")

print(f"[multi_ic_v3] Saved {txt_path}")
print(f"\n{'='*70}")
print("[multi_ic_v3] Done.")
print(f"  lambda: {mean_slope:.4f} +/- {std_slope:.4f} /yr  (bounded ICs)")
print(f"  bounded: {sum(all_bounded)}/{len(all_bounded)} ICs")
print(f"  Output: {OUT_DIR}/")
print(f"{'='*70}")
print("[multi_ic_v3] SCRIPT COMPLETE -- all figures and summary saved successfully.")
print(f"{'='*70}")
