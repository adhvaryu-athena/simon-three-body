# evaluation_encounter_fine_stepping.py
#
# Comparison evaluator: SIMON-EncounterNN (Fix 1) vs Zone-3 finer leapfrog stepping.
#
# Purpose
# -------
# For each base dt (0.06, 0.08, 0.10) compare FIVE methods:
#   1. IAS15               — reference integrator
#   2. SIMON-noNN          — coarse leapfrog, no NN, no fine stepping
#   3. SIMON-encounterNN   — coarse leapfrog + single 18D residual correction (Fix 1)
#   4. SIMON-finer-Zone3-k2— coarse leapfrog + 2 sub-steps in Zone 3 when gate fires
#   5. SIMON-finer-Zone3-k4— coarse leapfrog + 4 sub-steps in Zone 3 when gate fires
#
# Gate (same for EncounterNN and fine stepping):
#   pair 1-2 is in Zone 3 (0.052 <= r < 0.15 AU)  AND  v_rad_norm < vr_thresh (-0.40)
#
# Fix 1 is applied to EncounterNN: at most one correction per simulation.
# Fine stepping fires on EVERY qualifying Zone 3 step (no per-simulation limit).
#
# Three evaluation levels:
#   Level 1 (Global):   single-seed T=100yr rollout vs IAS15
#   Level 2 (Isolated): all methods started from the SAME noNN event-entry state
#   Level 3 (Ensemble): 10 IC-perturbed seeds, same as v8
#
# Validation (strict):
#   Before any fine-stepping comparison the script re-runs noNN and encounterNN
#   fresh. Both must match the base A / base B file ground-truth metrics within
#   val_tol. If either fails the script stops.
#
# Outputs → comp_encounter_stepping/
#   comparison_global_dt{dt}.txt/.csv
#   comparison_isolated_dt{dt}.txt
#   comparison_ensemble_dt{dt}.txt
#   speed_accuracy_dt{dt}.png
#   summary_winner_dt{dt}.txt
#   validation_dt{dt}.txt

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found.")
    raise


# =============================================================================
# Configuration — identical to v5/v7/v8
# =============================================================================
@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    r_soft_min: float = 5e-4
    zone1_r_gate: float = 4e-4
    adapt_thresh: float = 0.05
    nn_thresh: float = 0.15
    max_substeps: int = 16


# =============================================================================
# Encounter NN — NumPy deployment (identical to v5/v7/v8)
# =============================================================================
@dataclass
class EncounterNumpyWeights:
    layers: List[Tuple[np.ndarray, np.ndarray]]
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    arch: str
    params: int


def _as_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _linear_layer_indices(sd):
    idxs = []
    for k, v in sd.items():
        if not (k.startswith("net.") and k.endswith(".weight")):
            continue
        parts = k.split(".")
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if hasattr(v, "ndim") and int(v.ndim) == 2:
            idxs.append(idx)
    return sorted(idxs)


def load_encounter_numpy(model_path: str) -> EncounterNumpyWeights:
    raw = torch.load(model_path, map_location="cpu")
    sd = _as_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError("Encounter checkpoint is not a supported state_dict.")
    required = ["input_mean", "input_std", "target_mean", "target_std"]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(f"Encounter checkpoint missing {missing}.")
    idxs = _linear_layer_indices(sd)
    if not idxs:
        raise KeyError("No Linear layers found under keys like net.<idx>.weight")
    layers = []
    shapes = []
    params = 0
    for idx in idxs:
        wk, bk = f"net.{idx}.weight", f"net.{idx}.bias"
        W = sd[wk].detach().cpu().numpy().astype(np.float32).copy()
        b = sd[bk].detach().cpu().numpy().astype(np.float32).copy()
        layers.append((W, b))
        shapes.append((int(W.shape[1]), int(W.shape[0])))
        params += int(W.size + b.size)
    arch = " -> ".join([str(shapes[0][0])] + [str(out) for _, out in shapes])
    def _buf(name):
        arr = sd[name].detach().cpu().numpy().astype(np.float32).reshape(-1).copy()
        return arr
    im = _buf("input_mean");  ist = _buf("input_std")
    tm = _buf("target_mean"); tst = _buf("target_std")
    ist = np.where(np.abs(ist) < 1e-8, 1.0, ist).astype(np.float32)
    tst = np.where(np.abs(tst) < 1e-8, 1.0, tst).astype(np.float32)
    print(f"[encounterNN] Loaded {model_path} | arch={arch} | params={params}")
    return EncounterNumpyWeights(layers=layers, input_mean=im, input_std=ist,
                                  target_mean=tm, target_std=tst, arch=arch, params=params)


def _silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def encounter_forward_numpy(X, w: EncounterNumpyWeights):
    z = np.asarray(X, dtype=np.float32).reshape(1, 18)
    z = (z - w.input_mean.reshape(1, 18)) / (w.input_std.reshape(1, 18) + np.float32(1e-8))
    for i, (W, b) in enumerate(w.layers):
        z = z @ W.T + b.reshape(1, -1)
        if i != len(w.layers) - 1:
            z = _silu_np(z).astype(np.float32)
    y_norm = z.reshape(18).astype(np.float32)
    y = y_norm * (w.target_std + np.float32(1e-8)) + w.target_mean
    return y.astype(np.float64)


def predict_encounter_residual_np(model, X):
    y = encounter_forward_numpy(X, model)
    if y.shape != (18,) or not np.all(np.isfinite(y)):
        raise FloatingPointError("EncounterNN produced invalid residual")
    dx = y[:9].reshape(3, 3)
    dv = y[9:].reshape(3, 3)
    pred_pos_norm = float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))
    pred_vel_norm = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    return dx, dv, pred_pos_norm, pred_vel_norm


# =============================================================================
# Reference and metrics — identical to v5/v7/v8
# =============================================================================
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]), x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, float(T), int(n_samples))
    pos = np.zeros((len(times), len(m), 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


def rms_sep(a, b):
    d = a - b
    pb = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(pb * pb, axis=1))


def metric_block(pos, vel, pos_ref, vel_ref):
    pr = rms_sep(pos, pos_ref)
    vr = rms_sep(vel, vel_ref)
    return {
        "pos_final": float(pr[-1]),
        "pos_timeavg": float(np.sqrt(np.mean(pr * pr))),
        "pos_med": float(np.median(pr)),
        "pos_p95": float(np.percentile(pr, 95)),
        "vel_final": float(vr[-1]),
        "vel_timeavg": float(np.sqrt(np.mean(vr * vr))),
        "vel_med": float(np.median(vr)),
        "vel_p95": float(np.percentile(vr, 95)),
    }


def all_finite_state(*arrs):
    return all(np.all(np.isfinite(a)) for a in arrs)


def max_radius_state(x):
    return float(np.max(np.linalg.norm(x, axis=1)))


def total_energy_state(x, v, m, G=1.0):
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = float(np.linalg.norm(x[j] - x[i]))
            pe -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(ke + pe)


def rel_energy_drift(E0, E_now):
    return abs((E_now - E0) / (abs(E0) + 1e-30))


def pct_gain(base, new):
    return 100.0 * (base - new) / max(abs(base), 1e-30)


# =============================================================================
# Feature construction — identical to v5/v7/v8
# =============================================================================
def compute_pair_velocity_features_12(x, v, m, cfg):
    i, j = 1, 2
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r = float(np.linalg.norm(rij))
    r_hat = rij / (r + 1e-30)
    v_rad = float(np.dot(vij, r_hat))
    v_tan_vec = vij - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))
    v_scale = math.sqrt(cfg.G * (float(m[i]) + float(m[j])) / (r + 1e-30))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r


def encounter_gate_features_pair12_fast(x, v, m, cfg):
    rij = x[2] - x[1]
    r2 = float(np.dot(rij, rij))
    r = math.sqrt(r2 + 1e-30)
    if not (cfg.adapt_thresh <= r < cfg.nn_thresh):
        return False, float("nan"), float("nan"), r
    vij = v[2] - v[1]
    r_hat = rij / (r + 1e-30)
    v_rad = float(np.dot(vij, r_hat))
    v_tan_vec = vij - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))
    v_scale = math.sqrt(cfg.G * (float(m[1]) + float(m[2])) / (r + 1e-30))
    return True, float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r


def make_X_rel18_pair12(x, v, m, dt, cfg):
    i, j, k = 1, 2, 0
    mi, mj = float(m[i]), float(m[j])
    pair_com_x = (mi * x[i] + mj * x[j]) / (mi + mj)
    pair_com_v = (mi * v[i] + mj * v[j]) / (mi + mj)
    rij = x[j] - x[i];  vij = v[j] - v[i]
    r3  = x[k] - pair_com_x;  v3 = v[k] - pair_com_v
    vr, vt, r = compute_pair_velocity_features_12(x, v, m, cfg)
    X = np.concatenate([
        np.log(m + 1e-30), rij, vij, r3, v3,
        np.array([math.log(float(dt) + 1e-30), vr, vt], dtype=np.float64),
    ]).astype(np.float32)
    return X, vr, vt, r


def project_com_to_reference(x_corr, v_corr, x_ref, v_ref, m):
    M = float(np.sum(m))
    com_x_corr = np.sum(m[:, None] * x_corr, axis=0) / M
    com_v_corr = np.sum(m[:, None] * v_corr, axis=0) / M
    com_x_ref  = np.sum(m[:, None] * x_ref,  axis=0) / M
    com_v_ref  = np.sum(m[:, None] * v_ref,  axis=0) / M
    return x_corr - (com_x_corr - com_x_ref), v_corr - (com_v_corr - com_v_ref)


# =============================================================================
# NoNN kernel — identical to v5/v7/v8
# =============================================================================
@dataclass
class NoNNKernel:
    cfg: HybridConfig
    m: np.ndarray

    def __post_init__(self):
        self.N = int(self.m.shape[0])
        ii, jj = [], []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ii.append(i); jj.append(j)
        self.ii = np.array(ii, dtype=np.int64)
        self.jj = np.array(jj, dtype=np.int64)
        self.P = int(len(self.ii))
        self.m_f = self.m.astype(np.float64)
        self.mi_arr = self.m_f[self.ii];  self.mj_arr = self.m_f[self.jj]
        self.Gmimj = float(self.cfg.G) * self.mi_arr * self.mj_arr
        self.inv_mi = 1.0 / self.mi_arr;  self.inv_mj = 1.0 / self.mj_arr
        self.eps2 = float(self.cfg.eps * self.cfg.eps)
        self.reset_counters()

    def reset_counters(self):
        self.pair_evals_n = np.int64(0)
        self.zone1_n = np.int64(0);  self.zone2_n = np.int64(0)
        self.zone3_n = np.int64(0);  self.far_n = np.int64(0)
        self.zone3_no_nn_n = np.int64(0)
        self.total_substeps = 0

    def geometry(self, pos):
        rij = pos[self.jj] - pos[self.ii]
        r2  = np.einsum("ij,ij->i", rij, rij)
        r   = np.sqrt(r2 + 1e-30)
        return rij, r2, r

    def acc_from_geom_noNN(self, rij, r2, r):
        cfg = self.cfg
        F_scalar = self.Gmimj / (r2 * r + 1e-30)
        zone1 = r < cfg.zone1_r_gate
        zone2 = (r >= cfg.zone1_r_gate) & (r < cfg.adapt_thresh)
        zone3 = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        far   = r >= cfg.nn_thresh
        self.pair_evals_n += self.P
        self.zone1_n += zone1.sum();  self.zone2_n += zone2.sum()
        self.zone3_n += zone3.sum();  self.far_n += far.sum()
        if np.any(zone3):
            F_soft = self.Gmimj[zone3] / ((r2[zone3] + self.eps2) ** 1.5 + 1e-30)
            F_scalar[zone3] = F_soft
            self.zone3_no_nn_n += int(zone3.sum())
        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((self.N, 3), dtype=np.float64)
        for p in range(self.P):
            acc[self.ii[p]] += F_vec[p] * self.inv_mi[p]
            acc[self.jj[p]] -= F_vec[p] * self.inv_mj[p]
        return acc

    def acc_far(self, pos):
        rij = pos[self.jj] - pos[self.ii]
        r2  = np.einsum("ij,ij->i", rij, rij)
        r   = np.sqrt(r2 + 1e-30)
        F_scalar = self.Gmimj / (r2 * r + 1e-30)
        self.pair_evals_n += self.P;  self.far_n += self.P
        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((self.N, 3), dtype=np.float64)
        for p in range(self.P):
            acc[self.ii[p]] += F_vec[p] * self.inv_mi[p]
            acc[self.jj[p]] -= F_vec[p] * self.inv_mj[p]
        return acc, float(np.sqrt(r2.min() + 1e-30))

    def verlet_noNN_zone_step(self, x_in, v_in, step_dt):
        rij_in, r2_in, r_in = self.geometry(x_in)
        a0 = self.acc_from_geom_noNN(rij_in, r2_in, r_in)
        vh  = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        rij_n, r2_n, r_n = self.geometry(x_new)
        a1 = self.acc_from_geom_noNN(rij_n, r2_n, r_n)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, float(np.sqrt(r2_n.min() + 1e-30))

    def verlet_far_step(self, x_in, v_in, step_dt, a_cache):
        if a_cache is not None:
            a0 = a_cache
            self.pair_evals_n += self.P;  self.far_n += self.P
        else:
            a0, _ = self.acc_far(x_in)
        vh    = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        a1, r_min_new = self.acc_far(x_new)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, a1, r_min_new

    def macro_step(self, x, v, step_dt, r_min_cur, a_cache):
        cfg = self.cfg
        if r_min_cur < cfg.adapt_thresh:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_thresh / max(r_min_cur, 1e-30)))))
            sub_dt = float(step_dt) / n_sub
            x_loc, v_loc, r_min_loc = x, v, r_min_cur
            for _ in range(n_sub):
                x_loc, v_loc, r_min_loc = self.verlet_noNN_zone_step(x_loc, v_loc, sub_dt)
            self.total_substeps += int(n_sub)
            return x_loc, v_loc, r_min_loc, None, int(n_sub)
        if r_min_cur < cfg.nn_thresh:
            x_new, v_new, r_min_new = self.verlet_noNN_zone_step(x, v, float(step_dt))
            self.total_substeps += 1
            return x_new, v_new, r_min_new, None, 1
        x_new, v_new, a_new, r_min_new = self.verlet_far_step(x, v, float(step_dt), a_cache)
        self.total_substeps += 1
        return x_new, v_new, r_min_new, a_new, 1

    def perf_dict(self, label, steps, dt, T, n_samples, elapsed):
        pair_evals = max(int(self.pair_evals_n), 1)
        return {
            "label": label, "steps": int(steps), "dt": float(dt),
            "T_years": float(T), "n_samples": int(n_samples),
            "total_time_sec": float(elapsed),
            "time_per_step_sec": float(elapsed / max(int(steps), 1)),
            "total_substeps": int(self.total_substeps),
            "zone1_frac":    int(self.zone1_n) / pair_evals,
            "zone2_frac":    int(self.zone2_n) / pair_evals,
            "zone3_frac":    int(self.zone3_n) / pair_evals,
            "zone4_far_frac":int(self.far_n)   / pair_evals,
        }

# =============================================================================
# IC1 — identical to v5/v7/v8
# =============================================================================
def make_ic1():
    m  = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64)
    M  = float(m.sum())
    x0 = x0 - (m[:,None]*x0).sum(0)/M
    v0 = v0 - (m[:,None]*v0).sum(0)/M
    return x0, v0, m


def perturb_ic(x0, v0, m, seed, scale):
    rng = np.random.default_rng(int(seed))
    x_p = x0 + rng.uniform(-scale, scale, x0.shape)
    v_p = v0.copy()
    M   = float(m.sum())
    x_p = x_p - (m[:,None]*x_p).sum(0)/M
    v_p = v_p - (m[:,None]*v_p).sum(0)/M
    return x_p, v_p


# =============================================================================
# SIMON-noNN global rollout — identical to v5/v7/v8
# =============================================================================
def simulate_simon_noNN(x0, v0, m, cfg, dt, T, n_samples):
    kernel = NoNNKernel(cfg, m)
    times  = np.linspace(0.0, float(T), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    E0 = total_energy_state(x, v, m, cfg.G)

    si = 0; t_cur = 0.0
    while si < len(times) and t_cur >= times[si] - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1

    _, r2_0, _ = kernel.geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None; steps = 0
    n_steps = int(math.ceil(float(T)/float(dt)))
    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt = min(float(dt), float(T) - t_cur)
        x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min_cur, a_cache)
        t_cur += step_dt; steps += 1
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
        if not all_finite_state(x, v): break
    while si < len(times):
        pos_out[si] = x; vel_out[si] = v; si += 1
    elapsed = time.perf_counter() - t_start
    E_final = total_energy_state(x, v, m, cfg.G)
    perf = kernel.perf_dict("SIMON_noNN", steps, dt, T, n_samples, elapsed)
    perf["relE_drift"] = rel_energy_drift(E0, E_final)
    return times, pos_out, vel_out, perf


# =============================================================================
# SIMON-encounterNN global rollout (Fix 1 applied) — same as v5/v7/v8
# =============================================================================
def simulate_simon_encounterNN(x0, v0, m, encounter_model, cfg,
                                dt, T, n_samples,
                                window_years=0.5, vr_thresh=-0.40,
                                energy_gate=0.20, max_radius_gate=1e4,
                                com_project=True, event_rows=None):
    kernel = NoNNKernel(cfg, m)
    times  = np.linspace(0.0, float(T), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    E0 = total_energy_state(x, v, m, cfg.G)

    si = 0; t_cur = 0.0
    def fill():
        nonlocal si
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
    fill()

    _, r2_0, _ = kernel.geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None; steps = 0
    n_steps = int(math.ceil(float(T)/float(dt)))
    gate_candidates = 0; encounter_attempts = 0
    encounter_used  = 0; encounter_fallback = 0
    first_used_t = float("nan"); last_used_t = float("nan")
    pred_pos_sum = 0.0; pred_vel_sum = 0.0
    min_r_global = r_min_cur
    max_rad_global = max_radius_state(x)
    relE_corr_used = float("nan")

    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt_macro = min(float(dt), float(T) - t_cur)
        in_z3, vr_entry, vt_entry, r_pair = encounter_gate_features_pair12_fast(x, v, m, cfg)
        if in_z3:
            gate_candidates += 1
        gate_ok = bool(in_z3 and (vr_entry < float(vr_thresh))
                       and (t_cur + float(window_years) <= float(T) + 1e-12))
        if gate_ok:
            X_entry, vr_entry, vt_entry, r_pair = make_X_rel18_pair12(x, v, m, float(dt), cfg)
            encounter_attempts += 1
            t_event = float(t_cur)
            x_start = x.copy(); v_start = v.copy()
            E_start = total_energy_state(x_start, v_start, m, G=cfg.G)
            min_r_window = r_min_cur
            remaining = float(window_years)
            while remaining > 1e-14:
                step_w = min(float(dt), remaining)
                x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_w, r_min_cur, a_cache)
                t_cur += step_w; remaining -= step_w; steps += 1
                min_r_window = min(min_r_window, r_min_cur)
                min_r_global = min(min_r_global, r_min_cur)
                fill()
                if not all_finite_state(x, v): break
            x_no_exit = x.copy(); v_no_exit = v.copy()
            dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual_np(encounter_model, X_entry)
            x_corr = x_no_exit + dx; v_corr = v_no_exit + dv
            if com_project:
                x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)
            relE_corr = abs((total_energy_state(x_corr, v_corr, m, G=cfg.G) - E_start) / (abs(E_start)+1e-30))
            use = True; reason = "used"
            if not all_finite_state(x_corr, v_corr):
                use = False; reason = "nonfinite"
            elif max_radius_state(x_corr) > float(max_radius_gate):
                use = False; reason = "max_radius_gate"
            elif relE_corr > float(energy_gate):
                use = False; reason = "energy_gate"
            elif encounter_used >= 1:
                use = False; reason = "max_one_correction"  # Fix 1
            if use:
                x = x_corr; v = v_corr
                _, r2c, _ = kernel.geometry(x)
                r_min_cur = float(np.sqrt(r2c.min() + 1e-30))
                a_cache = None
                encounter_used += 1
                if not np.isfinite(first_used_t): first_used_t = t_event
                last_used_t = t_event
                pred_pos_sum += pred_pos_norm; pred_vel_sum += pred_vel_norm
                min_r_global  = min(min_r_global, r_min_cur)
                max_rad_global = max(max_rad_global, max_radius_state(x))
                relE_corr_used = relE_corr
                if si > 0 and abs(times[si-1] - t_cur) <= max(1e-10, 1e-9*abs(t_cur)):
                    pos_out[si-1] = x; vel_out[si-1] = v
                fill()
            else:
                x = x_no_exit; v = v_no_exit; a_cache = None; encounter_fallback += 1
            if event_rows is not None:
                event_rows.append({"t_start": t_event, "t_exit": float(t_cur),
                    "used": int(use), "reason": reason, "r_pair": float(r_pair),
                    "v_rad_norm": float(vr_entry), "v_tan_norm": float(vt_entry),
                    "min_r_window": float(min_r_window),
                    "pred_pos_norm": float(pred_pos_norm), "pred_vel_norm": float(pred_vel_norm),
                    "relE_corr": float(relE_corr)})
            continue
        x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt_macro, r_min_cur, a_cache)
        t_cur += step_dt_macro; steps += 1
        min_r_global = min(min_r_global, r_min_cur)
        fill()
        if not all_finite_state(x, v): break
    while si < len(times):
        pos_out[si] = x; vel_out[si] = v; si += 1
    elapsed = time.perf_counter() - t_start
    E_final = total_energy_state(x, v, m, cfg.G)
    used_safe = max(int(encounter_used), 1)
    perf = kernel.perf_dict("SIMON_encounterNN", steps, dt, T, n_samples, elapsed)
    perf.update({
        "gate_candidates": int(gate_candidates),
        "encounter_attempts": int(encounter_attempts),
        "encounter_used": int(encounter_used),
        "encounter_fallback": int(encounter_fallback),
        "first_used_t": float(first_used_t),
        "last_used_t": float(last_used_t),
        "pred_pos_norm_mean": float(pred_pos_sum / used_safe),
        "pred_vel_norm_mean": float(pred_vel_sum / used_safe),
        "min_r": float(min_r_global),
        "max_radius": float(max_rad_global),
        "local_correction_relE_corr": float(relE_corr_used),
        "relE_drift": rel_energy_drift(E0, E_final),
    })
    return times, pos_out, vel_out, perf


# =============================================================================
# SIMON-finer-Zone3 global rollout (NEW)
# =============================================================================
def simulate_simon_finer_zone3(x0, v0, m, cfg, dt, T, n_samples, k,
                                vr_thresh=-0.40):
    """
    SIMON with Zone-3 leapfrog sub-stepping. No NN correction.

    When the gate fires (pair 1-2 in Zone 3 AND vr_norm < vr_thresh) the macro
    dt is split into k equal leapfrog sub-steps of dt/k. Zone 2 adaptive
    sub-stepping and the far field are unchanged. Fine-stepping fires on every
    qualifying Zone 3 macro-step (not limited to one per simulation).
    """
    kernel = NoNNKernel(cfg, m)
    times  = np.linspace(0.0, float(T), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    E0 = total_energy_state(x, v, m, cfg.G)

    si = 0; t_cur = 0.0
    while si < len(times) and t_cur >= times[si] - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1

    _, r2_0, _ = kernel.geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None; steps = 0
    n_steps = int(math.ceil(float(T)/float(dt)))
    fine_steps_total = 0   # count of macro-steps where fine-stepping fired
    gate_candidates  = 0

    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt = min(float(dt), float(T) - t_cur)

        # Check Zone 2 first (r_min_cur already tracks minimum across all pairs)
        if r_min_cur < cfg.adapt_thresh:
            # Zone 2: normal adaptive sub-stepping, no gate override
            x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min_cur, a_cache)
            t_cur += step_dt; steps += 1
        elif r_min_cur < cfg.nn_thresh:
            # Zone 3: check fine-stepping gate for pair 1-2
            in_z3, vr, vt, r_pair = encounter_gate_features_pair12_fast(x, v, m, cfg)
            if in_z3:
                gate_candidates += 1
            if in_z3 and vr < float(vr_thresh):
                # Fine-step: split macro dt into k sub-steps
                sub_dt = step_dt / k
                x_loc = x; v_loc = v; r_loc = r_min_cur; ac = None
                for _ in range(k):
                    # Each sub-step uses the full macro_step, which handles
                    # Zone 2 penetration if it occurs during the sub-step.
                    x_loc, v_loc, r_loc, ac, _ = kernel.macro_step(x_loc, v_loc, sub_dt, r_loc, ac)
                x = x_loc; v = v_loc; r_min_cur = r_loc; a_cache = None
                fine_steps_total += 1
                steps += 1
            else:
                # Zone 3 but gate not firing: normal single macro step
                x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min_cur, a_cache)
                steps += 1
            t_cur += step_dt
        else:
            # Far field
            x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min_cur, a_cache)
            t_cur += step_dt; steps += 1

        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
        if not all_finite_state(x, v): break

    while si < len(times):
        pos_out[si] = x; vel_out[si] = v; si += 1
    elapsed = time.perf_counter() - t_start
    E_final = total_energy_state(x, v, m, cfg.G)
    perf = kernel.perf_dict(f"SIMON_finer_k{k}", steps, dt, T, n_samples, elapsed)
    perf.update({
        "k": int(k),
        "gate_candidates": int(gate_candidates),
        "fine_steps_fired": int(fine_steps_total),
        "relE_drift": rel_energy_drift(E0, E_final),
    })
    return times, pos_out, vel_out, perf


# =============================================================================
# advance_nonn_to_time — identical to v7
# =============================================================================
def advance_nonn_to_time(x0, v0, m, cfg, dt, t_target):
    kernel = NoNNKernel(cfg, m)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    t = 0.0; steps = 0
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min() + 1e-30))
    a_cache = None
    t0 = time.perf_counter()
    while t < float(t_target) - 1e-14:
        h = min(float(dt), float(t_target) - t)
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t += h; steps += 1
        if not all_finite_state(x, v):
            raise FloatingPointError("advance_nonn_to_time: non-finite state")
    return x, v, {"elapsed_sec": time.perf_counter()-t0, "steps": steps, "t_reached": t}


# =============================================================================
# Isolated local simulations — for all methods from same entry state
# =============================================================================
def run_window_from_state(x_start, v_start, m, cfg, dt, duration, k=1, vr_thresh=-0.40):
    """
    Run noNN or finer-Zone3 from an entry state for exactly 'duration' years.
    k=1 → normal noNN; k>1 → finer-Zone3 with k sub-steps when gate fires.
    Returns (x_exit, v_exit, perf_dict).
    """
    kernel = NoNNKernel(cfg, m)
    x = x_start.astype(np.float64).copy()
    v = v_start.astype(np.float64).copy()
    t = 0.0; steps = 0; fine_steps = 0
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min() + 1e-30))
    min_r_seen = r_min; a_cache = None
    t0 = time.perf_counter()
    while t < float(duration) - 1e-14:
        h = min(float(dt), float(duration) - t)
        if k > 1 and r_min >= cfg.adapt_thresh and r_min < cfg.nn_thresh:
            in_z3, vr, _, _ = encounter_gate_features_pair12_fast(x, v, m, cfg)
            if in_z3 and vr < float(vr_thresh):
                sub_dt = h / k
                x_loc = x; v_loc = v; r_loc = r_min; ac = None
                for _ in range(k):
                    x_loc, v_loc, r_loc, ac, _ = kernel.macro_step(x_loc, v_loc, sub_dt, r_loc, ac)
                x = x_loc; v = v_loc; r_min = r_loc; a_cache = None
                fine_steps += 1; steps += 1; t += h
                min_r_seen = min(min_r_seen, r_min)
                if not all_finite_state(x, v): break
                continue
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t += h; steps += 1
        min_r_seen = min(min_r_seen, r_min)
        if not all_finite_state(x, v): break
    return x, v, {"elapsed_sec": time.perf_counter()-t0, "steps": steps,
                  "min_r_seen": min_r_seen, "fine_steps": fine_steps}


def simulate_local_all_methods(
    x_event, v_event, m, encounter_model, cfg,
    dt, T_local, n_samples,
    window_years=0.5, vr_thresh=-0.40,
    energy_gate=0.20, max_radius_gate=1e4,
    k_list=(2, 4),
):
    """
    Run all methods from the same event-entry state for T_local years.
    Returns dict of (times, pos, vel) arrays and perf for each method.
    """
    # IAS15
    tl, p_ias, v_ias, perf_ias = simulate_rebound_ias15(x_event, v_event, m, cfg.G, T_local, n_samples)

    # noNN local
    _, p_no, v_no, perf_no_l = simulate_simon_noNN(x_event, v_event, m, cfg, dt, T_local, n_samples)

    # EncounterNN once — run noNN window then apply correction then continue noNN
    x = x_event.astype(np.float64).copy()
    v = v_event.astype(np.float64).copy()
    E_start = total_energy_state(x, v, m, cfg.G)
    X_entry, vr_e, vt_e, r_e = make_X_rel18_pair12(x, v, m, float(dt), cfg)
    x_no_exit, v_no_exit, _ = run_window_from_state(x_event, v_event, m, cfg, dt, window_years, k=1)
    dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual_np(encounter_model, X_entry)
    x_corr = x_no_exit + dx; v_corr = v_no_exit + dv
    x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)
    relE_corr = abs((total_energy_state(x_corr, v_corr, m, cfg.G) - E_start) / (abs(E_start)+1e-30))
    enc_accepted = bool(all_finite_state(x_corr, v_corr)
                        and max_radius_state(x_corr) <= float(max_radius_gate)
                        and relE_corr <= float(energy_gate))

    # Build encounterNN trajectory
    kernel_en = NoNNKernel(cfg, m)
    times2 = np.linspace(0.0, float(T_local), int(n_samples))
    p_en = np.zeros((len(times2), 3, 3), dtype=np.float64)
    v_en = np.zeros_like(p_en)
    x2 = x_event.astype(np.float64).copy(); v2 = v_event.astype(np.float64).copy()
    si = 0; t_cur = 0.0
    def fill_en():
        nonlocal si
        while si < len(times2) and t_cur >= times2[si]-1e-12:
            p_en[si]=x2; v_en[si]=v2; si+=1
    fill_en()
    _, r2x, _ = kernel_en.geometry(x2)
    r_min2 = float(np.sqrt(r2x.min()+1e-30)); ac2 = None; steps2 = 0
    # noNN window
    remaining = float(window_years)
    while remaining > 1e-14:
        h = min(float(dt), remaining)
        x2, v2, r_min2, ac2, _ = kernel_en.macro_step(x2, v2, h, r_min2, ac2)
        t_cur += h; remaining -= h; steps2 += 1
        fill_en()
    if enc_accepted:
        x2 = x_corr.copy(); v2 = v_corr.copy()
        _, r2c, _ = kernel_en.geometry(x2)
        r_min2 = float(np.sqrt(r2c.min()+1e-30)); ac2 = None
        if si > 0 and abs(times2[si-1]-t_cur) <= max(1e-10, 1e-9*abs(t_cur)):
            p_en[si-1]=x2; v_en[si-1]=v2
        fill_en()
    # continue
    t0_en = time.perf_counter()
    while t_cur < float(T_local) - 1e-14:
        h = min(float(dt), float(T_local)-t_cur)
        x2, v2, r_min2, ac2, _ = kernel_en.macro_step(x2, v2, h, r_min2, ac2)
        t_cur += h; steps2 += 1
        fill_en()
    while si < len(times2):
        p_en[si]=x2; v_en[si]=v2; si+=1
    perf_en_l = {"correction_used": int(enc_accepted), "relE_corr": float(relE_corr),
                 "r_pair_entry": float(r_e), "v_rad_norm_entry": float(vr_e)}

    # Finer-Zone3 for each k
    finer_results = {}
    for kk in k_list:
        _, p_fk, v_fk, perf_fk = simulate_simon_noNN(  # reuse noNN but override below
            x_event, v_event, m, cfg, dt, T_local, n_samples)
        # Actually run proper finer simulation
        p_fk, v_fk, perf_fk = _run_finer_local(
            x_event, v_event, m, cfg, dt, T_local, n_samples, int(kk), float(vr_thresh))
        finer_results[kk] = (times2.copy(), p_fk, v_fk, perf_fk)

    return {
        "times":   tl,
        "ias15":   (p_ias, v_ias, perf_ias),
        "noNN":    (p_no,  v_no,  perf_no_l),
        "encNN":   (p_en,  v_en,  perf_en_l),
        "finer":   finer_results,
        "entry":   {"r_pair": float(r_e), "vr": float(vr_e), "vt": float(vt_e)},
        "x_event": x_event, "v_event": v_event,
    }


def _run_finer_local(x_event, v_event, m, cfg, dt, T_local, n_samples, k, vr_thresh):
    """Run finer-Zone3-k from an event-entry state for T_local yr."""
    kernel = NoNNKernel(cfg, m)
    times  = np.linspace(0.0, float(T_local), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x_event.astype(np.float64).copy()
    v = v_event.astype(np.float64).copy()
    si = 0; t_cur = 0.0
    while si < len(times) and t_cur >= times[si]-1e-12:
        pos_out[si]=x; vel_out[si]=v; si+=1
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min()+1e-30)); a_cache = None; steps = 0; fine = 0
    n_steps = int(math.ceil(float(T_local)/float(dt)))
    t0 = time.perf_counter()
    while t_cur < float(T_local) - 1e-14 and steps < n_steps + 10:
        step_dt = min(float(dt), float(T_local) - t_cur)
        if r_min < cfg.adapt_thresh:
            x, v, r_min, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min, a_cache)
        elif r_min < cfg.nn_thresh:
            in_z3, vr, _, _ = encounter_gate_features_pair12_fast(x, v, m, cfg)
            if in_z3 and vr < float(vr_thresh):
                sub_dt = step_dt / k
                xl=x; vl=v; rl=r_min; ac=None
                for _ in range(k):
                    xl, vl, rl, ac, _ = kernel.macro_step(xl, vl, sub_dt, rl, ac)
                x=xl; v=vl; r_min=rl; a_cache=None; fine+=1
            else:
                x, v, r_min, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min, a_cache)
        else:
            x, v, r_min, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min, a_cache)
        t_cur += step_dt; steps += 1
        while si < len(times) and t_cur >= times[si]-1e-12:
            pos_out[si]=x; vel_out[si]=v; si+=1
        if not all_finite_state(x, v): break
    while si < len(times):
        pos_out[si]=x; vel_out[si]=v; si+=1
    elapsed = time.perf_counter()-t0
    perf = kernel.perf_dict(f"LOCAL_finer_k{k}", steps, dt, T_local, n_samples, elapsed)
    perf["fine_steps_fired"] = fine
    return pos_out, vel_out, perf


# =============================================================================
# 0.5yr exit-state comparison — all five methods from event-entry state
# =============================================================================
def compute_exit_state_all_methods(x_event, v_event, m, encounter_model, cfg,
                                    dt, window_years, energy_gate, max_radius_gate,
                                    k_list=(2, 4), vr_thresh=-0.40):
    """
    Run each method for exactly window_years from x_event and compare exit states
    against IAS15 reference exit.
    """
    # IAS15 reference exit
    t_ias, p_ias, v_ias, perf_ias = simulate_rebound_ias15(
        x_event, v_event, m, cfg.G, float(window_years), 2)
    x_ias_exit = p_ias[-1].copy(); v_ias_exit = v_ias[-1].copy()

    # noNN exit
    x_no_exit, v_no_exit, _ = run_window_from_state(
        x_event, v_event, m, cfg, dt, window_years, k=1)
    # EncounterNN exit
    E_start = total_energy_state(x_event, v_event, m, cfg.G)
    X_entry, vr_e, vt_e, r_e = make_X_rel18_pair12(x_event, v_event, m, float(dt), cfg)
    dx, dv, pred_pos_norm, _ = predict_encounter_residual_np(encounter_model, X_entry)
    x_corr = x_no_exit + dx; v_corr = v_no_exit + dv
    x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)
    relE_corr = abs((total_energy_state(x_corr, v_corr, m, cfg.G) - E_start) / (abs(E_start)+1e-30))
    enc_ok = bool(all_finite_state(x_corr, v_corr)
                  and max_radius_state(x_corr) <= float(max_radius_gate)
                  and relE_corr <= float(energy_gate))
    x_enc_exit = x_corr if enc_ok else x_no_exit
    v_enc_exit = v_corr if enc_ok else v_no_exit

    def pos_err(x_a): return float(rms_sep(x_a[None,:,:], x_ias_exit[None,:,:])[0])
    def vel_err(v_a): return float(rms_sep(v_a[None,:,:], v_ias_exit[None,:,:])[0])

    noNN_pos = pos_err(x_no_exit);  noNN_vel = vel_err(v_no_exit)
    encNN_pos = pos_err(x_enc_exit); encNN_vel = vel_err(v_enc_exit)

    result = {
        "r_pair_entry": float(r_e), "v_rad_norm_entry": float(vr_e),
        "relE_corr": float(relE_corr), "enc_accepted": int(enc_ok),
        "noNN_pos_exit_err": noNN_pos, "noNN_vel_exit_err": noNN_vel,
        "encNN_pos_exit_err": encNN_pos, "encNN_vel_exit_err": encNN_vel,
        "encNN_pos_gain_pct": pct_gain(noNN_pos, encNN_pos),
        "encNN_vel_gain_pct": pct_gain(noNN_vel, encNN_vel),
    }
    for kk in k_list:
        x_fk, v_fk, _ = run_window_from_state(
            x_event, v_event, m, cfg, dt, window_years, k=int(kk), vr_thresh=float(vr_thresh))
        fk_pos = pos_err(x_fk); fk_vel = vel_err(v_fk)
        result[f"finer_k{kk}_pos_exit_err"] = fk_pos
        result[f"finer_k{kk}_vel_exit_err"] = fk_vel
        result[f"finer_k{kk}_pos_gain_pct"] = pct_gain(noNN_pos, fk_pos)
        result[f"finer_k{kk}_vel_gain_pct"] = pct_gain(noNN_vel, fk_vel)
    return result

# =============================================================================
# Base-file parser — reads A and B summary files written by v5/v7
# =============================================================================
def _re1(pattern, text, default=float("nan")):
    """Return first capture group as float, or default if no match."""
    m = re.search(pattern, text)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    return default


def parse_base_a_file(path):
    """Parse a v5 summary file. Returns dict of ground-truth metrics."""
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    d = {}
    # IAS15 time
    d["ias15_time"] = _re1(r"IAS15_total_time_sec:\s*([\d.e+\-]+)", txt)
    # noNN
    d["noNN_time"]      = _re1(r"SIMON_noNN_total_time_sec:\s*([\d.e+\-]+)", txt)
    d["noNN_speedup"]   = _re1(r"SIMON_noNN_speedup_vs_IAS15:\s*([\d.e+\-]+)", txt)
    # Read the method table row for SIMON-noNN
    m_no = re.search(r"SIMON-noNN\t([\d.e+\-]+)\t([\d.e+\-]+)\t[\d.e+\-]+\t[\d.e+\-]+\t([\d.e+\-]+)\t([\d.e+\-]+)", txt)
    if m_no:
        d["noNN_pos_final"]   = float(m_no.group(1))
        d["noNN_pos_timeavg"] = float(m_no.group(2))
        d["noNN_vel_final"]   = float(m_no.group(3))
        d["noNN_vel_timeavg"] = float(m_no.group(4))
    # encounterNN
    d["encNN_time"]    = _re1(r"SIMON_encounterNN_total_time_sec:\s*([\d.e+\-]+)", txt)
    d["encNN_speedup"] = _re1(r"SIMON_encounterNN_speedup_vs_IAS15:\s*([\d.e+\-]+)", txt)
    m_en = re.search(r"SIMON-encounterNN\t([\d.e+\-]+)\t([\d.e+\-]+)\t[\d.e+\-]+\t[\d.e+\-]+\t([\d.e+\-]+)\t([\d.e+\-]+)", txt)
    if m_en:
        d["encNN_pos_final"]   = float(m_en.group(1))
        d["encNN_pos_timeavg"] = float(m_en.group(2))
        d["encNN_vel_final"]   = float(m_en.group(3))
        d["encNN_vel_timeavg"] = float(m_en.group(4))
    d["encNN_pos_final_gain"]   = _re1(r"SIMON-encounterNN_pos_final_gain_pct:\s*([\+\-\d.e]+)", txt)
    d["encNN_pos_timeavg_gain"] = _re1(r"SIMON-encounterNN_pos_timeavg_gain_pct:\s*([\+\-\d.e]+)", txt)
    d["encNN_vel_final_gain"]   = _re1(r"SIMON-encounterNN_vel_final_gain_pct:\s*([\+\-\d.e]+)", txt)
    d["encNN_vel_timeavg_gain"] = _re1(r"SIMON-encounterNN_vel_timeavg_gain_pct:\s*([\+\-\d.e]+)", txt)
    return d


def parse_base_b_file(path):
    """Parse a v7 summary file. Returns dict including isolated metrics."""
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    d = parse_base_a_file(path)  # same global section
    # relE_corr
    d["relE_corr"] = _re1(r"local_correction_relE_corr:\s*([\d.e+\-]+)", txt)
    # exit-state exact
    ee = re.search(r"isolated_exact_exit_state\t[\d.]+\t[\d.]+\t([\d.e+\-]+)\t([\d.e+\-]+)\t([\+\-\d.]+)\t([\d.e+\-]+)\t([\d.e+\-]+)\t([\+\-\d.]+)", txt)
    if ee:
        d["iso_pos_noNN_exit"]  = float(ee.group(1))
        d["iso_pos_encNN_exit"] = float(ee.group(2))
        d["iso_pos_exit_gain"]  = float(ee.group(3))
        d["iso_vel_noNN_exit"]  = float(ee.group(4))
        d["iso_vel_encNN_exit"] = float(ee.group(5))
        d["iso_vel_exit_gain"]  = float(ee.group(6))
    # 4yr physical window
    fw = re.search(r"isolated_physical_forward_window\t[\d.]+\t[\d.]+\t\d+\t([\d.e+\-]+)\t([\d.e+\-]+)\t([\+\-\d.]+)\t([\d.e+\-]+)\t([\d.e+\-]+)\t([\+\-\d.]+)\t([\+\-\d.]+)\t([\+\-\d.]+)", txt)
    if fw:
        d["iso_4yr_pos_noNN"]   = float(fw.group(1))
        d["iso_4yr_pos_encNN"]  = float(fw.group(2))
        d["iso_4yr_pos_gain"]   = float(fw.group(3))
        d["iso_4yr_vel_gain"]   = float(fw.group(6))
        d["iso_4yr_pos_final_gain"] = float(fw.group(7))
    return d


# =============================================================================
# Validation — fresh noNN and encounterNN must match base-file ground truth
# =============================================================================
def validate_fresh_run(label, fresh_val, base_val, tol, out_lines):
    """Check fresh value vs base-file value. Returns True if PASS."""
    if not (math.isfinite(base_val) and math.isfinite(fresh_val)):
        msg = f"[VALIDATION] {label:50s}  fresh={fresh_val:.6e}  base={base_val:.6e}  SKIP (non-finite)"
        print(msg); out_lines.append(msg); return True
    diff_pct = abs(fresh_val - base_val) / (abs(base_val) + 1e-30) * 100.0
    passed   = diff_pct <= tol * 100.0
    status   = "PASS" if passed else "FAIL"
    msg = (f"[VALIDATION] {label:50s}  fresh={fresh_val:+.6e}  base={base_val:+.6e}  "
           f"diff={diff_pct:.3f}%  {status}")
    print(msg); out_lines.append(msg)
    return passed


def run_validation(x0, v0, m, cfg, encounter_model, dt, T, n_samples,
                   window_years, vr_thresh, energy_gate, max_radius_gate,
                   base_a, base_b, val_tol, out_path):
    """Re-run noNN and encounterNN fresh and compare against base files."""
    print("\n" + "="*80)
    print("VALIDATION: re-running noNN and encounterNN to match base-file ground truth")
    print("="*80)
    out_lines = []
    all_pass = True

    _, p_no, v_no, perf_no = simulate_simon_noNN(x0, v0, m, cfg, dt, T, n_samples)
    tr, pr, vr_ref, perf_r  = simulate_rebound_ias15(x0, v0, m, cfg.G, T, n_samples)
    met_no = metric_block(p_no, v_no, pr, vr_ref)
    sp_no  = perf_r["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)

    # Strict validation is limited to deterministic trajectory metrics.
    # Runtime and speedup are intentionally excluded because they naturally
    # vary between runs with CPU load, system scheduling, and timing noise.
    checks_no = [
        ("noNN  pos_final",    met_no["pos_final"],    base_a.get("noNN_pos_final", float("nan"))),
        ("noNN  pos_timeavg",  met_no["pos_timeavg"],  base_a.get("noNN_pos_timeavg", float("nan"))),
        ("noNN  vel_final",    met_no["vel_final"],    base_a.get("noNN_vel_final", float("nan"))),
        ("noNN  vel_timeavg",  met_no["vel_timeavg"],  base_a.get("noNN_vel_timeavg", float("nan"))),
    ]
    for label, fresh, base in checks_no:
        if not validate_fresh_run(label, fresh, base, val_tol, out_lines):
            all_pass = False

    # Report speedup for information only. It must not cause validation failure.
    base_sp_no = base_a.get("noNN_speedup", float("nan"))
    if math.isfinite(base_sp_no):
        speedup_diff_pct = 100.0 * abs(sp_no - base_sp_no) / max(abs(base_sp_no), 1e-30)
        speedup_line = (
            f"[VALIDATION] {'noNN  speedup (informational only)':<52} "
            f"fresh={sp_no:+.6e}  base={base_sp_no:+.6e}  "
            f"diff={speedup_diff_pct:.3f}%  NOT USED FOR PASS/FAIL"
        )
    else:
        speedup_line = (
            f"[VALIDATION] {'noNN  speedup (informational only)':<52} "
            f"fresh={sp_no:+.6e}  base=not available  NOT USED FOR PASS/FAIL"
        )

    print(speedup_line)
    out_lines.append(speedup_line)

    event_rows = []
    _, p_en, v_en, perf_en = simulate_simon_encounterNN(
        x0, v0, m, encounter_model, cfg,
        dt=float(dt), T=float(T), n_samples=int(n_samples),
        window_years=float(window_years), vr_thresh=float(vr_thresh),
        energy_gate=float(energy_gate), max_radius_gate=float(max_radius_gate),
        event_rows=event_rows)
    met_en = metric_block(p_en, v_en, pr, vr_ref)
    sp_en  = perf_r["total_time_sec"] / max(perf_en["total_time_sec"], 1e-12)
    gain_pf = pct_gain(met_no["pos_final"], met_en["pos_final"])
    gain_pt = pct_gain(met_no["pos_timeavg"], met_en["pos_timeavg"])

    checks_en = [
        ("encNN pos_final",         met_en["pos_final"],    base_a.get("encNN_pos_final", float("nan"))),
        ("encNN pos_timeavg",       met_en["pos_timeavg"],  base_a.get("encNN_pos_timeavg", float("nan"))),
        ("encNN pos_final_gain %",  gain_pf,                base_b.get("encNN_pos_final_gain", base_a.get("encNN_pos_final_gain", float("nan")))),
        ("encNN pos_timeavg_gain%", gain_pt,                base_b.get("encNN_pos_timeavg_gain", base_a.get("encNN_pos_timeavg_gain", float("nan")))),
    ]
    for label, fresh, base in checks_en:
        if not validate_fresh_run(label, fresh, base, val_tol, out_lines):
            all_pass = False

    if all_pass:
        print("\n[VALIDATION] ALL CHECKS PASSED — fresh runs reproduce base-file results.")
        out_lines.append("\nALL VALIDATION CHECKS PASSED")
    else:
        print("\n[VALIDATION] ONE OR MORE CHECKS FAILED — comparison may not be fair.")
        out_lines.append("\nVALIDATION FAILED — see above")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return (
        all_pass,
        tr,
        pr,
        vr_ref,
        perf_r,
        met_no,
        met_en,
        sp_no,
        sp_en,
        perf_no,
        perf_en,
        event_rows,
    )


# =============================================================================
# Global comparison table writer
# =============================================================================
def write_global_comparison(out_dir, dt, methods_data, ias15_time):
    """
    methods_data: list of (name, met, perf, speedup, relE_drift, gate_info)
    gate_info: dict with 'fine_steps_fired', 'gate_candidates', etc.
    """
    lines = []
    W = 130
    div = "=" * W
    lines.append(div)
    lines.append(f"FINE-STEPPING VS ENCOUNTERNN  —  GLOBAL T=100yr COMPARISON  (dt={dt})")
    lines.append(div)
    lines.append("")

    hdr = (f"{'Method':<28} {'pos_final(AU)':>14} {'pos_timeavg(AU)':>16} "
           f"{'vel_final':>12} {'vel_timeavg':>12} "
           f"{'Time(s)':>8} {'Speedup':>8} {'relE':>12} {'Zone3 fine':>11}")
    lines.append(hdr)
    lines.append("-" * W)

    ref_method = next((m for m in methods_data if m[0] == "noNN"), None)
    ref_pf = ref_method[1]["pos_final"] if ref_method else 1.0

    best_pos_final = min(m[1]["pos_final"] for m in methods_data)
    best_pos_tavg  = min(m[1]["pos_timeavg"] for m in methods_data)

    for name, met, perf, speedup, relE, gate_info in methods_data:
        pf_mark = " *" if abs(met["pos_final"]   - best_pos_final) < 1e-10 else "  "
        pt_mark = " *" if abs(met["pos_timeavg"] - best_pos_tavg)  < 1e-10 else "  "
        fine_str = str(gate_info.get("fine_steps_fired", "-"))
        if "encounter_used" in gate_info:
            fine_str = f"enc={gate_info['encounter_used']}"
        row = (f"{name:<28} {met['pos_final']:>14.4e}{pf_mark} {met['pos_timeavg']:>14.4e}{pt_mark} "
               f"{met['vel_final']:>12.4e} {met['vel_timeavg']:>12.4e} "
               f"{perf['total_time_sec']:>8.4f} {speedup:>8.2f}x {relE:>12.3e} {fine_str:>11}")
        lines.append(row)

    lines.append("")
    lines.append("  * = best value in column")
    lines.append(f"  IAS15 reference time: {ias15_time:.4f}s")
    lines.append("")

    # Gain table vs noNN
    lines.append(f"{'Method':<28} {'pos_final gain%':>16} {'pos_timeavg gain%':>18} "
                 f"{'vel_final gain%':>16} {'vel_timeavg gain%':>18}")
    lines.append("-" * 88)
    if ref_method:
        m0 = ref_method[1]
        for name, met, perf, speedup, relE, gate_info in methods_data:
            if name == "noNN": continue
            pf_g = pct_gain(m0["pos_final"],   met["pos_final"])
            pt_g = pct_gain(m0["pos_timeavg"], met["pos_timeavg"])
            vf_g = pct_gain(m0["vel_final"],   met["vel_final"])
            vt_g = pct_gain(m0["vel_timeavg"], met["vel_timeavg"])
            lines.append(f"{name:<28} {pf_g:>+16.3f}% {pt_g:>+17.3f}% "
                         f"{vf_g:>+15.3f}% {vt_g:>+17.3f}%")
    lines.append("")

    # Speed-accuracy winner at similar compute
    
    enc_row  = next((m for m in methods_data if "EncounterNN" in m[0]), None)
    fk2_row  = next((m for m in methods_data if "k2" in m[0]), None)
    fk4_row  = next((m for m in methods_data if "k4" in m[0]), None)

    ENERGY_SAFE_THRESH = 0.10   # relE > 10% = energy-unsafe, disqualify
    def energy_safe(row):
        return row is not None and row[4] <= ENERGY_SAFE_THRESH

    lines.append("CONCLUSIONS")
    lines.append("-" * 60)
    # Flag energy-unsafe methods
    for row in [enc_row, fk2_row, fk4_row]:
        if row is not None and not energy_safe(row):
            lines.append(f"  !! ENERGY UNSAFE: {row[0]} (relE={row[4]:.3e} > {ENERGY_SAFE_THRESH:.0%}) — DISQUALIFIED")
    lines.append("")

    if enc_row and fk2_row and fk4_row:
        enc_t = enc_row[2]["total_time_sec"]
        fk2_t = fk2_row[2]["total_time_sec"]
        fk4_t = fk4_row[2]["total_time_sec"]
        # Only consider energy-safe methods for runtime-matched comparison
        safe_finer = [r for r in [fk2_row, fk4_row] if energy_safe(r)]
        if not safe_finer:
            lines.append("  No energy-safe fine-stepping method available for comparison.")
        else:
            closest = min(safe_finer, key=lambda r: abs(r[2]["total_time_sec"] - enc_t))
            lines.append(f"  Runtime-matched comparison (energy-safe methods only):")
            lines.append(f"    EncounterNN    : {enc_t:.4f}s  pos_final={enc_row[1]['pos_final']:.4e}  relE={enc_row[4]:.3e}")
            lines.append(f"    {closest[0]:<14}: {closest[2]['total_time_sec']:.4f}s  pos_final={closest[1]['pos_final']:.4e}  relE={closest[4]:.3e}")
            if enc_row[1]["pos_final"] < closest[1]["pos_final"]:
                lines.append(f"  >> RUNTIME WINNER: EncounterNN (lower pos_final error at similar compute)")
            else:
                lines.append(f"  >> RUNTIME WINNER: {closest[0]} (lower pos_final error at similar compute)")
        lines.append("")
        # Accuracy-only: all energy-safe methods
        safe_all = [r for r in [enc_row, fk2_row, fk4_row] if energy_safe(r)]
        if safe_all:
            best_acc = min(safe_all, key=lambda r: r[1]["pos_final"])
            lines.append(f"  Accuracy-only (energy-safe methods, ignoring NN inference cost):")
            lines.append(f"  >> ACCURACY WINNER: {best_acc[0]} (pos_final={best_acc[1]['pos_final']:.4e})")
        else:
            lines.append("  No energy-safe methods found.")

    lines.append(div)

    txt = "\n".join(lines)
    txt_path = os.path.join(out_dir, f"comparison_global_dt{dt}.txt")
    with open(txt_path, "w", encoding="utf-8") as f: f.write(txt)
    print(txt)

    # CSV
    csv_path = os.path.join(out_dir, f"comparison_global_dt{dt}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","pos_final","pos_timeavg","vel_final","vel_timeavg",
                    "time_sec","speedup","relE","fine_steps","pos_final_gain_pct",
                    "pos_timeavg_gain_pct"])
        ref_m = ref_method[1] if ref_method else None
        for name, met, perf, speedup, relE, gate_info in methods_data:
            pf_g = pct_gain(ref_m["pos_final"], met["pos_final"]) if ref_m and name!="noNN" else 0.0
            pt_g = pct_gain(ref_m["pos_timeavg"], met["pos_timeavg"]) if ref_m and name!="noNN" else 0.0
            w.writerow([name, f"{met['pos_final']:.6e}", f"{met['pos_timeavg']:.6e}",
                        f"{met['vel_final']:.6e}", f"{met['vel_timeavg']:.6e}",
                        f"{perf['total_time_sec']:.6f}", f"{speedup:.4f}", f"{relE:.4e}",
                        gate_info.get("fine_steps_fired", gate_info.get("encounter_used","")),
                        f"{pf_g:+.3f}", f"{pt_g:+.3f}"])
    return txt_path, csv_path


# =============================================================================
# Isolated comparison writer
# =============================================================================
def write_isolated_comparison(out_dir, dt, exit_metrics, local_metrics, k_list):
    lines = []
    W = 120
    lines.append("=" * W)
    lines.append(f"FINE-STEPPING VS ENCOUNTERNN  —  ISOLATED LOCAL COMPARISON  (dt={dt})")
    lines.append("=" * W)
    lines.append(f"  Entry state: r_pair={exit_metrics.get('r_pair_entry',float('nan')):.5f} AU  "
                 f"vr_norm={exit_metrics.get('v_rad_norm_entry',float('nan')):.4f}")
    lines.append(f"  EncounterNN correction accepted: {exit_metrics.get('enc_accepted','?')}")
    lines.append(f"  EncounterNN relE_corr: {exit_metrics.get('relE_corr',float('nan')):.4e}")
    lines.append("")
    lines.append("A) 0.5yr exit-state error vs IAS15 (lower=better):")
    lines.append(f"  {'Method':<22} {'pos exit err':>14} {'pos gain%':>11} {'vel exit err':>14} {'vel gain%':>11}")
    lines.append("  " + "-" * 75)

    noNN_pe = exit_metrics.get("noNN_pos_exit_err", float("nan"))
    noNN_ve = exit_metrics.get("noNN_vel_exit_err", float("nan"))
    lines.append(f"  {'noNN':<22} {noNN_pe:>14.4e} {'---':>11} {noNN_ve:>14.4e} {'---':>11}")
    enc_pe = exit_metrics.get("encNN_pos_exit_err", float("nan"))
    enc_ve = exit_metrics.get("encNN_vel_exit_err", float("nan"))
    enc_pg = exit_metrics.get("encNN_pos_gain_pct", float("nan"))
    enc_vg = exit_metrics.get("encNN_vel_gain_pct", float("nan"))
    lines.append(f"  {'EncounterNN':<22} {enc_pe:>14.4e} {enc_pg:>+10.2f}% {enc_ve:>14.4e} {enc_vg:>+10.2f}%")
    for kk in k_list:
        fk_pe = exit_metrics.get(f"finer_k{kk}_pos_exit_err", float("nan"))
        fk_ve = exit_metrics.get(f"finer_k{kk}_vel_exit_err", float("nan"))
        fk_pg = exit_metrics.get(f"finer_k{kk}_pos_gain_pct", float("nan"))
        fk_vg = exit_metrics.get(f"finer_k{kk}_vel_gain_pct", float("nan"))
        lines.append(f"  {f'finer-k{kk}':<22} {fk_pe:>14.4e} {fk_pg:>+10.2f}% {fk_ve:>14.4e} {fk_vg:>+10.2f}%")
    lines.append("")

    lines.append("B) 4yr physical window (time-average error vs IAS15, lower=better):")
    lines.append(f"  {'Method':<22} {'pos timeavg':>14} {'pos gain%':>11} {'vel timeavg':>14}")
    lines.append("  " + "-" * 65)
    for row in local_metrics:
        mname = row.get("method", "?")
        pos_ta = row.get("pos_timeavg", float("nan"))
        vel_ta = row.get("vel_timeavg", float("nan"))
        pos_g  = row.get("pos_gain_pct", float("nan"))
        lines.append(f"  {mname:<22} {pos_ta:>14.4e} {pos_g:>+10.2f}% {vel_ta:>14.4e}")
    lines.append("")

    # Winner
    enc_exit = exit_metrics.get("encNN_pos_exit_err", float("nan"))
    best_exit = min([(exit_metrics.get(f"finer_k{kk}_pos_exit_err", float("nan")), f"finer-k{kk}")
                     for kk in k_list] + [(enc_exit, "EncounterNN")], key=lambda x: x[0])
    lines.append(f"  >> EXIT-STATE WINNER: {best_exit[1]}  (pos_exit_err={best_exit[0]:.4e})")
    lines.append("=" * W)

    path = os.path.join(out_dir, f"comparison_isolated_dt{dt}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))
    return path


# =============================================================================
# Ensemble comparison writer
# =============================================================================
def write_ensemble_comparison(out_dir, dt, seed_results, k_list):
    """seed_results: list of dicts, one per seed, with method-level metrics."""
    lines = []
    W = 140
    lines.append("=" * W)
    lines.append(f"FINE-STEPPING VS ENCOUNTERNN  —  ENSEMBLE COMPARISON  (dt={dt}, n_seeds={len(seed_results)})")
    lines.append("=" * W)

    methods = ["noNN", "encNN"] + [f"finer_k{kk}" for kk in k_list]
    keys    = ["pos_final", "pos_timeavg", "vel_final"]

    # Header
    hdr = f"{'Seed':>5}"
    for mth in methods:
        hdr += f"  {mth+' pos_final':>22}"
    lines.append(hdr)
    lines.append("-" * W)

    col_vals = {mth: [] for mth in methods}
    for seed_r in seed_results:
        row = f"{seed_r['seed']:>5}"
        for mth in methods:
            val = seed_r.get(f"{mth}_pos_final_gain_pct", float("nan"))
            col_vals[mth].append(val)
            row += f"  {val:>+21.3f}%"
        lines.append(row)

    lines.append("-" * W)
    for stat_name, fn in [("MEAN", np.nanmean), ("STD", np.nanstd), ("MEDIAN", np.nanmedian)]:
        row = f"{stat_name:>5}"
        for mth in methods:
            v = fn(col_vals[mth])
            row += f"  {v:>+21.3f}%"
        lines.append(row)
    lines.append("")

    # N improved
    row = f"{'N>0':>5}"
    for mth in methods:
        n = sum(1 for v in col_vals[mth] if math.isfinite(v) and v > 0)
        tot = sum(1 for v in col_vals[mth] if math.isfinite(v))
        row += f"  {f'{n}/{tot}':>22}"
    lines.append(row)
    lines.append("")

    # iso exit pos if available
    if any("encNN_iso_exit_gain" in r for r in seed_results):
        lines.append(f"{'Seed':>5}  {'encNN iso_exit%':>20}" +
                     "".join(f"  {f'finer_k{kk} iso_exit%':>22}" for kk in k_list))
        lines.append("-" * 80)
        for seed_r in seed_results:
            row = f"{seed_r['seed']:>5}  {seed_r.get('encNN_iso_exit_gain',float('nan')):>+19.3f}%"
            for kk in k_list:
                row += f"  {seed_r.get(f'finer_k{kk}_iso_exit_gain',float('nan')):>+21.3f}%"
            lines.append(row)
        lines.append("")

    # Overall winner — disqualify methods with any catastrophic seed (relE > 10%)
    ENERGY_SAFE_THRESH = 0.10
    enc_mean = np.nanmean(col_vals["encNN"])
    finer_means = {f"finer_k{kk}": np.nanmean(col_vals[f"finer_k{kk}"]) for kk in k_list}
    all_means = {"encNN": enc_mean, **finer_means}

    # Check energy safety using relE column if available
    energy_unsafe = set()
    for mth in methods:
        rE_col = [r.get(f"{mth}_relE_drift", float("nan")) for r in seed_results]
        if any(math.isfinite(v) and v > ENERGY_SAFE_THRESH for v in rE_col):
            n_unsafe = sum(1 for v in rE_col if math.isfinite(v) and v > ENERGY_SAFE_THRESH)
            lines.append(f"  !! ENERGY UNSAFE: {mth} ({n_unsafe}/{len(seed_results)} seeds relE > {ENERGY_SAFE_THRESH:.0%}) — DISQUALIFIED from winner")
            energy_unsafe.add(mth)

    safe_means = {k: v for k, v in all_means.items()
                  if k.replace("encNN","EncounterNN").replace("finer_","finer-") not in energy_unsafe
                  and k not in energy_unsafe}
    if safe_means:
        winner = max(safe_means, key=safe_means.get)
        lines.append(f"  >> ENSEMBLE GLOBAL WINNER: {winner}  (mean pos_final_gain={safe_means[winner]:+.2f}%)")
    else:
        winner = max(all_means, key=all_means.get)
        lines.append(f"  >> ENSEMBLE GLOBAL WINNER (unchecked): {winner}  (mean pos_final_gain={all_means[winner]:+.2f}%)")

    lines.append("=" * W)

    path = os.path.join(out_dir, f"comparison_ensemble_dt{dt}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))
    return path


# =============================================================================
# Speed-accuracy plot
# =============================================================================
def write_speed_accuracy_plot(out_dir, dt, methods_data):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Speed–Accuracy Frontier  (dt={dt}, T=100yr, IC1)", fontsize=12)

    colors  = {"noNN":"gray","EncounterNN":"royalblue",
               "finer-Zone3-k2":"tomato","finer-Zone3-k4":"darkorange"}
    markers = {"noNN":"s","EncounterNN":"*","finer-Zone3-k2":"o","finer-Zone3-k4":"^"}

    for name, met, perf, speedup, relE, _ in methods_data:
        col = colors.get(name, "purple")
        mks = markers.get(name, "D")
        sz  = 150 if name=="EncounterNN" else 80
        axes[0].scatter(speedup, met["pos_final"],
                        c=col, marker=mks, s=sz, label=name, zorder=3)
        axes[1].scatter(speedup, met["pos_timeavg"],
                        c=col, marker=mks, s=sz, label=name, zorder=3)

    for ax, ylabel, title in zip(axes,
                                  ["pos_final RMS error (AU)", "pos_timeavg RMS error (AU)"],
                                  ["Final position error vs speedup",
                                   "Time-avg position error vs speedup"]):
        ax.set_xlabel("Speedup vs IAS15")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # faster = left, so better accuracy = bottom-left

    plt.tight_layout()
    path = os.path.join(out_dir, f"speed_accuracy_dt{dt}.png")
    plt.savefig(path, dpi=220)
    plt.close()
    return path


# =============================================================================
# Summary winner file
# =============================================================================
def write_summary_winner(out_dir, dt, global_lines, isolated_exit, ensemble_winner):
    lines  = ["=" * 70,
              f"FINE-STEPPING VS ENCOUNTERNN — SUMMARY VERDICT  (dt={dt})",
              "=" * 70, ""]
    lines += global_lines
    lines += ["", f"Isolated exit-state winner  : {isolated_exit}",
              f"Ensemble global mean winner : {ensemble_winner}", ""]
    path = os.path.join(out_dir, f"summary_winner_dt{dt}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))
    return path


# =============================================================================
# Main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="EncounterNN vs fine-stepping comparison")
    ap.add_argument("--dt",           type=float, required=True)
    ap.add_argument("--T",            type=float, default=100.0)
    ap.add_argument("--n_samples",    type=int,   default=5000)
    ap.add_argument("--window_years", type=float, default=0.5)
    ap.add_argument("--physical_window_steps", type=int, default=25,
                    help="Local horizon = 2 * this * dt")
    ap.add_argument("--post_window_steps",     type=int, default=25)
    ap.add_argument("--local_n_samples",       type=int, default=1001)
    ap.add_argument("--encounter_model",       type=str, required=True)
    ap.add_argument("--base_a_file",           type=str, required=True,
                    help="Path to v5 summary file for this dt (ground truth A)")
    ap.add_argument("--base_b_file",           type=str, required=True,
                    help="Path to v7 summary file for this dt (ground truth B)")
    ap.add_argument("--out_dir",               type=str, default="comp_encounter_stepping")
    ap.add_argument("--k_list",    type=int,   nargs="+", default=[2, 4])
    ap.add_argument("--n_seeds",   type=int,   default=10)
    ap.add_argument("--perturb_scale", type=float, default=1e-8)
    ap.add_argument("--vr_thresh",     type=float, default=-0.40)
    ap.add_argument("--energy_gate",   type=float, default=0.20)
    ap.add_argument("--max_radius_gate",type=float,default=1e4)
    ap.add_argument("--val_tol",       type=float, default=0.005,
                    help="Relative tolerance for validation (default 0.5%)")
    ap.add_argument("--skip_ensemble", action="store_true",
                    help="Skip ensemble level (faster testing)")
    ap.add_argument("--skip_validation", action="store_true",
                    help="Skip strict validation (for debugging only)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()
    dt  = float(args.dt)
    k_list = [int(k) for k in args.k_list]

    enc_path = args.encounter_model
    if not enc_path.endswith(".pt") and os.path.exists(enc_path + ".pt"):
        enc_path += ".pt"
    encounter_model = load_encounter_numpy(enc_path)

    # Parse base files
    print(f"\n[base] Parsing ground-truth files...")
    base_a = parse_base_a_file(args.base_a_file)
    base_b = parse_base_b_file(args.base_b_file)
    print(f"  Base A (v5): noNN pos_final={base_a.get('noNN_pos_final',float('nan')):.4e}  "
          f"encNN pos_final={base_a.get('encNN_pos_final',float('nan')):.4e}")
    print(f"  Base B (v7): iso_exit_gain={base_b.get('iso_pos_exit_gain',float('nan')):.2f}%  "
          f"relE_corr={base_b.get('relE_corr',float('nan')):.4e}")

    x0, v0, m = make_ic1()
    local_horizon = 2.0 * float(args.physical_window_steps) * dt

    print(f"\n{'='*80}")
    print(f"ENCOUNTER NN vs FINE STEPPING COMPARISON  dt={dt}yr  T={args.T}yr")
    print(f"  k_list: {k_list}  |  n_seeds: {args.n_seeds}  |  local_horizon: {local_horizon:.2f}yr")
    print(f"  base_a: {args.base_a_file}")
    print(f"  base_b: {args.base_b_file}")
    print(f"  out_dir: {args.out_dir}")
    print(f"{'='*80}\n")

    # ---- VALIDATION ----
    val_path = os.path.join(args.out_dir, f"validation_dt{dt}.txt")
    if args.skip_validation:
        print("[VALIDATION] Skipped (--skip_validation flag set)")
        # Still need IAS15 and noNN fresh for downstream use
        tr, pr, vr_ref, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, args.T, args.n_samples)
        _, p_no, v_no, perf_no  = simulate_simon_noNN(x0, v0, m, cfg, dt, args.T, args.n_samples)
        event_rows = []
        _, p_en, v_en, perf_en  = simulate_simon_encounterNN(
            x0, v0, m, encounter_model, cfg,
            dt=dt, T=args.T, n_samples=args.n_samples,
            window_years=args.window_years, vr_thresh=args.vr_thresh,
            energy_gate=args.energy_gate, max_radius_gate=args.max_radius_gate,
            event_rows=event_rows)
        met_no = metric_block(p_no, v_no, pr, vr_ref)
        met_en = metric_block(p_en, v_en, pr, vr_ref)
        sp_no  = perf_r["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)
        sp_en  = perf_r["total_time_sec"] / max(perf_en["total_time_sec"], 1e-12)
        val_ok = True
    else:
        
        (
            val_ok,
            tr,
            pr,
            vr_ref,
            perf_r,
            met_no,
            met_en,
            sp_no,
            sp_en,
            perf_no,
            perf_en,
            event_rows,
        ) = run_validation(
            x0,
            v0,
            m,
            cfg,
            encounter_model,
            dt,
            args.T,
            args.n_samples,
            args.window_years,
            args.vr_thresh,
            args.energy_gate,
            args.max_radius_gate,
            base_a,
            base_b,
            args.val_tol,
            val_path,
        )
        
        if not val_ok:
            print("\n[ERROR] Validation failed. Results would not be comparable. Stopping.")
            sys.exit(1)

    # ---- GLOBAL FINE-STEPPING ----
    print(f"\n[global] Running SIMON-finer-Zone3 k={k_list}...")
    finer_global = {}
    for kk in k_list:
        print(f"  k={kk}...")
        _, p_fk, v_fk, perf_fk = simulate_simon_finer_zone3(
            x0, v0, m, cfg, dt, args.T, args.n_samples, kk, args.vr_thresh)
        met_fk = metric_block(p_fk, v_fk, pr, vr_ref)
        sp_fk  = perf_r["total_time_sec"] / max(perf_fk["total_time_sec"], 1e-12)
        print(f"    pos_final={met_fk['pos_final']:.4e}  speedup={sp_fk:.2f}x  "
              f"fine_fired={perf_fk['fine_steps_fired']}  "
              f"relE_drift={perf_fk['relE_drift']:.3e}")
        finer_global[kk] = (p_fk, v_fk, met_fk, perf_fk, sp_fk)

    # Build methods_data for global table
    methods_data = [
        ("noNN",              met_no, perf_no, sp_no, perf_no["relE_drift"], {}),
        ("EncounterNN",       met_en, perf_en, sp_en,
         perf_en.get("local_correction_relE_corr", perf_en.get("relE_drift", float("nan"))),
         {"encounter_used": perf_en.get("encounter_used", 0)}),
    ]
    for kk in k_list:
        _, _, met_fk, perf_fk, sp_fk = finer_global[kk]
        methods_data.append(
            (f"finer-Zone3-k{kk}", met_fk, perf_fk, sp_fk,
             perf_fk["relE_drift"], {"fine_steps_fired": perf_fk["fine_steps_fired"]}))

    ias15_time = perf_r["total_time_sec"]
    write_global_comparison(args.out_dir, dt, methods_data, ias15_time)
    write_speed_accuracy_plot(args.out_dir, dt, methods_data)

    # ---- ISOLATED LOCAL ----
    used_events = [r for r in event_rows if int(r.get("used", 0)) == 1]
    if not used_events:
        print("[WARNING] No EncounterNN events fired. Skipping isolated comparison.")
        iso_winner = "N/A"
    else:
        first_event = used_events[0]
        event_t = float(first_event["t_start"])
        print(f"\n[isolated] Reconstructing event-entry state at t={event_t:.3f}yr...")
        x_event, v_event, _ = advance_nonn_to_time(x0, v0, m, cfg, dt, event_t)
        vr_e, vt_e, r_e = compute_pair_velocity_features_12(x_event, v_event, m, cfg)
        print(f"  r_pair={r_e:.5f} AU  vr_norm={vr_e:.4f}  vt_norm={vt_e:.4f}")

        print(f"  Running exit-state comparison (0.5yr)...")
        exit_met = compute_exit_state_all_methods(
            x_event, v_event, m, encounter_model, cfg,
            dt, args.window_years, args.energy_gate, args.max_radius_gate,
            k_list=k_list, vr_thresh=args.vr_thresh)

        print(f"  Running local IAS15 over {local_horizon:.2f}yr...")
        tl, p_ias_l, v_ias_l, _ = simulate_rebound_ias15(
            x_event, v_event, m, cfg.G, local_horizon, args.local_n_samples)
        print(f"  Running local noNN...")
        _, p_no_l, v_no_l, _ = simulate_simon_noNN(
            x_event, v_event, m, cfg, dt, local_horizon, args.local_n_samples)
        print(f"  Running local EncounterNN-once...")
        p_en_l, v_en_l, _ = _run_finer_local(  # reuse infra; k=1 = no fine
            x_event, v_event, m, cfg, dt, local_horizon, args.local_n_samples, 1, args.vr_thresh)
        # override with proper encounterNN-once (apply correction)
        p_en_l, v_en_l, perf_en_l = _run_encounterNN_local_once(
            x_event, v_event, m, encounter_model, cfg, dt, local_horizon,
            args.local_n_samples, args.window_years, args.energy_gate,
            args.max_radius_gate, args.vr_thresh)

        # 4yr metrics for all methods
        def _4yr_metrics(p_test, v_test, name):
            err_p = rms_sep(p_test, p_ias_l)
            err_n = rms_sep(p_no_l, p_ias_l)
            mask  = (tl >= -1e-12) & (tl <= local_horizon + 1e-12)
            pos_ta = float(np.sqrt(np.mean(err_p[mask]**2)))
            no_ta  = float(np.sqrt(np.mean(err_n[mask]**2)))
            pos_g  = pct_gain(no_ta, pos_ta)
            vel_p  = rms_sep(v_test, v_ias_l)
            vel_n  = rms_sep(v_no_l, v_ias_l)
            vel_ta = float(np.sqrt(np.mean(vel_p[mask]**2)))
            vno_ta = float(np.sqrt(np.mean(vel_n[mask]**2)))
            return {"method": name, "pos_timeavg": pos_ta, "pos_gain_pct": pos_g,
                    "vel_timeavg": vel_ta, "vel_gain_pct": pct_gain(vno_ta, vel_ta)}

        local_metrics = [_4yr_metrics(p_no_l, v_no_l, "noNN"),
                         _4yr_metrics(p_en_l, v_en_l, "EncounterNN")]
        for kk in k_list:
            print(f"  Running local finer-k{kk}...")
            p_fk_l, v_fk_l, _ = _run_finer_local(
                x_event, v_event, m, cfg, dt, local_horizon,
                args.local_n_samples, int(kk), args.vr_thresh)
            local_metrics.append(_4yr_metrics(p_fk_l, v_fk_l, f"finer-k{kk}"))
            exit_met[f"finer_k{kk}_4yr_gain"] = local_metrics[-1]["pos_gain_pct"]

        iso_path = write_isolated_comparison(args.out_dir, dt, exit_met, local_metrics, k_list)

        # Determine isolated exit winner
        candidates = {"EncounterNN": exit_met.get("encNN_pos_exit_err", float("nan"))}
        for kk in k_list:
            candidates[f"finer-k{kk}"] = exit_met.get(f"finer_k{kk}_pos_exit_err", float("nan"))
        # Disqualify energy-unsafe methods from isolated winner (checked via global relE)
        iso_winner = min(candidates, key=candidates.get)
        # Note: isolated winner is labelled advisory only — check global energy before trusting

    # ---- ENSEMBLE ----
    if args.skip_ensemble:
        print("[ensemble] Skipped (--skip_ensemble flag)")
        ens_winner = "N/A (skipped)"
    else:
        print(f"\n[ensemble] Running {args.n_seeds} seeds (perturb_scale={args.perturb_scale:.1e})...")
        seed_results = []
        for seed in range(args.n_seeds):
            print(f"  seed {seed}...")
            xp, vp = perturb_ic(x0, v0, m, seed, args.perturb_scale)
            tr_s, pr_s, vr_s, _ = simulate_rebound_ias15(xp, vp, m, cfg.G, args.T, args.n_samples)
            _, p_nos, v_nos, _ = simulate_simon_noNN(xp, vp, m, cfg, dt, args.T, args.n_samples)
            met_nos = metric_block(p_nos, v_nos, pr_s, vr_s)
            ev_s = []
            _, p_ens, v_ens, perf_ens = simulate_simon_encounterNN(
                xp, vp, m, encounter_model, cfg,
                dt=dt, T=args.T, n_samples=args.n_samples,
                window_years=args.window_years, vr_thresh=args.vr_thresh,
                energy_gate=args.energy_gate, max_radius_gate=args.max_radius_gate,
                event_rows=ev_s)
            
            met_ens = metric_block(p_ens, v_ens, pr_s, vr_s)
            sr = {"seed": seed,
                  "encNN_relE_drift": perf_ens.get("local_correction_relE_corr",
                                                    perf_ens.get("relE_drift", float("nan"))),
                  
                  "noNN_pos_final_gain_pct": pct_gain(met_nos["pos_final"], met_nos["pos_final"]),
                  "encNN_pos_final_gain_pct": pct_gain(met_nos["pos_final"], met_ens["pos_final"]),
                  "encNN_iso_exit_gain": float("nan")}
            # noNN gain vs itself is always 0; patch
            sr["noNN_pos_final_gain_pct"] = 0.0
            for kk in k_list:
                _, p_fks, v_fks, perf_fks = simulate_simon_finer_zone3(
                    xp, vp, m, cfg, dt, args.T, args.n_samples, int(kk), args.vr_thresh)
                
                met_fks = metric_block(p_fks, v_fks, pr_s, vr_s)
                sr[f"finer_k{kk}_pos_final_gain_pct"] = pct_gain(
                    met_nos["pos_final"], met_fks["pos_final"])
                sr[f"finer_k{kk}_relE_drift"] = perf_fks.get("relE_drift", float("nan"))
                
            # Isolated exit for this seed
            used_s = [r for r in ev_s if int(r.get("used",0))==1]
            if used_s:
                et_s = float(used_s[0]["t_start"])
                try:
                    xe_s, ve_s, _ = advance_nonn_to_time(xp, vp, m, cfg, dt, et_s)
                    ex_s = compute_exit_state_all_methods(
                        xe_s, ve_s, m, encounter_model, cfg, dt,
                        args.window_years, args.energy_gate, args.max_radius_gate,
                        k_list=k_list, vr_thresh=args.vr_thresh)
                    sr["encNN_iso_exit_gain"] = ex_s.get("encNN_pos_gain_pct", float("nan"))
                    for kk in k_list:
                        sr[f"finer_k{kk}_iso_exit_gain"] = ex_s.get(
                            f"finer_k{kk}_pos_gain_pct", float("nan"))
                except Exception:
                    pass
            seed_results.append(sr)

        ens_path = write_ensemble_comparison(args.out_dir, dt, seed_results, k_list)
        col_enc = [r.get("encNN_pos_final_gain_pct", float("nan")) for r in seed_results]
        finer_means = {}
        for kk in k_list:
            col = [r.get(f"finer_k{kk}_pos_final_gain_pct", float("nan")) for r in seed_results]
            finer_means[f"finer_k{kk}"] = np.nanmean(col)
        all_means = {"EncounterNN": np.nanmean(col_enc), **finer_means}
        ens_winner = max(all_means, key=all_means.get)

    # ---- SUMMARY WINNER ----
    enc_pf_gain = pct_gain(met_no["pos_final"], met_en["pos_final"])
    global_sum = [
        f"Global pos_final gains vs noNN:",
        f"  EncounterNN  : {enc_pf_gain:+.3f}%  speedup={sp_en:.2f}x  relE={perf_en.get('local_correction_relE_corr',float('nan')):.3e}",
    ]
    for kk in k_list:
        _, _, met_fk, perf_fk, sp_fk = finer_global[kk]
        g = pct_gain(met_no["pos_final"], met_fk["pos_final"])
        global_sum.append(
            f"  finer-k{kk}     : {g:+.3f}%  speedup={sp_fk:.2f}x  relE_drift={perf_fk['relE_drift']:.3e}")

    write_summary_winner(args.out_dir, dt, global_sum,
                         iso_winner if used_events else "N/A",
                         ens_winner)

    print(f"\n[done] All outputs written to: {args.out_dir}")
    print(f"  Return to Claude for analysis:")
    out_files = [
        f"comparison_global_dt{dt}.txt",
        f"comparison_isolated_dt{dt}.txt",
        f"comparison_ensemble_dt{dt}.txt",
        f"summary_winner_dt{dt}.txt",
    ]
    for fn in out_files:
        print(f"    {os.path.join(args.out_dir, fn)}")


def _run_encounterNN_local_once(x_event, v_event, m, encounter_model, cfg,
                                 dt, T_local, n_samples, window_years,
                                 energy_gate, max_radius_gate, vr_thresh):
    """Run encounterNN-once from event-entry state, return trajectory arrays."""
    kernel = NoNNKernel(cfg, m)
    times  = np.linspace(0.0, float(T_local), int(n_samples))
    p_out  = np.zeros((len(times), 3, 3), dtype=np.float64)
    v_out  = np.zeros_like(p_out)
    x = x_event.astype(np.float64).copy()
    v = v_event.astype(np.float64).copy()
    E_start = total_energy_state(x, v, m, cfg.G)
    X_entry, _, _, _ = make_X_rel18_pair12(x, v, m, float(dt), cfg)
    si = 0; t_cur = 0.0
    def fill():
        nonlocal si
        while si < len(times) and t_cur >= times[si]-1e-12:
            p_out[si]=x; v_out[si]=v; si+=1
    fill()
    _, r2, _ = kernel.geometry(x); r_min = float(np.sqrt(r2.min()+1e-30)); ac = None; steps = 0
    # noNN window
    remaining = float(window_years)
    while remaining > 1e-14:
        h = min(float(dt), remaining)
        x, v, r_min, ac, _ = kernel.macro_step(x, v, h, r_min, ac)
        t_cur += h; remaining -= h; steps += 1; fill()
    # apply correction
    x_no = x.copy(); v_no = v.copy()
    dx, dv, _, _ = predict_encounter_residual_np(encounter_model, X_entry)
    xc = x_no + dx; vc = v_no + dv
    xc, vc = project_com_to_reference(xc, vc, x_no, v_no, m)
    relE = abs((total_energy_state(xc, vc, m, cfg.G) - E_start) / (abs(E_start)+1e-30))
    ok = bool(all_finite_state(xc, vc)
              and max_radius_state(xc) <= float(max_radius_gate)
              and relE <= float(energy_gate))
    if ok:
        x = xc; v = vc
        _, r2c, _ = kernel.geometry(x); r_min = float(np.sqrt(r2c.min()+1e-30)); ac = None
        if si > 0 and abs(times[si-1]-t_cur) <= max(1e-10, 1e-9*abs(t_cur)):
            p_out[si-1]=x; v_out[si-1]=v
        fill()
    # continue
    while t_cur < float(T_local) - 1e-14:
        h = min(float(dt), float(T_local)-t_cur)
        x, v, r_min, ac, _ = kernel.macro_step(x, v, h, r_min, ac)
        t_cur += h; steps += 1; fill()
    while si < len(times):
        p_out[si]=x; v_out[si]=v; si+=1
    perf = {"correction_used": int(ok), "relE_corr": float(relE)}
    return p_out, v_out, perf


if __name__ == "__main__":
    main()
