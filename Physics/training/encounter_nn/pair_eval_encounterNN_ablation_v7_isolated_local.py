# pair_eval_encounterNN_ablation_v6_window_compare.py
#
# Focused SIMON encounter-level residual-NN evaluator with fast gate path and local encounter-window comparisons.
#
# Purpose
# -------
# Compare only:
#   1. IAS15 reference
#   2. SIMON-noNN baseline
#   3. SIMON-encounterNN = SIMON-noNN local window + 18D residual correction
#
# In addition to the full 100-year metrics, this v6 evaluator measures local
# encounter-window metrics around each used EncounterNN event. This separates
# the question "did the NN help during the encounter?" from the later chaotic
# branch divergence over the remaining trajectory.
#
# This file deliberately contains NO step-level Zone-3 NN code and does NOT require
# <step-level Zone-3 NN checkpoint>. The step-level Zone-3 NN result can be compared later
# from the already-known v4 evaluator outputs.
#
# EncounterNN deployment formula
# ------------------------------
#   At a gated encounter-entry state I:
#       F_S,new = SIMON-noNN exit state after a 0.5 yr local window
#       R_hat   = EncounterNN(I), an 18D residual [dx_flat9, dv_flat9]
#       F_corr  = F_S,new + R_hat
#   Then the global rollout resumes from F_corr.
#
# Efficiency choices
# ------------------
# - The encounter model is loaded once from the PyTorch checkpoint.
# - Its Linear-layer weights and normalization buffers are extracted to NumPy.
# - Timed rollout uses no torch/CUDA inference calls.
# - The noNN integrator uses the same optimized ideas as the original v4
#   evaluator: vectorized three-pair geometry, cached far-field acceleration,
#   no separate min-distance pass at the top of every step, and lightweight
#   counters.
#
# Typical run from C:\Aarush\Physics\training\encounter_nn:
#   python -B pair_eval_encounterNN_ablation_v6_window_compare.py ^
#       --encounter_model encounter_surrogate_v3_rollout_local_velocitysafe.pt ^
#       --dt 0.08 --T 100 --n_samples 5000 ^
#       --out_dir encounterNN_ablation_v6_window_compare_out

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
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
    print("[ERROR] rebound not found. Install it in your local cenv.")
    raise


# =============================================================================
# Configuration
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
# Encounter residual NN: checkpoint -> NumPy deployment
# =============================================================================
@dataclass
class EncounterNumpyWeights:
    layers: List[Tuple[np.ndarray, np.ndarray]]  # each (W[out,in], b[out])
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    arch: str
    params: int


def _as_state_dict(obj):
    """Accept either raw state_dict or a checkpoint wrapper."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _linear_layer_indices(sd: Dict[str, torch.Tensor]) -> List[int]:
    idxs: List[int] = []
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
        raise TypeError("Encounter checkpoint is not a state_dict or supported checkpoint wrapper.")

    required = ["input_mean", "input_std", "target_mean", "target_std"]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(
            f"Encounter checkpoint missing {missing}. Expected the final velocity-safe "
            "18D residual model with input/target normalization buffers."
        )

    idxs = _linear_layer_indices(sd)
    if not idxs:
        raise KeyError("No Linear layers found under keys like net.<idx>.weight")

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    shapes: List[Tuple[int, int]] = []
    params = 0
    for idx in idxs:
        wk, bk = f"net.{idx}.weight", f"net.{idx}.bias"
        if bk not in sd:
            raise KeyError(f"Missing bias for {wk}: expected {bk}")
        W = sd[wk].detach().cpu().numpy().astype(np.float32).copy()
        b = sd[bk].detach().cpu().numpy().astype(np.float32).copy()
        if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
            raise ValueError(f"Bad layer shape at net.{idx}: W={W.shape}, b={b.shape}")
        layers.append((W, b))
        shapes.append((int(W.shape[1]), int(W.shape[0])))
        params += int(W.size + b.size)

    if shapes[0][0] != 18:
        raise ValueError(f"Encounter model must take 18 inputs; first layer in/out={shapes[0]}")
    if shapes[-1][1] != 18:
        raise ValueError(f"Encounter model must output 18 residuals; final layer in/out={shapes[-1]}")

    def _buffer(name: str) -> np.ndarray:
        arr = sd[name].detach().cpu().numpy().astype(np.float32).reshape(-1).copy()
        if arr.shape != (18,):
            raise ValueError(f"{name} must have shape (18,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        return arr

    input_mean = _buffer("input_mean")
    input_std = _buffer("input_std")
    target_mean = _buffer("target_mean")
    target_std = _buffer("target_std")
    input_std = np.where(np.abs(input_std) < 1e-8, 1.0, input_std).astype(np.float32)
    target_std = np.where(np.abs(target_std) < 1e-8, 1.0, target_std).astype(np.float32)

    arch = " -> ".join([str(shapes[0][0])] + [str(out_dim) for _, out_dim in shapes])
    print(f"[encounterNN] Loaded {model_path} as NumPy MLP | arch={arch} | params={params}")
    return EncounterNumpyWeights(
        layers=layers,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        arch=arch,
        params=params,
    )


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def encounter_forward_numpy(X: np.ndarray, w: EncounterNumpyWeights) -> np.ndarray:
    z = np.asarray(X, dtype=np.float32).reshape(1, 18)
    z = (z - w.input_mean.reshape(1, 18)) / (w.input_std.reshape(1, 18) + np.float32(1e-8))
    for layer_i, (W, b) in enumerate(w.layers):
        z = z @ W.T + b.reshape(1, -1)
        if layer_i != len(w.layers) - 1:
            z = _silu_np(z).astype(np.float32)
    y_norm = z.reshape(18).astype(np.float32)
    y = y_norm * (w.target_std + np.float32(1e-8)) + w.target_mean
    return y.astype(np.float64)


# =============================================================================
# Reference and metrics
# =============================================================================
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
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


def rms_sep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    pb = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(pb * pb, axis=1))


def metric_block(pos: np.ndarray, vel: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray) -> Dict[str, float]:
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


def all_finite_state(*arrs) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrs)


def max_radius_state(x: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(x, axis=1)))


def total_energy_state(x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float = 1.0) -> float:
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = float(np.linalg.norm(x[j] - x[i]))
            pe -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(ke + pe)


# =============================================================================
# Feature construction for final encounter model
# =============================================================================
def compute_pair_velocity_features_12(x: np.ndarray, v: np.ndarray, m: np.ndarray, cfg: HybridConfig) -> Tuple[float, float, float]:
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


def encounter_gate_features_pair12_fast(x: np.ndarray, v: np.ndarray, m: np.ndarray, cfg: HybridConfig) -> Tuple[bool, float, float, float]:
    """
    Cheap encounter gate for pair 1-2.

    This intentionally does NOT build the full 18D NN input. Most macro steps
    are far from the validated encounter gate, so the hot loop first checks only
    the pair distance. It computes velocity features only for pair 1-2 when the
    pair is actually in Zone 3. The full X_rel18 vector is built only on the rare
    step where the gate passes.
    """
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


def make_X_rel18_pair12(x: np.ndarray, v: np.ndarray, m: np.ndarray, dt: float, cfg: HybridConfig) -> Tuple[np.ndarray, float, float, float]:
    # This is the compact relative-state representation used by the final rollout-local residual model.
    i, j, k = 1, 2, 0
    mi, mj = float(m[i]), float(m[j])
    pair_com_x = (mi * x[i] + mj * x[j]) / (mi + mj)
    pair_com_v = (mi * v[i] + mj * v[j]) / (mi + mj)
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r3 = x[k] - pair_com_x
    v3 = v[k] - pair_com_v
    vr, vt, r = compute_pair_velocity_features_12(x, v, m, cfg)
    X = np.concatenate([
        np.log(m + 1e-30),
        rij, vij, r3, v3,
        np.array([math.log(float(dt) + 1e-30), vr, vt], dtype=np.float64),
    ]).astype(np.float32)
    if X.shape != (18,):
        raise RuntimeError(f"X_rel18 has bad shape {X.shape}")
    return X, vr, vt, r


def project_com_to_reference(x_corr: np.ndarray, v_corr: np.ndarray,
                             x_ref: np.ndarray, v_ref: np.ndarray,
                             m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    com_x_corr = np.sum(m[:, None] * x_corr, axis=0) / M
    com_v_corr = np.sum(m[:, None] * v_corr, axis=0) / M
    com_x_ref = np.sum(m[:, None] * x_ref, axis=0) / M
    com_v_ref = np.sum(m[:, None] * v_ref, axis=0) / M
    return x_corr - (com_x_corr - com_x_ref), v_corr - (com_v_corr - com_v_ref)


def predict_encounter_residual_np(model: EncounterNumpyWeights, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    y = encounter_forward_numpy(X, model)
    if y.shape != (18,) or not np.all(np.isfinite(y)):
        raise FloatingPointError(f"EncounterNN produced invalid residual with shape {y.shape}")
    dx = y[:9].reshape(3, 3)
    dv = y[9:].reshape(3, 3)
    pred_pos_norm = float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))
    pred_vel_norm = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    return dx, dv, pred_pos_norm, pred_vel_norm


# =============================================================================
# Optimized SIMON-noNN kernel and encounterNN deployment
# =============================================================================
@dataclass
class NoNNKernel:
    cfg: HybridConfig
    m: np.ndarray

    def __post_init__(self):
        self.N = int(self.m.shape[0])
        if self.N != 3:
            raise ValueError("This focused evaluator is written for the 3-body IC1 case.")
        ii, jj = [], []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ii.append(i); jj.append(j)
        self.ii = np.array(ii, dtype=np.int64)
        self.jj = np.array(jj, dtype=np.int64)
        self.P = int(len(self.ii))
        self.m_f = self.m.astype(np.float64)
        self.mi_arr = self.m_f[self.ii]
        self.mj_arr = self.m_f[self.jj]
        self.Gmimj = float(self.cfg.G) * self.mi_arr * self.mj_arr
        self.inv_mi = 1.0 / self.mi_arr
        self.inv_mj = 1.0 / self.mj_arr
        self.eps2 = float(self.cfg.eps * self.cfg.eps)
        self.reset_counters()

    def reset_counters(self):
        self.pair_evals_n = np.int64(0)
        self.zone1_n = np.int64(0)
        self.zone2_n = np.int64(0)
        self.zone3_n = np.int64(0)
        self.far_n = np.int64(0)
        self.zone3_no_nn_n = np.int64(0)
        self.total_substeps = 0

    def geometry(self, pos: np.ndarray):
        rij = pos[self.jj] - pos[self.ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        return rij, r2, r

    def acc_from_geom_noNN(self, rij: np.ndarray, r2: np.ndarray, r: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        F_scalar = self.Gmimj / (r2 * r + 1e-30)

        zone1_mask = r < cfg.zone1_r_gate
        zone2_mask = (r >= cfg.zone1_r_gate) & (r < cfg.adapt_thresh)
        zone3_mask = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        far_mask = r >= cfg.nn_thresh

        self.pair_evals_n += self.P
        self.zone1_n += zone1_mask.sum()
        self.zone2_n += zone2_mask.sum()
        self.zone3_n += zone3_mask.sum()
        self.far_n += far_mask.sum()

        if np.any(zone3_mask):
            # noNN Zone 3: c=1 * softened force, exactly matching the original noNN baseline.
            F_soft = self.Gmimj[zone3_mask] / ((r2[zone3_mask] + self.eps2) ** 1.5 + 1e-30)
            F_scalar[zone3_mask] = F_soft
            self.zone3_no_nn_n += int(zone3_mask.sum())

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((self.N, 3), dtype=np.float64)
        for p in range(self.P):
            acc[self.ii[p]] += F_vec[p] * self.inv_mi[p]
            acc[self.jj[p]] -= F_vec[p] * self.inv_mj[p]
        return acc

    def acc_far(self, pos: np.ndarray) -> Tuple[np.ndarray, float]:
        rij = pos[self.jj] - pos[self.ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        F_scalar = self.Gmimj / (r2 * r + 1e-30)

        self.pair_evals_n += self.P
        self.far_n += self.P

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((self.N, 3), dtype=np.float64)
        for p in range(self.P):
            acc[self.ii[p]] += F_vec[p] * self.inv_mi[p]
            acc[self.jj[p]] -= F_vec[p] * self.inv_mj[p]
        return acc, float(np.sqrt(r2.min() + 1e-30))

    def verlet_noNN_zone_step(self, x_in: np.ndarray, v_in: np.ndarray, step_dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
        rij_in, r2_in, r_in = self.geometry(x_in)
        a0 = self.acc_from_geom_noNN(rij_in, r2_in, r_in)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        rij_n, r2_n, r_n = self.geometry(x_new)
        a1 = self.acc_from_geom_noNN(rij_n, r2_n, r_n)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, float(np.sqrt(r2_n.min() + 1e-30))

    def verlet_far_step(self, x_in: np.ndarray, v_in: np.ndarray, step_dt: float,
                        a_cache: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if a_cache is not None:
            a0 = a_cache
            # Preserve original report semantics: account for skipped first far-force call.
            self.pair_evals_n += self.P
            self.far_n += self.P
        else:
            a0, _ = self.acc_far(x_in)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        a1, r_min_new = self.acc_far(x_new)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, a1, r_min_new

    def macro_step(self, x: np.ndarray, v: np.ndarray, step_dt: float,
                   r_min_cur: float, a_cache: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], int]:
        cfg = self.cfg
        if r_min_cur < cfg.adapt_thresh:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_thresh / max(r_min_cur, 1e-30)))))
            sub_dt = float(step_dt) / n_sub
            x_loc = x
            v_loc = v
            r_min_loc = r_min_cur
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

    def perf_dict(self, label: str, steps: int, dt: float, T: float, n_samples: int, elapsed: float) -> Dict[str, float]:
        pair_evals = max(int(self.pair_evals_n), 1)
        return {
            "label": label,
            "steps": int(steps),
            "dt": float(dt),
            "T_years": float(T),
            "n_samples": int(n_samples),
            "total_time_sec": float(elapsed),
            "time_per_step_sec": float(elapsed / max(int(steps), 1)),
            "total_substeps": int(self.total_substeps),
            "zone1_frac": int(self.zone1_n) / pair_evals,
            "zone2_frac": int(self.zone2_n) / pair_evals,
            "zone3_frac": int(self.zone3_n) / pair_evals,
            "zone4_far_frac": int(self.far_n) / pair_evals,
            "zone3_no_nn_frac": int(self.zone3_no_nn_n) / pair_evals,
        }


def simulate_simon_noNN(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, cfg: HybridConfig,
                        dt: float, T: float, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    kernel = NoNNKernel(cfg, m)
    times = np.linspace(0.0, float(T), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()

    si = 0
    t_cur = 0.0
    while si < len(times) and t_cur >= times[si] - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    _, r2_0, _ = kernel.geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None
    steps = 0
    n_steps = int(math.ceil(float(T) / float(dt)))
    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt = min(float(dt), float(T) - t_cur)
        x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt, r_min_cur, a_cache)
        t_cur += step_dt
        steps += 1
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
        if not all_finite_state(x, v):
            break
    while si < len(times):
        pos_out[si] = x
        vel_out[si] = v
        si += 1
    elapsed = time.perf_counter() - t_start
    return times, pos_out, vel_out, kernel.perf_dict("SIMON_noNN", steps, dt, T, n_samples, elapsed)


def simulate_simon_encounterNN(
    x0: np.ndarray,
    v0: np.ndarray,
    m: np.ndarray,
    encounter_model: EncounterNumpyWeights,
    cfg: HybridConfig,
    dt: float,
    T: float,
    n_samples: int,
    window_years: float = 0.5,
    vr_thresh: float = -0.40,
    energy_gate: float = 0.20,
    max_radius_gate: float = 1e4,
    com_project: bool = True,
    event_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    kernel = NoNNKernel(cfg, m)
    times = np.linspace(0.0, float(T), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()

    si = 0
    t_cur = 0.0

    def fill_outputs_current():
        nonlocal si
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1

    fill_outputs_current()

    _, r2_0, _ = kernel.geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None
    steps = 0
    n_steps = int(math.ceil(float(T) / float(dt)))

    gate_candidates = 0
    encounter_attempts = 0
    encounter_used = 0
    encounter_fallback = 0
    first_used_t = float("nan")
    last_used_t = float("nan")
    pred_pos_norm_sum = 0.0
    pred_vel_norm_sum = 0.0
    min_r_global = r_min_cur
    max_radius_global = max_radius_state(x)

    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt_macro = min(float(dt), float(T) - t_cur)

        # Fast encounter gate at macro-step entry. Build the expensive 18D input
        # only if the cheap pair-1/2 distance+velocity gate passes.
        in_zone3_pair12, vr_entry, vt_entry, r_pair = encounter_gate_features_pair12_fast(x, v, m, cfg)
        if in_zone3_pair12:
            gate_candidates += 1
        gate_ok = bool(
            in_zone3_pair12
            and (vr_entry < float(vr_thresh))
            and (t_cur + float(window_years) <= float(T) + 1e-12)
        )

        if gate_ok:
            # The residual model itself still receives the exact X_rel18 feature
            # representation used in training. This is computed only on the rare
            # gated event, not on every macro step.
            X_entry, vr_entry, vt_entry, r_pair = make_X_rel18_pair12(x, v, m, float(dt), cfg)

            encounter_attempts += 1
            t_event = float(t_cur)
            x_start = x.copy()
            v_start = v.copy()
            E_start = total_energy_state(x_start, v_start, m, G=cfg.G)
            min_r_window = r_min_cur

            # First run the local SIMON-noNN window to produce F_S,new.
            remaining = float(window_years)
            while remaining > 1e-14:
                step_dt_w = min(float(dt), remaining)
                x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt_w, r_min_cur, a_cache)
                t_cur += step_dt_w
                remaining -= step_dt_w
                steps += 1
                min_r_window = min(min_r_window, r_min_cur)
                min_r_global = min(min_r_global, r_min_cur)
                fill_outputs_current()
                if not all_finite_state(x, v):
                    break

            x_no_exit = x.copy()
            v_no_exit = v.copy()

            # Then apply the 18D residual correction predicted from the entry state.
            dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual_np(encounter_model, X_entry)
            x_corr = x_no_exit + dx
            v_corr = v_no_exit + dv
            if com_project:
                x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)

            relE_corr = abs((total_energy_state(x_corr, v_corr, m, G=cfg.G) - E_start) / (abs(E_start) + 1e-30))
            use = True
            reason = "used"
            if not all_finite_state(x_corr, v_corr):
                use = False; reason = "nonfinite"
            elif max_radius_state(x_corr) > float(max_radius_gate):
                use = False; reason = "max_radius_gate"
            elif relE_corr > float(energy_gate):
                use = False; reason = "energy_gate"
            elif encounter_used >= 1:
                use = False; reason = "max_one_correction"

            if use:
                x = x_corr
                v = v_corr
                _, r2c, _ = kernel.geometry(x)
                r_min_cur = float(np.sqrt(r2c.min() + 1e-30))
                a_cache = None  # corrected state invalidates cached far acceleration
                encounter_used += 1
                if not np.isfinite(first_used_t):
                    first_used_t = t_event
                last_used_t = t_event
                pred_pos_norm_sum += pred_pos_norm
                pred_vel_norm_sum += pred_vel_norm
                min_r_global = min(min_r_global, r_min_cur)
                max_radius_global = max(max_radius_global, max_radius_state(x))
                # If an output sample was just written at the exit using noNN, overwrite with corrected state.
                if si > 0 and abs(times[si - 1] - t_cur) <= max(1e-10, 1e-9 * abs(t_cur)):
                    pos_out[si - 1] = x
                    vel_out[si - 1] = v
                fill_outputs_current()
            else:
                x = x_no_exit
                v = v_no_exit
                a_cache = None
                encounter_fallback += 1

            if event_rows is not None:
                event_rows.append({
                    "t_start": t_event,
                    "t_exit": float(t_cur),
                    "used": int(use),
                    "reason": reason,
                    "r_pair": float(r_pair),
                    "v_rad_norm": float(vr_entry),
                    "v_tan_norm": float(vt_entry),
                    "min_r_window": float(min_r_window),
                    "pred_pos_norm": float(pred_pos_norm),
                    "pred_vel_norm": float(pred_vel_norm),
                    "relE_corr": float(relE_corr),
                })
            continue

        # Standard optimized noNN macro step.
        x, v, r_min_cur, a_cache, _ = kernel.macro_step(x, v, step_dt_macro, r_min_cur, a_cache)
        t_cur += step_dt_macro
        steps += 1
        min_r_global = min(min_r_global, r_min_cur)
        fill_outputs_current()
        if not all_finite_state(x, v):
            break

    while si < len(times):
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    elapsed = time.perf_counter() - t_start
    perf = kernel.perf_dict("SIMON_encounterNN", steps, dt, T, n_samples, elapsed)
    used_safe = max(int(encounter_used), 1)
    perf.update({
        "gate_candidates": int(gate_candidates),
        "encounter_attempts": int(encounter_attempts),
        "encounter_used": int(encounter_used),
        "encounter_fallback": int(encounter_fallback),
        "first_used_t": float(first_used_t),
        "last_used_t": float(last_used_t),
        "pred_pos_norm_mean": float(pred_pos_norm_sum / used_safe),
        "pred_vel_norm_mean": float(pred_vel_norm_sum / used_safe),
        "min_r": float(min_r_global),
        "max_radius": float(max_radius_global),
    })
    return times, pos_out, vel_out, perf


# =============================================================================
# Output helpers
# =============================================================================
def write_event_csv(path: str, rows: List[Dict[str, object]]) -> None:
    headers = [
        "t_start", "t_exit", "used", "reason", "r_pair", "v_rad_norm", "v_tan_norm",
        "min_r_window", "pred_pos_norm", "pred_vel_norm", "relE_corr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pct_gain(base: float, new: float) -> float:
    return 100.0 * (base - new) / max(abs(base), 1e-30)




# =============================================================================
# Local encounter-window metrics
# =============================================================================
def _safe_div(num: float, den: float) -> float:
    return float(num) / max(abs(float(den)), 1e-30)


def _window_mask(times: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    """Return a boolean mask for samples inside [t_start, t_end]."""
    lo = max(float(times[0]), float(t_start))
    hi = min(float(times[-1]), float(t_end))
    if hi < lo:
        return np.zeros_like(times, dtype=bool)
    return (times >= lo - 1e-12) & (times <= hi + 1e-12)


def _union_intervals(intervals: List[Tuple[float, float]], T: float) -> List[Tuple[float, float]]:
    """Merge overlapping time intervals after clipping to [0,T]."""
    cleaned: List[Tuple[float, float]] = []
    for a, b in intervals:
        lo = max(0.0, float(a))
        hi = min(float(T), float(b))
        if hi >= lo:
            cleaned.append((lo, hi))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])
    merged = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 1e-12:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _mask_from_intervals(times: np.ndarray, intervals: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(times, dtype=bool)
    for lo, hi in intervals:
        mask |= _window_mask(times, lo, hi)
    return mask


def _rms_timeavg(err: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")
    e = np.asarray(err[mask], dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


def _point_metric_row(
    label: str,
    event_index: str,
    t_point: float,
    times: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
) -> Dict[str, object]:
    idx = int(np.argmin(np.abs(times - float(t_point))))
    pos_err_no = rms_sep(pos_no, pos_ref)
    pos_err_en = rms_sep(pos_en, pos_ref)
    vel_err_no = rms_sep(vel_no, vel_ref)
    vel_err_en = rms_sep(vel_en, vel_ref)
    return {
        "scope": label,
        "event_index": event_index,
        "t_start": float(times[idx]),
        "t_end": float(times[idx]),
        "duration_years": 0.0,
        "n_samples": 1,
        "pos_noNN_timeavg": float(pos_err_no[idx]),
        "pos_encounterNN_timeavg": float(pos_err_en[idx]),
        "pos_gain_pct": pct_gain(float(pos_err_no[idx]), float(pos_err_en[idx])),
        "vel_noNN_timeavg": float(vel_err_no[idx]),
        "vel_encounterNN_timeavg": float(vel_err_en[idx]),
        "vel_gain_pct": pct_gain(float(vel_err_no[idx]), float(vel_err_en[idx])),
        "pos_noNN_median": float(pos_err_no[idx]),
        "pos_encounterNN_median": float(pos_err_en[idx]),
        "vel_noNN_median": float(vel_err_no[idx]),
        "vel_encounterNN_median": float(vel_err_en[idx]),
        "pos_noNN_p95": float(pos_err_no[idx]),
        "pos_encounterNN_p95": float(pos_err_en[idx]),
        "vel_noNN_p95": float(vel_err_no[idx]),
        "vel_encounterNN_p95": float(vel_err_en[idx]),
        "first_sample_t": float(times[idx]),
        "last_sample_t": float(times[idx]),
        "interpretation_hint": "nearest sampled trajectory point to the requested instant; use as an exit-state proxy, not an exact IAS15 dense-output event metric",
    }


def _window_metric_row(
    label: str,
    event_index: str,
    t_start: float,
    t_end: float,
    times: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
) -> Dict[str, object]:
    mask = _window_mask(times, t_start, t_end)
    n = int(np.sum(mask))
    pos_err_no = rms_sep(pos_no, pos_ref)
    pos_err_en = rms_sep(pos_en, pos_ref)
    vel_err_no = rms_sep(vel_no, vel_ref)
    vel_err_en = rms_sep(vel_en, vel_ref)

    if n == 0:
        return {
            "scope": label,
            "event_index": event_index,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "duration_years": float(max(0.0, t_end - t_start)),
            "n_samples": 0,
            "pos_noNN_timeavg": float("nan"),
            "pos_encounterNN_timeavg": float("nan"),
            "pos_gain_pct": float("nan"),
            "vel_noNN_timeavg": float("nan"),
            "vel_encounterNN_timeavg": float("nan"),
            "vel_gain_pct": float("nan"),
            "pos_noNN_median": float("nan"),
            "pos_encounterNN_median": float("nan"),
            "vel_noNN_median": float("nan"),
            "vel_encounterNN_median": float("nan"),
            "pos_noNN_p95": float("nan"),
            "pos_encounterNN_p95": float("nan"),
            "vel_noNN_p95": float("nan"),
            "vel_encounterNN_p95": float("nan"),
            "first_sample_t": float("nan"),
            "last_sample_t": float("nan"),
            "interpretation_hint": "no samples in requested window",
        }

    pos_no_ta = _rms_timeavg(pos_err_no, mask)
    pos_en_ta = _rms_timeavg(pos_err_en, mask)
    vel_no_ta = _rms_timeavg(vel_err_no, mask)
    vel_en_ta = _rms_timeavg(vel_err_en, mask)
    return {
        "scope": label,
        "event_index": event_index,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "duration_years": float(max(0.0, float(t_end) - float(t_start))),
        "n_samples": n,
        "pos_noNN_timeavg": pos_no_ta,
        "pos_encounterNN_timeavg": pos_en_ta,
        "pos_gain_pct": pct_gain(pos_no_ta, pos_en_ta),
        "vel_noNN_timeavg": vel_no_ta,
        "vel_encounterNN_timeavg": vel_en_ta,
        "vel_gain_pct": pct_gain(vel_no_ta, vel_en_ta),
        "pos_noNN_median": float(np.median(pos_err_no[mask])),
        "pos_encounterNN_median": float(np.median(pos_err_en[mask])),
        "vel_noNN_median": float(np.median(vel_err_no[mask])),
        "vel_encounterNN_median": float(np.median(vel_err_en[mask])),
        "pos_noNN_p95": float(np.percentile(pos_err_no[mask], 95)),
        "pos_encounterNN_p95": float(np.percentile(pos_err_en[mask], 95)),
        "vel_noNN_p95": float(np.percentile(vel_err_no[mask], 95)),
        "vel_encounterNN_p95": float(np.percentile(vel_err_en[mask], 95)),
        "first_sample_t": float(times[mask][0]),
        "last_sample_t": float(times[mask][-1]),
        "interpretation_hint": "positive gain means EncounterNN is closer to IAS15 than SIMON-noNN in this window",
    }


def _union_metric_row(
    label: str,
    intervals: List[Tuple[float, float]],
    times: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    T: float,
) -> Dict[str, object]:
    merged = _union_intervals(intervals, T)
    mask = _mask_from_intervals(times, merged)
    pos_err_no = rms_sep(pos_no, pos_ref)
    pos_err_en = rms_sep(pos_en, pos_ref)
    vel_err_no = rms_sep(vel_no, vel_ref)
    vel_err_en = rms_sep(vel_en, vel_ref)
    n = int(np.sum(mask))
    if n == 0:
        return {
            "scope": label,
            "event_index": "ALL_USED",
            "t_start": float("nan"),
            "t_end": float("nan"),
            "duration_years": 0.0,
            "n_samples": 0,
            "pos_noNN_timeavg": float("nan"),
            "pos_encounterNN_timeavg": float("nan"),
            "pos_gain_pct": float("nan"),
            "vel_noNN_timeavg": float("nan"),
            "vel_encounterNN_timeavg": float("nan"),
            "vel_gain_pct": float("nan"),
            "pos_noNN_median": float("nan"),
            "pos_encounterNN_median": float("nan"),
            "vel_noNN_median": float("nan"),
            "vel_encounterNN_median": float("nan"),
            "pos_noNN_p95": float("nan"),
            "pos_encounterNN_p95": float("nan"),
            "vel_noNN_p95": float("nan"),
            "vel_encounterNN_p95": float("nan"),
            "first_sample_t": float("nan"),
            "last_sample_t": float("nan"),
            "interpretation_hint": "no used EncounterNN windows",
        }
    pos_no_ta = _rms_timeavg(pos_err_no, mask)
    pos_en_ta = _rms_timeavg(pos_err_en, mask)
    vel_no_ta = _rms_timeavg(vel_err_no, mask)
    vel_en_ta = _rms_timeavg(vel_err_en, mask)
    total_duration = sum(hi - lo for lo, hi in merged)
    return {
        "scope": label,
        "event_index": "ALL_USED",
        "t_start": float(merged[0][0]),
        "t_end": float(merged[-1][1]),
        "duration_years": float(total_duration),
        "n_samples": n,
        "pos_noNN_timeavg": pos_no_ta,
        "pos_encounterNN_timeavg": pos_en_ta,
        "pos_gain_pct": pct_gain(pos_no_ta, pos_en_ta),
        "vel_noNN_timeavg": vel_no_ta,
        "vel_encounterNN_timeavg": vel_en_ta,
        "vel_gain_pct": pct_gain(vel_no_ta, vel_en_ta),
        "pos_noNN_median": float(np.median(pos_err_no[mask])),
        "pos_encounterNN_median": float(np.median(pos_err_en[mask])),
        "vel_noNN_median": float(np.median(vel_err_no[mask])),
        "vel_encounterNN_median": float(np.median(vel_err_en[mask])),
        "pos_noNN_p95": float(np.percentile(pos_err_no[mask], 95)),
        "pos_encounterNN_p95": float(np.percentile(pos_err_en[mask], 95)),
        "vel_noNN_p95": float(np.percentile(vel_err_no[mask], 95)),
        "vel_encounterNN_p95": float(np.percentile(vel_err_en[mask], 95)),
        "first_sample_t": float(times[mask][0]),
        "last_sample_t": float(times[mask][-1]),
        "interpretation_hint": "merged union of all used EncounterNN windows; positive gain means EncounterNN is closer to IAS15 than SIMON-noNN",
    }


def compute_encounter_window_metrics(
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    event_rows: List[Dict[str, object]],
    dt: float,
    T: float,
    physical_window_steps: int,
    post_window_steps: int,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Tuple[float, float]]]]:
    """Compute local encounter-window metrics from already-generated trajectories."""
    rows: List[Dict[str, object]] = []
    used_events = [r for r in event_rows if int(r.get("used", 0)) == 1]
    half_width = float(physical_window_steps) * float(dt)
    post_width = float(post_window_steps) * float(dt)

    physical_intervals: List[Tuple[float, float]] = []
    post_to_physical_end_intervals: List[Tuple[float, float]] = []
    post_fixed_intervals: List[Tuple[float, float]] = []
    training_intervals: List[Tuple[float, float]] = []

    for idx, ev in enumerate(used_events, start=1):
        t_entry = float(ev["t_start"])
        t_exit = float(ev["t_exit"])
        physical_start = max(0.0, t_entry - half_width)
        physical_end = min(float(T), t_entry + half_width)
        post_phys_start = max(0.0, t_exit)
        post_phys_end = min(float(T), max(t_exit, physical_end))
        post_fixed_start = max(0.0, t_exit)
        post_fixed_end = min(float(T), t_exit + post_width)
        train_start = max(0.0, t_entry)
        train_end = min(float(T), t_exit)

        physical_intervals.append((physical_start, physical_end))
        post_to_physical_end_intervals.append((post_phys_start, post_phys_end))
        post_fixed_intervals.append((post_fixed_start, post_fixed_end))
        training_intervals.append((train_start, train_end))

        event_label = str(idx)
        rows.append(_window_metric_row(
            "event_training_window_entry_to_exit", event_label, train_start, train_end,
            times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref,
        ))
        rows.append(_point_metric_row(
            "event_exit_nearest_sample", event_label, t_exit,
            times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref,
        ))
        rows.append(_window_metric_row(
            "event_physical_window_entry_centered", event_label, physical_start, physical_end,
            times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref,
        ))
        rows.append(_window_metric_row(
            "event_post_correction_to_physical_end", event_label, post_phys_start, post_phys_end,
            times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref,
        ))
        rows.append(_window_metric_row(
            "event_post_correction_fixed_steps", event_label, post_fixed_start, post_fixed_end,
            times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref,
        ))

    rows.append(_union_metric_row(
        "ALL_training_window_entry_to_exit_union", training_intervals,
        times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref, T,
    ))
    rows.append(_union_metric_row(
        "ALL_physical_window_entry_centered_union", physical_intervals,
        times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref, T,
    ))
    rows.append(_union_metric_row(
        "ALL_post_correction_to_physical_end_union", post_to_physical_end_intervals,
        times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref, T,
    ))
    rows.append(_union_metric_row(
        "ALL_post_correction_fixed_steps_union", post_fixed_intervals,
        times, pos_no, vel_no, pos_en, vel_en, pos_ref, vel_ref, T,
    ))

    return rows, {
        "training": _union_intervals(training_intervals, T),
        "physical": _union_intervals(physical_intervals, T),
        "post_to_physical_end": _union_intervals(post_to_physical_end_intervals, T),
        "post_fixed": _union_intervals(post_fixed_intervals, T),
    }


def write_window_metrics_csv(path: str, rows: List[Dict[str, object]]) -> None:
    headers = [
        "scope", "event_index", "t_start", "t_end", "duration_years", "n_samples",
        "pos_noNN_timeavg", "pos_encounterNN_timeavg", "pos_gain_pct",
        "vel_noNN_timeavg", "vel_encounterNN_timeavg", "vel_gain_pct",
        "pos_noNN_median", "pos_encounterNN_median", "vel_noNN_median", "vel_encounterNN_median",
        "pos_noNN_p95", "pos_encounterNN_p95", "vel_noNN_p95", "vel_encounterNN_p95",
        "first_sample_t", "last_sample_t", "interpretation_hint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_local_timeseries_csv(
    path: str,
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    event_rows: List[Dict[str, object]],
    intervals: Dict[str, List[Tuple[float, float]]],
) -> None:
    """Write per-sample errors for the local windows, useful for plotting/diagnosis."""
    pos_no_err = rms_sep(pos_no, pos_ref)
    pos_en_err = rms_sep(pos_en, pos_ref)
    vel_no_err = rms_sep(vel_no, vel_ref)
    vel_en_err = rms_sep(vel_en, vel_ref)
    mask_any = np.zeros_like(times, dtype=bool)
    for vals in intervals.values():
        mask_any |= _mask_from_intervals(times, vals)

    used_events = [r for r in event_rows if int(r.get("used", 0)) == 1]
    with open(path, "w", encoding="utf-8", newline="") as f:
        headers = [
            "t_yr", "nearest_event_index", "relative_to_entry", "relative_to_exit",
            "in_training_window", "in_physical_window", "in_post_to_physical_end", "in_post_fixed",
            "pos_noNN_err", "pos_encounterNN_err", "pos_gain_instant_pct",
            "vel_noNN_err", "vel_encounterNN_err", "vel_gain_instant_pct",
        ]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for i, t in enumerate(times):
            if not mask_any[i]:
                continue
            if used_events:
                dists = [abs(float(t) - float(ev["t_start"])) for ev in used_events]
                j = int(np.argmin(dists))
                ev = used_events[j]
                event_index = j + 1
                rel_entry = float(t) - float(ev["t_start"])
                rel_exit = float(t) - float(ev["t_exit"])
            else:
                event_index = -1
                rel_entry = float("nan")
                rel_exit = float("nan")
            row = {
                "t_yr": float(t),
                "nearest_event_index": int(event_index),
                "relative_to_entry": rel_entry,
                "relative_to_exit": rel_exit,
                "in_training_window": int(_mask_from_intervals(np.array([t]), intervals.get("training", []))[0]),
                "in_physical_window": int(_mask_from_intervals(np.array([t]), intervals.get("physical", []))[0]),
                "in_post_to_physical_end": int(_mask_from_intervals(np.array([t]), intervals.get("post_to_physical_end", []))[0]),
                "in_post_fixed": int(_mask_from_intervals(np.array([t]), intervals.get("post_fixed", []))[0]),
                "pos_noNN_err": float(pos_no_err[i]),
                "pos_encounterNN_err": float(pos_en_err[i]),
                "pos_gain_instant_pct": pct_gain(float(pos_no_err[i]), float(pos_en_err[i])),
                "vel_noNN_err": float(vel_no_err[i]),
                "vel_encounterNN_err": float(vel_en_err[i]),
                "vel_gain_instant_pct": pct_gain(float(vel_no_err[i]), float(vel_en_err[i])),
            }
            w.writerow(row)


def maybe_write_local_plot(
    path: str,
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    intervals: Dict[str, List[Tuple[float, float]]],
) -> None:
    """Optional diagnostic plot. Safe no-op if matplotlib is unavailable."""
    if not HAS_MPL:
        return
    mask = _mask_from_intervals(times, intervals.get("physical", []))
    if int(np.sum(mask)) < 2:
        return
    pos_no_err = rms_sep(pos_no, pos_ref)
    pos_en_err = rms_sep(pos_en, pos_ref)
    vel_no_err = rms_sep(vel_no, vel_ref)
    vel_en_err = rms_sep(vel_en, vel_ref)

    plt.figure(figsize=(9, 5))
    plt.plot(times[mask], pos_no_err[mask], label="noNN position error")
    plt.plot(times[mask], pos_en_err[mask], label="EncounterNN position error")
    plt.plot(times[mask], vel_no_err[mask], linestyle="--", label="noNN velocity error")
    plt.plot(times[mask], vel_en_err[mask], linestyle="--", label="EncounterNN velocity error")
    plt.xlabel("time (yr)")
    plt.ylabel("RMS error vs IAS15")
    plt.title("Local physical encounter-window errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()



# =============================================================================
# Isolated local replay metrics (corrected methodology)
# =============================================================================
def advance_nonn_to_time(
    x0: np.ndarray,
    v0: np.ndarray,
    m: np.ndarray,
    cfg: HybridConfig,
    dt: float,
    t_target: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Advance the scalar-free noNN kernel from t=0 to t_target.

    This is used to reconstruct the exact encounter-entry state from the same
    noNN branch that the full EncounterNN rollout uses before the first event.
    """
    kernel = NoNNKernel(cfg, m)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    t = 0.0
    steps = 0
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min() + 1e-30))
    a_cache = None
    t0 = time.perf_counter()
    while t < float(t_target) - 1e-14:
        h = min(float(dt), float(t_target) - t)
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t += h
        steps += 1
        if not all_finite_state(x, v):
            raise FloatingPointError("advance_nonn_to_time produced non-finite state")
    elapsed = time.perf_counter() - t0
    return x, v, {"elapsed_sec": float(elapsed), "steps": int(steps), "t_reached": float(t), "r_min": float(r_min)}


def run_nonn_window_from_state(
    x_start: np.ndarray,
    v_start: np.ndarray,
    m: np.ndarray,
    cfg: HybridConfig,
    dt: float,
    duration: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run SIMON-noNN from an arbitrary state for a short duration and return the exit state."""
    kernel = NoNNKernel(cfg, m)
    x = x_start.astype(np.float64).copy()
    v = v_start.astype(np.float64).copy()
    t = 0.0
    steps = 0
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min() + 1e-30))
    min_r_seen = r_min
    a_cache = None
    t0 = time.perf_counter()
    while t < float(duration) - 1e-14:
        h = min(float(dt), float(duration) - t)
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t += h
        steps += 1
        min_r_seen = min(min_r_seen, r_min)
        if not all_finite_state(x, v):
            raise FloatingPointError("run_nonn_window_from_state produced non-finite state")
    elapsed = time.perf_counter() - t0
    return x, v, {"elapsed_sec": float(elapsed), "steps": int(steps), "t_reached": float(t), "min_r_seen": float(min_r_seen)}


def simulate_encounterNN_once_from_state(
    x_start: np.ndarray,
    v_start: np.ndarray,
    m: np.ndarray,
    encounter_model: EncounterNumpyWeights,
    cfg: HybridConfig,
    dt: float,
    T_local: float,
    n_samples: int,
    window_years: float = 0.5,
    energy_gate: float = 0.20,
    max_radius_gate: float = 1e4,
    com_project: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Dict[str, object]]:
    """
    Isolated one-event EncounterNN replay.

    Unlike the full rollout deployment, this function starts all local branches
    from the same event-entry state. It applies the EncounterNN once at
    t = window_years, then continues with the noNN kernel. This cleanly isolates
    the effect of the encounter correction from the 82.56 years of pre-encounter
    divergence.
    """
    if float(T_local) < float(window_years) - 1e-14:
        raise ValueError("T_local must be at least window_years")

    kernel = NoNNKernel(cfg, m)
    times = np.linspace(0.0, float(T_local), int(n_samples))
    pos_out = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)

    x = x_start.astype(np.float64).copy()
    v = v_start.astype(np.float64).copy()
    E_start = total_energy_state(x, v, m, G=cfg.G)
    X_entry, vr_entry, vt_entry, r_pair = make_X_rel18_pair12(x, v, m, float(dt), cfg)

    in_zone3_pair12, gate_vr, gate_vt, gate_r = encounter_gate_features_pair12_fast(x, v, m, cfg)
    gate_ok = bool(in_zone3_pair12)

    si = 0
    t_cur = 0.0

    def fill_outputs_current():
        nonlocal si
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1

    fill_outputs_current()
    _, r2, _ = kernel.geometry(x)
    r_min = float(np.sqrt(r2.min() + 1e-30))
    min_r_seen = r_min
    a_cache = None
    steps = 0
    used = False
    reason = "not_attempted"
    pred_pos_norm = float("nan")
    pred_vel_norm = float("nan")
    relE_corr = float("nan")
    x_no_exit = None
    v_no_exit = None
    x_corr = None
    v_corr = None

    t0 = time.perf_counter()

    # 1) Run the noNN window to the residual-correction exit.
    remaining = float(window_years)
    while remaining > 1e-14:
        h = min(float(dt), remaining)
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t_cur += h
        remaining -= h
        steps += 1
        min_r_seen = min(min_r_seen, r_min)
        fill_outputs_current()
        if not all_finite_state(x, v):
            reason = "nonfinite_during_window"
            break

    x_no_exit = x.copy()
    v_no_exit = v.copy()

    # 2) Apply one residual correction at the exit, if safe.
    if reason != "nonfinite_during_window":
        dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual_np(encounter_model, X_entry)
        x_corr = x_no_exit + dx
        v_corr = v_no_exit + dv
        if com_project:
            x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)
        relE_corr = abs((total_energy_state(x_corr, v_corr, m, G=cfg.G) - E_start) / (abs(E_start) + 1e-30))
        used = True
        reason = "used"
        if not all_finite_state(x_corr, v_corr):
            used = False
            reason = "nonfinite_correction"
        elif max_radius_state(x_corr) > float(max_radius_gate):
            used = False
            reason = "max_radius_gate"
        elif relE_corr > float(energy_gate):
            used = False
            reason = "energy_gate"

        if used:
            x = x_corr.copy()
            v = v_corr.copy()
            _, r2c, _ = kernel.geometry(x)
            r_min = float(np.sqrt(r2c.min() + 1e-30))
            min_r_seen = min(min_r_seen, r_min)
            a_cache = None
            # If a sample was just written at the exit from noNN state, overwrite it
            # with the corrected state so sampled trajectory reflects deployment.
            if si > 0 and abs(times[si - 1] - t_cur) <= max(1e-10, 1e-9 * abs(t_cur)):
                pos_out[si - 1] = x
                vel_out[si - 1] = v
            fill_outputs_current()
        else:
            x = x_no_exit.copy()
            v = v_no_exit.copy()
            a_cache = None

    # 3) Continue with noNN dynamics after the one correction.
    while t_cur < float(T_local) - 1e-14:
        h = min(float(dt), float(T_local) - t_cur)
        x, v, r_min, a_cache, _ = kernel.macro_step(x, v, h, r_min, a_cache)
        t_cur += h
        steps += 1
        min_r_seen = min(min_r_seen, r_min)
        fill_outputs_current()
        if not all_finite_state(x, v):
            reason = "nonfinite_after_correction"
            break

    while si < len(times):
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    elapsed = time.perf_counter() - t0
    perf = kernel.perf_dict("LOCAL_EncounterNN_once", steps, dt, T_local, n_samples, elapsed)
    perf.update({
        "correction_used": int(used),
        "reason": reason,
        "t_entry_local": 0.0,
        "t_exit_local": float(window_years),
        "r_pair_entry": float(r_pair),
        "v_rad_norm_entry": float(vr_entry),
        "v_tan_norm_entry": float(vt_entry),
        "gate_ok_at_entry": int(gate_ok),
        "gate_r_pair": float(gate_r),
        "gate_v_rad_norm": float(gate_vr),
        "gate_v_tan_norm": float(gate_vt),
        "pred_pos_norm": float(pred_pos_norm),
        "pred_vel_norm": float(pred_vel_norm),
        "relE_corr": float(relE_corr),
        "min_r_seen": float(min_r_seen),
    })
    event = {
        "t_entry_local": 0.0,
        "t_exit_local": float(window_years),
        "used": int(used),
        "reason": reason,
        "r_pair_entry": float(r_pair),
        "v_rad_norm_entry": float(vr_entry),
        "v_tan_norm_entry": float(vt_entry),
        "gate_ok_at_entry": int(gate_ok),
        "min_r_seen": float(min_r_seen),
        "pred_pos_norm": float(pred_pos_norm),
        "pred_vel_norm": float(pred_vel_norm),
        "relE_corr": float(relE_corr),
    }
    if x_no_exit is not None:
        event["x_no_exit"] = x_no_exit
        event["v_no_exit"] = v_no_exit
    if x_corr is not None:
        event["x_corr_exit"] = x_corr
        event["v_corr_exit"] = v_corr
    return times, pos_out, vel_out, perf, event


def compute_exact_isolated_exit_metrics(
    x_event: np.ndarray,
    v_event: np.ndarray,
    m: np.ndarray,
    encounter_model: EncounterNumpyWeights,
    cfg: HybridConfig,
    dt: float,
    window_years: float,
    energy_gate: float,
    max_radius_gate: float,
) -> Dict[str, object]:
    """Exact local exit-state comparison, all branches starting from the same event state."""
    times_ias, pos_ias, vel_ias, perf_ias = simulate_rebound_ias15(x_event, v_event, m, cfg.G, float(window_years), 2)
    x_ias_exit = pos_ias[-1].copy()
    v_ias_exit = vel_ias[-1].copy()
    x_no_exit, v_no_exit, perf_no = run_nonn_window_from_state(x_event, v_event, m, cfg, dt, window_years)

    X_entry, vr_entry, vt_entry, r_pair = make_X_rel18_pair12(x_event, v_event, m, float(dt), cfg)
    dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual_np(encounter_model, X_entry)
    x_corr = x_no_exit + dx
    v_corr = v_no_exit + dv
    x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)

    E_start = total_energy_state(x_event, v_event, m, G=cfg.G)
    relE_corr = abs((total_energy_state(x_corr, v_corr, m, G=cfg.G) - E_start) / (abs(E_start) + 1e-30))
    accepted = bool(all_finite_state(x_corr, v_corr) and max_radius_state(x_corr) <= float(max_radius_gate) and relE_corr <= float(energy_gate))

    pos_no_err = float(rms_sep(x_no_exit[None, :, :], x_ias_exit[None, :, :])[0])
    pos_corr_err = float(rms_sep(x_corr[None, :, :], x_ias_exit[None, :, :])[0])
    vel_no_err = float(rms_sep(v_no_exit[None, :, :], v_ias_exit[None, :, :])[0])
    vel_corr_err = float(rms_sep(v_corr[None, :, :], v_ias_exit[None, :, :])[0])

    return {
        "scope": "isolated_exact_exit_state",
        "t_start_local": 0.0,
        "t_exit_local": float(window_years),
        "pos_noNN_exit_error": pos_no_err,
        "pos_encounterNN_exit_error": pos_corr_err,
        "pos_gain_pct": pct_gain(pos_no_err, pos_corr_err),
        "vel_noNN_exit_error": vel_no_err,
        "vel_encounterNN_exit_error": vel_corr_err,
        "vel_gain_pct": pct_gain(vel_no_err, vel_corr_err),
        "r_pair_entry": float(r_pair),
        "v_rad_norm_entry": float(vr_entry),
        "v_tan_norm_entry": float(vt_entry),
        "pred_pos_norm": float(pred_pos_norm),
        "pred_vel_norm": float(pred_vel_norm),
        "relE_corr": float(relE_corr),
        "accepted_by_safety": int(accepted),
        "ias15_exit_time_sec": float(perf_ias["total_time_sec"]),
        "nonn_exit_time_sec": float(perf_no["elapsed_sec"]),
    }


def _local_window_row(
    scope: str,
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    t_start: float,
    t_end: float,
) -> Dict[str, object]:
    mask = _window_mask(times, t_start, t_end)
    n = int(np.sum(mask))
    if n == 0:
        return {
            "scope": scope,
            "t_start_local": float(t_start),
            "t_end_local": float(t_end),
            "n_samples": 0,
            "pos_noNN_timeavg": float("nan"),
            "pos_encounterNN_timeavg": float("nan"),
            "pos_gain_pct": float("nan"),
            "vel_noNN_timeavg": float("nan"),
            "vel_encounterNN_timeavg": float("nan"),
            "vel_gain_pct": float("nan"),
            "pos_noNN_final": float("nan"),
            "pos_encounterNN_final": float("nan"),
            "pos_final_gain_pct": float("nan"),
            "vel_noNN_final": float("nan"),
            "vel_encounterNN_final": float("nan"),
            "vel_final_gain_pct": float("nan"),
        }
    pos_no_err = rms_sep(pos_no, pos_ref)
    pos_en_err = rms_sep(pos_en, pos_ref)
    vel_no_err = rms_sep(vel_no, vel_ref)
    vel_en_err = rms_sep(vel_en, vel_ref)
    idx = np.where(mask)[0]
    last = int(idx[-1])
    pos_no_ta = _rms_timeavg(pos_no_err, mask)
    pos_en_ta = _rms_timeavg(pos_en_err, mask)
    vel_no_ta = _rms_timeavg(vel_no_err, mask)
    vel_en_ta = _rms_timeavg(vel_en_err, mask)
    return {
        "scope": scope,
        "t_start_local": float(t_start),
        "t_end_local": float(t_end),
        "n_samples": n,
        "pos_noNN_timeavg": float(pos_no_ta),
        "pos_encounterNN_timeavg": float(pos_en_ta),
        "pos_gain_pct": pct_gain(pos_no_ta, pos_en_ta),
        "vel_noNN_timeavg": float(vel_no_ta),
        "vel_encounterNN_timeavg": float(vel_en_ta),
        "vel_gain_pct": pct_gain(vel_no_ta, vel_en_ta),
        "pos_noNN_final": float(pos_no_err[last]),
        "pos_encounterNN_final": float(pos_en_err[last]),
        "pos_final_gain_pct": pct_gain(float(pos_no_err[last]), float(pos_en_err[last])),
        "vel_noNN_final": float(vel_no_err[last]),
        "vel_encounterNN_final": float(vel_en_err[last]),
        "vel_final_gain_pct": pct_gain(float(vel_no_err[last]), float(vel_en_err[last])),
    }


def compute_isolated_local_metrics(
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    window_years: float,
    local_horizon_years: float,
    post_window_years: float,
) -> List[Dict[str, object]]:
    """Trajectory-window metrics for the isolated local replay."""
    rows: List[Dict[str, object]] = []
    rows.append(_local_window_row(
        "isolated_training_window_0_to_exit", times, pos_ref, vel_ref, pos_no, vel_no, pos_en, vel_en,
        0.0, float(window_years)
    ))
    rows.append(_local_window_row(
        "isolated_physical_forward_window", times, pos_ref, vel_ref, pos_no, vel_no, pos_en, vel_en,
        0.0, float(local_horizon_years)
    ))
    rows.append(_local_window_row(
        "isolated_post_correction_to_physical_end", times, pos_ref, vel_ref, pos_no, vel_no, pos_en, vel_en,
        float(window_years), float(local_horizon_years)
    ))
    rows.append(_local_window_row(
        "isolated_post_correction_fixed_steps", times, pos_ref, vel_ref, pos_no, vel_no, pos_en, vel_en,
        float(window_years), min(float(local_horizon_years), float(window_years) + float(post_window_years))
    ))
    return rows


def write_isolated_metrics_csv(path: str, rows: List[Dict[str, object]]) -> None:
    headers = [
        "scope", "t_start_local", "t_end_local", "n_samples",
        "pos_noNN_timeavg", "pos_encounterNN_timeavg", "pos_gain_pct",
        "vel_noNN_timeavg", "vel_encounterNN_timeavg", "vel_gain_pct",
        "pos_noNN_final", "pos_encounterNN_final", "pos_final_gain_pct",
        "vel_noNN_final", "vel_encounterNN_final", "vel_final_gain_pct",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def write_exact_exit_csv(path: str, row: Dict[str, object]) -> None:
    headers = [
        "scope", "t_start_local", "t_exit_local",
        "pos_noNN_exit_error", "pos_encounterNN_exit_error", "pos_gain_pct",
        "vel_noNN_exit_error", "vel_encounterNN_exit_error", "vel_gain_pct",
        "r_pair_entry", "v_rad_norm_entry", "v_tan_norm_entry",
        "pred_pos_norm", "pred_vel_norm", "relE_corr", "accepted_by_safety",
        "ias15_exit_time_sec", "nonn_exit_time_sec",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow({h: row.get(h, "") for h in headers})


def write_isolated_timeseries_csv(
    path: str,
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    window_years: float,
) -> None:
    pos_no_err = rms_sep(pos_no, pos_ref)
    pos_en_err = rms_sep(pos_en, pos_ref)
    vel_no_err = rms_sep(vel_no, vel_ref)
    vel_en_err = rms_sep(vel_en, vel_ref)
    with open(path, "w", encoding="utf-8", newline="") as f:
        headers = [
            "local_t_yr", "phase",
            "pos_noNN_err", "pos_encounterNN_err", "pos_gain_instant_pct",
            "vel_noNN_err", "vel_encounterNN_err", "vel_gain_instant_pct",
        ]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for i, t in enumerate(times):
            phase = "pre_correction_window" if float(t) < float(window_years) - 1e-12 else "post_correction"
            w.writerow({
                "local_t_yr": float(t),
                "phase": phase,
                "pos_noNN_err": float(pos_no_err[i]),
                "pos_encounterNN_err": float(pos_en_err[i]),
                "pos_gain_instant_pct": pct_gain(float(pos_no_err[i]), float(pos_en_err[i])),
                "vel_noNN_err": float(vel_no_err[i]),
                "vel_encounterNN_err": float(vel_en_err[i]),
                "vel_gain_instant_pct": pct_gain(float(vel_no_err[i]), float(vel_en_err[i])),
            })


def maybe_write_isolated_plot(
    path: str,
    times: np.ndarray,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    pos_no: np.ndarray,
    vel_no: np.ndarray,
    pos_en: np.ndarray,
    vel_en: np.ndarray,
    window_years: float,
) -> None:
    if not HAS_MPL:
        return
    pos_no_err = rms_sep(pos_no, pos_ref)
    pos_en_err = rms_sep(pos_en, pos_ref)
    vel_no_err = rms_sep(vel_no, vel_ref)
    vel_en_err = rms_sep(vel_en, vel_ref)

    plt.figure(figsize=(9, 5))
    plt.plot(times, pos_no_err, label="local noNN position error")
    plt.plot(times, pos_en_err, label="local EncounterNN position error")
    plt.plot(times, vel_no_err, linestyle="--", label="local noNN velocity error")
    plt.plot(times, vel_en_err, linestyle="--", label="local EncounterNN velocity error")
    plt.axvline(float(window_years), linestyle=":", label="correction applied")
    plt.xlabel("local time since event entry (yr)")
    plt.ylabel("RMS error vs local IAS15")
    plt.title("Isolated local replay from identical encounter-entry state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def find_row(rows: List[Dict[str, object]], scope: str) -> Optional[Dict[str, object]]:
    for r in rows:
        if r.get("scope") == scope:
            return r
    return None

def make_ic1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    M = float(m.sum())
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M
    return x0, v0, m


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Focused SIMON-noNN vs SIMON-encounterNN evaluator with corrected isolated local replay. No scalar NN required."
    )
    ap.add_argument("--encounter_model", default="encounter_surrogate_v3_rollout_local_velocitysafe.pt",
                    help="Final rollout-local 18D residual encounterNN model.")
    ap.add_argument("--out_dir", default="encounterNN_ablation_v7_isolated_local_out")
    ap.add_argument("--dt", type=float, default=0.08,
                    help="Validated dt for the final encounterNN model. Use dt=0.08 unless intentionally testing out of scope.")
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--window_years", type=float, default=0.5,
                    help="The trained residual window length. Correction is applied at this local time.")
    ap.add_argument("--physical_window_steps", type=int, default=25,
                    help="Half-width used in the earlier equivalent-step logic. In this corrected isolated replay, the forward physical horizon is 2*physical_window_steps*dt. For dt=0.08 and 25, this is 4.0 yr.")
    ap.add_argument("--post_window_steps", type=int, default=25,
                    help="Fixed post-correction persistence window in macro steps after corrected exit.")
    ap.add_argument("--local_n_samples", type=int, default=1001,
                    help="Number of samples for the isolated local replay. Default gives high resolution for a 4 yr window.")
    ap.add_argument("--vr_thresh", type=float, default=-0.40)
    ap.add_argument("--energy_gate", type=float, default=0.20)
    ap.add_argument("--max_radius_gate", type=float, default=1e4)
    ap.add_argument("--write_arrays", action="store_true", help="Save rollout and isolated local arrays to NPZ; off by default.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    encounter_path = args.encounter_model
    if not encounter_path.endswith(".pt") and os.path.exists(encounter_path + ".pt"):
        encounter_path += ".pt"

    if abs(float(args.dt) - 0.08) > 1e-12:
        print("[WARNING] The final encounterNN model was validated for dt=0.08 only.")

    local_horizon_years = 2.0 * float(args.physical_window_steps) * float(args.dt)
    post_window_years = float(args.post_window_steps) * float(args.dt)
    if local_horizon_years < float(args.window_years):
        raise ValueError("2*physical_window_steps*dt must be at least window_years")

    print("=" * 96)
    print("SIMON ENCOUNTERNN ABLATION v7 FAST-GATE + CORRECT ISOLATED LOCAL REPLAY")
    print(f"  encounter_model        : {encounter_path}")
    print(f"  global dt/T/samples    : {args.dt:.6f} yr / {args.T:.1f} yr / {args.n_samples}")
    print(f"  gate                   : pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh:.3f}")
    print(f"  trained residual window: {args.window_years:.3f} yr")
    print(f"  isolated local horizon : {local_horizon_years:.3f} yr (= 2*{args.physical_window_steps}*dt)")
    print(f"  post-correction window : {post_window_years:.3f} yr (= {args.post_window_steps}*dt)")
    print(f"  out_dir                : {args.out_dir}")
    print("  stepLevelNN            : NOT loaded, NOT run")
    print("  encounter infer        : NumPy weights, no torch/CUDA in timed full rollout")
    print("  local methodology      : all local branches restart from the SAME event-entry state")
    print("=" * 96)

    encounter_model = load_encounter_numpy(encounter_path)
    x0, v0, m = make_ic1()

    print("[1/4] Global IAS15 reference from original IC...")
    tr, pr, vr_ref, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, float(args.T), int(args.n_samples))
    print(f"      IAS15 time={perf_r['total_time_sec']:.6f}s")

    print("[2/4] Global SIMON-noNN baseline using scalar-free optimized noNN kernel...")
    _, p_no, v_no, perf_no = simulate_simon_noNN(x0, v0, m, cfg, float(args.dt), float(args.T), int(args.n_samples))
    met_no = metric_block(p_no, v_no, pr, vr_ref)
    sp_no = perf_r["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)
    print(f"      noNN pos_timeavg={met_no['pos_timeavg']:.6e} pos_final={met_no['pos_final']:.6e} time={perf_no['total_time_sec']:.6f}s speedup={sp_no:.2f}x")

    print("[3/4] Global SIMON-encounterNN residual-correction rollout...")
    event_rows: List[Dict[str, object]] = []
    _, p_en, v_en, perf_en = simulate_simon_encounterNN(
        x0, v0, m, encounter_model, cfg,
        dt=float(args.dt), T=float(args.T), n_samples=int(args.n_samples),
        window_years=float(args.window_years), vr_thresh=float(args.vr_thresh),
        energy_gate=float(args.energy_gate), max_radius_gate=float(args.max_radius_gate),
        com_project=True, event_rows=event_rows,
    )
    met_en = metric_block(p_en, v_en, pr, vr_ref)
    sp_en = perf_r["total_time_sec"] / max(perf_en["total_time_sec"], 1e-12)
    print(f"      encounterNN pos_timeavg={met_en['pos_timeavg']:.6e} pos_final={met_en['pos_final']:.6e} time={perf_en['total_time_sec']:.6f}s speedup={sp_en:.2f}x used={int(perf_en['encounter_used'])}")

    used_events = [r for r in event_rows if int(r.get("used", 0)) == 1]
    if not used_events:
        raise RuntimeError("No used EncounterNN events were found; isolated local replay cannot be constructed.")
    first_event = used_events[0]
    event_t = float(first_event["t_start"])
    print("[4/4] Correct isolated local replay from same encounter-entry state...")
    print(f"      reconstructing noNN event-entry state at t={event_t:.12f} yr")
    x_event, v_event, adv_info = advance_nonn_to_time(x0, v0, m, cfg, float(args.dt), event_t)
    vr_evt, vt_evt, r_evt = compute_pair_velocity_features_12(x_event, v_event, m, cfg)
    print(f"      event features: r_pair={r_evt:.8f} AU, v_rad_norm={vr_evt:.8f}, v_tan_norm={vt_evt:.8f}")
    print("      local branches: LOCAL-IAS15, LOCAL-noNN, LOCAL-EncounterNN all start from this identical state")

    # Exact exit-state isolation: this is the cleanest test of the trained 0.5 yr residual map.
    exact_exit = compute_exact_isolated_exit_metrics(
        x_event, v_event, m, encounter_model, cfg,
        dt=float(args.dt), window_years=float(args.window_years),
        energy_gate=float(args.energy_gate), max_radius_gate=float(args.max_radius_gate),
    )

    # Full local trajectory isolation over the physical encounter timescale.
    print(f"      running local IAS15 over {local_horizon_years:.3f} yr...")
    tl, p_ref_l, v_ref_l, perf_ref_l = simulate_rebound_ias15(x_event, v_event, m, cfg.G, local_horizon_years, int(args.local_n_samples))
    print(f"      running local noNN over {local_horizon_years:.3f} yr...")
    _, p_no_l, v_no_l, perf_no_l = simulate_simon_noNN(x_event, v_event, m, cfg, float(args.dt), local_horizon_years, int(args.local_n_samples))
    print(f"      running local EncounterNN-once over {local_horizon_years:.3f} yr...")
    _, p_en_l, v_en_l, perf_en_l, local_event = simulate_encounterNN_once_from_state(
        x_event, v_event, m, encounter_model, cfg,
        dt=float(args.dt), T_local=local_horizon_years, n_samples=int(args.local_n_samples),
        window_years=float(args.window_years), energy_gate=float(args.energy_gate),
        max_radius_gate=float(args.max_radius_gate), com_project=True,
    )
    isolated_rows = compute_isolated_local_metrics(
        tl, p_ref_l, v_ref_l, p_no_l, v_no_l, p_en_l, v_en_l,
        window_years=float(args.window_years),
        local_horizon_years=local_horizon_years,
        post_window_years=post_window_years,
    )

    event_csv = os.path.join(args.out_dir, "encounter_events_global.csv")
    write_event_csv(event_csv, event_rows)
    exact_csv = os.path.join(args.out_dir, "isolated_exact_exit_metrics.csv")
    write_exact_exit_csv(exact_csv, exact_exit)
    isolated_csv = os.path.join(args.out_dir, "isolated_local_replay_metrics.csv")
    write_isolated_metrics_csv(isolated_csv, isolated_rows)
    isolated_ts_csv = os.path.join(args.out_dir, "isolated_local_replay_timeseries.csv")
    write_isolated_timeseries_csv(isolated_ts_csv, tl, p_ref_l, v_ref_l, p_no_l, v_no_l, p_en_l, v_en_l, float(args.window_years))
    isolated_plot = os.path.join(args.out_dir, "isolated_local_replay_errors.png")
    maybe_write_isolated_plot(isolated_plot, tl, p_ref_l, v_ref_l, p_no_l, v_no_l, p_en_l, v_en_l, float(args.window_years))

    if args.write_arrays:
        np.savez_compressed(
            os.path.join(args.out_dir, "ablation_and_isolated_arrays.npz"),
            global_times=tr,
            pos_ias15_global=pr,
            vel_ias15_global=vr_ref,
            pos_noNN_global=p_no,
            vel_noNN_global=v_no,
            pos_encounterNN_global=p_en,
            vel_encounterNN_global=v_en,
            local_times=tl,
            pos_ias15_local=p_ref_l,
            vel_ias15_local=v_ref_l,
            pos_noNN_local=p_no_l,
            vel_noNN_local=v_no_l,
            pos_encounterNN_local=p_en_l,
            vel_encounterNN_local=v_en_l,
            x_event=x_event,
            v_event=v_event,
            x0=x0,
            v0=v0,
            m=m,
        )

    # Print the key comparison messages to the screen, as requested.
    global_pos_ta_gain = pct_gain(met_no["pos_timeavg"], met_en["pos_timeavg"])
    global_vel_ta_gain = pct_gain(met_no["vel_timeavg"], met_en["vel_timeavg"])
    physical_row = find_row(isolated_rows, "isolated_physical_forward_window")
    post_row = find_row(isolated_rows, "isolated_post_correction_to_physical_end")
    fixed_post_row = find_row(isolated_rows, "isolated_post_correction_fixed_steps")

    print("\n" + "=" * 96)
    print("KEY COMPARISONS")
    print("=" * 96)
    print("A) Full 0-100 yr global rollout versus global IAS15 branch")
    print(f"   pos_timeavg gain: {global_pos_ta_gain:+.3f}% | vel_timeavg gain: {global_vel_ta_gain:+.3f}%")
    print(f"   pos_final gain  : {pct_gain(met_no['pos_final'], met_en['pos_final']):+.3f}% | vel_final gain  : {pct_gain(met_no['vel_final'], met_en['vel_final']):+.3f}%")
    print("B) Correct isolated 0.5 yr exit-state test from the SAME event-entry state")
    print(f"   pos exit error: noNN={exact_exit['pos_noNN_exit_error']:.6e}, EncounterNN={exact_exit['pos_encounterNN_exit_error']:.6e}, gain={exact_exit['pos_gain_pct']:+.3f}%")
    print(f"   vel exit error: noNN={exact_exit['vel_noNN_exit_error']:.6e}, EncounterNN={exact_exit['vel_encounterNN_exit_error']:.6e}, gain={exact_exit['vel_gain_pct']:+.3f}%")
    if physical_row is not None:
        print(f"C) Correct isolated physical forward window 0 -> {local_horizon_years:.3f} yr")
        print(f"   pos timeavg gain: {float(physical_row['pos_gain_pct']):+.3f}% | vel timeavg gain: {float(physical_row['vel_gain_pct']):+.3f}%")
        print(f"   pos final gain  : {float(physical_row['pos_final_gain_pct']):+.3f}% | vel final gain  : {float(physical_row['vel_final_gain_pct']):+.3f}%")
    if post_row is not None:
        print(f"D) Correct isolated post-correction window {args.window_years:.3f} -> {local_horizon_years:.3f} yr")
        print(f"   pos timeavg gain: {float(post_row['pos_gain_pct']):+.3f}% | vel timeavg gain: {float(post_row['vel_gain_pct']):+.3f}%")
    if fixed_post_row is not None:
        print(f"E) Correct isolated fixed post-correction window {args.window_years:.3f} -> {min(local_horizon_years, args.window_years + post_window_years):.3f} yr")
        print(f"   pos timeavg gain: {float(fixed_post_row['pos_gain_pct']):+.3f}% | vel timeavg gain: {float(fixed_post_row['vel_gain_pct']):+.3f}%")
    print("=" * 96)

    summary_path = os.path.join(args.out_dir, "encounterNN_ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SIMON encounterNN ablation v7 fast-gate no-scalar NumPy with corrected isolated local replay\n")
        f.write("=" * 96 + "\n")
        f.write("Purpose: compare IAS15, SIMON-noNN, and SIMON-encounterNN on IC1, then isolate the EncounterNN effect by restarting all local branches from the same encounter-entry state.\n")
        f.write("Only the encounter-level residual NN is evaluated here; scalarNN is not loaded or run.\n")
        f.write("EncounterNN deployment formula: corrected_exit = noNN_exit + predicted_residual.\n")
        f.write("Full rollout inference uses NumPy weights only; no torch/CUDA call occurs inside the timed rollout.\n")
        f.write("Corrected local methodology: the local IAS15, local noNN, and local EncounterNN branches all start from the identical noNN event-entry state. This removes all pre-encounter drift and isolates the NN effect during the encounter period.\n\n")

        f.write("Configuration\n")
        f.write("-" * 96 + "\n")
        f.write(f"encounter_model: {encounter_path}\n")
        f.write(f"dt: {args.dt:.8f}\nT_years_global: {args.T:.8f}\nn_samples_global: {args.n_samples}\n")
        f.write(f"window_years_training_residual: {args.window_years:.8f}\n")
        f.write(f"physical_window_steps_half_width_from_discussion: {args.physical_window_steps}\n")
        f.write(f"isolated_forward_physical_horizon_years: {local_horizon_years:.8f}\n")
        f.write(f"post_window_steps: {args.post_window_steps}\npost_window_years: {post_window_years:.8f}\n")
        f.write(f"local_n_samples: {args.local_n_samples}\n")
        f.write(f"vr_thresh: {args.vr_thresh:.8f}\nenergy_gate: {args.energy_gate:.8f}\n")
        f.write("encounter_inference: numpy_weights_no_torch_in_hot_loop\n\n")

        f.write("Global runtime\n")
        f.write("-" * 96 + "\n")
        f.write(f"IAS15_total_time_sec: {perf_r['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_total_time_sec: {perf_no['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_speedup_vs_IAS15: {sp_no:.6f}\n")
        f.write(f"SIMON_encounterNN_total_time_sec: {perf_en['total_time_sec']:.9f}\n")
        f.write(f"SIMON_encounterNN_speedup_vs_IAS15: {sp_en:.6f}\n\n")

        f.write("Global accuracy vs global IAS15 branch\n")
        f.write("-" * 96 + "\n")
        f.write("method\tpos_final\tpos_timeavg\tpos_med\tpos_p95\tvel_final\tvel_timeavg\tvel_med\tvel_p95\ttotal_time_sec\tspeedup_vs_ias15\n")
        f.write(f"SIMON-noNN\t{met_no['pos_final']:.9e}\t{met_no['pos_timeavg']:.9e}\t{met_no['pos_med']:.9e}\t{met_no['pos_p95']:.9e}\t{met_no['vel_final']:.9e}\t{met_no['vel_timeavg']:.9e}\t{met_no['vel_med']:.9e}\t{met_no['vel_p95']:.9e}\t{perf_no['total_time_sec']:.9f}\t{sp_no:.6f}\n")
        f.write(f"SIMON-encounterNN\t{met_en['pos_final']:.9e}\t{met_en['pos_timeavg']:.9e}\t{met_en['pos_med']:.9e}\t{met_en['pos_p95']:.9e}\t{met_en['vel_final']:.9e}\t{met_en['vel_timeavg']:.9e}\t{met_en['vel_med']:.9e}\t{met_en['vel_p95']:.9e}\t{perf_en['total_time_sec']:.9f}\t{sp_en:.6f}\n\n")

        f.write("Global delta vs SIMON-noNN\n")
        f.write("-" * 96 + "\n")
        f.write(f"SIMON-encounterNN_pos_final_gain_pct: {pct_gain(met_no['pos_final'], met_en['pos_final']):+.6f}\n")
        f.write(f"SIMON-encounterNN_pos_timeavg_gain_pct: {global_pos_ta_gain:+.6f}\n")
        f.write(f"SIMON-encounterNN_vel_final_gain_pct: {pct_gain(met_no['vel_final'], met_en['vel_final']):+.6f}\n")
        f.write(f"SIMON-encounterNN_vel_timeavg_gain_pct: {global_vel_ta_gain:+.6f}\n\n")

        f.write("Global EncounterNN event stats\n")
        f.write("-" * 96 + "\n")
        for key in ["gate_candidates", "encounter_attempts", "encounter_used", "encounter_fallback", "first_used_t", "last_used_t", "pred_pos_norm_mean", "pred_vel_norm_mean", "min_r", "max_radius"]:
            f.write(f"{key}: {perf_en.get(key)}\n")
        f.write(f"event_csv: {event_csv}\n\n")

        f.write("Correct isolated local replay methodology\n")
        f.write("-" * 96 + "\n")
        f.write("The previous global-window method looked only at samples near the encounter but still used trajectories that had already evolved for 82.56 yr from t=0. That does not fully isolate the NN effect because pre-encounter branch drift is already present.\n")
        f.write("This v7 method reconstructs the exact noNN event-entry state at the first used event, then launches three new local branches from that identical state: LOCAL-IAS15, LOCAL-noNN, and LOCAL-EncounterNN-once.\n")
        f.write("The EncounterNN branch runs noNN for the trained 0.5 yr window, applies the predicted 18D residual at the exit, and then continues with noNN for the rest of the local physical horizon.\n")
        f.write("Positive gain_pct means EncounterNN is closer to the LOCAL IAS15 branch than LOCAL noNN. Negative gain_pct means it is worse in that isolated local test.\n\n")

        f.write("Reconstructed event-entry state\n")
        f.write("-" * 96 + "\n")
        f.write(f"global_event_time_yr: {event_t:.12f}\n")
        f.write(f"advance_to_event_steps: {adv_info['steps']}\n")
        f.write(f"r_pair_entry: {r_evt:.12e}\n")
        f.write(f"v_rad_norm_entry: {vr_evt:.12e}\n")
        f.write(f"v_tan_norm_entry: {vt_evt:.12e}\n")
        f.write(f"local_correction_used: {local_event.get('used')}\n")
        f.write(f"local_correction_reason: {local_event.get('reason')}\n")
        f.write(f"local_correction_relE_corr: {local_event.get('relE_corr')}\n\n")

        f.write("Exact isolated exit-state metric\n")
        f.write("-" * 96 + "\n")
        f.write("scope\tt_start_local\tt_exit_local\tpos_noNN_exit_error\tpos_encounterNN_exit_error\tpos_gain_pct\tvel_noNN_exit_error\tvel_encounterNN_exit_error\tvel_gain_pct\taccepted_by_safety\n")
        f.write(
            f"{exact_exit['scope']}\t{exact_exit['t_start_local']:.9f}\t{exact_exit['t_exit_local']:.9f}\t"
            f"{exact_exit['pos_noNN_exit_error']:.9e}\t{exact_exit['pos_encounterNN_exit_error']:.9e}\t{exact_exit['pos_gain_pct']:+.6f}\t"
            f"{exact_exit['vel_noNN_exit_error']:.9e}\t{exact_exit['vel_encounterNN_exit_error']:.9e}\t{exact_exit['vel_gain_pct']:+.6f}\t{exact_exit['accepted_by_safety']}\n\n"
        )
        f.write(f"isolated_exact_exit_metrics_csv: {exact_csv}\n\n")

        f.write("Isolated local trajectory-window metrics\n")
        f.write("-" * 96 + "\n")
        f.write("scope\tt_start_local\tt_end_local\tn_samples\tpos_noNN_timeavg\tpos_encounterNN_timeavg\tpos_gain_pct\tvel_noNN_timeavg\tvel_encounterNN_timeavg\tvel_gain_pct\tpos_final_gain_pct\tvel_final_gain_pct\n")
        for row in isolated_rows:
            f.write(
                f"{row['scope']}\t{float(row['t_start_local']):.9f}\t{float(row['t_end_local']):.9f}\t{int(row['n_samples'])}\t"
                f"{float(row['pos_noNN_timeavg']):.9e}\t{float(row['pos_encounterNN_timeavg']):.9e}\t{float(row['pos_gain_pct']):+.6f}\t"
                f"{float(row['vel_noNN_timeavg']):.9e}\t{float(row['vel_encounterNN_timeavg']):.9e}\t{float(row['vel_gain_pct']):+.6f}\t"
                f"{float(row['pos_final_gain_pct']):+.6f}\t{float(row['vel_final_gain_pct']):+.6f}\n"
            )
        f.write("\n")
        f.write(f"isolated_local_replay_metrics_csv: {isolated_csv}\n")
        f.write(f"isolated_local_replay_timeseries_csv: {isolated_ts_csv}\n")
        if HAS_MPL:
            f.write(f"isolated_local_replay_plot_png: {isolated_plot}\n")
        f.write("\n")

        f.write("Paper-story interpretation guide\n")
        f.write("-" * 96 + "\n")
        f.write("1. Use the exact isolated exit-state metric as the cleanest proof of whether the residual model learned the 0.5 yr encounter map.\n")
        f.write("2. Use the isolated physical forward-window metric to test whether that correction remains beneficial over the physically motivated encounter timescale.\n")
        f.write("3. Use the full 100 yr global rollout only as a branch-sensitivity / deployment result, not as the clean local NN-effect metric.\n")
        f.write("4. The scalar Zone-3 production baseline remains separate; this file intentionally does not load or run it.\n\n")

        f.write("Notes\n")
        f.write("- This evaluator is intentionally focused on the encounterNN method only.\n")
        f.write("- Step-level Zone-3 NN comparisons should be taken from the established v4/v5 evaluator outputs, not rerun here.\n")
        f.write("- EncounterNN is currently validated for dt=0.08, IC1, active pair 1-2, window_years=0.5.\n")

    print("\nFinal global ablation table")
    print("method              pos_final     pos_timeavg   vel_final     vel_timeavg   time(s)   speedup")
    print("-" * 94)
    print(f"SIMON-noNN         {met_no['pos_final']:11.4e} {met_no['pos_timeavg']:12.4e} {met_no['vel_final']:11.4e} {met_no['vel_timeavg']:12.4e} {perf_no['total_time_sec']:8.4f} {sp_no:8.3f}x")
    print(f"SIMON-encounterNN  {met_en['pos_final']:11.4e} {met_en['pos_timeavg']:12.4e} {met_en['vel_final']:11.4e} {met_en['vel_timeavg']:12.4e} {perf_en['total_time_sec']:8.4f} {sp_en:8.3f}x")
    print(f"\n[done] wrote {summary_path}")
    print(f"[done] wrote {event_csv}")
    print(f"[done] wrote {exact_csv}")
    print(f"[done] wrote {isolated_csv}")
    print(f"[done] wrote {isolated_ts_csv}")
    if HAS_MPL and os.path.exists(isolated_plot):
        print(f"[done] wrote {isolated_plot}")


if __name__ == "__main__":
    main()
