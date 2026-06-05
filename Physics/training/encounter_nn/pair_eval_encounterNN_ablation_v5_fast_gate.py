# pair_eval_encounterNN_ablation_v5_fast_gate.py
#
# Focused SIMON encounter-level residual-NN evaluator with fast gate path.
#
# Purpose
# -------
# Compare only:
#   1. IAS15 reference
#   2. SIMON-noNN baseline
#   3. SIMON-encounterNN = SIMON-noNN local window + 18D residual correction
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
#   python -B pair_eval_encounterNN_ablation_v5_fast_gate.py ^
#       --encounter_model encounter_surrogate_v3_rollout_local_velocitysafe.pt ^
#       --dt 0.08 --T 100 --n_samples 5000 ^
#       --out_dir encounterNN_ablation_v5_fast_gate_out

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
        description="Focused SIMON-noNN vs SIMON-encounterNN evaluator. No scalar NN required."
    )
    ap.add_argument("--encounter_model", default="encounter_surrogate_v3_rollout_local_velocitysafe.pt",
                    help="Final rollout-local 18D residual encounterNN model.")
    ap.add_argument("--out_dir", default="encounterNN_ablation_v5_fast_gate_out")
    ap.add_argument("--dt", type=float, default=0.08,
                    help="Validated dt for the final encounterNN model. Use dt=0.08 unless intentionally testing out of scope.")
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--window_years", type=float, default=0.5)
    ap.add_argument("--vr_thresh", type=float, default=-0.40)
    ap.add_argument("--energy_gate", type=float, default=0.20)
    ap.add_argument("--max_radius_gate", type=float, default=1e4)
    ap.add_argument("--write_arrays", action="store_true", help="Save rollout arrays to NPZ; off by default.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    encounter_path = args.encounter_model
    if not encounter_path.endswith(".pt") and os.path.exists(encounter_path + ".pt"):
        encounter_path += ".pt"

    if abs(float(args.dt) - 0.08) > 1e-12:
        print("[WARNING] The final encounterNN model was validated for dt=0.08 only.")

    print("=" * 88)
    print("SIMON ENCOUNTERNN ABLATION v5 FAST-GATE NO-SCALAR NUMPY")
    print(f"  encounter_model : {encounter_path}")
    print(f"  dt/T/samples    : {args.dt:.6f} yr / {args.T:.1f} yr / {args.n_samples}")
    print(f"  gate            : pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh:.3f}")
    print(f"  window          : {args.window_years:.3f} yr")
    print(f"  out_dir         : {args.out_dir}")
    print("  stepLevelNN        : NOT loaded, NOT run")
    print("  encounter infer : NumPy weights, no torch/CUDA in timed rollout")
    print("=" * 88)

    encounter_model = load_encounter_numpy(encounter_path)
    x0, v0, m = make_ic1()

    print("[1/3] IAS15 reference...")
    tr, pr, vr_ref, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, float(args.T), int(args.n_samples))
    print(f"      IAS15 time={perf_r['total_time_sec']:.6f}s")

    print("[2/3] SIMON-noNN baseline using scalar-free optimized noNN kernel...")
    _, p_no, v_no, perf_no = simulate_simon_noNN(x0, v0, m, cfg, float(args.dt), float(args.T), int(args.n_samples))
    met_no = metric_block(p_no, v_no, pr, vr_ref)
    sp_no = perf_r["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)
    print(f"      noNN pos_timeavg={met_no['pos_timeavg']:.6e} pos_final={met_no['pos_final']:.6e} time={perf_no['total_time_sec']:.6f}s speedup={sp_no:.2f}x")

    print("[3/3] SIMON-encounterNN residual-correction rollout...")
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

    event_csv = os.path.join(args.out_dir, "encounter_events.csv")
    write_event_csv(event_csv, event_rows)

    if args.write_arrays:
        np.savez_compressed(
            os.path.join(args.out_dir, "ablation_arrays.npz"),
            times=tr,
            pos_ias15=pr,
            vel_ias15=vr_ref,
            pos_noNN=p_no,
            vel_noNN=v_no,
            pos_encounterNN=p_en,
            vel_encounterNN=v_en,
            x0=x0,
            v0=v0,
            m=m,
        )

    summary_path = os.path.join(args.out_dir, "encounterNN_ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SIMON encounterNN ablation v5 fast-gate no-scalar NumPy\n")
        f.write("=" * 88 + "\n")
        f.write("Purpose: compare IAS15, SIMON-noNN, and SIMON-encounterNN on IC1.\n")
        f.write("Only the encounter-level residual NN is evaluated here; scalarNN is not loaded or run.\n")
        f.write("EncounterNN deployment formula: corrected_exit = noNN_exit + predicted_residual.\n")
        f.write("EncounterNN inference uses NumPy weights only; no torch/CUDA call occurs inside timed rollout. Full X_rel18 features are built only when the cheap pair-1/2 gate passes.\n\n")
        f.write(f"encounter_model: {encounter_path}\n")
        f.write(f"dt: {args.dt:.8f}\nT_years: {args.T:.8f}\nn_samples: {args.n_samples}\n")
        f.write(f"window_years: {args.window_years:.8f}\nvr_thresh: {args.vr_thresh:.8f}\nenergy_gate: {args.energy_gate:.8f}\n")
        f.write("encounter_inference: numpy_weights_no_torch_in_hot_loop\n\n")

        f.write("Runtime\n")
        f.write("-" * 88 + "\n")
        f.write(f"IAS15_total_time_sec: {perf_r['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_total_time_sec: {perf_no['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_speedup_vs_IAS15: {sp_no:.6f}\n")
        f.write(f"SIMON_encounterNN_total_time_sec: {perf_en['total_time_sec']:.9f}\n")
        f.write(f"SIMON_encounterNN_speedup_vs_IAS15: {sp_en:.6f}\n\n")

        f.write("Accuracy vs IAS15\n")
        f.write("-" * 88 + "\n")
        f.write("method\tpos_final\tpos_timeavg\tpos_med\tpos_p95\tvel_final\tvel_timeavg\tvel_med\tvel_p95\ttotal_time_sec\tspeedup_vs_ias15\n")
        f.write(f"SIMON-noNN\t{met_no['pos_final']:.9e}\t{met_no['pos_timeavg']:.9e}\t{met_no['pos_med']:.9e}\t{met_no['pos_p95']:.9e}\t{met_no['vel_final']:.9e}\t{met_no['vel_timeavg']:.9e}\t{met_no['vel_med']:.9e}\t{met_no['vel_p95']:.9e}\t{perf_no['total_time_sec']:.9f}\t{sp_no:.6f}\n")
        f.write(f"SIMON-encounterNN\t{met_en['pos_final']:.9e}\t{met_en['pos_timeavg']:.9e}\t{met_en['pos_med']:.9e}\t{met_en['pos_p95']:.9e}\t{met_en['vel_final']:.9e}\t{met_en['vel_timeavg']:.9e}\t{met_en['vel_med']:.9e}\t{met_en['vel_p95']:.9e}\t{perf_en['total_time_sec']:.9f}\t{sp_en:.6f}\n\n")

        f.write("EncounterNN event stats\n")
        f.write("-" * 88 + "\n")
        for key in ["gate_candidates", "encounter_attempts", "encounter_used", "encounter_fallback", "first_used_t", "last_used_t", "pred_pos_norm_mean", "pred_vel_norm_mean", "min_r", "max_radius"]:
            f.write(f"{key}: {perf_en.get(key)}\n")
        f.write(f"event_csv: {event_csv}\n\n")

        f.write("Delta vs SIMON-noNN\n")
        f.write("-" * 88 + "\n")
        f.write(f"SIMON-encounterNN_pos_final_gain_pct: {pct_gain(met_no['pos_final'], met_en['pos_final']):+.6f}\n")
        f.write(f"SIMON-encounterNN_pos_timeavg_gain_pct: {pct_gain(met_no['pos_timeavg'], met_en['pos_timeavg']):+.6f}\n")
        f.write(f"SIMON-encounterNN_vel_final_gain_pct: {pct_gain(met_no['vel_final'], met_en['vel_final']):+.6f}\n")
        f.write(f"SIMON-encounterNN_vel_timeavg_gain_pct: {pct_gain(met_no['vel_timeavg'], met_en['vel_timeavg']):+.6f}\n\n")

        f.write("Notes\n")
        f.write("- This evaluator is intentionally focused on the encounterNN method only.\n")
        f.write("- Step-level Zone-3 NN comparisons should be taken from the established v4 evaluator, not rerun here.\n")
        f.write("- EncounterNN is currently validated for dt=0.08, IC1, active pair 1-2, window_years=0.5.\n")

    print("\nFinal ablation table")
    print("method              pos_final     pos_timeavg   vel_final     vel_timeavg   time(s)   speedup")
    print("-" * 94)
    print(f"SIMON-noNN         {met_no['pos_final']:11.4e} {met_no['pos_timeavg']:12.4e} {met_no['vel_final']:11.4e} {met_no['vel_timeavg']:12.4e} {perf_no['total_time_sec']:8.4f} {sp_no:8.3f}x")
    print(f"SIMON-encounterNN  {met_en['pos_final']:11.4e} {met_en['pos_timeavg']:12.4e} {met_en['vel_final']:11.4e} {met_en['vel_timeavg']:12.4e} {perf_en['total_time_sec']:8.4f} {sp_en:8.3f}x")
    print(f"\n[done] wrote {summary_path}")
    print(f"[done] wrote {event_csv}")


if __name__ == "__main__":
    main()
