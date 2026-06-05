"""
train_pair_correction_v3.py

Train the velocity-aware Zone 3 scalar correction NN for SIMON.

v3 target and features
----------------------
Dataset: encounter_data_zone3_v3.npz
Features:
    [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
Target:
    log(c_opt)

This is different from the older analytic softening target.  c_opt is the
trajectory-optimal scalar selected by matching one leapfrog macro-step to an
IAS15 one-step reference.  The velocity features are required because the
optimal correction differs strongly for approaching and receding encounters.

Outputs by default:
    pair_correction_nn_v3.pt
    pair_correction_nn_v3_split.npz
    pair_correction_nn_v3_metrics.npz
    training_log_v3.png              (if matplotlib is available)
    train_pair_correction_v3_summary.txt

Example:
    python -B train_pair_correction_v3.py --data encounter_data_zone3_v3.npz
    python -B train_pair_correction_v3.py --data encounter_data_zone3_v3.npz --epochs 8000 --hidden 64
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]
FEATURE_NAMES = ["log_r_soft", "log_mi", "log_mj", "log_dt", "v_rad_norm", "v_tan_norm"]
DT_LIST = [0.005, 0.010, 0.020, 0.040, 0.050, 0.060, 0.080, 0.100]


class PairCorrectionNN(nn.Module):
    """
    6 -> H -> H -> H -> 1 with SiLU.

    Inputs:
        [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
    Output:
        log(c_opt)
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm).squeeze(-1)


@dataclass
class DatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    raw: Dict[str, np.ndarray]


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    return s


def load_zone3_v3_data(npz_path: str) -> DatasetBundle:
    data = np.load(npz_path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(
            f"{npz_path} is missing required v3 fields: {missing}\n"
            "The v3 model requires v_rad_norm and v_tan_norm."
        )

    raw = {k: data[k].astype(np.float32) for k in REQUIRED_FIELDS}

    for k, arr in raw.items():
        if arr.ndim != 1:
            raise ValueError(f"Field {k} must be 1D, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Field {k} contains non-finite values")

    n = len(raw["r_AU"])
    if any(len(raw[k]) != n for k in REQUIRED_FIELDS):
        raise ValueError("Not all required fields have the same length")

    log_r_soft = np.log(raw["r_soft"].astype(np.float64) + 1e-30).astype(np.float32)
    features = np.stack([
        log_r_soft,
        raw["log_mi"],
        raw["log_mj"],
        raw["log_dt"],
        raw["v_rad_norm"],
        raw["v_tan_norm"],
    ], axis=1).astype(np.float32)

    targets = raw["log_c_opt"].astype(np.float32)

    print(f"[data] Loaded {npz_path}")
    print(f"[data] samples={n}")
    print("[data] feature ranges:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"       {name:12s} min={features[:, i].min(): .4f}  "
              f"med={np.median(features[:, i]): .4f}  max={features[:, i].max(): .4f}")
    print(f"[data] target log_c_opt: mean={targets.mean():.5f}  std={targets.std():.5f}  "
          f"min={targets.min():.5f}  max={targets.max():.5f}")

    dt_yr = np.exp(raw["log_dt"].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt_yr])
    c = raw["c_opt"].astype(np.float64)
    print("[data] per-dt target medians:")
    for dt in sorted(set(rounded)):
        mask = rounded == dt
        print(f"       dt={dt:0.4f}  n={int(mask.sum()):5d}  c_med={np.median(c[mask]):.5f}  "
              f"q25={np.percentile(c[mask],25):.5f}  q75={np.percentile(c[mask],75):.5f}")

    return DatasetBundle(features=features, targets=targets, raw=raw)


def make_splits(n: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    if len(test_idx) == 0:
        raise ValueError("Test split is empty; use a larger dataset")
    return train_idx, val_idx, test_idx


def evaluate_arrays(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    err_log = y_pred_log - y_true_log
    c_true = np.exp(y_true_log)
    c_pred = np.exp(y_pred_log)
    err_c = c_pred - c_true
    return {
        "mse_log": float(np.mean(err_log ** 2)),
        "rmse_log": float(np.sqrt(np.mean(err_log ** 2))),
        "mae_log": float(np.mean(np.abs(err_log))),
        "mae_c": float(np.mean(np.abs(err_c))),
        "median_abs_c": float(np.median(np.abs(err_c))),
        "p95_abs_c": float(np.percentile(np.abs(err_c), 95)),
        "corr_c": float(np.corrcoef(c_true, c_pred)[0, 1]) if len(c_true) > 2 else float("nan"),
        "pred_c_min": float(np.min(c_pred)),
        "pred_c_med": float(np.median(c_pred)),
        "pred_c_max": float(np.max(c_pred)),
    }


def predict_numpy(model: PairCorrectionNN, features: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            xb = torch.tensor(features[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def train_model(bundle: DatasetBundle, hidden: int, epochs: int, lr: float, seed: int,
                device: str, batch_size: int, output_prefix: str) -> Tuple[PairCorrectionNN, Dict[str, np.ndarray], Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = bundle.features
    targets = bundle.targets
    n = len(features)
    train_idx, val_idx, test_idx = make_splits(n, seed=seed)

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)
    feat_std = _safe_std(features[train_idx])

    model = PairCorrectionNN(hidden=hidden).to(device)
    model.input_mean = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    model.input_std = torch.tensor(feat_std, dtype=torch.float32, device=device)

    X_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 100.0)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    train_hist = []
    val_hist = []
    t0 = time.perf_counter()

    n_train = len(train_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss_sum = 0.0
        ep_seen = 0
        for start in range(0, n_train, batch_size):
            ids = perm[start:start + batch_size]
            xb = X_train[ids]
            yb = y_train[ids]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss_sum += float(loss.item()) * len(ids)
            ep_seen += len(ids)
        sched.step()
        train_loss = ep_loss_sum / max(ep_seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), y_val).item())
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep + 1 == epochs:
            print(f"  ep {ep+1:5d}/{epochs}  train={train_loss:.7f}  val={val_loss:.7f}")

    elapsed = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    np.savez_compressed(
        f"{output_prefix}_split.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        feature_names=np.array(FEATURE_NAMES),
        input_mean=feat_mean,
        input_std=feat_std,
        seed=np.array([seed], dtype=np.int64),
    )

    pred_train = predict_numpy(model, features[train_idx], device)
    pred_val = predict_numpy(model, features[val_idx], device)
    pred_test = predict_numpy(model, features[test_idx], device)

    metrics = {}
    for name, idx, pred in [
        ("train", train_idx, pred_train),
        ("val", val_idx, pred_val),
        ("test", test_idx, pred_test),
    ]:
        m = evaluate_arrays(targets[idx].astype(np.float64), pred)
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v

    metrics["best_val_mse"] = float(best_val)
    metrics["elapsed_sec"] = float(elapsed)
    metrics["n_train"] = float(len(train_idx))
    metrics["n_val"] = float(len(val_idx))
    metrics["n_test"] = float(len(test_idx))

    histories = {
        "train_loss": np.array(train_hist, dtype=np.float64),
        "val_loss": np.array(val_hist, dtype=np.float64),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return model, histories, metrics


def print_test_diagnostics(bundle: DatasetBundle, model: PairCorrectionNN, indices: np.ndarray, device: str) -> None:
    features = bundle.features[indices]
    targets = bundle.targets[indices].astype(np.float64)
    pred_log = predict_numpy(model, features, device)
    c_true = np.exp(targets)
    c_pred = np.exp(pred_log)
    raw = bundle.raw
    dt = np.exp(raw["log_dt"][indices].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    vr = raw["v_rad_norm"][indices].astype(np.float64)
    vt = raw["v_tan_norm"][indices].astype(np.float64)
    impr = raw["improvement"][indices].astype(np.float64)

    print("\n[test] Overall held-out metrics")
    m = evaluate_arrays(targets, pred_log)
    for k in ["mse_log", "rmse_log", "mae_log", "mae_c", "median_abs_c", "p95_abs_c", "corr_c"]:
        print(f"       {k:14s}: {m[k]:.6g}")
    print(f"       pred c range : {c_pred.min():.4f} / {np.median(c_pred):.4f} / {c_pred.max():.4f}")
    print(f"       true c range : {c_true.min():.4f} / {np.median(c_true):.4f} / {c_true.max():.4f}")
    print(f"       out-of-gate predicted c (<0.2 or >5): {np.mean((c_pred < 0.2) | (c_pred > 5.0)):.2%}")

    print("\n[test] Per-dt medians on held-out test set")
    print(f"  {'dt':>7} | {'n':>5} | {'true_med':>9} | {'pred_med':>9} | {'MAE_c':>9} | {'impr_med':>9}")
    print("  " + "-" * 61)
    for d in sorted(set(rounded)):
        mask = rounded == d
        print(f"  {d:7.4f} | {int(mask.sum()):5d} | {np.median(c_true[mask]):9.4f} | "
              f"{np.median(c_pred[mask]):9.4f} | {np.mean(np.abs(c_pred[mask]-c_true[mask])):9.4f} | "
              f"{np.median(impr[mask]):9.2%}")

    print("\n[test] Approaching vs receding medians on held-out test set")
    print(f"  {'dt':>7} | {'app_n':>5} | {'app_true':>9} | {'app_pred':>9} | {'rec_n':>5} | {'rec_true':>9} | {'rec_pred':>9}")
    print("  " + "-" * 85)
    for d in sorted(set(rounded)):
        base = rounded == d
        app = base & (vr < 0)
        rec = base & (vr >= 0)
        def med_or_nan(a, mask):
            return float(np.median(a[mask])) if np.any(mask) else float("nan")
        print(f"  {d:7.4f} | {int(app.sum()):5d} | {med_or_nan(c_true,app):9.4f} | {med_or_nan(c_pred,app):9.4f} | "
              f"{int(rec.sum()):5d} | {med_or_nan(c_true,rec):9.4f} | {med_or_nan(c_pred,rec):9.4f}")

    identity = np.abs(c_true - 1.0) < 1e-6
    if np.any(identity):
        print("\n[test] Near-identity targets on held-out test set")
        print(f"       n identity={int(identity.sum())}")
        print(f"       predicted c median={np.median(c_pred[identity]):.5f}")
        print(f"       within [0.90, 1.10]={np.mean((c_pred[identity] >= 0.90) & (c_pred[identity] <= 1.10)):.2%}")

    # A direct sanity probe that specifically checks the new velocity feature.
    print("\n[probe] Fixed-state velocity-direction check")
    print("        r=0.10 AU, m_i=1, m_j=0.01, v_tan_norm=0.65")
    print(f"  {'dt':>7} | {'c_pred approach(vr=-0.8)':>26} | {'c_pred recede(vr=+0.8)':>24}")
    print("  " + "-" * 68)
    for d in DT_LIST:
        r_soft = float(np.sqrt(0.10 ** 2 + (3e-4) ** 2))
        rows = np.array([
            [np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), -0.8, 0.65],
            [np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), +0.8, 0.65],
        ], dtype=np.float32)
        pred = np.exp(predict_numpy(model, rows, device))
        print(f"  {d:7.4f} | {pred[0]:26.5f} | {pred[1]:24.5f}")


def maybe_plot_training(histories: Dict[str, np.ndarray], output_prefix: str) -> None:
    if not HAS_MPL:
        print("[plot] matplotlib not available; skipping training_log_v3.png")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(histories["train_loss"], label="train")
    ax.plot(histories["val_loss"], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on log(c_opt)")
    ax.set_title("Zone 3 v3 6-input NN training curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = "training_log_v3.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[plot] Saved {out}")


def write_summary(path: str, args: argparse.Namespace, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Zone 3 v3 training summary\n")
        f.write("=" * 72 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\nMetrics\n")
        f.write("-" * 72 + "\n")
        for k in sorted(metrics):
            f.write(f"{k}: {metrics[k]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SIMON Zone 3 v3 6-input trajectory-correction NN.")
    ap.add_argument("--data", default="encounter_data_zone3_v3.npz")
    ap.add_argument("--hidden", type=int, default=64,
                    help="Hidden width. 64 is default for the noisier 6-input target.")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="pair_correction_nn_v3.pt")
    args = ap.parse_args()

    output_prefix = os.path.splitext(args.output)[0]
    print(f"[config] data={args.data}")
    print(f"[config] output={args.output}")
    print(f"[config] hidden={args.hidden} epochs={args.epochs} lr={args.lr} batch_size={args.batch_size} seed={args.seed}")
    print(f"[config] device={args.device}")

    bundle = load_zone3_v3_data(args.data)
    model, histories, metrics = train_model(
        bundle=bundle,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        output_prefix=output_prefix,
    )

    torch.save(model.cpu().state_dict(), args.output)
    print(f"\n[save] Saved model: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"[save] Saved split: {output_prefix}_split.npz")

    # Metrics and sanity diagnostics use CPU now that the model has been saved to CPU.
    model_cpu = PairCorrectionNN(hidden=args.hidden)
    model_cpu.load_state_dict(torch.load(args.output, map_location="cpu"))
    model_cpu.eval()
    print_test_diagnostics(bundle, model_cpu, histories["test_idx"], device="cpu")

    np.savez_compressed(
        f"{output_prefix}_metrics.npz",
        **{k: np.array([v], dtype=np.float64) for k, v in metrics.items()},
        train_loss=histories["train_loss"],
        val_loss=histories["val_loss"],
    )
    print(f"[save] Saved metrics: {output_prefix}_metrics.npz")

    maybe_plot_training(histories, output_prefix)
    summary_path = "train_pair_correction_v3_summary.txt"
    write_summary(summary_path, args, metrics)
    print(f"[save] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
