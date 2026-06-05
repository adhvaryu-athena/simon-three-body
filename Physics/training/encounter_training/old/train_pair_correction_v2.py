# train_pair_correction_v2.py
#
# Trains the revised Zone 3 scalar correction NN for SIMON.
#
# Key differences from train_pair_correction_new.py:
#   1. Input dimension: 3 → 4  (adds log_dt as 4th feature)
#   2. Training data: loaded from encounter_data_zone3.npz
#      Target is log(c_opt) from one-step trajectory comparison vs ias15.
#      This is NOT the analytic softening correction — it is a genuine
#      dt-dependent leapfrog discretisation correction.
#   3. Sanity check: shows c vs dt at fixed r (validates dt-dependence).
#
# Everything else carried forward from the original:
#   Same architecture depth (3 hidden layers, SiLU), hidden=32.
#   Same AdamW + CosineAnnealingLR schedule, 5000 epochs.
#   Same 80/20 split, best-val checkpoint.
#
# Output:
#   pair_correction_nn_v2.pt  -- new model weights (4-input)
#   training_log_v2.png       -- train/val loss curves
#
# Run:
#   python train_pair_correction_v2.py
#   python train_pair_correction_v2.py --data encounter_data_zone3.npz
#   python train_pair_correction_v2.py --epochs 8000 --hidden 64

import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Model
# =============================================================================
class PairCorrectionNN(nn.Module):
    """
    4->H->H->H->1 with SiLU.

    Inputs: [log(r_soft), log(m_i), log(m_j), log(dt)]
    Output: log(c_opt)  where c_opt is the trajectory-optimal force scalar.

    Hidden=32 is sufficient: the target surface c(r, dt) is smooth and
    low-dimensional. Masses have weak effect on c_opt (as in v1).
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Input normalisation buffers (4-dimensional, set during training)
        self.register_buffer('input_mean', torch.zeros(4))
        self.register_buffer('input_std',  torch.ones(4))

    def forward(self, x):
        return self.net(
            (x - self.input_mean) / (self.input_std + 1e-8)
        ).squeeze(-1)


# =============================================================================
# Data loader
# =============================================================================
def load_zone3_data(npz_path):
    """
    Load Zone 3 training data from encounter_data_zone3.npz.

    Features: [log_r_soft, log_mi, log_mj, log_dt]   shape (N, 4)
    Targets:  log_c_opt                                shape (N,)

    The log_c_opt values are negative for Zone 3 (c_opt < 1.0 at large dt).
    This is correct: the leapfrog over-binds at large dt and c < 1 damps it.
    """
    data    = np.load(npz_path)
    log_c   = data["log_c_opt"].astype(np.float32)

    # Build 4-feature matrix: [log(r_soft), log(mi), log(mj), log(dt)]
    feat    = np.stack([
        data["r_soft"].astype(np.float32),   # already r_soft, not log yet
        data["log_mi"].astype(np.float32),
        data["log_mj"].astype(np.float32),
        data["log_dt"].astype(np.float32),
    ], axis=1)

    # r_soft is stored as raw AU, need log
    feat[:, 0] = np.log(feat[:, 0] + 1e-30)

    n = len(log_c)
    print(f"[data] Loaded {npz_path}")
    print(f"[data] {n} samples")
    print(f"[data] log_c_opt: mean={log_c.mean():.4f}  std={log_c.std():.4f}"
          f"  min={log_c.min():.4f}  max={log_c.max():.4f}")

    # Show per-dt c_opt_median to confirm data quality
    dt_yr   = np.exp(data["log_dt"].astype(np.float64))
    c_opt   = data["c_opt"].astype(np.float64)
    dt_vals = sorted(set(round(float(v), 4) for v in dt_yr))
    print(f"[data] Per-dt c_opt_median:")
    for dt in dt_vals:
        mask = np.abs(dt_yr - dt) < 1e-5
        print(f"       dt={dt:.3f}  n={mask.sum():6d}  "
              f"c_opt_med={float(np.median(c_opt[mask])):.5f}")

    return feat, log_c


# =============================================================================
# Training loop  (identical structure to original)
# =============================================================================
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Per-feature normalisation (now 4-dimensional)
    feat_mean = features.mean(0)    # (4,)
    feat_std  = features.std(0)     # (4,)

    # 80/20 train/val split
    N   = len(features)
    idx = np.random.permutation(N)
    nv  = int(0.2 * N)

    Xv = torch.tensor(features[idx[:nv]],  device=device)
    yv = torch.tensor(targets[idx[:nv]],   device=device)
    Xt = torch.tensor(features[idx[nv:]],  device=device)
    yt = torch.tensor(targets[idx[nv:]],   device=device)

    model = PairCorrectionNN(hidden=hidden).to(device)
    model.input_mean = torch.tensor(feat_mean, device=device)
    model.input_std  = torch.tensor(feat_std,  device=device)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, epochs, eta_min=lr / 100)
    loss_fn = nn.MSELoss()

    best_vl = 1e9
    best_sd = None
    tl_hist = []
    vl_hist = []
    t0      = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()

        tl_hist.append(loss.item())
        vl_hist.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs}  "
                  f"train={loss.item():.6f}  val={vl:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"[train] {elapsed:.1f}s  best_val={best_vl:.6f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# =============================================================================
# Sanity check  (updated to show dt-dependence -- the key new feature)
# =============================================================================
def sanity_check(model, device="cpu"):
    """
    Show predicted c at representative (r, dt) pairs in Zone 3.

    Expected pattern (from training data):
        dt=0.005: c ≈ 1.00  (leapfrog resolves encounter, no correction needed)
        dt=0.040: c ≈ 0.93  (meaningful impulse reduction at operational dt)
        dt=0.080: c ≈ 0.81  (stronger correction as dt becomes coarser)
        dt=0.100: c ≈ 0.76  (large dt needs largest correction)

    If the model outputs this pattern, dt-awareness is working correctly.
    """
    EPS = 3e-4
    model.eval()

    print("\nSanity check — predicted c vs dt at r=0.10 AU (Zone 3 mid):")
    print(f"  {'dt':>7} | {'c_pred':>8} | {'expected':>10}")
    print("  " + "-" * 32)

    # Expected medians from 80k training data
    expected = {0.005: 0.999, 0.010: 0.994, 0.020: 0.981,
                0.040: 0.925, 0.050: 0.896, 0.060: 0.868,
                0.080: 0.810, 0.100: 0.764}

    r    = 0.10          # AU  -- mid-Zone 3
    mi   = 1.0;  mj = 0.01   # representative masses
    r_s  = float(np.sqrt(r**2 + EPS**2))

    for dt in [0.005, 0.010, 0.020, 0.040, 0.050, 0.060, 0.080, 0.100]:
        feat = torch.tensor(
            [[np.log(r_s), np.log(mi), np.log(mj), np.log(dt)]],
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            c_pred = float(torch.exp(model(feat)).item())
        exp    = expected.get(dt, float("nan"))
        marker = "  OK" if abs(c_pred - exp) < 0.05 else "  CHECK"
        print(f"  {dt:7.3f} | {c_pred:8.5f} | {exp:10.3f}{marker}")

    print()
    print("Sanity check — predicted c vs r at dt=0.04 (fixed dt):")
    print(f"  {'r (AU)':>8} | {'c_pred':>8} | note")
    print("  " + "-" * 36)

    for r in [0.05, 0.06, 0.08, 0.10, 0.12, 0.14]:
        r_s  = float(np.sqrt(r**2 + EPS**2))
        feat = torch.tensor(
            [[np.log(r_s), np.log(1.0), np.log(0.01), np.log(0.04)]],
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            c_pred = float(torch.exp(model(feat)).item())
        print(f"  {r:8.3f} | {c_pred:8.5f}")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train Zone 3 trajectory-correction NN (v2, 4-input)"
    )
    ap.add_argument("--data",   default="encounter_data_zone3.npz",
                    help="Path to Zone 3 training data npz")
    ap.add_argument("--hidden", type=int,   default=32)
    ap.add_argument("--epochs", type=int,   default=5000)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                         else "cpu")
    ap.add_argument("--output", default="pair_correction_nn_v2.pt")
    ap.add_argument("--seed",   type=int,   default=42)
    args = ap.parse_args()

    print(f"[config] hidden={args.hidden}  epochs={args.epochs}"
          f"  lr={args.lr}  device={args.device}")
    print(f"[config] data={args.data}")
    print(f"[config] output={args.output}")

    # Load data
    feat, targ = load_zone3_data(args.data)

    # Train
    model, tl, vl = train_model(
        feat, targ,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    # Sanity check
    sanity_check(model, device=args.device)

    # Save model
    torch.save(model.cpu().state_dict(), args.output)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nSaved {args.output} ({size_kb:.1f} KB)")

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tl, label="train", alpha=0.8)
    ax.plot(vl, label="val",   alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Zone 3 trajectory-correction NN — training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_log_v2.png", dpi=150)
    plt.close(fig)
    print("Saved training_log_v2.png")


if __name__ == "__main__":
    main()
