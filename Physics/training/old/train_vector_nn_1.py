# train_vector_nn.py  (v2 — fixed relative loss + best_vl init)
# Trains a Vector NN baseline: predicts full pairwise force vector (Fx, Fy, Fz) directly.
# Companion to train_pair_correction_new.py — same architecture size, same training
# hyperparameters, same data distribution.
#
# Key differences from SIMON (PairCorrectionNN):
#   - 5 inputs  [rx, ry, rz, log_mi, log_mj]  vs SIMON's 3  [log_r, log_mi, log_mj]
#   - 3 outputs [Fx, Fy, Fz]                  vs SIMON's 1  [log_c]
#   - Relative MSE loss (scale-invariant)      vs SIMON's MSE on log(c)
#   - Learns force direction from data         vs SIMON's analytic direction (always exact)
#
# Run:  python train_vector_nn.py
# Out:  vector_nn.pt  +  training_log_vector.png

import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ── Model ─────────────────────────────────────────────────────────────────────
class VectorNN(nn.Module):
    """
    5 -> 32 -> 32 -> 32 -> 3  with SiLU activations.

    Inputs  (5 features, float32):
        [rx, ry, rz, log(m_i), log(m_j)]
        (rx, ry, rz) = r_ij = pos_j - pos_i, the raw relative position vector.

    Output  (3 values):
        [Fx, Fy, Fz] -- pairwise gravitational force vector on body i from body j,
        targeting the analytic softened force: G*mi*mj/r_soft^3 * r_ij.

    Design note:
        SIMON receives only log(r_soft) -- a scalar -- so its output is rotationally
        invariant and it recovers direction from exact geometry at every timestep.
        VectorNN receives the full displacement vector and must learn both magnitude
        and direction simultaneously. At inference time inside the simulation, any
        directional error the NN makes propagates directly into trajectories -- unlike
        SIMON where direction is always geometrically exact.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer('input_mean', torch.zeros(5))
        self.register_buffer('input_std',  torch.ones(5))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8))


# ── Loss ──────────────────────────────────────────────────────────────────────
def relative_mse_loss(pred, true):
    """
    Scale-invariant relative MSE loss.

        loss_i = |F_pred_i - F_true_i|^2 / |F_true_i|^2

    Why not plain MSE:
        Force magnitudes span ~15 orders of magnitude in this dataset (2e-8 to 1e7).
        Plain MSE is dominated by the largest forces so the model ignores the far field.
        Relative MSE weights every sample equally regardless of distance scale --
        the same reason SIMON uses log-space MSE on log(c) rather than raw MSE on c.
    """
    true_norm_sq = (true * true).sum(dim=1, keepdim=True).clamp(min=1e-30)
    rel_err = (pred - true) / true_norm_sq.sqrt()
    return (rel_err * rel_err).mean()


# ── Training data generator ───────────────────────────────────────────────────
def generate_vector_training_data(n=50000, eps=3e-4, G=1.0, seed=42):
    """
    Generate pairwise force vector training data analytically (no REBOUND needed).

    Sampling mirrors train_pair_correction_new.py exactly:
      60% transition zone  r in [0.5*eps, 50*eps]
      40% far field        r in [50*eps,  10.0]

    For each sample:
      1. Sample scalar distance r from log-uniform distribution.
      2. Sample random unit direction r_hat uniformly on S^2 (Gaussian -> normalise).
      3. r_ij = r * r_hat  (relative position vector).
      4. Analytic softened force: F = G*mi*mj/r_soft^3 * r_ij

    Features:  [rx, ry, rz, log(m_i), log(m_j)]    shape (n, 5)  float32
    Targets:   [Fx, Fy, Fz]                          shape (n, 3)  float32
    """
    rng = np.random.RandomState(seed)

    n_trans = int(0.6 * n)
    n_far   = n - n_trans
    log_r = np.concatenate([
        rng.uniform(np.log(0.5 * eps), np.log(50 * eps), n_trans),
        rng.uniform(np.log(50 * eps),  np.log(10.0),     n_far),
    ])
    rng.shuffle(log_r)
    r = np.exp(log_r)

    # Random unit directions uniformly on S^2
    raw  = rng.randn(n, 3)
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-30)
    r_ij = r[:, None] * dirs                                # (n, 3)

    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)
    mi = np.exp(log_mi)
    mj = np.exp(log_mj)

    r_soft = np.sqrt(r**2 + eps**2)
    F_mag  = G * mi * mj / r_soft**3
    F_vec  = F_mag[:, None] * r_ij                          # (n, 3) force vector

    features = np.stack([
        r_ij[:, 0], r_ij[:, 1], r_ij[:, 2],
        log_mi, log_mj,
    ], axis=1).astype(np.float32)
    targets = F_vec.astype(np.float32)

    F_norms = np.linalg.norm(F_vec, axis=1)
    print(f"[train_vector] {n} samples generated")
    print(f"  r   range : [{r.min():.2e}, {r.max():.2e}]")
    print(f"  |F| range : [{F_norms.min():.2e}, {F_norms.max():.2e}]"
          f"  mean={F_norms.mean():.2e}  std={F_norms.std():.2e}")
    print(f"  Loss: relative MSE (scale-invariant across all force magnitudes)")
    return features, targets


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_mean = features.mean(0)
    feat_std  = features.std(0)

    N   = len(features)
    idx = np.random.permutation(N)
    nv  = int(0.2 * N)

    Xt = torch.tensor(features[idx[nv:]], device=device)
    yt = torch.tensor(targets[idx[nv:]],  device=device)
    Xv = torch.tensor(features[idx[:nv]], device=device)
    yv = torch.tensor(targets[idx[:nv]],  device=device)

    model = VectorNN(hidden=hidden).to(device)
    model.input_mean = torch.tensor(feat_mean, device=device)
    model.input_std  = torch.tensor(feat_std,  device=device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 100)

    # FIX 1: float('inf') -- val loss starts at ~10^10, must initialise above that
    best_vl = float('inf')
    best_sd = None

    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)
        # FIX 2: relative MSE -- scale-invariant across 15 orders of magnitude
        l = relative_mse_loss(pred, yt)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vl = relative_mse_loss(model(Xv), yv).item()

        tl_hist.append(l.item())
        vl_hist.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs}  train={l.item():.6f}  val={vl:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"[train_vector] {elapsed:.1f}s  |  best val relative-MSE = {best_vl:.6f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, eps=3e-4, G=1.0, device="cpu"):
    """
    Test on x-axis pairs: r_ij = [r, 0, 0], mi=mj=1.
    A well-trained model should have:
      err%      < 10% across all tested r values
      cos(angle) > 0.99  (direction nearly correct)
    """
    model.eval()
    print(f"\n  {'r':>8} | {'|F_true|':>12} | {'|F_pred|':>12} | {'err%':>7} | {'cos(angle)':>10}")
    print("  " + "-"*62)
    for r in [1.5e-4, 5e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0]:
        r_soft     = np.sqrt(r**2 + eps**2)
        F_true_mag = G * 1.0 * 1.0 / r_soft**3 * r
        F_true_vec = np.array([F_true_mag, 0.0, 0.0])
        inp = torch.tensor([[r, 0.0, 0.0, 0.0, 0.0]],
                           dtype=torch.float32, device=device)
        with torch.no_grad():
            F_pred_vec = model(inp).cpu().numpy()[0]
        F_pred_mag = np.linalg.norm(F_pred_vec)
        err_pct    = abs(F_pred_mag - F_true_mag) / (abs(F_true_mag) + 1e-30) * 100
        cos_angle  = np.dot(F_pred_vec, F_true_vec) / (
                     F_pred_mag * np.linalg.norm(F_true_vec) + 1e-30)
        print(f"  {r:8.1e} | {F_true_mag:12.4e} | {F_pred_mag:12.4e}"
              f" | {err_pct:6.2f}% | {cos_angle:10.6f}")
    print()
    print("  cos(angle) near 1.0 = direction correct")
    print("  err% < 10%          = magnitude well learned")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int,   default=32)
    ap.add_argument("--epochs", type=int,   default=5000)
    ap.add_argument("--n",      type=int,   default=50000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="vector_nn.pt")
    args = ap.parse_args()

    print(f"[train_vector] hidden={args.hidden}, epochs={args.epochs}, "
          f"n={args.n}, device={args.device}")

    features, targets = generate_vector_training_data(n=args.n)

    model, tl, vl = train_model(
        features, targets,
        hidden=args.hidden,
        epochs=args.epochs,
        device=args.device,
    )

    sanity_check(model, device=args.device)

    torch.save(model.cpu().state_dict(), args.output)
    size_kb = os.path.getsize(args.output) / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_vector] Saved {args.output}  ({size_kb:.1f} KB)  params={n_params}")

    plt.figure(figsize=(8, 4))
    plt.plot(tl, label="train (relative MSE)")
    plt.plot(vl, label="val (relative MSE)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative MSE loss")
    plt.title("Vector NN Training Log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_log_vector.png", dpi=150)
    print("[train_vector] Saved training_log_vector.png")
    print("[train_vector] Done.")


if __name__ == "__main__":
    main()
