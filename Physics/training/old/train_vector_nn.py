# train_vector_nn.py
# Trains a Vector NN baseline: predicts full pairwise force vector (Fx, Fy, Fz) directly.
# Companion to train_pair_correction_new.py — same architecture, same training setup,
# different input/output: receives relative position vector + masses, outputs force vector.
#
# Run:  python train_vector_nn.py
# Saves: vector_nn.pt  +  training_log_vector.png

import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ── Model ─────────────────────────────────────────────────────────────────────
class VectorNN(nn.Module):
    """
    5 → 32 → 32 → 32 → 3  with SiLU activations.

    Inputs  (5 features):
        [rx, ry, rz, log(m_i), log(m_j)]
        where (rx, ry, rz) = r_ij = pos_j - pos_i  (raw relative position)

    Output  (3 values):
        [Fx, Fy, Fz]  — the pairwise gravitational force vector on body i from body j
        (unsoftened analytic target: G*mi*mj / r_soft^3 * r_ij)

    Design note:
        The full relative position vector (not just scalar distance) is provided
        so the NN can learn both magnitude AND direction. This is the key difference
        from SIMON (PairCorrectionNN), which receives only log(r_soft) and predicts
        a scalar correction — making SIMON rotationally invariant by construction.
        VectorNN must learn rotational symmetry from data, which is harder and may
        accumulate directional error over long chaotic rollouts.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),          # 3 outputs: Fx, Fy, Fz
        )
        # Input normalisation buffers (same pattern as PairCorrectionNN)
        self.register_buffer('input_mean', torch.zeros(5))
        self.register_buffer('input_std',  torch.ones(5))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8))


# ── Training data generator ───────────────────────────────────────────────────
def generate_vector_training_data(n=50000, eps=3e-4, G=1.0, seed=42):
    """
    Generate pairwise force vector training data analytically (no REBOUND needed).

    Sampling strategy mirrors train_pair_correction_new.py:
      - 60% of samples from transition zone  r ∈ [0.5ε, 50ε]
      - 40% of samples from far field        r ∈ [50ε, 10.0]

    For each sample we:
      1. Sample a scalar distance r from the same log-uniform distribution.
      2. Sample a random unit direction vector r̂ uniformly on S².
      3. Construct the relative position vector r_ij = r * r̂.
      4. Compute the analytic softened force vector:
             F = G * m_i * m_j / r_soft³  *  r_ij
         where r_soft = sqrt(r² + ε²).

    Features:  [rx, ry, rz, log(m_i), log(m_j)]    shape (n, 5)  float32
    Targets:   [Fx, Fy, Fz]                          shape (n, 3)  float32
    """
    rng = np.random.RandomState(seed)

    # ── Distance sampling (identical to current training script) ──────────────
    n_trans = int(0.6 * n)
    n_far   = n - n_trans
    log_r = np.concatenate([
        rng.uniform(np.log(0.5 * eps), np.log(50 * eps), n_trans),
        rng.uniform(np.log(50 * eps),  np.log(10.0),     n_far),
    ])
    rng.shuffle(log_r)
    r = np.exp(log_r)                                       # (n,)

    # ── Random unit direction vectors (uniform on S²) ─────────────────────────
    # Use Gaussian → normalise method, which gives perfect spherical uniformity.
    raw_dirs = rng.randn(n, 3)
    norms    = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    dirs     = raw_dirs / (norms + 1e-30)                   # (n, 3) unit vectors

    # ── Relative position vectors ─────────────────────────────────────────────
    r_ij = r[:, None] * dirs                                # (n, 3)  r_ij = r * r̂

    # ── Mass sampling (same range as current script) ─────────────────────────
    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)
    mi = np.exp(log_mi)                                     # (n,)
    mj = np.exp(log_mj)                                     # (n,)

    # ── Analytic softened force vector ────────────────────────────────────────
    r_soft = np.sqrt(r**2 + eps**2)                         # (n,)
    F_mag  = G * mi * mj / r_soft**3                        # (n,)  force magnitude
    F_vec  = F_mag[:, None] * r_ij                          # (n, 3) force vector

    # ── Assemble features ─────────────────────────────────────────────────────
    features = np.stack([
        r_ij[:, 0],   # rx
        r_ij[:, 1],   # ry
        r_ij[:, 2],   # rz
        log_mi,       # log mass i
        log_mj,       # log mass j
    ], axis=1).astype(np.float32)                           # (n, 5)
    targets = F_vec.astype(np.float32)                      # (n, 3)

    # ── Sanity report ─────────────────────────────────────────────────────────
    F_norms = np.linalg.norm(F_vec, axis=1)
    print(f"[train_vector] {n} samples generated")
    print(f"  r  range: [{r.min():.2e}, {r.max():.2e}]")
    print(f"  |F| range: [{F_norms.min():.2e}, {F_norms.max():.2e}]  "
          f"mean={F_norms.mean():.2e}  std={F_norms.std():.2e}")
    return features, targets


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    """
    Train VectorNN. Mirrors train_pair_correction_new.py structure exactly:
      - AdamW, weight_decay=1e-5
      - CosineAnnealingLR, eta_min = lr/100
      - 80/20 train/val split
      - Best val checkpoint saved
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalisation statistics (over training features)
    feat_mean = features.mean(0)
    feat_std  = features.std(0)

    # Train / val split
    N   = len(features)
    idx = np.random.permutation(N)
    nv  = int(0.2 * N)
    Xt  = torch.tensor(features[idx[nv:]], device=device)
    yt  = torch.tensor(targets[idx[nv:]],  device=device)
    Xv  = torch.tensor(features[idx[:nv]], device=device)
    yv  = torch.tensor(targets[idx[:nv]],  device=device)

    # Model
    model = VectorNN(hidden=hidden).to(device)
    model.input_mean = torch.tensor(feat_mean, device=device)
    model.input_std  = torch.tensor(feat_std,  device=device)

    # Optimiser + scheduler
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 100)
    loss_fn = nn.MSELoss()

    best_vl = 1e9
    best_sd = None
    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)                    # (N_train, 3)
        l    = loss_fn(pred, yt)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()

        tl_hist.append(l.item())
        vl_hist.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs}  train={l.item():.6f}  val={vl:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"[train_vector] {elapsed:.1f}s  |  best val MSE = {best_vl:.6f}")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, eps=3e-4, G=1.0, device="cpu"):
    """
    Verify that the trained Vector NN roughly reproduces the analytic force
    for a few test cases along the x-axis (r_ij = [r, 0, 0], equal masses).
    """
    model.eval()
    print(f"\n{'r':>10} | {'|F_true|':>10} | {'|F_pred|':>10} | {'err%':>8} | {'cos(angle)':>10}")
    for r in [1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 3.0]:
        r_soft = np.sqrt(r**2 + eps**2)
        F_true_mag = G * 1.0 * 1.0 / r_soft**3 * r   # mi=mj=1, direction=[1,0,0]
        F_true_vec = np.array([F_true_mag, 0.0, 0.0])
        inp = torch.tensor([[r, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            F_pred_vec = model(inp).cpu().numpy()[0]
        F_pred_mag = np.linalg.norm(F_pred_vec)
        err_pct    = abs(F_pred_mag - F_true_mag) / (F_true_mag + 1e-30) * 100
        cosine     = np.dot(F_pred_vec, F_true_vec) / (F_pred_mag * F_true_mag + 1e-30)
        print(f"  {r:10.1e} | {F_true_mag:10.4f} | {F_pred_mag:10.4f} | {err_pct:7.2f}% | {cosine:10.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train Vector NN force baseline")
    ap.add_argument("--hidden",  type=int,   default=32)
    ap.add_argument("--epochs",  type=int,   default=5000)
    ap.add_argument("--n",       type=int,   default=50000, help="Training samples")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output",  default="vector_nn.pt")
    args = ap.parse_args()

    print(f"[train_vector] hidden={args.hidden}, epochs={args.epochs}, "
          f"n={args.n}, device={args.device}")

    # Generate data
    features, targets = generate_vector_training_data(n=args.n)

    # Train
    model, tl, vl = train_model(
        features, targets,
        hidden=args.hidden,
        epochs=args.epochs,
        device=args.device,
    )

    # Sanity check
    sanity_check(model, device=args.device)

    # Save weights
    torch.save(model.cpu().state_dict(), args.output)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n[train_vector] Saved {args.output}  ({size_kb:.1f} KB)")
    print(f"[train_vector] Params: {sum(p.numel() for p in model.parameters())}")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(tl, label="train MSE")
    plt.plot(vl, label="val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (force vector)")
    plt.title("Vector NN Training Log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_log_vector.png", dpi=150)
    print("[train_vector] Saved training_log_vector.png")
    print("[train_vector] Done.")


if __name__ == "__main__":
    main()
