# train_vector_nn.py  (v3 — proper input decomposition + log-magnitude/direction loss)
# Trains a Vector NN baseline that predicts the full pairwise force vector [Fx, Fy, Fz].
# Companion to train_pair_correction_new.py.
#
# DESIGN CHOICES (explained in comments throughout):
#
#   Inputs  (6 features): [rx/r, ry/r, rz/r, log(r_soft), log(m_i), log(m_j)]
#   Outputs (3 values):   [Fx, Fy, Fz]
#   Loss:   log-magnitude MSE  +  direction cosine loss
#
# WHY these choices are correct (and different from the previous broken version):
#
#   Previous version used raw [rx, ry, rz] as inputs.
#   Problem: a close encounter at r=1.5e-4 AU gives rx=±0.00015, which after
#   normalisation (dividing by std_rx ~ a few AU) becomes ~0.00005, indistinguishable
#   from zero. The NN could not tell different close-encounter directions apart and
#   learned to output a constant vector instead.
#
#   This version uses unit direction [rx/r, ry/r, rz/r] + log distance [log(r_soft)].
#   Every input is now well-conditioned:
#     - Unit direction components are always in [-1, +1], mean=0, std=1/sqrt(3)
#     - log(r_soft) is the SAME as SIMON's primary input -- NN sees the same distance info
#     - log masses are the same as SIMON's other inputs
#   This gives the Vector NN the BEST POSSIBLE inputs to compete against SIMON.
#   If SIMON still produces lower divergence rates despite the Vector NN having well-
#   conditioned inputs, the scientific conclusion is stronger: analytic direction
#   enforcement matters for chaotic stability even when learned direction has good data.
#
# Run:  python train_vector_nn.py
# Out:  vector_nn.pt  +  training_log_vector.png

import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ── Model ─────────────────────────────────────────────────────────────────────
class VectorNN(nn.Module):
    """
    6 -> 32 -> 32 -> 32 -> 3  with SiLU activations.

    Inputs  (6 features, float32):
        [rx/r,  ry/r,  rz/r,  log(r_soft),  log(m_i),  log(m_j)]
         unit direction vector  log distance   log masses

    Output  (3 values):
        [Fx, Fy, Fz] -- pairwise gravitational force vector on body i from body j.

    Comparison with SIMON (PairCorrectionNN, 3->32->32->32->1):
        SIMON inputs:  [log(r_soft), log(m_i), log(m_j)]         -- 3 features, no direction
        SIMON output:  scalar log(c) correction factor             -- 1 value
        SIMON direction: computed from exact geometry r_ij/|r_ij| -- always perfect

        VectorNN inputs: [rx/r, ry/r, rz/r, log(r_soft), log(m_i), log(m_j)] -- 6 features
        VectorNN output: [Fx, Fy, Fz] directly                                -- 3 values
        VectorNN direction: F_pred / |F_pred| from NN -- may contain small errors

    The extra 3 inputs (unit direction) give VectorNN full information about the
    geometry, so training can converge. The key difference at SIMULATION TIME is
    that VectorNN direction comes from the NN output, while SIMON direction comes
    from the exact current positions -- making SIMON's direction always geometrically
    correct, even if the NN magnitude prediction has small errors.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),      nn.SiLU(),   # 6 inputs instead of 5
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),                    # Fx, Fy, Fz
        )
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std',  torch.ones(6))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8))


# ── Loss ──────────────────────────────────────────────────────────────────────
def vector_nn_loss(pred, true):
    """
    Two-component loss for force vector prediction.

    Component 1 -- Log-magnitude MSE:
        MSE( log|F_pred|, log|F_true| )
        Range: ~0 (perfect) to ~large (order-of-magnitude error).
        This is analogous to SIMON's MSE on log(c) -- both use log-space to handle
        the wide range of force magnitudes.

    Component 2 -- Direction cosine loss:
        mean( 1 - cosine_similarity(F_pred, F_true) )
        Range: 0 (perfect alignment) to 2 (opposite direction).
        Penalises directional error independently of magnitude error.

    Total loss ~ 0 for a perfect model, ~ 5-15 at random initialisation.
    Both components are dimensionless and on a similar scale, so no weighting needed.
    """
    # Magnitudes
    pred_mag = torch.norm(pred, dim=1).clamp(min=1e-30)
    true_mag = torch.norm(true, dim=1).clamp(min=1e-30)

    # Log-magnitude MSE (same idea as SIMON's log(c) MSE)
    log_mag_loss = F.mse_loss(torch.log(pred_mag), torch.log(true_mag))

    # Direction cosine loss
    dir_loss = (1.0 - F.cosine_similarity(pred, true, dim=1)).mean()

    return log_mag_loss + dir_loss


# ── Training data generator ───────────────────────────────────────────────────
def generate_vector_training_data(n=50000, eps=3e-4, G=1.0, seed=42):
    """
    Generate pairwise force vector training data analytically (no REBOUND needed).

    Sampling mirrors train_pair_correction_new.py exactly:
      60% transition zone  r in [0.5*eps, 50*eps]  where correction c != 1
      40% far field        r in [50*eps,  10.0]

    Features: [rx/r, ry/r, rz/r, log(r_soft), log(m_i), log(m_j)]   shape (n, 6) float32
    Targets:  [Fx, Fy, Fz]                                            shape (n, 3) float32

    All feature components are well-conditioned:
      rx/r, ry/r, rz/r  in [-1, +1]  -- unit direction, mean=0, no normalisation needed
      log(r_soft)        in [-8, +2]  -- same as SIMON's first input
      log(m_i), log(m_j) in [-7, +1] -- same as SIMON's other inputs
    """
    rng = np.random.RandomState(seed)

    # Distance sampling -- identical to train_pair_correction_new.py
    n_trans = int(0.6 * n)
    n_far   = n - n_trans
    log_r = np.concatenate([
        rng.uniform(np.log(0.5 * eps), np.log(50 * eps), n_trans),
        rng.uniform(np.log(50 * eps),  np.log(10.0),     n_far),
    ])
    rng.shuffle(log_r)
    r = np.exp(log_r)                                       # (n,) scalar distances

    # Random unit directions uniformly on S^2 (Gaussian -> normalise)
    raw  = rng.randn(n, 3)
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-30)   # (n, 3)

    # Relative position vectors
    r_ij   = r[:, None] * dirs                              # (n, 3)  displacement vector
    r_soft = np.sqrt(r**2 + eps**2)                         # (n,)    softened distance

    # Mass sampling -- same range as train_pair_correction_new.py
    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)
    mi = np.exp(log_mi)
    mj = np.exp(log_mj)

    # Analytic softened force vector  F = G*mi*mj/r_soft^3 * r_ij
    F_mag = G * mi * mj / r_soft**3                         # (n,)
    F_vec = F_mag[:, None] * r_ij                           # (n, 3)

    # Features: unit direction + log distance + log masses
    features = np.stack([
        dirs[:, 0],            # rx / r  (unit direction x-component)
        dirs[:, 1],            # ry / r
        dirs[:, 2],            # rz / r
        np.log(r_soft),        # log(r_soft) -- same as SIMON's primary input
        log_mi,                # log(m_i)
        log_mj,                # log(m_j)
    ], axis=1).astype(np.float32)                           # (n, 6)

    targets = F_vec.astype(np.float32)                      # (n, 3)

    # Diagnostics
    F_norms = np.linalg.norm(F_vec, axis=1)
    print(f"[train_vector] {n} samples generated")
    print(f"  r     range : [{r.min():.2e}, {r.max():.2e}]")
    print(f"  |F|   range : [{F_norms.min():.2e}, {F_norms.max():.2e}]"
          f"  mean={F_norms.mean():.2e}")
    print(f"  Inputs: [rx/r, ry/r, rz/r, log(r_soft), log_mi, log_mj]  -- 6 features")
    print(f"  Loss  : log-magnitude MSE + direction cosine loss")
    return features, targets


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    """
    Mirrors train_pair_correction_new.py structure:
      AdamW, weight_decay=1e-5, CosineAnnealingLR, 80/20 split, best-val checkpoint.
    """
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

    best_vl = float('inf')          # must be inf, not 1e9
    best_sd = None

    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)
        l    = vector_nn_loss(pred, yt)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vl = vector_nn_loss(model(Xv), yv).item()

        tl_hist.append(l.item())
        vl_hist.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs}  train={l.item():.6f}  val={vl:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"[train_vector] {elapsed:.1f}s  |  best val loss = {best_vl:.6f}")
    print(f"  (loss = log-mag MSE + dir-cosine loss; ~0 is perfect, ~10 is random)")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, eps=3e-4, G=1.0, device="cpu"):
    """
    Test on x-axis pairs: direction=(1,0,0), mi=mj=1, log_mi=log_mj=0.
    Input: [rx/r=1, ry/r=0, rz/r=0, log(r_soft), log(1), log(1)]
           = [1, 0, 0, log(sqrt(r^2+eps^2)), 0, 0]

    Success criteria:
      err%      < 5%    across all r   -- magnitude learned
      cos(angle) > 0.999              -- direction learned (should be near-perfect
                                         since direction is directly in the input)
    """
    model.eval()
    print(f"\n  {'r':>8} | {'|F_true|':>12} | {'|F_pred|':>12} | {'err%':>7} | {'cos(angle)':>10}")
    print("  " + "-"*62)
    for r in [1.5e-4, 5e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0]:
        r_soft     = np.sqrt(r**2 + eps**2)
        F_true_mag = G * 1.0 * 1.0 / r_soft**3 * r
        F_true_vec = np.array([F_true_mag, 0.0, 0.0])
        # Input: [rx/r=1, ry/r=0, rz/r=0, log(r_soft), log_mi=0, log_mj=0]
        inp = torch.tensor(
            [[1.0, 0.0, 0.0, float(np.log(r_soft)), 0.0, 0.0]],
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            F_pred_vec = model(inp).cpu().numpy()[0]
        F_pred_mag = np.linalg.norm(F_pred_vec)
        err_pct    = abs(F_pred_mag - F_true_mag) / (abs(F_true_mag) + 1e-30) * 100
        cos_angle  = np.dot(F_pred_vec, F_true_vec) / (
                     F_pred_mag * np.linalg.norm(F_true_vec) + 1e-30)
        print(f"  {r:8.1e} | {F_true_mag:12.4e} | {F_pred_mag:12.4e}"
              f" | {err_pct:6.2f}% | {cos_angle:10.6f}")
    print()
    print("  Target: err% < 5% and cos(angle) > 0.999 across all rows")


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
    size_kb  = os.path.getsize(args.output) / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_vector] Saved {args.output}  ({size_kb:.1f} KB)  params={n_params}")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(tl, label="train loss")
    plt.plot(vl, label="val loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("log-mag MSE + direction loss")
    plt.title("Vector NN Training Log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_log_vector.png", dpi=150)
    print("[train_vector] Saved training_log_vector.png")
    print("[train_vector] Done.")


if __name__ == "__main__":
    main()
