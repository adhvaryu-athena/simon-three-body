# train_vector_nn.py  (v4 — dual-head architecture, 7 inputs)
#
# WHAT CHANGED FROM v3 AND WHY:
#
#   v3 problem 1 — single output layer trying to learn magnitude AND direction
#   simultaneously via the same weights. The gradient that improves magnitude
#   pushes direction wrong and vice versa. Model gets stuck.
#   Fix: dual-head — separate branches for log-magnitude (scalar) and
#   unit-direction (3-vector). Independent weights, no interference.
#
#   v3 problem 2 — model had to infer log(r) from log(r_soft) nonlinearly.
#   For close encounters (r << eps), log(r_soft) ≈ constant so the model
#   couldn't distinguish different close-encounter distances.
#   Fix: include log(r) as an explicit 7th input feature. Then log(|F|) =
#   log_mi + log_mj + log_r - 3*log_r_soft is LINEAR in the inputs — trivially
#   learnable. log(r) is already computed in the eval loop from r=sqrt(r2).
#
# ARCHITECTURE:
#   7 inputs: [rx/r, ry/r, rz/r, log(r_soft), log(r), log_mi, log_mj]
#   Shared trunk: 7→32→32→32 (SiLU)
#   Magnitude head: 32→1 → predicts log(|F_vec|)
#   Direction head: 32→3 → F.normalize'd to unit vector
#   Output: F_vec = exp(log_mag) * dir_unit     shape (3,)
#
# COMPARISON WITH SIMON:
#   SIMON  inputs:  [log(r_soft), log_mi, log_mj]           3 features, no direction
#   SIMON  output:  scalar log(c) correction factor          1 value
#   SIMON  direction: exact geometry r_ij/|r_ij|            always perfect
#
#   VectorNN inputs: [rx/r,ry/r,rz/r, log(r_soft), log(r), log_mi, log_mj]  7 features
#   VectorNN output: [Fx, Fy, Fz] directly                  3 values
#   VectorNN direction: exp(log_mag)*dir_unit from NN       may have NN errors
#
#   Even if VectorNN trains to near-zero error, at INFERENCE TIME the direction
#   comes from NN weights (floating-point approximation), while SIMON's direction
#   comes from exact current positions. In chaotic systems, even tiny directional
#   errors accumulate. This is what the simulation comparison will test.
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
    Dual-head architecture: separate magnitude and direction branches.

    Inputs (7 features, float32):
        [rx/r, ry/r, rz/r,  log(r_soft),  log(r),  log(m_i),  log(m_j)]
         unit direction      log-distance   log-r    log masses

    Internal:
        Trunk  : 7→32→32→32 with SiLU (3 hidden layers, shared representation)
        Mag    : trunk→32→1 : predicts log(|F_vec|)
        Dir    : trunk→32→3 : predicts direction, F.normalized to unit vector

    Output (3 values):
        F_vec = exp(log_mag) * dir_unit   [Fx, Fy, Fz]

    Why dual-head works:
        Single output layer gradient conflicts: improving magnitude pushes direction
        wrong and vice versa. Separate heads have independent weights so each
        objective gets clean gradients.

    Why 7 inputs:
        log(|F_vec|) = log(G) + log_mi + log_mj + log_r - 3*log_r_soft
        This is LINEAR in {log_r, log_r_soft, log_mi, log_mj}. With log(r) as
        an explicit input, the magnitude head can learn this exactly in a few
        hundred epochs. Without it, the model must learn the nonlinear relationship
        between log(r_soft) and log(r), which is ambiguous for r < eps.
    """
    def __init__(self, hidden=32):
        super().__init__()
        # Shared trunk: 3 SiLU layers
        self.trunk = nn.Sequential(
            nn.Linear(7, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # Magnitude head: scalar log(|F|)
        self.mag_head = nn.Linear(hidden, 1)
        # Direction head: 3D direction (normalized in forward pass)
        self.dir_head = nn.Linear(hidden, 3)

        self.register_buffer('input_mean', torch.zeros(7))
        self.register_buffer('input_std',  torch.ones(7))

    def forward(self, x):
        h        = self.trunk((x - self.input_mean) / (self.input_std + 1e-8))
        log_mag  = self.mag_head(h).squeeze(-1)              # (B,) log force magnitude
        dir_raw  = self.dir_head(h)                          # (B, 3) unnormalised
        dir_unit = F.normalize(dir_raw, dim=1, eps=1e-8)     # (B, 3) unit vector
        # Combine: force vector = magnitude * direction
        F_vec    = torch.exp(log_mag).unsqueeze(1) * dir_unit  # (B, 3)
        return F_vec


# ── Loss ──────────────────────────────────────────────────────────────────────
def dual_head_loss(pred, true):
    """
    Two-component loss, one per head.

    Magnitude (log-space MSE):
        MSE( log|F_pred|, log|F_true| )
        Same idea as SIMON's MSE on log(c). Scale-invariant across 15 orders.
        With log(r) in the input, true log|F| is LINEAR in inputs → should
        converge to near-zero just like SIMON.

    Direction (unit-vector MSE):
        MSE( dir_pred, dir_true )   [both normalised to unit vectors]
        Range: 0 (perfect) to 4 (antipodal). Expected ~2 at random init.
        With direction in the input features, model should learn dir≈input quickly.

    Both components are dimensionless and comparable in scale.
    Target: magnitude loss << 0.01, direction loss << 0.001 at convergence.
    """
    pred_mag  = torch.norm(pred, dim=1).clamp(min=1e-30)
    true_mag  = torch.norm(true, dim=1).clamp(min=1e-30)

    # Log-magnitude MSE
    log_mag_loss = F.mse_loss(torch.log(pred_mag), torch.log(true_mag))

    # Unit-vector MSE
    pred_unit = pred / pred_mag.unsqueeze(1)
    true_unit = true / true_mag.unsqueeze(1)
    dir_loss  = F.mse_loss(pred_unit, true_unit)

    return log_mag_loss + dir_loss


# ── Training data generator ───────────────────────────────────────────────────
def generate_vector_training_data(n=50000, eps=3e-4, G=1.0, seed=42):
    """
    Generate pairwise force vector training data analytically (no REBOUND needed).

    Sampling mirrors train_pair_correction_new.py exactly:
        60% from transition zone  r in [0.5*eps, 50*eps]
        40% from far field        r in [50*eps,  10.0]

    Features (7): [rx/r, ry/r, rz/r, log(r_soft), log(r), log_mi, log_mj]
    Targets  (3): [Fx, Fy, Fz]  where F_vec = G*mi*mj/r_soft^3 * r_ij

    KEY: with log(r) included as feature 5:
        log(|F_vec|) = log_mi + log_mj + log(r) - 3*log(r_soft)  [exact, linear]
    The magnitude head only needs to learn a linear combination — very easy.
    """
    rng = np.random.RandomState(seed)

    # Distance sampling — same as train_pair_correction_new.py
    n_trans = int(0.6 * n)
    n_far   = n - n_trans
    log_r   = np.concatenate([
        rng.uniform(np.log(0.5 * eps), np.log(50 * eps), n_trans),
        rng.uniform(np.log(50 * eps),  np.log(10.0),     n_far),
    ])
    rng.shuffle(log_r)
    r = np.exp(log_r)

    # Random unit directions uniformly on S^2
    raw  = rng.randn(n, 3)
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-30)  # (n, 3)

    # Relative position vector and softened distance
    r_ij   = r[:, None] * dirs            # (n, 3)
    r_soft = np.sqrt(r**2 + eps**2)       # (n,)

    # Mass sampling — same range as train_pair_correction_new.py
    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)
    mi     = np.exp(log_mi)
    mj     = np.exp(log_mj)

    # Analytic softened force vector:  F = G*mi*mj/r_soft^3 * r_ij
    F_mag = G * mi * mj / r_soft**3        # (n,) scalar magnitude
    F_vec = F_mag[:, None] * r_ij          # (n, 3) force vector

    # Check: log(|F_vec|) = log_mi + log_mj + log(r) - 3*log(r_soft)  [linear in features!]
    log_F_expected = log_mi + log_mj + log_r - 3 * np.log(r_soft)
    log_F_actual   = np.log(np.linalg.norm(F_vec, axis=1) + 1e-30)
    max_log_err    = np.max(np.abs(log_F_expected - log_F_actual))

    # 7 input features
    features = np.stack([
        dirs[:, 0],        # rx / r  (unit x-component of direction)
        dirs[:, 1],        # ry / r
        dirs[:, 2],        # rz / r
        np.log(r_soft),    # log(r_soft) — same as SIMON's primary input
        log_r,             # log(r)     — NEW: makes log|F| linear in inputs
        log_mi,            # log(m_i)   — same as SIMON
        log_mj,            # log(m_j)
    ], axis=1).astype(np.float32)

    targets = F_vec.astype(np.float32)

    F_norms = np.linalg.norm(F_vec, axis=1)
    print(f"[train_vector] {n} samples generated")
    print(f"  r      range : [{r.min():.2e}, {r.max():.2e}]")
    print(f"  |F|    range : [{F_norms.min():.2e}, {F_norms.max():.2e}]")
    print(f"  log|F| linear check: max error = {max_log_err:.2e}  (should be ~1e-6)")
    print(f"  Features: [rx/r, ry/r, rz/r, log_r_soft, log_r, log_mi, log_mj]")
    return features, targets


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    """
    Mirrors train_pair_correction_new.py: AdamW, cosine LR, 80/20 split, best-val ckpt.
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

    best_vl = float('inf')
    best_sd = None
    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)
        l    = dual_head_loss(pred, yt)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vl = dual_head_loss(model(Xv), yv).item()

        tl_hist.append(l.item())
        vl_hist.append(vl)

        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs}  train={l.item():.6f}  val={vl:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"[train_vector] {elapsed:.1f}s  |  best val loss = {best_vl:.6f}")
    print(f"  loss = log-mag MSE + unit-dir MSE")
    print(f"  target: << 0.01 for magnitude, << 0.001 for direction")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, eps=3e-4, G=1.0, device="cpu"):
    """
    Test on x-axis pairs: direction=(1,0,0), mi=mj=1 (log_mi=log_mj=0).

    Input: [rx/r=1, ry/r=0, rz/r=0, log(r_soft), log(r), 0, 0]

    SUCCESS criteria:
        err%      < 2%    across all r   -- magnitude learned (linear target)
        cos(angle) > 0.9999             -- direction learned (direct input)

    If BOTH hold: Vector NN force is accurate. Comparison with SIMON then
    comes down to long-horizon simulation stability, not training accuracy.
    """
    model.eval()
    print(f"\n  {'r':>8} | {'|F_true|':>12} | {'|F_pred|':>12} | {'err%':>7} | {'cos(angle)':>10}")
    print("  " + "-"*64)
    for r in [1.5e-4, 5e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0]:
        r_soft     = np.sqrt(r**2 + eps**2)
        F_true_mag = G * 1.0 * 1.0 / r_soft**3 * r        # |F_vec| for mi=mj=1
        F_true_vec = np.array([F_true_mag, 0.0, 0.0])

        # 7 features: [1,0,0, log_r_soft, log_r, 0, 0]
        inp = torch.tensor([[
            1.0, 0.0, 0.0,
            float(np.log(r_soft)),
            float(np.log(r)),
            0.0, 0.0
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            F_pred_vec = model(inp).cpu().numpy()[0]

        F_pred_mag = np.linalg.norm(F_pred_vec)
        err_pct    = abs(F_pred_mag - F_true_mag) / (abs(F_true_mag) + 1e-30) * 100
        cos_angle  = np.dot(F_pred_vec, F_true_vec) / (
                     F_pred_mag * np.linalg.norm(F_true_vec) + 1e-30)
        print(f"  {r:8.1e} | {F_true_mag:12.4e} | {F_pred_mag:12.4e}"
              f" | {err_pct:6.2f}% | {cos_angle:10.6f}")
    print()
    print("  SUCCESS: err% < 2% AND cos(angle) > 0.9999 for all rows")


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tl, label="train"); ax1.plot(vl, label="val")
    ax1.set_yscale("log"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss (log-mag + dir)"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Zoom into last 80% of training to see convergence
    start = len(tl) // 5
    ax2.plot(range(start, len(tl)), tl[start:], label="train")
    ax2.plot(range(start, len(vl)), vl[start:], label="val")
    ax2.set_yscale("log"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss (last 80% of training)"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_log_vector.png", dpi=150)
    print("[train_vector] Saved training_log_vector.png")
    print("[train_vector] Done.")


if __name__ == "__main__":
    main()
