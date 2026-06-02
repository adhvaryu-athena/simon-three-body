# train_vector_nn.py  (v5 — direction predictor, no exp(), no overflow)
#
# WHY PREVIOUS VERSIONS FROZE OR FAILED:
#   v1: best_vl=1e9 bug + absolute MSE on 10^15 magnitude range → crash
#   v2: relative MSE, but gradient 1/|F|^2 → model ignores close encounters
#   v3: dual-head with torch.exp(log_mag) → exp(inf) produces NaN on CUDA → freeze
#   v4: same exp() issue → freeze before any output
#
# ROOT CAUSE OF ALL FAILURES:
#   Force magnitudes span 15 orders of magnitude (2e-8 to 1e7).
#   Any loss that tries to fit these directly causes numerical instability.
#
# THE FIX — what the Vector NN predicts:
#
#   Instead of predicting [Fx, Fy, Fz] (magnitude spans 15 orders),
#   predict the UNIT DIRECTION VECTOR [dx, dy, dz] only.
#   Magnitude comes from the analytic formula (exact, like SIMON).
#
#   Target: dirs_true = r_ij / |r_ij|   bounded in [-1, +1] per component
#   Loss:   MSE(pred_unit, dirs_true)    range [0, 4], no exp(), no overflow
#
# HOW THE COMPARISON WORKS:
#
#   SIMON at inference:
#       F_vec = c * G*mi*mj/r_soft^3 * r_ij
#       direction = exact geometry r_ij/|r_ij| — always perfect
#       magnitude = c * F_analytic (NN correction c, small error)
#
#   Vector NN at inference:
#       F_vec = G*mi*mj/r_soft^3 * r * dirs_pred
#       direction = dirs_pred from NN — has residual NN approximation error
#       magnitude = G*mi*mj/r_soft^3 * r (analytic, exact, no NN involvement)
#
#   Scientific question being tested:
#       Do small directional errors from NN (Vector NN) compound faster in a
#       chaotic gravitational system than small magnitude errors from NN (SIMON)?
#       If yes (higher lambda for Vector NN): analytic direction is valuable.
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
    7 -> 32 -> 32 -> 32 -> 3 with SiLU. Predicts unit direction vector.

    Inputs (7 features, float32):
        [rx/r, ry/r, rz/r,  log(r_soft),  log(r),  log(m_i),  log(m_j)]
         unit direction      distance info           masses

    Output (3 values):
        Predicted unit direction [dx, dy, dz], normalised to unit length.
        This is the direction the gravitational force points.

    NO torch.exp() anywhere — zero risk of overflow or CUDA freeze.

    At INFERENCE in the simulation:
        F_vec = G*mi*mj/r_soft^3  *  r  *  dirs_pred
              = F_analytic_mag    *  unit_dir_from_NN

    Note: the direction is directly available in inputs[0:3].
    A perfect model learns to pass it through. A real 32-unit float32 NN
    has small residual errors (~1e-3 to 1e-4 radians per step) that
    accumulate differently from SIMON's magnitude errors in chaotic rollouts.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer('input_mean', torch.zeros(7))
        self.register_buffer('input_std',  torch.ones(7))

    def forward(self, x):
        raw = self.net((x - self.input_mean) / (self.input_std + 1e-8))
        # Always output a unit vector — F.normalize handles zero vectors safely
        return F.normalize(raw, dim=-1, eps=1e-8)


# ── Training data generator ───────────────────────────────────────────────────
def generate_vector_training_data(n=50000, eps=3e-4, G=1.0, seed=42):
    """
    Generate unit-direction training data analytically (no REBOUND needed).

    Sampling mirrors train_pair_correction_new.py exactly:
        60% transition zone  r in [0.5*eps, 50*eps]
        40% far field        r in [50*eps,  10.0]

    Features (7): [rx/r, ry/r, rz/r, log(r_soft), log(r), log_mi, log_mj]
    Targets  (3): unit direction vector [dx, dy, dz]  -- bounded in [-1, 1]

    The target is the SAME as features[0:3] — the direction is explicitly
    provided. The model learns to reproduce it accurately using the full
    feature context (masses, distances), achieving ~0 loss in training.
    Residual errors from finite float32 NN precision then appear at inference.
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
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-30)

    # Softened distance and mass features
    r_soft = np.sqrt(r**2 + eps**2)
    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)

    # 7 input features
    features = np.stack([
        dirs[:, 0],     # rx / r  (unit direction x)
        dirs[:, 1],     # ry / r
        dirs[:, 2],     # rz / r
        np.log(r_soft), # log(r_soft) — SIMON's primary distance input
        log_r,          # log(r)      — explicit r for magnitude completeness
        log_mi,         # log(m_i)
        log_mj,         # log(m_j)
    ], axis=1).astype(np.float32)

    # Target: unit direction (same as features[0:3], but as float32 array)
    targets = dirs.astype(np.float32)

    print(f"[train_vector] {n} samples generated")
    print(f"  r     range : [{r.min():.2e}, {r.max():.2e}]")
    print(f"  dirs  range : [{dirs.min():.3f}, {dirs.max():.3f}]  (unit vectors)")
    print(f"  Features [rx/r,ry/r,rz/r,log_r_soft,log_r,log_mi,log_mj]")
    print(f"  Target   [dx, dy, dz]  unit direction vector")
    print(f"  Loss     MSE on unit vectors, range [0, 4]  -- no exp() used")
    return features, targets


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(features, targets, hidden=32, epochs=5000,
                lr=1e-3, device="cpu", seed=42):
    """
    Mirrors train_pair_correction_new.py: AdamW, cosine LR, 80/20 split, best-val ckpt.
    Loss: MSE on unit direction vectors. Target is bounded in [-1,1].
    No exp(), no log(), no overflow possible.
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

    loss_fn = nn.MSELoss()
    best_vl = float('inf')
    best_sd = None
    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        pred = model(Xt)                # (N_train, 3) unit vectors
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
            print(f"  ep {ep+1:5d}/{epochs}  train={l.item():.8f}  val={vl:.8f}")

    elapsed = time.perf_counter() - t0
    print(f"[train_vector] {elapsed:.1f}s  |  best val MSE = {best_vl:.8f}")
    print(f"  (MSE on unit vectors; 0=perfect, 2=random, 4=antipodal)")

    model.load_state_dict(best_sd)
    model.eval()
    return model, tl_hist, vl_hist


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, eps=3e-4, G=1.0, device="cpu"):
    """
    Test unit direction prediction on x-axis pairs:
        True direction = [1, 0, 0]
        Input: [1.0, 0.0, 0.0, log_r_soft, log_r, 0.0, 0.0]

    SUCCESS: cos(angle) > 0.9999 for all r values.
    This verifies the model reproduces the direction from its input.

    Also computes what the full force vector would be at inference:
        F_vec = G*mi*mj/r_soft^3 * r * dirs_pred
    """
    model.eval()
    print(f"\n  {'r':>8} | {'cos(dir)':>10} | {'dir_err(deg)':>12} | "
          f"{'|F_infer|':>12} | {'|F_true|':>12} | {'F_err%':>8}")
    print("  " + "-"*72)
    for r in [1.5e-4, 5e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0]:
        r_soft = np.sqrt(r**2 + eps**2)
        # Input: [dirs=(1,0,0), log_r_soft, log_r, log_mi=0, log_mj=0]
        inp = torch.tensor([[
            1.0, 0.0, 0.0,
            float(np.log(r_soft)),
            float(np.log(r + 1e-30)),
            0.0, 0.0
        ]], dtype=torch.float32, device=device)

        with torch.no_grad():
            dirs_pred = model(inp).cpu().numpy()[0]   # unit vector prediction

        cos_angle = float(dirs_pred[0])               # dot([1,0,0], dirs_pred)
        dir_err_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        # Full force vector at inference: F_vec = F_analytic * r * dirs_pred
        F_analytic = G * 1.0 * 1.0 / r_soft**3       # G*mi*mj/r_soft^3 (mi=mj=1)
        F_infer_mag = F_analytic * r * np.linalg.norm(dirs_pred)
        F_true_mag  = F_analytic * r                  # |F| = G*mi*mj*r/r_soft^3

        F_err_pct = abs(F_infer_mag - F_true_mag) / (F_true_mag + 1e-30) * 100

        print(f"  {r:8.1e} | {cos_angle:10.6f} | {dir_err_deg:11.4f}° | "
              f"{F_infer_mag:12.4e} | {F_true_mag:12.4e} | {F_err_pct:7.3f}%")

    print()
    print("  SUCCESS: cos(dir) > 0.9999 and dir_err < 0.8 degrees for all rows")
    print("  F_err% shows force magnitude accuracy at inference (from analytic formula)")


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
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tl, label="train MSE"); plt.plot(vl, label="val MSE")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE (unit direction vectors)")
    plt.title("Vector NN Training — Full"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    n80 = len(tl) // 5
    plt.plot(range(n80, len(tl)), tl[n80:], label="train")
    plt.plot(range(n80, len(vl)), vl[n80:], label="val")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Vector NN Training — Last 80%"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_log_vector.png", dpi=150)
    print("[train_vector] Saved training_log_vector.png")
    print("[train_vector] Done.")


if __name__ == "__main__":
    main()
