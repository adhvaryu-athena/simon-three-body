# train_pair_correction.py
import os, time, argparse
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

class PairCorrectionNN(nn.Module):
    """3->H->H->H->1 with SiLU. Hidden=32: the mapping log(r)->log(c) is
    smooth and 1D (masses cancel). 32 units is plenty for a monotonic function."""
    def __init__(self, hidden=32, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std', torch.ones(3))
    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)

def generate_training_data(n=50000, eps=3e-4, seed=42):
    rng = np.random.RandomState(seed)
    n_trans = int(0.6 * n)
    n_far = n - n_trans
    log_r = np.concatenate([
        rng.uniform(np.log(0.5*eps), np.log(50*eps), n_trans),
        rng.uniform(np.log(50*eps), np.log(10.0), n_far)
    ])
    rng.shuffle(log_r)
    r = np.exp(log_r)
    log_mi = rng.uniform(np.log(0.001), np.log(2.0), n)
    log_mj = rng.uniform(np.log(0.001), np.log(2.0), n)
    r_soft = np.sqrt(r*r + eps*eps)
    log_c = 3.0 * np.log(r_soft / r)
    features = np.stack([np.log(r_soft), log_mi, log_mj], axis=1).astype(np.float32)
    targets = log_c.astype(np.float32)
    print(f"[train] {n} samples | log(c): mean={targets.mean():.4f} std={targets.std():.4f} "
          f"min={targets.min():.4f} max={targets.max():.4f}")
    return features, targets

def train_model(features, targets, hidden=32, epochs=5000, lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    feat_mean = features.mean(0); feat_std = features.std(0)
    N = len(features); idx = np.random.permutation(N); nv = int(0.2*N)
    Xt = torch.tensor(features[idx[nv:]], device=device)
    yt = torch.tensor(targets[idx[nv:]], device=device)
    Xv = torch.tensor(features[idx[:nv]], device=device)
    yv = torch.tensor(targets[idx[:nv]], device=device)
    model = PairCorrectionNN(hidden=hidden).to(device)
    model.input_mean = torch.tensor(feat_mean, device=device)
    model.input_std = torch.tensor(feat_std, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr/100)
    loss_fn = nn.MSELoss(); best_vl = 1e9; best_sd = None
    tl_hist, vl_hist = [], []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train(); l = loss_fn(model(Xt), yt)
        opt.zero_grad(); l.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv), yv).item()
        tl_hist.append(l.item()); vl_hist.append(vl)
        if vl < best_vl: best_vl = vl; best_sd = {k:v.clone() for k,v in model.state_dict().items()}
        if (ep+1) % 1000 == 0 or ep == 0:
            print(f"  ep {ep+1:5d}/{epochs} train={l.item():.6f} val={vl:.6f}")
    print(f"[train] {time.perf_counter()-t0:.1f}s | best val={best_vl:.6f}")
    model.load_state_dict(best_sd); model.eval()
    return model, tl_hist, vl_hist

def sanity_check(model, eps=3e-4, device="cpu"):
    model.eval()
    print(f"\n{'r':>10} | {'c_true':>8} | {'c_pred':>8} | {'err':>8}")
    for r in [1e-5,1e-4,3e-4,1e-3,0.01,0.1,1.0,5.0]:
        rs = np.sqrt(r**2+eps**2); ct = (rs/r)**3
        with torch.no_grad():
            cp = torch.exp(model(torch.tensor([[np.log(rs),0.,0.]], dtype=torch.float32, device=device))).item()
        print(f"  {r:10.1e} | {ct:8.3f} | {cp:8.3f} | {abs(cp-ct)/max(ct,1e-12):8.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="pair_correction_nn.pt")
    args = ap.parse_args()
    print(f"Hidden={args.hidden}, Epochs={args.epochs}, Device={args.device}")
    feat, targ = generate_training_data()
    model, tl, vl = train_model(feat, targ, hidden=args.hidden, epochs=args.epochs, device=args.device)
    sanity_check(model, device=args.device)
    torch.save(model.cpu().state_dict(), args.output)
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    plt.figure(figsize=(8,4)); plt.plot(tl,label="train"); plt.plot(vl,label="val")
    plt.yscale("log"); plt.legend(); plt.grid(True,alpha=0.3)
    plt.savefig("training_log.png",dpi=150); print("Saved training_log.png")

if __name__ == "__main__":
    main()
