"""
Stage 2: train both correctors on GPU. Save numpy weights for fast deploy + curves.
 A1: features(7) -> log(c)            (MLP 7-64-64-1)
 A2: features(7) -> normalized 18-d residual  (MLP 7-128-128-18)
Random state-level val split for early-stopping; the TRUE generalization test is
the held-out configs in evaluate.py (IC1/IC4/IC6 + unseen TEST configs).
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT+"/experiments/nn_correctors")
import numpy as np, torch, torch.nn as nn
OUT = ROOT+"/experiments/nn_correctors"; dev="cuda" if torch.cuda.is_available() else "cpu"
D = np.load(OUT+"/datasets.npz")
curves = {}

def mlp(din,dh,dout,nh=2):
    layers=[nn.Linear(din,dh),nn.SiLU()]
    for _ in range(nh-1): layers+=[nn.Linear(dh,dh),nn.SiLU()]
    layers+=[nn.Linear(dh,dout)]; return nn.Sequential(*layers)

def train(X,y,dh,dout,epochs,name,asym=False):
    n=len(X); idx=np.random.RandomState(0).permutation(n); ntr=int(0.85*n)
    Xt=torch.tensor(X[idx[:ntr]],device=dev); yt=torch.tensor(y[idx[:ntr]],device=dev)
    Xv=torch.tensor(X[idx[ntr:]],device=dev); yv=torch.tensor(y[idx[ntr:]],device=dev)
    net=mlp(X.shape[1],dh,dout).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    def loss(p,t):
        if asym:
            e=p.squeeze(-1)-t; return torch.mean(torch.where(e>0,4.0*e*e,e*e))  # over-predict c worse (under-resolve)
        return torch.mean((p-t)**2) if t.dim()>1 else torch.mean((p.squeeze(-1)-t)**2)
    tr=[]; vl=[]; best=1e9; bw=None
    for ep in range(epochs):
        net.train(); l=loss(net(Xt),yt); opt.zero_grad(); l.backward(); opt.step()
        if ep%50==0 or ep==epochs-1:
            net.eval()
            with torch.no_grad(): v=float(loss(net(Xv),yv))
            tr.append(float(l)); vl.append(v)
            if v<best: best=v; bw={k:vv.clone() for k,vv in net.state_dict().items()}
    net.load_state_dict(bw); net.eval()
    curves[name]={"epoch":list(range(0,epochs,50)),"train":tr,"val":vl,"best_val":best,"n":int(n)}
    print(f"[train] {name}: n={n} best_val={best:.5f}", flush=True)
    return {k:v.cpu().numpy() for k,v in net.state_dict().items()}

# A1
w1 = train(((D["Xc"]-D["Xc_mu"])/D["Xc_sd"]).astype(np.float32), D["Yc"].astype(np.float32),
           64, 1, 2500, "A1_c_opt", asym=True)
np.savez(OUT+"/a1_weights.npz", **w1, mu=D["Xc_mu"], sd=D["Xc_sd"])
# A2 (if enough data)
if len(D["Xr"])>=40:
    Yn=((D["Yr"]-D["Yr_mu"])/D["Yr_sd"]).astype(np.float32)
    w2 = train(((D["Xr"]-D["Xr_mu"])/D["Xr_sd"]).astype(np.float32), Yn, 128, 18, 3000, "A2_residual")
    np.savez(OUT+"/a2_weights.npz", **w2, mu=D["Xr_mu"], sd=D["Xr_sd"], ymu=D["Yr_mu"], ysd=D["Yr_sd"])
    print("[train] A2 weights saved")
else:
    print(f"[train] A2 SKIPPED — only {len(D['Xr'])} windows (too few)")
json.dump(curves, open(OUT+"/train_curves.json","w"), indent=2)
print("TRAIN_DONE")
