"""
Track 3: NN-learned stepsize multiplier on the time-symmetric (tsalf) base.
Target mu* = largest multiplier on h0=eta*t_dyn keeping one-step local position
error < TOL. Features: geometry only (tightest pair r,v,ecc-proxy,mass,GM,
t_orb/t_flyby ratio, third-body perturbation ratio). Asymmetric loss.
Train on random configs + IC2/IC3/IC5; HELD-OUT eval on IC1/IC4/IC6.
CLICK iff NN-tsalf < analytic-tsalf force-evals at equal accuracy on held-out.
"""
import sys, os, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np, torch, torch.nn as nn
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/track3_nn_stepper"; os.makedirs(OUT, exist_ok=True)
G = sc.G_TOY; ETA = 0.1; TOL = 1e-6
MU_GRID = np.array([0.25,0.354,0.5,0.707,1.0,1.414,2.0,2.83,4.0,5.66,8.0])
dev = "cuda" if torch.cuda.is_available() else "cpu"
L=[]; log=lambda s:(print(s,flush=True),L.append(s))

def acc_factory(m):
    P=sc._prep(m,G); ii,jj=P['ii'],P['jj']; eps2=sc.EPS**2
    def acc(x): return sc.compute_acc(x,ii,jj,P['Gmimj'],P['inv_mi'],P['inv_mj'],P['log_mi'],P['log_mj'],None,eps2,'newton')
    return acc, ii, jj

def features(x, v, m, ii, jj):
    rij=x[jj]-x[ii]; vij=v[jj]-v[ii]
    r=np.sqrt(np.einsum('ij,ij->i',rij,rij)+1e-30)
    k=int(np.argmin(r))                          # tightest pair
    i,j=ii[k],jj[k]
    rv=r[k]; vrel=math.sqrt(float(vij[k]@vij[k])+1e-30)
    eproxy=abs(float(rij[k]@vij[k])/(rv*vrel+1e-30))
    GM=G*(m[i]+m[j]); t_orb=2*np.pi*math.sqrt(rv**3/(GM+1e-30)); t_fly=rv/vrel
    # third body perturbation on tightest pair
    kk=[b for b in range(len(m)) if b not in (i,j)][0]
    a_int=GM/rv**2
    a_tid=G*m[kk]*(1.0/(np.linalg.norm(x[i]-x[kk])**2+1e-9)+1.0/(np.linalg.norm(x[j]-x[kk])**2+1e-9))
    pert=a_tid/(a_int+1e-30)
    return np.array([math.log(rv), math.log(vrel+1e-12), eproxy, math.log(GM+1e-30),
                     math.log(t_orb/(t_fly+1e-30)+1e-30), math.log(pert+1e-30),
                     math.log(ETA*min(t_orb,t_fly)+1e-30)], dtype=np.float32)

def one_step(x,v,a,h,acc):
    vh=v+0.5*h*a; x1=x+h*vh; a1=acc(x1); v1=vh+0.5*h*a1; return x1,v1
def converged(x,v,a,h,acc,K=32):
    xx,vv,aa=x,v,a; hh=h/K
    for _ in range(K):
        vh=vv+0.5*hh*aa; xx=xx+hh*vh; aa=acc(xx); vv=vh+0.5*hh*aa
    return xx,vv
def mu_star(x,v,a,acc,h0):
    best=MU_GRID[0]
    for mu in MU_GRID:
        h=mu*h0
        x1,_=one_step(x,v,a,h,acc); xs,_=converged(x,v,a,h,acc)
        e=float(np.max(np.abs(x1-xs)))
        if e<TOL: best=mu
        else: break
    return best

def gen_config(rng):
    mS=1.0; m1=10**rng.uniform(-3.0,-1.3); m2=10**rng.uniform(-3.3,-1.3)
    r1=rng.uniform(0.5,3.0); r2=rng.uniform(1.2,6.0)
    th1=rng.uniform(0,2*np.pi); th2=rng.uniform(0,2*np.pi); f1=rng.uniform(0.65,1.15); f2=rng.uniform(0.65,1.15)
    vc1=math.sqrt(G*mS/r1); vc2=math.sqrt(G*mS/r2)
    m=np.array([mS,m1,m2])
    x=np.array([[0,0,0],[r1*np.cos(th1),r1*np.sin(th1),0],[r2*np.cos(th2),r2*np.sin(th2),0]],float)
    v=np.array([[0,0,0],[-vc1*f1*np.sin(th1),vc1*f1*np.cos(th1),0],[-vc2*f2*np.sin(th2),vc2*f2*np.cos(th2),0]],float)
    M=m.sum(); x-=(m[:,None]*x).sum(0)/M; v-=(m[:,None]*v).sum(0)/M
    KE=0.5*np.sum(m[:,None]*v**2); PE=-sum(G*m[i]*m[j]/np.linalg.norm(x[i]-x[j]) for i in range(3) for j in range(i+1,3))
    return (m,x,v) if KE+PE<0 else None

# ---------- data generation ----------
log(f"[track3] device={dev} generating data ..."); t0=time.time()
rng=np.random.RandomState(7)
Xf=[]; Yf=[]
train_cfgs=[]
for ic in ["IC2","IC3","IC5"]:
    train_cfgs.append(sc.get_ic(ic))
while len(train_cfgs) < 60:
    c=gen_config(rng)
    if c is not None: train_cfgs.append(c)
TARGET=18000
for (m,x0,v0) in train_cfgs:
    acc,ii,jj=acc_factory(m); x=x0.copy(); v=v0.copy(); a=acc(x)
    for step in range(700):
        h0=ETA*si.t_dyn_min(x,v,m,ii,jj,G)
        if step%3==0:
            try:
                Xf.append(features(x,v,m,ii,jj)); Yf.append(math.log2(mu_star(x,v,a,acc,h0)))
            except Exception: pass
        # advance with analytic tsalf-like step (1 symmetric iter)
        h=h0
        x1,v1=one_step(x,v,a,h,acc); h=0.5*(h0+ETA*si.t_dyn_min(x1,v1,m,ii,jj,G))
        vh=v+0.5*h*a; x=x+h*vh; a=acc(x); v=vh+0.5*h*a
        if not np.all(np.isfinite(x)): break
    if len(Xf)>=TARGET: break
Xf=np.array(Xf,np.float32); Yf=np.array(Yf,np.float32)
log(f"[track3] dataset {Xf.shape} mu* log2 mean={Yf.mean():.3f} std={Yf.std():.3f} [{time.time()-t0:.0f}s]")
np.savez(OUT+"/data.npz", X=Xf, Y=Yf)

# feature importance sanity: correlation of each feature with target
for fi,nm in enumerate(["log_r","log_v","eproxy","log_GM","log_torb/tfly","log_pert","log_h0"]):
    c=np.corrcoef(Xf[:,fi],Yf)[0,1]; log(f"   corr({nm},log2mu*)={c:+.3f}")

# ---------- train ----------
mu=Xf.mean(0); sd=Xf.std(0)+1e-6
Xt=torch.tensor((Xf-mu)/sd,device=dev); Yt=torch.tensor(Yf,device=dev)
n=len(Xt); idx=torch.randperm(n); ntr=int(0.85*n)
tr,va=idx[:ntr],idx[ntr:]
net=nn.Sequential(nn.Linear(7,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1)).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
UNDER_W=5.0   # asymmetric: predicting mu too HIGH (under-resolution) penalized more
def aloss(pred,tgt):
    e=pred.squeeze(-1)-tgt; return torch.mean(torch.where(e>0,UNDER_W*e*e,e*e))
best=1e9; bw=None
for ep in range(3000):
    net.train(); p=net(Xt[tr]); l=aloss(p,Yt[tr]); opt.zero_grad(); l.backward(); opt.step()
    if ep%200==0 or ep==2999:
        net.eval()
        with torch.no_grad(): vl=aloss(net(Xt[va]),Yt[va]).item()
        if vl<best: best=vl; bw={k:v.clone() for k,v in net.state_dict().items()}
        if ep%600==0: log(f"   ep{ep} train={l.item():.4f} val={vl:.4f}")
net.load_state_dict(bw); net.eval()
log(f"[track3] trained, best val={best:.4f}")
# numpy weights for fast deploy
sdw={k:v.cpu().numpy() for k,v in net.state_dict().items()}
torch.save(net.state_dict(), OUT+"/stepper_nn.pt")
np.savez(OUT+"/stepper_norm.npz", mu=mu, sd=sd)

def silu(z): return z/(1+np.exp(-z))
def nn_predict(feat):
    h=(feat-mu)/sd
    h=silu(h@sdw['0.weight'].T+sdw['0.bias'])
    h=silu(h@sdw['2.weight'].T+sdw['2.bias'])
    return float(h@sdw['4.weight'].T+sdw['4.bias'])

# ---------- NN-controlled tsalf ----------
def tsalf_nn(m,x0,v0,T,NS,predict=None,eta=ETA,max_steps=3_000_000):
    acc,ii,jj=acc_factory(m); x=x0.copy(); v=v0.copy(); a=acc(x)
    ts=[0.0]; xs=[x.copy()]; vs=[v.copy()]; t=0.0; ns=0
    while t<T and ns<max_steps:
        td=si.t_dyn_min(x,v,m,ii,jj,G); h0=eta*td
        if predict is not None:
            mu_=2.0**predict(features(x,v,m,ii,jj)); mu_=min(8.0,max(0.25,mu_)); h0=h0*mu_
        h=h0; x1,v1=one_step(x,v,a,h,acc); td1=si.t_dyn_min(x1,v1,m,ii,jj,G)
        mu1=1.0
        if predict is not None: mu1=min(8.0,max(0.25,2.0**predict(features(x1,v1,m,ii,jj))))
        h=0.5*(h0+eta*td1*mu1)
        if t+h>T: h=T-t
        vh=v+0.5*h*a; x=x+h*vh; a=acc(x); v=vh+0.5*h*a; t+=h; ns+=1
        ts.append(t); xs.append(x.copy()); vs.append(v.copy())
        if not np.all(np.isfinite(x)): break
    ts=np.array(ts); xs=np.array(xs); vs=np.array(vs)
    return sc.max_dE(xs,vs,m,G,eps=1e-9), ns, (t>=T-1e-6)

# ---------- HELD-OUT eval: NN vs analytic ----------
log(""); log("HELD-OUT EVAL (IC1/IC4/IC6): analytic tsalf (mu=1) vs NN tsalf")
log(f"{'IC':<6}{'analytic dE%':>14}{'analytic fe':>12}{'NN dE%':>12}{'NN fe':>10}{'cost ratio':>11}")
eval_rows=[]
for ic in ["IC1","IC4","IC6"]:
    m,x0,v0=sc.get_ic(ic)
    dEa,fea,oka=tsalf_nn(m,x0,v0,100.0,5000,predict=None)
    dEn,fen,okn=tsalf_nn(m,x0,v0,100.0,5000,predict=nn_predict)
    ratio=fen/max(fea,1)
    log(f"{ic:<6}{dEa:>14.4f}{fea:>12d}{dEn:>12.4f}{fen:>10d}{ratio:>11.3f}")
    eval_rows.append(dict(ic=ic,analytic_dE=dEa,analytic_fe=fea,nn_dE=dEn,nn_fe=fen,cost_ratio=ratio,nn_bounded=okn))

# verdict: NN must reach comparable accuracy (within 3x dE) at materially lower fe (<0.85x), bounded
clicks=[r for r in eval_rows if r['nn_bounded'] and r['nn_dE']<=3*r['analytic_dE'] and r['cost_ratio']<0.85]
regress=[r for r in eval_rows if (not r['nn_bounded']) or r['nn_dE']>3*r['analytic_dE']]
log("")
if len(clicks)>=2 and not regress: verdict="CLICK"
elif clicks and not regress: verdict="PARTIAL"
elif regress: verdict="PARTIAL/REGRESS"
else: verdict="NULL (NN ~matches analytic; ship analytic by Occam)"
log(f"TRACK 3 VERDICT: {verdict}  (clicks={[r['ic'] for r in clicks]}, regress={[r['ic'] for r in regress]})")
open(OUT+"/track3_results.txt","w").write("\n".join(L)+"\n")
json.dump(dict(verdict=verdict,eval=eval_rows),open(OUT+"/track3_results.json","w"),indent=2,default=str)
print("TRACK3_DONE")
