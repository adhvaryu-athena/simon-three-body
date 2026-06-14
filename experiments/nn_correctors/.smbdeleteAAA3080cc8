"""
Stage 1: generate population + both training datasets.
 A1 (c_opt): per-step (geometry features, reference-matched optimal scalar c).
 A2 (residual): per W-step window at close encounters (entry features, 18-d exit
     residual leapfrog-vs-ias15 from a common start).
Config-level split; IC1/IC4/IC6 fully held out. Save npz + held-out configs JSON.
"""
import sys, os, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src"); sys.path.insert(0, ROOT + "/experiments/nn_correctors")
import numpy as np
import simon_core as sc
import corr_lib as cl

OUT = ROOT + "/experiments/nn_correctors"; os.makedirs(OUT, exist_ok=True)
DT, T, W, GATE = 0.04, 30.0, 12, cl.NN_THRESH
N_TRAIN, N_TEST = 220, 24
rng = np.random.RandomState(20260601)
t0 = time.time(); log=lambda s: print(s, flush=True)

def make_configs(n):
    out=[]; tried=0
    while len(out)<n:
        tried+=1; c=cl.gen_config(rng)
        if c is not None and cl.trainable(*c): out.append(c)
        if tried>30*n: break
    return out

log("[gen] building configs ...")
train_cfgs = make_configs(N_TRAIN); test_cfgs = make_configs(N_TEST)
heldout = {ic: sc.get_ic(ic) for ic in ["IC1","IC4","IC6"]}

# ---- A1 dataset: per-step c_opt ----
log("[gen] A1 c_opt dataset ...")
Xc=[]; Yc=[]
for ci,(m,x0,v0) in enumerate(train_cfgs):
    P=cl.prep(m); x=x0.copy(); v=v0.copy()
    a=cl.acc_full(x,*P)
    nstep=int(round(T/DT))
    for step in range(nstep):
        c,nc=cl.c_opt_step(x,v,DT,P)              # sample every step; close-enc steps are the valuable ones
        if nc>0 and np.isfinite(c) and 0.05<c<20:
            Xc.append(cl.features(x,v,m,P)); Yc.append(math.log(c))
        vh=v+0.5*DT*a; x=x+DT*vh; a=cl.acc_full(x,*P); v=vh+0.5*DT*a
        if not np.all(np.isfinite(x)): break
    if ci%40==0: log(f"   A1 cfg {ci}/{N_TRAIN}  states={len(Xc)}  [{time.time()-t0:.0f}s]")
Xc=np.array(Xc,np.float32); Yc=np.array(Yc,np.float32)
log(f"[gen] A1 dataset {Xc.shape}  log(c): mean={Yc.mean():.4f} std={Yc.std():.4f}")

# ---- A2 dataset: per-window exit residual (leapfrog vs ias15 from common start) ----
log("[gen] A2 residual dataset ...")
Xr=[]; Yr=[]
def leap_W(m,x,v,steps):
    P=cl.prep(m); a=cl.acc_full(x,*P); x=x.copy(); v=v.copy()
    for _ in range(steps):
        vh=v+0.5*DT*a; x=x+DT*vh; a=cl.acc_full(x,*P); v=vh+0.5*DT*a
    return x,v
for ci,(m,x0,v0) in enumerate(train_cfgs):
    P=cl.prep(m); ii,jj=P[0],P[1]; x=x0.copy(); v=v0.copy()
    nstep=int(round(T/DT)); s=0
    while s+W<=nstep:
        rmin=float(np.min(np.sqrt(np.einsum('ij,ij->i',x[jj]-x[ii],x[jj]-x[ii])+1e-30)))
        if rmin<GATE:
            feat=cl.features(x,v,m,P)
            xl,vl=leap_W(m,x,v,W)
            try:
                _,pi,vi,_=cl.ias15_truth(m,x,v,W*DT,2)   # 2 samples: start + exit
                resid=np.concatenate([(xl-pi[-1]).ravel(),(vl-vi[-1]).ravel()])
                if np.all(np.isfinite(resid)) and np.linalg.norm(resid)<1e3:
                    Xr.append(feat); Yr.append(resid.astype(np.float32))
                x,v=xl,vl; s+=W; continue
            except Exception: pass
        # advance one step
        a=cl.acc_full(x,*P); vh=v+0.5*DT*a; x=x+DT*vh; a=cl.acc_full(x,*P); v=vh+0.5*DT*a; s+=1
        if not np.all(np.isfinite(x)): break
    if ci%40==0: log(f"   A2 cfg {ci}/{N_TRAIN}  windows={len(Xr)}  [{time.time()-t0:.0f}s]")
Xr=np.array(Xr,np.float32); Yr=np.array(Yr,np.float32)
log(f"[gen] A2 dataset {Xr.shape}")

# ---- held-out eval set: configs + ias15 truth (T) + measured lambda ----
log("[gen] held-out eval truth + lambda ...")
NS_EVAL=600
eval_set={}
def pack(name,m,x0,v0):
    t,p,vv,steps=cl.ias15_truth(m,x0,v0,T,NS_EVAL)
    lam=cl.lyap_estimate(m,x0,v0,T,NS_EVAL)
    eval_set[name]=dict(m=m.tolist(),x0=x0.tolist(),v0=v0.tolist(),
                        ias15_pos=p.tolist(),ias15_vel=vv.tolist(),ias15_steps=steps,lam=lam)
for ic,(m,x0,v0) in heldout.items(): pack(ic,m,x0,v0)
for k,(m,x0,v0) in enumerate(test_cfgs): pack(f"TEST{k:02d}",m,x0,v0)
log(f"[gen] eval set: {list(heldout)} + {len(test_cfgs)} TEST configs")

np.savez(OUT+"/datasets.npz", Xc=Xc,Yc=Yc,Xr=Xr,Yr=Yr,
         Xc_mu=Xc.mean(0),Xc_sd=Xc.std(0)+1e-6,
         Xr_mu=Xr.mean(0) if len(Xr) else np.zeros(7),Xr_sd=(Xr.std(0)+1e-6) if len(Xr) else np.ones(7),
         Yr_mu=Yr.mean(0) if len(Yr) else np.zeros(18),Yr_sd=(Yr.std(0)+1e-6) if len(Yr) else np.ones(18))
json.dump(dict(DT=DT,T=T,W=W,NS_EVAL=NS_EVAL,eval_set=eval_set),
          open(OUT+"/eval_set.json","w"), default=str)
log(f"[gen] saved datasets.npz + eval_set.json  [{time.time()-t0:.0f}s]")
print("GEN_DONE")
