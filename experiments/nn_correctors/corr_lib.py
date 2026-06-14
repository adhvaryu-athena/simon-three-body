"""
corr_lib.py — shared core for the NN-corrector vs equal-compute-physics experiment.
Backbone: pure-Newtonian velocity-Verlet (no softening), so correctors and the
finer-step baseline are compared against the true dynamics with no softening confound.
"""
import math, numpy as np
import simon_core as sc

G = sc.G_TOY                  # toy units; configs generated in these units
NN_THRESH = 0.15              # "close pair" radius (correctors act here)

def make_pairs(N):
    ii,jj=[],[]
    for i in range(N):
        for j in range(i+1,N): ii.append(i); jj.append(j)
    return np.array(ii), np.array(jj)

def gen_config(rng):
    """Close-encounter-rich bounded 3-body: inner body on an ECCENTRIC orbit with
    controlled perihelion (so r01 dips into the corrector's regime), perturber
    circular outside. Returns (m,x0,v0) or None (unbound)."""
    mS=1.0; m1=10**rng.uniform(-2.5,-0.7); m2=10**rng.uniform(-3.0,-1.0)
    r_apo=rng.uniform(0.6,2.5)
    r_peri=max(0.07, rng.uniform(0.08,0.45)*r_apo); r_peri=min(r_peri,0.9*r_apo)
    a=0.5*(r_apo+r_peri); GM1=G*(mS+m1)
    v_apo=math.sqrt(max(GM1*(2.0/r_apo-1.0/a),1e-9))
    th=rng.uniform(0,2*np.pi); ct,st=math.cos(th),math.sin(th)
    x1=np.array([r_apo*ct,r_apo*st,0.0]); v1=np.array([-v_apo*st,v_apo*ct,0.0])
    r2=rng.uniform(2.0,6.0); th2=rng.uniform(0,2*np.pi); c2,s2=math.cos(th2),math.sin(th2)
    vc2=math.sqrt(G*(mS+m1)/r2)*rng.uniform(0.75,1.1)
    x2=np.array([r2*c2,r2*s2,0.0]); v2=np.array([-vc2*s2,vc2*c2,0.0])
    m=np.array([mS,m1,m2]); x=np.array([[0,0,0],x1,x2]); v=np.array([[0,0,0],v1,v2])
    M=m.sum(); x-=(m[:,None]*x).sum(0)/M; v-=(m[:,None]*v).sum(0)/M
    KE=0.5*np.sum(m[:,None]*v**2); PE=-sum(G*m[i]*m[j]/np.linalg.norm(x[i]-x[j]) for i in range(3) for j in range(i+1,3))
    return (m,x,v) if KE+PE<0 else None

def trainable(m,x0,v0,dt=0.04,T=30.0,NS=300):
    """Keep config only if plain leapfrog stays bounded (not exploded) AND has a
    close encounter (min sep < NN_THRESH) — the regime where a corrector can act."""
    _,p,_,info=leapfrog(m,x0,v0,dt,T,NS)
    if not np.all(np.isfinite(p)): return False
    com=(p*m[None,:,None]).sum(1,keepdims=True)/m.sum()
    if np.max(np.linalg.norm(p-com,axis=2))>30.0: return False
    ii,jj=make_pairs(len(m))
    mins=min(float(np.min(np.linalg.norm(p[k][jj]-p[k][ii],axis=1))) for k in range(0,NS,3))
    return mins < NN_THRESH

def prep(m):
    ii,jj=make_pairs(len(m)); mi,mj=m[ii],m[jj]
    return ii,jj,G*mi*mj,1.0/mi,1.0/mj

def acc_split(x, ii, jj, Gmimj, inv_mi, inv_mj):
    """Newtonian accelerations split into far (r>=NN_THRESH) and close (r<NN_THRESH)."""
    rij=x[jj]-x[ii]; r2=np.einsum('ij,ij->i',rij,rij); r=np.sqrt(r2+1e-30)
    F=Gmimj/(r2*r+1e-30); Fvec=F[:,None]*rij
    close=r<NN_THRESH
    N=x.shape[0]; a_far=np.zeros((N,3)); a_close=np.zeros((N,3))
    for k in range(len(ii)):
        tgt=a_close if close[k] else a_far
        tgt[ii[k]]+=Fvec[k]*inv_mi[k]; tgt[jj[k]]-=Fvec[k]*inv_mj[k]
    return a_far, a_close, int(close.sum())

def acc_full(x, ii, jj, Gmimj, inv_mi, inv_mj):
    af,ac,_=acc_split(x,ii,jj,Gmimj,inv_mi,inv_mj); return af+ac

def converged_onestep(x,v,dt,accf,K=32):
    a=accf(x); xx,vv=x.copy(),v.copy(); h=dt/K
    for _ in range(K):
        vh=vv+0.5*h*a; xx=xx+h*vh; a=accf(xx); vv=vh+0.5*h*a
    return xx,vv

def c_opt_step(x,v,dt,P):
    """Reference-matched optimum scalar c on the close-pair force (closed-form LSQ
    vs converged-Newtonian one-step). c=1 if no close pair."""
    ii,jj,Gmimj,inv_mi,inv_mj=P
    af,ac,nc=acc_split(x,ii,jj,Gmimj,inv_mi,inv_mj)
    if nc==0: return 1.0, 0
    p=x+dt*v+0.5*dt*dt*af; q=0.5*dt*dt*ac
    xref,_=converged_onestep(x,v,dt,lambda xx:acc_full(xx,ii,jj,Gmimj,inv_mi,inv_mj))
    num=float(np.sum((xref-p)*q)); den=float(np.sum(q*q))+1e-30
    return num/den, nc

def features(x,v,m,P):
    """Geometry-only features (tightest pair + third-body perturbation)."""
    ii,jj=P[0],P[1]
    rij=x[jj]-x[ii]; vij=v[jj]-v[ii]
    r=np.sqrt(np.einsum('ij,ij->i',rij,rij)+1e-30); k=int(np.argmin(r)); i,j=ii[k],jj[k]
    rv=r[k]; vrel=math.sqrt(float(vij[k]@vij[k])+1e-30); eproxy=abs(float(rij[k]@vij[k])/(rv*vrel+1e-30))
    GM=G*(m[i]+m[j]); t_orb=2*np.pi*math.sqrt(rv**3/(GM+1e-30)); t_fly=rv/vrel
    kk=[b for b in range(len(m)) if b not in (i,j)][0]; a_int=GM/rv**2
    a_tid=G*m[kk]*(1.0/(np.linalg.norm(x[i]-x[kk])**2+1e-9)+1.0/(np.linalg.norm(x[j]-x[kk])**2+1e-9))
    return np.array([math.log(rv),math.log(vrel+1e-12),eproxy,math.log(GM+1e-30),
                     math.log(t_orb/(t_fly+1e-30)+1e-30),math.log(a_tid/(a_int+1e-30)+1e-30),
                     math.log(rv/0.15)],dtype=np.float32)

# ---------- integrators (leapfrog backbone, optional correctors) ----------
def leapfrog(m,x0,v0,dt,T,NS,c_predict=None,res_predict=None,res_feat_norm=None,
             gate=0.15,window=12,max_steps=4_000_000):
    """Plain leapfrog (c_predict=res_predict=None), or c_opt-corrected (c_predict),
    or residual-corrected (res_predict). Returns times,pos,vel,info(fe, nn_calls)."""
    ii,jj,Gmimj,inv_mi,inv_mj=prep(m); P=(ii,jj,Gmimj,inv_mi,inv_mj)
    x=x0.astype(float).copy(); v=v0.astype(float).copy(); N=len(m)
    n_steps=int(round(T/dt)); times=np.linspace(0,T,NS)
    pos=np.full((NS,N,3),np.nan); vel=np.full((NS,N,3),np.nan)
    fe=[0]; nn=[0]
    def accc(xx, c=1.0):
        fe[0]+=1; af,ac,nc=acc_split(xx,ii,jj,Gmimj,inv_mi,inv_mj); return af+c*ac, nc
    a,_=accc(x); pos[0]=x; vel[0]=v; si=0
    steps_remaining=-1; res_pending=None      # Issue 2: -1 means no correction pending
    for step in range(min(n_steps,max_steps)):
        c=1.0
        if c_predict is not None:
            af,ac,nc=acc_split(x,ii,jj,Gmimj,inv_mi,inv_mj)
            if nc>0:
                nn[0]+=1; c=float(np.clip(c_predict(features(x,v,m,P)),0.2,5.0))
                # Issue 3: apply the current c to the START half-kick as well. Reuse the
                # split already computed above for gating (no extra force evaluation); the
                # same c is used for both half-kicks of this step.
                a=af+c*ac
        vh=v+0.5*dt*a; x=x+dt*vh; a,_=accc(x,c); v=vh+0.5*dt*a
        # residual corrector (A2): schedule at encounter entry, then apply the predicted
        # residual after EXACTLY `window` future leapfrog steps, so deployment timing
        # matches the W-step training target (Issue 2). Target sign is IAS15-leapfrog, so
        # we add it (Issue 1, set in gen_data.py).
        if res_predict is not None:
            rmin=float(np.min(np.sqrt(np.einsum('ij,ij->i',x[jj]-x[ii],x[jj]-x[ii])+1e-30)))
            if steps_remaining<0 and rmin<gate:
                nn[0]+=1; feat=features(x,v,m,P)
                res_pending=res_predict(feat)*(res_feat_norm if res_feat_norm else 1.0)
                steps_remaining=window
            elif steps_remaining>0:
                steps_remaining-=1
                if steps_remaining==0:
                    x=x+res_pending[:N*3].reshape(N,3); v=v+res_pending[N*3:].reshape(N,3)
                    a,_=accc(x)
                    res_pending=None; steps_remaining=-1
        t_cur=(step+1)*dt
        while si<NS-1 and times[si+1]<=t_cur+1e-9: si+=1; pos[si]=x; vel[si]=v
        if not np.all(np.isfinite(x)): break
    while si<NS-1: si+=1; pos[si]=x; vel[si]=v
    return times,pos,vel,dict(fe=fe[0],nn_calls=nn[0],n_steps=n_steps)

def ias15_truth(m,x0,v0,T,NS):
    import rebound
    sim=rebound.Simulation(); sim.integrator="ias15"; sim.G=G
    for i in range(len(m)):
        sim.add(m=float(m[i]),x=float(x0[i,0]),y=float(x0[i,1]),z=float(x0[i,2]),
                vx=float(v0[i,0]),vy=float(v0[i,1]),vz=float(v0[i,2]))
    sim.move_to_com()
    times=np.linspace(0,T,NS); pos=np.zeros((NS,len(m),3)); vel=np.zeros((NS,len(m),3))
    for k,t in enumerate(times):
        sim.integrate(t)
        for i,p in enumerate(sim.particles): pos[k,i]=[p.x,p.y,p.z]; vel[k,i]=[p.vx,p.vy,p.vz]
    return times,pos,vel,int(sim.steps_done)

# ---------- metrics ----------
def energy(pos,vel,m,eps=1e-9):
    KE=0.5*np.einsum("kij,i->k",vel**2,m); PE=np.zeros(pos.shape[0]); e2=eps*eps
    for i in range(len(m)):
        for j in range(i+1,len(m)):
            d=pos[:,i,:]-pos[:,j,:]; r2=np.einsum("ki,ki->k",d,d); PE-=G*m[i]*m[j]/np.sqrt(r2+e2)
    return KE+PE
def maxdE(pos,vel,m):
    E=energy(pos,vel,m)
    if not np.all(np.isfinite(E)): return float("inf")
    return float(np.max(np.abs((E-E[0])/max(abs(E[0]),1e-30)))*100)
def rms_sep(a,b): d=a-b; return np.sqrt(np.mean(np.sum(d**2,axis=-1),axis=1))
def lyap_estimate(m,x0,v0,T=30.0,NS=600):
    _,p1,_,_=ias15_truth(m,x0,v0,T,NS)
    x0p=x0.copy(); x0p[1,0]+=1e-8
    _,p2,_,_=ias15_truth(m,x0p,v0,T,NS); d=rms_sep(p1,p2)
    t=np.linspace(0,T,NS); msk=(d>1e-7)&(d<0.1)
    if msk.sum()<5: return 0.1
    sl=np.polyfit(t[msk],np.log(d[msk]),1)[0]; return max(sl,0.02)
def short_horizon_rms(pos,pref,t,lam):
    tl=min(2.0/max(lam,1e-3),t[-1]); msk=t<=tl; d=rms_sep(pos,pref)
    return float(np.sqrt(np.mean(d[msk]**2))) if msk.sum() else float("nan")
def time_to_diverge(pos,pref,t,thresh=0.1):
    d=rms_sep(pos,pref); idx=np.where(d>thresh)[0]
    return float(t[idx[0]]) if len(idx) else float(t[-1])
def pair_fidelity(pos,pref,m):
    """max over pairs of the worse normalized drift in BOTH min and max bound-pair
    separation (closest-approach and widest-excursion fidelity). Issue 5: the min term
    was named in the docstring but never compared; both are now included."""
    worst=0.0
    for i in range(len(m)):
        for j in range(i+1,len(m)):
            dm=np.linalg.norm(pos[:,i]-pos[:,j],axis=1); dr=np.linalg.norm(pref[:,i]-pref[:,j],axis=1)
            worst=max(worst,
                      abs(dm.max()-dr.max())/(dr.max()+1e-9),
                      abs(dm.min()-dr.min())/(dr.min()+1e-9))
    return float(worst)
