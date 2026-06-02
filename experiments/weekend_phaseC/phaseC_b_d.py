"""
weekend_phaseC part 2: complete the operating-envelope table + plot + verdict.
 (b) ORACLE encounter correction = upper bound on ANY encounter NN: at every
     macro-step where the encounter gate fires (pair close AND approaching, or
     simply close), snap the binary (bodies 0,1) to the ias15 truth. If even the
     oracle can't improve on (a), no trained NN can.
 (d) tsalf (Track 2) = the resolution fix (step-capped; report completed time).
Builds operating_envelope.{txt,png}. Honest CLICK/FOLD verdict.
"""
import sys, os, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/weekend_phaseC"
G = sc.G_REAL; T, dt, NS = 100.0, 0.04, 5000
R_BINS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]
m = np.array([1.0, 0.5, 0.1]); mb = 1.5
L=[]; log=lambda s:(print(s,flush=True),L.append(s))

def binary_single_ic(r_bin):
    x = np.array([[-r_bin*0.5/mb,0,0],[r_bin*1.0/mb,0,0],[3.0,0,0]], float)
    vrel = math.sqrt(G*mb/r_bin); v2 = math.sqrt(G*mb/3.0)
    v = np.array([[0,-vrel*0.5/mb,0],[0,vrel*1.0/mb,0],[0,v2,0]], float)
    M=m.sum(); x-=(m[:,None]*x).sum(0)/M; v-=(m[:,None]*v).sum(0)/M
    return x, v

def binrange(pos): d=np.linalg.norm(pos[:,0,:]-pos[:,1,:],axis=1); return float(d.min()),float(d.max())
def tavg_rms(p,pr): return float(np.sqrt(np.mean(sc.rms_sep(p,pr)**2)))

# ---- ORACLE encounter-NN upper bound: heuristic backbone, snap binary to ias15
# at each macro-step start whenever the gate fires. Two gates tested:
#   'approach' = Sushant-style (close AND approaching)   ; 'any-close' = close only (more charitable)
def heuristic_with_oracle(x0, v0, pref, vref, times_ref, gate='any-close'):
    P = sc._prep(m, G); ii, jj = P['ii'], P['jj']; eps2 = sc.EPS**2
    x = x0.copy(); v = v0.copy()
    n_steps = int(round(T/dt)); times = np.linspace(0,T,NS)
    pos = np.full((NS,3,3), np.nan); vel = np.full((NS,3,3), np.nan)
    fe=[0]
    def acc(xx): fe[0]+=1; return sc.compute_acc(xx,ii,jj,P['Gmimj'],P['inv_mi'],P['inv_mj'],P['log_mi'],P['log_mj'],None,eps2,'soft')
    def ias_at(t):  # nearest ias15 sample
        k=min(NS-1,int(round(t/T*(NS-1)))); return pref[k], vref[k]
    a=acc(x); pos[0]=x; vel[0]=v; si=0; gate_fires=0; snaps=0
    adapt=0.05; maxs=16
    for step in range(n_steps):
        t_cur=step*dt
        r01=np.linalg.norm(x[0]-x[1]); vr=float((x[1]-x[0])@(v[1]-v[0]))/(r01+1e-30)
        vcirc=math.sqrt(G*mb/max(r01,1e-9))
        fired = (r01<0.15) if gate=='any-close' else (r01<0.15 and vr< -0.4*vcirc)
        if fired:
            gate_fires+=1
            pr_t, vr_t = ias_at(t_cur)           # ORACLE: snap binary (0,1) to ias15 truth
            for b in (0,1): x[b]=pr_t[b]; v[b]=vr_t[b]
            a=acc(x); snaps+=1
        rmin=float(np.min(np.sqrt(np.einsum('ij,ij->i',x[jj]-x[ii],x[jj]-x[ii])+1e-30)))
        if rmin<adapt:
            nsub=min(maxs,max(2,int(np.ceil(adapt/rmin)))); h=dt/nsub
            for _ in range(nsub):
                vh=v+0.5*h*a; x=x+h*vh; a=acc(x); v=vh+0.5*h*a
        else:
            vh=v+0.5*dt*a; x=x+dt*vh; a=acc(x); v=vh+0.5*dt*a
        tc=(step+1)*dt
        while si<NS-1 and times[si+1]<=tc+1e-9: si+=1; pos[si]=x; vel[si]=v
    while si<NS-1: si+=1; pos[si]=x; vel[si]=v
    return pos, vel, dict(force_evals=fe[0], gate_pct=100.0*gate_fires/n_steps, snaps=snaps)

log("="*108); log("OPERATING ENVELOPE — binary-single (G=4pi^2, masses [1,0.5,0.1], T=100, dt=0.04)")
log("(a)=SIMON heuristic noNN  (b)=ORACLE encounter-NN upper bound  (c)=ias15  (d)=tsalf resolution fix")
log("="*108)
hdr=f"{'r_bin':>7}{'gate%(b)':>9}{'RMS(a-c)':>12}{'RMS(b-c)':>12}{'improve%':>10}{'|dE|a%':>11}{'|dE|tsalf%':>12}{'tsalf done@yr':>14}"
log(hdr); log("-"*len(hdr))
rows=[]
for rb in R_BINS:
    x0,v0 = binary_single_ic(rb)
    tref, pc, vc = sc.ias15_reference(m, x0, v0, T, NS, G)
    _, pa, va, ia = sc.simulate_leapfrog(m,x0,v0,dt,T,NS,G,mode='soft',adaptive=True)
    rms_a = tavg_rms(pa,pc); dEa = sc.max_dE(pa,va,m,G,eps=sc.EPS)
    # (b) oracle
    pb, vb, ib = heuristic_with_oracle(x0,v0,pc,vc,tref,gate='any-close')
    rms_b = tavg_rms(pb,pc)
    improve = 100.0*(rms_a-rms_b)/rms_a if rms_a>0 else 0.0
    # (d) tsalf, step-capped
    t0=time.time()
    td,pd,vd,idd = si.tsalf_simulate(m,x0,v0,0.05,T,NS,G,mode='soft',max_steps=2_000_000)
    dEd = idd['max_dE_steps']; done_yr = T if idd['completed'] else (idd['step_states'][0][-1])
    log(f"{rb:>7}{ib['gate_pct']:>8.1f}%{rms_a:>12.2f}{rms_b:>12.2f}{improve:>9.1f}%{dEa:>10.1f}%{dEd:>11.3g}%{done_yr:>14.2f}")
    rows.append(dict(r_bin=rb, gate_pct_b=ib['gate_pct'], rms_a=rms_a, rms_b=rms_b, improve_pct=improve,
                     maxdE_a=dEa, maxdE_tsalf=dEd, tsalf_done_yr=float(done_yr),
                     tsalf_completed=idd['completed'], tsalf_fe=idd['force_evals'],
                     binrange_a=list(binrange(pa)), binrange_ias15=list(binrange(pc)), binrange_tsalf=list(binrange(pd))))
log("")
imps=[r['improve_pct'] for r in rows]
verdict = ("FOLD — NN (even oracle) improvement null across all r_bin; failure is binary under-resolution "
           "(timestep problem), unfixable by any force/encounter correction. Fix = adaptive resolution (tsalf/ias15).")
if max(imps)>20 and rows[0]['improve_pct']>20: verdict="CLEAN ENVELOPE — NN helps at small r_bin"
elif max(imps)>5: verdict="PARTIAL — modest NN help somewhere; inspect"
log("VERDICT: "+verdict)
log("Constructive: tsalf bounds the binary where the heuristic explodes (see |dE|tsalf vs |dE|a, binary ranges).")

# ---- plot: improvement vs r_bin (the operating-envelope result) + the resolution story ----
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
rbv=np.array(R_BINS)
ax[0].semilogx(rbv,[r['improve_pct'] for r in rows],'o-',color='#DC2626',lw=2,label='NN(oracle) improvement')
ax[0].axhline(20,ls=':',color='gray',label='20% target (small r_bin)'); ax[0].axhline(5,ls=':',color='green')
ax[0].axhline(0,color='k',lw=0.6); ax[0].set_xlabel('binary separation r_bin (AU, log)')
ax[0].set_ylabel('NN improvement over heuristic (%)'); ax[0].set_title('NN operating envelope (oracle upper bound)')
ax[0].legend(fontsize=8); ax[0].grid(True,which='both',alpha=0.3)
ax[1].loglog(rbv,[max(r['maxdE_a'],1e-3) for r in rows],'s-',color='#DC2626',lw=2,label='(a) heuristic |dE|max')
ax[1].loglog(rbv,[max(r['maxdE_tsalf'],1e-6) for r in rows],'o-',color='#16A34A',lw=2,label='(d) tsalf |dE|max')
ax[1].axhline(1.0,ls=':',color='gray'); ax[1].set_xlabel('binary separation r_bin (AU, log)')
ax[1].set_ylabel('max |dE/E0| (%, log)'); ax[1].set_title('Resolution fix: tsalf bounds energy, heuristic explodes')
ax[1].legend(fontsize=8); ax[1].grid(True,which='both',alpha=0.3)
fig.tight_layout(); fig.savefig(OUT+"/operating_envelope.png",dpi=170); plt.close()
log("wrote operating_envelope.png")
open(OUT+"/operating_envelope.txt","w").write("\n".join(L)+"\n")
json.dump(dict(verdict=verdict,rows=rows),open(OUT+"/operating_envelope.json","w"),indent=2,default=str)
print("PHASEC_DONE")
