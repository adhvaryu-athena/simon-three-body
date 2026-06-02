"""
Track 0/1: verification guard + matched-dt ablation reconciliation.
Guard: SIMON (adaptive+NN) at dt=0.04 must reproduce IC1 1.39%, IC3 0.67%,
IC4 13.77%, IC6 0.0006% |dE/E0|max. Then run A/C/D at dt in {0.04,0.02,0.01}
on IC1 and report the matched-dt panel to settle the "Run A ejects" claim.
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import simon_core as sc

OUT = ROOT + "/experiments/track1_reconciliation"
os.makedirs(OUT, exist_ok=True)
W = sc.load_scalar_nn_weights(ROOT + "/_orig/training (1)/pair_correction_nn.pt")
T, NS, G = 100.0, 5000, sc.G_TOY
L = []
def log(s): print(s); L.append(s)

t_start = time.time()
log("="*78)
log("VERIFICATION GUARD — SIMON (adaptive+NN), dt=0.04, T=100  (toy G=1.0)")
log("="*78)
EXP = {"IC1": 1.39, "IC3": 0.67, "IC4": 13.77, "IC6": 0.0006}
guard_ok = True
for ic in ["IC1", "IC3", "IC4", "IC6"]:
    m, x0, v0 = sc.get_ic(ic)
    t, pos, vel, info = sc.simulate_leapfrog(m, x0, v0, 0.04, T, NS, G, mode='nn', w=W, adaptive=True)
    dE = sc.max_dE(pos, vel, m, G)
    exp = EXP[ic]; ok = abs(dE - exp) <= max(0.05*exp, 0.05); guard_ok &= ok
    log(f"  {ic}: |dE/E0|max = {dE:8.4f}%  (exp ~{exp})  fe={info['force_evals']:6d}  -> {'OK' if ok else 'MISMATCH'}")
log(f"GUARD: {'PASS' if guard_ok else 'FAIL — downstream results suspect'}")

log("")
log("="*78)
log("TRACK 1 — matched-dt ablation on IC1  (A=NN/no-adapt, C=soft/adapt, D=full SIMON)")
log("Matched-dt reconciliation: Run A (no adaptive) vs Run D (full SIMON).")
log("="*78)
ic = "IC1"; m, x0, v0 = sc.get_ic(ic)
_, pref, vref = sc.ias15_reference(m, x0, v0, T, NS, G)
lam = sc.ic_meta(ic)["exp_lambda"]
RUNS = [("A_no_adaptive(NN,fixed)", dict(mode='nn', adaptive=False)),
        ("C_no_nn(soft,adaptive)",  dict(mode='soft', adaptive=True)),
        ("D_full_SIMON(NN,adaptive)", dict(mode='nn', adaptive=True))]
rows = []
hdr = f"{'run':<26}{'dt':>7}{'|dE/E0|max':>13}{'rms_short':>11}{'rms_final':>11}{'bounded':>9}{'force_evals':>12}"
log(hdr); log("-"*len(hdr))
for dt in [0.04, 0.02, 0.01]:
    for name, kw in RUNS:
        t, pos, vel, info = sc.simulate_leapfrog(m, x0, v0, dt, T, NS, G, w=W, **kw)
        pan = sc.metric_panel(t, pos, vel, pref, m, G, info['force_evals'], lyap=lam)
        rows.append(dict(run=name, dt=dt, **pan))
        dE = pan['max_dE_pct']; dEs = "inf" if not np.isfinite(dE) else f"{dE:.4f}%"
        log(f"{name:<26}{dt:>7}{dEs:>13}{pan['rms_short']:>11.4f}{pan['rms_final']:>11.3f}"
            f"{str(pan['bounded']):>9}{pan['force_evals']:>12d}")
    log("-"*len(hdr))

# Verdict on the reconciliation question
A_eject = {dt: next(r for r in rows if r['run'].startswith('A') and r['dt']==dt)['bounded'] for dt in [0.04,0.02,0.01]}
log("")
log("RECONCILIATION ANSWER:")
for dt in [0.04, 0.02, 0.01]:
    rA = next(r for r in rows if r['run'].startswith('A') and r['dt']==dt)
    log(f"  Run A @ dt={dt}: bounded={rA['bounded']}  |dE/E0|max={rA['max_dE_pct']:.3f}%  final_rms={rA['rms_final']:.2f} AU")
log(f"  -> At the operational dt=0.04, Run A {'EJECTS' if not A_eject[0.04] else 'STAYS BOUNDED'}.")
log(f"  (Published 161.7 AU ejection is {'consistent with' if not A_eject[0.04] else 'NOT reproduced at matched dt; likely a finer-dt / sampling-bug artifact'}.)")

log("")
log(f"[wall {time.time()-t_start:.1f}s]")
open(OUT + "/reconciliation.txt", "w").write("\n".join(L) + "\n")
json.dump(dict(guard_ok=guard_ok, rows=rows), open(OUT + "/reconciliation.json", "w"), indent=2, default=str)
print(f"\nwrote {OUT}/reconciliation.txt + .json")
