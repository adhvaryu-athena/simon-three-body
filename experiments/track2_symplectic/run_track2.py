"""
Track 2: time-symmetric adaptive (tsalf) + 4th-order symplectic (yoshida4) vs
baselines (fixed leapfrog, heuristic adaptive) on IC1/IC4/IC6.
Force model = 'soft' (SIMON physics minus the inert NN). Metric panel:
|dE/E0|max (softened PE), short-horizon RMS (~2 Lyap), force-evals, bounded.
PASS = energy bounded across ICs (esp IC1) at cost competitive with fine uniform dt.
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/track2_symplectic"; os.makedirs(OUT, exist_ok=True)
T, NS, G = 100.0, 5000, sc.G_TOY
MODE = 'soft'
ICS = ["IC1", "IC4", "IC6"]
L = []
def log(s): print(s); L.append(s)

def panel_for(times, pos, vel, pref, m, fe, lam, maxdE_override=None):
    pan = sc.metric_panel(times, pos, vel, pref, m, G, fe, lyap=lam)
    if maxdE_override is not None:
        pan['max_dE_pct'] = maxdE_override
    return pan

rows = []
t0 = time.time()
for ic in ICS:
    m, x0, v0 = sc.get_ic(ic); lam = abs(sc.ic_meta(ic)["exp_lambda"]) or 0.1
    _, pref, vref = sc.ias15_reference(m, x0, v0, T, NS, G)
    log("="*96); log(f"{ic}  (lambda~{lam:.3f})  force-model={MODE}"); log("="*96)
    hdr = f"{'method':<26}{'param':>9}{'|dE/E0|max':>13}{'rms_short':>11}{'rms_final':>11}{'bnd':>6}{'force_evals':>12}"
    log(hdr); log("-"*len(hdr))
    methods = []
    # baselines
    for dt in [0.04, 0.005]:
        t, p, vv, info = sc.simulate_leapfrog(m, x0, v0, dt, T, NS, G, mode=MODE, adaptive=False)
        methods.append((f"leapfrog_fixed", f"dt={dt}", panel_for(t, p, vv, pref, m, info['force_evals'], lam)))
    t, p, vv, info = sc.simulate_leapfrog(m, x0, v0, 0.04, T, NS, G, mode=MODE, adaptive=True)
    methods.append(("leapfrog_adaptive(heur)", "dt=0.04", panel_for(t, p, vv, pref, m, info['force_evals'], lam)))
    # Track 2 candidates
    for ds in [0.04, 0.02]:
        t, p, vv, info = si.yoshida4_simulate(m, x0, v0, ds, T, NS, G, mode=MODE)
        methods.append(("yoshida4", f"ds={ds}", panel_for(t, p, vv, pref, m, info['force_evals'], lam)))
    for eta in [0.1, 0.05, 0.02]:
        t, p, vv, info = si.tsalf_simulate(m, x0, v0, eta, T, NS, G, mode=MODE)
        methods.append(("tsalf", f"eta={eta}", panel_for(t, p, vv, pref, m, info['force_evals'], lam,
                                                          maxdE_override=info['max_dE_steps'])))
    for name, param, pan in methods:
        dE = pan['max_dE_pct']; dEs = "inf" if not np.isfinite(dE) else f"{dE:.4f}%"
        log(f"{name:<26}{param:>9}{dEs:>13}{pan['rms_short']:>11.4f}{pan['rms_final']:>11.3f}"
            f"{str(pan['bounded']):>6}{pan['force_evals']:>12d}")
        rows.append(dict(ic=ic, method=name, param=param, **pan))
    log("")

# --- verdict per IC and overall ---
log("="*96); log("TRACK 2 VERDICT"); log("="*96)
def best_bounded(ic, name_prefix):
    cand = [r for r in rows if r['ic']==ic and r['method'].startswith(name_prefix) and r['bounded'] and np.isfinite(r['max_dE_pct'])]
    return min(cand, key=lambda r: r['max_dE_pct']) if cand else None

for ic in ICS:
    lf04 = next(r for r in rows if r['ic']==ic and r['method']=='leapfrog_fixed' and r['param']=='dt=0.04')
    lf005= next(r for r in rows if r['ic']==ic and r['method']=='leapfrog_fixed' and r['param']=='dt=0.005')
    ts   = best_bounded(ic, 'tsalf'); y4 = best_bounded(ic, 'yoshida4')
    log(f"{ic}: leapfrog dt=0.04 dE={lf04['max_dE_pct']:.3f}% bnd={lf04['bounded']} fe={lf04['force_evals']} | "
        f"dt=0.005 dE={lf005['max_dE_pct']:.3f}% fe={lf005['force_evals']}")
    if y4:  log(f"     yoshida4 best: {y4['param']} dE={y4['max_dE_pct']:.4f}% bnd=True fe={y4['force_evals']}")
    if ts:  log(f"     tsalf    best: {ts['param']} dE={ts['max_dE_pct']:.4f}% bnd=True fe={ts['force_evals']}")

# headline checks
ic1_lf04 = next(r for r in rows if r['ic']=='IC1' and r['method']=='leapfrog_fixed' and r['param']=='dt=0.04')
ic1_ts = best_bounded('IC1','tsalf')
ic4_lf005 = next(r for r in rows if r['ic']=='IC4' and r['method']=='leapfrog_fixed' and r['param']=='dt=0.005')
ic4_y4 = best_bounded('IC4','yoshida4')
log("")
log("HEADLINE CHECKS:")
log(f"  IC1 close-encounter: tsalf bounded@{ic1_ts['param'] if ic1_ts else 'NONE'} "
    f"dE={ic1_ts['max_dE_pct']:.3f}% fe={ic1_ts['force_evals'] if ic1_ts else '-'} "
    f"(vs leapfrog dt=0.04 bnd={ic1_lf04['bounded']} dE={ic1_lf04['max_dE_pct']:.2f}%)")
if ic4_y4:
    log(f"  IC4 under-resolution: yoshida4 {ic4_y4['param']} dE={ic4_y4['max_dE_pct']:.4f}% fe={ic4_y4['force_evals']} "
        f"vs leapfrog dt=0.005 dE={ic4_lf005['max_dE_pct']:.3f}% fe={ic4_lf005['force_evals']} "
        f"-> cost ratio {ic4_y4['force_evals']/ic4_lf005['force_evals']:.2f}x")
log("")
log(f"[wall {time.time()-t0:.1f}s]")
open(OUT+"/track2_results.txt","w").write("\n".join(L)+"\n")
json.dump(rows, open(OUT+"/track2_results.json","w"), indent=2, default=str)
print(f"\nwrote {OUT}/track2_results.{{txt,json}}")
