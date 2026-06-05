import numpy as np, sys
sys.path.insert(0, r"C:\Aarush\Physics\set 3_after_adaptive")
from pair_eval_after_adaptive import simulate_rebound_ias15, HybridConfig
cfg = HybridConfig()
m  = np.array([1.0, 0.5, 0.25])
x0 = np.array([[0,0,0],[1,0,0],[-0.5,0.8,0]], dtype=np.float64)
v0 = np.array([[0,0,0],[0,0.6,0],[-0.4,-0.3,0]], dtype=np.float64)
M  = m.sum()
x0 -= (m[:,None]*x0).sum(0)/M
v0 -= (m[:,None]*v0).sum(0)/M
tr, pr, vr, _ = simulate_rebound_ias15(x0, v0, m, cfg.G, 100.0, 5000)
max_r = np.max(np.linalg.norm(pr[-1], axis=1))
print(f"ias15 final max body distance from origin: {max_r:.2f} AU")
print(f"Ejection by ias15: {'YES -- physically unstable IC' if max_r > 10 else 'NO -- SIMON is causing it'}")