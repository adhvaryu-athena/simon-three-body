import numpy as np

data = np.load(r"C:\Aarush\Physics\training\encounter_training\old dataset1\encounter_data.npz")

print("Fields:", data.files)
print("Total samples:", len(data["r_AU"]))

# Check if zone field exists
if "zone" in data.files:
    zones = data["zone"]
    print(f"Zone 2 samples: {(zones == 2).sum()}")
    print(f"Zone 3 samples: {(zones == 3).sum()}")
    z3 = zones == 3
else:
    # No zone field -- use r to separate
    r = data["r_AU"]
    z3 = (r >= 0.05) & (r < 0.15)
    print(f"Zone 3 by r (0.05-0.15 AU): {z3.sum()} samples")
    print(f"Outside Zone 3: {(~z3).sum()} samples")

# Per-dt statistics for Zone 3
dt_yr = np.exp(data["log_dt"][z3])
c_opt = data["c_opt"][z3]
c_ana = data["c_ana"][z3]
impr  = data["improvement"][z3]

dt_vals = sorted(set(round(float(v), 4) for v in dt_yr))
print(f"\nZone 3 per-dt statistics:")
print(f"{'dt':>7} | {'n':>6} | {'c_opt_med':>10} | {'c_ana_med':>10} | {'impr_med':>10}")
print("-" * 55)
for dt in dt_vals:
    mask = np.abs(dt_yr - dt) < 1e-5
    print(f"  {dt:5.3f} | {mask.sum():6d} | "
          f"{float(np.median(c_opt[mask])):10.5f} | "
          f"{float(np.median(c_ana[mask])):10.7f} | "
          f"{float(np.median(impr[mask])):10.2%}")