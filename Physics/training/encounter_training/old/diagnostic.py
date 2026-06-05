import numpy as np
import torch

data = np.load("encounter_data_zone3.npz")

# Check feature ranges
r_soft = data["r_soft"].astype(np.float64)
log_mi = data["log_mi"].astype(np.float64)
log_mj = data["log_mj"].astype(np.float64)
log_dt = data["log_dt"].astype(np.float64)
c_opt  = data["c_opt"].astype(np.float64)
log_c  = data["log_c_opt"].astype(np.float64)

log_rs = np.log(r_soft + 1e-30)

print("Feature ranges (these go into the NN):")
print(f"  log(r_soft): {log_rs.min():.3f} to {log_rs.max():.3f}")
print(f"  log_mi:      {log_mi.min():.3f} to {log_mi.max():.3f}")
print(f"  log_mj:      {log_mj.min():.3f} to {log_mj.max():.3f}")
print(f"  log_dt:      {log_dt.min():.3f} to {log_dt.max():.3f}")
print()
print("Target range:")
print(f"  log_c_opt:   {log_c.min():.3f} to {log_c.max():.3f}")
print(f"  c_opt:       {c_opt.min():.3f} to {c_opt.max():.3f}")
print()

# Compute what normalisation values the model stored
feat = np.stack([log_rs, log_mi, log_mj, log_dt], axis=1).astype(np.float32)
print("Normalisation (mean, std) per feature:")
for i, name in enumerate(["log_r_soft", "log_mi", "log_mj", "log_dt"]):
    print(f"  {name}: mean={feat[:,i].mean():.4f}  std={feat[:,i].std():.4f}")

# Load model and check what it stored
model_sd = torch.load("pair_correction_nn_v2.pt", map_location="cpu")
print()
print("Model stored normalisation:")
print(f"  input_mean: {model_sd['input_mean'].numpy()}")
print(f"  input_std:  {model_sd['input_std'].numpy()}")