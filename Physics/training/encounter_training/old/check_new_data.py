import numpy as np

f = "encounter_data_zone3_v3_dt0p005_batch001.npz"
data = np.load(f)

print("FIELDS:")
print(data.files)

required = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement"
]

print("\nMISSING:")
print([k for k in required if k not in data.files])

print("\nSHAPES:")
for k in required:
    print(k, data[k].shape, data[k].dtype)

print("\nRANGES:")
for k in ["r_AU", "r_soft", "log_dt", "v_rad_norm", "v_tan_norm", "c_opt", "log_c_opt"]:
    x = data[k]
    print(f"{k:12s} min={x.min(): .6f}  med={np.median(x): .6f}  max={x.max(): .6f}")

print("\nCHECKS:")
print("r in Zone 3:", np.all((data["r_AU"] > 0.052) & (data["r_AU"] < 0.148)))
print("v_tan_norm nonnegative:", np.all(data["v_tan_norm"] >= 0))
print("finite all:", all(np.all(np.isfinite(data[k])) for k in required))