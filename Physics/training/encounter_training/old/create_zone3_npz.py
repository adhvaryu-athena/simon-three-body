import numpy as np

data = np.load(
    r"C:\Aarush\Physics\training\encounter_training\old dataset1\encounter_data.npz")

z3 = data["zone"] == 3
fields = ["r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
          "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement"]

np.savez_compressed(
    r"C:\Aarush\Physics\training\encounter_training\encounter_data_zone3.npz",
    **{k: data[k][z3] for k in fields}
)
print(f"Saved {z3.sum()} Zone 3 samples")
print("Fields:", fields)