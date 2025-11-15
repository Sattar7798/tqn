# scripts/generate_scaler_minimal.py

import pickle
import numpy as np
from pathlib import Path

# Example: assuming state = [T_zone_1..5, T_out, hour_sin, hour_cos, day_sin, day_cos]
# We just define a simple hand-crafted scaler with reasonable ranges.

state_dim = 5 + 1 + 4  # 5 zones + T_out + 4 encoded time features
means = np.zeros(state_dim, dtype=np.float32)
stds = np.ones(state_dim, dtype=np.float32)

# Rough priors:
# zone temps ~ 23°C, T_out ~ 18°C, time encodings in [-1,1]
means[0:5] = 23.0       # zones
means[5]   = 18.0       # T_out
# time encodings keep mean 0, std 1
stds[0:5] = 3.0         # 3°C variability
stds[5]   = 10.0        # larger outdoor range

scaler = {
    "mean": means,
    "std": stds,
    "eps": 1e-8,
}

out_path = Path("data")
out_path.mkdir(parents=True, exist_ok=True)
with (out_path / "scaler_minimal.pkl").open("wb") as f:
    pickle.dump(scaler, f)

print("Saved minimal scaler to data/scaler_minimal.pkl")
