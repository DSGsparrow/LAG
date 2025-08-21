import numpy as np
import matplotlib.pyplot as plt

# 参数
miss_min_reward = -1.0
half_km = 5.0

# tau 由 half_km 推出：exp(-half_km / tau) = 0.5
tau = half_km / np.log(2.0)

def exp_miss_reward(d_km):
    s_close = np.exp(-d_km / tau)
    return miss_min_reward * (1.0 - s_close)

# 距离范围 0-20 km
d = np.linspace(0, 20, 500)
r = exp_miss_reward(d)

plt.figure(figsize=(7,5))
plt.plot(d, r, label="Exponential miss reward")
plt.axvline(half_km, color='r', linestyle='--', label=f"half_km={half_km} km")
plt.axhline(miss_min_reward/2, color='g', linestyle=':', label="half reward")
plt.axhline(miss_min_reward, color='k', linestyle='--', label="min reward")
plt.title("Exponential Miss Reward Curve")
plt.xlabel("Distance to target (km)")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
