# Re-plot with distance_bin_km = 0.5 km
import numpy as np
import matplotlib.pyplot as plt

hit_reward = 1.0
miss_base_penalty = -0.75
miss_bonus_max = 1.5

distance_mid_km = 7.0
distance_steep_km = 1.2
distance_bin_km = 0.5  # updated as requested

def logistic01(d_km, mid_km, steep_km):
    return 1.0 / (1.0 + np.exp((d_km - mid_km) / max(1e-6, steep_km)))

def reward_fn(d_m, quantized=False):
    d_km = d_m / 1000.0
    if d_km <= 0.3:
        return hit_reward
    if quantized:
        d_km = np.round(d_km / distance_bin_km) * distance_bin_km
    shape = logistic01(d_km, distance_mid_km, distance_steep_km)
    return miss_base_penalty + miss_bonus_max * shape

d_m = np.linspace(0, 15000, 2000)
r_cont = np.array([reward_fn(x, quantized=False) for x in d_m])
r_quant = np.array([reward_fn(x, quantized=True) for x in d_m])

plt.figure(figsize=(8,5))
plt.plot(d_m/1000.0, r_cont, label='Continuous', linewidth=2)
plt.plot(d_m/1000.0, r_quant, linestyle='solid', label=f'Binned ({distance_bin_km:.1f} km)', linewidth=2)
plt.axvline(x=0.3, linestyle=':', linewidth=1.5)
plt.text(0.31, hit_reward-0.05, 'Hit @ 0.3 km', fontsize=9, va='top')

plt.title('Reward vs Final Miss Distance (hit if ≤ 0.3 km)')
plt.xlabel('Final distance d (km)')
plt.ylabel('Reward R(d)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()

out_path = '/mnt/data/reward_curve_300m_logistic_bin0p5.png'
plt.show()
# plt.savefig(out_path, dpi=180)
# out_path
