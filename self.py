import numpy as np
import matplotlib.pyplot as plt

# 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)

def speed_reward(distance):
    reward = 0
    if distance > 10000:
        # 太远
        reward -= 1
    elif distance <= 10000:

        if distance <= 9000:
            # 高斯函数下降慢一点：sigma 调大
            bonus = np.exp(-((distance - 9000) ** 2) / (2 * 1000 ** 2))
        else:  # if distance <= 10000:
            # 线性递减：从 1 到 -1
            ratio = (distance - 9000) / (10000 - 9000)  # 0 到 1
            bonus = 1 - 2 * ratio

        reward += 1 * bonus



vs = np.linspace(6000, 11000, 500)
rs = [speed_reward(v) for v in vs]

plt.plot(vs, rs)
plt.title("Speed Reward Curve")
plt.xlabel("Speed (mh)")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
