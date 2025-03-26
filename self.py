import matplotlib.pyplot as plt
import numpy as np

def distance_reward_linear(distance, center=7000, max_distance=10000, sigma=1000):
    """
    分段连续距离奖励函数：
    - <= center: 高斯函数（下降慢）
    - > center: 线性从 1 降到 -1（直观明确）
    - > max_distance: 固定为 -1
    """
    if distance <= center:
        # 高斯函数下降慢一点：sigma 调大
        return np.exp(-((distance - center) ** 2) / (2 * sigma ** 2))
    elif distance <= max_distance:
        # 线性递减：从 1 到 -1
        ratio = (distance - center) / (max_distance - center)  # 0 到 1
        return 1 - 2 * ratio
    else:
        return -1.0



xs = np.linspace(5000, 10500, 500)
ys = [distance_reward_linear(x) for x in xs]

plt.plot(xs, ys)
plt.axvline(9500, color='gray', linestyle='--', label='center')
plt.axvline(10000, color='red', linestyle='--', label='max_distance')
plt.title("Simple Distance-Based Reward (Gaussian + Linear)")
plt.xlabel("Distance (m)")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
