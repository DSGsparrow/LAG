import numpy as np
import matplotlib.pyplot as plt

# 定义函数
reward_func = lambda R: 1 * (R < 5) + \
                        (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + \
                        np.clip(np.exp(-0.16 * R), 0, 0.2)

# 绘制函数图像
R = np.linspace(0, 50, 500)
reward = reward_func(R)

plt.plot(R, reward, label='Reward Function')
plt.xlabel('R')
plt.ylabel('Reward')
plt.title('Reward vs. Distance (R)')
plt.grid()
plt.legend()
plt.show()
