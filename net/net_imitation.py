import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from gym.spaces import MultiDiscrete
from sklearn.model_selection import train_test_split

from net.net_shoot_imitation import MLPBase, GRULayer

# 单纯模仿学习的网络
# 模仿学习策略网络（直接输出连续动作）
class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.feature_extractor = MLPBase(obs_dim, 128)
        self.gru = GRULayer(128, 128)
        self.action_head = nn.Linear(128, 5)  # 连续动作输出 5维

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.gru(feat)
        raw = self.action_head(feat)
        # 前3维 tanh [-1, 1]，第4维 sigmoid * 0.5 + 0.4 => [0.4, 0.9]，第5维 sigmoid => [0,1]
        aileron = torch.tanh(raw[:, 0:1])
        elevator = torch.tanh(raw[:, 1:2])
        rudder = torch.tanh(raw[:, 2:3])
        throttle = torch.sigmoid(raw[:, 3:4]) * 0.5 + 0.4
        shoot = torch.sigmoid(raw[:, 4:5])
        return torch.cat([aileron, elevator, rudder, throttle, shoot], dim=-1)