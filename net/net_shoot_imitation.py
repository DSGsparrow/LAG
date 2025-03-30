import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ========== 1. MLPBase（特征提取） ==========
class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBase, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)

# ========== 2. GRULayer ==========
class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(1)
        x, _ = self.gru(x)
        return x.squeeze(1)

# ========== 3. ACTLayer（连续输出 5 维） ==========
class ACTLayer(nn.Module):
    def __init__(self, input_dim):
        super(ACTLayer, self).__init__()
        self.output = nn.Linear(input_dim, 5)  # 输出5维连续动作

    def forward(self, x):
        raw = self.output(x)
        # 映射到合法范围（仅在需要时）
        aileron = torch.tanh(raw[:, 0:1])                 # [-1, 1]
        elevator = torch.tanh(raw[:, 1:2])                # [-1, 1]
        rudder = torch.tanh(raw[:, 2:3])                  # [-1, 1]
        throttle = torch.sigmoid(raw[:, 3:4]) * 0.5 + 0.4 # [0.4, 0.9]
        shoot = torch.sigmoid(raw[:, 4:5])                # [0, 1]
        return torch.cat([aileron, elevator, rudder, throttle, shoot], dim=-1)

# ========== 4. CustomPolicy ==========
class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super(CustomPolicy, self).__init__(observation_space, features_dim=128)
        input_dim = observation_space.shape[0]
        hidden_dim = 128

        self.feature_extractor = MLPBase(input_dim, hidden_dim)
        self.gru_layer = GRULayer(hidden_dim, hidden_dim)
        self.act_layer = ACTLayer(hidden_dim)  # 不再需要 action_dim 参数

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.gru_layer(features)
        return features  # 给 SB3 用于 actor 和 critic 分别处理

    def get_action(self, x):
        # 如果你想在外部直接取动作
        features = self.forward(x)
        return self.act_layer(features)
