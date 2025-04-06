import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# PPO 调用 .pt 的网络模型
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


# ========== 4. CustomPolicy ==========
class CustomImitationPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super(CustomImitationPolicy, self).__init__(observation_space, features_dim=128)
        input_dim = observation_space.shape[0]
        hidden_dim = 128

        self.feature_extractor = MLPBase(input_dim, hidden_dim)
        self.gru = GRULayer(hidden_dim, hidden_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.gru(features)
        return features  # 给 SB3 用于 actor 和 critic 分别处理
