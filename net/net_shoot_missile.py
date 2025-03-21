import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



# ========== 2. MLPBase（特征提取） ==========
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


# ========== 3. GRULayer（时间序列建模） ==========
class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, x):
        if x.dim() == 2:  # (batch, features)
            x = x.unsqueeze(1)  # 变成 (batch, 1, features)，保证 GRU 兼容
        elif x.dim() == 4:  # (batch, 1, seq_len, features)
            x = x.squeeze(1)  # 去掉多余的 batch 维度
        x, _ = self.gru(x)  # GRU 处理
        return x.squeeze(1)  # (batch, features)


# ========== 4. ACTLayer（动作决策层） ==========
class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ACTLayer, self).__init__()

        action_dims = list(action_dim.nvec)  # 转换成 Python list 以确保兼容性

        # 创建独立的输出层，每个维度一个 Linear 层
        self.action_heads = nn.ModuleList([nn.Linear(input_dim, dim) for dim in action_dims])

    def forward(self, x):
        return [head(x) for head in self.action_heads]


# ========== 5. 自定义策略网络（Feature + GRU + Action） ==========
class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, action_dim):
        super(CustomPolicy, self).__init__(observation_space, features_dim=128)

        input_dim = observation_space.shape[0]
        hidden_dim = 128  # 隐藏层大小
        self.feature_extractor = MLPBase(input_dim, hidden_dim)
        self.gru_layer = GRULayer(hidden_dim, hidden_dim)  # GRU 处理时间序列
        self.act_layer = ACTLayer(hidden_dim, action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.gru_layer(features)
        return features  # 只返回 Tensor，不要返回 Tuple