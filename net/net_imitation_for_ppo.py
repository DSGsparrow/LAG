import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):  # [B, T, obs_dim] or [B, obs_dim]
        if x.dim() == 2:  # [B, D]
            x = x.unsqueeze(1)  # → [B, 1, D]
        B, T, D = x.shape
        x = x.view(B * T, D)
        out = self.network(x)
        return out.view(B, T, -1)  # [B, T, H]

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):  # x: [B, T, H]
        out, _ = self.gru(x)
        return out[:, -1]  # 取最后一个隐藏状态作为整体序列表示

class PPOGRUPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__(observation_space, features_dim=hidden_dim)
        input_dim = observation_space.shape[-1]  # 单帧状态维度

        self.mlp = MLPBase(input_dim, hidden_dim)
        self.gru = GRULayer(hidden_dim, hidden_dim)

        # Actor 从模仿学习中加载后微调
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic 自行强化训练
        self.critic = nn.Linear(hidden_dim, 1)

        # 固定 log_std 可训练
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):  # x: [B, T, obs_dim]
        features = self.mlp(x)        # → [B, T, H]
        context = self.gru(features)  # → [B, H]
        return context

    def act(self, obs):  # obs: [B, T, obs_dim]
        x = self.forward(obs)
        mu = self.actor(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):  # obs: [B, T, obs_dim], actions: [B, act_dim]
        x = self.forward(obs)
        mu = self.actor(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return log_prob, entropy, value


class PPOMLPPolicy(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule=None,
        net_arch=None,
        activation_fn=None,
        *args,
        **kwargs
    ):
        super().__init__(observation_space, features_dim=128)
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0] if action_space is not None else 4
        hidden_dim = 128  # 可从 kwargs 中提取自定义值

        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):  # x: [B, obs_dim]
        return self.feature_extractor(x)  # [B, hidden_dim]

    def act(self, obs):  # obs: [B, obs_dim]
        x = self.forward(obs)
        mu = self.actor(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        x = self.forward(obs)
        mu = self.actor(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x).squeeze(-1)
        return log_prob, entropy, value



class GRUPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):  # x: [B, T, obs_dim]
        B, T, D = x.shape
        x = self.mlp(x.view(B * T, D)).view(B, T, -1)  # x: [B, T, hidden_dim]
        x, _ = self.gru(x)                             # x: [B, T, hidden_dim]
        out = self.actor(x[:, -1])                     # 只用最后时刻特征
        return out                                     # [B, act_dim]


class MLPFeatureExtractor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):  # x: [B, obs_dim]
        return self.feature_net(x)  # [B, hidden_dim]


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.feature_extractor = MLPFeatureExtractor(obs_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):  # x: [B, obs_dim]
        features = self.feature_extractor(x)
        return self.actor(features)  # [B, act_dim]
