import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ===== 模仿学习结构复制 =====
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
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(1)
        x, _ = self.gru(x)
        return x.squeeze(1)

# ===== 自定义特征提取器+ActorCritic策略网络 =====
class CustomImitationShootBackPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.mlp = MLPBase(obs_dim, 128)
        self.gru = GRULayer(128, features_dim)
        self.output_dim = features_dim  # 最终输出维度

    def forward(self, obs):
        x = self.mlp(obs)
        x = self.gru(x)
        return x

class CustomActorCriticShootBackPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 替换特征提取器为我们的网络
        # self.features_extractor = CustomImitationShootBackPolicy(self.observation_space)

        # 自定义 actor 和 critic head
        latent_dim = self.features_extractor.output_dim
        self.action_net = nn.Linear(latent_dim, self.action_space.shape[0])
        self.value_net = nn.Linear(latent_dim, 1)

        # 连续动作用到的 log_std
        self.log_std = nn.Parameter(torch.zeros(1, self.action_space.shape[0]))

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.constant_(self.action_net.bias, 0)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)
        nn.init.constant_(self.value_net.bias, 0)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(features)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent):
        mean_actions = self.action_net(latent)
        log_std = self.log_std.expand_as(mean_actions)
        return self.action_dist.proba_distribution(mean_actions, log_std)

    def _predict(self, observation, deterministic=False):
        features = self.extract_features(observation)
        mean_actions = self.action_net(features)
        if deterministic:
            return mean_actions
        log_std = self.log_std.expand_as(mean_actions)
        return self.action_dist.proba_distribution(mean_actions, log_std).sample()