import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Bernoulli
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter

# ========== 初始化工具 ==========
def init(m, weight_init, bias_init, gain=1):
    weight_init(m.weight.data, gain=gain)
    bias_init(m.bias.data)
    return m

# ========== 特征提取器（MLP + GRU） ==========
class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(1)
        x, _ = self.gru(x)
        return x.squeeze(1)

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        hidden_dim = features_dim
        self.mlp = MLPBase(input_dim, hidden_dim)
        self.gru = GRULayer(hidden_dim, hidden_dim)

    def forward(self, obs):
        return self.gru(self.mlp(obs))

# ========== BetaShootBernoulli ==========
class FixedBernoulli(Bernoulli):
    def log_prob(self, value):
        return super().log_prob(value).view(-1)

class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super().__init__()
        self.net = init(nn.Linear(num_inputs, num_outputs), nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, alpha0=2.0, beta0=5.0):
        x = self.constraint(self.net(x))
        x = 100 - self.constraint(100 - x)
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        p = (alpha + alpha0) / (alpha + alpha0 + beta + beta0)
        return FixedBernoulli(probs=p), p

    @property
    def output_size(self):
        return self._num_outputs

# ========== CustomActorCriticPolicy ==========
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = CustomExtractor
        kwargs["features_extractor_kwargs"] = dict(features_dim=128)
        super().__init__(*args, **kwargs)

        hidden_dim = self.mlp_extractor.latent_dim_pi

        # 主动作 MultiDiscrete([3,5,3])
        self.multi_action_net = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in self.action_space.nvec[:-1]]
        )

        # 发射动作头：输出 alpha, beta
        self.fire_head = BetaShootBernoulli(hidden_dim, 2)
        self.value_net = nn.Linear(hidden_dim, 1)

        # Tensorboard 记录器（可选）
        self.writer = SummaryWriter(log_dir="./ppo_fire_debug")
        self.debug_step = 0

    def compute_prior_from_obs(self, obs):
        AO = obs[:, 11] / np.pi * 180
        TA = obs[:, 12] / np.pi * 180
        distance = obs[:, 13] * 10000

        alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
        beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
        alpha0[distance <= 12000] = 6
        alpha0[distance <= 8000] = 10
        beta0[AO <= 45] = 6
        beta0[AO <= 22.5] = 3

        # distance_factor = torch.clamp((distance - 6000) / 4000, min=0.0, max=1.0)
        # ao_factor = AO.abs() / 45.0
        # ta_factor = TA.abs() / 90.0
        #
        # alpha0 = 2.0 + (1.0 - distance_factor) * 1.5
        # beta0 = 2.0 + 3.0 * distance_factor + 1.0 * ao_factor + 0.5 * ta_factor

        return alpha0, beta0

    def _get_action_dist_from_latent(self, latent_pi, alpha0=2.0, beta0=5.0, log_debug=False, obs=None):
        logits_list = [head(latent_pi) for head in self.multi_action_net]
        multi_cat_dist = MultiCategoricalDistribution(self.action_space.nvec[:-1])
        multi_cat_dist.distribution = [Categorical(logits=logits) for logits in logits_list]

        fire_dist, fire_prob = self.fire_head(latent_pi, alpha0=alpha0, beta0=beta0)

        if log_debug and obs is not None:
            with torch.no_grad():
                self.writer.add_scalar("debug/fire_prob_mean", fire_prob.mean().item(), self.debug_step)
                self.writer.add_scalar("debug/distance_mean", (obs[:, 13] * 10000).mean().item(), self.debug_step)
                self.writer.add_scalar("debug/AO_mean", (obs[:, 11] * 180 / np.pi).abs().mean().item(), self.debug_step)
                self.writer.add_scalar("debug/TA_mean", (obs[:, 12] * 180 / np.pi).abs().mean().item(), self.debug_step)
                self.debug_step += 1

        return multi_cat_dist, fire_dist

    def forward(self, obs, deterministic=False):
        alpha0, beta0 = self.compute_prior_from_obs(obs)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        multi_dist, fire_dist = self._get_action_dist_from_latent(latent_pi, alpha0, beta0, log_debug=True, obs=obs)
        main_actions = multi_dist.get_actions(deterministic=deterministic)
        fire_action = fire_dist.sample() if not deterministic else (fire_dist.probs > 0.5).float()
        action = torch.cat([main_actions.float(), fire_action.unsqueeze(-1)], dim=-1)
        value = self.value_net(latent_vf)
        log_prob = multi_dist.log_prob(main_actions) + fire_dist.log_prob(fire_action)
        return action, value, log_prob

    def _predict(self, obs, deterministic=False):
        alpha0, beta0 = self.compute_prior_from_obs(obs)
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        multi_dist, fire_dist = self._get_action_dist_from_latent(latent_pi, alpha0, beta0)
        main_actions = multi_dist.get_actions(deterministic=deterministic)
        fire_action = fire_dist.sample() if not deterministic else (fire_dist.probs > 0.5).float()
        return torch.cat([main_actions.float(), fire_action.unsqueeze(-1)], dim=-1)

    def evaluate_actions(self, obs, actions):
        alpha0, beta0 = self.compute_prior_from_obs(obs)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        multi_dist, fire_dist = self._get_action_dist_from_latent(latent_pi, alpha0, beta0)
        main_actions = actions[:, :-1].long()
        fire_action = actions[:, -1]
        log_prob = multi_dist.log_prob(main_actions) + fire_dist.log_prob(fire_action)
        entropy = multi_dist.entropy().sum(-1) + fire_dist.entropy()
        value = self.value_net(latent_vf)
        return value, log_prob, entropy