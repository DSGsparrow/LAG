import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from gym.spaces import MultiDiscrete

from net.net_shoot_missile import MLPBase, GRULayer, ACTLayer, CustomPolicy
from adapter.adapter_dodge_missile import SB3SingleCombatEnv


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

# 专家数据集
class ExpertDataset(Dataset):
    def __init__(self, data_dir):
        self.obs = []
        self.actions = []
        for file in os.listdir(data_dir):
            if file.endswith(".npz"):
                data = np.load(os.path.join(data_dir, file))
                self.obs.append(data["obs"])
                self.actions.append(data["action"])
        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return torch.tensor(self.obs[idx], dtype=torch.float32), \
               torch.tensor(self.actions[idx], dtype=torch.float32)

# 训练并导出 PPO zip 模型

def train_imitation_and_export(data_dir, env, zip_path="trained_model/imitation_shoot/imitation_pretrained.zip"):
    dataset = ExpertDataset(data_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    obs_dim = dataset[0][0].shape[0]
    imit_model = ImitationPolicy(obs_dim)
    optim = torch.optim.Adam(imit_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        total_loss = 0
        for obs_batch, act_batch in loader:
            pred = imit_model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * obs_batch.size(0)
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataset):.4f}")

    # 初始化 SB3 PPO 模型
    policy_kwargs = dict(
        features_extractor_class=CustomPolicy
    )
    ppo = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    # 拷贝参数（只迁移 feature + gru）
    imit_sd = imit_model.state_dict()
    ppo.policy.features_extractor.feature_extractor.load_state_dict({
        k.replace("feature_extractor.", ""): v for k, v in imit_sd.items() if "feature_extractor" in k
    }, strict=False)
    ppo.policy.features_extractor.gru.load_state_dict({
        k.replace("gru.", ""): v for k, v in imit_sd.items() if "gru" in k
    }, strict=False)

    # 强化学习阶段会重新初始化动作头，所以这里只迁移特征提取器部分
    ppo.save(zip_path)
    print(f"✅ 已保存为 SB3 PPO 模型: {zip_path}")



env = SB3SingleCombatEnv(0, config_name='1v1/DodgeMissile/HierarchyVsBaselineSelf')
train_imitation_and_export(data_dir="render_train/dodge2/imitation", env=env)
