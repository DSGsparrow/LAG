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

def train_imitation_and_export(data_dir, env, zip_path="imitation_pretrained.zip", patience=5):
    dataset = ExpertDataset(data_dir)

    # ✅ 拆分训练/验证集
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.1, shuffle=True)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    obs_dim = dataset[0][0].shape[0]
    imit_model = ImitationPolicy(obs_dim)
    optim = torch.optim.Adam(imit_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):  # 最多训练 100 轮
        imit_model.train()
        total_loss = 0
        for obs_batch, act_batch in train_loader:
            pred = imit_model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * obs_batch.size(0)
        train_loss = total_loss / len(train_dataset)

        # 验证集 loss
        imit_model.eval()
        with torch.no_grad():
            val_loss = 0
            for obs_batch, act_batch in val_loader:
                pred = imit_model(obs_batch)
                loss = loss_fn(pred, act_batch)
                val_loss += loss.item() * obs_batch.size(0)
            val_loss = val_loss / len(val_dataset)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_sd = imit_model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ 早停触发：验证集 {patience} 次未提升，提前停止训练")
                break

    best_model_sd = torch.load('imitation_pretrained_pytorch.pt')

    # 恢复最佳参数
    imit_model.load_state_dict(best_model_sd)

    # ✅ 保存 PyTorch 原始模型（防止 zip 失败）
    torch.save(best_model_sd, zip_path.replace(".zip", "_pytorch.pt"))
    print(f"📦 已保存为原始 PyTorch 模型: {zip_path.replace('.zip', '_pytorch.pt')}")

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
    ppo.policy.features_extractor.gru_layer.load_state_dict({
        k.replace("gru.", ""): v for k, v in imit_sd.items() if "gru" in k
    }, strict=False)

    # 强化学习阶段会重新初始化动作头，所以这里只迁移特征提取器部分
    ppo.save(zip_path)
    print(f"✅ 已保存为 SB3 PPO 模型: {zip_path}")



env = SB3SingleCombatEnv(0, config_name='1v1/DodgeMissile/HierarchyVsBaselineSelf')
train_imitation_and_export(data_dir="render_train/dodge2/imitation", env=env)
