import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PPOAgent:
    def __init__(self, policy, vec_env, model_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.policy = policy.to(self.device)
        self.old_policy = policy.__class__().to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        if model_path and os.path.exists(model_path):
            print(f"✅ 加载预训练模型: {model_path}")
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.old_policy.load_state_dict(torch.load(model_path, map_location=self.device))

        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(obs)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def collect_rollout(self, rollout_len=2048):
        obs = self.vec_env.reset()
        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []

        for _ in range(rollout_len):
            action, logp, value = self.select_action(obs)
            next_obs, rewards, dones, infos = self.vec_env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp)
            val_list.append(value)
            rew_list.append(rewards)
            done_list.append(dones)

            obs = next_obs

        obs_array = np.array(obs_list)
        act_array = np.array(act_list)
        logp_array = np.array(logp_list)
        val_array = np.array(val_list)
        rew_array = np.array(rew_list)
        done_array = np.array(done_list)

        with torch.no_grad():
            _, _, next_value = self.select_action(obs)

        advantages = np.zeros_like(rew_array)
        returns = np.zeros_like(rew_array)
        for env_idx in range(self.num_envs):
            adv, ret = self.compute_gae(
                rew_array[:, env_idx],
                val_array[:, env_idx],
                done_array[:, env_idx],
                next_value[env_idx]
            )
            advantages[:, env_idx] = adv
            returns[:, env_idx] = ret

        adv_flat = advantages.flatten()
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)  # ✅ Normalize advantages

        return (
            obs_array.reshape(-1, obs_array.shape[-1]),
            act_array.reshape(-1, act_array.shape[-1]),
            logp_array.flatten(),
            returns.flatten(),
            adv_flat
        )

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = np.append(values, next_value)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = advantages + values[:-1].tolist()
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def ppo_update(self, obs, actions, log_probs_old, returns, advantages, batch_size=64, epochs=10):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_old = log_probs_old[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # ✅ use old_policy to compute old log prob
                old_log_probs, _, _ = self.old_policy.evaluate(mb_obs, mb_actions)
                new_log_probs, entropy, values = self.policy.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = ((values - mb_returns) ** 2).mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        print(f"✅ 模型保存: {path}")