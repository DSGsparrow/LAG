# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceExpertDataset(Dataset):
    def __init__(self, root_dir, window_size=10, stride=1):
        self.samples = []
        self.window_size = window_size
        self.stride = stride

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".npz"):
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with np.load(fpath, allow_pickle=False) as data:
                            states = data["states"]
                            actions = data["actions"]
                            if (
                                states.ndim == 2 and actions.ndim == 2 and
                                states.shape[0] == actions.shape[0] and
                                states.shape[0] >= window_size
                            ):
                                for i in range(0, len(states) - window_size + 1, stride):
                                    state_seq = states[i:i+window_size]
                                    action = actions[i+window_size-1]
                                    self.samples.append((state_seq, action))
                    except Exception as e:
                        print(f"⚠️ 读取失败: {fpath}, 原因: {e}")

        print(f"✅ 加载完成样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state_seq, action = self.samples[idx]
        state_tensor = torch.tensor(state_seq, dtype=torch.float32)  # [T, obs_dim]
        action_tensor = torch.tensor(action, dtype=torch.float32)    # [act_dim]
        return state_tensor, action_tensor



class SimpleExpertDataset(Dataset):
    def __init__(self, root_dir):
        self.state_list = []
        self.action_list = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".npz"):
                    try:
                        data = np.load(os.path.join(dirpath, fname))
                        states = data["states"]
                        actions = data["actions"]
                        if (
                            states.ndim == 2 and actions.ndim == 2 and
                            states.shape[0] == actions.shape[0]
                        ):
                            self.state_list.append(states)
                            self.action_list.append(actions)
                    except Exception as e:
                        print(f"⚠️ 读取失败: {fname}, 原因: {e}")

        self.states = np.concatenate(self.state_list, axis=0)
        self.actions = np.concatenate(self.action_list, axis=0)
        print(f"✅ 加载完成，样本数: {len(self.states)}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return state, action
