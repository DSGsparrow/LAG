import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from net.net_imitation_for_ppo import MLPPolicy
from utils.for_imitation.data_set import SimpleExpertDataset

from net.net_imitation_for_ppo import GRUPolicy
from utils.for_imitation.data_set import SequenceExpertDataset


def main_gru():
    # train_gru_imitation.py
    # === 配置参数 ===
    root_dir = "imitation_data"
    window_size = 10
    batch_size = 64
    hidden_dim = 128
    epochs = 200
    lr = 1e-3
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "trained_model/imitation_shoot/gru_imitation_policy.pt"

    # === 加载数据集并划分训练/验证集 ===
    dataset = SequenceExpertDataset(root_dir, window_size=window_size, stride=1)
    obs_dim = dataset[0][0].shape[-1]
    act_dim = dataset[0][1].shape[-1]
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # === 初始化模型和优化器 ===
    model = GRUPolicy(obs_dim, act_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # === 训练 + 验证 + 早停 ===
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for state_seq, action in train_loader:
            state_seq = state_seq.to(device)      # [B, T, obs_dim]
            action = action.to(device)            # [B, act_dim]
            pred_action = model(state_seq)
            loss = F.mse_loss(pred_action, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(state_seq)
        avg_train_loss = total_train_loss / train_size

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for state_seq, action in val_loader:
                state_seq = state_seq.to(device)
                action = action.to(device)
                pred_action = model(state_seq)
                loss = F.mse_loss(pred_action, action)
                total_val_loss += loss.item() * len(state_seq)
        avg_val_loss = total_val_loss / val_size

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存模型到 {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ 早停触发，训练终止")
                break


def main_mlp():
    # train_mlp_imitation.py
    # 配置参数
    root_dir = "imitation_data"
    batch_size = 256
    hidden_dim = 128
    lr = 1e-3
    epochs = 200
    patience = 10
    save_path = "mlp_imitation_policy.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集并划分训练/验证集
    dataset = SimpleExpertDataset(root_dir)
    obs_dim = dataset[0][0].shape[0]
    act_dim = dataset[0][1].shape[0]
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 初始化模型和优化器
    model = MLPPolicy(obs_dim, act_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练 + 验证 + 早停
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for state, action in train_loader:
            state, action = state.to(device), action.to(device)
            pred = model(state)
            loss = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(state)
        avg_train_loss = total_train_loss / train_size

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for state, action in val_loader:
                state, action = state.to(device), action.to(device)
                pred = model(state)
                loss = F.mse_loss(pred, action)
                total_val_loss += loss.item() * len(state)
        avg_val_loss = total_val_loss / val_size

        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存至 {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ 早停触发，训练终止")
                break


if __name__ == '__main__':
    main_gru()