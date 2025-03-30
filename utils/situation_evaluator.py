import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# **1. 读取 JSON 数据**
def load_data(json_file):
    data = []
    scores = []
    with open(json_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # 提取输入特征
            features = [
                entry["distance"],
                entry["angle"],
                entry["alt"],
                entry["speed"],
                # 1 if entry["success"] else 0,  # 布尔值转数值
                # entry["reward"],
                # entry["total_steps"]
            ]

            # 增加高分样本
            if entry["success"]:
                for i in range(50):
                    data.append(features)
                    scores.append(entry["situation_score"])  # 目标值（分数）
            else:
                data.append(features)
                scores.append(entry["situation_score"])  # 目标值（分数）

    return np.array(data, dtype=np.float32), np.array(scores, dtype=np.float32)


# **2. 定义神经网络**
class SituationNet(nn.Module):
    def __init__(self):
        super(SituationNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 线性回归输出
        )

    def forward(self, x):
        return self.model(x)


# **3. 定义加权损失函数**
class WeightedMSELoss(nn.Module):
    def __init__(self, high_weight=5):  # 适当降低权重
        super(WeightedMSELoss, self).__init__()
        self.high_weight = high_weight

    def forward(self, predictions, targets):
        weights = torch.where(targets > 1.5, self.high_weight, 1.0)  # 高分（>1.5）的样本权重更大
        loss = weights * (predictions - targets) ** 2
        return loss.mean()


# **4. 预测函数（可从外部调用）**
def predict_situation(input_data, model_path="situation_model.pth", scaler_path="scaler.npy"):
    """
    输入：战场态势数据字典
    输出：预测的态势评分
    """
    # **加载标准化参数**
    scaler_mean, scaler_std = np.load(scaler_path, allow_pickle=True)
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_std

    # **加载模型**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SituationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # **预处理输入数据**
    input_features = np.array([
        input_data["distance"],
        input_data["angle"],
        input_data["alt"],
        input_data["speed"],
        # 1 if input_data["success"] else 0,
        # input_data["reward"],
        # input_data["total_steps"]
    ], dtype=np.float32).reshape(1, -1)

    input_scaled = torch.tensor(scaler.transform(input_features), dtype=torch.float32).to(device)

    # **模型预测**
    with torch.no_grad():
        prediction = model(input_scaled).item()

    return prediction


# **主程序：训练模型**
if __name__ == "__main__":
    model_path = "trained_model/shoot_prediction/situation_model2.pth"
    scaler_path = "trained_model/shoot_prediction/scaler2.npy"


    # **加载数据**
    X, y = load_data("./test_result/dodge_test/evaluated_results_all.json")

    # **数据归一化**
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # **保存归一化参数**
    np.save(scaler_path, [scaler.mean_, scaler.scale_])

    # **划分训练集和测试集**
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # **转换为 PyTorch Tensor**
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)

    # **批量训练**
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # **初始化模型**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SituationNet().to(device)
    criterion = WeightedMSELoss(high_weight=10)  # 降低高分样本的权重
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    # **训练模型**
    num_epochs = 100
    patience = 10
    best_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # **验证**
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # **Early Stopping**
        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "trained_model/shoot_prediction/situation_model.pth")  # 保存最优模型
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    # **测试模型**
    model.load_state_dict(torch.load("trained_model/shoot_prediction/situation_model.pth"))  # 加载最佳模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"最终测试集 Loss: {test_loss:.4f}")

    # **测试预测**
    test_input = {
        "distance": 500,
        "angle": 30,
        "alt": 1000,
        "speed": 400,
        # "success": True,
        # "reward": 820.314,
        # "total_steps": 300
    }

    predicted_score = predict_situation(test_input, model_path, scaler_path)
    print(f"预测态势评分: {predicted_score:.4f}")



# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
#
#
# # **1. 读取 JSON 数据**
# def load_data(json_file):
#     data = []
#     scores = []
#     with open(json_file, 'r') as f:
#         for line in f:
#             entry = json.loads(line)
#             # 提取输入特征
#             features = [
#                 entry["distance"],
#                 entry["angle"],
#                 entry["alt"],
#                 entry["speed"],
#                 1 if entry["success"] else 0,  # 布尔值转数值
#                 entry["reward"],
#                 entry["total_steps"]
#             ]
#
#             # 增加高分样本
#             if entry["success"]:
#                 for i in range(10):
#                     data.append(features)
#                     scores.append(entry["situation_score"])  # 目标值（分数）
#             else:
#                 data.append(features)
#                 scores.append(entry["situation_score"])  # 目标值（分数）
#
#     return np.array(data, dtype=np.float32), np.array(scores, dtype=np.float32)
#
#
# # **2. 加载数据**
# X, y = load_data("./test_result/dodge_test/merged_evaluation.json")
#
# # **3. 数据归一化**
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # **4. 划分训练集和测试集**
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # # 放大目标值（训练时）
# # y_train_scaled = y_train * 10
# # y_test_scaled = y_test * 10
#
# # **5. 转换为 PyTorch Tensor**
# X_train_tensor = torch.tensor(X_train)
# y_train_tensor = torch.tensor(y_train).view(-1, 1)
# X_test_tensor = torch.tensor(X_test)
# y_test_tensor = torch.tensor(y_test).view(-1, 1)
#
# # **6. 使用 DataLoader 进行批量训练**
# batch_size = 64
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#
# # **7. 定义 MLP 神经网络**
# class SituationNet(nn.Module):
#     def __init__(self):
#         super(SituationNet, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(7, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1)  # 线性回归输出
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# # **8. 初始化模型**
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SituationNet().to(device)
#
# class WeightedMSELoss(nn.Module):
#     def __init__(self, high_weight=10):
#         super(WeightedMSELoss, self).__init__()
#         self.high_weight = high_weight
#
#     def forward(self, predictions, targets):
#         weights = torch.where(targets > 1.5, self.high_weight, 1.0)  # 高分（>1.5）的样本权重更大
#         loss = weights * (predictions - targets) ** 2
#         return loss.mean()
#
# criterion = WeightedMSELoss(high_weight=10)  # 高分的损失加权 10 倍
#
#
# # **9. 定义损失函数和优化器**
# # criterion = nn.MSELoss()  # 均方误差
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # L2 正则化
#
# # **10. 训练模型**
# num_epochs = 100
# patience = 10  # Early Stopping
# best_loss = float("inf")
# early_stopping_counter = 0
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     # **验证**
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_X, batch_y in test_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             val_loss += loss.item()
#
#     val_loss /= len(test_loader)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#     # **Early Stopping**
#     if val_loss < best_loss:
#         best_loss = val_loss
#         early_stopping_counter = 0
#         torch.save(model.state_dict(), "situation_model.pth")  # 保存最优模型
#     else:
#         early_stopping_counter += 1
#         if early_stopping_counter >= patience:
#             print("Early stopping triggered.")
#             break
#
# # **11. 评估模型**
# model.load_state_dict(torch.load("situation_model.pth"))  # 加载最佳模型
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#         test_loss += loss.item()
#
# test_loss /= len(test_loader)
# print(f"最终测试集 Loss: {test_loss:.4f}")
#
#
# # **12. 预测函数**
# def predict_situation(input_data):
#     """
#     输入：战场态势数据字典
#     输出：预测的态势评分
#     """
#     input_features = np.array([
#         input_data["distance"],
#         input_data["angle"],
#         input_data["alt"],
#         input_data["speed"],
#         1 if input_data["success"] else 0,
#         input_data["reward"],
#         input_data["total_steps"]
#     ], dtype=np.float32).reshape(1, -1)
#
#     input_scaled = torch.tensor(scaler.transform(input_features), dtype=torch.float32).to(device)
#
#     model.eval()
#     with torch.no_grad():
#         prediction = model(input_scaled).item()
#
#     return prediction
#
#
# # **测试预测**
# test_input = {
#     "distance": 500,
#     "angle": 30,
#     "alt": 1000,
#     "speed": 400,
#     "success": True,
#     "reward": 820.314,
#     "total_steps": 300
# }
#
# predicted_score = predict_situation(test_input)
# print(f"预测态势评分: {predicted_score:.4f}")
#
#
# # if __name__ == "__main__":