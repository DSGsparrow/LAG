import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.situation_evaluator import SituationNet,predict_situation

# **测试预测**
test_input = {
    "distance": 8000.0, "angle": 0.0, "alt": 1111.1111111111095, "speed": 928.5714285714286,
    # "success": True,
    # "reward": 820.314,
    # "total_steps": 300
}

model_path = "trained_model/shoot_prediction/situation_model2.pth"
scaler_path = "trained_model/shoot_prediction/scaler2.npy"
predicted_score = predict_situation(test_input, model_path, scaler_path)
print(f"预测态势评分: {predicted_score:.4f}")