from utils.situation_evaluator import predict_situation

data = \
    {"distance": 8000.0, "angle": 10.285714285714286, "alt": -500.0, "speed": 928.5714285714286, "success": False,
     "reward": 1293.04889864, "total_steps": 296, "counter": 87, "state": "null", "situation_score": 2.0833820665177685}

prediction = predict_situation(data)
print(prediction)