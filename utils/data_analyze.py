import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ========== 1. 加载数据 ==========
# 替换为你的真实数据文件路径
# 支持 CSV / JSON / JSONL 等格式（你只需保证字段为：distance, angle, alt, speed, success）
df = pd.read_csv("./test_result/dodge_test/evaluated_results_all.json")  # 或 pd.read_json("your_data.json")

# 将 success 转为 0/1
df["success"] = df["success"].astype(int)

# ========== 2. 决策树分析特征重要性 ==========
X = df[['distance', 'angle', 'alt', 'speed']]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 可视化特征重要性
importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 可选：输出分类性能
print("决策树分类报告：\n")
print(classification_report(y_test, clf.predict(X_test)))

# ========== 3. 聚类命中成功区域 ==========
success_df = df[df['success'] == 1].copy()

scaler = StandardScaler()
X_success_scaled = scaler.fit_transform(success_df[['distance', 'angle', 'alt', 'speed']])

# 聚成 3 类（可调整）
kmeans = KMeans(n_clusters=3, random_state=42)
success_df['cluster'] = kmeans.fit_predict(X_success_scaled)

# 可视化聚类结果（选两个特征维度）
plt.figure(figsize=(8, 6))
sns.scatterplot(data=success_df, x='distance', y='alt', hue='cluster', palette='tab10')
plt.title("Clusters of Successful Hits (Distance vs Alt)")
plt.tight_layout()
plt.savefig("clusters.png")
plt.close()

# ========== 4. 热力图查看密集命中区域 ==========
plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=success_df,
    x="distance",
    y="angle",
    fill=True,
    cmap="Reds",
    thresh=0.05,
)
plt.title("Density of Successful Hits (Distance vs Angle)")
plt.tight_layout()
plt.savefig("density_heatmap.png")
plt.close()

print("分析完成！输出图已保存为：feature_importance.png、clusters.png、density_heatmap.png")
