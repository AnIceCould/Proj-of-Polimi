import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'dataset/Spinach_Raw_THz_Stress_Dataset_middle.csv'
try:
    df = pd.read_csv(file_path)
    print(f"成功加载数据集: {file_path}")
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}，请检查路径。")
    # 如果本地没有文件，以下代码将无法运行
    exit()

# 2. 数据预处理
# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 标签编码 (将 Normal, A_Drought 等转换为 0, 1, 2...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# 3. 特征缩放 (KNN 必须步骤)
# 使用 StandardScaler 使数据符合均值为0，方差为1的分布
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. 构建并训练 KNN 模型
# n_neighbors: 邻居数量，通常取奇数以避免平票，这里先设为 5
# weights: 'distance' 表示距离越近的权重越高
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model.fit(X_train, y_train)

# 6. 进行预测
y_pred = knn_model.predict(X_test)

# 7. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
# 使用 weighted 考虑类别不平衡
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"KNN 模型评估结果 (k=5):")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1 分数 (F1-Score): {f1:.4f}")
print("-" * 30)

# 8. 打印详细报告
print("\n各类别详细指标:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 9. 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN - Confusion Matrix')
plt.show()

# 10. (可选) 寻找最优的 K 值
# 尝试不同的 K 查看对准确率的影响
k_range = range(1, 21)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(8, 4))
plt.plot(k_range, scores, marker='o')
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.title('Finding Optimal K')
plt.grid(True)
plt.show()