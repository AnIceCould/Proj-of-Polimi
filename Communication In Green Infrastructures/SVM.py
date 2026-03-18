import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'dataset/Spinach_Raw_THz_Stress_Dataset_middle.csv'
try:
    df = pd.read_csv(file_path)
    print(f"成功加载数据集: {file_path}")
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}")
    exit()

# 2. 数据预处理
X = df.drop('label', axis=1)
y = df['label']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# 3. 特征缩放 (SVM 必须步骤)
# SVM 通过计算超平面距离工作，特征量纲必须统一
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. 构建并训练 SVM 模型
# C: 惩罚系数（控制误分类的容忍度）
# kernel: 'rbf' (高斯核) 适合处理非线性太赫兹特征空间
# gamma: 'scale' 自动调整核函数影响范围
svm_model = SVC(kernel='rbf', C=100, gamma=0.1, probability=True, class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# 6. 进行预测
y_pred = svm_model.predict(X_test)

# 7. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 30)
print(f"SVM 模型评估结果 (Kernel=RBF):")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM - Confusion Matrix')
plt.show()
