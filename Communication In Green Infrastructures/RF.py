import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'Spinach_Raw_THz_Stress_Dataset.csv'
df = pd.read_csv(file_path)

# 2. 数据预处理
X = df.drop('label', axis=1)
y = df['label']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# 划分训练集和测试集 (保持与之前实验一致的 80/20 比例和随机种子)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. 使用最佳参数构建最终模型
# 参数来源: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 300}
final_rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=5,
    class_weight='balanced', # 建议保留，以应对植物样本间可能的类别不平衡
    random_state=42
)

# 4. 训练模型
final_rf_model.fit(X_train, y_train)

# 5. 进行预测
y_pred = final_rf_model.predict(X_test)

# 6. 计算核心指标
accuracy = accuracy_score(y_test, y_pred)
# 针对多分类使用 weighted 平均值
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=" * 40)
print(f"最终随机森林模型评估 (最佳参数版):")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1 分数 (F1-Score): {f1:.4f}")
print("=" * 40)

# 7. 详细分类报告
print("\n各类别详细指标:")
print(classification_report(y_test, y_pred, target_names=class_names))

# # 8. 可视化特征重要性
# # 既然模型已优化，查看哪些太赫兹特征贡献最大非常有科研价值
# importances = final_rf_model.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# plt.figure(figsize=(12, 7))
# plt.title("Optimized RF - Feature Importances")
# sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
# plt.xlabel("Relative Importance Score")
# plt.tight_layout()
# plt.show()

# 9. 混淆矩阵
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('RF Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()