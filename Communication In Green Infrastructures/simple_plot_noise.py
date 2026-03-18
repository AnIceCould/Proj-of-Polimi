import matplotlib.pyplot as plt

# 准备数据
noise_floors = ["-10", "-20", "-25", "-30", "-35", "-40", "-100"]
accuracy = [0.5500, 0.7000, 0.6900, 0.9000, 0.9100, 0.9500, 0.9400]
precision = [0.5601, 0.7049, 0.7151, 0.9038, 0.9187, 0.9518, 0.9440]
recall = [0.5500, 0.7000, 0.6900, 0.9000, 0.9100, 0.9500, 0.9400]
f1_score = [0.5531, 0.7003, 0.6893, 0.8981, 0.9097, 0.9499, 0.9398]

# 设置绘图风格
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-muted') # 或者使用 'ggplot'

# 绘制各项指标
plt.plot(noise_floors, accuracy, marker='o', linestyle='-', linewidth=2, label='Accuracy')
plt.plot(noise_floors, precision, marker='s', linestyle='--', linewidth=2, label='Precision')
plt.plot(noise_floors, recall, marker='^', linestyle='-.', linewidth=2, label='Recall')
plt.plot(noise_floors, f1_score, marker='d', linestyle=':', linewidth=2, label='F1-Score')

# 添加标题和标签
plt.title('Model Performance vs. Noise Floor', fontsize=14)
plt.xlabel('Noise Floor (dB)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0.5, 1.0) # 根据数据范围调整 y 轴区间
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()