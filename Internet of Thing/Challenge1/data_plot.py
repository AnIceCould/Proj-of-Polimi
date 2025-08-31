import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('transmission_power.csv')

# 将Timestamp列转换为datetime类型
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 计算从第一个时间点开始的秒数
start_time = df['Timestamp'].min()
df['Seconds'] = (df['Timestamp'] - start_time).dt.total_seconds()

# 创建一个新的图表
plt.figure(figsize=(12, 6))

# 绘制数据
plt.plot(df['Seconds'], df['Data'], color='blue', linewidth=1)

# 设置图表标题和轴标签
plt.title('Power Consumption Over Time', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Power (mW)', fontsize=12)

# 设置x轴刻度，每5秒一个刻度
max_seconds = df['Seconds'].max()
ticks = np.arange(0, max_seconds, 5)  # 从0开始，每5秒一个刻度，确保覆盖整个数据范围
plt.xticks(ticks)

# 添加网格线以提高可读性
plt.grid(True, linestyle='--', alpha=0.7)

# 设置y轴的合适范围
y_min = df['Data'].min() * 0.95  # 数据最小值的95%
y_max = df['Data'].max() * 1.05  # 数据最大值的105%
plt.ylim(y_min, y_max)

# 筛选数据点，计算平均值
filtered_data = df[(df['Data'] > 750) & (df['Data'] < 900)]['Data']
if len(filtered_data) > 0:  # 确保有符合条件的数据点
    mean_value = filtered_data.mean()
    plt.axhline(y=mean_value, color='r', linestyle='--', 
                label=f'Average (Wi-Fi 2dB): {mean_value:.2f} mW')

# 添加图例
plt.legend()

# 使图表更美观
plt.tight_layout()

# 保存图表
plt.savefig('power_consumption_plot.png', dpi=300)

# 显示图表
plt.show()