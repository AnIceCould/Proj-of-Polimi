import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 指定文件夹路径
folder_path = r"C:\Users\IceCould\OneDrive - Politecnico di Milano\Education\Polimi\B_研一下\实验室经验\processed\ob\ga"

# 获取所有符合条件的 CSV 文件
csv_files = glob.glob(os.path.join(folder_path, "Probe_*_*_DL_*_*_trial*_MS1.csv"))

# 提取算法类型函数
def get_algorithm_from_filename(filename):
    parts = filename.split('_')
    if 'cubic' in parts:
        return 'cubic'
    elif 'bbr' in parts:
        return 'bbr'
    elif 'reno' in parts:
        return 'reno'
    else:
        return 'unknown'

# 指标列表
metrics = [
    "NR PCC DL Avg MCS",
    "NR PCC UL Avg MCS",
    "NR PCC DL Initial BLER(%)",
    "NR PCC UL Initial BLER(%)",
    "NR Serving SS-RSRP(dBm)"
]

# 显示中文字符（如果需要）和负号
plt.rcParams['font.sans-serif'] = ['Arial']  # 避免乱码
plt.rcParams['axes.unicode_minus'] = False

# 创建图像
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 12), constrained_layout=True)

# 读取和绘图
for idx, file in enumerate(csv_files):
    algo = get_algorithm_from_filename(file)
    df = pd.read_csv(file)
    
    for j, metric in enumerate(metrics):
        ax = axes[idx, j]
        if metric in df.columns:
            ax.plot(df[metric], label=algo, linewidth=1.2)
            ax.set_title(f"{algo.upper()} - {metric}", fontsize=9)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(metric)
            ax.grid(True)

plt.show()
output_file = "probe_ob_ul.png"
fig.savefig(output_file, dpi=300)