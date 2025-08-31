import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

def process_data(folder_path):
    """处理文件夹中的数据并按算法类型返回整理后的结果"""
    # 存储结果的字典
    algorithms = ['bbr', 'cubic', 'reno']  # 预设的三种算法
    data = {alg: {'throughput': None, 'rtt': None, 'cwnd': None, 'ssthresh': None} for alg in algorithms}
    
    # 查找所有iperf数据文件
    iperf_files = glob.glob(os.path.join(folder_path, "*iperf*_*.csv"))
    for file in iperf_files:
        # 从文件名中提取算法，处理可能存在的obs部分
        match = re.search(r'_([a-zA-Z0-9]+)(?:_NLOS)?_trial', file)
        if match:
            algorithm = match.group(1).lower()
            if algorithm in algorithms:
                try:
                    df = pd.read_csv(file)
                    # 确保包含Throughput列
                    if 'Throughput' in df.columns:
                        data[algorithm]['throughput'] = df['Throughput'].values
                except Exception as e:
                    print(f"处理{file}时出错: {e}")
    
    # 查找所有ss数据文件
    ss_files = glob.glob(os.path.join(folder_path, "*ss*_*.csv"))
    for file in ss_files:
        # 从文件名中提取算法，处理可能存在的obs部分
        match = re.search(r'_([a-zA-Z0-9]+)(?:_NLOS)?_trial', file)
        if match:
            algorithm = match.group(1).lower()
            if algorithm in algorithms:
                try:
                    df = pd.read_csv(file)
                    # 确保包含所需列
                    if 'rtt' in df.columns:
                        data[algorithm]['rtt'] = df['rtt'].values
                    if 'cwnd' in df.columns:
                        data[algorithm]['cwnd'] = df['cwnd'].values
                    if 'ssthresh' in df.columns:
                        data[algorithm]['ssthresh'] = df['ssthresh'].values
                except Exception as e:
                    print(f"处理{file}时出错: {e}")
    
    return data

def create_visualization(data):
    """创建4×3的可视化图表，每行对应一个算法，只在顶部添加列标题"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # 数据类型
    data_types = ['throughput', 'rtt', 'cwnd', 'ssthresh']
    column_titles = ['Throughput', 'RTT', 'CWND', 'SSThresh']
    
    # 算法顺序（对应行）
    algorithms = ['bbr', 'cubic', 'reno']
    
    for row, algorithm in enumerate(algorithms):
        for col, (data_type, column_title) in enumerate(zip(data_types, column_titles)):
            ax = axes[row, col]
            
            # 绘制特定算法的特定数据类型，跳过第一个数据点
            values = data[algorithm][data_type]
            if values is not None and len(values) > 1:  # 确保至少有两个点（因为要删除第一个）
                #values = values[1:]  # 删除第一个数据点
                x_values = range(len(values))
                ax.plot(x_values, values, linestyle='-')
            
            # 设置坐标轴标签
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(column_title)
            ax.grid(True)
            
            # 只在第一行添加标题，表示整列
            if row == 0:
                ax.set_title(column_title, fontsize=14, fontweight='bold')
            
            # 在每行的第一个图中添加算法标签
            if col == 0:
                # 在Y轴标签前面添加算法名称
                ax.set_ylabel(f"{algorithm.upper()}\n{column_title}")
    
    plt.tight_layout()
    return fig

def main():
    # 指定数据所在的文件夹路径
    folder_path = "C:/Users/IceCould/OneDrive - Politecnico di Milano/Education/Polimi/B_研一下/实验室经验/processed/ob/ga"
    
    print(f"正在处理文件夹: {folder_path}")
    
    # 处理数据
    data = process_data(folder_path)
    
    # 创建可视化
    fig = create_visualization(data)
    
    # 保存图表
    output_file = "ga_ob_dl.png"
    fig.savefig(output_file, dpi=300)
    print(f"可视化已保存至: {output_file}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()