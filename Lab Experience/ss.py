import re
import csv
import os
from datetime import datetime

def extract_tcp_stats(input_file, output_file):
    """
    从TCP状态输出文件中提取信息并保存为CSV格式，
    根据文件名中的算法信息过滤数据
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出CSV文件路径
    """
    # 从文件名中获取期望的算法
    file_basename = os.path.basename(input_file)
    expected_algorithm = None
    
    if 'bbr' in file_basename.lower():
        expected_algorithm = 'bbr'
    elif 'cubic' in file_basename.lower():
        expected_algorithm = 'cubic'
    elif 'reno' in file_basename.lower():
        expected_algorithm = 'reno'
    
    if not expected_algorithm:
        print(f"警告: 无法从文件名 '{file_basename}' 中检测到算法(bbr/cubic/reno)")
    else:
        print(f"从文件名中检测到算法: {expected_algorithm}")
    
    # 初始化结果列表
    results = []
    
    # 读取输入文件
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 按时间戳分割数据块
    blocks = content.split('Timestamp: ')
    
    for block in blocks[1:]:  # 跳过第一个空块
        # 提取时间戳
        timestamp_match = re.match(r'(.+?)\n', block)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
        else:
            continue
        
        # 查找所有连接信息块
        connections = re.findall(r'ESTAB.+?\n\t (.+?)(?=\n\w|\Z)', block, re.DOTALL)
        
        for conn in connections:
            # 确定使用的拥塞控制算法
            algorithm = None
            if conn.startswith('cubic'):
                algorithm = 'cubic'
            elif conn.startswith('bbr'):
                algorithm = 'bbr'
            elif conn.startswith('reno'):
                algorithm = 'reno'
            else:
                continue
            
            # 如果算法与文件名中的不匹配，则跳过此记录
            if expected_algorithm and algorithm != expected_algorithm:
                continue
            
            # 提取RTT值
            rtt_match = re.search(r'rtt:(\d+\.\d+)/\d+\.\d+', conn)
            rtt = rtt_match.group(1) if rtt_match else ''
            
            # 提取CWND值
            cwnd_match = re.search(r'cwnd:(\d+)', conn)
            cwnd = cwnd_match.group(1) if cwnd_match else ''
            
            # 提取ssthresh
            ssthresh_match = re.search(r'ssthresh:(\d+)', conn)
            ssthresh = ssthresh_match.group(1) if ssthresh_match else ''
            
            # 只添加ssthresh不为空的记录
            if ssthresh and ssthresh != "":
                results.append({
                    'timestamp': timestamp,
                    'algorithm': algorithm,
                    'rtt': rtt,
                    'cwnd': cwnd,
                    'ssthresh': ssthresh
                })
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'algorithm', 'rtt', 'cwnd', 'ssthresh']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"已成功提取数据并保存到 {output_file}")
    print(f"已删除ssthresh为空的行和与文件名算法({expected_algorithm if expected_algorithm else '未检测'})不匹配的行")

def process_ss_files_in_folder(folder_path):
    """处理文件夹中所有包含'ss'的文件"""
    # 获取文件夹中所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否包含'ss'(不区分大小写)并且是文本文件
        if 'ss' in filename.lower() and filename.endswith('.txt'):
            input_file = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(folder_path, f"{base_name}.csv")
            
            print(f"\n正在处理文件: {input_file}")
            try:
                extract_tcp_stats(input_file, output_file)
                print(f"成功处理: {input_file} -> {output_file}")
            except Exception as e:
                print(f"处理文件 {input_file} 时出错: {str(e)}")

def main():
    # 指定包含ss文件的文件夹路径
    folder_path = "C:/Users/IceCould/OneDrive - Politecnico di Milano/Education/Polimi/B_研一下/实验室经验/raw/nlos_ga"
    
    # 处理文件夹中所有ss文件
    process_ss_files_in_folder(folder_path)
    print("\n已完成文件夹中所有ss文件的处理")

if __name__ == "__main__":
    main()