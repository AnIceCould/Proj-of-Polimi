import re
import csv

def extract_time_data(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 找出所有完整的数据组
    # 每组数据包含5行：setup time、idle time、busy time、trans time和sleep time
    all_data = []
    
    # 使用正则表达式找出所有数据组
    pattern = r'setup time: (\d+) ms\s+idle time: (\d+) ms\s+busy time: (\d+) ms\s+trans time: (\d+) (us|ms)\s+sleep time: (\d+) ms'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        setup_time = match.group(1)
        idle_time = match.group(2)
        busy_time = match.group(3)
        trans_time_value = float(match.group(4))
        trans_time_unit = match.group(5)
        sleep_time = match.group(6)
        
        # 如果trans time单位是微秒，转换为毫秒
        if trans_time_unit == 'us':
            trans_time = trans_time_value / 1000  # 1 ms = 1000 us
        else:
            trans_time = trans_time_value
            
        # 将数据添加到列表中
        all_data.append({
            'setup time': setup_time,
            'idle time': idle_time,
            'busy time': busy_time,
            'trans time': str(trans_time),
            'sleep time': sleep_time
        })
    
    # 写入CSV文件
    if all_data:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['setup time', 'idle time', 'busy time', 'trans time', 'sleep time'])
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"找到并保存了 {len(all_data)} 组数据到 {output_file}")
    else:
        print("未找到任何匹配的数据组")

# 如果这个脚本被直接运行
if __name__ == "__main__":
    input_file = "raw_time_data.txt"
    output_file = "time_data.csv"
    extract_time_data(input_file, output_file)