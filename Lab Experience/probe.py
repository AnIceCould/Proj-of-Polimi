import pandas as pd
import os
import sys

def extract_columns(input_filepath, output_filepath):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_filepath)
        
        # 提取指定的列
        columns_to_extract = [
            'Date & Time',
            'NR PCC DL Avg MCS',
            'NR PCC UL Avg MCS',
            'NR PCC DL Initial BLER(%)',
            'NR PCC UL Initial BLER(%)',
            'NR Serving SS-RSRP(dBm)'
        ]
        
        # 检查所有列是否存在
        missing_columns = [col for col in columns_to_extract if col not in df.columns]
        if missing_columns:
            print(f"警告: 在文件 {os.path.basename(input_filepath)} 中以下列不存在: {', '.join(missing_columns)}")
            # 跳过不存在的列，只提取存在的列
            columns_to_extract = [col for col in columns_to_extract if col in df.columns]
            if not columns_to_extract:
                print(f"错误: 文件 {os.path.basename(input_filepath)} 中没有所需的列，跳过处理")
                return False
        
        # 创建只包含所需列的新数据框
        extracted_df = df[columns_to_extract]
        
        # 保存到新文件
        extracted_df.to_csv(output_filepath, index=False)
        print(f"成功处理文件 {os.path.basename(input_filepath)} 并保存到 {output_filepath}")
        return True
    
    except Exception as e:
        print(f"处理文件 {os.path.basename(input_filepath)} 时出错: {str(e)}")
        return False

def batch_process_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有CSV文件
    processed_files = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            if extract_columns(input_filepath, output_filepath):
                processed_files += 1
    
    print(f"\n处理完成！共处理了 {processed_files} 个文件。输出文件保存在 {output_folder}")

def main():
    # 输入文件夹路径
    input_folder = "C:/Users/IceCould/OneDrive - Politecnico di Milano/Education/Polimi/B_研一下/实验室经验/raw/Probe_NLOS_obstacle_LabExp"
    
    # 输出文件夹路径（在当前目录下创建processed文件夹）
    output_folder = "processed"
    
    batch_process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()