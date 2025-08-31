import re
import csv
import os
from datetime import datetime

def parse_iperf_data(input_text):
    # Extract only the measurement data lines
    measurement_pattern = r'(\w+ \w+\s+\d+ \d+:\d+:\d+ \d+) \[\s*(\d+)\]\s+(\d+\.\d+-\d+\.\d+)\s+sec\s+(\d+\.\d+) MBytes\s+(\d+) Mbits/sec'
    
    data = []
    for line in input_text.strip().split('\n'):
        match = re.match(measurement_pattern, line)
        if match:
            timestamp, id_num, interval, transfer, bitrate = match.groups()
            
            # Parse interval to get just the numbers
            start_time, end_time = interval.split('-')
            interval_value = float(end_time) - float(start_time)
            
            # Convert transfer to float
            transfer_value = float(transfer)
            
            # Convert bitrate to float
            bitrate_value = float(bitrate)
            
            # Calculate throughput (interval * bitrate)
            throughput = interval_value * bitrate_value
            
            data.append([
                timestamp, 
                id_num, 
                interval_value, 
                transfer_value, 
                bitrate_value, 
                throughput
            ])
    
    return data

def extract_measurement_data(input_text):
    """Extract only the measurement data from the iperf output."""
    lines = input_text.strip().split('\n')
    measurement_data = []
    
    # Find the start line (the header line with "Interval Transfer Bitrate")
    start_index = None
    for i, line in enumerate(lines):
        if "Interval" in line and "Transfer" in line and "Bitrate" in line:
            start_index = i + 1  # Start from the line after the header
            break
    
    if start_index is None:
        return ""  # Header not found
    
    # Find the end line (the line with dashes "- - - - - -")
    end_index = None
    for i, line in enumerate(lines[start_index:], start_index):
        if "- - - - - -" in line:
            end_index = i
            break
    
    if end_index is None:
        end_index = len(lines)  # If no end marker, use all remaining lines
    
    # Extract only the measurement data lines
    return '\n'.join(lines[start_index:end_index])

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Timestamp', 'ID', 'Interval', 'Transfer', 'Bitrate', 'Throughput'])
        # Write data
        writer.writerows(data)
    
    print(f"Data successfully written to {output_file}")

def process_iperf_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        input_text = f.read()
    
    # Extract only the measurement data
    measurement_data = extract_measurement_data(input_text)
    
    # Parse the measurement data
    data = parse_iperf_data(measurement_data)
    
    # Write to CSV
    write_to_csv(data, output_file)

def process_iperf_files_in_folder(folder_path):
    """Process all files containing 'iperf' in the given folder"""
    # Get all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file contains 'iperf' (case insensitive) and is a text file
        if 'iperf' in filename.lower() and filename.endswith('.txt'):
            input_file = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(folder_path, f"{base_name}.csv")
            
            print(f"Processing file: {input_file}")
            try:
                process_iperf_file(input_file, output_file)
                print(f"Successfully processed: {input_file} -> {output_file}")
            except Exception as e:
                print(f"Error processing file {input_file}: {str(e)}")

def main():
    # Specify the folder containing iperf files
    folder_path = "C:/Users/IceCould/OneDrive - Politecnico di Milano/Education/Polimi/B_研一下/实验室经验/raw/nlos_ga"
    
    # Process all iperf files in the folder
    process_iperf_files_in_folder(folder_path)
    print("Processing complete for all iperf files in the folder.")

if __name__ == "__main__":
    main()