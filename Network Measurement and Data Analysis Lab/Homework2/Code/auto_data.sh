#!/bin/bash

# Defining variables
OUTPUT_DIR="captures"            # Output Directory
CSV_FILE="website_ip_list.csv"   # Input CSV file
INTERFACE="eth0"                 # Interface
CAPTURE_DURATION=5              # Capture duration (seconds)
CURL_COUNT=10                     # curl times

# Create dir
mkdir -p "$OUTPUT_DIR"

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "error: CSV $CSV_FILE not exist!"
    exit 1
fi

# Skip header row of CSV file
tail -n +2 "$CSV_FILE" | while IFS=, read -r TARGET_URL TARGET_IP
do
    # Extract the website name from the URL as a file name prefix
    SITE_NAME=$(echo "$TARGET_URL" | sed -E 's|https?://||' | awk -F/ '{print $1}' | sed -E 's/^www[0-9]?\.//' | awk -F. '{if (NF>2 && $(NF-1) != "co" && $(NF-1) != "com") {print $(NF-2)} else {print $(NF-2)}}' | tr -d '.')
    
    echo "======================================"
    echo "website: $TARGET_URL (IP: $TARGET_IP)"
    echo "======================================"
    
    # Perform multiple curl requests to each website
    for ((i=1; i<=CURL_COUNT; i++)); do
        # Create a unique file name for each request
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PCAP="${OUTPUT_DIR}/${SITE_NAME}_${i}_${TIMESTAMP}.pcap"
        OUTPUT_CSV="${OUTPUT_DIR}/${SITE_NAME}_${i}_${TIMESTAMP}.csv"
        
        echo "===== Request $i/$CURL_COUNT to $SITE_NAME ====="
        
        # Start tcpdump
        echo "Start tcpdump, Lasts $CAPTURE_DURATION s..."
        sudo tcpdump -i "$INTERFACE" -nn "" -w "$OUTPUT_PCAP" &
        TCPDUMP_PID=$!
        
        # wait
        sleep 1
        
        # Start curl
        echo "Execute a curl request to $TARGET_URL..."
        curl --user-agent "Mozilla/4.0" -s -kL "$TARGET_URL" > /dev/null
        
        # wait
        echo "Wait for capture to complete..."
        sleep $((CAPTURE_DURATION - 1))
        
        # stop tcpdump
        sudo kill "$TCPDUMP_PID"
        echo "Capture complete. Data saved to $OUTPUT_PCAP"
        
        # Use tshark to convert .pcap to .csv
        echo "Convert $OUTPUT_PCAP to $OUTPUT_CSV..."
        tshark -r "$OUTPUT_PCAP" -T fields \
            -e frame.number \
            -e frame.time \
            -e frame.len \
            -e frame.cap_len \
            -e ip.id \
            -e ip.checksum \
            -e ip.proto \
            -e ip.len \
            -e ip.src \
            -e ip.dst \
            -e tcp.hdr_len \
            -e tcp.len \
            -e tcp.srcport \
            -e tcp.dstport \
            -e tcp.seq \
            -e tcp.ack \
            -e tcp.window_size_value \
            -e tcp.checksum \
            -E header=y \
            -E separator=, \
            -E quote=d \
            -E occurrence=f > "$OUTPUT_CSV"
        
        echo "Completed. Saved in the $OUTPUT_CSV"
        echo "------------------------"
    done
done

echo "Completed. Saved in the $OUTPUT_DIR."