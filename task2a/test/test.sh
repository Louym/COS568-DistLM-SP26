# python dist_test.py --rank 0 --master_addr 10.10.1.2 --world_size 4
# python dist_test.py --rank 1 --master_addr 10.10.1.2 --world_size 4
# python dist_test.py --rank 2 --master_addr 10.10.1.2 --world_size 4
# python dist_test.py --rank 3 --master_addr 10.10.1.2 --world_size 4


#!/bin/bash

# Configuration
MASTER_IP="10.10.1.2"  # Change to your actual 10.10.1.* IP
WORLD_SIZE=4
PYTHON_PATH="python"   # Or path to your conda env: /raid/home/yumingl/anaconda3/bin/python
SCRIPT_DIR="$(pwd)/test.py"

# Array of your node hostnames or experimental IPs
NODES=("10.10.1.2" "10.10.1.3" "10.10.1.4" "10.10.1.1")

for RANK in {0..3}; do
    NODE_IP=${NODES[$RANK]}
    echo "Launching Rank $RANK on $NODE_IP..."
    
    # Run in background via SSH
    # Note: We use -n to redirect stdin from /dev/null to prevent SSH from hanging
    ssh -n -o "StrictHostKeyChecking=no" $NODE_IP "$PYTHON_PATH $SCRIPT_DIR --rank $RANK --master_addr $MASTER_IP --world_size $WORLD_SIZE" &
done

# Wait for all background processes to finish
wait
echo "All nodes have finished the test."