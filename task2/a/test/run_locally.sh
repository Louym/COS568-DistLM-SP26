#!/bin/bash

# 1. Define your nodes (Use the IPs/Hostnames your local computer uses to connect)
# Make sure the Master Node is the FIRST one in this list
NODES=("128.105.146.92" "128.105.144.60" "128.105.144.69" "128.105.146.99")
USER="Louym"

MASTER_NODE=${NODES[0]}

echo "--- Step 1: Fetching Public Key from Master Node ($MASTER_NODE) ---"
# This fetches the key from the Master node and stores it in a local variable
MASTER_PUB_KEY=$(ssh $USER@$MASTER_NODE "cat ~/.ssh/id_ed25519.pub")

if [ -z "$MASTER_PUB_KEY" ]; then
    echo "Error: Could not find public key on Master. Did you run ssh-keygen on it?"
    exit 1
fi

echo "--- Step 2: Distributing Master Key to all nodes ---"
for NODE in "${NODES[@]}"; do
    echo "Processing $NODE..."
    # We use SSH to append the key and fix permissions in one go
    ssh $USER@$NODE "mkdir -p ~/.ssh && echo '$MASTER_PUB_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
done

echo "--- Done! ---"
echo "Now your Master Node ($MASTER_NODE) should be able to SSH into all other nodes via the 10.10.1.* network."