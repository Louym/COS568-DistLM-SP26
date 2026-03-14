import os
import torch
import torch.distributed as dist
import argparse
import socket
import time

def main():
    parser = argparse.ArgumentParser()
    # The IP address of the Master node (Rank 0) on the 10.10.1.* network
    parser.add_argument('--master_addr', type=str, required=True, help="Master IP (e.g., 10.10.1.1)")
    parser.add_argument('--master_port', type=str, default='12345')
    parser.add_argument('--world_size', type=int, default=4, help="Total number of nodes")
    parser.add_argument('--rank', type=int, required=True, help="Rank of current node (0 to world_size-1)")
    args = parser.parse_args()

    print(f"[{args.rank}] Attempting to connect to Master at {args.master_addr}:{args.master_port}...")

    try:
        # Initialize the process group
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank
        )
        print(f"[{args.rank}] Initialization successful! Hostname: {socket.gethostname()}")

        # Create a tensor unique to this rank (e.g., Rank 0 creates [1], Rank 1 creates [2]...)
        tensor = torch.ones(1) * (args.rank + 1)
        print(f"[{args.rank}] Local data: {tensor.item()}")

        # Synchronize: All nodes wait here until everyone has reached this point
        dist.barrier()

        # All-Reduce: Sum the values from all nodes
        # Expected result for 4 nodes: 1 + 2 + 3 + 4 = 10.0
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        print(f"[{args.rank}] Result after All-Reduce (Expected 10.0): {tensor.item()}")

    except Exception as e:
        print(f"[{args.rank}] Error during distributed init: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[{args.rank}] Process group destroyed. Test complete.")

if __name__ == "__main__":
    main()