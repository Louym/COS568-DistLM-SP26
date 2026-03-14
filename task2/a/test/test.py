import os
import torch
import torch.distributed as dist
import argparse
import socket
import time

def main():
    parser = argparse.ArgumentParser(description="Complex Distributed GEMM & Reduce Example")
    parser.add_argument('--master_addr', type=str, required=True, help="IP address of the Master node")
    parser.add_argument('--master_port', type=str, default='29500')
    parser.add_argument('--world_size', type=int, default=2, help="Total number of nodes/processes")
    parser.add_argument('--rank', type=int, required=True, help="Rank of the current process")
    args = parser.parse_args()

    # 1. Initialization
    # Use 'gloo' for CPU-based distributed computing. 
    # Switch to 'nccl' if you are using NVIDIA GPUs.
    try:
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank
        )
        print(f"Rank {args.rank}: Process group initialized on {socket.gethostname()}")
    except Exception as e:
        print(f"Rank {args.rank}: Initialization failed: {e}")
        return

    # 2. Setup Data for GEMM (General Matrix Multiplication)
    # We simulate a large linear layer: Y = X @ W
    matrix_dim = 2048
    slice_size = matrix_dim // args.world_size
    torch.manual_seed(42 + args.rank) # Different seed per rank for varied data
    
    # Local shard of the input matrix X
    local_X = torch.randn(matrix_dim, matrix_dim)
    # Shared Weight matrix W (in real Model Parallel, W might be sharded too)
    shared_W = torch.randn(matrix_dim, matrix_dim)

    # 3. Synchronization Before Computation
    # Ensure all nodes have allocated memory and are ready to start the benchmark
    dist.barrier()
    comp_start_time = time.time()

    # 4. Perform Local GEMM
    # Each rank computes a partial result of the global operation
    local_output = torch.matmul(local_X[:, args.rank * slice_size : (args.rank + 1) * slice_size], shared_W[:, args.rank * slice_size : (args.rank + 1) * slice_size])
    
    # Optional: Brief local work to simulate real-world compute variance
    # time.sleep(0.1 * args.rank) 

    comp_end_time = time.time()

    # 5. Synchronization Before Communication
    # This barrier ensures we measure communication latency separately from computation
    dist.barrier()
    comm_start_time = time.time()

    # 6. Global Reduce (All-Reduce)
    # Aggregates the local_output from all ranks into a single sum
    # After this call, every rank's 'local_output' will contain the global sum
    dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
    
    comm_end_time = time.time()

    # 7. Reporting and Validation
    if args.rank == 0:
        total_time = comm_end_time - comp_start_time
        compute_duration = comp_end_time - comp_start_time
        comm_duration = comm_end_time - comm_start_time
        ref=torch.matmul(local_X, shared_W)
        if torch.allclose(local_output, ref):
            print("Result is correct")
        else:
            print("Result is incorrect")
            print(f"Reference: {ref.sum().item():.2f}")
            print(f"Result: {local_output.sum().item():.2f}")
            print(f"Difference: {torch.abs(local_output - ref).sum().item():.2f}")
            print(f"Difference percentage: {torch.abs(local_output - ref).sum().item() / ref.sum().item() * 100:.2f}%")
            print(f"Reference: {ref.sum().item():.2f}")
        print("\n" + "="*50)
        print(f"DISTRIBUTED PERFORMANCE REPORT (World Size: {args.world_size})")
        print("-"*50)
        print(f"Matrix Shape:      {matrix_dim}x{matrix_dim}")
        print(f"Compute Time:      {compute_duration:.4f} seconds")
        print(f"Communication:     {comm_duration:.4f} seconds (All-Reduce)")
        print(f"Total Latency:     {total_time:.4f} seconds")
        print(f"Comm Overhead:     {(comm_duration/total_time)*100:.2f}%")
        print("-"*50)
        print(f"Result checksum:   {local_output.sum().item():.2f}")
        print("="*50 + "\n")

    # 8. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()