import os
import sys
import time
import torch
import numpy as np
import random
from rwkv_emb.model import EmbeddingRWKV

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Set to '1' for CUDA kernel (Crucial for correct speed benchmarking)
os.environ["RWKV_CUDA_ON"] = '1' 

MODEL_PATH = sys.argv[1] # <--- 修改为你的模型路径
BATCH_SIZE = int(sys.argv[2]) # Set to 1 to measure pure Latency per single request
                         # Set to 8 or 16 to measure maximum Throughput
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384] # 实验变量
N_WARMUP = 5          # 预热次数
N_RUNS = 20           # 正式运行次数取平均
VOCAB_SIZE = 65535

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def generate_synthetic_batch(batch_size, seq_len):
    """
    Generates a batch of random token IDs.
    Returns: List[List[int]] (The format EmbeddingRWKV expects)
    """
    batch = []
    for _ in range(batch_size):
        # Generate random tokens (simulating real text indices)
        # We append 0 at the start just to be consistent with some padding logic if needed,
        # but here we generate exact lengths so no padding inside model is triggered.
        tokens = [random.randint(1, VOCAB_SIZE-1) for _ in range(seq_len-1)] + [65535]
        batch.append(tokens)
    return batch

def format_memory(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.2f} MB"
    else:
        return f"{bytes_val/(1024**3):.2f} GB"

# -----------------------------------------------------------------------------
# Main Benchmark Loop
# -----------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Benchmarking on CPU is not representative.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = EmbeddingRWKV(model_path=MODEL_PATH)
    
    # Force model to CUDA explicitly if not already (wrapper usually handles it)
    # Accessing internal rwkv to check device
    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    print(f"Benchmark Config: Batch Size={BATCH_SIZE}, Runs={N_RUNS}")
    print("-" * 80)
    print(f"{'Seq Len':<10} | {'Latency (ms)':<15} | {'Throughput (doc/s)':<20} | {'Peak VRAM':<15}")
    print("-" * 80)

    results = []

    for seq_len in SEQ_LENGTHS:
        # 1. Prepare Data
        # Ensure all sequences are exactly seq_len (No internal padding overhead)
        batch_tokens = generate_synthetic_batch(BATCH_SIZE, seq_len)
        
        # 2. Reset Memory Stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 3. Warmup
        # (JIT compilation and GPU caching happens here)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                model.forward(batch_tokens, None)
        
        torch.cuda.synchronize() # Wait for warmup to finish
        
        # 4. Benchmarking
        latencies = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                model.forward(batch_tokens, None)
            
            torch.cuda.synchronize() # Wait for kernel execution
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) # Convert to ms

        # 5. Statistics
        avg_latency = np.mean(latencies)
        # Throughput = (Batch Size) / (Avg Latency in seconds)
        throughput = BATCH_SIZE / (avg_latency / 1000)
        
        # Memory (Peak allocated during the forward pass)
        peak_mem = torch.cuda.max_memory_allocated()
        
        print(f"{seq_len:<10} | {avg_latency:<15.2f} | {throughput:<20.2f} | {format_memory(peak_mem):<15}")
        
        results.append({
            "len": seq_len,
            "latency": avg_latency,
            "throughput": throughput,
            "memory": peak_mem
        })

    print("-" * 80)
    print("Done. Copy the data above for your paper plotting.")

if __name__ == "__main__":
    main()