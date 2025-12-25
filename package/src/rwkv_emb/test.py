import os
import sys
import time
import torch
import numpy as np

# 确保在 import model 之前设置，或者依赖环境
# os.environ["RWKV_CUDA_ON"] = '0' 

from model import EmbeddingRWKV

# RWKV-7 的参考代码(Goose)严重依赖 CUDA，确保环境支持
assert torch.cuda.is_available(), "RWKV-7 requires CUDA"

# EOS token
EOS_INDEX = 65535

# ==============================================================================
# 1. 初始化模型
# ==============================================================================
if len(sys.argv) < 2:
    print("Usage: python test.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]
model = EmbeddingRWKV(model_path=model_path)

# ==============================================================================
# 2. 正确性测试 (Correctness Tests)
# ==============================================================================
print("\n=== Correctness Tests ===")

print("\n--- Test 1: Single Full Pass ---")
tokens_full = [187, 510, 1563, 310, 247, EOS_INDEX]
# 第一次 forward，传入 None 让模型自动生成 zero state
emb_full, state_full = model.forward(tokens_full, None)

print(f"Embedding shape: {emb_full.shape}")
print("Embedding (first 10):", emb_full.detach().cpu().numpy().flatten()[:10])

# RWKV-7 State 结构: List[Tensor, Tensor]
# State[0] (TokenShift): (n_layer, 2, Bsz, n_embd)  <- 如果是BS=1模式，Bsz维度会被移除
# State[1] (TimeMix):    (n_layer, Bsz, H, N, N)
print(f"State length: {len(state_full)}")
print(f"State[0] shape: {state_full[0].shape}") 
print(f"State[1] shape: {state_full[1].shape}")


print("\n--- Test 2: Split Pass (State Consistency Check) ---")
# 测试分段输入 (State Passing)
state = None

# Part A
emb, state = model.forward([187, 510], state)
# Part B (Linear execution, no need to clone state yet)
emb, state = model.forward([1563], state)
# Part C
emb, state = model.forward([310, 247, EOS_INDEX], state)

print("Embedding (first 10):", emb.detach().cpu().numpy().flatten()[:10])

# 验证一致性
diff = torch.max(torch.abs(emb_full - emb))
print(f"Difference between Full and Split pass: {diff.item():.6f}")

if diff < 1e-4:
    print("SUCCESS: State works correctly.")
else:
    print("WARNING: State mismatch detected.")


# ==============================================================================
# 3. 性能测试 (Benchmarks)
# ==============================================================================
print("\n=== Performance Benchmarks ===")

def run_benchmark(label, input_tokens, n_loop=50):
    print(f"\nRunning: {label}")
    
    # 统计 Token 总量
    if isinstance(input_tokens[0], list):
        # Batch 模式
        total_tokens = sum(len(t) for t in input_tokens)
        bsz = len(input_tokens)
    else:
        # Single 模式
        total_tokens = len(input_tokens)
        bsz = 1
    
    print(f"  Batch Size: {bsz}")
    print(f"  Total Tokens per pass: {total_tokens}")

    # Warmup (预热 CUDA kernel)
    model.forward(input_tokens, None)
    torch.cuda.synchronize()

    # Start Timer
    start_time = time.perf_counter()
    
    for _ in range(n_loop):
        # 模拟 Embedding 任务：每次都处理新的文本，所以 state=None
        model.forward(input_tokens, None)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 计算指标
    total_time_seconds = end_time - start_time
    avg_time_seconds = total_time_seconds / n_loop
    avg_time_ms = avg_time_seconds * 1000
    
    # TPS = (每次 pass 的 token 总数) / (每次 pass 的秒数)
    tps = total_tokens / avg_time_seconds

    print(f"  Result: {avg_time_ms:.2f} ms per pass | TPS: {tps:,.2f}")

# 构造一些随机数据用于测试
# 假设词表大小约为 65536
dummy_vocab_range = (0, 60000)
seq_len = 512
bsz = 8

# 1. Batch Size = 1
tokens_bs1 = list(np.random.randint(*dummy_vocab_range, size=seq_len).tolist())

# 2. Batch Size > 1, Same Length
tokens_bs_same = [list(np.random.randint(*dummy_vocab_range, size=seq_len).tolist()) for _ in range(bsz)]

# 3. Batch Size > 1, Variable Length
# 构造长度不一的序列 (例如：从 seq_len//2 到 seq_len)
tokens_bs_var = []
for i in range(bsz):
    # 随机长度
    this_len = np.random.randint(seq_len // 4, seq_len)
    tokens_bs_var.append(list(np.random.randint(*dummy_vocab_range, size=this_len).tolist()))


# 执行测试
run_benchmark("1. Batch Size = 1 (Single Sequence)", tokens_bs1)
run_benchmark(f"2. Batch Size = {bsz}, Same Length ({seq_len})", tokens_bs_same)
run_benchmark(f"3. Batch Size = {bsz}, Variable Length (Random {seq_len//4}-{seq_len})", tokens_bs_var)

print('\nDone.')