import os
import sys
import time
import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 需要的话可以改或删掉

DEVICE = "cuda"
MODEL_NAME = "intfloat/e5-large-v2" # 335M

BATCH_SIZE = int(sys.argv[1])
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]    # Transformer 往上可能会 OOM
N_WARMUP = 5
N_RUNS = 5
VOCAB_SIZE = 30522  # e5/BERT 系 tokenizer 的 vocab size 大概这个量级，用作合成 token 上限

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def generate_synthetic_batch(batch_size, seq_len):
    """
    生成一批随机 token id，直接绕过 tokenizer，和 RWKV 脚本保持一致的“合成数据”设定。
    返回: input_ids, attention_mask; 都是 torch.LongTensor, shape = [B, L]
    """
    data = []
    for _ in range(batch_size):
        tokens = [random.randint(1, VOCAB_SIZE - 1) for _ in range(seq_len)]
        data.append(tokens)
    input_ids = torch.tensor(data, dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    return input_ids, attention_mask

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
# Main
# -----------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return

    print(f"Loading baseline model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 模型直接以 bf16 加载，并指定 flash_attention_2（如果模型/版本支持）
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,               # 🔑 关键：权重直接以 bf16 加载
        attn_implementation="sdpa",  # BERT系列不支持 flash_attention，目前只能用 sdpa
        device_map=DEVICE                      # 一般就是 "cuda" 或 "auto"
    ).eval()
    print(f"torch_dtype: {model.config.torch_dtype}, attn_implementation: {model.config._attn_implementation}")
    print(f"Model loaded on: {DEVICE}")
    print(f"Benchmark Config: Batch Size={BATCH_SIZE}, Runs={N_RUNS}")
    print("-" * 80)
    print(f"{'Seq Len':<10} | {'Latency (ms)':<15} | {'Throughput (doc/s)':<20} | {'Peak VRAM':<15}")
    print("-" * 80)

    results = []

    for L in SEQ_LENGTHS:
        # 1. 构造合成输入（不计 tokenizer 开销，和 RWKV 一致）
        input_ids, attention_mask = generate_synthetic_batch(BATCH_SIZE, L)
        # [B, L] 的 token_type_ids，全 0，避免用到内部 512 长度的 buffer
        token_type_ids = torch.zeros_like(input_ids, device=DEVICE)

        # 位置 id：0,1,...,L-1 再对 max_pos 取模，保证 index 落在 [0, max_pos-1]
        pos = torch.arange(L, device=DEVICE) % 512       # [L]
        position_ids = pos.unsqueeze(0).expand(BATCH_SIZE, -1)  # [B, L]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        # 2. 重置显存统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 3. Warmup
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(**inputs)
        torch.cuda.synchronize()

        # 4. 正式计时
        latencies = []
        with torch.no_grad():
            for _ in range(N_RUNS):
                torch.cuda.synchronize()
                start = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end = time.time()
                latencies.append((end - start) * 1000)  # ms

        avg_latency = float(np.mean(latencies))
        throughput = BATCH_SIZE / (avg_latency / 1000.0)
        peak_mem = torch.cuda.max_memory_allocated()

        print(f"{L:<10} | {avg_latency:<15.2f} | {throughput:<20.2f} | {format_memory(peak_mem):<15}")

        results.append(
            {
                "len": L,
                "latency": avg_latency,
                "throughput": throughput,
                "memory": peak_mem,
            }
        )

    print("-" * 80)
    print("Done. Copy the data above for plotting with RWKV benchmark.")

if __name__ == "__main__":
    main()
