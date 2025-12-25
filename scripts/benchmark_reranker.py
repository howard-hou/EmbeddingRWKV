import os
import sys
import time
import torch
import numpy as np
from types import SimpleNamespace
from transformers import AutoConfig, AutoModelForSequenceClassification

# -----------------------------------------------------------------------------
# 1. Environment & Config
# -----------------------------------------------------------------------------
os.environ["RWKV_CUDA_ON"] = '1'  # Force CUDA kernel
# only enable old qk^T → softmax → @ v for standard attention benchmark
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

from rwkv_emb.model import EmbeddingRWKV, RWKVReRanker

# Paths (Update these!)
RWKV_EMB_PATH = sys.argv[1]
RWKV_RERANK_PATH = sys.argv[2]

# Benchmark Settings
BATCH_SIZE = 100          # Use a realistic batch size for throughput
QUERY_LEN = 64           # Fixed query length
DOC_LENS = [512, 1024, 2048, 4096] # The variable we are scanning
N_WARMUP = 5
N_RUNS = 5
BERT_MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base" # small (149M)
BERT_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2" # base (0.5B)
BERT_MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v2" # large (1.5B)

device = "cuda"

# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------
def get_bert_model(model_name, attn_implementation):
    """
    Initializes a BERT model with expanded position embeddings 
    to benchmark long-context performance (O(N^2)) without crashing.
    """
    config = AutoConfig.from_pretrained(model_name)
    config.max_position_embeddings = 8192 # Force support for long context benchmarking
    config.torch_dtype = torch.bfloat16  # Load weights in bf16 to save memory
    config._attn_implementation = attn_implementation   # sdpa or flash_attention_2
    # We use random weights; we only care about speed, not accuracy here
    model = AutoModelForSequenceClassification.from_config(config).to(device).eval()
    print(f"torch_dtype: {model.config.torch_dtype}, attn_implementation: {model.config._attn_implementation}")
    return model

def generate_ids(bsz, seq_len):
    return [torch.randint(0, 10000, (seq_len,)).tolist() for _ in range(bsz)]

def generate_dummy_state(model, bsz):
    """
    Generates a dummy cached state tensor matching the shape of RWKV-7 states.
    Simulates loading a pre-computed document state from disk/RAM.
    """
    n_layer = model.rwkv.args.n_layer
    n_embd = model.rwkv.args.n_embd
    n_head = model.rwkv.n_head
    head_size = model.rwkv.head_size

    # State[0]: TimeMix [Layers, 2, Batch, Hidden]
    state_0 = torch.randn(n_layer, 2, bsz, n_embd, device=device, dtype=torch.half)
    
    # State[1]: Attention [Layers, Batch, Heads, HeadSize, HeadSize]
    state_1 = torch.randn(n_layer, bsz, n_head, head_size, head_size, device=device, dtype=torch.float) # Attn state is float32
    
    return [state_0, state_1]

def bert_forward(bert_model, input_ids):
    """
    Simple wrapper to call ModernBERT with a valid attention_mask.
    """
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    with torch.no_grad():
        return bert_model(input_ids=input_ids, attention_mask=attention_mask)
    
def format_mem(bytes_val):
    return f"{bytes_val / (1024**3):.2f} GB"

# -----------------------------------------------------------------------------
# 3. Main Benchmark
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Standard Attention Baseline (Synthetic)...")
    sdpa_model = get_bert_model(BERT_MODEL_NAME, attn_implementation="sdpa")
    print(f"{BERT_MODEL_NAME} with Standard Attention loaded.")

    doc_len2sdpa_str = {}
    doc_len2sdpa_tpt = {}
    for doc_len in DOC_LENS:
        total_len = doc_len + QUERY_LEN
        
        # ------------------------------------------------------
        # A. BERT (Cross-Encoder)
        # Input: [CLS] Query [SEP] Doc [SEP] -> Length: L_q + L_d
        # Complexity: O((Lq+Ld)^2)
        # ------------------------------------------------------
        dummy_input = torch.randint(0, 1000, (BATCH_SIZE, total_len), device=device)
        
        # Warmup
        for _ in range(N_WARMUP):
            bert_forward(sdpa_model, dummy_input)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N_RUNS):
            with torch.no_grad():
                bert_forward(sdpa_model, dummy_input)
        torch.cuda.synchronize()
        sdpa_latency = (time.time() - start) / N_RUNS * 1000
        sdpa_tpt = BATCH_SIZE / (sdpa_latency / 1000)
        sdpa_mem = torch.cuda.max_memory_allocated()

        doc_len2sdpa_str[doc_len] = f"{'Standard Attention':<20} | {doc_len:<5} | {sdpa_latency:<10.2f} | {'---':<10} | {'---':<10} | {format_mem(sdpa_mem):<10} | {sdpa_tpt:<12.1f} | {'1.0x'}"
        doc_len2sdpa_tpt[doc_len] = sdpa_tpt

    # free up SDPA memory
    del sdpa_model
    del dummy_input
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # -----------------------------------------------------------------------------
    # flash attention model loading
    print(f"Loading Flash Attention Baseline (Synthetic)...")
    flash_model = get_bert_model(BERT_MODEL_NAME, attn_implementation="flash_attention_2")
    print(f"{BERT_MODEL_NAME} with Flash Attention loaded.")

    doc_len2flash_str = {}
    doc_len2flash_tpt = {}
    for doc_len in DOC_LENS:
        total_len = doc_len + QUERY_LEN
        
        # ------------------------------------------------------
        # A. BERT (Cross-Encoder)
        # Input: [CLS] Query [SEP] Doc [SEP] -> Length: L_q + L_d
        # Complexity: O((Lq+Ld)^2)
        # ------------------------------------------------------
        dummy_input = torch.randint(0, 1000, (BATCH_SIZE, total_len), device=device)
        
        # Warmup
        for _ in range(N_WARMUP):
            bert_forward(flash_model, dummy_input)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(N_RUNS):
            with torch.no_grad():
                bert_forward(flash_model, dummy_input)
        torch.cuda.synchronize()
        flash_latency = (time.time() - start) / N_RUNS * 1000
        flash_tpt = BATCH_SIZE / (flash_latency / 1000)
        sdpa_tpt = doc_len2sdpa_tpt[doc_len]
        flash_mem = torch.cuda.max_memory_allocated()

        doc_len2flash_str[doc_len] = f"{'Flash Attention':<20} | {doc_len:<5} | {flash_latency:<10.2f} | {'---':<10} | {'---':<10} | {format_mem(flash_mem):<10} | {flash_tpt:<12.1f} | {flash_tpt/sdpa_tpt:.1f}x"
        doc_len2flash_tpt[doc_len] = flash_tpt
    
    # free up Flash memory
    del flash_model
    del dummy_input
    torch.cuda.empty_cache()
    # -----------------------------------------------------------------------------
    # 3. RWKV Benchmarking
    print(f"Loading RWKV Models...")
    emb_model = EmbeddingRWKV(model_path=RWKV_EMB_PATH)
    reranker = RWKVReRanker(model_path=RWKV_RERANK_PATH)
    doc_len2rwkv_on_str = {}
    doc_len2rwkv_off_str = {}
    for doc_len in DOC_LENS:
        total_len = doc_len + QUERY_LEN
        # ------------------------------------------------------
        # B. RWKV Online
        # Input: Instruct + Doc + Query -> Length: L_q + L_d
        # Complexity: O(Lq+Ld)
        # ------------------------------------------------------
        # We simulate the full tokens passing through
        online_tokens = generate_ids(BATCH_SIZE, total_len)
        
        # Warmup
        for _ in range(N_WARMUP):
            out, s = emb_model.forward(online_tokens, None)
            reranker.forward(s[1])

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t_emb_accum = 0.0
        t_rnk_accum = 0.0
        
        start_total = time.time()
        for _ in range(N_RUNS):
            with torch.no_grad():
                # 1. Embedding Part
                t0 = time.time()
                _, state = emb_model.forward(online_tokens, None)
                torch.cuda.synchronize()
                t1 = time.time()
                
                # 2. Reranker Part
                logits = reranker.forward(state[1])
                torch.cuda.synchronize()
                t2 = time.time()

                t_emb_accum += (t1 - t0)
                t_rnk_accum += (t2 - t1)
        
        total_latency = (time.time() - start_total) / N_RUNS * 1000
        emb_latency = (t_emb_accum / N_RUNS) * 1000
        rnk_latency = (t_rnk_accum / N_RUNS) * 1000
        
        rwkv_mem = torch.cuda.max_memory_allocated()
        online_tpt = BATCH_SIZE / (total_latency / 1000)
        sdpa_tpt = doc_len2sdpa_tpt[doc_len]
        doc_len2rwkv_on_str[doc_len] = f"{'RWKV Online':<20} | {doc_len:<5} | {total_latency:<10.2f} | {emb_latency:<10.2f} | {rnk_latency:<10.2f} | {format_mem(rwkv_mem):<10} | {online_tpt:<12.1f} | {online_tpt/sdpa_tpt:.1f}x"

        del online_tokens
        torch.cuda.empty_cache()
        # ------------------------------------------------------
        # C. RWKV Offline (Cached)
        # ------------------------------------------------------
        query_tokens = generate_ids(BATCH_SIZE, QUERY_LEN)
        cached_state = generate_dummy_state(emb_model, BATCH_SIZE)

        for _ in range(N_WARMUP):
            out, s = emb_model.forward(query_tokens, cached_state)
            reranker.forward(s[1])

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        t_emb_accum = 0.0
        t_rnk_accum = 0.0
        
        start_total = time.time()
        for _ in range(N_RUNS):
            with torch.no_grad():
                # 1. Embedding Part (Query Only)
                t0 = time.time()
                # Note: Not cloning here for pure compute benchmark speed
                _, final_state = emb_model.forward(query_tokens, cached_state)
                torch.cuda.synchronize()
                t1 = time.time()
                
                # 2. Reranker Part
                logits = reranker.forward(final_state[1])
                torch.cuda.synchronize()
                t2 = time.time()
                
                t_emb_accum += (t1 - t0)
                t_rnk_accum += (t2 - t1)

        total_latency = (time.time() - start_total) / N_RUNS * 1000
        emb_latency = (t_emb_accum / N_RUNS) * 1000
        rnk_latency = (t_rnk_accum / N_RUNS) * 1000
        
        rwkv_mem = torch.cuda.max_memory_allocated()
        offline_tpt = BATCH_SIZE / (total_latency / 1000)
        sdpa_tpt = doc_len2sdpa_tpt[doc_len]

        doc_len2rwkv_off_str[doc_len] = f"{'RWKV Offline':<20} | {doc_len:<5} | {total_latency:<10.2f} | {emb_latency:<10.2f} | {rnk_latency:<10.2f} | {format_mem(rwkv_mem):<10} | {offline_tpt:<12.1f} | {offline_tpt/sdpa_tpt:.1f}x"

    # -----------------------------------------------------------------------------
    # 4. Print Results
    print("\n" + "="*125)
    header = f"{'Mode':<20} | {'Len':<5} | {'Total(ms)':<10} | {'Emb(ms)':<10} | {'Rnk(ms)':<10} | {'Peak VRAM':<10} | {'Tpt(pair/s)':<12} | {'Speedup'}"
    print(header)
    print("="*125)
    for doc_len in DOC_LENS:
        print(doc_len2sdpa_str[doc_len])
        print(doc_len2flash_str[doc_len])
        print(doc_len2rwkv_on_str[doc_len])
        print(doc_len2rwkv_off_str[doc_len])
        print("-"*125)

if __name__ == "__main__":
    main()