# EmbeddingRWKV

A high-efficiency text embedding and reranking model based on RWKV architecture.

## 📦 Installation

```bash
pip install rwkv-emb
```

## 🤖 Models & Weights

You can download the weights from the [HuggingFace Repository](https://huggingface.co/howard-hou/EmbeddingRWKV/tree/main).

| Size / Level | Embedding Model (Main) | Matching Reranker (Paired) | Notes |
| :--- | :--- | :--- | :--- |
| **Tiny** | `rwkv0b1-emb-curriculum.pth` | `rwkv0b1-reranker.pth` | Ultra-fast, minimal memory. |
| **Base** | `rwkv0b4-emb-curriculum.pth` | `rwkv0b3-reranker.pth` | Balanced speed & performance. |
| **Large** | `rwkv1b4-emb-curriculum.pth` | `rwkv1b3-reranker.pth` | Best performance, higher VRAM usage. |

## 🚀 Quick Start (End-to-End)

Get text embeddings in just a few lines. The tokenizer and model are designed to work seamlessly together.

> **Note**: Always set `add_eos=True` during tokenization. The model relies on the EOS token (`65535`) to mark the end of a sentence for correct embedding generation.

```python
import os
from torch.nn import functional as F
# Set environment for JIT compilation (Optional, set to '1' for CUDA acceleration)
os.environ["RWKV_CUDA_ON"] = '1'

from rwkv_emb.tokenizer import RWKVTokenizer
from rwkv_emb.model import EmbeddingRWKV

# Fast retrieval, good for initial candidate filtering.
emb_model = EmbeddingRWKV(model_path='/path/to/model.pth')
tokenizer = RWKVTokenizer()

query = "What represents the end of a sequence?"
documents = [
    "The EOS token is used to mark the end of a sentence.",
    "Apples are red and delicious fruits.",
    "Machine learning requires large datasets."
]
# Encode Query
q_tokens = tokenizer.encode(query, add_eos=True)
q_emb, _ = emb_model.forward_text_only(q_tokens, None) # shape: [1, Dim]

# Encode Documents (Batch)
doc_batch = [tokenizer.encode(doc, add_eos=True) for doc in documents]
max_doc_len = max(len(t) for t in doc_batch)
for i in range(len(doc_batch)):
    pad_len = max_doc_len - len(doc_batch[i])
    # Prepend 0s (Left Padding)
    doc_batch[i] = [0] * pad_len + doc_batch[i]

d_embs, _ = emb_model.forward_text_only(doc_batch, None)

# Calculate Cosine Similarity
scores_emb = F.cosine_similarity(q_emb, d_embs)
print("\nEmbeddingRWKV Cosine Similarity:")
for doc, score in zip(documents, scores_emb):
    print(f"[{score.item():.4f}] {doc}")
```

For production use cases, running inference in batches is significantly faster.

### ⚠️ Critical Performance Tip: Pad to Same Length

While the model supports batches with variable sequence lengths, **we strongly recommend padding all sequences to the same length** for maximum GPU throughput.

  - **Pad Token**: `0`
  - **Performance**: Fixed-length batches allow the CUDA kernel to parallelize computation efficiently. Variable-length batches will trigger a slower execution path.


### Image-Only Example

```python
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor
from rwkv_emb.model import EmbeddingRWKV

emb_model = EmbeddingRWKV(model_path='/path/to/vision-enabled-model.pth')

# Preprocess images to match SigLIP2 input (B, C, H, W) with 256x256 resolution
image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-256", use_fast=True)

def load_image_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

images = [
    load_image_from_url("https://images.unsplash.com/photo-1518791841217-8f162f1e1131"),
    load_image_from_url("https://images.unsplash.com/photo-1517423440428-a5a00ad493e8"),
]
pixel_values = image_processor(images=images, return_tensors="pt")['pixel_values'].squeeze(0)

# Compute embeddings for images only (if tokens is not given)
img_embs, _ = emb_model.forward_image_only(pixel_values, full_output=False)
print(img_embs.shape)  # [batch, Dim]
```

### Multimodal (Text + Image) Example

```python
import torch
from PIL import Image
from torchvision import transforms
from rwkv_emb.model import EmbeddingRWKV
from rwkv_emb.tokenizer import RWKVTokenizer

emb_model = EmbeddingRWKV(model_path='/path/to/vision-enabled-model.pth')
tokenizer = RWKVTokenizer()
# Preprocess images to match SigLIP2 input (B, C, H, W) with 256x256 resolution
image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-256", use_fast=True)

def load_image_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

images = [
    load_image_from_url("https://images.unsplash.com/photo-1518791841217-8f162f1e1131"),
    load_image_from_url("https://images.unsplash.com/photo-1517423440428-a5a00ad493e8"),
]
pixel_values = image_processor(images=images, return_tensors="pt")['pixel_values'].squeeze(0)

# Tokenize and left-pad to the same length for best performance
captions = [
    "A cat resting on a wooden floor.",
    "A happy dog playing in the park.",
]

token_batch = [tokenizer.encode(text, add_eos=True) for text in captions]
max_len = max(len(t) for t in token_batch)
token_batch = [[0] * (max_len - len(t)) + t for t in token_batch]

# Jointly encode text-image pairs (tokens and images must be paired)
mm_embs, _ = emb_model.forward_multimodal(token_batch, pixel_values)
print(mm_embs.shape)  # [batch, Dim]
```



## 🎯 RWKVReRanker (State-based Reranker)

The `RWKVReRanker` utilizes the final hidden state produced by the main `EmbeddingRWKV` model to score the relevance between a query and a document.

### Online Mode

#### Workflow
1.  **Format** Query and Document based on Online template.
2.  Run the **Embedding Model** to generate the final State.
3.  Feed the **TimeMixing State** (`state[1]`) into the **ReRanker** to get a relevance score.

#### 📝 Online Mode Usage Example

```python
import torch
from rwkv_emb.tokenizer import RWKVTokenizer
from rwkv_emb.model import EmbeddingRWKV, RWKVReRanker

# 1. Load Models
# The ReRanker weights are stored in the differernt checkpoint
emb_model = EmbeddingRWKV(model_path='/path/to/EmbeddingRWKV.pth')
reranker = RWKVReRanker(model_path='/path/to/RWKVReRanker.pth')

tokenizer = RWKVTokenizer()

# 2. Prepare Data (Query + Candidate Documents)
query = "What represents the end of a sequence?"
documents = [
    "The EOS token is used to mark the end of a sentence.",
    "Apples are red and delicious fruits.",
    "Machine learning requires large datasets."
]

# 3. Construct Input Pairs
# We treat the Query and Document as a single sequence.
pairs = []
online_template = "Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\nQuery: {query}"
for doc in documents:
    # Format: Instruct + Document + Query
    text = online_template.format(document=doc, query=query)
    pairs.append(text)

# 4. Tokenize & Pad (Critical for Batch Performance)
batch_tokens = [tokenizer.encode(p, add_eos=True) for p in pairs]

# Left pad to same length for efficiency
max_len = max(len(t) for t in batch_tokens)
for i in range(len(batch_tokens)):
    batch_tokens[i] = [0] * (max_len - len(batch_tokens[i])) + batch_tokens[i]

# 5. Get States from Embedding Model
# We don't need the embedding output here, we only need the final 'state'
_, state = emb_model.forward(batch_tokens, None)

# 6. Score with ReRanker
# The ReRanker expects the TimeMixing State: state[1]
# state[1] shape: [Layers, Batch, Heads, HeadSize, HeadSize]
logits = reranker.forward(state[1])
scores = torch.sigmoid(logits) # Convert logits to probabilities (0-1)

# 7. Print Results
print("\nRWKVReRanker Online Scores:")
for doc, score in zip(documents, scores):
    print(f"[{score:.4f}] {doc}")
```

### Offline Mode (Cached Doc State)
For scenarios where documents are static but queries change (e.g., Search Engines, RAG), you can **pre-compute and cache the document states**. This reduces query-time latency from O(L_doc + L_query) to just O(L_query).

#### Workflow

1.  **Indexing**: Process `Instruct + Document` -\> Save State.
2.  **Querying**: Load State -\> Process `Query` -\> Score.

#### 📝 Offline Mode Usage Example

```python
# --- Phase 1: Indexing (Pre-computation) ---
# Note: Do NOT add EOS here, because the sequence continues with the query later.
doc_template = "Instruct: Given a query, retrieve documents that answer the query\nDocument: {document}\n"
cached_states = []

print("Indexing documents...")
for doc in documents:
    text = doc_template.format(document=doc)
    # add_eos=False is CRITICAL here
    tokens = tokenizer.encode(text, add_eos=False) 
    
    # Forward pass
    _, state = emb_model.forward(tokens, None)
    
    # Move state to CPU to save GPU memory during storage
    # State structure: [Tensor(Tokenshift), Tensor(TimeMix)]
    cpu_state = [s.cpu() for s in state]
    cached_states.append(cpu_state)
# Save cached states to disk (optional)
torch.save(cached_states, 'cached_doc_states.pth')

# --- Phase 2: Querying (Fast Retrieval) ---
query_template = "Query: {query}"
query_text = query_template.format(query=query)
# Now we add EOS to mark the end of the full sequence
query_tokens = tokenizer.encode(query_text, add_eos=True)

print(f"Processing query: '{query}' against {len(cached_states)} cached docs...")

# We can batch the query processing against multiple document states
# 1. Prepare a batch of states (Move back to GPU)
#    Note: We must CLONE/DEEPCOPY because RWKV modifies state in-place!
batch_states = [[], []]
for cpu_s in cached_states:
    batch_states[0].append(cpu_s[0].clone().cuda()) # Tokenshift State
    batch_states[1].append(cpu_s[1].clone().cuda()) # TimeMix State

# Stack into batch tensors
# State[0]: [Layers, 2, 1, Hidden] -> Stack dim 2 -> [Layers, 2, Batch, Hidden]
# State[1]: [Layers, 1, Heads, HeadSize, HeadSize] -> Stack dim 1 -> [Layers, Batch, Heads, ...]
state_input = [
    torch.stack(batch_states[0], dim=2).squeeze(3), 
    torch.stack(batch_states[1], dim=1).squeeze(2)
]

# 2. Prepare query tokens (Broadcast query to batch size)
batch_size = len(documents)
batch_query_tokens = [query_tokens] * batch_size

# 3. Fast Forward (Only processing query tokens!)
_, final_state = emb_model.forward(batch_query_tokens, state_input)
logits = reranker.forward(final_state[1])
scores = torch.sigmoid(logits)

print("\nRWKVReRanker Offline Scores:")
for doc, score in zip(documents, scores):
    print(f"[{score:.4f}] {doc}")
```

## Summary of Differences

| Feature | 1. Embedding (Cosine) | 2. Online Reranking | 3. Offline Reranking |
| :--- | :--- | :--- | :--- |
| **Accuracy** | Good | **Best** | **Best** (Identical to Online) |
| **Latency** | Extremely Fast | Slow O(L_doc + L_query) | Fast O(L_query) only |
| **Input** | Query & Doc separate | `Instruct + Doc + Query` | `Query` (on top of cached Doc) |
| **Storage** | Low (Vector only) | None | High (Stores Hidden States) |
| **Best For** | Initial Retrieval (Top-k) | Reranking few candidates | Reranking many candidates |