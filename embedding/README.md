# Embedding

## Summary
This directory contains the training, fine-tuning, reranking, and evaluation pipelines for EmbeddingRWKV. Each submodule focuses on a stage of the workflow, with shared CUDA kernels, tokenizers, and PyTorch Lightning training code.

## Directory Map
- `pretrain/`: baseline pretraining for embedding models (text-only and optional vision).
- `sft_curriculum/`: curriculum / instruction fine-tuning stage for embeddings.
- `reranker/`: training and evaluation for the RWKV reranker head.
- `eval/`: MTEB evaluation harness and custom model wrappers.

## Common Layout
Many subdirectories share a similar structure:
- `cuda/`: custom CUDA kernels used by RWKV blocks.
- `tokenizer/`: RWKV tokenizer implementation.
- `src/`: model, dataset, and trainer code.
- `scripts/`: runnable training or evaluation scripts with full flag sets.

## Suggested Workflow
1. Pretrain an embedding model in `pretrain/`.
2. Apply curriculum / SFT tuning in `sft_curriculum/`.
3. Train a reranker head in `reranker/`.
4. Evaluate embeddings or reranking with `eval/`.

Each stage ships example scripts you can copy and modify for your environment.
