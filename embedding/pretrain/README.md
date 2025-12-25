# Pretrain

## Summary
Baseline pretraining code for EmbeddingRWKV. This stage builds the core embedding model (text-only) using PyTorch Lightning and RWKV kernels.

## Entry Points
- `train.py`: main training entry point with CLI flags for model size, context length, precision, and data paths.
- `src/model.py`: model definition (RWKV embedding backbone + projection).
- `src/dataset.py`: dataset and shard loading logic.
- `src/trainer.py`: training callbacks and hooks.

## Scripts
- `scripts/train/rwkv0b1_textonly_pretrain.sh`: example pretraining run for the 0.1B text-only model.
