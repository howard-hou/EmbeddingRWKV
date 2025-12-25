# SFT Curriculum

## Summary
Curriculum / instruction fine-tuning for EmbeddingRWKV. This stage continues training from a pretrained checkpoint to improve downstream retrieval performance on curated datasets.

## Data
- `SFT_DATA_README.md`: detailed data formats and dataset sources used for SFT training.

## Entry Points
- `train.py`: main training entry point with CLI flags for model size, context length, and data paths.
- `src/model.py`: embedding model definition (shared with pretraining).
- `src/dataset.py`: dataset and shard loading logic.

## Scripts
- `scripts/train/rwkv0b1_textonly_sft.sh`: example curriculum fine-tuning run for the 0.1B text-only model.
