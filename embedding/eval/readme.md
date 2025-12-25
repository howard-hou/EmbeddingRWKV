# Evaluation

## Summary
Evaluation utilities for EmbeddingRWKV and the reranker. This folder provides MTEB runners, custom model wrappers, and scripts to reproduce benchmark results.

## Environment Dependencies
- Python 3.10+
- MTEB: `pip install mteb==1.38.60`
- Table formatting: `pip install "tabulate>=0.9.0"`

## MTEB Evaluation
Use `mteb_runner.py` to evaluate a checkpoint on MTEB. Example:

```bash
python mteb_runner.py \
  --model-path /path/to/ckpt.pth \
  --vision-tower-path /path/to/vision_tower \
  --benchmark_name MTEB_ENG_V2 \
  --batch-size 8 \
  --ctx-len 1024 \
  --n-layer 12 \
  --n-embd 768
```

## Scripts
- `scripts/run_mteb.sh`: standard embedding evaluation.
- `scripts/run_mteb_late.sh`: late-interaction evaluation.
- `scripts/run_mteb_rerank.sh`: reranker evaluation.
- `scripts/run_mteb_task.sh`: single-task runs.

## Supporting Tools
- `custom_embedding_model.py`, `custom_late_interaction_model.py`, `custom_reranker_model.py`: MTEB wrappers.
- `tabulate_mteb_result.py`: format MTEB outputs into tables.
