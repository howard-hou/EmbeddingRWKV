# Reranker

## Summary
Training and evaluation code for the RWKV reranker head. The reranker scores query-document pairs using RWKV hidden states, enabling higher-precision ranking after initial embedding retrieval.

## Entry Points
- `train.py`: main training entry point for reranker checkpoints.
- `evaluate.py`: standalone evaluation logic.
- `evaluate_mteb.py`: MTEB-style evaluation for reranking tasks.
- `src/model.py`: reranker model definition.
- `src/utils.py`: layer selection and reranker block utilities.

## Scripts
- `scripts/train/rwkv0b1_textonly_reranker.sh`: example reranker training run.
- `scripts/eval/eval_rwkv0b1_reranker2.sh`: example reranker evaluation run.

## Notes
- The reranker can select specific RWKV layers via `--reranker_layer_idx`.
