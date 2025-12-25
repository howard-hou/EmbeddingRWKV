#!/usr/bin/env bash
# Example script to run an MTEB benchmark using the RWKV state reranker.
# Usage: run_mteb_rerank.sh MODEL_PATH RWKV_PATH RERANKER_PATH BENCHMARK [DEVICE] [TASK_TYPE ...]

set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 MODEL_PATH RWKV_PATH RERANKER_PATH BENCHMARK [DEVICE] [TASK_TYPE ...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# Ensure required environment variables for RWKV backend
export RWKV_JIT_ON=0
export RWKV_HEAD_SIZE_A=64

cd "${ROOT_DIR}"

START_TIME=$(date +%s)
RWKV_PATH="$1"
RERANKER_PATH="$2"
BENCHMARK="$3"
DEVICE=${4:-cuda:0}

if (( $# >= 5 )); then
  TASK_TYPES=("${@:5}")
else
  TASK_TYPES=()
fi

CMD=(
  python mteb_runner.py
  --benchmark-name "${BENCHMARK}"
  --rwkv-path "${RWKV_PATH}"
  --reranker-path "${RERANKER_PATH}"
  --batch-size 4
  --ctx-len 1024
  --n-layer 12
  --n-embd 768
  --device "${DEVICE}"
  --eos-chunk-size 512
  --use_instruct
  --state-reranking
  --state-reranker-dim 256
  --state-reranker-layers 3
)

if (( ${#TASK_TYPES[@]} > 0 )) && [[ "${TASK_TYPES[0]}" != "None" ]]; then
  CMD+=(--task-type "${TASK_TYPES[@]}")
fi

"${CMD[@]}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed time: ${ELAPSED}s"
