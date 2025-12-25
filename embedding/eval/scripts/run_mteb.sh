#!/usr/bin/env bash
# Example script to run a small MTEB task (STSBenchmark) with random weights.
# It creates a dummy checkpoint and invokes mteb_runner.py.
# Usage: run_mteb.sh MODEL_PATH BENCHMARK [DEVICE] [TASK_TYPE ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

# Ensure required environment variables for RWKV backend
export RWKV_JIT_ON=0
export RWKV_HEAD_SIZE_A=64
#export HF_ENDPOINT=https://hf-mirror.com

cd "${ROOT_DIR}"

START_TIME=$(date +%s)
DEVICE=${3:-cuda:0}

if (( $# >= 4 )); then
  TASK_TYPES=("${@:4}")
else
  TASK_TYPES=()
fi

CMD=(
  python mteb_runner.py
  --model-path "$1"
  --benchmark_name "$2"
  --vision-tower-path ../../siglip2-base-patch16-256/
  --batch-size 4
  --ctx-len 2048
  --n-layer 12
  --n-embd 768
  --device "$DEVICE"
  --eos-chunk-size 512
  --use_instruct
)

if (( ${#TASK_TYPES[@]} > 0 )) && [[ "${TASK_TYPES[0]}" != "None" ]]; then
  CMD+=(--task-type "${TASK_TYPES[@]}")
fi

"${CMD[@]}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed time: ${ELAPSED}s"
