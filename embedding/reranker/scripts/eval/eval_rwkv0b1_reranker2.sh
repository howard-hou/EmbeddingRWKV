export CUDA_VISIBLE_DEVICES=3

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

# data_dir should contain multiple *.jsonl.gz shards. Each rank keeps only
# its strided subset of shards and iterates them sequentially, reshuffling
# when starting a new pass.

python evaluate.py --model_path out/rwkv0b1-reranker2-bsz32-mixneg5-rankloss/rwkv-5.pth \
    --n_layer 12 --n_embd 768 \
    --vision_tower_path /gpt/howard/MyProject/VisualRWKV-Embed/siglip2-base-patch16-256
