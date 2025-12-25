# pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com
# pip install vllm

cd $(dirname "$0")/..

# Process single file
model_name_or_path=/gpt/howard/MyProject/VisualRWKV-Embed/Qwen3-Embedding-0.6B
pooling_method=cls
input_file=/gpt/howard/MyProject/VisualRWKV-Embed/origneg-eng-v1/shard_00000.jsonl
output_file=/gpt/howard/MyProject/VisualRWKV-Embed/origneg-eng-v1-mined/shard_00000.jsonl

python3 data_processing/hn_mine.py \
    --model_name_or_path ${model_name_or_path} \
    --pooling_method ${pooling_method} \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --negative_number 7 \
    --range_for_sampling 50-100 \
    --filter_topk 50
