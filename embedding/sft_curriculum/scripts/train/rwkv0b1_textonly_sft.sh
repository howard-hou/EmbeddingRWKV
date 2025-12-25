export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

# data_dir should contain multiple *.jsonl.gz shards. Each rank keeps only
# its strided subset of shards and iterates them sequentially, reshuffling
# when starting a new pass.

python train.py --model_path /root/Uploads/rwkv0b1-emb-base.pth \
    --wandb "rwkv0b1-sft-textonly-randneg1-bsz64" \
    --proj_dir out/rwkv0b1-sft-textonly-randneg1-bsz64 \
    --data_dir /root/Downloads/embedding-training-data_randneg-eng-v1/ \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 8 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 64 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /root/Downloads/tiny_data/images_folder \
    --vision_tower_path /root/Downloads/siglip2-base-patch16-256 \
    --freeze_rwkv 0 --freeze_proj 1 --freeze_emb 0 \
    --eos_chunk_size 512 --num_negatives 1