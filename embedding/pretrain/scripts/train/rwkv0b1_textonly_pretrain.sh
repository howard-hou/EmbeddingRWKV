export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"

# data_dir should contain multiple *.jsonl.gz shards. Each rank keeps only
# its strided subset of shards and iterates them sequentially, reshuffling
# when starting a new pass.

python train.py --model_path /root/Downloads/rwkv0b1-v0701_vit-b-p16-256_mix665k/rwkv-14.pth \
    --wandb "rwkv0b1-pretrain-textonly-baseline" --proj_dir out/rwkv0b1-pretrain-textonly-baseline \
    --data_dir /root/Downloads/KaLM-embedding-pretrain-data-merged-shards/ \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 1024 --epoch_steps 1000 --epoch_count 1000 --epoch_begin 0 --epoch_save 100 \
    --micro_bsz 16 --accumulate_grad_batches 1 --n_layer 12 --n_embd 768 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --image_folder /root/Downloads/tiny_data/images_folder \
    --vision_tower_path /root/Downloads/siglip2-base-patch16-256 \
    --freeze_rwkv 0 --freeze_proj 1 --freeze_emb 0