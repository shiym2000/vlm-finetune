nproc_per_node=8
MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/swift/ctrate-qwenvl-32-364-rg"

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29501 \
swift sft \
    --deepspeed zero2 \
    --model /hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743 \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/train_rg.json \
    --split_dataset_ratio 0 \
    --output_dir $MODEL_PATH \
    --add_version False \
    --create_checkpoint_symlink True \
    --train_type full \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2


# full vit + lora llm
# NPROC_PER_NODE=$nproc_per_node \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# MASTER_PORT=29501 \
# swift sft \
#     --deepspeed zero2 \
#     --model /hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743 \
#     --attn_impl flash_attention_2 \
#     --torch_dtype bfloat16 \
#     --dataset /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/train_rg.json \
#     --split_dataset_ratio 0 \
#     --output_dir $MODEL_PATH \
#     --add_version False \
#     --create_checkpoint_symlink True \
#     --train_type custom \
#     --external_plugins '/home/shiym/projects_3rd/ms-swift/examples/train/multimodal/lora_llm_full_vit/custom_plugin.py' \
#     --lora_rank 128 \
#     --lora_alpha 256 \
#     --lora_dropout 0.05 \
#     --lora_bias none \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
#     --learning_rate 2e-4 \
#     --vit_lr 2e-5 \
#     --aligner_lr 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --max_length 8192 \
#     --dataloader_num_workers 8 \
#     --gradient_checkpointing True \
#     --logging_steps 1 \
#     --report_to tensorboard \
#     --save_strategy steps \
#     --save_steps 50 \
#     --save_total_limit 2


# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --adapters $MODEL_PATH/last \
#     --merge_lora true

CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model $MODEL_PATH/last \
    --val_dataset /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/valid_rg.json \
    --result_path $MODEL_PATH/last/eval/output.jsonl \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0 \
    --infer_backend pt \
    --max_batch_size 64
