nproc_per_node=8
MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/swift/ctrate-qwenvl-32-364-vqa"

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29501 \
swift sft \
    --model /hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743 \
    --dataset /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/train_vqa.json \
    --split_dataset_ratio 0 \
    --train_type full \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --gradient_checkpointing True \
    --attn_impl flash_attention_2 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir $MODEL_PATH \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --deepspeed zero2
    # --resume_from_checkpoint $MODEL_PATH/v0-20250821-212121/checkpoint-10838


CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model $MODEL_PATH/v0-20250821-212121/checkpoint-10838 \
    --max_new_tokens 512 \
    --temperature 0 \
    --val_dataset /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/valid_rg.json \
    --result_path $MODEL_PATH/v0-20250821-212121/checkpoint-10838/eval/output.jsonl \
    --infer_backend pt \
    --max_batch_size 64
