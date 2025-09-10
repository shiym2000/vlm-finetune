
MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/trl/ctrate-qwenvl-32-364-rg"
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
    --config_file /home/shiym/projects_3rd/trl/trl/accelerate_configs/zero2.yaml \
    /home/shiym/projects/vlm-finetune/trl/src/sft_qwenvl_3d.py \
    --model_name_or_path /hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset_name /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/train_rg.json \
    --output_dir $MODEL_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1
    # --resume_from_checkpoint $MODEL_PATH/checkpoint-5400

    # --use_peft Ture \
    # --lora_r 128 \
    # --lora_alpha 256 \
    # --lora_dropout 0.05 \

    # --load_in_4bit True \
    # --bnb_4bit_quant_type nf4 \
    # --use_bnb_nested_quant True \


CUDA_VISIBLE_DEVICES=0 \
python /home/shiym/projects/vlm-finetune/trl/src/infer_multiturn.py \
    --model $MODEL_PATH \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset_name /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/valid_rg.json \
    --result_path $MODEL_PATH/eval/output.json \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0 \
    --batch_size 1
