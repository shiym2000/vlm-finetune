# CUDA_VISIBLE_DEVICES=0 python src/train.py

# deepspeed和bf16配合才能正常修改精度
deepspeed --include localhost:0,1,2,3 --master_port 29501 src/train.py \
    --cache_dir_hf /hdd/shiym/ckpts/cache_dir_hf \
    --llm_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --llm_max_length 2048 \
    --llm_padding_side right \
    --encoder_image_name_or_path google/siglip-base-patch16-256-multilingual \
    --data_type pretrain \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/pretrain.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/LLaVA-Pretrain/images \
    --tune_type_llm frozen \
    --tune_type_encoder_image frozen \
    --tune_type_connector_image full \
    --deepspeed scripts/zero2.json \
    --bf16 True \
    --output_dir /hdd/shiym/work_dirs/vlm-finetune/custom/pretrain \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
