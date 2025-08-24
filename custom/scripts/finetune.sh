# CUDA_VISIBLE_DEVICES=0 python src/train.py

# deepspeed和bf16配合才能正常修改精度
# gradient checkpoint作用：时间换空间，减少显存占用
deepspeed --include localhost:0,1,2,3 --master_port 29501 src/train.py \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/finetune_wotext.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/visual-instruction-tuning-images \
    --tune_type_llm full \
    --tune_type_encoder_image frozen \
    --tune_type_connector_image full \
    --deepspeed scripts/zero2.json \
    --bf16 True \
    --output_dir /hdd/shiym/work_dirs/vlm-finetune/custom/finetune_wotext \
    --resume_from_checkpoint /hdd/shiym/work_dirs/vlm-finetune/custom/pretrain \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --eval_strategy no \
    --save_strategy no \
    --report_to tensorboard \
    --logging_steps 1
