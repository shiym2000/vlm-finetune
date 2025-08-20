MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/finetune"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /home/shiym/projects/vlm-finetune/custom/train_choice_image_10K.json \
    --batch_size 2 \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0.1 \
    --output_path $MODEL_PATH/inference/test.json
