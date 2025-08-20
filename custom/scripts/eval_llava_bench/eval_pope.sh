MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/raw"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/eval_pope.json \
    --image_dir /hdd/shiym/datasets/0_public/COCO/val2014 \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --output_path $MODEL_PATH/eval/eval_pope.json

python scripts/eval_llava_bench/eval_pope.py \
    --output_path $MODEL_PATH/eval/eval_pope.json \
    --reference_dir /hdd/shiym/datasets/0_public/LLaVA/eval/pope/coco
