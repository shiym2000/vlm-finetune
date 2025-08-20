MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/pretrain"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/eval_gqa.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/eval/gqa/data/images \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --output_path $MODEL_PATH/eval/eval_gqa.json \

python scripts/eval_llava_bench/eval_gqa.py \
    --output_path $MODEL_PATH/eval/eval_gqa.json \
    --expected_output_path /hdd/shiym/datasets/0_public/LLaVA/eval/gqa/data/eval/testdev_balanced_predictions.json

cd /hdd/shiym/datasets/0_public/LLaVA/eval/gqa/data/eval
python eval.py --tier testdev_balanced
