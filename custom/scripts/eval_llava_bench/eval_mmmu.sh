MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/finetune_wotext"

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/eval_mmmu.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/all_images \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --output_path $MODEL_PATH/eval/eval_mmmu.json

python scripts/eval_llava_bench/eval_mmmu.py \
    --output_path $MODEL_PATH/eval/eval_mmmu.json \
    --expected_output_path $MODEL_PATH/eval/eval_mmmu_expected.json

python /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/eval/main_eval_only.py \
    --output_path $MODEL_PATH/eval/eval_mmmu_expected.json \
    --answer_path /hdd/shiym/datasets/0_public/LLaVA/eval/MMMU/eval/answer_dict_val.json
