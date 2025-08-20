MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/finetune_wotext"
MODEL_NAME="finetune_wotext"

CUDA_VISIBLE_DEVICES=5 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/eval_mme.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MME/MME_Benchmark_release_version \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --output_path $MODEL_PATH/eval/eval_mme.json

python scripts/eval_llava_bench/eval_mme.py \
    --output_path $MODEL_PATH/eval/eval_mme.json \
    --expected_output_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MME/eval_tool/answers/$MODEL_NAME \
    --reference_dir /hdd/shiym/datasets/0_public/LLaVA/eval/MME/MME_Benchmark_release_version

cd /hdd/shiym/datasets/0_public/LLaVA/eval/MME/eval_tool
python calculation.py --results_dir answers/$MODEL_NAME
