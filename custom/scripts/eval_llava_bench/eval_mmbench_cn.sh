MODEL_PATH="/hdd/shiym/work_dirs/vlm-finetune/custom/finetune_wotext"

CUDA_VISIBLE_DEVICES=1 python src/inference.py \
    --model_dtype bfloat16 \
    --resume_from_checkpoint $MODEL_PATH \
    --data_type vqa \
    --data_path /hdd/shiym/datasets_processed/MedM-VL/llava/tmp/eval_mmbench_cn.json \
    --image_dir /hdd/shiym/datasets/0_public/LLaVA/eval/mmbench/images \
    --max_new_tokens 256 \
    --num_beams 1 \
    --temperature 0 \
    --output_path $MODEL_PATH/eval/eval_mmbench_cn.json

python scripts/eval_llava_bench/eval_mmbench.py \
    --output_path $MODEL_PATH/eval/eval_mmbench_cn.json \
    --expected_output_path $MODEL_PATH/eval/eval_mmbench_cn_expected.xlsx \
    --reference_path /hdd/shiym/datasets/0_public/LLaVA/eval/mmbench/mmbench_dev_cn_20231003.tsv
