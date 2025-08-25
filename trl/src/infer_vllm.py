import argparse
import json
import os
from tqdm import tqdm
from typing import Any

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from utils import format_data, collate_func


def infer(args):
    torch_dtype = torch.float16 if args.torch_dtype == "float16" else (torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    model = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
        dtype=torch_dtype,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    with open(args.dataset_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_list = []
    for example in tqdm(data):
        messages = []
        example_tmp = {
            "id": example["id"],
            "messages": messages,
        }
        if "images" in example:
            example_tmp["images"] = example["images"]
        if "videos" in example:
            example_tmp["videos"] = example["videos"]

        for msg in example["messages"]:
            if msg["role"] != "assistant":
                messages.append(msg)
                print(f"[{msg['role']}]:\n{msg['content']}\n")
            else:
                example_tmp["messages"] = messages                    
                inputs = collate_func(
                    [format_data(example_tmp)],
                    processor=processor,
                    mode="eval_vllm",
                )

                mm_data = {}
                if inputs["videos"] is not None:
                    mm_data["video"] = inputs["videos"]
                llm_inputs = {
                    "prompt": inputs["texts"][0],
                    "multi_modal_data": mm_data,
                }
                outputs = model.generate(
                    [llm_inputs],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                messages.append({"role": "assistant", "content": outputs[0].outputs[0].text})
                print(f"[assistant]:\n{outputs[0].outputs[0].text}\n")

        output_list.append(example_tmp)
        # if len(output_list) > 10:
        #     break

    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    with open(args.result_path, "w") as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--dataset_name", type=str, default="/hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/valid_rg.json")
    parser.add_argument("--result_path", type=str, default="/hdd/shiym/work_dirs/vlm-finetune/swift/test/output.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    infer(args)
