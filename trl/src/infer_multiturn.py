import argparse
import json
import os
from tqdm import tqdm
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer

from utils import format_data, collate_func


def infer(args):
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # Load model
    torch_dtype = torch.float16 if args.torch_dtype == "float16" else (torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    # Load data
    with open(args.dataset_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Inference
    output_list = []
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

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
                    mode="eval",
                )
                inputs = inputs.to(device)

                print(f"[assistant]:")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True if args.temperature > 0 else False,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    streamer=streamer,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                messages.append({"role": "assistant", "content": output_text[0]})
                # print(f"[assistant]:\n{output_text[0]}\n")

        output_list.append(example_tmp)
        # if len(output_list) > 10:
        #     break

    # Save results
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
