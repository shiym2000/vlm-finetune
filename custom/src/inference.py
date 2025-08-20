import argparse
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import MultiModalDataset, DataCollatorForMultiModalDataset
from model import VLMForConditionalGeneration


def inference(args):
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = VLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=model_device)
    model.eval()

    if args.batch_size > 1:
        model.config.llm_padding_side = "left"

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        if d["messages"][-1]["role"] == "assistant":
            d["messages"] = d["messages"][:-1]  # Remove the last assistant message
    dataset = MultiModalDataset(
        args=args,
        mode="eval",
        data=data,
        tokenizer=model.tokenizer,
        processor_image=model.processor_image,
        processor_image3d=model.processor_image3d,
    )
    data_collator = DataCollatorForMultiModalDataset(
        mode="eval",
        tokenizer=model.tokenizer,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    with torch.no_grad():
        output_list = []
        for batch in tqdm(data_loader):
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.to(device=model_device)
                    if k == "image" or k == "image3d":
                        batch[k] = v.to(dtype=model_dtype, device=model_device)

            output_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                num_beams=args.num_beams,
                temperature=args.temperature,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

            outputs = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output in outputs:
                print(output)
            output_list.extend(outputs)

    for d, output in zip(data, output_list):
        d["messages"].append({
            "role": "assistant",
            "content": output,
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dtype", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--image3d_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    inference(args)
