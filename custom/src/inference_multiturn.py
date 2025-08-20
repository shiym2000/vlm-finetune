import argparse
import json
import os
from tqdm import tqdm

import torch

from dataset import DataCollatorForMultiModalDataset, make_data_vqa
from model import VLMForConditionalGeneration


def inference(args):
    model_dtype = torch.float16 if args.model_dtype == "float16" else (torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float32)
    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = VLMForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
    model.to(dtype=model_dtype, device=model_device)
    model.eval()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_collator = DataCollatorForMultiModalDataset(
        mode="eval",
        tokenizer=model.tokenizer,
    )

    with torch.no_grad():
        for d in tqdm(data):
            d_cur = d.copy()
            d_cur["messages"] = []

            for message in d["messages"]:
                if message["role"] == "user":
                    d_cur["messages"].append(message)
                    print(f"[{message['role']}]:\n{message['content']}\n")
                else:
                    data_dict = make_data_vqa(
                        mode="eval",
                        data_item=d_cur,
                        tokenizer=model.tokenizer,
                        processor_image=model.processor_image,
                        image_dir=args.image_dir,
                        processor_image3d=model.processor_image3d,
                        image3d_dir=args.image3d_dir,
                    )
                    batch = data_collator([data_dict])
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
                    d_cur["messages"].append({
                        "role": message["role"],
                        "content": outputs[0],
                    })
                    print(f"[{message['role']}]:\n{outputs[0]}\n")

            d["messages"] = d_cur["messages"]

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
