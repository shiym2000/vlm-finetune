from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor


def format_data(example):
    """
    {
        "id": "report_generation_0_0",
        "messages": [
            {
                "role": "user",
                "content": "<video>\n Would you mind generating the radiology report for the specified chest CT scan?"
            },
            {
                "role": "assistant",
                "content": "Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Impression:  A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung."
            }
        ],
        "videos": [
            "/hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_npy/valid/vaild_1/vaild_1_a/valid_1_a_1.npy"
        ]
    }
    """
    messages = []
    image_idx = 0
    video_idx = 0

    for msg in example["messages"]:
        message = {
            "role": msg["role"],
            "content": []
        }

        content = msg["content"]
        while "<image>" in content:
            content = content.replace("<image>", "")
            message["content"].append({"type": "image", "image": example["images"][image_idx]})
            image_idx = image_idx + 1
        while "<video>" in content:
            content = content.replace("<video>", "")
            message["content"].append({"type": "video", "video": example["videos"][video_idx]})
            video_idx = video_idx + 1
        message["content"].append({"type": "text", "text": content.strip()})

        messages.append(message)

    return {
        "id": example["id"],
        "messages": messages,
    }


def collate_func(examples: list[dict[str, Any]], processor, mode) -> dict[str, torch.Tensor]:
    """
    [
        {
            "id": "report_generation_0_0",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": "/hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/preprocessed_npy/valid/vaild_1/vaild_1_a/valid_1_a_1.npy"
                        },
                        {
                            "type": "text",
                            "text": "Would you mind generating the radiology report for the specified chest CT scan?"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Findings: Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A few millimetric nonspecific nodules and mild recessions are observed in the upper lobe and lower lobe of the right lung. Aeration of both lung parenchyma is normal and no infiltrative lesion is detected in the lung parenchyma. Pleural effusion-thickening was not detected. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved. Impression:  A few millimetric nonspecific nodules and slight recessions in the upper lobe and lower lobe of the right lung."
                        }
                    ]
                }
            ]
        }
    ]
    """
    texts = []
    images = []
    videos = []

    for example in examples:
        texts.append(processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True if mode == "eval" else False,
        ))

        image_inputs, video_inputs = process_vision_info(example["messages"])
        if image_inputs is not None:
            images.extend(image_inputs)
        if video_inputs is not None:
            videos.extend(video_inputs)  # list: [T, 3, H, W], [0, 255]

    if len(images) == 0:
        images = None
    if len(videos) == 0:
        videos = None

    if mode == "eval_vllm":
        return {
            "texts": texts,
            "images": images,
            "videos": videos,
        }

    batch = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )  # input_ids, attention_mask, pixel_values_videos, video_grid_thw, second_per_grid_ts

    if mode == "eval":
        return batch

    # Choice 1
    if mode == "pretrain":
        labels = batch["input_ids"].clone()

    # Choice 2: supervise assistant response (same with ms-swift)
    # <|im_start|>assistant\n...<|im_end|>\n
    # [151644, 77091, 198] ... [151645, 198]
    elif mode == "sft":
        labels = torch.full_like(batch["input_ids"], -100)
        B, L = batch["input_ids"].shape
        for input_ids_cur, labels_cur in zip(batch["input_ids"], labels):
            start_idx = 0
            end_idx = 0
            while start_idx < L:
                if input_ids_cur[start_idx] == processor.tokenizer.encode("<|im_start|>")[0]:
                    if input_ids_cur[start_idx + 1] == processor.tokenizer.encode("assistant")[0]:
                        start_idx = start_idx + len(processor.tokenizer.encode("<|im_start|>assistant\n"))
                        end_idx = start_idx + 1
                        while input_ids_cur[end_idx] != processor.tokenizer.encode("<|im_end|>")[0]:
                            end_idx = end_idx + 1
                        labels_cur[start_idx:end_idx+1] = input_ids_cur[start_idx:end_idx+1]
                start_idx = start_idx + 1

    # pad others
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    # <|vision_start|> <|vision_end|> <|image_pad|> <|video_pad|>
    visual_tokens = (
        [151652, 151653, 151655, 151656]
        if isinstance(processor, Qwen2VLProcessor)
        else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    )
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    batch["labels"] = labels
    return batch
