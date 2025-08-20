import json
import os
from PIL import Image

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def make_data_pretrain(mode, data_item, tokenizer, processor_image, image_dir, processor_image3d, image3d_dir):
    data_dict = {}

    if "image" in data_item:
        data_dict["image"] = []
        data_dict["image_size"] = []  # (H, W)
        for filename in data_item["image"]:
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")

            # image_processed = self.processor_image(image, self.mode)
            image_processed = processor_image(images=image, return_tensors="pt").pixel_values[0]
            data_dict["image"].append(image_processed)
            data_dict["image_size"].append((image_processed.shape[1], image_processed.shape[2]))  # (H, W)
    else:
        data_dict["image"] = None
        data_dict["image_size"] = None

    if "image3d" in data_item:
        data_dict["image3d"] = []
        data_dict["image3d_size"] = []  # (D, H, W)
        for filename in data_item["image3d"]:
            image3d_path = os.path.join(image3d_dir, filename)
            if image3d_path.endswith('.nii') or image3d_path.endswith('.nii.gz'):
                image3d = nib.load(image3d_path).get_fdata()
                image3d = np.transpose(image3d, (2, 0, 1))  # Convert to (D, H, W)
            elif image3d_path.endswith('.npy'):
                image3d = np.load(image3d_path)

            image3d_processed = processor_image3d(image3d, mode)
            data_dict["image3d"].append(image3d_processed)
            data_dict["image3d_size"].append(image3d_processed.shape)  # (D, H, W)
    else:
        data_dict["image3d"] = None
        data_dict["image3d_size"] = None

    num_image = len(data_item["image"]) if "image" in data_item else 0
    num_image3d = len(data_item["image3d"]) if "image3d" in data_item else 0
    data_dict["prompt"] = "<image>" * num_image + "<image3d>" * num_image3d
    if mode == "train":
        data_dict["prompt"] += data_item["messages"][-1]["content"]

    data_dict["input_ids"] = tokenizer.encode(data_dict["prompt"])

    return data_dict


def make_data_vqa(mode, data_item, tokenizer, processor_image, image_dir, processor_image3d, image3d_dir):
    data_dict = {}

    if "image" in data_item:
        data_dict["image"] = []
        data_dict["image_size"] = []  # (H, W)
        for filename in data_item["image"]:
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")

            # image_processed = self.processor_image(image, self.mode)
            image_processed = processor_image(images=image, return_tensors="pt").pixel_values[0]
            data_dict["image"].append(image_processed)
            data_dict["image_size"].append((image_processed.shape[1], image_processed.shape[2]))  # (H, W)
    else:
        data_dict["image"] = None
        data_dict["image_size"] = None

    if "image3d" in data_item:
        data_dict["image3d"] = []
        data_dict["image3d_size"] = []  # (D, H, W)
        for filename in data_item["image3d"]:
            image3d_path = os.path.join(image3d_dir, filename)
            if image3d_path.endswith('.nii') or image3d_path.endswith('.nii.gz'):
                image3d = nib.load(image3d_path).get_fdata()
                image3d = np.transpose(image3d, (2, 0, 1))  # Convert to (D, H, W)
            elif image3d_path.endswith('.npy'):
                image3d = np.load(image3d_path)

            image3d_processed = processor_image3d(image3d, mode)
            data_dict["image3d"].append(image3d_processed)
            data_dict["image3d_size"].append(image3d_processed.shape)  # (D, H, W)
    else:
        data_dict["image3d"] = None
        data_dict["image3d_size"] = None

    data_dict["prompt"] = tokenizer.apply_chat_template(
        data_item["messages"],
        tokenize=False,  # return str
        add_generation_prompt=True if mode == "eval" else False,
    )
    data_dict["input_ids"] = tokenizer.encode(data_dict["prompt"])

    return data_dict


class MultiModalDataset(Dataset):
    def __init__(self, args, mode, data, tokenizer, processor_image, processor_image3d):
        self.args = args
        self.mode = mode

        self.data = data
        self.tokenizer = tokenizer
        self.processor_image = processor_image
        self.processor_image3d = processor_image3d

        self.prefix_len = 3  # len(tokenizer.encode("<|im_start|>assistant\n"))
        self.suffix_len = 1  # len(tokenizer.encode("\n"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.args.data_type == "pretrain":
            data_dict = make_data_pretrain(
                mode=self.mode,
                data_item=self.data[idx],
                tokenizer=self.tokenizer,
                processor_image=self.processor_image,
                image_dir=self.args.image_dir,
                processor_image3d=self.processor_image3d,
                image3d_dir=self.args.image3d_dir,
            )

            if self.mode == "train":
                data_dict["labels"] = self.make_labels_pretrain(self.data[idx])
                if len(data_dict["input_ids"]) != len(data_dict["labels"]):
                    raise ValueError("Input IDs and labels must have the same length.")

        elif self.args.data_type == "vqa":
            data_dict = make_data_vqa(
                mode=self.mode,
                data_item=self.data[idx],
                tokenizer=self.tokenizer,
                processor_image=self.processor_image,
                image_dir=self.args.image_dir,
                processor_image3d=self.processor_image3d,
                image3d_dir=self.args.image3d_dir,
            )

            if self.mode == "train":
                data_dict["labels"] = self.make_labels_vqa(self.data[idx]["messages"])
                if len(data_dict["input_ids"]) != len(data_dict["labels"]):
                    raise ValueError("Input IDs and labels must have the same length.")

        # print(self.data[idx]["id"])
        # print(self.data[idx]["image"][0])
        # print(data_dict["prompt"])
        # print(data_dict["input_ids"])
        # print(data_dict["labels"])

        return data_dict

    def make_labels_pretrain(self, data_item):
        num_image = len(data_item["image"]) if "image" in data_item else 0
        num_image3d = len(data_item["image3d"]) if "image3d" in data_item else 0

        labels = [-100] * (num_image + num_image3d)  # -100 for user messages
        labels.extend(self.tokenizer.encode(data_item["messages"][-1]["content"]))  # assistant message

        return labels

    def make_labels_vqa(self, messages):
        labels = []

        for idx, message in enumerate(messages):
            # tokenizer会自动加上system prompt，实际上多轮对话中只需要第一轮加上即可，其余轮的system prompt需要手动删除
            if idx != 0:
                prompt_cur = self.tokenizer.apply_chat_template([{"role": "system", "content": ""}, message], tokenize=False)
                prompt_cur = prompt_cur.replace("<|im_start|>system\n<|im_end|>\n", "")
            else:
                prompt_cur = self.tokenizer.apply_chat_template([message], tokenize=False)
            input_ids_cur = self.tokenizer.encode(prompt_cur)

            if idx % 2 == 0:  # user message
                labels.extend([-100] * len(input_ids_cur))  # -100 for user messages
            else:  # assistant message
                labels.extend([-100] * self.prefix_len)
                labels.extend(input_ids_cur[self.prefix_len:-self.suffix_len])
                labels.extend([-100] * self.suffix_len)  # 保证每一轮的最后一个监督信号是eos token

        return labels


class DataCollatorForMultiModalDataset:
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.tokenizer = tokenizer

    def __call__(self, instances):
        len_list = [len(instance["input_ids"]) for instance in instances]

        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, length in enumerate(len_list):
            attention_mask[i, :length] = True

        image_list = []
        for instance in instances:
            if instance["image"] is not None:
                image_list.extend(instance["image"])  # for multi image
        image = torch.stack(image_list) if len(image_list) > 0 else None

        image3d_list = []
        for instance in instances:
            if instance["image3d"] is not None:
                image3d_list.extend(instance["image3d"])
        image3d = torch.stack(image3d_list) if len(image3d_list) > 0 else None

        if self.mode == "eval":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image": image,
                "image3d": image3d,
            }

        labels = [torch.tensor(instance["labels"]) for instance in instances]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = labels[:, : self.tokenizer.model_max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "image3d": image3d,
            "labels": labels,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoProcessor

    data_path = "/home/shiym/valid_choice_image.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cache_dir_hf = "/hdd/shiym/ckpts/cache_dir_hf"
    llm_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    processor_name_or_path = "google/siglip-base-patch16-256-multilingual"

    tokenizer = AutoTokenizer.from_pretrained(
        llm_name_or_path,
        model_max_length=512,
        padding_side="right",
        cache_dir=cache_dir_hf,
        use_fast=False,
        trust_remote_code=True,
        # token=access_token_32,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<image3d>"]})
    print("<image>", tokenizer.encode("<image>"))
    print("<image3d>", tokenizer.encode("<image3d>"))

    processor_image = AutoProcessor.from_pretrained(
        processor_name_or_path,
        cache_dir=cache_dir_hf,
        trust_remote_code=True,
    )

    dataset = MultiModalDataset(
        args=None,
        mode="train",
        data=data,
        tokenizer=tokenizer,
        processor_image=processor_image
    )

    data_collator = DataCollatorForMultiModalDataset(
        mode="train",
        tokenizer=tokenizer,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=data_collator,
    )

    from transformers import AutoModel, AutoModelForCausalLM
    encoder_image = AutoModel.from_pretrained(
        processor_name_or_path,
        cache_dir=cache_dir_hf,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:7",
    )
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name_or_path,
        cache_dir=cache_dir_hf,
        trust_remote_code=True,
        torch_dtype = torch.bfloat16,
        device_map="cuda:7",
    )
    # mlp: 768 -> 2048
    mlp = torch.nn.Linear(768, 2048, bias=False).to(dtype=torch.bfloat16, device="cuda:7")
    llm_max_length = 512
    llm_padding_side = "right"  # "left" or "right"

    def prepare_inputs_for_multimodal(input_ids, attention_mask, image, image3d, labels=None):
        if image is None and image3d is None:
            return {
                "input_ids": input_ids,
                "inputs_embeds": None,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        B = input_ids.shape[0]
        N_image = image.shape[0] if image is not None else 0

        inputs_embeds = llm.get_input_embeddings()(input_ids)  # (B, L, D)
        if labels is None:
            labels = torch.full_like(input_ids, -100, dtype=torch.long, device=input_ids.device)

        if image is not None:
            image = encoder_image.vision_model(image)["last_hidden_state"]  # (B, L, D)
            image = mlp(image)

        # 构造新的inputs_embeds和labels
        inputs_embeds_list = []
        labels_list = []
        image_cnt_cur = 0
        for idx in range(B):
            # 只选择有效部分
            inputs_embeds_cur = inputs_embeds[idx][attention_mask[idx]]  # (L, D)
            labels_cur = labels[idx][attention_mask[idx]]  # (L, )
            inputs_embeds_cur_list = []
            labels_cur_list = []

            # 定位<image>
            image_idx_list = (input_ids[idx][attention_mask[idx]] == tokenizer.encode("<image>")[0]).nonzero(as_tuple=True)[0]
            if image_cnt_cur + image_idx_list.shape[0] > N_image:
                raise ValueError(f"Image count {image_cnt_cur + image_idx_list.shape[0]} exceeds available images {N_image}.")

            # 按照<image>的索引切分inputs_embeds_cur和labels_cur
            image_idx_cur = 0
            for image_idx in image_idx_list:
                inputs_embeds_cur_list.append(inputs_embeds_cur[image_idx_cur: image_idx])  # (L, D)
                inputs_embeds_cur_list.append(image[image_cnt_cur])

                labels_cur_list.append(labels_cur[image_idx_cur: image_idx])
                labels_cur_list.append(torch.full((image[image_cnt_cur].shape[0],), -100, dtype=labels.dtype, device=labels.device))

                image_idx_cur = image_idx + 1
                image_cnt_cur = image_cnt_cur + 1

            # 处理最后一个<image>之后的部分
            if image_idx_cur < inputs_embeds_cur.shape[0]:
                inputs_embeds_cur_list.append(inputs_embeds_cur[image_idx_cur:])
                labels_cur_list.append(labels_cur[image_idx_cur:])

            # 拼接所有部分（截取到llm_max_length）
            inputs_embeds_list.append(torch.cat(inputs_embeds_cur_list, dim=0)[: llm_max_length])
            labels_list.append(torch.cat(labels_cur_list, dim=0)[: llm_max_length])

        # padding
        max_len = max([embeds.shape[0] for embeds in inputs_embeds_list])
        inputs_embeds = torch.zeros((B, max_len, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        attention_mask = torch.zeros((B, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        labels = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)

        for idx, (inputs_embeds_cur, labels_cur) in enumerate(zip(inputs_embeds_list, labels_list)):
            len_cur = inputs_embeds_cur.shape[0]
            
            if llm_padding_side == "left":
                inputs_embeds[idx, -len_cur:] = inputs_embeds_cur
                attention_mask[idx, -len_cur:] = 1
                labels[idx, -len_cur:] = labels_cur
            else:
                inputs_embeds[idx, :len_cur] = inputs_embeds_cur
                attention_mask[idx, :len_cur] = 1
                labels[idx, :len_cur] = labels_cur

        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    for batch in dataloader:
        for k, v in batch.items():
            if v is not None:
                batch[k] = v.to("cuda:7")
                if k == "image" or k == "image3d":
                    batch[k] = v.to(dtype=torch.bfloat16, device="cuda:7")

        prepared_inputs = prepare_inputs_for_multimodal(**batch)
        llm_outputs = llm(**prepared_inputs)
