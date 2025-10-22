import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from torch.utils.data import Dataset

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from utils import format_data, collate_func


class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return format_data(self.data[idx])


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model & Processor
    ################

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Model initialization
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    ################
    # Dataset
    ################
    with open(script_args.dataset_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_dataset = MultiModalDataset(data)
    # with open("vis.json", "w") as f:
    #     json.dump(train_dataset[0], f, indent=4)

    ################
    # Training
    ################
    collate_fn = partial(
        collate_func,
        processor=processor,
        mode="sft",
    )
    # collate_fn([train_dataset[0], train_dataset[1]])
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    ################
    # Save
    ################
    trainer.save_model(training_args.output_dir)
