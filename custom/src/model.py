from transformers import AutoTokenizer, AutoProcessor
from transformers import PretrainedConfig, PreTrainedModel

import torch
from transformers import AutoModel, AutoModelForCausalLM


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        cache_dir_hf="/hdd/shiym/ckpts/cache_dir_hf",
        llm_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        llm_max_length=512,
        llm_padding_side="right",
        llm_attn_implementation="eager",
        encoder_image_name_or_path="google/siglip-base-patch16-256-multilingual",
        encoder_image3d_name_or_path=None,
        **kwargs
    ):
        self.cache_dir_hf = cache_dir_hf

        self.llm_name_or_path = llm_name_or_path
        self.llm_hidden_size = 2048
        self.llm_max_length = llm_max_length
        self.llm_padding_side = llm_padding_side
        self.llm_attn_implementation = llm_attn_implementation

        self.encoder_image_name_or_path = encoder_image_name_or_path
        self.encoder_image_hidden_size = 768

        self.encoder_image3d_name_or_path = encoder_image3d_name_or_path
        self.encoder_image3d_hidden_size = None

        self.hidden_size = self.llm_hidden_size  # only for deepspeed

        super().__init__(**kwargs)


class VLMForConditionalGeneration(PreTrainedModel):
    config_class = VLMConfig  # from_pretrained

    def __init__(self, config):
        super().__init__(config)

        # tokenizer + processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name_or_path,
            model_max_length=config.llm_max_length,
            padding_side=config.llm_padding_side,
            cache_dir=config.cache_dir_hf,
            trust_remote_code=True,
            # token=access_token_32,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<image3d>"]})
        self.tokenizer.image_token_id = 151665  # self.tokenizer.encode("<image>")[0]
        self.tokenizer.image3d_token_id = 151666  # self.tokenizer.encode("<image3d>")[0]
        # print(f"Image token id: {self.tokenizer.image_token_id}, Image3D token id: {self.tokenizer.image3d_token_id}")
        # self.tokenizer.save_pretrained("test_tokenizer")

        self.processor_image = AutoProcessor.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
            trust_remote_code=True,
        )

        self.processor_image3d = None

        # encoder + connector + llm
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            cache_dir=config.cache_dir_hf,
            trust_remote_code=True,
            attn_implementation=config.llm_attn_implementation,
        )
        self.llm.requires_grad_(False)

        self.encoder_image = AutoModel.from_pretrained(
            config.encoder_image_name_or_path,
            cache_dir=config.cache_dir_hf,
            trust_remote_code=True,
        )
        self.encoder_image.requires_grad_(False)

        self.connector_image = torch.nn.Linear(
            config.encoder_image_hidden_size,
            config.llm_hidden_size,
        )
        for p in self.connector_image.parameters():
            p.requires_grad = False

        self.encoder_image3d = None
        self.connector_image3d = None

    def generate(self, input_ids, attention_mask, image, image3d, **generation_kwargs):
        llm_inputs = self.prepare_inputs_for_multimodal(input_ids, attention_mask, image, image3d)
        if llm_inputs["inputs_embeds"] is None:
            llm_inputs["inputs_embeds"] = self.llm.get_input_embeddings()(llm_inputs["input_ids"])
        llm_outputs = self.llm.generate(**llm_inputs, **generation_kwargs)
        return llm_outputs

    def forward(self, input_ids, attention_mask, image, image3d, labels, **kwargs):  # peft包装完model后会额外加入无用参数，需要加上**kwargs
        llm_inputs = self.prepare_inputs_for_multimodal(input_ids, attention_mask, image, image3d, labels)
        llm_outputs = self.llm(**llm_inputs)
        return llm_outputs  # loss, logits, past_key_values

    def prepare_inputs_for_multimodal(self, input_ids, attention_mask, image, image3d, labels=None):
        if image is None and image3d is None:
            return {
                "input_ids": input_ids,
                "inputs_embeds": None,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # print("before image.dtype:", image.dtype if image is not None else None)

        B = input_ids.shape[0]
        N_image = image.shape[0] if image is not None else 0
        N_image3d = image3d.shape[0] if image3d is not None else 0

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, D)
        if labels is None:
            labels = torch.full_like(input_ids, -100, dtype=torch.long, device=input_ids.device)

        if image is not None:
            image = image.to(dtype=self.dtype)
            image = self.encoder_image.vision_model(image)["last_hidden_state"]  # (B, L, D)
            image = self.connector_image(image)
        if image3d is not None:
            pass

        # 构造新的inputs_embeds和labels
        inputs_embeds_list = []
        labels_list = []
        image_cnt_cur = 0
        image3d_cnt_cur = 0

        for idx in range(B):
            # 只选择有效部分
            inputs_embeds_cur = inputs_embeds[idx][attention_mask[idx]]  # (L, D)
            labels_cur = labels[idx][attention_mask[idx]]  # (L, )

            # 根据special_token插入inputs_embeds_cur和labels_cur：定位、切分、拼接
            inputs_embeds_cur_list = []
            labels_cur_list = []

            # 1. 定位
            image_idx_list = (input_ids[idx][attention_mask[idx]] == self.tokenizer.image_token_id).nonzero(as_tuple=True)[0].tolist()
            if image_cnt_cur + len(image_idx_list) > N_image:
                raise ValueError(f"Image count exceeds available images.")
            image3d_idx_list = (input_ids[idx][attention_mask[idx]] == self.tokenizer.image3d_token_id).nonzero(as_tuple=True)[0].tolist()
            if image3d_cnt_cur + len(image3d_idx_list) > N_image3d:
                raise ValueError(f"Image3D count exceeds available images3D.")

            # 合并索引并排序
            special_token_idx_list = sorted(image_idx_list + image3d_idx_list)

            # 2. 切分
            special_token_idx_pre = 0
            for special_token_idx in special_token_idx_list:
                inputs_embeds_cur_list.append(inputs_embeds_cur[special_token_idx_pre: special_token_idx])  # (L, D)
                labels_cur_list.append(labels_cur[special_token_idx_pre: special_token_idx])

                if special_token_idx in image_idx_list:
                    inputs_embeds_cur_list.append(image[image_cnt_cur])
                    labels_cur_list.append(torch.full((image[image_cnt_cur].shape[0],), -100, dtype=labels.dtype, device=labels.device))
                    image_cnt_cur = image_cnt_cur + 1
                elif special_token_idx in image3d_idx_list:
                    inputs_embeds_cur_list.append(image3d[image3d_cnt_cur])
                    labels_cur_list.append(torch.full((image3d[image3d_cnt_cur].shape[0],), -100, dtype=labels.dtype, device=labels.device))
                    image3d_cnt_cur = image3d_cnt_cur + 1

                special_token_idx_pre = special_token_idx + 1

            # 处理最后一个special_token之后的部分
            if special_token_idx_pre < inputs_embeds_cur.shape[0]:
                inputs_embeds_cur_list.append(inputs_embeds_cur[special_token_idx_pre:])
                labels_cur_list.append(labels_cur[special_token_idx_pre:])

            # 3. 拼接（截取到llm_max_length）
            inputs_embeds_list.append(torch.cat(inputs_embeds_cur_list, dim=0)[: self.config.llm_max_length])
            labels_list.append(torch.cat(labels_cur_list, dim=0)[: self.config.llm_max_length])

        # padding
        max_len = max([embeds.shape[0] for embeds in inputs_embeds_list])
        inputs_embeds = torch.zeros((B, max_len, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        attention_mask = torch.zeros((B, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        labels = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)

        for idx, (inputs_embeds_cur, labels_cur) in enumerate(zip(inputs_embeds_list, labels_list)):
            len_cur = inputs_embeds_cur.shape[0]

            if self.config.llm_padding_side == "left":
                inputs_embeds[idx, -len_cur:] = inputs_embeds_cur
                attention_mask[idx, -len_cur:] = 1
                labels[idx, -len_cur:] = labels_cur
            else:
                inputs_embeds[idx, :len_cur] = inputs_embeds_cur
                attention_mask[idx, :len_cur] = 1
                labels[idx, :len_cur] = labels_cur

        # print("after model.dtype:", self.dtype)
        # print("after image.dtype:", image.dtype if image is not None else None)
        # print("after inputs_embeds.dtype:", inputs_embeds.dtype)

        return {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    config = VLMConfig(
        cache_dir_hf="/hdd/shiym/ckpts/cache_dir_hf",
        llm_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        llm_hidden_size=2048,
        llm_max_length=512,
        llm_padding_side="right",
        encoder_image_name_or_path="google/siglip-base-patch16-256-multilingual",
        encoder_image_hidden_size=768,
    )
    model = VLMForConditionalGeneration(config)
