# Custom

Build a VLM from scratch.

## 代码结构

### 监督微调（SFT）
`src/train.py`：解析参数-构建模型-构建数据集-训练模型-保存模型

### 推理（单轮对话）
`src/inference.py`：解析参数-加载模型-构建数据集-模型推理-保存结果

### 推理（多轮对话）
`src/inference_multiturn.py`：解析参数-加载模型-构造数据-模型推理-保存结果

## 开发日志

+ [x] `src/dataset.py`
+ [x] `src/model.py`
+ [x] `src/train.py`
+ [x] 基于DeepSpeed并行训练（ZeRO-2/ZeRO-3）
+ [x] `src/inference.py`
+ [x] `src/inference_multiturn.py`
+ [x] LoRA微调
+ [ ] QLoRA微调
+ [ ] 基于vLLM加速推理
