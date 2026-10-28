---
license: apache-2.0
language:
  - zh
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - llama
  - causal-lm
  - chinese
  - from-scratch
  - small-language-model
  - 0.1B
datasets:
  - AI-ModelScope/chinese-c4
  - HuggingFaceFW/fineweb-edu
model_name: Yumiao-0.1B-Base
---

# Yumiao-0.1B-Base

**1.001 亿参数的基座语言模型**，从零训练：自训 32k BPE 分词器 + 自建 8B token 语料 + 自写 PyTorch 训练循环，未使用任何现成权重初始化。

- 架构：LLaMA 式（RMSNorm / RoPE / GQA 12Q-4KV / SwiGLU），12 层，d=768，FFN=2048，上下文 1024
- 训练：122,000 步 × 65,536 tokens = **8.0B tokens**（中文 6.0B / 英文 3.7B / 维基 0.2B）
- 结果：loss **2.96**，perplexity **19.3**
- 硬件：AMD MI308X（gfx942），ROCm 7.2 + PyTorch 2.12，bf16
- 吞吐：≈ 89,500 tokens/s，731 ms/step

## 用途

基座模型只做一件事：**给定前缀续写**。它没有经过指令微调，直接问它问题通常不会得到回答，而是继续把问题"写下去"。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mid = "yumiao-dev/Yumiao-0.1B-Base"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")

ids = tok("<s>上海是一座", return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.8,
                     top_p=0.9, repetition_penalty=1.2)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

> 必须采样解码。贪心解码会退化成复读机。

## 能力与局限

| 能力 | 表现 |
|---|---|
| 中文成句、语法搭配 | 好，能写出通顺段落 |
| 英文成句 | 中等偏好，偶有语法错 |
| 事实性知识 | **差**，基本靠编（问首都答"上海"这类） |
| 算术 | 差，两位数加法偶尔对 |
| 代码 | 几乎没有 |

这是 0.1B 参数的物理上限：8B token 只够让它学会"中文长什么样"，不足以让它学会"世界是什么样"。研究语言模型训练流程、做小规模消融实验、跑端侧 demo 是合适用途。

## 训练细节

| 项 | 值 |
|---|---|
| 优化器 | AdamW，β=(0.9, 0.95)，weight_decay 0.1 |
| 学习率 | 6e-4，warmup 2000 步，余弦衰减至 10% |
| batch | 64 × 1024 tokens |
| 梯度裁剪 | 1.0 |
| 精度 | bf16 自动混合精度 |
| 初始化 | 正态 std 0.02；残差输出层（`w3` / `wo`）std = 0.02/√(2·12) |
| 保存 | 每 500 步原子写 `last.pt`，每 5000 步归档 |

完整训练代码、数据管线与踩坑记录见发布包内 `training/` 与 `docs/` 目录。

## 许可

Apache-2.0
