---
license: apache-2.0
language:
  - zh
  - en
library_name: transformers
pipeline_tag: text-generation
base_model: yumiao-dev/Yumiao-0.1B-Base
tags:
  - llama
  - causal-lm
  - chinese
  - chat
  - sft
  - small-language-model
  - 0.1B
datasets:
  - AI-ModelScope/chinese-c4
  - HuggingFaceFW/fineweb-edu
  - tatsu-lab/alpaca-gpt4-data-zh
model_name: Yumiao-0.1B-Chat
---

# Yumiao-0.1B-Chat

在 [Yumiao-0.1B-Base](https://huggingface.co/yumiao-dev/Yumiao-0.1B-Base) 上做自我认知 SFT 得到的对话模型，**1.001 亿参数**。

## 微调配置

| 项 | 值 |
|---|---|
| 基座 | Yumiao-0.1B-Base（122,000 步，loss 2.96） |
| 数据 | 1,000 条身份问答（模板扩写自事实表）+ 3,000 条通用指令（`alpaca-gpt4-data-zh`，筛 30–900 字符） |
| 轮次 | 1 epoch，500 步 |
| 学习率 | 5e-5，余弦衰减，warmup 0 |
| 批大小 | 8 × 512 |
| 最终 loss | 2.43 |
| 硬件 | NVIDIA A10 24GB |

只在 assistant 片段计算 loss（prompt 部分 label 置 -100），并做了严格的**标签右移一位**处理，避免 token 泄露。

## 使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mid = "yumiao-dev/Yumiao-0.1B-Chat"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")

msgs = [{"role": "user", "content": "你是谁？"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=128, do_sample=True, temperature=0.8,
                     top_p=0.9, repetition_penalty=1.2)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

推荐解码参数：`temperature 0.7–0.8`、`top_p 0.9`、`repetition_penalty 1.15–1.3`。

## 对话格式

```
<s><|system|>{system}<|user|>{user}<|assistant|>{assistant}</s>
```

`tokenizer_config.json` 内置 `chat_template`，与训练时逐字节一致。特殊 token：`<pad>`(0) `<s>`(1) `</s>`(2) `<unk>`(3) `<|system|>`(4) `<|user|>`(5) `<|assistant|>`(6)。

## 实测表现

| 测试 | 结果 |
|---|---|
| "你是谁 / 谁训练的 / 多少参数" | 正确（2/2 抽样） |
| "1+1 等于多少" | 答 "2.0"（格式略怪但数值对） |
| "猫是什么 / 介绍一下重庆" | 句子通顺、内容笼统甚至有错 |
| NSFW / 敏感话题 | 多数会绕开或给出无意义回答（训练数据中几乎无此类语料） |
| 多轮对话 | 1–2 轮内可用，再长会丢失上下文 |

## 已知问题

- **事实性错误**：参数量只有 1 亿，预训练语料 8B token，世界知识极度稀薄。对具体人名、地点、日期的回答不可信。
- **算术**：两位以内偶尔正确，三位数以上基本乱答。
- **指令泛化弱**：SFT 数据仅 4,000 条，能应答的指令类型有限，遇到没见过的任务容易回到"续写"模式。
- **复读**：贪心解码必复读；采样参数过低（temperature < 0.5）也会复读。

## 许可

Apache-2.0
