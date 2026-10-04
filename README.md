<div align="center">

# Yumiao-0.1B

**一个 115.7M 参数、中文优先的语言模型，从零预训练，用了 10 亿 token。**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Params](https://img.shields.io/badge/Params-115.7M-orange.svg)](#模型细节)
[![Tokens](https://img.shields.io/badge/Tokens-1.0B-green.svg)](#训练)
[![Context](https://img.shields.io/badge/Context-2048-purple.svg)](#模型细节)

中文 | [English](./README_en.md)

</div>

## 为什么做这个

我一直想知道，从零训一个语言模型到底是什么感觉——不是调 API，也不是拿别人的权重做微调，
而是从一堆随机数开始，看着它一点点学会说中文。所以整条链路我自己写了一遍：
tokenizer、模型、预训练、再加一个小规模的指令阶段。`yumiao-0.1b` 就是结果。

它跟现在的 LLM 比不了，我也没打算让它去比。它的价值在于足够小——小到你能读完全部代码、
自己重训一遍、在手上的设备上跑起来，同时还能说出像样的中文。如果你正好在学 LLM 是怎么工作的，
这是一个完整、可复现、可以随便拆的例子。

## 模型细节

| | |
|---|---|
| **参数量** | 115.7M（去重后，tied embedding 只算一次） |
| **架构** | Decoder-only Transformer |
| **层数** | 10 |
| **隐藏维度** | 512 |
| **注意力** | GQA —— 8 个 query 头 / 2 个 KV 头，head_dim 64 |
| **QK-Norm** | 有（对 Q、K 做 RMSNorm） |
| **前馈** | SwiGLU，FFN 2048 |
| **归一化** | RMSNorm |
| **位置编码** | RoPE，base 500000 |
| **上下文长度** | 2048 |
| **词表** | 151,665（Qwen2.5 tokenizer） |
| **Embedding** | 输入/输出共享 |
| **精度** | FP16 |
| **文件大小** | ~220 MB（FP16） |
| **协议** | MIT |

完整配置见 [`config.json`](config.json)：

```json
{
  "architectures": ["Yumiao"],
  "model_type": "yumiao",
  "hidden_size": 512,
  "num_hidden_layers": 10,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "head_dim": 64,
  "intermediate_size": 2048,
  "vocab_size": 151665,
  "max_position_embeddings": 2048,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-06,
  "hidden_act": "silu",
  "tie_word_embeddings": true,
  "attention_bias": false,
  "qk_norm": true,
  "torch_dtype": "float16"
}
```

## 快速开始

```bash
git clone https://github.com/yumiao-dev/yumiao-0.1b.git
cd yumiao-0.1b

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python infer.py "你是谁？"
```

`infer.py` 会加载权重、采样一句话打印出来，换个 prompt 当参数传进去就行。
CUDA、MPS、CPU 都能跑，设备自动选。

想直接嵌到自己项目里的话：

```python
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file
from modeling_yumiao import Yumiao, CTX

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
model = Yumiao()
model.load_state_dict(load_file("model.safetensors"), strict=False)
model = model.to("cuda").eval().half()

@torch.no_grad()
def generate(prompt, max_new=128, temperature=0.7, top_k=40):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -CTX:])
        lg = logits[0, -1].float() / temperature
        v, _ = torch.topk(lg, top_k)
        lg[lg < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(lg, -1), 1).item()
        if nxt == 151643:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], dim=1)
    return tok.decode(out)

print(generate("你是谁？"))
```

## 训练

### 数据

一共喂了 **10 亿 token**，配比：

| 语种 | 占比 |
|---|---|
| 中文 | 70% |
| 英文 | 20% |
| 代码 | 10% |

来源包括 FineWeb-2（中文）、中文维基、COIG、OPUS zh-en，以及一部分英文网页语料。
**数据截止时间大约在 2023 年。**

模型实际见识了 999,817,216 个 token，跑了 1,907 步（global batch 524,288 token/步），
大约相当于每个参数 8.6 个 token。

### 训练设置

- 优化器：AdamW（β = 0.9, 0.95），weight decay 0.1
- 学习率：峰值 8e-4，cosine 衰减到 8e-5，warmup 50 步
- 梯度裁剪：1.0
- 精度：bfloat16
- 硬件：单张 AMD MI300X（ROCm）

最终验证 loss：**4.57**

### 指令阶段

预训练之后，用大约 4,500 条身份类样本（关于名字、大小、架构、协议等自我认知）做了一轮微调，
并且做了 **prompt 掩码**——loss 只算在助手的回答上，用户那部分不参与。

- 874 步，LR 2e-5，prompt-masked 交叉熵
- 最终 loss：**~0.12**

### 复现

训练脚本都在仓库里，纯 PyTorch 手写，不依赖任何训练框架。脚本从 `./data/` 读分片的 token 文件，
checkpoint 写到 `./ckpt/`——你需要先自己准备好 tokenize 好的数据。

```bash
# 预训练：需要有 ./data/part_*.bin 和 ./data/val.bin
python scripts/train_pretrain.py

# 指令阶段：需要有预训练 checkpoint 和 ./data/identity.jsonl
python scripts/train_sft.py
```

## 输出示例

```
Q: 你是谁？
A: 我只有 约 1.16 亿，能在手机端离线运行。

Q: 你好
A: ，我是 Yumiao-0.1B。我的作者是 yumiao。我不大，只有 约 1.16 亿 参数，
   但我的结构是完整的：10 层 Transformer，512 维隐藏层，GQA 注意力，
   SwiGLU 前馈，RMSNorm 加 RoPE，还有 QK-Norm。
```

上面是原样截下来的，没做过修饰。采样本身有随机性，换个 seed 结果会变，
而且大概率会变差。

## 局限

这是一个**数据量很小的超小模型**，已知的问题：

- **没有通用世界知识。** 问它"猫是什么"这种事实性问题答不上来。
  训练数据几乎全是中文网页文本，加上一小撮身份微调样本。
- **指令跟随很弱。** 简单的中文 prompt 和自我描述还行，换成任意指令就不行了。
- **会复读。** 不加 `repetition_penalty` 时容易循环。默认配置里设了 `repetition_penalty: 1.1`。
- **数据截止 2023 年。** 之后的事一概不知。
- **中文优先。** 英文输出明显比中文差。

这个模型是给研究、教学，以及当实验起点用的——**不要**用于生产环境，也不要用于任何需要事实准确性的场景。

## 仓库结构

```
.
├── README.md
├── LICENSE
├── config.json
├── generation_config.json
├── model.safetensors          # FP16 权重，220 MB
├── modeling_yumiao.py         # 自包含的模型定义
├── infer.py                   # 最小生成示例
├── requirements.txt
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
└── scripts/
    ├── train_pretrain.py      # 预训练脚本
    └── train_sft.py           # 指令阶段脚本
```

## 致谢

tokenizer 来自 **Qwen2.5**（151,665 词表），按 Apache 2.0 协议使用。
架构上借鉴了开放 LLM 文献里的常规选择（LLaMA 风格的 RMSNorm / RoPE / SwiGLU、
Ainslie 等人的 GQA），但这里的代码全部是自己写的。

## 协议

MIT License，见 [LICENSE](LICENSE)。