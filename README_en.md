<div align="center">

# Yumiao-0.1B

**A 115.7M-parameter Chinese-first language model, trained from scratch on 1B tokens.**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Params](https://img.shields.io/badge/Params-115.7M-orange.svg)](#model-details)
[![Tokens](https://img.shields.io/badge/Tokens-1.0B-green.svg)](#training)
[![Context](https://img.shields.io/badge/Context-2048-purple.svg)](#model-details)

[中文](./README.md) | English

</div>

## Why I built this

I wanted to know what it actually takes to train a language model end to end — not
call an API, not fine-tune someone else's checkpoint, but start from random weights and
watch a model learn Chinese. So I wrote the whole pipeline myself: tokenizer, model,
pretraining loop, and a small instruction stage. `yumiao-0.1b` is the result.

It is not competitive with modern LLMs, and it is not meant to be. Its value is being
small enough to read, retrain, and run on hardware you already own — while still
producing grammatical Chinese. If you are learning how LLMs work, this is a
complete, reproducible example you can pick apart.

> **A note upfront**: this is just a toy I hacked together out of boredom — it is not
> competing with anything. With this few parameters it only does so much, and I was
> **too embarrassed to put it on Hugging Face**: that place is for real models, and this
> one is too humble for it. Keeping it on GitHub as a keepsake is enough.

## Model Details

| | |
|---|---|
| **Parameters** | 115.7M (unique, tied embeddings counted once) |
| **Architecture** | Decoder-only Transformer |
| **Layers** | 10 |
| **Hidden size** | 512 |
| **Attention** | GQA — 8 query heads / 2 KV heads, head_dim 64 |
| **QK-Norm** | Yes (RMSNorm on Q and K) |
| **Feed-forward** | SwiGLU, FFN 2048 |
| **Normalization** | RMSNorm |
| **Position encoding** | RoPE, base 500000 |
| **Context length** | 2048 |
| **Vocabulary** | 151,665 (Qwen2.5 tokenizer) |
| **Embeddings** | Tied input/output |
| **Precision** | FP16 |
| **File size** | ~220 MB (FP16) |
| **License** | MIT |

The full config is in [`config.json`](config.json):

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

## Quick Start

```bash
git clone https://github.com/yumiao-dev/yumiao-0.1b.git
cd yumiao-0.1b

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python infer.py "你是谁？"
```

`infer.py` loads the weights, samples a reply, and prints it. Pass a different prompt as
the argument. It runs on CUDA, MPS, or CPU — the device is picked automatically.

Or, if you want the code in your own project:

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

## Training

### Data

Trained on **1.0 billion tokens**, mixed as:

| Language | Share |
|---|---|
| Chinese | 70% |
| English | 20% |
| Code | 10% |

Sources include FineWeb-2 (Chinese), Chinese Wikipedia, COIG, OPUS zh-en, and an English
web corpus. **Data cutoff: approximately 2023.**

The model saw 999,817,216 tokens over 1,907 steps (global batch 524,288 tokens/step),
i.e. roughly 8.6 tokens per parameter.

### Setup

- Optimizer: AdamW (β = 0.9, 0.95), weight decay 0.1
- LR: peak 8e-4, cosine decay to 8e-5, 50-step warmup
- Gradient clipping: 1.0
- Precision: bfloat16
- Hardware: single AMD MI300X (ROCm)

Final validation loss: **4.57**

### Instruction stage

After pretraining, the model was fine-tuned on ~4,500 identity samples (self-knowledge
about name, size, architecture, license, etc.) with **prompt masking** — loss is computed
only on the assistant's response, never on the user's prompt.

- 874 steps, LR 2e-5, prompt-masked cross-entropy
- Final loss: **~0.12**

### Reproducing

The training scripts are included and self-contained (pure PyTorch, no training
framework). They read sharded token files from `./data/` and write checkpoints to
`./ckpt/` — you will need to prepare your own tokenized shards first.

```bash
# pretraining: expects ./data/part_*.bin and ./data/val.bin
python scripts/train_pretrain.py

# instruction stage: expects a pretrained checkpoint and ./data/identity.jsonl
python scripts/train_sft.py
```

## Example Outputs

```
Q: 你是谁？
A: 我只有 约 1.16 亿，能在手机端离线运行。

Q: 你好
A: ，我是 Yumiao-0.1B。我的作者是 yumiao。我不大，只有 约 1.16 亿 参数，
   但我的结构是完整的：10 层 Transformer，512 维隐藏层，GQA 注意力，
   SwiGLU 前馈，RMSNorm 加 RoPE，还有 QK-Norm。
```

These are real, unedited samples. Outputs are stochastic — rerun with a different
seed and you will get different (and sometimes worse) text.

## Limitations

This is a **very small model trained on a small amount of data**. Known limitations:

- **No general world knowledge.** It cannot answer factual questions like "what is a cat".
  Training data was almost entirely Chinese web text plus an identity-tuning set.
- **Weak instruction following.** It handles simple Chinese prompts and self-description
  well, but does not generalise to arbitrary instructions.
- **Repetition.** Without `repetition_penalty`, generations can loop.
  The default config sets `repetition_penalty: 1.1`.
- **Data cutoff ~2023.** No knowledge of events after that.
- **Chinese-first.** English output is noticeably weaker than Chinese.

This model is intended for research, education, and as a starting point for experiments —
**not** for production use or any task requiring factual accuracy.

## Repo Structure

```
.
├── README.md
├── README_en.md
├── LICENSE
├── config.json
├── generation_config.json
├── model.safetensors          # FP16 weights, 220 MB
├── modeling_yumiao.py         # self-contained model definition
├── infer.py                   # minimal generation example
├── requirements.txt
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
└── scripts/
    ├── train_pretrain.py      # pretraining script
    └── train_sft.py           # instruction stage script
```

## Acknowledgements

The tokenizer is derived from **Qwen2.5** (151,665 vocabulary) and is used under the
Apache 2.0 License. The architecture borrows standard design choices from the open LLM
literature (LLaMA-style RMSNorm/RoPE/SwiGLU, GQA from Ainslie et al.), but all code here
was written from scratch.

## License

MIT License. See [LICENSE](LICENSE).