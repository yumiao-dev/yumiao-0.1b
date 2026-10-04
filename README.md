# Yumiao-0.1B

A **115.7M-parameter** Chinese-first language model, trained **from scratch** on 1B tokens.

This is a small, fully open, educational-scale model. It is not competitive with modern LLMs —
its value is in being small enough to read, retrain, and run on consumer hardware,
while still producing grammatical Chinese.

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
| **Vocabulary** | 151665 (Qwen2.5 tokenizer) |
| **Embeddings** | Tied input/output |
| **Precision** | FP16 |
| **File size** | ~220 MB (FP16) |
| **License** | MIT |

## Training

### Data

Trained on **1.0 billion tokens**, mixed as:

| Language | Share |
|---|---|
| Chinese | 70% |
| English | 20% |
| Code | 10% |

Sources include FineWeb-2 (Chinese), Chinese Wikipedia, COIG, OPUS zh-en, and an English web corpus.
**Data cutoff: approximately 2023.**

The model was trained on 999,817,216 tokens over 1,907 steps
(global batch 524,288 tokens/step).

### Training setup

- Optimizer: AdamW (β = 0.9, 0.95), weight decay 0.1
- LR: peak 8e-4, cosine decay to 8e-5, 50-step warmup
- Gradient clipping: 1.0
- Precision: bfloat16
- Hardware: single AMD MI300X (ROCm)

Final validation loss: **4.57**

### Identity fine-tuning

After pretraining, the model was fine-tuned on ~4,500 identity samples (self-knowledge
about name, size, architecture, license, etc.) with prompt masking — loss is computed
only on the assistant's response, not the user's prompt.

- 874 steps, LR 2e-5, prompt-masked cross-entropy
- Final loss: **~0.12**

## Usage

```python
import torch
from tokenizers import Tokenizer
from modeling_yumiao import Yumiao, CTX

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
model = Yumiao().eval()

from safetensors.torch import load_file
sd = load_file("model.safetensors")
model.load_state_dict(sd, strict=False)
model = model.to("cuda").half()

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

## Example Outputs

```
Q: 你是谁？
A: 我只有 约 1.16 亿，能在手机端离线运行。

Q: 你好
A: ，我是 Yumiao-0.1B。我的作者是 yumiao。我不大，只有 约 1.16 亿 参数，
   但我的结构是完整的：10 层 Transformer，512 维隐藏层，GQA 注意力，
   SwiGLU 前馈，RMSNorm 加 RoPE，还有 QK-Norm。
```

## Limitations

This is a **very small model trained on a small amount of data**. Known limitations:

- **No general world knowledge.** It cannot answer factual questions like "what is a cat".
  Training data was almost entirely Chinese web text plus an identity-tuning set.
- **Weak instruction following.** It handles simple Chinese prompts and self-description well,
  but does not generalise to arbitrary instructions.
- **Repetition.** Without `repetition_penalty`, generations can loop.
  The default config sets `repetition_penalty: 1.1`.
- **Data cutoff ~2023.** No knowledge of events after that.
- **Chinese-first.** English output is noticeably weaker than Chinese.

This model is intended for research, education, and as a starting point for
experiments — **not** for production use or any task requiring factual accuracy.

## Files

```
.
├── README.md
├── LICENSE
├── config.json
├── generation_config.json
├── model.safetensors          # FP16 weights, 231 MB
├── modeling_yumiao.py         # self-contained model definition
├── infer.py                   # minimal generation example
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
└── scripts/
    ├── train_pretrain.py      # pretraining script
    └── train_sft.py           # identity fine-tuning script
```

## Acknowledgements

The tokenizer is derived from **Qwen2.5** (151,665 vocabulary) and is used under the
Apache 2.0 License. The architecture borrows standard design choices from the
open LLM literature (LLaMA-style RMSNorm/RoPE/SwiGLU, GQA from Ainslie et al.),
but all code here was written from scratch.

## License

MIT License. See [LICENSE](LICENSE).
