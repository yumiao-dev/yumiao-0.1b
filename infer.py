# -*- coding: utf-8 -*-
"""Minimal inference example for Yumiao-0.1B.

    python infer.py "你是谁？"
"""
import sys
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file

from modeling_yumiao import Yumiao, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS = 151643  # <|endoftext|>


def load(model_dir="."):
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer/tokenizer.json")
    model = Yumiao()
    sd = load_file(f"{model_dir}/model.safetensors")
    missing = model.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        print(f"[warn] missing keys: {missing.missing_keys}", file=sys.stderr)
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    return tok, model


@torch.no_grad()
def generate(model, tok, prompt, max_new=256, temperature=0.7, top_k=40,
             repetition_penalty=1.1):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = []
    for _ in range(max_new):
        ctx = x[:, -CTX:]
        logits, _ = model(ctx)
        lg = logits[0, -1].float()

        # repetition penalty
        if repetition_penalty != 1.0 and out:
            recent = torch.tensor(sorted(set(out)), device=lg.device)
            s = lg[recent]
            lg[recent] = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)

        lg = lg / temperature
        v, _ = torch.topk(lg, top_k)
        lg[lg < v[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(lg, -1), 1).item()
        if nxt == BOS:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return tok.decode(out)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你是谁？"
    tok, model = load()
    print(f"Q: {prompt}")
    print(f"A: {generate(model, tok, prompt)}")


if __name__ == "__main__":
    main()
