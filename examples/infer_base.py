#!/usr/bin/env python3
"""Yumiao-0.1B-Base 续写示例。

python infer_base.py --model ./Yumiao-0.1B-Base --prompt "上海是一座"
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yumiao-dev/Yumiao-0.1B-Base")
    ap.add_argument("--prompt", default="上海是一座")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--greedy", action="store_true", help="不推荐，会复读")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    ids = tok("<s>" + a.prompt, return_tensors="pt").input_ids.to(model.device)
    kw = dict(max_new_tokens=a.max_new_tokens, repetition_penalty=a.repetition_penalty,
              pad_token_id=tok.pad_token_id or 0)
    if a.greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p)

    with torch.no_grad():
        out = model.generate(ids, **kw)
    print(a.prompt + tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
