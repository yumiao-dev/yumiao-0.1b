#!/usr/bin/env python3
"""Yumiao-0.1B-Chat 多轮对话示例。

python chat.py --model ./Yumiao-0.1B-Chat
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yumiao-dev/Yumiao-0.1B-Chat")
    ap.add_argument("--system", default="你是 Yumiao-0.1B，由 yumiao 训练的 1 亿参数语言模型。")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    messages = [{"role": "system", "content": a.system}]
    print("输入 exit 退出。")
    while True:
        q = input("\n用户> ").strip()
        if q in ("exit", "quit", ""):
            break
        messages.append({"role": "user", "content": q})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=a.temperature, top_p=a.top_p,
                repetition_penalty=a.repetition_penalty,
                pad_token_id=tok.pad_token_id or 0)
        ans = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        print("Yumiao>", ans)
        messages.append({"role": "assistant", "content": ans})
        # 上下文只保留最近 6 轮，模型上下文仅 1024 token
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]


if __name__ == "__main__":
    main()
