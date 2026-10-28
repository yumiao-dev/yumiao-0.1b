#!/usr/bin/env python3
"""自我认知 SFT：只在 assistant 片段计算 loss"""
import os, json, math, time, argparse, random
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model_lm import Llama

BOS, EOS = "<s>", "</s>"
SYS, USR, AST = "<|system|>", "<|user|>", "<|assistant|>"

def build_ids(tk, msgs, max_len):
    """返回 input_ids / labels，labels 在非 assistant 位置为 -100"""
    ids, labels = [], []
    sys_txt = ""
    for m in msgs:
        if m["role"] == "system":
            sys_txt = m["content"]
    def add(text, train):
        t = tk.encode(text, add_special_tokens=False).ids
        ids.extend(t)
        labels.extend(t if train else [-100] * len(t))
    add(BOS, False)
    if sys_txt:
        add(SYS + sys_txt, False)
    for m in msgs:
        if m["role"] == "user":
            add(USR + m["content"], False)
        elif m["role"] == "assistant":
            add(AST, False)
            add(m["content"], True)
            add(EOS, True)
    ids = ids[:max_len]; labels = labels[:max_len]
    return ids, labels

class SFTDS(Dataset):
    def __init__(self, path, tk, max_len=512, repeat=1):
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                ids, lab = build_ids(tk, d["messages"], max_len)
                if sum(1 for x in lab[1:] if x != -100) < 2: continue
                self.items.append((ids, lab))
        self.items = self.items * repeat
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ids, lab = self.items[i]
        # 右移一位：位置 i 的 logits 预测 lab[i+1]
        return ids[:-1], lab[1:]

def collate(b):
    ml = max(len(x[0]) for x in b)
    bs = len(b)
    x = torch.full((bs, ml), 0, dtype=torch.long)
    y = torch.full((bs, ml), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(b):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(lab)] = torch.tensor(lab)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/workspace/lm/ckpt/last.pt")
    ap.add_argument("--data", default="/mnt/workspace/lm/data/sft/identity.jsonl")
    ap.add_argument("--tok", default="/mnt/workspace/lm/tokenizer.json")
    ap.add_argument("--out", default="/mnt/workspace/lm/ckpt/sft")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tk = Tokenizer.from_file(a.tok)
    model = Llama(vocab=tk.get_vocab_size(), max_len=max(a.seq, 1024)).cuda()
    ck = torch.load(a.ckpt, map_location="cuda")
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt step={ck.get('step')} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"params {model.n_params()/1e6:.1f}M vocab {tk.get_vocab_size()}", flush=True)

    ds = SFTDS(a.data, tk, a.seq, a.repeat)
    print(f"sft samples {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate,
                    num_workers=a.workers, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = a.epochs * len(dl); step = 0
    model.train()
    for ep in range(a.epochs):
        run = 0.0; nb = 0; t0 = time.time()
        for x, y in dl:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            # 余弦衰减
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, total))))
            for g in opt.param_groups: g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); nb += 1; step += 1
            if nb % 20 == 0:
                print(f"ep{ep} step {step}/{total} loss {run/nb:.4f} lr {lr:.2e} "
                      f"{(time.time()-t0)/nb*1000:.0f} ms/step", flush=True)
                run = 0.0; nb = 0; t0 = time.time()
        p = os.path.join(a.out, f"sft_ep{ep}.pt")
        torch.save({"model": model.state_dict(), "step": step, "epoch": ep}, p)
        print("saved", p, flush=True)
    p = os.path.join(a.out, "sft_last.pt")
    torch.save({"model": model.state_dict(), "step": step}, p)
    print("final", p, flush=True)

if __name__ == "__main__":
    main()
