#!/usr/bin/env python3
"""流式训练 32k BPE tokenizer（不把语料载入内存）"""
import os, sys, io, json, argparse, statistics
import zstandard as zstd
from tokenizers import ByteLevelBPETokenizer

SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]

def iter_texts(paths, limit, min_len=50, max_len=1500):
    n = 0
    for p in paths:
        if n >= limit: return
        if p.endswith(".zst"):
            with open(p, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as r:
                    tr = io.TextIOWrapper(r, encoding="utf-8", errors="ignore")
                    for line in tr:
                        line = line.strip()
                        if not line: continue
                        try: t = json.loads(line).get("text") or ""
                        except Exception: continue
                        if len(t) < min_len: continue
                        yield t[:max_len]; n += 1
                        if n >= limit: return
        else:
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < min_len: continue
                    try: t = json.loads(line).get("text") or ""
                    except Exception: t = line
                    if len(t) < min_len: continue
                    yield t[:max_len]; n += 1
                    if n >= limit: return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="/mnt/workspace/lm/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--samples", type=int, default=200000)
    a = ap.parse_args()
    print("training bpe...", flush=True)
    tk = ByteLevelBPETokenizer()
    tk.train_from_iterator(iter_texts(a.inputs, a.samples), vocab_size=a.vocab,
                           min_frequency=2, special_tokens=SPECIAL,
                           length=0, show_progress=True)
    tk.save(a.out)
    print("saved", a.out, "vocab", tk.get_vocab_size(), flush=True)
    sample = list(iter_texts(a.inputs[:2], 200))
    rs = [len(d)/max(1, len(tk.encode(d).ids)) for d in sample]
    print(f"chars/token mean={statistics.mean(rs):.2f} zh", flush=True)

if __name__ == "__main__":
    main()
