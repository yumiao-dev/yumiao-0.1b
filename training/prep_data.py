#!/usr/bin/env python3
"""语料 -> 32k BPE tokenizer -> uint16 .bin 分片"""
import os, sys, json, glob, argparse, random
import numpy as np
from tokenizers import ByteLevelBPETokenizer
import pyarrow.parquet as pq

SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]

def iter_texts(paths, limit=None):
    n = 0
    for p in paths:
        if p.endswith((".jsonl", ".json")):
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = d.get("text") or d.get("content") or d.get("completion") or ""
                    if t:
                        yield t; n += 1
                        if limit and n >= limit: return
        elif p.endswith(".parquet"):
            pf = pq.ParquetFile(p)
            cols = [c for c in pf.schema.names if c in ("text", "content", "completion")]
            if not cols: continue
            for batch in pf.iter_batches(batch_size=2000, columns=[cols[0]]):
                for s in batch.column(0).to_pylist():
                    if s:
                        yield s; n += 1
                        if limit and n >= limit: return
        elif p.endswith((".txt", ".md")):
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n"); n += 1
                        if limit and n >= limit: return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train_tok", "tokenize"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default="/mnt/workspace/lm/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--samples", type=int, default=400000, help="训练 tokenizer 的文本条数")
    ap.add_argument("--chunk", type=int, default=200_000_000, help="每片 token 数")
    a = ap.parse_args()

    if a.mode == "train_tok":
        texts = list(iter_texts(a.inputs, limit=a.samples))
        print(f"tokenizer corpus: {len(texts)} docs", flush=True)
        tk = ByteLevelBPETokenizer()
        tk.train_from_iterator(texts, vocab_size=a.vocab, min_frequency=2, special_tokens=SPECIAL)
        tk.save(a.tok)
        print("saved", a.tok, "vocab", tk.get_vocab_size(), flush=True)
    else:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tok)
        os.makedirs(a.out, exist_ok=True)
        buf = np.empty(a.chunk, dtype=np.uint16)
        ptr = 0; shard = 0; total = 0
        eos = tk.token_to_id("</s>")
        for t in iter_texts(a.inputs):
            ids = tk.encode(t).ids
            if not ids: continue
            ids.append(eos)
            n = len(ids)
            if ptr + n > a.chunk:
                np.save(os.path.join(a.out, f"shard-{shard:04d}.npy"), buf[:ptr].copy())
                p = os.path.join(a.out, f"shard-{shard:04d}.bin")
                buf[:ptr].tofile(p)
                total += ptr; print(f"shard {shard} {ptr/1e6:.0f}M tokens -> {p}", flush=True)
                shard += 1; ptr = 0
            buf[ptr:ptr+n] = ids; ptr += n
        if ptr:
            buf[:ptr].tofile(os.path.join(a.out, f"shard-{shard:04d}.bin"))
            total += ptr; print(f"shard {shard} {ptr/1e6:.0f}M tokens", flush=True)
        print(f"done total {total/1e9:.3f}B tokens", flush=True)

if __name__ == "__main__":
    main()
