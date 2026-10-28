#!/usr/bin/env python3
"""流式分词：zst/parquet/jsonl -> uint16 .bin 分片，多进程，按 token 配额停"""
import os, sys, io, json, argparse, time, multiprocessing as mp
import numpy as np
from tokenizers import Tokenizer
import zstandard as zstd
import pyarrow.parquet as pq

def iter_texts(path):
    if path.endswith(".zst"):
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as r:
                tr = io.TextIOWrapper(r, encoding="utf-8", errors="ignore")
                for line in tr:
                    line = line.strip()
                    if not line: continue
                    try: d = json.loads(line)
                    except Exception: continue
                    t = d.get("text") or d.get("content") or ""
                    if t: yield t
    elif path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        col = "text" if "text" in pf.schema.names else pf.schema.names[0]
        for b in pf.iter_batches(batch_size=1000, columns=[col]):
            for s in b.column(0).to_pylist():
                if s: yield s
    else:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except Exception:
                    yield line; continue
                t = d.get("text") or d.get("content") or ""
                if t: yield t

def work(job):
    wid, files, out, tok_path, target, shard_tokens, prefix, min_len = job
    tk = Tokenizer.from_file(tok_path)
    eos = tk.token_to_id("</s>")
    if eos is None: eos = 0
    buf = np.empty(shard_tokens, dtype=np.uint16)
    ptr = 0; sc = 0; total = 0; docs = 0; skipped = 0
    t0 = time.time()

    def flush():
        nonlocal ptr, sc
        if ptr == 0: return
        p = os.path.join(out, f"{prefix}-w{wid:02d}-{sc:05d}.bin")
        buf[:ptr].tofile(p)
        sc += 1; ptr = 0

    batch = []
    stop = False
    for fp in files:
        if stop: break
        for t in iter_texts(fp):
            if len(t) < min_len:
                skipped += 1; continue
            batch.append(t)
            if len(batch) >= 512:
                for e in tk.encode_batch(batch):
                    ids = e.ids
                    if not ids: continue
                    n = len(ids) + 1
                    if ptr + n > shard_tokens:
                        flush()
                    buf[ptr:ptr+len(ids)] = ids
                    buf[ptr+len(ids)] = eos
                    ptr += n
                    total += n; docs += 1
                batch = []
                if total >= target:
                    stop = True; break
    # 收尾
    if batch and not stop:
        for e in tk.encode_batch(batch):
            ids = e.ids
            if not ids: continue
            n = len(ids) + 1
            if ptr + n > shard_tokens:
                flush()
            if ptr + n > shard_tokens: continue
            buf[ptr:ptr+len(ids)] = ids
            buf[ptr+len(ids)] = eos
            ptr += n; total += n; docs += 1
    flush()
    dt = time.time() - t0
    return {"wid": wid, "tokens": total, "docs": docs, "shards": sc,
            "skip": skipped, "sec": round(dt, 1), "tok_s": round(total/max(dt, 1))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--target", type=float, required=True, help="总 token 配额")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard_tokens", type=int, default=50_000_000)
    ap.add_argument("--prefix", default="zh")
    ap.add_argument("--min_len", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(a.inputs)
    per = a.target / a.workers
    jobs = []
    for w in range(a.workers):
        sub = files[w::a.workers]
        jobs.append((w, sub, a.out, a.tok, per, a.shard_tokens, a.prefix, a.min_len))
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(work, jobs)
    tot = sum(r["tokens"] for r in res)
    dt = time.time() - t0
    print(f"done prefix={a.prefix} tokens={tot/1e9:.3f}B "
          f"docs={sum(r['docs'] for r in res)} shards={sum(r['shards'] for r in res)} "
          f"{tot/max(dt,1):.0f} tok/s {dt:.0f}s", flush=True)
    for r in res:
        print(" ", r, flush=True)

if __name__ == "__main__":
    main()
