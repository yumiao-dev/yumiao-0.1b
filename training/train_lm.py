#!/usr/bin/env python3
"""0.1B 语言模型预训练：memmap 二进制数据 + bf16 + 断点续训"""
import os, glob, math, time, argparse
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from model_lm import Llama


class BinDS(Dataset):
    """uint16 二进制 token 流，按 seq 切块"""
    def __init__(self, root, seq=1024):
        self.seq = seq
        self.files = sorted(glob.glob(os.path.join(root, "*.bin")))
        if not self.files:
            raise RuntimeError(f"no .bin in {root}")
        self.mm = []
        self.lens = []
        for f in self.files:
            a = np.memmap(f, dtype=np.uint16, mode="r")
            self.mm.append(a); self.lens.append(len(a))
        self.cum = np.cumsum([0] + self.lens)
        self.total = int(self.cum[-1])
        self.n = max(1, self.total // (seq + 1))
    def __len__(self): return self.n
    def __getitem__(self, i):
        pos = i * self.seq
        fi = int(np.searchsorted(self.cum, pos, side="right")) - 1
        fi = min(fi, len(self.files) - 1)
        off = int(pos - self.cum[fi])
        if off + self.seq + 1 > self.lens[fi]:
            off = max(0, self.lens[fi] - self.seq - 1)
        a = self.mm[fi][off:off + self.seq + 1].astype(np.int64)
        x = torch.from_numpy(a[:-1]); y = torch.from_numpy(a[1:])
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/workspace/lm/data/bin")
    ap.add_argument("--out", default="/mnt/workspace/lm/ckpt")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=305000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = Llama(vocab=32000, max_len=a.seq).cuda()
    print(f"params {model.n_params()/1e6:.1f}M  seq {a.seq}  bs {a.bs}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd,
                            betas=(0.9, 0.95), fused=False)
    start = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]; print("resume from", start, flush=True)

    ds = BinDS(a.data, a.seq)
    print(f"dataset blocks {len(ds)}  tokens {ds.total/1e9:.2f}B", flush=True)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4,
                    shuffle=True)

    def lr_at(s):
        if s < a.warmup: return a.lr * s / max(1, a.warmup)
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    model.train()
    step = start; run = 0.0; nb = 0; t0 = time.time(); tok = 0
    while step < a.steps:
        for x, y in dl:
            if step >= a.steps: break
            lr = lr_at(step)
            for g in opt.param_groups: g["lr"] = lr
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); nb += 1; tok += x.numel(); step += 1
            if step % a.log_every == 0:
                dt = time.time() - t0
                print(f"step {step} loss {run/nb:.4f} ppl {math.exp(min(run/nb,20)):.1f} "
                      f"lr {lr:.2e} {tok/dt:.0f} tok/s {dt/nb*1000:.0f} ms/step", flush=True)
                run = 0.0; nb = 0; t0 = time.time(); tok = 0
            if step % a.save_every == 0:
                st = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
                last = os.path.join(a.out, "last.pt"); tmp = last + ".tmp"
                torch.save(st, tmp); os.replace(tmp, last)
                if step % (a.save_every * 10) == 0:
                    p = os.path.join(a.out, f"ckpt_{step:07d}.pt")
                    torch.save(st, p)
                    olds = sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")))
                    for o in olds[:-3]: os.remove(o)
                    print("archived", p, flush=True)
                print("saved", last, flush=True)


if __name__ == "__main__":
    main()
