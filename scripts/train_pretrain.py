# -*- coding: utf-8 -*-
"""Yumiao-0.1B 预训练脚本 —— AMD MI300 (ROCm) 单卡
架构: Decoder-only, hidden 512, 10 层, GQA(8Q/2KV), SwiGLU, RoPE, RMSNorm, QK-Norm, tied embedding
参数: 115.7M  |  目标: 1B tokens (~1907 步, ~2.5h @ 111K tok/s)
"""
import os, sys, math, time, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 超参 ----------------
HIDDEN = 512
LAYERS = 10
HEADS = 8
KVHEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 64
FFN = 2048
VOCAB = 151665
CTX = 2048                          # 训练序列长度
DROPOUT = 0.0

LR_PEAK = 8e-4
LR_MIN = 8e-5
WARMUP = 50
TOTAL_STEPS = 1907                  # 1B tokens / 524288
WEIGHT_DECAY = 0.1
BETA = (0.9, 0.95)
GRAD_CLIP = 1.0
MICRO_BS = 32
GRAD_ACCUM = 8                       # global batch = 32*2048*8 = 524288 tok/step
MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", "14400"))    # 4h 安全线
CKPT_DIR = os.environ.get("CKPT_DIR", "./ckpt")
LOG_PATH = os.environ.get("LOG_PATH", "./train.log")
DATA_GLOB = os.environ.get("DATA_GLOB", "./data/part_*.bin")
VAL_BIN = os.environ.get("VAL_BIN", "./data/val.bin")

os.makedirs(CKPT_DIR, exist_ok=True)

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------------- 模型 ----------------
def rope_cache(seq, dim, device, base=500000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).bfloat16(), torch.sin(freqs).bfloat16()

def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(HIDDEN, KVHEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(HIDDEN, KVHEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.qn = nn.RMSNorm(HEAD_DIM)
        self.kn = nn.RMSNorm(HEAD_DIM)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, T, KVHEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, T, KVHEADS, HEAD_DIM).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)           # QK-Norm
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if KVHEADS != HEADS:                     # GQA 展开
            rep = HEADS // KVHEADS
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.RMSNorm(HIDDEN)
        self.attn = Attention()
        self.n2 = nn.RMSNorm(HIDDEN)
        self.up = nn.Linear(HIDDEN, 2 * FFN, bias=False)   # SwiGLU gate+up
        self.down = nn.Linear(FFN, HIDDEN, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        g, u = self.up(self.n2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(g) * u)
        return x

class Yumiao(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.nf = nn.RMSNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.head.weight = self.emb.weight                # tied
        self.apply(self._init)
        for name, p in self.named_parameters():           # embed/lm_head 不衰减
            if 'emb.weight' in name or 'head.weight' in name:
                p.DO_NOT_WD = True

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = rope_cache(T, HEAD_DIM, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

# ---------------- 数据 ----------------
class BinStream:
    """多 part 流式读取 uint32 token，随机起始，循环消费"""
    def __init__(self):
        _min = MICRO_BS * CTX * 4 * 8
        self._min = _min
        self.parts = sorted(p for p in glob.glob(DATA_GLOB) if os.path.getsize(p) > _min)
        self.fhs = [open(p, 'rb', buffering=0) for p in self.parts]
        self._rescans = 0
        log(f"data parts: {[os.path.basename(p) for p in self.parts]}")
        if not self.parts:
            raise RuntimeError("NO DATA PARTS FOUND in " + DATA_GLOB)

    def rescan(self):
        """重新扫描目录，把 tokenize 新产出的 part 纳入（边训边等数据）"""
        cur = set(self.parts)
        new = [p for p in sorted(glob.glob(DATA_GLOB))
               if p not in cur and os.path.getsize(p) > self._min]
        if new:
            for p in new:
                self.parts.append(p)
                self.fhs.append(open(p, 'rb', buffering=0))
            self.parts.sort()
            self.fhs = [open(p, 'rb', buffering=0) for p in self.parts]
            for f in self.fhs:
                pass
            log(f"data rescan: +{len(new)} new part(s) -> {[os.path.basename(p) for p in self.parts]}")
            return True
        return False

    def read_batch(self, bs, seq):
        need = bs * seq + 1
        out = np.empty(need, dtype=np.uint32)
        got = 0
        guard = 0
        while got < need:
            guard += 1
            if guard > 1000:
                raise RuntimeError("read_batch stuck")
            i = np.random.randint(0, len(self.fhs))
            fh = self.fhs[i]
            # 避开正在写入的尾部（tkz.py 追加写）——留 64MB 安全边距
            size = max(0, os.path.getsize(self.parts[i]) - 64 * 1024 * 1024) // 4
            span = size - need
            if span <= 0:
                continue
            start = np.random.randint(0, span)
            fh.seek(start * 4)
            nread = min(need - got, size - start)
            if nread <= 0:
                continue
            chunk = np.frombuffer(fh.read(nread * 4), dtype=np.uint32)
            if len(chunk) == 0:
                continue
            out[got:got + len(chunk)] = chunk
            got += len(chunk)
        x = torch.from_numpy(out[:-1].reshape(bs, seq).astype(np.int64))
        y = torch.from_numpy(out[1:].reshape(bs, seq).astype(np.int64))
        return x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)

# ---------------- 训练 ----------------
def get_lr(step):
    if step < WARMUP:
        return LR_PEAK * step / WARMUP
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(min(p, 1.0) * math.pi))

def save_ckpt(model, opt, step, tokens_done):
    tmp = os.path.join(CKPT_DIR, "latest.tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "tokens": tokens_done}, tmp)
    os.rename(tmp, os.path.join(CKPT_DIR, "latest.pt"))
    log(f"ckpt saved @ step {step}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    dev = 'cuda'
    model = Yumiao().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"model params: {nparam/1e6:.1f}M ({nparam*2/1024**3:.2f} GB bf16)")
    model = torch.compile(model)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if getattr(p, 'DO_NOT_WD', False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=LR_PEAK, betas=BETA, eps=1e-8, fused=True)

    step, tokens_done = 0, 0
    ck = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(ck):
        try:
            d = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"])
            step, tokens_done = d["step"], d["tokens"]
            log(f"resumed from step {step}")
        except Exception as e:
            log(f"resume failed: {e}")

    data = BinStream()
    val_x, val_y = None, None
    vb = os.path.join(os.path.dirname(VAL_BIN), "val.bin")
    if os.path.exists(vb) and os.path.getsize(vb) > 100_000:
        try:
            v = np.fromfile(vb, dtype=np.uint32, count=CTX * 4 + 1)
            vx = torch.from_numpy(v[:-1].reshape(4, CTX).astype(np.int64)).to(dev)
            vy = torch.from_numpy(v[1:].reshape(4, CTX).astype(np.int64)).to(dev)
            val_x, val_y = vx, vy
            log("val set loaded")
        except Exception as e:
            log(f"val load failed: {e}")

    model.train()
    t0 = time.time()
    log(f"=== training starts: ctx={CTX} micro_bs={MICRO_BS} accum={GRAD_ACCUM} "
        f"total_steps={TOTAL_STEPS} ===")
    while step < TOTAL_STEPS:
        lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = data.read_batch(MICRO_BS, CTX)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1
        tokens_done += MICRO_BS * GRAD_ACCUM * CTX

        if step % 20 == 0:
            el = time.time() - t0
            tps = tokens_done / max(1e-9, el)
            log(f"step {step}/{TOTAL_STEPS} | loss {loss_sum/GRAD_ACCUM:.3f} | lr {lr:.2e} | "
                f"tok {tokens_done/1e9:.3f}G | {el:.0f}s | {tps/1000:.1f}K tok/s")
        if step % 100 == 0:
            try:
                data.rescan()
            except Exception as e:
                log(f"rescan failed: {e}")
        if step % 250 == 0:
            save_ckpt(model, opt, step, tokens_done)
            if val_x is not None:
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    vl = model(val_x, val_y)[1].item()
                log(f"val loss @ step {step}: {vl:.3f}")
                model.train()
        if time.time() - t0 > MAX_SECONDS:
            save_ckpt(model, opt, step, tokens_done)
            log("time budget reached, saved & exit")
            break
    save_ckpt(model, opt, step, tokens_done)
    log(f"=== training loop ended: {tokens_done/1e9:.3f}G tokens, {step} steps ===")

if __name__ == "__main__":
    main()
