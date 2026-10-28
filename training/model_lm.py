#!/usr/bin/env python3
"""0.1B 标准 Llama 式语言模型：RMSNorm + RoPE + GQA + SwiGLU"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def precompute_rope(head_dim, max_len=4096, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: B,H,T,D
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    s = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    def __init__(self, d, nh, nkv):
        super().__init__()
        self.nh, self.nkv = nh, nkv
        self.dh = d // nh
        self.wq = nn.Linear(d, nh * self.dh, bias=False)
        self.wk = nn.Linear(d, nkv * self.dh, bias=False)
        self.wv = nn.Linear(d, nkv * self.dh, bias=False)
        self.wo = nn.Linear(nh * self.dh, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.dh).transpose(1, 2)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d, nh, nkv, hidden):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attention(d, nh, nkv)
        self.n2 = RMSNorm(d); self.mlp = SwiGLU(d, hidden)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Llama(nn.Module):
    def __init__(self, vocab=32000, d=768, nlayer=12, nh=12, nkv=4, hidden=2048, max_len=2048):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nh, nkv, hidden) for _ in range(nlayer)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight          # tie
        self.max_len = max_len
        cos, sin = precompute_rope(d // nh, max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # 残差输出缩放，避免深层累积
        for n, p in self.named_parameters():
            if n.endswith("w3.weight") or n.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * nlayer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        if targets is None: return logits
        tg = targets.view(-1)
        if (tg < 0).any():
            # ROCm 上 ignore_index 的 backward 会 kernel failure，改手动 mask
            mask = (tg >= 0).float()
            safe = tg.clamp(min=0)
            ls = F.cross_entropy(logits.view(-1, logits.size(-1)), safe, reduction="none")
            loss = (ls * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tg)
        return logits, loss
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = Llama()
    x = torch.randint(0, 32000, (2, 128))
    logits, loss = m(x, x)
    print("params", f"{m.n_params()/1e6:.1f}M", "out", tuple(logits.shape), "loss", round(loss.item(), 3))
