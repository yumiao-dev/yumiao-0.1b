# -*- coding: utf-8 -*-
"""Yumiao-0.1B model definition.

A 115.7M-parameter decoder-only Transformer trained from scratch on Chinese-first data.

Architecture:
    - Decoder-only Transformer, 10 layers
    - hidden size 512, 8 query heads / 2 KV heads (GQA), head_dim 64
    - SwiGLU feed-forward (FFN 2048)
    - RMSNorm + RoPE (base 500000)
    - QK-Norm on query/key
    - Tied input/output embeddings
    - Vocabulary 151665, context length 2048
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 512
LAYERS = 10
HEADS = 8
KVHEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 64
FFN = 2048
VOCAB = 151665
CTX = 2048
ROPE_BASE = 500000.0


def rope_cache(seq, dim, device, base=ROPE_BASE):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).bfloat16(), torch.sin(freqs).bfloat16()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D); cos/sin: (T, D/2)"""
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
        q, k = self.qn(q), self.kn(k)                       # QK-Norm
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if KVHEADS != HEADS:                                # GQA expansion
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
        self.up = nn.Linear(HIDDEN, 2 * FFN, bias=False)    # SwiGLU gate+up
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
        self.head.weight = self.emb.weight                  # tied embeddings
        self.apply(self._init)

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


def count_params(model):
    """Unique parameter count (tied embeddings counted once)."""
    return sum(p.numel() for p in set(model.parameters()))
