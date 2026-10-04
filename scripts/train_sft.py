# -*- coding: utf-8 -*-
"""Yumiao-0.1B 身份 SFT v2 —— 真正的指令微调
修正:
  1. 按样本切分，不再揉成长流
  2. prompt 掩码：只对「助手：」之后的内容计 loss
  3. lr 降到 2e-5，epoch 降 2，防过拟合背书
数据: ./identity.jsonl
输出: ./latest.pt
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_yumiao import Yumiao, VOCAB, CTX, rope_cache

TOK_PATH = os.environ.get("TOK_PATH", "./tokenizer/tokenizer.json")
DATA = os.environ.get("DATA", "./data/identity.jsonl")
PRETRAIN_CKPT = os.environ.get("PRETRAIN_CKPT", "./ckpt/pretrain.pt")
CKPT_DIR = os.environ.get("CKPT_DIR", "./ckpt_sft")
LOG_PATH = os.environ.get("LOG_PATH", "./train_sft.log")
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 2
BATCH = 8
LR = 2e-5
MIN_LR = 2e-6
WARMUP = 20
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EOT = 151643
MAX_SECONDS = float(os.environ.get("SFT_MAX_SECONDS", "3600"))
IGNORE = -100

# 「助手：」之后开始计 loss —— 覆盖生成里的各种写法
ANS_MARKERS = ["助手：", "助手:", "Assistant:", "答：", "答:", "A:", "A：", "回答："]
# 提示词计数 = 0（纯陈述句）时，最多保留多少条，防止"续写"样本稀释问答能力
MAX_PLAIN = 300


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


class SFTData:
    """按样本构建 (input_ids, labels)，labels 上 prompt 段 = IGNORE"""

    def __init__(self, tok):
        self.tok = tok
        self.samples = []          # list[(ids, ans_start)]
        texts = []
        with open(DATA, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = (o.get('text') or '').strip()
                if t:
                    texts.append(t)

        plain = []
        for t in texts:
            # 找答案起点（字符位置）
            cut = -1
            for mk in ANS_MARKERS:
                p = t.find(mk)
                if p >= 0:
                    cut = p + len(mk)
                    break
            if cut < 0:
                # 纯陈述句：整条都算答案，但限量
                plain.append(t)
                continue

            prompt, answer = t[:cut], t[cut:]
            if not answer.strip():
                continue

            p_ids = tok.encode(prompt).ids
            a_ids = tok.encode(answer).ids
            ids = p_ids + a_ids + [EOT]
            if len(ids) > CTX:
                ids = ids[:CTX]
            if len(ids) < 3:
                continue
            self.samples.append((ids, min(len(p_ids), CTX)))

        # 打散后补入限量纯陈述
        rng = np.random.RandomState(7)
        rng.shuffle(plain)
        for t in plain[:MAX_PLAIN]:
            ids = tok.encode(t).ids + [EOT]
            if len(ids) > CTX:
                ids = ids[:CTX]
            if len(ids) < 3:
                continue
            self.samples.append((ids, 0))

        rng.shuffle(self.samples)

        self.n = len(self.samples)
        tot = sum(len(s[0]) for s in self.samples)
        ans = sum(len(s[0]) - s[1] for s in self.samples)
        log(f"SFT2 数据: {self.n} 条 | {tot/1e6:.3f}M tokens | 监督 {ans/1e6:.3f}M "
            f"({100*ans/max(1,tot):.0f}%)")

    def batch(self, bs):
        """随机取 bs 个样本，右侧 pad 到等长，返回 x, y, mask"""
        idx = np.random.randint(0, self.n, size=bs)
        seqs, starts = [], []
        for i in idx:
            ids, st = self.samples[i]
            seqs.append(ids)
            starts.append(st)
        maxlen = max(len(s) for s in seqs)
        maxlen = min(maxlen, CTX)

        x = np.full((bs, maxlen), EOT, dtype=np.int64)
        y = np.full((bs, maxlen), IGNORE, dtype=np.int64)
        for b, (ids, st) in enumerate(zip(seqs, starts)):
            L = min(len(ids), maxlen)
            x[b, :L] = ids[:L]
            # labels: 预测下一位；prompt 段（含 pad）不计 loss
            lim = L - 1
            if lim <= 0:
                continue
            s = st - 1                      # 答案第一个 token 的前一位开始监督
            if s < 0:
                s = 0
            y[b, s:lim] = ids[s + 1:L]
        xt = torch.from_numpy(x).pin_memory().to('cuda', non_blocking=True)
        yt = torch.from_numpy(y).pin_memory().to('cuda', non_blocking=True)
        return xt, yt


def masked_loss(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        ignore_index=IGNORE)


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)
    tok = Tokenizer.from_file(TOK_PATH)
    data = SFTData(tok)

    dev = 'cuda'
    model = Yumiao().to(dev)
    sd = torch.load(PRETRAIN_CKPT, map_location=dev, weights_only=False)['model']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    mi = model.load_state_dict(sd, strict=False)
    log(f"预训练权重加载: missing={len(mi.missing_keys)} unexpected={len(mi.unexpected_keys)}")
    model = torch.compile(model)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if getattr(p, 'DO_NOT_WD', False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([
        {'params': decay, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=LR, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = max(1, data.n // BATCH)
    total_steps = steps_per_epoch * EPOCHS
    log(f"SFT2: {EPOCHS} epochs, {steps_per_epoch} steps/epoch, total {total_steps} steps")

    step = 0
    ck = os.path.join(CKPT_DIR, 'latest.pt')
    if os.path.exists(ck):
        try:
            d = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(d['model']); opt.load_state_dict(d['opt'])
            step = d['step']
            log(f"resumed SFT2 from step {step}")
        except Exception as e:
            log(f"SFT2 resume failed: {e}")

    model.train()
    t0 = time.time()
    log(f"=== SFT2 starts: batch={BATCH} ctx={CTX} lr={LR} ===")
    acc = []
    while step < total_steps:
        if step < WARMUP:
            lr = LR * (step + 1) / WARMUP
        else:
            p_ = (step - WARMUP) / max(1, total_steps - WARMUP)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(min(p_, 1.0) * math.pi))
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        x, y = data.batch(BATCH)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = masked_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1
        acc.append(loss.item())
        if step % 5 == 0:
            log(f"step {step}/{total_steps} | loss {np.mean(acc[-20:]):.4f} "
                f"| lr {lr:.2e} | {time.time()-t0:.0f}s")
        if step % 100 == 0 or step == total_steps:
            tmp = os.path.join(CKPT_DIR, 'latest.tmp')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'step': step}, tmp)
            os.rename(tmp, ck)
            log(f"SFT2 ckpt saved @ step {step}")
        if time.time() - t0 > MAX_SECONDS:
            log(f"SFT2 达到时间上限，停止")
            break

    tmp = os.path.join(CKPT_DIR, 'latest.tmp')
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, tmp)
    os.rename(tmp, ck)
    log(f"=== SFT2 结束: {step} steps, ckpt -> {ck}")


if __name__ == '__main__':
    main()
