#!/usr/bin/env python3
"""把自研 Llama 式权重导出为 HuggingFace 标准目录（LlamaForCausalLM 兼容）"""
import os, sys, json, inspect, torch
sys.path.insert(0, "/mnt/workspace/lm/scripts")
from model_lm import Llama, apply_rope
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file

print("ROPE_SRC:", " ".join(inspect.getsource(apply_rope).split())[:260])

ckpt_path = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
print("TOPKEYS:", [k for k in (ck.keys() if isinstance(ck, dict) else [])][:8])
print("STEP:", ck.get("step") if isinstance(ck, dict) else None,
      "TOKENS:", ck.get("tokens") if isinstance(ck, dict) else None)
print("NPARAM_KEYS:", len(sd), "DTYPE:", sd["emb.weight"].dtype)

V, D = sd["emb.weight"].shape
NL = len([k for k in sd if k.endswith(".n1.w")])
NH, NKV = 12, 4
HID = sd["blocks.0.mlp.w1.weight"].shape[0]
print("SHAPE:", V, D, NL, NH, NKV, HID)

def permute(w, n_head):
    """interleaved rope -> half rope (HF 布局)"""
    hd = w.shape[0] // n_head
    return w.view(n_head, hd // 2, 2, w.shape[1]).transpose(1, 2).reshape(w.shape[0], w.shape[1])

hf = {"model.embed_tokens.weight": sd["emb.weight"].float()}
for i in range(NL):
    p = "blocks.%d." % i
    q = "model.layers.%d." % i
    hf[q + "input_layernorm.weight"] = sd[p + "n1.w"].float()
    hf[q + "self_attn.q_proj.weight"] = permute(sd[p + "attn.wq.weight"].float(), NH)
    hf[q + "self_attn.k_proj.weight"] = permute(sd[p + "attn.wk.weight"].float(), NKV)
    hf[q + "self_attn.v_proj.weight"] = sd[p + "attn.wv.weight"].float()
    hf[q + "self_attn.o_proj.weight"] = sd[p + "attn.wo.weight"].float()
    hf[q + "post_attention_layernorm.weight"] = sd[p + "n2.w"].float()
    hf[q + "mlp.gate_proj.weight"] = sd[p + "mlp.w1.weight"].float()
    hf[q + "mlp.up_proj.weight"] = sd[p + "mlp.w2.weight"].float()
    hf[q + "mlp.down_proj.weight"] = sd[p + "mlp.w3.weight"].float()
hf["model.norm.weight"] = sd["norm.w"].float()

cfg = LlamaConfig(vocab_size=V, hidden_size=D, intermediate_size=HID,
                  num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=NKV,
                  max_position_embeddings=1024, rms_norm_eps=1e-6, rope_theta=10000.0,
                  tie_word_embeddings=True, torch_dtype="bfloat16")
m = LlamaForCausalLM(cfg)
miss, unexp = m.load_state_dict(hf, strict=False)
print("MISSING:", miss, "UNEXPECTED:", unexp)
m.eval()

own = Llama(vocab=V, d=D, nlayer=NL, nh=NH, nkv=NKV, hidden=HID, max_len=2048)
own.load_state_dict(sd)
own.eval()

ids = torch.tensor([[1, 500, 1200, 88, 77, 9, 300, 4521, 33, 7, 909, 1024]])
with torch.no_grad():
    a = m(ids).logits
    b = own(ids)
d1 = (a - b).abs()
print("VERIFY maxdiff=%.6f meandiff=%.6f logit_absmean=%.4f" % (d1.max().item(), d1.mean().item(), b.abs().mean().item()))
print("VERIFY argmax_equal=", bool((a.argmax(-1) == b.argmax(-1)).all().item()))

# 保存 bf16
save = {k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()}
save_file(save, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
cfg.save_pretrained(out_dir)
print("SAVED", out_dir)
print("SIZE_MB", round(sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6, 1))
