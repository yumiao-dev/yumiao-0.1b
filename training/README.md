# 训练代码

全部为自研实现，**预训练与 SFT 均不依赖 `transformers` 训练接口**（只用到 `torch`、`tokenizers`、`numpy`）。导出的权重可被 `transformers` 的 `LlamaForCausalLM` 直接加载（RoPE 布局已在导出时转换）。

## 文件

| 文件 | 作用 |
|---|---|
| `model_lm.py` | 模型定义：RMSNorm / RoPE / GQA Attention / SwiGLU / Block / Llama |
| `train_lm.py` | 预训练循环，`np.memmap` 读取 uint16 分片，AdamW + warmup + cosine |
| `train_lm.sh` | 启动脚本，自动检测 `last.pt` 续训 |
| `tokenizer_train.py` | 流式训练 32k BPE |
| `stream_tok.py` | 多进程分词，语料 → uint16 bin 分片，支持按 token 配额停止 |
| `sft_gen.py` | 事实表驱动的身份问答数据生成（含句式扩写与通用指令混合） |
| `build_sft2.py` | 合并身份数据与 `alpaca-gpt4-data-zh`，产出 SFT 混合集 |
| `sft_train.py` | SFT，只在 assistant 段算 loss |
| `export_hf.py` | 导出 HuggingFace 格式（含 RoPE 重排与数值验证） |

## 顺序

```bash
# 1) 训练分词器（约 20 分钟 / 8 核）
python tokenizer_train.py --out tokenizer.json

# 2) 语料 → uint16 分片（按配额停）
python stream_tok.py --src data/raw --out data/bin --tok tokenizer.json --target 8.0e9

# 3) 预训练
bash train_lm.sh          # 默认 122000 步，bs 64，seq 1024，lr 6e-4

# 4) 生成 + 合并 SFT 数据
python sft_gen.py --out data/sft/identity.jsonl --aug 0.22
python build_sft2.py

# 5) SFT
python sft_train.py --ckpt ckpt/last.pt --data data/sft/mix2.jsonl \
    --out ckpt/sft2 --lr 5e-5 --epochs 1 --seq 512 --bs 8

# 6) 导出 HF 格式
python export_hf.py ckpt/sft2/sft_last.pt hf_build/chat
```

## 硬性注意事项

1. **不要双重 shift**。DataLoader 负责 `ids[:-1] / labels[1:]`，模型内部就不能再错位一次。诊断方法：同一段文本分别按 `t+1`、`t+2` 对齐算 loss，正常应满足 `loss(t+1) < loss(t+2)` 且相差明显。
2. **多语料必须 shuffle**。分片文件名顺序会决定训练顺序，`en-*` 在 `zh-*` 前就会先训几万步纯英文。启动后立刻打印第一个 batch 的语言分布。
3. **SFT 的 labels 必须右移一位**，否则 loss 会掉到 0.001 量级，模型学成复读机。
4. **ROCm 环境避开 `cross_entropy(ignore_index=-100)`**。会在反向触发 `hipErrorLaunchFailure (719)`，改手动 mask（见 `sft_train.py`）。
5. **SFT 学习率别照搬大模型**。100M 模型用 5e-5，1e-4 会灾难性遗忘。

## 显存参考

| 阶段 | 配置 | 显存 |
|---|---|---|
| 预训练 | bs 64 × seq 1024，bf16 | ≈ 42 GB（含优化器状态） |
| SFT | bs 8 × seq 512，bf16 | ≈ 6 GB |
| 推理 | bs 1，bf16 | ≈ 0.6 GB |
