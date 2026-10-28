#!/bin/bash
cd /mnt/workspace/lm/scripts || exit 1
CKPT=/mnt/workspace/lm/ckpt
mkdir -p "$CKPT" /mnt/workspace/lm/logs
R=""
if [ -f "$CKPT/last.pt" ]; then
  R="--resume $CKPT/last.pt"
  echo "[$(date '+%F %T')] resume from last.pt"
else
  echo "[$(date '+%F %T')] fresh start"
fi
exec python -u train_lm.py \
  --data /mnt/workspace/lm/data/bin --out "$CKPT" \
  --steps 122000 --bs 64 --workers 8 --lr 6e-4 \
  --warmup 2000 --save_every 500 --log_every 50 $R
