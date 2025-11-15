#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="/root/336_assignmen1"
TRAIN_PY="$PROJECT_ROOT/cs336_basics/train_script.py"
CKPT_PATH="$PROJECT_ROOT/ckpt"
TRAIN_DATA="$PROJECT_ROOT/data/owt_train.npy"
VALID_DATA="$PROJECT_ROOT/data/owt_valid.npy"

export WANDB_PROJECT="owt"
GROUP="owt_$(date +%m%d_%H%M)"

for LR in 2e-3 1e-3; do
  WANDB_RUN_GROUP="$GROUP" uv run python "$TRAIN_PY" \
    --ckpt-path "$CKPT_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --valid-data-path "$VALID_DATA" \
    --ckpt_interval 5000 \
    --max_lr="$LR" \
    --exp_name "owt_${LR}" \
    --target_token=2704059661 \
    --grad_clip_cap=1.0
    --torch_compile 2>&1 | tee "$PROJECT_ROOT/logs/OWT_${LR}.log"
done
