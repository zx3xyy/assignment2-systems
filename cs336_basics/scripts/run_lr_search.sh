#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="/root/336_assignmen1"
TRAIN_PY="$PROJECT_ROOT/cs336_basics/train_script.py"
CKPT_PATH="$PROJECT_ROOT/ckpt"
TRAIN_DATA="$PROJECT_ROOT/data/TinyStoriesV2-GPT4-train.npy"
VALID_DATA="$PROJECT_ROOT/data/TinyStoriesV2-GPT4-valid.npy"

export WANDB_PROJECT="lr_bs_search"
GROUP="lr_sweep_$(date +%m%d_%H%M)"

for LR in 3e-4 1e-3 3e-3 1e-2; do
  WANDB_RUN_GROUP="$GROUP" uv run python "$TRAIN_PY" \
    --ckpt-path "$CKPT_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --valid-data-path "$VALID_DATA" \
    --ckpt_interval 0 \
    --max_lr="$LR" \
    --exp_name "lr_${LR}" \
    --torch_compile 2>&1 | tee "$PROJECT_ROOT/logs/${GROUP}__lr_${LR}_bs256.log"
done
