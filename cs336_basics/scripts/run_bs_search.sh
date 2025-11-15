#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="/root/336_assignmen1"
TRAIN_PY="$PROJECT_ROOT/cs336_basics/train_script.py"
CKPT_PATH="$PROJECT_ROOT/ckpt"
TRAIN_DATA="$PROJECT_ROOT/data/TinyStoriesV2-GPT4-train.npy"
VALID_DATA="$PROJECT_ROOT/data/TinyStoriesV2-GPT4-valid.npy"

export WANDB_PROJECT="lr_bs_search"
GROUP="bs_sweep_$(date +%m%d_%H%M)"

for BS in 64 128 256; do
  WANDB_RUN_GROUP="$GROUP" uv run python "$TRAIN_PY" \
    --ckpt-path "$CKPT_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --valid-data-path "$VALID_DATA" \
    --batch_size="$BS" \
    --exp_name "bs_${BS}_cfgLR" \
    --ckpt_interval 0 \
    --torch_compile 2>&1 | tee "$PROJECT_ROOT/logs/${GROUP}__bs_${BS}_cfgLR.log"
done
