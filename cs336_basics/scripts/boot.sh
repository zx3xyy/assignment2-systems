#!/usr/bin/env bash
set -euo pipefail

# ---- uv setup ----
curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

cd /root/336_assignmen1
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# ---- data ----
mkdir -p data
cd data

wget -q --show-progress -O TinyStoriesV2-GPT4-train.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt?download=1"

wget -q --show-progress -O TinyStoriesV2-GPT4-valid.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt?download=1"

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..

# ---- tokenize ----
uv run ./cs336_basics/scripts/tokenize_data.py \
  /root/336_assignmen1/data/TinyStoriesV2-GPT4-train.txt

uv run ./cs336_basics/scripts/tokenize_data.py \
  /root/336_assignmen1/data/TinyStoriesV2-GPT4-valid.txt

uv run ./cs336_basics/scripts/tokenize_data.py \
  /root/336_assignmen1/data/owt_train.txt

uv run ./cs336_basics/scripts/tokenize_data.py \
  /root/336_assignmen1/data/owt_valid.txt