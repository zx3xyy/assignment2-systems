#!/usr/bin/env python3
"""
Convert a plain text file into a NumPy array of GPT-2 token IDs with a progress bar.

Usage:
    python txt_to_np.py <input_txt> [<output_npy>]

Example:
    python txt_to_np.py data/TinyStoriesV2-GPT4-train.txt
"""

import sys
import numpy as np
import tiktoken
from tqdm import tqdm
from pathlib import Path


def txt_to_np_array(txt_file: str, np_file: str | None = None, chunk_size: int = 1_000_000) -> None:
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    total_chars = len(text)
    print(f"📘 Tokenizing {txt_file} ({total_chars:,} characters)...")

    tokens = []
    # tokenize in chunks with progress bar
    for i in tqdm(range(0, total_chars, chunk_size), desc="Tokenizing", ncols=100):
        chunk = text[i:i + chunk_size]
        tokens.extend(tokenizer.encode(chunk, allowed_special={"<|endoftext|>"}))

    arr = np.array(tokens, dtype=np.int32)

    if np_file is None:
        np_file = str(Path(txt_file).with_suffix(".npy"))

    np.save(np_file, arr)
    print(f"✅ Saved {len(arr):,} tokens to {np_file}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python txt_to_np.py <input_txt> [<output_npy>]")
        sys.exit(1)

    txt_file = sys.argv[1]
    np_file = sys.argv[2] if len(sys.argv) > 2 else None
    txt_to_np_array(txt_file, np_file)


if __name__ == "__main__":
    main()
