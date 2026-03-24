"""Pre-tokenized dataset cache.

Tokenizes a streaming HuggingFace dataset once and saves it as a flat numpy
array on disk.  Subsequent runs load from the cache file instantly (memory-
mapped), with no HTTP traffic and no on-the-fly BPE overhead.

Cache file naming:
    {cache_dir}/{dataset}_{split}_{tok_type}_{vocab_size}.npy
Example:
    data/cache/tinystories_train_bpe_3356.npy

The dtype is uint16 for vocab <= 65535, uint32 otherwise.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TokenizedCacheDataset(Dataset):
    """Fixed-length (x, y) chunks from a flat token array."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return (
            torch.tensor(chunk[:-1].astype(np.int64)),
            torch.tensor(chunk[1:].astype(np.int64)),
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path_for(config: dict, dataset_split_key: str, vocab_size: int) -> str:
    """Return the .npy path for this (dataset+split, tokenizer, vocab_size) triple."""
    tok_type  = config["data"]["tokenizer"]
    cache_dir = config["data"].get("cache_dir", "data/cache")
    fname = f"{dataset_split_key}_{tok_type}_{vocab_size}.npy"
    return os.path.join(cache_dir, fname)


def _build_cache(
    hf_dataset,
    tokenizer,
    cache_path: str,
    n_tokens_limit: int,
    text_key: str = "text",
) -> np.ndarray:
    """Stream-tokenize hf_dataset and write a flat numpy array to cache_path."""
    from tqdm import tqdm

    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32

    all_tokens: list[int] = []
    total = 0
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    with tqdm(total=n_tokens_limit, unit="tok", unit_scale=True,
              desc="    tokenizing for cache") as pbar:
        for item in hf_dataset:
            text = item.get(text_key, "")
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            all_tokens.extend(ids)
            added = len(ids)
            total += added
            pbar.update(added)
            if total >= n_tokens_limit:
                break

    arr = np.array(all_tokens[:n_tokens_limit], dtype=dtype)
    np.save(cache_path, arr)
    print(f"    cache saved ({len(arr):,} tokens, {arr.nbytes / 1e6:.1f} MB) -> {cache_path}")
    return arr


def load_or_build(
    hf_dataset,
    tokenizer,
    config: dict,
    dataset_split_key: str,
    n_tokens_limit: int,
    text_key: str = "text",
    force_rebuild: bool = False,
) -> TokenizedCacheDataset:
    """
    Return a TokenizedCacheDataset, building the cache file if needed.

    Args:
        hf_dataset:        streaming HuggingFace dataset (only consumed if cache missing)
        tokenizer:         fitted BaseTokenizer with .encode() and .vocab_size
        config:            full config dict
        dataset_split_key: e.g. "tinystories_train" — used in the cache filename
        n_tokens_limit:    max tokens to tokenize
        text_key:          field name in each HF example
        force_rebuild:     ignore existing cache and re-tokenize
    """
    seq_len    = config["data"]["seq_len"]
    vocab_size = tokenizer.vocab_size
    path       = cache_path_for(config, dataset_split_key, vocab_size)

    if not force_rebuild and os.path.exists(path):
        print(f"    loading token cache -> {path}")
        arr = np.load(path)
    else:
        if force_rebuild and os.path.exists(path):
            print(f"    rebuilding token cache (--no-cache flag)...")
        else:
            print(f"    token cache not found, building...")
        arr = _build_cache(hf_dataset, tokenizer, path, n_tokens_limit, text_key)

    return TokenizedCacheDataset(arr, seq_len)
