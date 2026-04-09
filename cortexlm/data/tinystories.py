"""TinyStories dataset (Eldan & Li 2023)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Iterator, List, Optional


class TokenSequenceDataset(Dataset):
    """Flat token array sliced into fixed-length chunks."""

    def __init__(self, tokens: List[int], seq_len: int):
        self.seq_len = seq_len
        # Number of complete chunks
        n_chunks = len(tokens) // (seq_len + 1)
        tokens = tokens[:n_chunks * (seq_len + 1)]
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.tokens) // (self.seq_len + 1)

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start: start + self.seq_len + 1]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]    # target tokens (shifted by 1)
        return x, y


class StreamingTokenDataset(IterableDataset):
    """
    Streaming token dataset.

    Tokenizes on-the-fly and yields (x, y) pairs of length seq_len.
    Limits total tokens consumed to n_tokens_limit.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        seq_len: int,
        n_tokens_limit: int,
        text_key: str = "text",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.n_tokens_limit = n_tokens_limit
        self.text_key = text_key

    def __iter__(self) -> Iterator:
        buffer: List[int] = []
        total = 0

        for item in self.hf_dataset:
            if total >= self.n_tokens_limit:
                break
            text = item.get(self.text_key, "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text)
            buffer.extend(ids)
            total += len(ids)

            # Yield complete chunks from buffer
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y


def load_tinystories(config: dict, tokenizer, split: str = "train",
                     force_rebuild: bool = False):
    """
    Load TinyStories.  Tokenizes once and caches to disk; subsequent calls
    load from the cache file instantly.

    Args:
        config:        full config dict
        tokenizer:     fitted BaseTokenizer
        split:         'train' or 'validation'
        force_rebuild: ignore existing cache and re-tokenize
    Returns:
        TokenizedCacheDataset  (regular Dataset, supports shuffle + workers)
    """
    from datasets import load_dataset
    from .cache import load_or_build

    default_limit = 521_044_049 if split == "train" else 5_232_110
    n_tokens_limit = config["data"].get("tinystories_token_limit", default_limit)

    print(f"    TinyStories ({split}) — limit {n_tokens_limit:,} tokens")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    return load_or_build(
        ds, tokenizer, config,
        dataset_split_key=f"tinystories_{split}",
        n_tokens_limit=n_tokens_limit,
        text_key="text",
        force_rebuild=force_rebuild,
    )
