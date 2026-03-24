"""Wikitext-2 and Wikitext-103 datasets."""

from __future__ import annotations
from cortexlm.data.tinystories import StreamingTokenDataset


_HF_NAMES = {
    "wikitext2":   ("wikitext", "wikitext-2-raw-v1"),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
}


def load_wikitext(config: dict, tokenizer, split: str = "train",
                  force_rebuild: bool = False):
    from datasets import load_dataset
    from .cache import load_or_build

    dataset_name = config["data"]["dataset"]
    hf_name, hf_config = _HF_NAMES[dataset_name]

    default_limit = 50_000_000 if split == "train" else 5_000_000
    n_tokens_limit = config["data"].get("wikitext_token_limit", default_limit)

    print(f"    {dataset_name} ({split}) — limit {n_tokens_limit:,} tokens")
    ds = load_dataset(hf_name, hf_config, split=split, streaming=True)

    return load_or_build(
        ds, tokenizer, config,
        dataset_split_key=f"{dataset_name}_{split}",
        n_tokens_limit=n_tokens_limit,
        text_key="text",
        force_rebuild=force_rebuild,
    )
