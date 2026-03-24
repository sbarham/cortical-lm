"""OpenWebText dataset — streaming mode to avoid downloading all 40GB."""

from __future__ import annotations
from cortexlm.data.tinystories import StreamingTokenDataset  # noqa: F401 (re-export)


def load_openwebtext(config: dict, tokenizer, split: str = "train",
                     force_rebuild: bool = False):
    from datasets import load_dataset
    from .cache import load_or_build

    ds = load_dataset("openwebtext", split=split, streaming=True)
    n_tokens_limit = config["data"].get("openwebtext_token_limit", 10_000_000)

    print(f"    OpenWebText ({split}) — limit {n_tokens_limit:,} tokens")
    return load_or_build(
        ds, tokenizer, config,
        dataset_split_key=f"openwebtext_{split}",
        n_tokens_limit=n_tokens_limit,
        text_key="text",
        force_rebuild=force_rebuild,
    )
