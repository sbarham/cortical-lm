"""Data pipeline factory."""

from __future__ import annotations
from typing import Tuple, Optional
from torch.utils.data import DataLoader, random_split

from .tokenizer import get_tokenizer, BaseTokenizer


def _get_sample_text_for_tokenizer(config: dict) -> str:
    """
    Return a sample of the training text to train BPE / char tokenizers.
    Uses a small subset to keep tokenizer training fast.
    """
    from tqdm import tqdm

    dataset = config["data"]["dataset"]
    sample_chars = config["data"].get("bpe_train_sample", 500_000)

    print(f"  Sampling up to {sample_chars:,} chars from {dataset} for tokenizer training...")

    # All sampling uses streaming=True so we only download what we need.
    if dataset == "tinystories":
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        text = ""
        with tqdm(total=sample_chars, unit="char", unit_scale=True, desc="  sampling") as pbar:
            for item in ds:
                chunk = item["text"] + "\n"
                text += chunk
                pbar.update(len(chunk))
                if len(text) >= sample_chars:
                    break
        return text[:sample_chars]

    elif dataset in ("wikitext2", "wikitext103"):
        from datasets import load_dataset
        from cortexlm.data.wikitext import _HF_NAMES
        hf_name, hf_config = _HF_NAMES[dataset]
        ds = load_dataset(hf_name, hf_config, split="train", streaming=True)
        text = ""
        with tqdm(total=sample_chars, unit="char", unit_scale=True, desc="  sampling") as pbar:
            for item in ds:
                chunk = item["text"] + "\n"
                text += chunk
                pbar.update(len(chunk))
                if len(text) >= sample_chars:
                    break
        return text[:sample_chars]

    elif dataset == "openwebtext":
        from datasets import load_dataset
        ds = load_dataset("openwebtext", split="train", streaming=True)
        text = ""
        with tqdm(total=sample_chars, unit="char", unit_scale=True, desc="  sampling") as pbar:
            for item in ds:
                chunk = item.get("text", "") + "\n"
                text += chunk
                pbar.update(len(chunk))
                if len(text) >= sample_chars:
                    break
        return text[:sample_chars]

    elif dataset == "ptb":
        from datasets import load_dataset
        try:
            ds = load_dataset("ptb_text_only", split="train", streaming=True)
            text = ""
            for item in ds:
                text += item["sentence"] + "\n"
                if len(text) >= sample_chars:
                    break
            return text[:sample_chars]
        except Exception:
            return ""

    return ""


def build_tokenizer(config: dict) -> BaseTokenizer:
    """Build and train a tokenizer from config."""
    tok_type = config["data"]["tokenizer"]
    needs_training = tok_type in ("char", "bpe", "byte_patch")

    train_text = None
    if needs_training:
        train_text = _get_sample_text_for_tokenizer(config)

    print(f"  Training {tok_type} tokenizer...")
    tok = get_tokenizer(config, train_text)
    print(f"  Tokenizer ready. vocab_size={tok.vocab_size:,}")
    return tok


def get_dataset(config: dict, tokenizer: Optional[BaseTokenizer] = None,
                force_rebuild: bool = False):
    """
    Build (train_dataset, val_dataset, test_dataset) from config.

    Tokenized data is cached to disk (data/cache/ by default) so subsequent
    runs with the same tokenizer skip re-tokenization entirely.

    Args:
        force_rebuild: ignore existing cache files and re-tokenize from scratch
    Returns: (train_ds, val_ds, test_ds, tokenizer)
    """
    if tokenizer is None:
        tokenizer = build_tokenizer(config)

    dataset_name = config["data"]["dataset"]
    rb = force_rebuild

    if dataset_name == "tinystories":
        from .tinystories import load_tinystories
        print("  Setting up TinyStories datasets...")
        train_ds = load_tinystories(config, tokenizer, split="train",      force_rebuild=rb)
        val_ds   = load_tinystories(config, tokenizer, split="validation", force_rebuild=rb)
        test_ds  = val_ds  # TinyStories has no separate test split

    elif dataset_name in ("wikitext2", "wikitext103"):
        from .wikitext import load_wikitext
        print(f"  Setting up {dataset_name} datasets...")
        train_ds = load_wikitext(config, tokenizer, split="train",      force_rebuild=rb)
        val_ds   = load_wikitext(config, tokenizer, split="validation", force_rebuild=rb)
        test_ds  = load_wikitext(config, tokenizer, split="test",       force_rebuild=rb)

    elif dataset_name == "openwebtext":
        from .openwebtext import load_openwebtext
        print("  Setting up OpenWebText dataset...")
        train_ds = load_openwebtext(config, tokenizer, split="train", force_rebuild=rb)
        val_ds   = train_ds
        test_ds  = val_ds

    elif dataset_name == "ptb":
        from .ptb import load_ptb
        print("  Tokenizing PTB train split...")
        train_ds = load_ptb(config, tokenizer, split="train")
        print("  Tokenizing PTB validation split...")
        val_ds   = load_ptb(config, tokenizer, split="validation")
        print("  Tokenizing PTB test split...")
        test_ds  = load_ptb(config, tokenizer, split="test")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds, tokenizer


def make_dataloader(dataset, config: dict, shuffle: bool = True) -> DataLoader:
    batch_size  = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 0)
    from torch.utils.data import IterableDataset
    if isinstance(dataset, IterableDataset):
        shuffle = False
        num_workers = 0   # streaming datasets have 1 shard; extra workers cause warnings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
