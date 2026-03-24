"""Penn Treebank character-level dataset."""

from __future__ import annotations
import os
from typing import Optional
from cortexlm.data.tinystories import TokenSequenceDataset


_PTB_SPLITS = {
    "train": "ptb.train.txt",
    "validation": "ptb.valid.txt",
    "test": "ptb.test.txt",
}

_PTB_HF_SPLITS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def load_ptb(config: dict, tokenizer, split: str = "train", data_dir: Optional[str] = None):
    """
    Load Penn Treebank character-level.

    Tries local files first (data_dir), then HuggingFace datasets.

    Returns: TokenSequenceDataset
    """
    text = None

    if data_dir is not None:
        fname = _PTB_SPLITS.get(split)
        if fname:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()

    if text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("ptb_text_only", split=_PTB_HF_SPLITS.get(split, split),
                              trust_remote_code=True)
            text = "\n".join(item["sentence"] for item in ds)
        except Exception:
            raise RuntimeError(
                "Could not load PTB. Either provide data_dir with ptb.{train,valid,test}.txt "
                "or ensure HuggingFace datasets can access ptb_text_only."
            )

    tokens = tokenizer.encode(text)
    seq_len = config["data"]["seq_len"]
    return TokenSequenceDataset(tokens, seq_len)
