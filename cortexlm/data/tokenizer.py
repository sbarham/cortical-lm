"""Tokenizer implementations: char, BPE, bytes, byte_patch, tiktoken."""

from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import torch


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    def avg_bytes_per_token(self) -> float:
        """Average UTF-8 bytes per token, computed from vocabulary.
        Default: 1.0 (correct for char/byte tokenizers)."""
        return 1.0


# ── Character tokenizer ────────────────────────────────────────────────────

class CharTokenizer(BaseTokenizer):
    """Maps unique characters to integer IDs. Vocab inferred from training text."""

    def __init__(self, text: Optional[str] = None):
        self._char2id: Dict[str, int] = {}
        self._id2char: Dict[int, str] = {}
        if text is not None:
            self.fit(text)

    def fit(self, text: str):
        chars = sorted(set(text))
        self._char2id = {c: i for i, c in enumerate(chars)}
        self._id2char = {i: c for c, i in self._char2id.items()}

    def encode(self, text: str) -> List[int]:
        return [self._char2id.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._id2char.get(i, "") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._char2id)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"char2id": self._char2id}, f)

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        import json
        tok = cls()
        with open(path) as f:
            d = json.load(f)
        tok._char2id = d["char2id"]
        tok._id2char = {int(i): c for c, i in tok._char2id.items()}
        return tok


# ── BPE tokenizer ─────────────────────────────────────────────────────────

class BPETokenizer(BaseTokenizer):
    """HuggingFace tokenizers BPE, trained on provided text with configurable vocab_size."""

    def __init__(self, vocab_size: int = 4096):
        self._vocab_size = vocab_size
        self._tokenizer = None

    def train(self, text: str):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=["[UNK]", "[PAD]"],
            show_progress=True,
        )
        tokenizer.train_from_iterator([text], trainer=trainer)
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("BPETokenizer must be trained before encoding")
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("BPETokenizer must be trained before decoding")
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return self._vocab_size

    def avg_bytes_per_token(self) -> float:
        """Compute average UTF-8 bytes per token from the vocabulary.
        Excludes special tokens. Used to convert bpt → bpb."""
        if self._tokenizer is None:
            return 1.0
        special = {"[UNK]", "[PAD]"}
        vocab = self._tokenizer.get_vocab()  # {token_str: id}
        total_bytes = 0
        n = 0
        for token_str, token_id in vocab.items():
            if token_str in special:
                continue
            try:
                text = self._tokenizer.decode([token_id])
                total_bytes += len(text.encode("utf-8"))
                n += 1
            except Exception:
                pass
        return total_bytes / max(n, 1)

    def save(self, path: str):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained")
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        from tokenizers import Tokenizer
        tok = cls()
        tok._tokenizer = Tokenizer.from_file(path)
        tok._vocab_size = tok._tokenizer.get_vocab_size()
        return tok


# ── Bytes tokenizer ────────────────────────────────────────────────────────

class BytesTokenizer(BaseTokenizer):
    """Raw bytes tokenizer. vocab_size=256. No OOV."""

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256


# ── Byte patch tokenizer ───────────────────────────────────────────────────

class BytePatchTokenizer(BaseTokenizer):
    """
    Groups bytes into non-overlapping patches of patch_size bytes.

    patch_size=2: vocab_size = 256^2 = 65536 (feasible)
    patch_size=4: too large; use hash-based vocabulary capped at max_vocab_size
    patch_size=8: too large; same.

    For patch_size >= 4: build a vocabulary of the most frequent patches seen
    during training, up to max_vocab_size. Unknown patches map to a special ID.
    """

    def __init__(self, patch_size: int = 4, max_vocab_size: int = 8192):
        self.patch_size = patch_size
        self.max_vocab_size = max_vocab_size
        self._patch2id: Dict[bytes, int] = {}
        self._id2patch: Dict[int, bytes] = {}
        self._unk_id = 0

        if patch_size == 2:
            # Enumerate all 65536 patches
            self._build_full_vocab()

    def _build_full_vocab(self):
        """For patch_size=2, enumerate all possible byte pairs."""
        self._patch2id = {}
        idx = 0
        for b0 in range(256):
            for b1 in range(256):
                patch = bytes([b0, b1])
                self._patch2id[patch] = idx
                self._id2patch[idx] = patch
                idx += 1
        self._unk_id = 0  # no UNK needed; all patches covered

    def fit(self, text: str):
        """For patch_size >= 4: build vocab from most frequent patches in text."""
        if self.patch_size == 2:
            return  # full vocab already built
        raw = text.encode("utf-8")
        from collections import Counter
        patches = []
        for i in range(0, len(raw) - self.patch_size + 1, self.patch_size):
            patches.append(bytes(raw[i:i + self.patch_size]))
        counts = Counter(patches)
        # Reserve ID 0 for UNK
        self._patch2id = {"[UNK]": 0}  # type: ignore
        self._id2patch = {0: b""}
        for idx, (patch, _) in enumerate(counts.most_common(self.max_vocab_size - 1), start=1):
            self._patch2id[patch] = idx
            self._id2patch[idx] = patch
        self._unk_id = 0

    def encode(self, text: str) -> List[int]:
        raw = text.encode("utf-8")
        # Pad to multiple of patch_size
        pad = (self.patch_size - len(raw) % self.patch_size) % self.patch_size
        raw = raw + b"\x00" * pad
        ids = []
        for i in range(0, len(raw), self.patch_size):
            patch = bytes(raw[i:i + self.patch_size])
            ids.append(self._patch2id.get(patch, self._unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = b"".join(self._id2patch.get(i, b"") for i in ids)
        return parts.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        if self.patch_size == 2:
            return 65536
        return len(self._patch2id)


# ── tiktoken tokenizer ─────────────────────────────────────────────────────

class TiktokenTokenizer(BaseTokenizer):
    """Thin wrapper around tiktoken. Fixed vocabulary (cl100k_base: 100k+ tokens)."""

    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab


# ── Factory ────────────────────────────────────────────────────────────────

def get_tokenizer(config: dict, train_text: Optional[str] = None) -> BaseTokenizer:
    """
    Build tokenizer from config. If tokenizer needs training (bpe, char, byte_patch≥4),
    train_text must be provided.
    """
    tok_type  = config["data"]["tokenizer"]
    vocab_size = config["data"].get("vocab_size", 4096)

    if tok_type == "char":
        tok = CharTokenizer()
        if train_text:
            tok.fit(train_text)
        return tok

    elif tok_type == "bpe":
        tok = BPETokenizer(vocab_size=vocab_size or 4096)
        if train_text:
            tok.train(train_text)
        return tok

    elif tok_type == "bytes":
        return BytesTokenizer()

    elif tok_type == "byte_patch":
        ps = config["data"].get("byte_patch_size", 4)
        tok = BytePatchTokenizer(patch_size=ps, max_vocab_size=vocab_size or 8192)
        if ps >= 4 and train_text:
            tok.fit(train_text)
        return tok

    elif tok_type == "tiktoken":
        enc = config["data"].get("tiktoken_encoding", "cl100k_base")
        return TiktokenTokenizer(encoding=enc)

    else:
        raise ValueError(f"Unknown tokenizer: {tok_type}")
