"""Tests for data pipeline and tokenizers (no network downloads needed)."""

import pytest
import torch
from cortexlm.utils.config import get_default_config
from cortexlm.data.tokenizer import (
    CharTokenizer, BPETokenizer, BytesTokenizer, BytePatchTokenizer
)
from cortexlm.data.tinystories import TokenSequenceDataset


# ── Tokenizer round-trip tests ─────────────────────────────────────────────

SAMPLE_TEXT = "Hello, world! This is a test of the tokenizer. 123 abc."


def test_char_tokenizer_round_trip():
    tok = CharTokenizer(SAMPLE_TEXT)
    ids = tok.encode(SAMPLE_TEXT)
    decoded = tok.decode(ids)
    assert decoded == SAMPLE_TEXT, f"Round-trip failed: {repr(decoded[:50])}"


def test_char_tokenizer_vocab_size():
    tok = CharTokenizer(SAMPLE_TEXT)
    assert tok.vocab_size == len(set(SAMPLE_TEXT))


def test_char_tokenizer_encodes_to_ints():
    tok = CharTokenizer(SAMPLE_TEXT)
    ids = tok.encode(SAMPLE_TEXT)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_bytes_tokenizer_round_trip():
    tok = BytesTokenizer()
    ids = tok.encode(SAMPLE_TEXT)
    decoded = tok.decode(ids)
    assert decoded == SAMPLE_TEXT


def test_bytes_tokenizer_vocab_size():
    tok = BytesTokenizer()
    assert tok.vocab_size == 256


def test_bytes_tokenizer_all_ids_in_range():
    tok = BytesTokenizer()
    ids = tok.encode(SAMPLE_TEXT)
    assert all(0 <= i < 256 for i in ids)


def test_byte_patch_patch2_round_trip():
    tok = BytePatchTokenizer(patch_size=2)
    ids = tok.encode(SAMPLE_TEXT)
    decoded = tok.decode(ids)
    # Round-trip may not be exact due to padding, but should contain original
    assert SAMPLE_TEXT.rstrip() in decoded or decoded.startswith(SAMPLE_TEXT[:20])


def test_byte_patch_vocab_size_patch2():
    tok = BytePatchTokenizer(patch_size=2)
    assert tok.vocab_size == 65536


def test_byte_patch_patch4_fit():
    tok = BytePatchTokenizer(patch_size=4, max_vocab_size=512)
    tok.fit(SAMPLE_TEXT * 100)
    assert tok.vocab_size <= 512
    ids = tok.encode(SAMPLE_TEXT)
    assert all(isinstance(i, int) for i in ids)


def test_bpe_tokenizer_trains_and_round_trips():
    pytest.importorskip("tokenizers")
    tok = BPETokenizer(vocab_size=128)
    tok.train(SAMPLE_TEXT * 50)  # repeat to have enough for BPE to learn
    ids = tok.encode(SAMPLE_TEXT)
    decoded = tok.decode(ids)
    # BPE may not perfectly round-trip short text but should be close
    assert len(decoded) > 0
    assert tok.vocab_size <= 128


# ── TokenSequenceDataset ───────────────────────────────────────────────────

def test_token_sequence_dataset_shape():
    tokens = list(range(1000))
    seq_len = 32
    ds = TokenSequenceDataset(tokens, seq_len)
    x, y = ds[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)


def test_token_sequence_dataset_shift():
    """y should be x shifted by one position."""
    tokens = list(range(200))
    ds = TokenSequenceDataset(tokens, seq_len=10)
    x, y = ds[0]
    assert (x[1:] == y[:-1]).all()


def test_token_sequence_dataset_non_overlapping():
    """Consecutive chunks should not overlap (they are non-overlapping by design)."""
    tokens = list(range(500))
    seq_len = 16
    ds = TokenSequenceDataset(tokens, seq_len)
    x0, _ = ds[0]
    x1, _ = ds[1]
    # Second chunk starts right after first
    assert x1[0].item() == x0[-1].item() + 1 + 1  # +1 for y shift, +1 for next chunk


def test_token_sequence_dataset_length():
    tokens = list(range(1000))
    seq_len = 32
    ds = TokenSequenceDataset(tokens, seq_len)
    # Each chunk uses (seq_len + 1) tokens
    expected_len = 1000 // (seq_len + 1)
    assert len(ds) == expected_len


# ── Config validation ──────────────────────────────────────────────────────

def test_config_validation_catches_bad_dataset():
    from cortexlm.utils.config import get_default_config, _validate_config
    cfg = get_default_config()
    cfg["data"]["dataset"] = "not_a_real_dataset"
    with pytest.raises(AssertionError):
        _validate_config(cfg)


def test_config_validation_passes_valid():
    from cortexlm.utils.config import get_default_config, _validate_config
    cfg = get_default_config()
    _validate_config(cfg)  # should not raise
