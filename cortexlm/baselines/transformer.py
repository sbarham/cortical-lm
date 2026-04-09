"""Causal transformer baseline with optional RoPE and SwiGLU."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import BaselineModel


# ── Rotary Position Embedding (RoPE) ──────────────────────────────────────────

class RoPE(nn.Module):
    """
    Rotary Position Embedding (Su et al. 2021).
    Applied to Q and K inside each attention head.
    Uses the "split-half" formulation (HuggingFace convention).
    """
    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        half = d_head // 2
        theta = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        pos   = torch.arange(max_seq_len).float()
        freqs = torch.outer(pos, theta)              # [T, half]
        cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, d_head]
        sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_heads, T, d_head]
        T   = x.shape[2]
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_head]
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


# ── Feed-forward variants ──────────────────────────────────────────────────────

class GELUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU (Shazeer 2020): silu(gate(x)) * up(x) → down.
    Uses d_ff = 8/3 * d_model (rounded to multiple of 64) so param count
    matches a standard 4× GELU FFN (3 matrices vs 2 at ⅔ the width).
    """
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = (int(d_model * 8 / 3) + 63) // 64 * 64
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Attention ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 rope: Optional[RoPE] = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.rope    = rope
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        scale = math.sqrt(self.d_head)
        att   = (q @ k.transpose(-2, -1)) / scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = F.softmax(att, dim=-1)
        out   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ── Transformer block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seq_len: int,
                 rope: Optional[RoPE] = None, activation: str = "gelu"):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, rope=rope)
        self.ln2  = nn.LayerNorm(d_model)
        if activation == "swiglu":
            self.ff = SwiGLUFeedForward(d_model)
        else:
            self.ff = GELUFeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────

class TransformerBaseline(BaselineModel):
    """
    Pre-norm causal transformer (GPT-2 style), extended with optional RoPE
    and SwiGLU.

    pos_encoding : "learned" | "rope"
    activation   : "gelu"   | "swiglu"

    All other defaults match the original GPT-2 style model so existing
    checkpoints and scripts remain compatible.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        seq_len: int,
        pos_encoding: str = "learned",
        activation: str = "gelu",
    ):
        super().__init__()
        self.pos_encoding = pos_encoding

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model) if pos_encoding == "learned" else None

        rope = RoPE(d_model // n_heads, max_seq_len=seq_len) if pos_encoding == "rope" else None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, seq_len,
                             rope=rope, activation=activation)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_sequence: torch.Tensor,
        initial_state=None,
    ) -> Tuple[torch.Tensor, None]:
        B, T = token_sequence.shape
        x = self.tok_emb(token_sequence)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=token_sequence.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x), None

    def init_state(self, batch_size: int):
        return None
