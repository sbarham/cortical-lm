"""ReadoutHead: maps L5 population activations to next-token logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Activation helpers ────────────────────────────────────────────────────────

class _SwiGLUBlock(nn.Module):
    """SwiGLU: silu(gate(x)) * up(x) — two parallel linear projections."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.up   = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.up(x)


_ACTIVATIONS = {
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "swish": nn.SiLU,
    "silu":  nn.SiLU,
}


class _ReadoutBlock(nn.Module):
    """One hidden block: (Linear|SwiGLU) → LayerNorm → Activation.

    SwiGLU replaces both the linear projection and the activation, so no
    separate activation module is added in that case.
    """
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        super().__init__()
        if activation == "swiglu":
            self.proj = _SwiGLUBlock(in_dim, out_dim)
        else:
            act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
            self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), act_cls())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


# ── ReadoutHead ───────────────────────────────────────────────────────────────

class ReadoutHead(nn.Module):
    """
    Maps concatenated L5_E outputs across all columns to next-token logits.

    Input:  [batch, n_columns * n_l5e]
    Output: [batch, vocab_size]  (no softmax — use F.cross_entropy)

    Architecture: n_layers hidden blocks of (Linear|SwiGLU) → LayerNorm,
    with optional residual skip connections from block input to output
    (applied on layers 2+ where dimensions match), then either:
      - a final Linear(hidden_dim, vocab_size)           [weight_tying: false]
      - F.linear(hidden, embedding.weight)               [weight_tying: true]

    Config keys (all under readout:):
      n_layers     int   Number of hidden blocks          (default: 1)
      hidden_dim   int   Hidden dimension per block       (default: 128)
      activation   str   relu | gelu | swish | swiglu     (default: relu)
      residual     bool  Add skip connections where dims match (default: false)
      weight_tying bool  Reuse embedding matrix as output projection (default: false)
    """

    def __init__(self, input_dim: int, vocab_size: int, config: dict):
        super().__init__()

        rcfg       = config.get("readout", {})
        n_layers   = rcfg.get("n_layers", 1)
        hidden_dim = rcfg.get("hidden_dim", 128)
        activation = rcfg.get("activation", "relu").lower()
        residual   = bool(rcfg.get("residual", False))
        self.weight_tying = rcfg.get("weight_tying", False)
        embed_dim  = config.get("embedding", {}).get("dim", 64)

        self._residual = residual

        self._blocks = nn.ModuleList()
        self._block_in_dims: list[int] = []
        in_dim = input_dim
        for _ in range(n_layers):
            self._blocks.append(_ReadoutBlock(in_dim, hidden_dim, activation))
            self._block_in_dims.append(in_dim)
            in_dim = hidden_dim

        if self.weight_tying:
            self._bridge = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else None
            self._tied_weight: torch.Tensor | None = None
        else:
            self._bridge = None
            self._out = nn.Linear(in_dim, vocab_size)

    # ── Weight tying ─────────────────────────────────────────────────────────

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """
        Share the output projection with the input embedding matrix.

        Call this once after both CortexLM and ReadoutHead are constructed:
            model.readout.tie_weights(model.embedding.weight)

        embedding_weight: the nn.Embedding.weight Parameter [vocab_size, embed_dim].
        F.linear(hidden, embedding_weight) computes hidden @ embedding_weight.T,
        mapping [batch, embed_dim] -> [batch, vocab_size].

        Uses object.__setattr__ to store the reference without PyTorch registering
        it as a parameter of this module — it remains a parameter of nn.Embedding
        only, preventing double-counting in parameter counts and optimizer state.
        """
        object.__setattr__(self, '_tied_weight', embedding_weight)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, l5_concat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l5_concat: [batch, n_columns * n_l5e]
        Returns:
            logits: [batch, vocab_size]
        """
        x = l5_concat
        for blk, in_dim in zip(self._blocks, self._block_in_dims):
            out = blk(x)
            if self._residual and in_dim == out.shape[-1]:
                out = out + x
            x = out

        if self.weight_tying:
            if self._bridge is not None:
                x = self._bridge(x)
            return F.linear(x, self._tied_weight)
        return self._out(x)
