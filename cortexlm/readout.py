"""ReadoutHead: maps L5 population activations to next-token logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadoutHead(nn.Module):
    """
    Maps concatenated L5_E outputs across all columns to next-token logits.

    Input:  [batch, n_columns * n_l5e]
    Output: [batch, vocab_size]  (no softmax — use F.cross_entropy)

    Architecture: n_readout_layers linear layers with LayerNorm + ReLU,
    then either:
      - a final Linear(hidden_dim, vocab_size)           [weight_tying: false]
      - F.linear(hidden, embedding.weight)               [weight_tying: true]

    Weight tying (readout.weight_tying: true):
        The output projection reuses the input embedding matrix transposed,
        so vocab-size parameters are counted only once.  Requires the final
        hidden representation to live in embedding_dim space; if hidden_dim !=
        embedding_dim a small bridging linear is added automatically.

    LayerNorm is used (not BatchNorm) for sequence-length independence
    and biological plausibility (approximates gain normalisation).
    """

    def __init__(self, input_dim: int, vocab_size: int, config: dict):
        super().__init__()

        rcfg       = config.get("readout", {})
        n_layers   = rcfg.get("n_layers", 2)
        hidden_dim = rcfg.get("hidden_dim", 256)
        self.weight_tying = rcfg.get("weight_tying", False)
        embed_dim  = config.get("embedding", {}).get("dim", 64)

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            in_dim = hidden_dim

        if self.weight_tying:
            # Bridge to embedding_dim if necessary, then use tied output projection.
            if in_dim != embed_dim:
                layers.append(nn.Linear(in_dim, embed_dim))
                in_dim = embed_dim
            # No final linear added here — forward() uses F.linear with tied weight.
            self._tied_weight: torch.Tensor | None = None
        else:
            layers.append(nn.Linear(in_dim, vocab_size))

        self.net = nn.Sequential(*layers)

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
        hidden = self.net(l5_concat)
        if self.weight_tying:
            # tied_weight: [vocab_size, embed_dim] — F.linear transposes it
            return F.linear(hidden, self._tied_weight)
        return hidden
