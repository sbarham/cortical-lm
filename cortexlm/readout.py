"""ReadoutHead: maps L5 population activations to next-token logits."""

import torch
import torch.nn as nn


class ReadoutHead(nn.Module):
    """
    Maps concatenated L5_E outputs across all columns to next-token logits.

    Input:  [batch, n_columns * n_l5e]
    Output: [batch, vocab_size]  (no softmax — use F.cross_entropy)

    Architecture: n_readout_layers linear layers with LayerNorm + ReLU,
    then a final linear to vocab_size.

    LayerNorm is used (not BatchNorm) for sequence-length independence
    and biological plausibility (approximates gain normalization).
    """

    def __init__(self, input_dim: int, vocab_size: int, config: dict):
        super().__init__()

        rcfg = config.get("readout", {})
        n_layers   = rcfg.get("n_layers", 2)
        hidden_dim = rcfg.get("hidden_dim", 256)

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, l5_concat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l5_concat: [batch, n_columns * n_l5e]
        Returns:
            logits: [batch, vocab_size]
        """
        return self.net(l5_concat)
