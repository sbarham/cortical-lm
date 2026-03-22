"""Vanilla RNN baseline."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import BaselineModel


class VanillaRNN(BaselineModel):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True, nonlinearity="tanh",
        )
        self.readout = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(
        self,
        token_sequence: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_sequence)       # [B, T, E]
        out, h = self.rnn(emb, initial_state)      # out: [B, T, H]
        logits = self.readout(out)                 # [B, T, V]
        return logits, h

    def init_state(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.rnn.hidden_size, device=device)
