"""LSTM baseline."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import BaselineModel


class LSTMBaseline(BaselineModel):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(
        self,
        token_sequence: torch.Tensor,
        initial_state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        emb = self.embedding(token_sequence)
        out, (h, c) = self.lstm(emb, initial_state)
        logits = self.readout(out)
        return logits, (h, c)

    def init_state(self, batch_size: int):
        device = next(self.parameters()).device
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
