"""LSTM + additive attention over hidden state history."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import BaselineModel


class LSTMWithAttention(BaselineModel):
    """
    LSTM with additive (Bahdanau-style) attention over the hidden history.
    Same attention mechanism as RNNWithAttention but with LSTM dynamics.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True,
        )
        self.attn_W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v  = nn.Linear(hidden_size, 1, bias=False)

        self.readout = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(
        self,
        token_sequence: torch.Tensor,
        initial_state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        emb = self.embedding(token_sequence)
        out, (h, c) = self.lstm(emb, initial_state)

        B, T, H = out.shape
        logits_list = []

        for t in range(T):
            h_t = out[:, t:t+1, :]
            if t == 0:
                context = h_t.squeeze(1)
            else:
                history = out[:, :t, :]
                scores = self.attn_v(
                    torch.tanh(self.attn_W1(h_t) + self.attn_W2(history))
                )
                weights = F.softmax(scores, dim=1)
                context = (weights * history).sum(dim=1)

            combined = torch.cat([out[:, t, :], context], dim=-1)
            logits_list.append(self.readout(combined))

        logits = torch.stack(logits_list, dim=1)
        return logits, (h, c)

    def init_state(self, batch_size: int):
        device = next(self.parameters()).device
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
