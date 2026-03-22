"""Vanilla RNN + additive attention over hidden state history."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import BaselineModel


class RNNWithAttention(BaselineModel):
    """
    Vanilla RNN with additive (Bahdanau-style) attention over the hidden history.

    At each step t, computes attention weights over h_0...h_{t-1},
    creates a context vector, concatenates with h_t, and feeds to readout.

    This isolates the contribution of attention vs. plain recurrence.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True, nonlinearity="tanh",
        )
        # Attention: score(h_t, h_s) = v^T tanh(W1*h_t + W2*h_s)
        self.attn_W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v  = nn.Linear(hidden_size, 1, bias=False)

        self.readout = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(
        self,
        token_sequence: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_sequence)           # [B, T, E]
        out, h = self.rnn(emb, initial_state)          # out: [B, T, H]

        B, T, H = out.shape
        logits_list = []

        for t in range(T):
            h_t = out[:, t:t+1, :]   # [B, 1, H]
            if t == 0:
                # No history; use h_t as context
                context = h_t.squeeze(1)
            else:
                history = out[:, :t, :]   # [B, t, H]
                # Additive attention scores: [B, t, 1]
                scores = self.attn_v(
                    torch.tanh(self.attn_W1(h_t) + self.attn_W2(history))
                )                                        # [B, t, 1]
                weights = F.softmax(scores, dim=1)       # [B, t, 1]
                context = (weights * history).sum(dim=1) # [B, H]

            combined = torch.cat([out[:, t, :], context], dim=-1)  # [B, 2H]
            logits_list.append(self.readout(combined))

        logits = torch.stack(logits_list, dim=1)   # [B, T, V]
        return logits, h

    def init_state(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
