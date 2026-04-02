"""Thalamic relay module: per-column projection from rich embedding to column input."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class ThalamicRelayModule(nn.Module):
    """
    Two-stage thalamic relay: rich embedding → per-column column-input projection.

    Stage 1 (upstream): nn.Embedding(vocab_size, embed_dim_large) lives in CortexLM.
    Stage 2 (this module): W_relay [n_cols, col_input_dim, embed_dim_large]
        thal_c = tok_emb @ W_relay[c].T  →  [batch, col_input_dim] per column

    Biological analogy: different thalamic nuclei project differently to different
    cortical areas.  Each column receives a distinct linear read-out of the shared
    rich token embedding.

    Stage 3 (optional): TRN-style divisive normalization across columns.
        norm_c = ||thal_c||_2  (per column, per batch item)
        thal_c_norm = thal_c / (1 + eta * mean_over_cols(norm_c))
    eta is a learnable scalar initialized to trn_eta_init.

    Config flags:
        thalamus.trn_competition: bool  (default False)
        thalamus.trn_eta_init:    float (default 0.1)
        thalamus.relay_init_scale: float (default 0.02)
    """

    def __init__(
        self,
        n_cols: int,
        embed_dim_large: int,
        col_input_dim: int,
        trn_competition: bool = False,
        trn_eta_init: float = 0.1,
        relay_init_scale: float = 0.02,
    ):
        super().__init__()
        self.n_cols = n_cols
        self.col_input_dim = col_input_dim
        self.trn_competition = trn_competition

        # W_relay: [n_cols, col_input_dim, embed_dim_large]
        self.W_relay = nn.Parameter(
            torch.randn(n_cols, col_input_dim, embed_dim_large) * relay_init_scale
        )

        if trn_competition:
            # Learnable TRN suppression strength; initialized to trn_eta_init
            self.trn_eta = nn.Parameter(torch.tensor(trn_eta_init))

    def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tok_emb: [batch, embed_dim_large]
        Returns:
            thal_relay: [batch, n_cols, col_input_dim]
        """
        # einsum: b=batch, e=embed_dim_large, c=n_cols, d=col_input_dim
        thal_relay = torch.einsum("be,cde->bcd", tok_emb, self.W_relay)

        if self.trn_competition:
            # Divisive normalization across columns
            norm_c = thal_relay.norm(dim=-1, keepdim=True)         # [batch, n_cols, 1]
            mean_norm = norm_c.mean(dim=1, keepdim=True)           # [batch, 1, 1]
            thal_relay = thal_relay / (1.0 + self.trn_eta * mean_norm)

        return thal_relay
