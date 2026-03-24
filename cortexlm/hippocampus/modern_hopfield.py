"""Modern Hopfield Network hippocampal module (Ramsauer et al. 2020)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .base import HippocampalModule


class ModernHopfieldHippocampus(HippocampalModule):
    """
    Modern Hopfield Network implementing CA3-style associative sequence memory.

    Mathematically equivalent to transformer attention, but with persistent
    stored memories (Xi) as learnable parameters — not recomputed per sequence.

    Update rule:
        state_new = Xi · softmax(β · Xi^T · state_query)

    Xi: [n_memories, d_model] — persistent memory patterns (learnable parameter)
    state_query: [batch, d_model] — projection of concatenated L5 outputs

    Output is projected back to column modulation dim and broadcast to all columns.

    Optional CA1 mismatch signal: L2 distance between retrieved pattern and
    current cortical state, broadcast as surprise scalar.
    """

    def __init__(self, config: dict, n_columns: int, n_l5e: int):
        super().__init__(config, n_columns, n_l5e)

        hcfg = config["hippocampus"]
        self.n_memories = hcfg.get("n_memories", 512)
        self.d_model    = hcfg.get("d_model", 256)
        self.beta       = hcfg.get("beta", 1.0)
        self.ca1        = hcfg.get("ca1", False)

        cortical_dim = n_columns * n_l5e  # concatenated L5 across all columns

        # Query projection: cortical state → d_model
        self.query_proj = nn.Linear(cortical_dim, self.d_model)

        # Stored memory patterns (persistent learnable parameter)
        self.Xi = nn.Parameter(torch.randn(self.n_memories, self.d_model) * 0.02)

        # Diagnostic: last attention weights [batch, n_memories], detached, set each forward pass
        self._last_attn_weights: Optional[torch.Tensor] = None

        # Output projection: d_model → per-column modulation
        # Each column gets a modulation vector of dim modulation_dim
        self.modulation_dim = max(1, self.d_model // n_columns)
        self.out_proj = nn.Linear(self.d_model, n_columns * self.modulation_dim)

        # CA1: project cortical state to d_model for comparison
        if self.ca1:
            self.ca1_proj = nn.Linear(cortical_dim, self.d_model)

    def forward(
        self,
        cortical_state_l5: torch.Tensor,   # [batch, n_columns * n_l5e]
        column_states=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = cortical_state_l5.shape[0]

        # Project cortical state to query
        query = self.query_proj(cortical_state_l5)   # [batch, d_model]

        # Modern Hopfield retrieval:
        # attention = softmax(β · Xi^T · query^T)   Xi: [M, d], query: [B, d]
        # scores: [B, M]
        scores = self.beta * (query @ self.Xi.t())   # [batch, n_memories]
        weights = F.softmax(scores, dim=-1)           # [batch, n_memories]
        self._last_attn_weights = weights.detach()    # store for diagnostics

        # Retrieved pattern: [batch, d_model]
        retrieved = weights @ self.Xi   # [batch, d_model]

        # Project to per-column modulation
        mod_flat = self.out_proj(retrieved)                              # [batch, n_cols*mod_dim]
        modulation = mod_flat.view(batch, self.n_columns, self.modulation_dim)

        # CA1 mismatch signal
        surprise = None
        if self.ca1:
            cortical_in_dmodel = self.ca1_proj(cortical_state_l5)  # [batch, d_model]
            surprise = torch.norm(retrieved - cortical_in_dmodel, dim=-1, keepdim=True)
            # [batch, 1]

        return modulation, surprise

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return {}
