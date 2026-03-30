"""Modern Hopfield Network hippocampal module (Ramsauer et al. 2020)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .base import HippocampalModule


class ModernHopfieldHippocampus(HippocampalModule):
    """
    Modern Hopfield Network implementing CA3+CA1 hippocampal memory.

    CA3 (base):
        Persistent stored memories Xi as learnable parameters.  Retrieval is
        modern Hopfield attention: retrieved = Xi · softmax(β · Xi^T · query).
        Output projected to per-column thalamic modulation.

    CA1 (optional, ca1: true):
        Models the entorhinal→CA1 direct pathway that bypasses CA3.

        In real hippocampus CA1 sits at the junction of two streams:
          - CA3 → CA1 (Schaffer collaterals): pattern-completed retrieved memory
          - EC layer III → CA1 (temporoammonic path): actual current input

        The mismatch (prediction error) serves two functions:
          1. Gates Xi writes: high surprise → Xi gradient allowed through → memory
             updated.  Low surprise → Xi gradient suppressed → familiar memory
             protected from overwriting.  Implemented by scaling 'retrieved' by
             a sigmoid gate before out_proj; gradients into Xi scale with surprise.
          2. Error feedback to cortex: the directional error vector is projected
             back to thalamic modulation space (CA1 → EC layer V → cortex),
             telling the cortex "here is how reality departs from the memory."

        This decomposition means the cortex receives both the CA3 context AND
        the CA1 correction, while Xi only consolidates surprising patterns.

    Parameters
    ----------
    n_memories       : number of stored memory patterns
    d_model          : dimensionality of the memory space
    beta             : inverse temperature for retrieval softmax
    ca1              : enable CA1 prediction-error gating (default False)
    normalize_xi     : normalize Xi rows to unit sphere after each optimizer step
                       (default False).  Stabilises attention score magnitudes and
                       prevents retrieval collapse.  Call normalize_xi_rows() from
                       the trainer after optimizer.step().
    gated_error_vec  : use gated_retrieved (not retrieved) to compute error_vec,
                       so the write gate also suppresses Xi gradients through the
                       CA1 feedback path (default False — legacy behaviour).
    forward_gate_ca1 : multiply ca1_mod by write_gate in the forward pass so
                       high-surprise states produce stronger CA1 error feedback
                       (default False — gradient-only gating).
    """

    def __init__(self, config: dict, n_columns: int, n_l5e: int):
        super().__init__(config, n_columns, n_l5e)

        hcfg = config["hippocampus"]
        self.n_memories       = hcfg.get("n_memories", 512)
        self.d_model          = hcfg.get("d_model", 256)
        self.beta             = hcfg.get("beta", 1.0)
        self.ca1              = hcfg.get("ca1", False)
        self.normalize_xi     = hcfg.get("normalize_xi", False)
        self.gated_error_vec  = hcfg.get("gated_error_vec", False)
        self.forward_gate_ca1 = hcfg.get("forward_gate_ca1", False)

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

        # CA1: entorhinal -> CA1 direct pathway (bypasses CA3).
        # ca1_proj     : cortical state -> d_model  (EC layer-III observation stream)
        # ca1_out_proj : error vector   -> per-column modulation  (CA1 -> EC layer V -> cortex)
        # surprise_scale: learnable scalar log-temperature for the sigmoid write gate
        if self.ca1:
            self.ca1_proj      = nn.Linear(cortical_dim, self.d_model)
            self.ca1_out_proj  = nn.Linear(self.d_model, n_columns * self.modulation_dim)
            # Learnable temperature: gate = sigmoid(exp(surprise_scale) * surprise_norm)
            # Init near 0 so gate starts at sigmoid(0) = 0.5 (half-open) for all inputs.
            self.surprise_scale = nn.Parameter(torch.zeros(1))

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

        surprise = None
        if self.ca1:
            # ── CA1: prediction-error gating ──────────────────────────────────
            # EC layer-III observation stream: project current cortical state to d_model
            ec_obs = self.ca1_proj(cortical_state_l5)             # [batch, d_model]

            # ── Surprise magnitude (computed from ungated error for scale) ────
            # Use ungated retrieved here so surprise reflects true prediction error.
            raw_error     = retrieved - ec_obs                     # [batch, d_model]
            surprise_norm = torch.norm(raw_error, dim=-1, keepdim=True)  # [batch, 1]
            surprise = surprise_norm                               # stored for logging

            # ── Function 1: gate Xi writes ────────────────────────────────────
            # sigmoid(temperature * surprise) in [0,1]:
            #   high surprise -> gate near 1 -> gradients flow into Xi -> memory updated
            #   low surprise  -> gate near 0 -> Xi gradients suppressed -> memory protected
            # 'retrieved' enters the gate; gradients through Xi scale with gate value.
            # retrieved.detach() provides the "no update" baseline that the gate
            # interpolates against.
            write_gate = torch.sigmoid(
                self.surprise_scale.exp() * surprise_norm
            )                                                       # [batch, 1]
            gated_retrieved = (
                write_gate * retrieved
                + (1.0 - write_gate) * retrieved.detach()
            )                                                       # [batch, d_model]

            # ── Project gated retrieval to cortical modulation (CA3 path) ────
            mod_flat   = self.out_proj(gated_retrieved)            # [batch, n_cols*mod_dim]
            modulation = mod_flat.view(batch, self.n_columns, self.modulation_dim)

            # ── Function 2: error feedback to cortex (CA1 -> EC layer V) ─────
            # gated_error_vec=True (1k/1l): use gated_retrieved so the write gate
            #   also suppresses Xi gradients through this path (fixes gradient leak).
            # gated_error_vec=False (legacy 1i): use raw retrieved, leaking gradients.
            error_vec  = (gated_retrieved if self.gated_error_vec else retrieved) - ec_obs
            ca1_flat   = self.ca1_out_proj(error_vec)              # [batch, n_cols*mod_dim]
            ca1_mod    = ca1_flat.view(batch, self.n_columns, self.modulation_dim)

            # forward_gate_ca1=True (1l): scale ca1_mod by write_gate so surprised
            #   states produce stronger cortical error feedback.
            # forward_gate_ca1=False (1k and legacy): gradient-only gating.
            if self.forward_gate_ca1:
                ca1_mod = write_gate.unsqueeze(1) * ca1_mod

            modulation = modulation + ca1_mod

        else:
            # No CA1: standard CA3 retrieval projected to modulation
            mod_flat   = self.out_proj(retrieved)                  # [batch, n_cols*mod_dim]
            modulation = mod_flat.view(batch, self.n_columns, self.modulation_dim)

        return modulation, surprise

    def normalize_xi_rows(self) -> None:
        """Normalize Xi rows to unit sphere (call from trainer after optimizer.step()).

        Stabilises Hopfield attention score magnitudes and prevents retrieval
        collapse when softmax concentrates on a small number of memories.
        Only active when hippocampus.normalize_xi: true in config.
        """
        if self.normalize_xi:
            with torch.no_grad():
                self.Xi.data = F.normalize(self.Xi.data, dim=-1)

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return {}
