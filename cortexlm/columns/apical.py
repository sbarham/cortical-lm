"""Apical dendritic pathway — top-down / skip-connection inputs to L5 (or E) neurons.

Provides four variants that can be layered onto any CortexLM architecture
(phase 1a simple_ei through phase 1e Hopfield) via a single config flag:

    column:
      apical_pathway: none | skip | additive | multiplicative | corticortical

Biological motivation
---------------------
L5 pyramidal neurons have two functionally distinct input zones:

  Basal dendrites (current model)  — feedforward input from L4 via L2/3
  Apical dendrites (this module)   — top-down / feedback input arriving at L1

Apical depolarisation can trigger a dendritic calcium spike that nonlinearly
amplifies somatic output (Larkum 2013).  Even a simple additive projection
provides a gradient highway analogous to the transformer residual stream:
early in training, when inter-layer weights are uninformative, the apical
signal keeps loss gradients flowing to the readout.

Variants
--------
  skip           Direct embed → L5E additive projection.
                 Biologically: higher-order thalamus (pulvinar) bypassing L4,
                 projecting directly to L5.  Fan-in init; contributes from step 0.

  additive       Same projection with a per-neuron learnable sigmoid gate.
                 Gate initialised to ~0 — starts nearly silent, grows as learned.

  multiplicative Larkum two-compartment model:
                   I_l5e_out = I_l5e * (1 + tanh(apical_proj(embed)))
                 Weights initialised near 0 → tanh ≈ 0 → identity at start.

  corticortical  Previous-timestep L5E of column (k+1) % n_cols projects to
                 L23E of column k.  Circular top-down feedback; implementable
                 across all phase variants including simple_ei.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class ApicalPathway(nn.Module):
    """
    Apical dendritic input module.

    Parameters
    ----------
    config           : full model config dict
    n_cols           : number of columns
    embed_dim        : token embedding dimension
    n_apical_target  : neurons receiving apical input  (n_l5e for layered, n_e for simple_ei)
    n_cortical_l23   : neurons in the corticortical target layer  (n_l23e / n_e)
    n_cortical_l5    : neurons in the corticortical source layer  (n_l5e  / n_e)
    """

    VARIANTS = ("none", "skip", "additive", "multiplicative", "corticortical")

    def __init__(
        self,
        config: dict,
        n_cols: int,
        embed_dim: int,
        n_apical_target: int,
        n_cortical_l23: int,
        n_cortical_l5: int,
    ):
        super().__init__()
        self.variant = config["column"].get("apical_pathway", "none")
        if self.variant not in self.VARIANTS:
            raise ValueError(
                f"column.apical_pathway must be one of {self.VARIANTS}, got {self.variant!r}"
            )

        self.n_cols = n_cols

        if self.variant == "skip":
            # Fan-in init — same scale as the thalamic L4 projection.
            self.l5_proj = nn.Parameter(
                torch.randn(n_cols, n_apical_target, embed_dim) * (1.0 / math.sqrt(embed_dim))
            )

        elif self.variant == "additive":
            # Projection + per-neuron sigmoid gate.
            # Gate init to -3  →  sigmoid(-3) ≈ 0.05 (nearly silent at start).
            self.l5_proj = nn.Parameter(
                torch.randn(n_cols, n_apical_target, embed_dim) * (1.0 / math.sqrt(embed_dim))
            )
            self.apical_gate = nn.Parameter(torch.full((n_cols, n_apical_target), -3.0))

        elif self.variant == "multiplicative":
            # Larkum calcium spike.  Weights init near 0  →  tanh ≈ 0  →  identity at start.
            self.l5_proj = nn.Parameter(
                torch.randn(n_cols, n_apical_target, embed_dim) * (0.01 / math.sqrt(embed_dim))
            )

        elif self.variant == "corticortical":
            # L5E of column (k+1) % n_cols  →  L23E of column k.  Fan-in = n_cortical_l5.
            self.l5_to_l23 = nn.Parameter(
                torch.randn(n_cols, n_cortical_l23, n_cortical_l5) * (1.0 / math.sqrt(n_cortical_l5))
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def l5_additive(self, thal_full: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Additive contribution to I_l5e  (skip / additive variants).

        thal_full : [batch, n_cols, embed_dim]
        returns   : [batch, n_cols, n_apical_target]  or  None
        """
        if self.variant == "skip":
            return torch.einsum("bce,coe->bco", thal_full, self.l5_proj)

        elif self.variant == "additive":
            proj = torch.einsum("bce,coe->bco", thal_full, self.l5_proj)
            gate = torch.sigmoid(self.apical_gate).unsqueeze(0)   # [1, n_cols, n]
            return gate * proj

        return None

    def l5_multiplicative(
        self, I_l5e: torch.Tensor, thal_full: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiplicative modulation of I_l5e  (multiplicative variant only).
        Returns I_l5e unchanged for all other variants.

        I_l5e    : [batch, n_cols, n_apical_target]
        thal_full: [batch, n_cols, embed_dim]
        returns  : [batch, n_cols, n_apical_target]
        """
        if self.variant != "multiplicative":
            return I_l5e
        apical = torch.einsum("bce,coe->bco", thal_full, self.l5_proj)
        return I_l5e * (1.0 + torch.tanh(apical))

    def l23_corticortical(self, r_l5e_prev: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Top-down corticortical input to L23E  (corticortical variant only).

        r_l5e_prev : [batch, n_cols, n_cortical_l5]  ← state["r_l5e"] / state["r_e"]
        returns    : [batch, n_cols, n_cortical_l23]  or  None
        """
        if self.variant != "corticortical":
            return None
        # Column k receives L5E from column (k+1) % n_cols — circular top-down
        r_shifted = torch.roll(r_l5e_prev, shifts=-1, dims=1)
        return torch.einsum("bce,coe->bco", r_shifted, self.l5_to_l23)
