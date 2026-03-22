"""Assembles full inter-column connectivity from config."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .local import gaussian_connectivity_mask
from .small_world import small_world_connectivity_mask
from cortexlm.synapses.static import StaticSynapse
from cortexlm.synapses.stp import STPSynapse


class InterColumnSynapses(nn.Module):
    """
    Holds all inter-column synapse modules and implements the routing step.

    Laminar routing (biologically grounded):
      - Feedforward (lower → higher column index): l23_out → thalamic_input of target
      - Feedback (higher → lower column index):   l5_out  → l23_feedback of source

    For simple_ei columns: e_out → thalamic_input.
    """

    def __init__(
        self,
        config: dict,
        mask: torch.Tensor,              # [n_cols, n_cols] bool
        n_l23e_per_col: int,
        n_l5e_per_col: int,
        n_l4e_per_col: int,              # target dim for thalamic input projection
        embed_dim: int,
        use_stp: bool,
        is_simple_ei: bool,
    ):
        super().__init__()
        self.n_cols = mask.shape[0]
        self.mask = mask
        self.is_simple_ei = is_simple_ei
        self.use_stp = use_stp

        self.n_l23e = n_l23e_per_col
        self.n_l5e  = n_l5e_per_col
        self.embed_dim = embed_dim

        # Build synapse modules for each connected pair
        # Key: (src, tgt, direction) where direction in {'ff', 'fb'}
        self.synapses = nn.ModuleDict()

        for src in range(self.n_cols):
            for tgt in range(self.n_cols):
                if not mask[src, tgt]:
                    continue

                if src < tgt:
                    # Feedforward: src l23 → tgt l4 input (projected to embed_dim)
                    n_pre_e = n_l23e_per_col if not is_simple_ei else n_l23e_per_col
                    n_post  = embed_dim
                    key = f"ff_{src}_{tgt}"
                elif src > tgt:
                    # Feedback: src l5 → tgt l23_feedback
                    n_pre_e = n_l5e_per_col
                    n_post  = n_l23e_per_col
                    key = f"fb_{src}_{tgt}"
                else:
                    continue  # no self-connections

                if use_stp:
                    syn = STPSynapse(n_pre_e, 0, n_post, config)
                else:
                    syn = StaticSynapse(n_pre_e, 0, n_post)
                self.synapses[key] = syn

    def forward(
        self,
        layer_outputs: List[Dict[str, torch.Tensor]],  # one dict per column
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute inter-column signals and accumulate into target column inputs.

        Returns: list of input increment dicts for each column.
        Shape of each value depends on column type.
        """
        batch = next(
            v for lo in layer_outputs for v in lo.values()
        ).shape[0]
        device = next(
            v for lo in layer_outputs for v in lo.values()
        ).device

        # Initialize incremental inputs for each column
        col_inputs: List[Dict[str, torch.Tensor]] = [
            {
                "thalamic_input": torch.zeros(batch, self.embed_dim, device=device),
                "l23_feedback":   torch.zeros(batch, self.n_l23e,    device=device),
            }
            for _ in range(self.n_cols)
        ]

        dummy_zeros_i = torch.zeros(batch, 0, device=device)

        for src in range(self.n_cols):
            for tgt in range(self.n_cols):
                if not self.mask[src, tgt]:
                    continue

                if src < tgt:
                    # Feedforward
                    key = f"ff_{src}_{tgt}"
                    syn = self.synapses[key]
                    src_act = layer_outputs[src].get(
                        "l23_out", layer_outputs[src].get("e_out")
                    )  # [batch, n_l23e]
                    if src_act is None:
                        continue
                    if self.use_stp:
                        # STPSynapse.forward needs state — for now we use StaticSynapse-like path
                        # (STP state is managed in model.py; builder just does static projection here)
                        I = src_act @ syn.W_e.t()
                    else:
                        I = syn(src_act, dummy_zeros_i[:, :0] if dummy_zeros_i.shape[1] == 0
                                else dummy_zeros_i)
                    col_inputs[tgt]["thalamic_input"] = col_inputs[tgt]["thalamic_input"] + I

                elif src > tgt:
                    # Feedback
                    key = f"fb_{src}_{tgt}"
                    syn = self.synapses[key]
                    src_act = layer_outputs[src].get("l5_out")
                    if src_act is None:
                        continue
                    if self.use_stp:
                        I = src_act @ syn.W_e.t()
                    else:
                        I = syn(src_act, torch.zeros(batch, 0, device=device))
                    col_inputs[tgt]["l23_feedback"] = col_inputs[tgt]["l23_feedback"] + I

        return col_inputs


class ConnectivityBuilder:
    """Builds the inter-column connectivity given config and column specs."""

    def __init__(self, config: dict):
        self.config = config
        self.ccfg = config["column"]
        self.conn_cfg = config["connectivity"]

        n_cols = self.ccfg["n_columns"]
        conn_type = self.conn_cfg["type"]

        if conn_type == "gaussian_1d":
            self.mask = gaussian_connectivity_mask(
                n_cols,
                self.conn_cfg.get("p_max", 0.7),
                self.conn_cfg.get("sigma", 3.0),
            )
        elif conn_type == "small_world":
            self.mask = small_world_connectivity_mask(
                n_cols,
                self.conn_cfg.get("k", 4),
                self.conn_cfg.get("beta", 0.1),
            )
        elif conn_type == "random_sparse":
            p = self.conn_cfg.get("p_max", 0.3)
            self.mask = torch.bernoulli(
                torch.full((n_cols, n_cols), p)
            ).bool()
            self.mask.fill_diagonal_(False)
        else:
            raise ValueError(f"Unknown connectivity type: {conn_type}")

    def build(self) -> InterColumnSynapses:
        col_model = self.ccfg["model"]
        ls = self.ccfg.get("layer_sizes", {})
        embed_dim = self.config["embedding"]["dim"]

        if col_model == "layered":
            n_l23e = ls.get("l23", {}).get("n_e", 160)
            n_l5e  = ls.get("l5",  {}).get("n_e", 80)
            n_l4e  = ls.get("l4",  {}).get("n_e", 80)
            is_simple = False
        else:
            # simple_ei: use e_out as both l23 and l5 equivalent
            n_l23e = self.ccfg.get("n_e", 80)
            n_l5e  = n_l23e
            n_l4e  = n_l23e
            is_simple = True

        use_stp = self.config["synapse"].get("inter_column_stp", False)

        return InterColumnSynapses(
            config=self.config,
            mask=self.mask,
            n_l23e_per_col=n_l23e,
            n_l5e_per_col=n_l5e,
            n_l4e_per_col=n_l4e,
            embed_dim=embed_dim,
            use_stp=use_stp,
            is_simple_ei=is_simple,
        )
