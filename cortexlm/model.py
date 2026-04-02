"""CortexLM: top-level model assembling all components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cortexlm.columns import get_batched_columns
from cortexlm.connectivity.builder import ConnectivityBuilder
from cortexlm.hippocampus import get_hippocampus
from cortexlm.readout import ReadoutHead
from cortexlm.thalamus import ThalamicRelayModule
from cortexlm.utils.config import get_col_input_dim


@dataclass
class ModelState:
    """
    Encapsulates all dynamic state of CortexLM at a single timestep.

    column_states: batched state dict  {key: Tensor[batch, n_cols, n]}
    hpc_state:     hippocampal module state dict
    """
    column_states: Dict[str, torch.Tensor]
    hpc_state: Dict[str, torch.Tensor]

    def detach(self) -> ModelState:
        """Return new ModelState with all tensors detached (for truncated BPTT)."""
        return ModelState(
            column_states={k: v.detach() for k, v in self.column_states.items()},
            hpc_state={k: v.detach() for k, v in self.hpc_state.items()},
        )


class CortexLM(nn.Module):
    """
    Neurophysiologically structured language model.

    Components:
      - Token embedding (nn.Embedding) + linear projection to column_input_dim
      - n_columns CorticalColumn instances
      - InterColumnSynapses (connectivity module)
      - HippocampalModule (optional)
      - ReadoutHead (L5 → logits)

    All columns receive the same token embedding as thalamic input (analogous to
    thalamocortical broadcast of sensory input).
    """

    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        ccfg = config["column"]
        self.n_columns = ccfg["n_columns"]
        col_model = ccfg["model"]
        embed_dim  = config["embedding"]["dim"]   # vocab-facing (always)
        col_input_dim = get_col_input_dim(config)  # column-facing (may differ with relay)

        # Determine L5 output size for readout
        if col_model == "layered":
            ls = ccfg.get("layer_sizes", {})
            self.n_l5e = ls.get("l5", {}).get("n_e", 80)
            self.n_l23e = ls.get("l23", {}).get("n_e", 160)
        else:
            self.n_l5e  = ccfg.get("n_e", 80)
            self.n_l23e = self.n_l5e

        # Token embedding — always sized to embed_dim (vocab-facing)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Thalamic relay (optional): rich embedding → per-column col_input_dim projection
        tcfg = config.get("thalamus", {})
        self._thalamus_enabled = tcfg.get("enabled", False)
        if self._thalamus_enabled:
            self.thalamic_relay = ThalamicRelayModule(
                n_cols=self.n_columns,
                embed_dim_large=embed_dim,
                col_input_dim=col_input_dim,
                trn_competition=tcfg.get("trn_competition", False),
                trn_eta_init=tcfg.get("trn_eta_init", 0.1),
                relay_init_scale=tcfg.get("relay_init_scale", 0.02),
            )
        else:
            self.thalamic_relay = None

        # Cortical columns — single batched module processes all columns in parallel
        self.columns = get_batched_columns(config, self.n_columns)

        # Inter-column connectivity
        conn_builder = ConnectivityBuilder(config)
        self.connectivity = conn_builder.build()

        # Hippocampal module
        self.hippocampus = get_hippocampus(config, self.n_columns, self.n_l5e)

        # Hippocampal modulation projection (per-column, mod_dim → 1 scalar added to input)
        hpc_mod_dim = getattr(self.hippocampus, "modulation_dim", 1)
        # HPC modulation projects to col_input_dim (column-facing), not embed_dim
        self.hpc_input_proj = nn.Linear(hpc_mod_dim, col_input_dim, bias=False)
        self._col_input_dim = col_input_dim

        # Readout
        readout_source = config.get("readout", {}).get("source", "l5")
        if readout_source == "l5":
            readout_dim = self.n_columns * self.n_l5e
        elif readout_source == "l23":
            readout_dim = self.n_columns * self.n_l23e
        else:
            readout_dim = self.n_columns * (self.n_l5e + self.n_l23e)

        self.readout = ReadoutHead(readout_dim, vocab_size, config)
        if config.get("readout", {}).get("weight_tying", False):
            self.readout.tie_weights(self.embedding.weight)
        self.readout_source = readout_source
        self.dt = config.get("simulation", {}).get("dt", 1.0)

    # ── State management ────────────────────────────────────────────────────

    def init_state(self, batch_size: int) -> ModelState:
        col_state = self.columns.init_state(batch_size)
        hpc_state = self.hippocampus.init_state(batch_size)
        return ModelState(column_states=col_state, hpc_state=hpc_state)

    # ── Per-token step ───────────────────────────────────────────────────────

    def step(
        self,
        token_ids: torch.Tensor,    # [batch]
        model_state: ModelState,
    ) -> Tuple[torch.Tensor, ModelState]:
        """
        Single token step — all columns processed in one batched forward.

        1. Embed token_ids → [batch, embed_dim]
        2. Compute inter-column signals from previous state (connectivity module)
        3. Batched column forward over all n_cols simultaneously
        4. Hippocampal modulation folded into thalamic increments
        5. Readout L5 → logits

        Returns: (logits [batch, vocab_size], new_state)
        """
        batch = token_ids.shape[0]
        device = token_ids.device
        n_cols = self.n_columns

        # Token embedding — [batch, embed_dim] (always vocab-facing)
        tok_emb = self.embedding(token_ids)

        # Inter-column signals from previous state → [batch, n_cols, col_input_dim]
        prev_lo = self._col_state_to_list(model_state.column_states, batch, device)
        col_increments = self.connectivity(prev_lo)

        thal_inc = torch.stack(
            [col_increments[i]["thalamic_input"] for i in range(n_cols)], dim=1
        )   # [batch, n_cols, col_input_dim]
        l23_fb = torch.stack(
            [col_increments[i]["l23_feedback"] for i in range(n_cols)], dim=1
        )   # [batch, n_cols, n_l23e]

        # Hippocampal modulation: project to col_input_dim and add to thalamic increment
        l5_concat = self._get_l5_concat(model_state.column_states, device, batch)
        hpc_mod, hpc_surprise = self.hippocampus(l5_concat)   # [batch, n_cols, mod_dim]
        thal_inc = thal_inc + self.hpc_input_proj(hpc_mod)
        self._last_hpc_surprise = (
            hpc_surprise.mean().item() if hpc_surprise is not None else None
        )

        # Thalamic relay or legacy broadcast
        if self._thalamus_enabled:
            # Relay: tok_emb → per-column col_input_dim; fold into thal_inc
            thal_base = self.thalamic_relay(tok_emb)  # [batch, n_cols, col_input_dim]
            thal_inc = thal_inc + thal_base
            # Pass zeros for broadcast; apical_signal carries full embedding to L5
            thal_broadcast = tok_emb.new_zeros(batch, self._col_input_dim)
            apical_signal = tok_emb.unsqueeze(1).expand(-1, n_cols, -1)  # [batch, n_cols, embed_dim]
            layer_out, new_col_state = self.columns(
                thal_broadcast, thal_inc, l23_fb, model_state.column_states,
                apical_signal=apical_signal,
            )
        else:
            # Legacy: broadcast tok_emb directly; no apical_signal override
            layer_out, new_col_state = self.columns(
                tok_emb, thal_inc, l23_fb, model_state.column_states
            )
        # layer_out: {"l5_out": [batch, n_cols, n_l5e], ...}

        new_state = ModelState(
            column_states=new_col_state,
            hpc_state=model_state.hpc_state,
        )

        # Cache pre-readout input so e-prop trainer can re-run with grad enabled
        # without recomputing the full recurrent step.
        self._last_readout_input = self._compute_readout_input(layer_out, batch)
        logits = self.readout(self._last_readout_input)
        return logits, new_state

    def _col_state_to_list(
        self,
        col_state: Dict[str, torch.Tensor],
        batch: int,
        device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert batched column state Dict → List[Dict] for connectivity module."""
        outputs = []
        for i in range(self.n_columns):
            lo: Dict[str, torch.Tensor] = {}
            if "r_l23e" in col_state:
                lo["l23_out"] = col_state["r_l23e"][:, i, :]
                lo["l5_out"]  = col_state["r_l5e"][:, i, :]
                lo["l6_out"]  = col_state.get("r_l6e", col_state["r_l23e"])[:, i, :]
            elif "r_e" in col_state:
                act = col_state["r_e"][:, i, :]
                lo["e_out"] = act
                lo["l23_out"] = act
                lo["l5_out"]  = act
            outputs.append(lo)
        return outputs

    def _get_l5_concat(
        self, col_state: Dict[str, torch.Tensor], device, batch: int
    ) -> torch.Tensor:
        """Concatenate L5 E activations across all columns → [batch, n_cols*n_l5e]."""
        if "r_l5e" in col_state:
            return col_state["r_l5e"].reshape(batch, -1)
        elif "r_e" in col_state:
            return col_state["r_e"].reshape(batch, -1)
        return torch.zeros(batch, self.n_columns * self.n_l5e, device=device)

    def _compute_readout_input(
        self, layer_out: Dict[str, torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Return the pre-readout concatenated tensor [batch, readout_dim].

        Mirrors _readout exactly but stops before calling self.readout, so
        callers (e.g. e-prop trainer) can re-attach gradients to this tensor
        and re-run the readout head independently.
        """
        if self.readout_source == "l5":
            acts = layer_out.get("l5_out", layer_out.get("e_out"))
        elif self.readout_source == "l23":
            acts = layer_out.get("l23_out", layer_out.get("e_out"))
        else:
            l5  = layer_out.get("l5_out",  layer_out.get("e_out"))
            l23 = layer_out.get("l23_out", layer_out.get("e_out"))
            acts = torch.cat([l5, l23], dim=-1)
        return acts.reshape(batch, -1)

    def _readout(
        self, layer_out: Dict[str, torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Gather activations [batch, n_cols, n] → reshape → logits."""
        if self.readout_source == "l5":
            acts = layer_out.get("l5_out", layer_out.get("e_out"))
        elif self.readout_source == "l23":
            acts = layer_out.get("l23_out", layer_out.get("e_out"))
        else:
            l5  = layer_out.get("l5_out",  layer_out.get("e_out"))
            l23 = layer_out.get("l23_out", layer_out.get("e_out"))
            acts = torch.cat([l5, l23], dim=-1)   # [batch, n_cols, n_l5e+n_l23e]

        concat = acts.reshape(batch, -1)   # [batch, readout_dim]
        return self.readout(concat)

    # ── Full sequence unroll ─────────────────────────────────────────────────

    def forward(
        self,
        token_sequence: torch.Tensor,     # [batch, seq_len]
        initial_state: Optional[ModelState] = None,
    ) -> Tuple[torch.Tensor, ModelState]:
        """
        Unroll step() over a full sequence.

        Returns:
            all_logits: [batch, seq_len, vocab_size]
            final_state: ModelState at last timestep
        """
        batch, seq_len = token_sequence.shape
        device = token_sequence.device

        state = initial_state if initial_state is not None else self.init_state(batch)

        all_logits = []
        for t in range(seq_len):
            logits, state = self.step(token_sequence[:, t], state)
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)   # [batch, seq_len, vocab_size]
        return all_logits, state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
