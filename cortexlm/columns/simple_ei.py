"""Simple single-layer E/I column — minimal Phase 1 baseline."""

import torch
import torch.nn as nn
from typing import Dict, List

from .base import CorticalColumn
from .apical import ApicalPathway
from cortexlm.neurons import get_neuron_population, BatchedNeuronPop
from cortexlm.synapses.static import StaticSynapse, BatchedStaticSynapse


class SimpleEIColumn(CorticalColumn):
    """
    Single-layer excitatory/inhibitory column.

    Populations: E (n_e neurons), I (n_i neurons).
    Internal connectivity (all static, Dale's Law enforced):
        E → E (recurrent excitation)
        E → I (drive inhibition)
        I → E (feedback inhibition)
        I → I (optional)

    External input is added to E population's drive.

    Outputs: {'e_out': E activations [batch, n_e]}
    Inputs:  {'thalamic_input': [batch, embed_dim] projected to n_e}
    """

    def __init__(self, config: dict):
        super().__init__(config)

        ccfg = config["column"]
        self.n_e = ccfg.get("n_e", 80)
        self.n_i = ccfg.get("n_i", 20)
        embed_dim = config["embedding"]["dim"]

        # Neuron populations
        self.pop_e = get_neuron_population(config, self.n_e)
        self.pop_i = get_neuron_population(config, self.n_i)

        # Input projection: embed_dim → n_e
        self.input_proj = nn.Linear(embed_dim, self.n_e, bias=False)
        # Feedback projection: n_e → n_e (inter-column feedback mapped to same size)
        self.feedback_proj = nn.Linear(self.n_e, self.n_e, bias=False)

        # Internal synapses (static, Dale's Law)
        self.syn_ee = StaticSynapse(self.n_e, 0, self.n_e)
        self.syn_ei = StaticSynapse(self.n_e, 0, self.n_i)
        self.syn_ie = StaticSynapse(0, self.n_i, self.n_e)

        # Dummy tensors for synapses that don't use one population
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        thal = inputs.get("thalamic_input")   # [batch, embed_dim]
        batch_size = thal.shape[0]
        device = thal.device

        device = thal.device
        zeros0_e = torch.zeros(batch_size, 0, device=device)
        zeros0_i = torch.zeros(batch_size, 0, device=device)

        # Project external input to E population size
        I_ext = self.input_proj(thal)  # [batch, n_e]
        # Add inter-column feedback if provided
        l23_fb = inputs.get("l23_feedback")
        if l23_fb is not None:
            I_ext = I_ext + self.feedback_proj(l23_fb)

        # Previous activations
        r_e = state["r_e"]  # [batch, n_e]
        r_i = state["r_i"]  # [batch, n_i]

        # StaticSynapse(n_pre_e, n_pre_i=0, n_post) — pass zero-dim tensor for missing pop
        I_ee = self.syn_ee.forward(r_e, zeros0_i)   # [batch, n_e]
        I_ei = self.syn_ei.forward(r_e, zeros0_i)   # [batch, n_i]
        I_ie = self.syn_ie.forward(zeros0_e, r_i)   # [batch, n_e]

        # Total inputs
        I_e_total = I_ext + I_ee + I_ie   # [batch, n_e]
        I_i_total = I_ei                  # [batch, n_i]

        # Update neurons
        state_e = {k: state[f"e_{k}"] for k in self.pop_e.state_keys()}
        state_i = {k: state[f"i_{k}"] for k in self.pop_i.state_keys()}

        r_e_new, new_state_e = self.pop_e(I_e_total, state_e)
        r_i_new, new_state_i = self.pop_i(I_i_total, state_i)

        new_state = {"r_e": r_e_new, "r_i": r_i_new}
        for k, v in new_state_e.items():
            new_state[f"e_{k}"] = v
        for k, v in new_state_i.items():
            new_state[f"i_{k}"] = v

        layer_outputs = {"e_out": r_e_new}
        return layer_outputs, new_state

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        state = {
            "r_e": torch.zeros(batch_size, self.n_e, device=device),
            "r_i": torch.zeros(batch_size, self.n_i, device=device),
        }
        dummy_state_e = self.pop_e.init_state(batch_size)
        dummy_state_i = self.pop_i.init_state(batch_size)
        for k, v in dummy_state_e.items():
            state[f"e_{k}"] = v
        for k, v in dummy_state_i.items():
            state[f"i_{k}"] = v
        return state

    def input_keys(self) -> List[str]:
        return ["thalamic_input", "l23_feedback"]

    def output_keys(self) -> List[str]:
        return ["e_out"]


class BatchedSimpleEIColumns(nn.Module):
    """
    All simple_ei columns fused into a single batched module.

    State tensors have shape [batch, n_cols, n] instead of per-column [batch, n].
    Weight tensors have shape [n_cols, n_post, n_pre] — all columns processed in
    one einsum instead of a Python loop.
    """

    def __init__(self, config: dict, n_cols: int):
        super().__init__()
        ccfg = config["column"]
        embed_dim = config["embedding"]["dim"]

        self.n_cols = n_cols
        self.n_e = ccfg.get("n_e", 80)
        self.n_i = ccfg.get("n_i", 20)

        # Batched neuron populations — independent tau draw per column
        self.pop_e = BatchedNeuronPop(config, self.n_e, n_cols)
        self.pop_i = BatchedNeuronPop(config, self.n_i, n_cols)

        # Batched internal synapses [n_cols, n_post, n_pre]
        self.syn_ee = BatchedStaticSynapse(n_cols, self.n_e, 0, self.n_e)
        self.syn_ei = BatchedStaticSynapse(n_cols, self.n_e, 0, self.n_i)
        self.syn_ie = BatchedStaticSynapse(n_cols, 0, self.n_i, self.n_e)

        # Batched input & feedback projections
        # input_proj:   embed_dim → n_e,  per column  [n_cols, n_e, embed_dim]
        # feedback_proj: n_e      → n_e,  per column  [n_cols, n_e, n_e]
        import math as _math
        self.input_proj    = nn.Parameter(torch.randn(n_cols, self.n_e, embed_dim)  * (1.0 / _math.sqrt(embed_dim)))
        self.feedback_proj = nn.Parameter(torch.randn(n_cols, self.n_e, self.n_e)  * (1.0 / _math.sqrt(self.n_e)))

        # ── Optional apical pathway ───────────────────────────────────────────
        apical_mode = config["column"].get("apical_pathway", "none")
        if apical_mode != "none":
            # For simple_ei: E population serves as both L5 and L23 analogue
            self.apical = ApicalPathway(
                config, n_cols, embed_dim,
                n_apical_target=self.n_e,
                n_cortical_l23=self.n_e,
                n_cortical_l5=self.n_e,
            )
        else:
            self.apical = None

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        thal: torch.Tensor,              # [batch, embed_dim]
        thal_increments: torch.Tensor,   # [batch, n_cols, embed_dim]
        l23_fb: torch.Tensor,            # [batch, n_cols, n_e]
        state: Dict[str, torch.Tensor],  # batched state dict
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, n_cols = thal_increments.shape[:2]

        # Thalamic input: broadcast base embedding + per-column increments
        # [batch, embed_dim] → [batch, n_cols, embed_dim]
        thal_full = thal.unsqueeze(1).expand(-1, n_cols, -1) + thal_increments

        # Input projection: [batch, n_cols, embed_dim] × [n_cols, n_e, embed_dim]ᵀ
        I_ext = torch.einsum("bce,coe->bco", thal_full, self.input_proj)  # [batch, n_cols, n_e]
        I_fb  = torch.einsum("bce,coe->bco", l23_fb,    self.feedback_proj)
        I_ext = I_ext + I_fb

        r_e = state["r_e"]  # [batch, n_cols, n_e]
        r_i = state["r_i"]  # [batch, n_cols, n_i]

        z0e = r_e.new_zeros(batch, n_cols, 0)
        z0i = r_i.new_zeros(batch, n_cols, 0)

        I_ee = self.syn_ee(r_e, z0i)   # [batch, n_cols, n_e]
        I_ei = self.syn_ei(r_e, z0i)   # [batch, n_cols, n_i]
        I_ie = self.syn_ie(z0e, r_i)   # [batch, n_cols, n_e]

        I_e_total = I_ext + I_ee + I_ie

        # Apical pathway (all variants feed into E population for simple_ei)
        if self.apical is not None:
            apical_add = self.apical.l5_additive(thal_full)
            if apical_add is not None:
                I_e_total = I_e_total + apical_add
            I_e_total = self.apical.l5_multiplicative(I_e_total, thal_full)
            cortico = self.apical.l23_corticortical(state["r_e"])
            if cortico is not None:
                I_e_total = I_e_total + cortico

        I_i_total = I_ei

        state_e = {k: state[f"e_{k}"] for k in self.pop_e.state_keys()}
        state_i = {k: state[f"i_{k}"] for k in self.pop_i.state_keys()}

        r_e_new, new_state_e = self.pop_e(I_e_total, state_e)
        r_i_new, new_state_i = self.pop_i(I_i_total, state_i)

        new_state = {"r_e": r_e_new, "r_i": r_i_new}
        for k, v in new_state_e.items():
            new_state[f"e_{k}"] = v
        for k, v in new_state_i.items():
            new_state[f"i_{k}"] = v

        layer_outputs = {"e_out": r_e_new, "l23_out": r_e_new, "l5_out": r_e_new}
        return layer_outputs, new_state

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        n_cols, n_e, n_i = self.n_cols, self.n_e, self.n_i
        state = {
            "r_e": torch.zeros(batch_size, n_cols, n_e, device=device),
            "r_i": torch.zeros(batch_size, n_cols, n_i, device=device),
        }
        for k, v in self.pop_e.init_state(batch_size).items():
            state[f"e_{k}"] = v
        for k, v in self.pop_i.init_state(batch_size).items():
            state[f"i_{k}"] = v
        return state
