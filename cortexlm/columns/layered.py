"""Six-layer cortical column: L4, L2/3, L5, L6 with biologically grounded wiring."""

import torch
import torch.nn as nn
from typing import Dict, List

from .base import CorticalColumn
from .apical import ApicalPathway
from cortexlm.neurons import get_neuron_population, BatchedNeuronPop
from cortexlm.synapses.static import StaticSynapse, BatchedStaticSynapse


def _make_layer(config, n_e, n_i):
    """Create (E pop, I pop) for a layer."""
    pop_e = get_neuron_population(config, n_e)
    pop_i = get_neuron_population(config, n_i)
    return pop_e, pop_i


class LayeredColumn(CorticalColumn):
    """
    Six-layer cortical column (L4, L2/3, L5, L6) with biologically grounded
    inter-layer connectivity and Dale's Law throughout.

    Inter-layer wiring (excitatory drive from E populations only):
        Thalamic input → L4_E, L4_I
        L4_E  → L2/3_E (strong), L2/3_I
        L4_E  → L5_E (moderate)
        L2/3_E → L5_E, L5_I
        L2/3_E → L6_E
        L5_E  → L6_E, L6_I
        L6_E  → L4_E, L4_I (feedback, modulatory)

    Each layer also has recurrent E→E and E→I, I→E connections.

    Inputs:
        'thalamic_input': [batch, embed_dim] — projected to L4
        'l23_feedback':   [batch, n_l23_e] — from inter-column horizontal connections (optional)

    Outputs:
        'l23_out': [batch, n_l23_e]   — inter-column horizontal connections
        'l5_out':  [batch, n_l5_e]    — readout signal
        'l6_out':  [batch, n_l6_e]    — thalamic feedback (future use)
    """

    def __init__(self, config: dict):
        super().__init__(config)

        ccfg = config["column"]
        ls = ccfg.get("layer_sizes", {})
        embed_dim = config["embedding"]["dim"]

        self.n_l4e  = ls.get("l4",  {}).get("n_e", 80)
        self.n_l4i  = ls.get("l4",  {}).get("n_i", 20)
        self.n_l23e = ls.get("l23", {}).get("n_e", 160)
        self.n_l23i = ls.get("l23", {}).get("n_i", 40)
        self.n_l5e  = ls.get("l5",  {}).get("n_e", 80)
        self.n_l5i  = ls.get("l5",  {}).get("n_i", 20)
        self.n_l6e  = ls.get("l6",  {}).get("n_e", 80)
        self.n_l6i  = ls.get("l6",  {}).get("n_i", 20)

        # Neuron populations
        self.l4_e, self.l4_i     = _make_layer(config, self.n_l4e, self.n_l4i)
        self.l23_e, self.l23_i   = _make_layer(config, self.n_l23e, self.n_l23i)
        self.l5_e, self.l5_i     = _make_layer(config, self.n_l5e, self.n_l5i)
        self.l6_e, self.l6_i     = _make_layer(config, self.n_l6e, self.n_l6i)

        # Input projection: embed_dim → L4
        self.thal_proj_e = nn.Linear(embed_dim, self.n_l4e, bias=False)
        self.thal_proj_i = nn.Linear(embed_dim, self.n_l4i, bias=False)

        # L2/3 feedback projection (inter-column input)
        self.fb_proj = nn.Linear(self.n_l23e, self.n_l23e, bias=False)

        # ── Within-layer recurrent synapses ──────────────────────────────
        # L4
        self.syn_l4_ee  = StaticSynapse(self.n_l4e, 0,        self.n_l4e)
        self.syn_l4_ei  = StaticSynapse(self.n_l4e, 0,        self.n_l4i)
        self.syn_l4_ie  = StaticSynapse(0,          self.n_l4i, self.n_l4e)
        # L2/3
        self.syn_l23_ee = StaticSynapse(self.n_l23e, 0,         self.n_l23e)
        self.syn_l23_ei = StaticSynapse(self.n_l23e, 0,         self.n_l23i)
        self.syn_l23_ie = StaticSynapse(0,           self.n_l23i, self.n_l23e)
        # L5
        self.syn_l5_ee  = StaticSynapse(self.n_l5e, 0,         self.n_l5e)
        self.syn_l5_ei  = StaticSynapse(self.n_l5e, 0,         self.n_l5i)
        self.syn_l5_ie  = StaticSynapse(0,          self.n_l5i, self.n_l5e)
        # L6
        self.syn_l6_ee  = StaticSynapse(self.n_l6e, 0,         self.n_l6e)
        self.syn_l6_ei  = StaticSynapse(self.n_l6e, 0,         self.n_l6i)
        self.syn_l6_ie  = StaticSynapse(0,          self.n_l6i, self.n_l6e)

        # ── Inter-layer feedforward synapses ─────────────────────────────
        # L4_E → L2/3_E, L2/3_I (strong)
        self.syn_l4e_l23e = StaticSynapse(self.n_l4e, 0, self.n_l23e)
        self.syn_l4e_l23i = StaticSynapse(self.n_l4e, 0, self.n_l23i)
        # L4_E → L5_E (moderate)
        self.syn_l4e_l5e  = StaticSynapse(self.n_l4e, 0, self.n_l5e)
        # L2/3_E → L5_E, L5_I
        self.syn_l23e_l5e = StaticSynapse(self.n_l23e, 0, self.n_l5e)
        self.syn_l23e_l5i = StaticSynapse(self.n_l23e, 0, self.n_l5i)
        # L2/3_E → L6_E
        self.syn_l23e_l6e = StaticSynapse(self.n_l23e, 0, self.n_l6e)
        # L5_E → L6_E, L6_I
        self.syn_l5e_l6e  = StaticSynapse(self.n_l5e, 0, self.n_l6e)
        self.syn_l5e_l6i  = StaticSynapse(self.n_l5e, 0, self.n_l6i)
        # L6_E → L4_E, L4_I (feedback, modulatory)
        self.syn_l6e_l4e  = StaticSynapse(self.n_l6e, 0, self.n_l4e)
        self.syn_l6e_l4i  = StaticSynapse(self.n_l6e, 0, self.n_l4i)

        self.disinhibition = ccfg.get("disinhibition", False)

    # ── Helper: zeros ──────────────────────────────────────────────────────

    def _z(self, n, batch, device):
        return torch.zeros(batch, n, device=device)

    # ── Layer update ───────────────────────────────────────────────────────

    def _update_layer(self, pop_e, pop_i, I_e, I_i, state, prefix):
        state_e = {k: state[f"{prefix}_e_{k}"] for k in pop_e.state_keys()}
        state_i = {k: state[f"{prefix}_i_{k}"] for k in pop_i.state_keys()}
        r_e_new, new_se = pop_e(I_e, state_e)
        r_i_new, new_si = pop_i(I_i, state_i)
        out = {}
        for k, v in new_se.items():
            out[f"{prefix}_e_{k}"] = v
        for k, v in new_si.items():
            out[f"{prefix}_i_{k}"] = v
        return r_e_new, r_i_new, out

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        thal = inputs["thalamic_input"]             # [batch, embed_dim]
        l23_fb = inputs.get("l23_feedback", None)   # [batch, n_l23e] or None
        batch = thal.shape[0]
        device = thal.device

        # Previous activations
        r_l4e  = state["r_l4e"];  r_l4i  = state["r_l4i"]
        r_l23e = state["r_l23e"]; r_l23i = state["r_l23i"]
        r_l5e  = state["r_l5e"];  r_l5i  = state["r_l5i"]
        r_l6e  = state["r_l6e"];  r_l6i  = state["r_l6i"]

        z = self._z
        # shorthand for zero-inhibitory-input synapse forward
        def fe(syn, r):
            return syn(r, z(syn.n_pre_i, batch, device))
        def fi(syn, r):
            return syn(z(syn.n_pre_e, batch, device), r)

        # ── L4 ────────────────────────────────────────────────────────────
        I_l4e = (self.thal_proj_e(thal)
                 + fe(self.syn_l4_ee,  r_l4e)
                 + fi(self.syn_l4_ie,  r_l4i)
                 + fe(self.syn_l6e_l4e, r_l6e))
        I_l4i = (self.thal_proj_i(thal)
                 + fe(self.syn_l4_ei,  r_l4e)
                 + fe(self.syn_l6e_l4i, r_l6e))

        r_l4e_new, r_l4i_new, ns_l4 = self._update_layer(
            self.l4_e, self.l4_i, I_l4e, I_l4i, state, "l4")

        # ── L2/3 ──────────────────────────────────────────────────────────
        I_l23e = (fe(self.syn_l4e_l23e,  r_l4e_new)
                  + fe(self.syn_l23_ee,   r_l23e)
                  + fi(self.syn_l23_ie,   r_l23i))
        I_l23i = (fe(self.syn_l4e_l23i,  r_l4e_new)
                  + fe(self.syn_l23_ei,   r_l23e))

        # Inter-column feedback into L2/3
        if l23_fb is not None:
            I_l23e = I_l23e + self.fb_proj(l23_fb)

        r_l23e_new, r_l23i_new, ns_l23 = self._update_layer(
            self.l23_e, self.l23_i, I_l23e, I_l23i, state, "l23")

        # ── L5 ────────────────────────────────────────────────────────────
        I_l5e = (fe(self.syn_l4e_l5e,   r_l4e_new)
                 + fe(self.syn_l23e_l5e, r_l23e_new)
                 + fe(self.syn_l5_ee,    r_l5e)
                 + fi(self.syn_l5_ie,    r_l5i))
        I_l5i = (fe(self.syn_l23e_l5i,  r_l23e_new)
                 + fe(self.syn_l5_ei,    r_l5e))

        r_l5e_new, r_l5i_new, ns_l5 = self._update_layer(
            self.l5_e, self.l5_i, I_l5e, I_l5i, state, "l5")

        # ── L6 ────────────────────────────────────────────────────────────
        I_l6e = (fe(self.syn_l23e_l6e,  r_l23e_new)
                 + fe(self.syn_l5e_l6e,  r_l5e_new)
                 + fe(self.syn_l6_ee,    r_l6e)
                 + fi(self.syn_l6_ie,    r_l6i))
        I_l6i = (fe(self.syn_l5e_l6i,   r_l5e_new)
                 + fe(self.syn_l6_ei,    r_l6e))

        r_l6e_new, r_l6i_new, ns_l6 = self._update_layer(
            self.l6_e, self.l6_i, I_l6e, I_l6i, state, "l6")

        # Assemble new state
        new_state = {
            "r_l4e": r_l4e_new,  "r_l4i": r_l4i_new,
            "r_l23e": r_l23e_new, "r_l23i": r_l23i_new,
            "r_l5e": r_l5e_new,  "r_l5i": r_l5i_new,
            "r_l6e": r_l6e_new,  "r_l6i": r_l6i_new,
        }
        new_state.update(ns_l4)
        new_state.update(ns_l23)
        new_state.update(ns_l5)
        new_state.update(ns_l6)

        layer_outputs = {
            "l23_out": r_l23e_new,
            "l5_out":  r_l5e_new,
            "l6_out":  r_l6e_new,
        }
        return layer_outputs, new_state

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        state = {}
        for prefix, pop_e, pop_i, n_e, n_i in [
            ("l4",  self.l4_e,  self.l4_i,  self.n_l4e,  self.n_l4i),
            ("l23", self.l23_e, self.l23_i, self.n_l23e, self.n_l23i),
            ("l5",  self.l5_e,  self.l5_i,  self.n_l5e,  self.n_l5i),
            ("l6",  self.l6_e,  self.l6_i,  self.n_l6e,  self.n_l6i),
        ]:
            state[f"r_{prefix}e"] = torch.zeros(batch_size, n_e, device=device)
            state[f"r_{prefix}i"] = torch.zeros(batch_size, n_i, device=device)
            for k, v in pop_e.init_state(batch_size).items():
                state[f"{prefix}_e_{k}"] = v
            for k, v in pop_i.init_state(batch_size).items():
                state[f"{prefix}_i_{k}"] = v
        return state

    def input_keys(self) -> List[str]:
        return ["thalamic_input", "l23_feedback"]

    def output_keys(self) -> List[str]:
        return ["l23_out", "l5_out", "l6_out"]


class BatchedLayeredColumns(nn.Module):
    """
    All layered columns fused into a single batched module.

    Processes all n_cols columns in parallel via einsum instead of a Python loop.
    State tensors: [batch, n_cols, n].  Weight tensors: [n_cols, n_post, n_pre].

    Neuron populations are shared across columns (same tau distribution), which
    preserves per-neuron heterogeneity at modest loss of per-column diversity.
    """

    def __init__(self, config: dict, n_cols: int):
        super().__init__()
        ccfg = config["column"]
        ls = ccfg.get("layer_sizes", {})
        embed_dim = config["embedding"]["dim"]
        self.n_cols = n_cols

        self.n_l4e  = ls.get("l4",  {}).get("n_e", 80)
        self.n_l4i  = ls.get("l4",  {}).get("n_i", 20)
        self.n_l23e = ls.get("l23", {}).get("n_e", 160)
        self.n_l23i = ls.get("l23", {}).get("n_i", 40)
        self.n_l5e  = ls.get("l5",  {}).get("n_e", 80)
        self.n_l5i  = ls.get("l5",  {}).get("n_i", 20)
        self.n_l6e  = ls.get("l6",  {}).get("n_e", 80)
        self.n_l6i  = ls.get("l6",  {}).get("n_i", 20)

        # Independent neuron populations per column — each column gets its own tau draw
        def _pop(n): return BatchedNeuronPop(config, n, n_cols)
        self.l4_e,  self.l4_i  = _pop(self.n_l4e),  _pop(self.n_l4i)
        self.l23_e, self.l23_i = _pop(self.n_l23e), _pop(self.n_l23i)
        self.l5_e,  self.l5_i  = _pop(self.n_l5e),  _pop(self.n_l5i)
        self.l6_e,  self.l6_i  = _pop(self.n_l6e),  _pop(self.n_l6i)

        # Batched thalamic projections: [n_cols, n_l4e/i, embed_dim]
        # Fan-in = embed_dim; use 1/sqrt(embed_dim) so input variance ≈ 1 at init
        import math as _math
        self.thal_proj_e_w = nn.Parameter(torch.randn(n_cols, self.n_l4e, embed_dim) * (1.0 / _math.sqrt(embed_dim)))
        self.thal_proj_i_w = nn.Parameter(torch.randn(n_cols, self.n_l4i, embed_dim) * (1.0 / _math.sqrt(embed_dim)))

        # Batched L2/3 feedback projection: [n_cols, n_l23e, n_l23e]
        # Fan-in = n_l23e
        self.fb_proj_w = nn.Parameter(torch.randn(n_cols, self.n_l23e, self.n_l23e) * (1.0 / _math.sqrt(self.n_l23e)))

        # ── Within-layer recurrent synapses ──────────────────────────────────
        self.syn_l4_ee   = BatchedStaticSynapse(n_cols, self.n_l4e,  0,          self.n_l4e)
        self.syn_l4_ei   = BatchedStaticSynapse(n_cols, self.n_l4e,  0,          self.n_l4i)
        self.syn_l4_ie   = BatchedStaticSynapse(n_cols, 0,           self.n_l4i, self.n_l4e)
        self.syn_l23_ee  = BatchedStaticSynapse(n_cols, self.n_l23e, 0,          self.n_l23e)
        self.syn_l23_ei  = BatchedStaticSynapse(n_cols, self.n_l23e, 0,          self.n_l23i)
        self.syn_l23_ie  = BatchedStaticSynapse(n_cols, 0,           self.n_l23i, self.n_l23e)
        self.syn_l5_ee   = BatchedStaticSynapse(n_cols, self.n_l5e,  0,          self.n_l5e)
        self.syn_l5_ei   = BatchedStaticSynapse(n_cols, self.n_l5e,  0,          self.n_l5i)
        self.syn_l5_ie   = BatchedStaticSynapse(n_cols, 0,           self.n_l5i, self.n_l5e)
        self.syn_l6_ee   = BatchedStaticSynapse(n_cols, self.n_l6e,  0,          self.n_l6e)
        self.syn_l6_ei   = BatchedStaticSynapse(n_cols, self.n_l6e,  0,          self.n_l6i)
        self.syn_l6_ie   = BatchedStaticSynapse(n_cols, 0,           self.n_l6i, self.n_l6e)

        # ── Inter-layer feedforward synapses ─────────────────────────────────
        self.syn_l4e_l23e  = BatchedStaticSynapse(n_cols, self.n_l4e,  0, self.n_l23e)
        self.syn_l4e_l23i  = BatchedStaticSynapse(n_cols, self.n_l4e,  0, self.n_l23i)
        self.syn_l4e_l5e   = BatchedStaticSynapse(n_cols, self.n_l4e,  0, self.n_l5e)
        self.syn_l23e_l5e  = BatchedStaticSynapse(n_cols, self.n_l23e, 0, self.n_l5e)
        self.syn_l23e_l5i  = BatchedStaticSynapse(n_cols, self.n_l23e, 0, self.n_l5i)
        self.syn_l23e_l6e  = BatchedStaticSynapse(n_cols, self.n_l23e, 0, self.n_l6e)
        self.syn_l5e_l6e   = BatchedStaticSynapse(n_cols, self.n_l5e,  0, self.n_l6e)
        self.syn_l5e_l6i   = BatchedStaticSynapse(n_cols, self.n_l5e,  0, self.n_l6i)
        self.syn_l6e_l4e   = BatchedStaticSynapse(n_cols, self.n_l6e,  0, self.n_l4e)
        self.syn_l6e_l4i   = BatchedStaticSynapse(n_cols, self.n_l6e,  0, self.n_l4i)

        # ── Optional apical pathway ───────────────────────────────────────────
        apical_mode = config["column"].get("apical_pathway", "none")
        if apical_mode != "none":
            self.apical = ApicalPathway(
                config, n_cols, embed_dim,
                n_apical_target=self.n_l5e,
                n_cortical_l23=self.n_l23e,
                n_cortical_l5=self.n_l5e,
            )
        else:
            self.apical = None

        # ── Optional VIP disinhibition circuit (VIP→SST→PC) ──────────────────
        # VIP interneurons receive excitatory drive from E neurons in their layer,
        # and inhibit the SST (I) population, thereby disinhibiting pyramidal cells.
        # n_vip ≈ n_i // 2 per layer (VIP is a minority interneuron subtype).
        self.disinhibition = ccfg.get("disinhibition", False)
        if self.disinhibition:
            self.n_l4vip  = max(1, self.n_l4i  // 2)
            self.n_l23vip = max(1, self.n_l23i // 2)
            self.n_l5vip  = max(1, self.n_l5i  // 2)
            self.n_l6vip  = max(1, self.n_l6i  // 2)

            # VIP populations (one per layer)
            self.l4_vip  = _pop(self.n_l4vip)
            self.l23_vip = _pop(self.n_l23vip)
            self.l5_vip  = _pop(self.n_l5vip)
            self.l6_vip  = _pop(self.n_l6vip)

            # E→VIP: excitatory drive to VIP from local E population
            self.syn_l4_e_vip  = BatchedStaticSynapse(n_cols, self.n_l4e,  0, self.n_l4vip)
            self.syn_l23_e_vip = BatchedStaticSynapse(n_cols, self.n_l23e, 0, self.n_l23vip)
            self.syn_l5_e_vip  = BatchedStaticSynapse(n_cols, self.n_l5e,  0, self.n_l5vip)
            self.syn_l6_e_vip  = BatchedStaticSynapse(n_cols, self.n_l6e,  0, self.n_l6vip)

            # VIP→SST: VIP inhibits local SST (I) population (inhibitory synapse)
            self.syn_l4_vip_sst  = BatchedStaticSynapse(n_cols, 0, self.n_l4vip,  self.n_l4i)
            self.syn_l23_vip_sst = BatchedStaticSynapse(n_cols, 0, self.n_l23vip, self.n_l23i)
            self.syn_l5_vip_sst  = BatchedStaticSynapse(n_cols, 0, self.n_l5vip,  self.n_l5i)
            self.syn_l6_vip_sst  = BatchedStaticSynapse(n_cols, 0, self.n_l6vip,  self.n_l6i)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _update_vip(self, pop_vip, I_vip, state, prefix):
        """Update a VIP interneuron population.  Returns (r_vip_new, out_state)."""
        state_vip = {k: state[f"{prefix}_vip_{k}"] for k in pop_vip.state_keys()}
        r_vip_new, ns_vip = pop_vip(I_vip, state_vip)
        out_state = {f"{prefix}_vip_{k}": v for k, v in ns_vip.items()}
        return r_vip_new, out_state

    def _update_layer(self, pop_e, pop_i, I_e, I_i, state, prefix):
        """Neuron update for one layer.  I_e/I_i: [batch, n_cols, n]."""
        state_e = {k: state[f"{prefix}_e_{k}"] for k in pop_e.state_keys()}
        state_i = {k: state[f"{prefix}_i_{k}"] for k in pop_i.state_keys()}

        r_e_new, ns_e = pop_e(I_e, state_e)
        r_i_new, ns_i = pop_i(I_i, state_i)

        out_state = {f"{prefix}_e_{k}": v for k, v in ns_e.items()}
        out_state.update({f"{prefix}_i_{k}": v for k, v in ns_i.items()})
        return r_e_new, r_i_new, out_state

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        thal: torch.Tensor,             # [batch, embed_dim]
        thal_increments: torch.Tensor,  # [batch, n_cols, embed_dim]
        l23_fb: torch.Tensor,           # [batch, n_cols, n_l23e]
        state: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, n_cols = thal_increments.shape[:2]

        # Thalamic input: broadcast base embedding + per-column increments → [batch, n_cols, embed_dim]
        thal_full = thal.unsqueeze(1).expand(-1, n_cols, -1) + thal_increments

        # Thalamic projections to L4
        I_thal_e = torch.einsum("bce,coe->bco", thal_full, self.thal_proj_e_w)
        I_thal_i = torch.einsum("bce,coe->bco", thal_full, self.thal_proj_i_w)

        r_l4e  = state["r_l4e"];  r_l4i  = state["r_l4i"]
        r_l23e = state["r_l23e"]; r_l23i = state["r_l23i"]
        r_l5e  = state["r_l5e"];  r_l5i  = state["r_l5i"]
        r_l6e  = state["r_l6e"];  r_l6i  = state["r_l6i"]

        z0 = r_l4e.new_zeros  # shorthand

        # ── L4 ───────────────────────────────────────────────────────────────
        I_l4e = (I_thal_e
                 + self.syn_l4_ee(r_l4e,  z0(batch, n_cols, 0))
                 + self.syn_l4_ie(z0(batch, n_cols, 0), r_l4i)
                 + self.syn_l6e_l4e(r_l6e, z0(batch, n_cols, 0)))
        I_l4i = (I_thal_i
                 + self.syn_l4_ei(r_l4e,  z0(batch, n_cols, 0))
                 + self.syn_l6e_l4i(r_l6e, z0(batch, n_cols, 0)))

        if self.disinhibition:
            # VIP driven by prev-step L4 E; fires this step to inhibit L4 SST
            r_l4vip = state["r_l4vip"]
            I_l4_vip = self.syn_l4_e_vip(r_l4e, z0(batch, n_cols, 0))
            r_l4vip_new, ns_l4vip = self._update_vip(self.l4_vip, I_l4_vip, state, "l4")
            I_l4i = I_l4i + self.syn_l4_vip_sst(z0(batch, n_cols, 0), r_l4vip)

        r_l4e_new, r_l4i_new, ns_l4 = self._update_layer(
            self.l4_e, self.l4_i, I_l4e, I_l4i, state, "l4")

        # ── L2/3 ─────────────────────────────────────────────────────────────
        I_l23e = (self.syn_l4e_l23e(r_l4e_new,  z0(batch, n_cols, 0))
                  + self.syn_l23_ee(r_l23e,      z0(batch, n_cols, 0))
                  + self.syn_l23_ie(z0(batch, n_cols, 0), r_l23i)
                  + torch.einsum("bce,coe->bco", l23_fb, self.fb_proj_w))
        I_l23i = (self.syn_l4e_l23i(r_l4e_new,  z0(batch, n_cols, 0))
                  + self.syn_l23_ei(r_l23e,      z0(batch, n_cols, 0)))

        # Corticortical: previous-timestep L5E of col (k+1) → L23E of col k
        if self.apical is not None:
            cortico = self.apical.l23_corticortical(r_l5e)
            if cortico is not None:
                I_l23e = I_l23e + cortico

        if self.disinhibition:
            r_l23vip = state["r_l23vip"]
            I_l23_vip = self.syn_l23_e_vip(r_l23e, z0(batch, n_cols, 0))
            r_l23vip_new, ns_l23vip = self._update_vip(self.l23_vip, I_l23_vip, state, "l23")
            I_l23i = I_l23i + self.syn_l23_vip_sst(z0(batch, n_cols, 0), r_l23vip)

        r_l23e_new, r_l23i_new, ns_l23 = self._update_layer(
            self.l23_e, self.l23_i, I_l23e, I_l23i, state, "l23")

        # ── L5 ───────────────────────────────────────────────────────────────
        I_l5e = (self.syn_l4e_l5e(r_l4e_new,   z0(batch, n_cols, 0))
                 + self.syn_l23e_l5e(r_l23e_new, z0(batch, n_cols, 0))
                 + self.syn_l5_ee(r_l5e,         z0(batch, n_cols, 0))
                 + self.syn_l5_ie(z0(batch, n_cols, 0), r_l5i))
        I_l5i = (self.syn_l23e_l5i(r_l23e_new,  z0(batch, n_cols, 0))
                 + self.syn_l5_ei(r_l5e,         z0(batch, n_cols, 0)))

        # Apical pathway: embed → L5E (skip / additive / multiplicative)
        if self.apical is not None:
            apical_add = self.apical.l5_additive(thal_full)
            if apical_add is not None:
                I_l5e = I_l5e + apical_add
            I_l5e = self.apical.l5_multiplicative(I_l5e, thal_full)

        if self.disinhibition:
            r_l5vip = state["r_l5vip"]
            I_l5_vip = self.syn_l5_e_vip(r_l5e, z0(batch, n_cols, 0))
            r_l5vip_new, ns_l5vip = self._update_vip(self.l5_vip, I_l5_vip, state, "l5")
            I_l5i = I_l5i + self.syn_l5_vip_sst(z0(batch, n_cols, 0), r_l5vip)

        r_l5e_new, r_l5i_new, ns_l5 = self._update_layer(
            self.l5_e, self.l5_i, I_l5e, I_l5i, state, "l5")

        # ── L6 ───────────────────────────────────────────────────────────────
        I_l6e = (self.syn_l23e_l6e(r_l23e_new, z0(batch, n_cols, 0))
                 + self.syn_l5e_l6e(r_l5e_new,  z0(batch, n_cols, 0))
                 + self.syn_l6_ee(r_l6e,         z0(batch, n_cols, 0))
                 + self.syn_l6_ie(z0(batch, n_cols, 0), r_l6i))
        I_l6i = (self.syn_l5e_l6i(r_l5e_new,   z0(batch, n_cols, 0))
                 + self.syn_l6_ei(r_l6e,         z0(batch, n_cols, 0)))

        if self.disinhibition:
            r_l6vip = state["r_l6vip"]
            I_l6_vip = self.syn_l6_e_vip(r_l6e, z0(batch, n_cols, 0))
            r_l6vip_new, ns_l6vip = self._update_vip(self.l6_vip, I_l6_vip, state, "l6")
            I_l6i = I_l6i + self.syn_l6_vip_sst(z0(batch, n_cols, 0), r_l6vip)

        r_l6e_new, r_l6i_new, ns_l6 = self._update_layer(
            self.l6_e, self.l6_i, I_l6e, I_l6i, state, "l6")

        new_state = {
            "r_l4e":  r_l4e_new,  "r_l4i":  r_l4i_new,
            "r_l23e": r_l23e_new, "r_l23i": r_l23i_new,
            "r_l5e":  r_l5e_new,  "r_l5i":  r_l5i_new,
            "r_l6e":  r_l6e_new,  "r_l6i":  r_l6i_new,
        }
        new_state.update(ns_l4)
        new_state.update(ns_l23)
        new_state.update(ns_l5)
        new_state.update(ns_l6)
        if self.disinhibition:
            new_state["r_l4vip"]  = r_l4vip_new
            new_state["r_l23vip"] = r_l23vip_new
            new_state["r_l5vip"]  = r_l5vip_new
            new_state["r_l6vip"]  = r_l6vip_new
            new_state.update(ns_l4vip)
            new_state.update(ns_l23vip)
            new_state.update(ns_l5vip)
            new_state.update(ns_l6vip)

        layer_outputs = {
            "l23_out": r_l23e_new,
            "l5_out":  r_l5e_new,
            "l6_out":  r_l6e_new,
        }
        return layer_outputs, new_state

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        n_cols = self.n_cols
        state = {}
        for prefix, pop_e, pop_i, n_e, n_i in [
            ("l4",  self.l4_e,  self.l4_i,  self.n_l4e,  self.n_l4i),
            ("l23", self.l23_e, self.l23_i, self.n_l23e, self.n_l23i),
            ("l5",  self.l5_e,  self.l5_i,  self.n_l5e,  self.n_l5i),
            ("l6",  self.l6_e,  self.l6_i,  self.n_l6e,  self.n_l6i),
        ]:
            state[f"r_{prefix}e"] = torch.zeros(batch_size, n_cols, n_e, device=device)
            state[f"r_{prefix}i"] = torch.zeros(batch_size, n_cols, n_i, device=device)
            for k, v in pop_e.init_state(batch_size).items():
                state[f"{prefix}_e_{k}"] = v
            for k, v in pop_i.init_state(batch_size).items():
                state[f"{prefix}_i_{k}"] = v
        if self.disinhibition:
            for prefix, pop_vip, n_vip in [
                ("l4",  self.l4_vip,  self.n_l4vip),
                ("l23", self.l23_vip, self.n_l23vip),
                ("l5",  self.l5_vip,  self.n_l5vip),
                ("l6",  self.l6_vip,  self.n_l6vip),
            ]:
                state[f"r_{prefix}vip"] = torch.zeros(batch_size, n_cols, n_vip, device=device)
                for k, v in pop_vip.init_state(batch_size).items():
                    state[f"{prefix}_vip_{k}"] = v
        return state
