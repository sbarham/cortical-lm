"""Batched neuron population: independent tau draws per column, no reshape needed."""

import torch
import torch.nn as nn
from typing import Dict, List

from .utils import make_taus, get_nonlinearity


class BatchedNeuronPop(nn.Module):
    """
    Neuron population that operates on [batch, n_cols, n_neurons] tensors.

    Each column gets its own independently drawn tau values — stored as
    [n_cols, n_neurons] buffers rather than sharing one [n_neurons] vector.
    This preserves both per-neuron AND per-column timescale diversity.

    Supports rate and rate_adex models transparently.
    No reshape/flatten needed: the extra column dimension broadcasts cleanly.
    """

    def __init__(self, config: dict, n_neurons: int, n_cols: int):
        super().__init__()
        ncfg = config["neuron"]
        model = ncfg["model"]
        learn = ncfg.get("learn_taus", False)
        self.dt = config.get("simulation", {}).get("dt", 1.0)
        self.n_neurons = n_neurons
        self.n_cols = n_cols
        self._model = model

        # ── tau_m: [n_cols, n_neurons] — one independent draw per column ──────
        dist_m = ncfg.get("tau_m_dist", "lognormal")
        lo_m, hi_m = ncfg.get("tau_m_range", [2.0, 30.0])
        tau_m = torch.stack([
            make_taus(n_neurons, dist_m, lo_m, hi_m, learn=False)
            for _ in range(n_cols)
        ])   # [n_cols, n_neurons]
        if learn:
            self.tau_m = nn.Parameter(tau_m)
        else:
            self.register_buffer("tau_m", tau_m)

        # ── rate_adex extras ──────────────────────────────────────────────────
        self._has_adex = (model == "rate_adex")
        if self._has_adex:
            dist_w = ncfg.get("tau_w_dist", "lognormal")
            lo_w, hi_w = ncfg.get("tau_w_range", [30.0, 500.0])
            tau_w = torch.stack([
                make_taus(n_neurons, dist_w, lo_w, hi_w, learn=False)
                for _ in range(n_cols)
            ])   # [n_cols, n_neurons]
            if learn:
                self.tau_w = nn.Parameter(tau_w)
            else:
                self.register_buffer("tau_w", tau_w)

            self.E_L = 0.0
            self.R   = 1.0
            self.a   = ncfg.get("adaptation_a", 0.1)
        else:
            nonlin = ncfg.get("nonlinearity", "tanh")
            self._f = get_nonlinearity(nonlin)
            # LayerNorm over synaptic inputs: normalizes I → keeps tanh' ≈ 1
            # so per-layer gradient ≈ output_activation'(v) × ||W|| rather than
            # output_activation'(v) × tanh'(I) × ||W||.  Applied over neuron dim.
            self.input_norm = nn.LayerNorm(n_neurons)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                    # [batch, n_cols, n_neurons]
        state: Dict[str, torch.Tensor],     # {"v": [batch, n_cols, n], ...}
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # alpha: [1, n_cols, n_neurons]  (unsqueeze batch dim for broadcast)
        alpha_m = (self.dt / self.tau_m).unsqueeze(0)
        v = state["v"]

        if self._has_adex:
            alpha_w = (self.dt / self.tau_w).unsqueeze(0)
            w = state["w"]
            dv = alpha_m * (-(v - self.E_L) + self.R * x - w)
            dw = alpha_w * (self.a * (v - self.E_L) - w)
            v_new = v + dv
            w_new = w + dw
            r = (torch.tanh(v_new) + 1.0) / 2.0
            return r, {"v": v_new, "w": w_new}
        else:
            # LayerNorm on synaptic input: keeps tanh' ≈ 1 at every step,
            # eliminating tanh from the spatial gradient product.
            # (tanh(v)+1)/2 replaces sigmoid: max gradient 0.5 vs 0.25,
            # and avoids the double-squashing that collapses inter-layer gradients.
            x_norm = self.input_norm(x)
            v_new = v + alpha_m * (-v + self._f(x_norm))
            r = (torch.tanh(v_new) + 1.0) / 2.0
            return r, {"v": v_new}

    # ── state helpers ─────────────────────────────────────────────────────────

    def state_keys(self) -> List[str]:
        return ["v", "w"] if self._has_adex else ["v"]

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.tau_m.device
        E_L = self.E_L if self._has_adex else 0.0
        s = {"v": torch.full((batch_size, self.n_cols, self.n_neurons), E_L, device=device)}
        if self._has_adex:
            s["w"] = torch.zeros(batch_size, self.n_cols, self.n_neurons, device=device)
        return s
