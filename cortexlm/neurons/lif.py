"""
Leaky Integrate-and-Fire (LIF) spiking neurons.

WARNING: Spiking output is non-differentiable. This module uses the
Straight-Through Estimator (STE) for the Heaviside threshold:
  - Forward: spike = (v >= threshold).float()
  - Backward: gradient passes through as if spike = sigmoid(v) (surrogate)

NOT used in Phases 1-2 (BPTT on rate-coded models). Included for Phase 3+
or for ablation studies with surrogate gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .base import NeuronPopulation
from .utils import make_taus


class HeavisideSTE(torch.autograd.Function):
    """Heaviside with straight-through gradient estimator."""
    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        # Surrogate: sigmoid derivative centered at threshold
        sig = torch.sigmoid(v - ctx.threshold)
        surrogate = sig * (1 - sig)
        return grad_output * surrogate, None


class LIFNeurons(NeuronPopulation):
    """
    Standard LIF with hard threshold and refractory period.

    State: v (membrane), ref_count (refractory countdown)

    Update:
        ref_count[t+1] = max(0, ref_count[t] - 1) + spike[t] * refractory_steps
        mask = (ref_count[t] == 0)
        dv = (dt/tau_m) * (-v[t] + I_syn[t])
        v[t+1] = v[t] + dv (where mask) else v_reset
        spike[t] = Heaviside(v[t] - threshold)  [STE in backward]
    """

    def __init__(self, n_neurons: int, config: dict):
        super().__init__(n_neurons, config)

        ncfg = config["neuron"]
        dist = ncfg.get("tau_m_dist", "lognormal")
        lo, hi = ncfg.get("tau_m_range", [2.0, 30.0])
        learn = ncfg.get("learn_taus", False)

        taus = make_taus(n_neurons, dist, lo, hi, learn)
        if learn:
            self.tau_m = nn.Parameter(taus)
        else:
            self.register_buffer("tau_m", taus)

        self.threshold = 1.0
        self.v_reset = 0.0
        self.refractory_steps = int(2.0 / self.dt)  # 2ms refractory

    def forward(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v = state["v"]
        ref = state["ref_count"]

        # Refractory mask: 1 where neuron can fire
        active = (ref == 0).float()

        # Spike detection (STE in backward)
        spikes = HeavisideSTE.apply(v, self.threshold)
        spikes = spikes * active  # can't spike if refractory

        # Update membrane (reset spiked neurons)
        alpha = self.dt / self.tau_m
        dv = alpha * (-v + x)
        v_new = (v + dv) * (1 - spikes) + self.v_reset * spikes

        # Update refractory counter
        ref_new = (ref - 1).clamp(min=0) + spikes * self.refractory_steps

        return spikes, {"v": v_new, "ref_count": ref_new}

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.tau_m.device
        return {
            "v": torch.zeros(batch_size, self.n_neurons, device=device),
            "ref_count": torch.zeros(batch_size, self.n_neurons, device=device),
        }

    def state_keys(self) -> List[str]:
        return ["v", "ref_count"]
