"""Rate-coded AdEx neurons: two state variables (v, w). Primary model."""

import torch
import torch.nn as nn
from typing import Dict, List

from .base import NeuronPopulation
from .utils import make_taus


class RateAdExNeurons(NeuronPopulation):
    """
    Rate-coded Adaptive Exponential Integrate-and-Fire neuron population.

    Two state variables per neuron: membrane potential v, adaptation current w.

    Euler update:
        v[t+1] = v[t] + (dt/tau_m) * (-(v[t]-E_L) + R*I_syn[t] - w[t])
        w[t+1] = w[t] + (dt/tau_w) * (a*(v[t]-E_L) - w[t])
        output[t] = sigmoid(v[t])

    tau_m: per-neuron, log-normal [2,30] timesteps (membrane timescale)
    tau_w: per-neuron, log-normal [30,500] timesteps (adaptation timescale, ~10x larger)
    a: subthreshold adaptation coupling (scalar), default 0.1

    Both taus are buffers by default; set learn_taus=True to make learnable.
    """

    def __init__(self, n_neurons: int, config: dict):
        super().__init__(n_neurons, config)

        ncfg = config["neuron"]
        learn = ncfg.get("learn_taus", False)

        # Membrane timescale
        dist_m = ncfg.get("tau_m_dist", "lognormal")
        lo_m, hi_m = ncfg.get("tau_m_range", [2.0, 30.0])
        taus_m = make_taus(n_neurons, dist_m, lo_m, hi_m, learn)

        # Adaptation timescale (~10x larger)
        dist_w = ncfg.get("tau_w_dist", "lognormal")
        lo_w, hi_w = ncfg.get("tau_w_range", [30.0, 500.0])
        taus_w = make_taus(n_neurons, dist_w, lo_w, hi_w, learn)

        if learn:
            self.tau_m = nn.Parameter(taus_m)
            self.tau_w = nn.Parameter(taus_w)
        else:
            self.register_buffer("tau_m", taus_m)
            self.register_buffer("tau_w", taus_w)

        # Fixed scalar parameters
        self.E_L = 0.0
        self.R = 1.0
        self.a = ncfg.get("adaptation_a", 0.1)

    def forward(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, n_neurons] total synaptic input
            state: {'v': [batch, n_neurons], 'w': [batch, n_neurons]}
        Returns:
            output: [batch, n_neurons] sigmoid(v_new), in [0,1]
            new_state: {'v': ..., 'w': ...}
        """
        v = state["v"]
        w = state["w"]

        dv = (self.dt / self.tau_m) * (-(v - self.E_L) + self.R * x - w)
        dw = (self.dt / self.tau_w) * (self.a * (v - self.E_L) - w)

        v_new = v + dv
        w_new = w + dw

        output = torch.sigmoid(v_new)
        return output, {"v": v_new, "w": w_new}

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.tau_m.device
        return {
            "v": torch.full((batch_size, self.n_neurons), self.E_L, device=device),
            "w": torch.zeros(batch_size, self.n_neurons, device=device),
        }

    def state_keys(self) -> List[str]:
        return ["v", "w"]
