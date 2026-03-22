"""Rate-coded neurons: single state variable (membrane potential)."""

import torch
import torch.nn as nn
from typing import Dict, List

from .base import NeuronPopulation
from .utils import make_taus, get_nonlinearity


class RateNeurons(NeuronPopulation):
    """
    Simple rate-coded neuron population with one state variable.

    Euler update:
        v[t+1] = v[t] + (dt / tau_m) * (-v[t] + f(I_syn[t]))
        output[t] = sigmoid(v[t])

    tau_m is per-neuron, drawn from a log-normal distribution by default.
    Registered as a buffer (non-learnable) unless config learn_taus=True.
    """

    def __init__(self, n_neurons: int, config: dict):
        super().__init__(n_neurons, config)

        ncfg = config["neuron"]
        dist = ncfg.get("tau_m_dist", "lognormal")
        lo, hi = ncfg.get("tau_m_range", [2.0, 30.0])
        learn = ncfg.get("learn_taus", False)
        nonlin = ncfg.get("nonlinearity", "tanh")

        self.f = get_nonlinearity(nonlin)

        taus = make_taus(n_neurons, dist, lo, hi, learn)
        if learn:
            self.tau_m = nn.Parameter(taus)
        else:
            self.register_buffer("tau_m", taus)

    def forward(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, n_neurons] synaptic input
            state: {'v': [batch, n_neurons]}
        Returns:
            output: [batch, n_neurons] in [0, 1]
            new_state: {'v': [batch, n_neurons]}
        """
        v = state["v"]
        alpha = self.dt / self.tau_m  # [n_neurons]
        v_new = v + alpha * (-v + self.f(x))
        output = torch.sigmoid(v_new)
        return output, {"v": v_new}

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.tau_m.device
        return {"v": torch.zeros(batch_size, self.n_neurons, device=device)}

    def state_keys(self) -> List[str]:
        return ["v"]
