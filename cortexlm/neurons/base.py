"""Abstract base class for neuron populations."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List


class NeuronPopulation(nn.Module, ABC):
    """
    Abstract base for all neuron population models.

    All subclasses must implement:
      - forward(x, state) -> (output, new_state)
      - init_state(batch_size) -> state (dict of tensors)
      - state_keys() -> list of str

    `x`: total synaptic input, shape [batch, n_neurons].
    `state`: dict of tensors, each shape [batch, n_neurons].
    `output`: firing rate / activation in [0, 1], shape [batch, n_neurons].
    """

    def __init__(self, n_neurons: int, config: dict):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.dt = config.get("simulation", {}).get("dt", 1.0)

    @abstractmethod
    def forward(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: synaptic input [batch, n_neurons]
            state: dict of state tensors, each [batch, n_neurons]
        Returns:
            output: firing rate in [0,1], [batch, n_neurons]
            new_state: updated state dict
        """
        ...

    @abstractmethod
    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize state tensors (zeros or resting values)."""
        ...

    @abstractmethod
    def state_keys(self) -> List[str]:
        """Return list of state variable names."""
        ...
