"""Abstract base class for cortical columns."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List


class CorticalColumn(nn.Module, ABC):
    """
    A cortical column receives input vectors and maintains internal neural state.

    Interface contract:
      forward(inputs, state) -> (layer_outputs, new_state)
        inputs: dict with keys from input_keys(), each Tensor [batch, n_input_i]
        state: dict of Tensors (internal neuron states; opaque outside column)
        layer_outputs: dict with keys from output_keys(), each Tensor [batch, n_output_i]
        new_state: updated state dict

      init_state(batch_size) -> state dict
      input_keys() -> list of str    e.g. ['thalamic_input', 'l23_feedback']
      output_keys() -> list of str   e.g. ['l23_out', 'l5_out', 'l6_out']
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ...

    @abstractmethod
    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def input_keys(self) -> List[str]:
        ...

    @abstractmethod
    def output_keys(self) -> List[str]:
        ...
