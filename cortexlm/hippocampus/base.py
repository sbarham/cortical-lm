"""Abstract base class for hippocampal modules."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class HippocampalModule(nn.Module, ABC):
    """
    Hippocampal module interface.

    forward(cortical_state_l5, column_states)
        -> modulation [batch, n_columns, modulation_dim]
           surprise   [batch, 1] or None

    init_state(batch_size) -> state dict
    """

    def __init__(self, config: dict, n_columns: int, n_l5e: int):
        super().__init__()
        self.config = config
        self.n_columns = n_columns
        self.n_l5e = n_l5e

    @abstractmethod
    def forward(
        self,
        cortical_state_l5: torch.Tensor,           # [batch, n_columns * n_l5e]
        column_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            modulation: [batch, n_columns, modulation_dim]
            surprise:   [batch, 1] or None
        """
        ...

    @abstractmethod
    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        ...
