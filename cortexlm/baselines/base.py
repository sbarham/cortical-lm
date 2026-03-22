"""Shared abstract interface for all baseline models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class BaselineModel(nn.Module, ABC):
    """
    Shared interface with CortexLM for comparison scripts.

    All baselines implement:
      forward(token_sequence, initial_state=None) -> (all_logits, final_state)
      init_state(batch_size) -> state (model-specific)
      count_parameters() -> int
    """

    @abstractmethod
    def forward(
        self,
        token_sequence: torch.Tensor,     # [batch, seq_len]
        initial_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Returns: (all_logits [batch, seq_len, vocab_size], final_state)"""
        ...

    @abstractmethod
    def init_state(self, batch_size: int) -> Any:
        """Return initial (zero) state."""
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
