"""Null hippocampal module (no hippocampus). Satisfies interface with zeros."""

import torch
from typing import Dict, Optional, Tuple

from .base import HippocampalModule


class NullHippocampus(HippocampalModule):
    """No-op hippocampus. Returns zero modulation and None surprise."""

    def forward(
        self,
        cortical_state_l5: torch.Tensor,
        column_states=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = cortical_state_l5.shape[0]
        device = cortical_state_l5.device
        modulation = torch.zeros(batch, self.n_columns, 1, device=device)
        return modulation, None

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return {}
