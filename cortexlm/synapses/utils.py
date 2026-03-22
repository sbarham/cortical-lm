"""Shared synapse utilities."""

import torch
import torch.nn.functional as F


def effective_excitatory(W_raw: torch.Tensor) -> torch.Tensor:
    """Dale's Law: excitatory weights are strictly >= 0."""
    return F.softplus(W_raw)


def effective_inhibitory(W_raw: torch.Tensor) -> torch.Tensor:
    """Dale's Law: inhibitory weights are strictly <= 0."""
    return -F.softplus(W_raw)
