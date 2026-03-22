"""Gaussian local inter-column connectivity."""

import torch
import numpy as np


def gaussian_connectivity_mask(
    n_columns: int,
    p_max: float,
    sigma: float,
) -> torch.Tensor:
    """
    Generate a binary connectivity mask using Gaussian fall-off.

    P(i→j) = p_max * exp(-(i-j)^2 / (2*sigma^2))
    No self-connections (diagonal = 0).

    Returns:
        mask: [n_columns, n_columns] bool tensor
    """
    i_idx = torch.arange(n_columns).float()
    j_idx = torch.arange(n_columns).float()
    diff = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)   # [n, n]
    probs = p_max * torch.exp(-(diff ** 2) / (2 * sigma ** 2))
    probs.fill_diagonal_(0.0)

    # Sample binary mask from probabilities
    mask = torch.bernoulli(probs).bool()
    return mask


def gaussian_connectivity_probs(
    n_columns: int,
    p_max: float,
    sigma: float,
) -> torch.Tensor:
    """Return probability matrix (float) without sampling."""
    i_idx = torch.arange(n_columns).float()
    j_idx = torch.arange(n_columns).float()
    diff = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)
    probs = p_max * torch.exp(-(diff ** 2) / (2 * sigma ** 2))
    probs.fill_diagonal_(0.0)
    return probs
