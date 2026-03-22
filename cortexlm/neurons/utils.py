"""Shared helpers for neuron models."""

import math
import torch
import torch.nn as nn


def init_lognormal_taus(n: int, lo: float, hi: float, device=None) -> torch.Tensor:
    """
    Draw n time constants log-uniformly from [lo, hi] timesteps.
    Log-normal such that ~5th percentile ≈ lo, ~95th percentile ≈ hi.
    """
    log_lo, log_hi = math.log(lo), math.log(hi)
    mu = (log_lo + log_hi) / 2
    sigma = (log_hi - log_lo) / 4   # 2-sigma span covers the range
    taus = torch.exp(torch.normal(mu, sigma, size=(n,))).clamp(lo, hi)
    if device is not None:
        taus = taus.to(device)
    return taus


def init_uniform_taus(n: int, lo: float, hi: float, device=None) -> torch.Tensor:
    taus = torch.empty(n).uniform_(lo, hi)
    if device is not None:
        taus = taus.to(device)
    return taus


def get_nonlinearity(name: str):
    """Return activation function by name."""
    fns = {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
    }
    if name not in fns:
        raise ValueError(f"Unknown nonlinearity: {name}")
    return fns[name]


def make_taus(n: int, dist: str, lo: float, hi: float, learn: bool, device=None):
    """
    Create tau tensor as buffer or parameter depending on learn flag.
    Returns the tensor; caller registers it appropriately.
    """
    if dist == "lognormal":
        taus = init_lognormal_taus(n, lo, hi, device)
    elif dist == "uniform":
        taus = init_uniform_taus(n, lo, hi, device)
    elif dist == "fixed":
        mu = (lo + hi) / 2
        taus = torch.full((n,), mu, device=device)
    else:
        raise ValueError(f"Unknown tau distribution: {dist}")
    return taus
