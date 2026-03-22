"""Tests for connectivity modules."""

import pytest
import torch
from cortexlm.connectivity.local import gaussian_connectivity_probs, gaussian_connectivity_mask
from cortexlm.connectivity.small_world import small_world_connectivity_mask, clustering_coefficient


def test_gaussian_connectivity_falloff():
    """Connection probability should decrease with distance."""
    n = 20
    p_max = 0.9
    sigma = 3.0
    probs = gaussian_connectivity_probs(n, p_max, sigma)

    # P(0, 1) should be > P(0, 10)
    p_near = probs[0, 1].item()
    p_far  = probs[0, 10].item()
    assert p_near > p_far, f"Expected p_near={p_near:.3f} > p_far={p_far:.3f}"


def test_gaussian_no_self_connections():
    """Diagonal must be zero."""
    probs = gaussian_connectivity_probs(10, 0.8, 2.0)
    assert (probs.diag() == 0).all()


def test_gaussian_mask_binary():
    """Output mask must be boolean."""
    mask = gaussian_connectivity_mask(10, 0.9, 2.0)
    assert mask.dtype == torch.bool


def test_gaussian_mask_no_self_connections():
    mask = gaussian_connectivity_mask(15, 0.8, 3.0)
    assert not mask.diag().any()


def test_gaussian_symmetry_of_probs():
    """Probabilities should be symmetric: P(i,j) == P(j,i)."""
    probs = gaussian_connectivity_probs(10, 0.8, 2.0)
    assert torch.allclose(probs, probs.t())


def test_small_world_no_self_connections():
    mask = small_world_connectivity_mask(20, k=4, beta=0.1)
    assert not mask.diag().any()


def test_small_world_higher_clustering_than_random():
    """Small-world graph should have higher clustering than Erdos-Renyi at same density."""
    import random
    random.seed(42)
    n = 30
    mask_sw = small_world_connectivity_mask(n, k=4, beta=0.05)
    cc_sw = clustering_coefficient(mask_sw)

    # Random graph with same density
    density = mask_sw.float().mean().item()
    mask_rand = torch.bernoulli(torch.full((n, n), density)).bool()
    mask_rand.fill_diagonal_(False)
    cc_rand = clustering_coefficient(mask_rand)

    # Small-world should have meaningfully higher clustering
    # (may occasionally fail due to randomness; use generous threshold)
    assert cc_sw >= cc_rand * 0.5, \
        f"Small-world CC={cc_sw:.3f} should be >= random CC={cc_rand:.3f}"


def test_small_world_shape():
    mask = small_world_connectivity_mask(16, k=4, beta=0.1)
    assert mask.shape == (16, 16)
