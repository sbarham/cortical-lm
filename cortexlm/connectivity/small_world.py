"""Watts-Strogatz small-world inter-column connectivity."""

import torch
import random


def small_world_connectivity_mask(
    n_columns: int,
    k: int,
    beta: float,
) -> torch.Tensor:
    """
    Watts-Strogatz small-world connectivity.

    Start with ring lattice: each column connected to k nearest neighbors.
    Rewire each edge with probability beta to a random target.
    No self-connections.

    Args:
        n_columns: number of columns
        k: number of nearest neighbors in ring lattice (should be even)
        beta: rewiring probability [0, 1]

    Returns:
        mask: [n_columns, n_columns] bool tensor
    """
    n = n_columns
    # Build ring lattice as adjacency set
    edges = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            nbr = (i + j) % n
            if i != nbr:
                edges.add((i, nbr))
                edges.add((nbr, i))

    # Rewire
    new_edges = set()
    for (i, j) in list(edges):
        if random.random() < beta:
            # Rewire i→j to i→random (no self, no duplicate)
            candidates = [c for c in range(n) if c != i and (i, c) not in new_edges]
            if candidates:
                new_j = random.choice(candidates)
                new_edges.add((i, new_j))
            else:
                new_edges.add((i, j))
        else:
            new_edges.add((i, j))

    mask = torch.zeros(n, n, dtype=torch.bool)
    for (i, j) in new_edges:
        mask[i, j] = True
    return mask


def clustering_coefficient(mask: torch.Tensor) -> float:
    """Compute average clustering coefficient of undirected graph."""
    n = mask.shape[0]
    m = mask.float()
    # Symmetrize
    m = ((m + m.t()) > 0).float()
    cc_list = []
    for i in range(n):
        neighbors = m[i].nonzero(as_tuple=True)[0]
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges among neighbors
        sub = m[neighbors][:, neighbors]
        n_edges = sub.sum().item() / 2
        possible = k * (k - 1) / 2
        cc_list.append(n_edges / possible)
    return sum(cc_list) / len(cc_list) if cc_list else 0.0
