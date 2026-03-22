"""Static synapse with Dale's Law enforcement via softplus."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import effective_excitatory, effective_inhibitory


def _softplus_inv(w: float) -> float:
    """Compute softplus^{-1}(w) = log(exp(w) - 1) so that softplus(result) == w."""
    return math.log(math.exp(w) - 1.0)


class StaticSynapse(nn.Module):
    """
    Simple static weight matrix with Dale's Law.

    Excitatory weights W_e >= 0 (via softplus on raw parameter).
    Inhibitory weights W_i <= 0 (via -softplus on raw parameter).

    forward(r_pre_e, r_pre_i) -> I_post = W_e @ r_pre_e + W_i @ r_pre_i
    """

    def __init__(self, n_pre_e: int, n_pre_i: int, n_post: int):
        super().__init__()
        self.n_pre_e = n_pre_e
        self.n_pre_i = n_pre_i
        self.n_post = n_post

        # Fan-in dependent init: target weight = 1/n_pre so that
        # total synaptic input ≈ n_pre × r̄ × (1/n_pre) = r̄ ≈ 0.5
        # regardless of fan-in.  Keeps tanh(I) ≈ 0.46 in the linear regime.
        # offset = softplus^{-1}(1/n_pre), noise scale 0.01 for symmetry-breaking.
        e_offset = _softplus_inv(1.0 / n_pre_e) if n_pre_e > 0 else -2.25
        i_offset = _softplus_inv(1.0 / n_pre_i) if n_pre_i > 0 else -2.25
        self.W_e_raw = nn.Parameter(torch.randn(n_post, n_pre_e) * 0.01 + e_offset)
        self.W_i_raw = nn.Parameter(torch.randn(n_post, n_pre_i) * 0.01 + i_offset)

    @property
    def W_e(self) -> torch.Tensor:
        return effective_excitatory(self.W_e_raw)

    @property
    def W_i(self) -> torch.Tensor:
        return effective_inhibitory(self.W_i_raw)

    def forward(
        self,
        r_pre_e: torch.Tensor,  # [batch, n_pre_e]
        r_pre_i: torch.Tensor,  # [batch, n_pre_i]
    ) -> torch.Tensor:
        """Returns post-synaptic current [batch, n_post]."""
        if self.n_pre_e > 0:
            I_e = r_pre_e @ self.W_e.t()
        else:
            I_e = torch.zeros(r_pre_e.shape[0], self.n_post, device=r_pre_e.device)
        if self.n_pre_i > 0:
            I_i = r_pre_i @ self.W_i.t()
        else:
            I_i = torch.zeros(r_pre_i.shape[0], self.n_post, device=r_pre_i.device)
        return I_e + I_i

    def enforce_dale(self):
        """No-op: softplus always enforces Dale's Law. Kept for interface consistency."""
        pass


class BatchedStaticSynapse(nn.Module):
    """
    Static synapse batched over n_cols columns.

    Weights: W_e [n_cols, n_post, n_pre_e], W_i [n_cols, n_post, n_pre_i]
    forward: r_e [batch, n_cols, n_pre_e], r_i [batch, n_cols, n_pre_i]
          -> I   [batch, n_cols, n_post]

    Handles n_pre_e=0 or n_pre_i=0 by omitting the corresponding weight.
    """

    def __init__(self, n_cols: int, n_pre_e: int, n_pre_i: int, n_post: int):
        super().__init__()
        self.n_cols  = n_cols
        self.n_pre_e = n_pre_e
        self.n_pre_i = n_pre_i
        self.n_post  = n_post

        if n_pre_e > 0:
            e_offset = _softplus_inv(1.0 / n_pre_e)
            self.W_e_raw = nn.Parameter(torch.randn(n_cols, n_post, n_pre_e) * 0.01 + e_offset)
        if n_pre_i > 0:
            i_offset = _softplus_inv(1.0 / n_pre_i)
            self.W_i_raw = nn.Parameter(torch.randn(n_cols, n_post, n_pre_i) * 0.01 + i_offset)

    @property
    def W_e(self) -> torch.Tensor:
        return effective_excitatory(self.W_e_raw)

    @property
    def W_i(self) -> torch.Tensor:
        return effective_inhibitory(self.W_i_raw)

    def forward(
        self,
        r_pre_e: torch.Tensor,  # [batch, n_cols, n_pre_e]  (may be size-0 on last dim)
        r_pre_i: torch.Tensor,  # [batch, n_cols, n_pre_i]
    ) -> torch.Tensor:
        """Returns post-synaptic current [batch, n_cols, n_post]."""
        batch = r_pre_e.shape[0]
        if self.n_pre_e > 0:
            # einsum: b=batch, c=col, p=pre → o=post
            I_e = torch.einsum("bcp,cop->bco", r_pre_e, self.W_e)
        else:
            I_e = r_pre_e.new_zeros(batch, self.n_cols, self.n_post)
        if self.n_pre_i > 0:
            I_i = torch.einsum("bcp,cop->bco", r_pre_i, self.W_i)
        else:
            I_i = r_pre_i.new_zeros(batch, self.n_cols, self.n_post)
        return I_e + I_i
