"""Tsodyks-Markram Short-Term Plasticity synapse."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .utils import effective_excitatory, effective_inhibitory
from .static import _softplus_inv
from cortexlm.neurons.utils import init_lognormal_taus


class STPSynapse(nn.Module):
    """
    Tsodyks-Markram STP synapse with Dale's Law.

    State per synapse connection:
        u: calcium / release probability (facilitation variable). Range [U0, 1].
        x: fraction of available vesicles (depression variable). Range [0, 1].

    Update (continuous-rate approximation, per token step):
        u[t+1] = u[t] + (dt/tau_f) * (U0 - u[t]) + U0 * (1 - u[t]) * r_pre[t]
        x[t+1] = x[t] + (dt/tau_d) * (1 - x[t]) - u[t] * x[t] * r_pre[t]
        A_eff[t] = u[t] * x[t]   (modulates base weight)

    I_post = (W_e * A_eff_e) @ r_pre_e + (W_i * A_eff_i) @ r_pre_i

    STP is applied only to inter-column synapses (as per spec). Within-column
    synapses use StaticSynapse.

    Note on dimensions: u and x are per-(pre, post) pair = n_pre_e * n_post for
    excitatory, etc. For efficiency, we store them as [n_post, n_pre] matrices.
    """

    def __init__(self, n_pre_e: int, n_pre_i: int, n_post: int, config: dict):
        super().__init__()
        self.n_pre_e = n_pre_e
        self.n_pre_i = n_pre_i
        self.n_post = n_post
        self.dt = config.get("simulation", {}).get("dt", 1.0)

        scfg = config.get("synapse", {})
        self.U0_e = scfg.get("U0_e", 0.2)
        self.U0_i = scfg.get("U0_i", 0.5)

        lo_f, hi_f = scfg.get("tau_f_range", [50.0, 300.0])
        lo_d, hi_d = scfg.get("tau_d_range", [100.0, 800.0])

        # Per-synapse time constants stored as [n_post, n_pre] matrices
        tau_f_e = init_lognormal_taus(n_post * n_pre_e, lo_f, hi_f).reshape(n_post, n_pre_e)
        tau_d_e = init_lognormal_taus(n_post * n_pre_e, lo_d, hi_d).reshape(n_post, n_pre_e)
        tau_f_i = init_lognormal_taus(n_post * n_pre_i, lo_f, hi_f).reshape(n_post, n_pre_i)
        tau_d_i = init_lognormal_taus(n_post * n_pre_i, lo_d, hi_d).reshape(n_post, n_pre_i)

        # Buffers (not parameters — STP time constants are fixed)
        self.register_buffer("tau_f_e", tau_f_e)
        self.register_buffer("tau_d_e", tau_d_e)
        self.register_buffer("tau_f_i", tau_f_i)
        self.register_buffer("tau_d_i", tau_d_i)

        # Weight matrices (learnable, Dale's Law via softplus)
        # Fan-in init: target = 1/n_pre so total input ≈ n_pre × 0.5 × (1/n_pre) = 0.5
        e_offset = _softplus_inv(1.0 / n_pre_e) if n_pre_e > 0 else -2.25
        i_offset = _softplus_inv(1.0 / n_pre_i) if n_pre_i > 0 else -2.25
        self.W_e_raw = nn.Parameter(torch.randn(n_post, n_pre_e) * 0.01 + e_offset)
        self.W_i_raw = nn.Parameter(torch.randn(n_post, n_pre_i) * 0.01 + i_offset)

    @property
    def W_e(self):
        return effective_excitatory(self.W_e_raw)

    @property
    def W_i(self):
        return effective_inhibitory(self.W_i_raw)

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        device = self.W_e_raw.device
        return {
            "u_e": torch.full((batch_size, self.n_post, self.n_pre_e), self.U0_e, device=device),
            "x_e": torch.ones(batch_size, self.n_post, self.n_pre_e, device=device),
            "u_i": torch.full((batch_size, self.n_post, self.n_pre_i), self.U0_i, device=device),
            "x_i": torch.ones(batch_size, self.n_post, self.n_pre_i, device=device),
        }

    def forward(
        self,
        r_pre_e: torch.Tensor,   # [batch, n_pre_e]
        r_pre_i: torch.Tensor,   # [batch, n_pre_i]
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            I_post: [batch, n_post]
            new_state: updated u, x tensors
        """
        u_e, x_e = state["u_e"], state["x_e"]
        u_i, x_i = state["u_i"], state["x_i"]

        # Expand r_pre for broadcasting: [batch, 1, n_pre_e]
        r_e = r_pre_e.unsqueeze(1)  # [batch, 1, n_pre_e]
        r_i = r_pre_i.unsqueeze(1)  # [batch, 1, n_pre_i]

        dt = self.dt

        # Excitatory STP update
        u_e_new = (u_e
                   + (dt / self.tau_f_e) * (self.U0_e - u_e)
                   + self.U0_e * (1 - u_e) * r_e)
        x_e_new = (x_e
                   + (dt / self.tau_d_e) * (1 - x_e)
                   - u_e * x_e * r_e)

        # Inhibitory STP update
        u_i_new = (u_i
                   + (dt / self.tau_f_i) * (self.U0_i - u_i)
                   + self.U0_i * (1 - u_i) * r_i)
        x_i_new = (x_i
                   + (dt / self.tau_d_i) * (1 - x_i)
                   - u_i * x_i * r_i)

        # Clamp state variables to valid range
        u_e_new = u_e_new.clamp(self.U0_e, 1.0)
        x_e_new = x_e_new.clamp(0.0, 1.0)
        u_i_new = u_i_new.clamp(self.U0_i, 1.0)
        x_i_new = x_i_new.clamp(0.0, 1.0)

        # Effective weights: modulate base weight by STP factor
        # A_eff: [batch, n_post, n_pre]; W: [n_post, n_pre]
        A_e = u_e_new * x_e_new   # [batch, n_post, n_pre_e]
        A_i = u_i_new * x_i_new   # [batch, n_post, n_pre_i]

        W_e = self.W_e.unsqueeze(0)   # [1, n_post, n_pre_e]
        W_i = self.W_i.unsqueeze(0)   # [1, n_post, n_pre_i]

        # I_e[b, j] = sum_i W_e[j,i] * A_e[b,j,i] * r_e[b,i]
        I_e = (W_e * A_e * r_e).sum(dim=-1)   # [batch, n_post]
        I_i = (W_i * A_i * r_i).sum(dim=-1)   # [batch, n_post]
        I_post = I_e + I_i

        new_state = {
            "u_e": u_e_new, "x_e": x_e_new,
            "u_i": u_i_new, "x_i": x_i_new,
        }
        return I_post, new_state

    def enforce_dale(self):
        """No-op: softplus always enforces Dale's Law."""
        pass
