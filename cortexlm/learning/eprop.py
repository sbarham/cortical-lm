"""
e-prop trainer (Bellec et al. 2020 / Nature Communications).

Online e-prop for rate-coded neurons. Replaces the recurrent backward pass
with local eligibility traces + global learning signal.

For rate neurons with tanh nonlinearity:
    ψ_j(t) = 1 - tanh²(v_j(t))   (exact derivative, not surrogate)

Eligibility trace:
    e_ij(t) = r_pre_i(t) * ψ_j(t)
    ē_ij(t) = γ * ē_ij(t-1) + e_ij(t),  γ = exp(-dt / τ_e)

Weight update:
    Δw_ij = η * L(t) * ē_ij(t)

Implementation:
- Eligibility traces computed via forward hooks on neuron populations.
- Global learning signal L(t) = gradient of CE loss w.r.t. readout input (autograd).
- Only recurrent weights use e-prop; readout uses autograd throughout.
- Dale's Law re-applied after each update.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from cortexlm.model import CortexLM, ModelState
from cortexlm.utils.metrics import compute_perplexity


class EligibilityTraceBuffer:
    """
    Holds eligibility traces for one weight matrix.

    Trace: ē_ij ∈ R^{n_post × n_pre}
    Updated online: ē(t) = γ * ē(t-1) + r_pre(t) ⊗ ψ_post(t)
    """

    def __init__(self, n_pre: int, n_post: int, gamma: float, device):
        self.gamma = gamma
        self.trace = torch.zeros(n_post, n_pre, device=device)

    def update(self, r_pre: torch.Tensor, psi_post: torch.Tensor):
        """
        r_pre:    [batch, n_pre]
        psi_post: [batch, n_post]
        Updates trace in-place (averaged over batch).
        """
        # Outer product averaged over batch: [n_post, n_pre]
        delta = torch.einsum("bi,bj->ji", r_pre, psi_post) / r_pre.shape[0]
        self.trace = self.gamma * self.trace + delta

    def reset(self):
        self.trace.zero_()


class EpropTrainer:
    """
    Online e-prop trainer.

    Architecture constraints for e-prop:
    - Only rate or rate_adex neurons (tanh nonlinearity → exact pseudo-derivative)
    - Only static synapses (STP makes trace computation more complex)

    The readout layer still uses standard autograd for L(t).
    """

    def __init__(self, model: CortexLM, config: dict, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        tcfg = config["training"]
        self.lr = tcfg.get("lr", 3e-4)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.reset_state = config["learning"].get("reset_state_between_batches", False)

        # Compute trace decay constant γ = exp(-dt / τ_e)
        dt = config.get("simulation", {}).get("dt", 1.0)
        tau_m_range = config["neuron"].get("tau_m_range", [2.0, 30.0])
        tau_e = config["learning"].get("eprop_tau_e", None)
        if tau_e is None:
            tau_e = math.exp((math.log(tau_m_range[0]) + math.log(tau_m_range[1])) / 2)
        self.gamma = math.exp(-dt / tau_e)

        # Readout optimizer (uses autograd)
        self.readout_optimizer = torch.optim.Adam(
            self.model.readout.parameters(), lr=self.lr
        )
        if hasattr(self.model, "embedding"):
            self.embed_optimizer = torch.optim.Adam(
                self.model.embedding.parameters(), lr=self.lr
            )
        else:
            self.embed_optimizer = None

        # Collect recurrent weight parameters and their eligibility traces
        self._setup_traces()

        # Forward hooks for collecting activations
        self._hooks = []
        self._activations: Dict[str, torch.Tensor] = {}

    def _setup_traces(self):
        """Find all recurrent weight matrices and allocate eligibility traces."""
        self.param_traces: List[Tuple[nn.Parameter, EligibilityTraceBuffer, str]] = []
        # We track W_e_raw of all StaticSynapse modules with n_pre_e > 0
        from cortexlm.synapses.static import StaticSynapse
        for name, module in self.model.named_modules():
            if isinstance(module, StaticSynapse):
                if module.n_pre_e > 0:
                    buf = EligibilityTraceBuffer(
                        module.n_pre_e, module.n_post, self.gamma, self.device
                    )
                    self.param_traces.append((module.W_e_raw, buf, "e"))

    def _psi(self, v: torch.Tensor) -> torch.Tensor:
        """Pseudo-derivative for tanh: ψ(v) = 1 - tanh²(v)."""
        return 1.0 - torch.tanh(v) ** 2

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model_state: Optional[ModelState] = None,
    ) -> Tuple[float, ModelState]:
        """Online e-prop step over one sequence."""
        x = x.to(self.device)
        y = y.to(self.device)
        batch, seq_len = x.shape

        if model_state is None:
            model_state = self.model.init_state(batch)

        # Reset trace decay (but not the traces themselves — they persist across tokens)
        # Traces accumulate over the sequence; reset between sequences is optional.
        total_loss = 0.0

        for t in range(seq_len):
            tok_t = x[:, t]
            tgt_t = y[:, t]

            # --- Forward step with autograd only on readout ---
            # Detach state to prevent autograd through recurrent dynamics
            model_state = model_state.detach()

            # Forward through recurrent part (no grad needed here for e-prop)
            with torch.no_grad():
                logits, new_state = self.model.step(tok_t, model_state)

            # Re-run just the readout with grad enabled to get L(t)
            # We need the L5 concat from the new state
            l5_concat = self.model._get_l5_concat(new_state.column_states, self.device, batch)
            l5_concat = l5_concat.detach().requires_grad_(True)

            logits_grad = self.model.readout(l5_concat)
            loss = F.cross_entropy(logits_grad, tgt_t)
            loss.backward()
            total_loss += loss.item()

            # Global learning signal: gradient of loss w.r.t. readout input
            # L(t) = mean(|grad|) as a scalar broadcast signal
            L_t = l5_concat.grad.detach()   # [batch, readout_dim]
            L_scalar = L_t.abs().mean().item()

            # Update readout
            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            self.readout_optimizer.step()
            self.readout_optimizer.zero_grad()

            if self.embed_optimizer is not None:
                self.embed_optimizer.step()
                self.embed_optimizer.zero_grad()

            # --- Update eligibility traces ---
            # Collect all E-population activations and v-states from columns.
            # Use concatenated activities as proxy for pre/post populations.
            # Collect E-population activations from batched column state
            # col_state tensors are [batch, n_cols, n]; flatten across cols
            cs = new_state.column_states
            r_e_tensor = cs.get("r_e", cs.get("r_l23e", None))
            v_e_tensor = cs.get("e_v", cs.get("l23_e_v", None))

            if r_e_tensor is not None and v_e_tensor is not None:
                # [batch, n_cols, n] → [batch, n_cols*n]
                r_concat = r_e_tensor.reshape(r_e_tensor.shape[0], -1).detach()
                v_concat = v_e_tensor.reshape(v_e_tensor.shape[0], -1).detach()
                r_concat_parts = [r_concat]
                v_concat_parts = [v_concat]
            else:
                r_concat_parts = []
                v_concat_parts = []

            if r_concat_parts and v_concat_parts:
                r_concat = r_concat_parts[0]
                v_concat = v_concat_parts[0]
                psi = self._psi(v_concat)  # [batch, total_e]
                # Update each trace with an appropriate slice of activity
                for param, trace_buf, pop_type in self.param_traces:
                    n_pre = trace_buf.trace.shape[1]
                    n_post = trace_buf.trace.shape[0]
                    r_pre_slice = r_concat[:, :n_pre] if n_pre <= r_concat.shape[1] else r_concat
                    psi_slice   = psi[:, :n_post]     if n_post <= psi.shape[1]     else psi
                    trace_buf.update(r_pre_slice, psi_slice)

            # Update recurrent weights using trace × L_scalar
            for param, trace_buf, pop_type in self.param_traces:
                # Apply weight update: Δw = lr * L(t) * ē(t)
                with torch.no_grad():
                    grad_estimate = L_scalar * trace_buf.trace
                    # Clip
                    grad_estimate = grad_estimate.clamp(-self.grad_clip, self.grad_clip)
                    param.data -= self.lr * grad_estimate

            # Enforce Dale's Law
            self._enforce_dale()

            model_state = new_state

        return total_loss / seq_len, model_state

    def _enforce_dale(self):
        for module in self.model.modules():
            if hasattr(module, "enforce_dale"):
                module.enforce_dale()

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        from tqdm import tqdm
        self.model.eval()
        total_loss = 0.0
        n = 0
        for i, (x, y) in enumerate(tqdm(val_loader, total=max_batches, desc="  evaluating",
                                         leave=False, unit="batch")):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            state = self.model.init_state(x.shape[0])
            logits, _ = self.model(x, state)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def train(self, train_loader, val_loader, logger=None):
        """Full training loop (mirrors BPTTTrainer.train)."""
        tcfg = self.config["training"]
        max_steps = tcfg.get("max_steps", 100_000)
        eval_interval = tcfg.get("eval_interval", 500)
        ckpt_interval = tcfg.get("checkpoint_interval", 5000)
        ckpt_dir = tcfg.get("checkpoint_dir", "checkpoints")

        import os
        os.makedirs(ckpt_dir, exist_ok=True)

        persistent_state = None
        step = 0
        train_iter = iter(train_loader)

        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            if persistent_state is None or self.reset_state:
                persistent_state = self.model.init_state(x.shape[0])

            loss, persistent_state = self.train_step(x, y, persistent_state)

            if logger and step % self.config["logging"].get("log_interval", 100) == 0:
                logger.log({
                    "train/loss": loss,
                    "train/perplexity": compute_perplexity(loss),
                }, step=step)

            if step % eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                if logger:
                    logger.log({
                        "val/loss": val_loss,
                        "val/perplexity": compute_perplexity(val_loss),
                    }, step=step)

            if step % ckpt_interval == 0 and step > 0:
                import os
                path = os.path.join(ckpt_dir, f"eprop_step_{step:07d}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }, path)

            step += 1
