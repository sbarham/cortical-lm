"""
e-prop trainer variants (Bellec et al. 2020 / Nature Communications).

Three learning rule implementations, selectable via config learning.rule:

    eprop_approx   Fast approximate e-prop. Scalar global learning signal
                   L(t) = mean|∂L/∂z|. Crude eligibility traces. Good for
                   quick sanity checks and baseline comparisons.

    eprop          Proper e-prop. Vector learning signal L_j(t) = ∂L/∂z_j —
                   one value per L5 neuron — giving each post-synaptic neuron
                   a credit signal proportional to its specific contribution
                   to the output error. Weight update: Δw_ij ∝ L_j * ē_ij.

    eprop_hybrid   Interleaved e-prop + BPTT consolidation. e-prop handles
                   online recurrent updates; BPTT periodically "consolidates"
                   the model. Biological motivation: sleep-wake cycle.
                   See README § Learning rule exploration for discussion.

Config options (all under learning:):
    eprop_tau_e            Eligibility trace timescale (ms). null = auto
                           (geometric mean of tau_m_range). Sweep [5,10,20,50].
    hybrid_eprop_steps     e-prop steps per cycle          (default: 100)
    hybrid_bptt_steps      BPTT consolidation steps/cycle  (default: 10)
    hybrid_bptt_scope      readout_only | full             (default: readout_only)
    hybrid_eprop_variant   eprop | eprop_approx            (default: eprop)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from cortexlm.model import CortexLM, ModelState
from cortexlm.utils.metrics import compute_perplexity


# ── Eligibility trace buffer ──────────────────────────────────────────────────

class EligibilityTraceBuffer:
    """
    Eligibility trace for one weight matrix.
    ē_ij(t) = γ * ē_ij(t-1) + r_pre_i(t) * ψ_post_j(t)
    """
    def __init__(self, n_pre: int, n_post: int, gamma: float, device):
        self.gamma = gamma
        self.trace = torch.zeros(n_post, n_pre, device=device)

    def update(self, r_pre: torch.Tensor, psi_post: torch.Tensor):
        """r_pre: [batch, n_pre]  psi_post: [batch, n_post]"""
        delta = torch.einsum("bi,bj->ji", r_pre, psi_post) / r_pre.shape[0]
        self.trace = self.gamma * self.trace + delta

    def reset(self):
        self.trace.zero_()


# ── Shared base ───────────────────────────────────────────────────────────────

class _EpropBase:
    """
    Shared infrastructure for all e-prop variants.

    Subclasses override train_step(x, y, state) → (loss, new_state).
    The train() loop here handles tokens, logging, eval, and checkpointing.
    """

    def __init__(self, model: CortexLM, config: dict, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        tcfg = config["training"]
        self.lr        = tcfg.get("lr", 3e-4)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.reset_state = config["learning"].get("reset_state_between_batches", False)

        # Trace decay γ = exp(-dt / τ_e)
        dt    = config.get("simulation", {}).get("dt", 1.0)
        tau_m = config["neuron"].get("tau_m_range", [2.0, 30.0])
        tau_e = config["learning"].get("eprop_tau_e", None)
        if tau_e is None:
            tau_e = math.exp((math.log(tau_m[0]) + math.log(tau_m[1])) / 2)
        self.gamma = math.exp(-dt / tau_e)

        # Readout + embedding use autograd throughout
        self.readout_optimizer = torch.optim.Adam(
            self.model.readout.parameters(), lr=self.lr
        )
        self.embed_optimizer = (
            torch.optim.Adam(self.model.embedding.parameters(), lr=self.lr)
            if hasattr(self.model, "embedding") else None
        )

        self.cosine_decay      = config["learning"].get("cosine_decay", False)
        self._current_lr       = self.lr

        # ── Series-2 diagnostic / fix flags ──────────────────────────────────
        # freeze_recurrent: skip all recurrent weight updates entirely.
        #   Diagnostic: if val divergence stops, the recurrent updates are the cause.
        self.freeze_recurrent  = config["learning"].get("freeze_recurrent", False)

        # adam_recurrent: use Adam (with moment estimates) for recurrent weight
        #   updates instead of direct param.data -= lr * g.  Gives adaptive per-
        #   parameter scaling, matching the readout/embedding optimizers.
        self.adam_recurrent    = config["learning"].get("adam_recurrent", False)

        # dale_interval: call enforce_dale() only every N inner timesteps instead
        #   of after every single timestep.  Reduces the chance that Dale clipping
        #   immediately undoes each e-prop update before it can accumulate.
        self.dale_interval     = config["learning"].get("dale_interval", 1)
        self._dale_t           = 0   # inner-timestep counter

        self._setup_traces()
        self._global_step = 0   # used by hybrid trainer for cycle tracking

    def _setup_traces(self):
        """Allocate eligibility traces (and Adam state if adam_recurrent) for all StaticSynapse modules."""
        from cortexlm.synapses.static import StaticSynapse
        self.param_traces: List[Tuple[nn.Parameter, EligibilityTraceBuffer]] = []
        for _, module in self.model.named_modules():
            if isinstance(module, StaticSynapse) and module.n_pre_e > 0:
                buf = EligibilityTraceBuffer(
                    module.n_pre_e, module.n_post, self.gamma, self.device
                )
                self.param_traces.append((module.W_e_raw, buf))

        # Adam moment buffers for recurrent weights (only allocated if needed)
        if getattr(self, "adam_recurrent", False):
            self._rec_m = {id(p): torch.zeros_like(p) for p, _ in self.param_traces}
            self._rec_v = {id(p): torch.zeros_like(p) for p, _ in self.param_traces}
            self._rec_t = {id(p): 0                   for p, _ in self.param_traces}

    def _psi(self, v: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.tanh(v) ** 2

    def _enforce_dale(self):
        for module in self.model.modules():
            if hasattr(module, "enforce_dale"):
                module.enforce_dale()

    def _maybe_enforce_dale(self):
        """Call enforce_dale() only every dale_interval inner timesteps."""
        self._dale_t += 1
        if self._dale_t % self.dale_interval == 0:
            self._enforce_dale()

    def _apply_recurrent_update(self, param: nn.Parameter, g: torch.Tensor):
        """Apply one recurrent weight update — direct SGD or Adam depending on config."""
        if self.adam_recurrent:
            pid = id(param)
            self._rec_t[pid] += 1
            self._rec_m[pid] = 0.9  * self._rec_m[pid] + 0.1  * g
            self._rec_v[pid] = 0.999 * self._rec_v[pid] + 0.001 * g.pow(2)
            m_hat = self._rec_m[pid] / (1 - 0.9  ** self._rec_t[pid])
            v_hat = self._rec_v[pid] / (1 - 0.999 ** self._rec_t[pid])
            param.data -= self._current_lr * m_hat / (v_hat.sqrt() + 1e-8)
        else:
            param.data -= self._current_lr * g

    def _update_traces(self, new_state: ModelState):
        """Update eligibility traces from column state activations."""
        cs = new_state.column_states
        r_e = cs.get("r_e", cs.get("r_l23e", None))
        v_e = cs.get("e_v", cs.get("l23_e_v", None))
        if r_e is None or v_e is None:
            return
        r_concat = r_e.reshape(r_e.shape[0], -1).detach()
        psi = self._psi(v_e.reshape(v_e.shape[0], -1).detach())
        for _, buf in self.param_traces:
            n_pre  = buf.trace.shape[1]
            n_post = buf.trace.shape[0]
            r_slice   = r_concat[:, :n_pre]  if n_pre  <= r_concat.shape[1] else r_concat
            psi_slice = psi[:, :n_post]      if n_post <= psi.shape[1]      else psi
            buf.update(r_slice, psi_slice)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, state: Optional[ModelState]
    ) -> Tuple[float, ModelState]:
        raise NotImplementedError

    # ── Evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        from tqdm import tqdm
        self.model.eval()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(tqdm(val_loader, total=max_batches,
                                         desc="  evaluating", leave=False)):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x, self.model.init_state(x.shape[0]))
            total += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, logger=None):
        tcfg   = self.config["training"]
        lcfg   = self.config["logging"]
        max_steps  = tcfg.get("max_steps", 100_000)
        ckpt_dir   = tcfg.get("checkpoint_dir", "checkpoints")
        seq_len    = self.config["data"].get("seq_len", 128)

        # Eval interval: token-based if available, else ~20 points across run
        tokens_per_step = tcfg.get("batch_size", 512) * seq_len
        eval_tokens = tcfg.get("eval_tokens", None)
        if eval_tokens is not None:
            eval_interval = max(1, eval_tokens // tokens_per_step)
        else:
            eval_interval = tcfg.get("eval_interval", max(1, max_steps // 20))

        # e-prop steps are cheap — log train at the same cadence as eval so charts align.
        # (BPTT's logging.log_interval: 100 is far too sparse for e-prop's step count.)
        log_interval = eval_interval
        ckpt_interval = tcfg.get("checkpoint_interval", max_steps)

        import os
        os.makedirs(ckpt_dir, exist_ok=True)

        state       = None
        step        = 0
        tokens_seen = 0
        train_iter  = iter(train_loader)

        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            if state is None or self.reset_state:
                state = self.model.init_state(x.shape[0])

            self._global_step = step

            # Cosine LR annealing — update optimizers and direct-update LR
            if self.cosine_decay:
                self._current_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * step / max_steps))
                for opt in (self.readout_optimizer, self.embed_optimizer):
                    if opt is not None:
                        for pg in opt.param_groups:
                            pg["lr"] = self._current_lr

            loss, state = self.train_step(x, y, state)
            tokens_seen += x.shape[0] * seq_len

            if logger and step % log_interval == 0:
                logger.log({
                    "train/loss":       loss,
                    "train/perplexity": compute_perplexity(loss),
                    "tokens":           tokens_seen,
                }, step=step)

            if step % eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                if logger:
                    logger.log({
                        "val/loss":       val_loss,
                        "val/perplexity": compute_perplexity(val_loss),
                        "tokens":         tokens_seen + 1,
                    }, step=step)

            if step % ckpt_interval == 0 and step > 0:
                path = os.path.join(ckpt_dir, f"eprop_step_{step:07d}.pt")
                torch.save({
                    "step": step, "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                }, path)

            step += 1


# ── Approximate e-prop ────────────────────────────────────────────────────────

class EpropApproxTrainer(_EpropBase):
    """
    Fast approximate e-prop.

    Learning signal: scalar L(t) = mean|∂loss/∂z| broadcast to all synapses.
    Eligibility traces: crude concat-and-slice (pre/post identity ignored).

    Use for quick sanity checks. Not faithful to the Bellec et al. formulation.
    """

    def train_step(self, x, y, state):
        x, y = x.to(self.device), y.to(self.device)
        batch, seq_len = x.shape
        if state is None:
            state = self.model.init_state(batch)
        total_loss = 0.0

        for t in range(seq_len):
            state = state.detach()
            with torch.no_grad():
                _, new_state = self.model.step(x[:, t], state)

            ri = self.model._last_readout_input.detach().requires_grad_(True)
            logits_grad = self.model.readout(ri)
            loss = F.cross_entropy(logits_grad, y[:, t])
            loss.backward()
            total_loss += loss.item()

            L_scalar = ri.grad.abs().mean().item()

            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.step(); self.embed_optimizer.zero_grad()

            if not self.freeze_recurrent:
                self._update_traces(new_state)
                with torch.no_grad():
                    for param, buf in self.param_traces:
                        g = (L_scalar * buf.trace).clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                self._maybe_enforce_dale()

            state = new_state

        return total_loss / seq_len, state


# ── Proper e-prop ─────────────────────────────────────────────────────────────

class EpropTrainer(_EpropBase):
    """
    Proper e-prop (Bellec et al. 2020).

    Learning signal: vector L_j(t) = ∂loss/∂z_j — one entry per readout-input
    neuron (L5 population). Each post-synaptic neuron j receives a credit
    signal proportional to its specific contribution to the output error.

    Weight update: Δw_ij = -lr * (1/B) Σ_b L_j^b(t) * ē_ij(t)
                         = -lr * L_mean_j * ē_ij(t)

    This preserves the directional information that the approx variant throws
    away, giving a much richer per-neuron credit assignment signal.
    """

    def train_step(self, x, y, state):
        x, y = x.to(self.device), y.to(self.device)
        batch, seq_len = x.shape
        if state is None:
            state = self.model.init_state(batch)
        total_loss = 0.0

        for t in range(seq_len):
            state = state.detach()
            with torch.no_grad():
                _, new_state = self.model.step(x[:, t], state)

            # Vector learning signal: gradient of loss w.r.t. each L5 neuron
            ri = self.model._last_readout_input.detach().requires_grad_(True)
            logits_grad = self.model.readout(ri)
            loss = F.cross_entropy(logits_grad, y[:, t])
            loss.backward()
            total_loss += loss.item()

            # L_j(t): mean over batch → [readout_dim]
            L_vec = ri.grad.detach().mean(dim=0)   # [readout_dim]

            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.step(); self.embed_optimizer.zero_grad()

            if not self.freeze_recurrent:
                self._update_traces(new_state)
                with torch.no_grad():
                    for param, buf in self.param_traces:
                        n_post = buf.trace.shape[0]
                        L_post = L_vec[:n_post] if n_post <= L_vec.shape[0] else L_vec
                        g = (L_post.unsqueeze(1) * buf.trace).clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                self._maybe_enforce_dale()

            state = new_state

        return total_loss / seq_len, state


# ── Hybrid e-prop / BPTT ──────────────────────────────────────────────────────

class EpropHybridTrainer(_EpropBase):
    """
    Interleaved e-prop + BPTT consolidation (sleep-wake learning cycle).

    Biological motivation
    ---------------------
    Complementary Learning Systems (McClelland et al. 1995) posits that the
    brain requires two learning phases:

        Awake (online)   — fast, local, approximate synaptic updates driven
                           by immediate experience. Implemented here as e-prop.

        Sleep (offline)  — slow, global consolidation via hippocampal replay.
                           Sharp-wave ripple events replay recent experiences
                           to cortex, allowing BPTT-like credit assignment over
                           longer time horizons. Implemented here as brief BPTT
                           "bursts" every hybrid_eprop_steps steps.

    The hybrid avoids two failure modes:
        • Pure e-prop: fast but noisy, prone to plateaus (gradient signal is
          approximate and may not escape poor basins).
        • Pure BPTT: accurate but slow, biologically implausible, requires
          storing full activation history.

    The interleaving lets e-prop do the bulk of the work cheaply, while BPTT
    bursts periodically correct the readout and (optionally) recurrent weights.

    Config options (under learning:)
    ---------------------------------
    hybrid_eprop_variant   eprop | eprop_approx         (default: eprop)
    hybrid_eprop_steps     e-prop steps per cycle        (default: 100)
    hybrid_bptt_steps      BPTT consolidation steps      (default: 10)
    hybrid_bptt_scope      readout_only | full           (default: readout_only)
        readout_only — BPTT updates only readout + embedding (fast; recurrent
                       weights remain local). Analogous to decision-layer
                       adaptation via direct cortical feedback.
        full         — BPTT updates all parameters (slower; true replay).
                       Analogous to hippocampal sharp-wave ripple consolidation.
    """

    def __init__(self, model: CortexLM, config: dict, device=None):
        super().__init__(model, config, device)
        lcfg = config["learning"]

        self.eprop_steps  = lcfg.get("hybrid_eprop_steps", 100)
        self.bptt_steps   = lcfg.get("hybrid_bptt_steps", 10)
        self.bptt_scope   = lcfg.get("hybrid_bptt_scope", "readout_only")
        variant           = lcfg.get("hybrid_eprop_variant", "eprop")

        # Inner e-prop trainer (shares model + traces with this instance)
        if variant == "eprop_approx":
            self._eprop = EpropApproxTrainer.__new__(EpropApproxTrainer)
        else:
            self._eprop = EpropTrainer.__new__(EpropTrainer)
        # Share all state — don't re-init, just bind
        self._eprop.__dict__ = self.__dict__

        # Full-model optimizer for BPTT consolidation steps
        self.full_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

    @property
    def _cycle_len(self):
        return self.eprop_steps + self.bptt_steps

    def _in_bptt_phase(self) -> bool:
        return (self._global_step % self._cycle_len) >= self.eprop_steps

    def train_step(self, x, y, state):
        if self._in_bptt_phase():
            return self._bptt_consolidation_step(x, y, state)
        else:
            return self._eprop.train_step(x, y, state)

    def _bptt_consolidation_step(self, x, y, state):
        x, y = x.to(self.device), y.to(self.device)
        state = state.detach()

        if self.bptt_scope == "full":
            # Full BPTT — update all parameters
            self.full_optimizer.zero_grad()
            logits, new_state = self.model(x, state)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.full_optimizer.step()
            return loss.item(), new_state

        else:
            # readout_only — collect L5 activations with frozen recurrent pass,
            # then update readout + embedding via autograd.
            l5_acts = []
            cur_state = state
            with torch.no_grad():
                for t in range(x.shape[1]):
                    _, cur_state = self.model.step(x[:, t], cur_state)
                    l5_acts.append(self.model._last_readout_input.clone())

            l5_stack = torch.stack(l5_acts, dim=1)       # [B, T, rdim]
            B, T, rdim = l5_stack.shape
            l5_flat = l5_stack.reshape(B * T, rdim)

            self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.zero_grad()

            logits = self.model.readout(l5_flat)
            loss = F.cross_entropy(logits, y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            self.readout_optimizer.step()
            if self.embed_optimizer:
                self.embed_optimizer.step()

            return loss.item(), cur_state
