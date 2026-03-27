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

    Supports both the non-batched case (single column, n_cols=1) and the
    batched case (n_cols > 1, one trace per column).

    trace shape: [n_cols, n_post, n_pre]
    grad property: returns trace squeezed to match W_e_raw.shape
      - n_cols=1  → [n_post, n_pre]
      - n_cols>1  → [n_cols, n_post, n_pre]
    """
    def __init__(self, n_pre: int, n_post: int, gamma: float, device, n_cols: int = 1):
        self.gamma  = gamma
        self.n_cols = n_cols
        self.trace  = torch.zeros(n_cols, n_post, n_pre, device=device)

    def update(self, r_pre: torch.Tensor, psi_post: torch.Tensor):
        """
        Non-batched: r_pre [B, n_pre],        psi_post [B, n_post]
        Batched:     r_pre [B, n_cols, n_pre], psi_post [B, n_cols, n_post]
        """
        B = r_pre.shape[0]
        if r_pre.dim() == 2:
            # [1, n_post, n_pre]
            delta = torch.einsum("bp,bq->qp", r_pre, psi_post).unsqueeze(0) / B
        else:
            # [n_cols, n_post, n_pre]
            delta = torch.einsum("bcp,bcq->cqp", r_pre, psi_post) / B
        self.trace = self.gamma * self.trace + delta

    @property
    def grad(self) -> torch.Tensor:
        """Trace in param-compatible shape: [n_post, n_pre] or [n_cols, n_post, n_pre]."""
        return self.trace[0] if self.n_cols == 1 else self.trace

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

        self.cosine_decay        = config["learning"].get("cosine_decay", False)
        self._current_lr         = self.lr
        # normalize_l_signal: rescale L_vec to unit mean-abs before applying to traces.
        # Prevents the learning signal from dying as the readout converges.
        self.normalize_l_signal  = config["learning"].get("normalize_l_signal", False)

        # ── Series-2 diagnostic / fix flags ──────────────────────────────────
        # freeze_recurrent: skip all recurrent weight updates entirely.
        #   Diagnostic: if val divergence stops, the recurrent updates are the cause.
        self.freeze_recurrent  = config["learning"].get("freeze_recurrent", False)

        # freeze_readout: skip readout + embedding optimizer steps entirely.
        #   Diagnostic: if val divergence stops, the autograd readout path is the cause.
        #   Recurrent weights still updated via e-prop traces.
        self.freeze_readout    = config["learning"].get("freeze_readout", False)

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

        # Diagnostic tracking — updated each train_step, logged at log_interval
        self._last_l_signal   = 0.0  # mean |learning signal| over last step's timesteps
        self._last_update_mag = 0.0  # mean |L × trace| gradient applied to recurrent weights

    def _setup_traces(self):
        """
        Allocate eligibility traces for all annotated synapse modules.

        Handles both StaticSynapse (single column) and BatchedStaticSynapse
        (multi-column).  Only modules that have eprop_pre_key / eprop_post_v_key
        set are included — these are annotated in BatchedLayeredColumns.__init__.

        param_traces entries: (param, buf, pre_key, post_v_key)
        """
        from cortexlm.synapses.static import StaticSynapse, BatchedStaticSynapse
        self.param_traces: List[Tuple] = []

        for _, module in self.model.named_modules():
            if not (hasattr(module, "eprop_pre_key") and hasattr(module, "eprop_post_v_key")):
                continue
            if isinstance(module, BatchedStaticSynapse) and module.n_pre_e > 0:
                buf = EligibilityTraceBuffer(
                    module.n_pre_e, module.n_post, self.gamma, self.device,
                    n_cols=module.n_cols,
                )
                self.param_traces.append(
                    (module.W_e_raw, buf, module.eprop_pre_key, module.eprop_post_v_key)
                )
            elif isinstance(module, StaticSynapse) and module.n_pre_e > 0:
                buf = EligibilityTraceBuffer(
                    module.n_pre_e, module.n_post, self.gamma, self.device,
                )
                self.param_traces.append(
                    (module.W_e_raw, buf, module.eprop_pre_key, module.eprop_post_v_key)
                )

        # Adam moment buffers for recurrent weights (only allocated if needed)
        if getattr(self, "adam_recurrent", False):
            self._rec_m = {id(p): torch.zeros_like(p) for p, _, _, _ in self.param_traces}
            self._rec_v = {id(p): torch.zeros_like(p) for p, _, _, _ in self.param_traces}
            self._rec_t = {id(p): 0                   for p, _, _, _ in self.param_traces}

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
        """
        Update eligibility traces using the correct pre/post population per synapse.

        Each param_traces entry carries (param, buf, pre_key, post_v_key) so we
        look up the exact firing-rate and voltage tensors for that connection.
        State tensors are [batch, n_cols, n] for batched columns.
        """
        cs = new_state.column_states
        for _, buf, pre_key, post_v_key in self.param_traces:
            r_pre = cs.get(pre_key)
            v_post = cs.get(post_v_key)
            if r_pre is None or v_post is None:
                continue
            buf.update(r_pre.detach(), self._psi(v_post.detach()))

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

        # Eval interval: token-based if available, else same cadence as train logging.
        tokens_per_step = tcfg.get("batch_size", 512) * seq_len
        log_tokens  = tcfg.get("log_tokens",  None)
        eval_tokens = tcfg.get("eval_tokens", log_tokens)  # default: same as log cadence
        if eval_tokens is not None:
            eval_interval = max(1, eval_tokens // tokens_per_step)
            log_interval  = max(1, (log_tokens or eval_tokens) // tokens_per_step)
        else:
            eval_interval = tcfg.get("eval_interval", max(1, max_steps // 20))
            log_interval  = eval_interval
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
                for _, buf, _, _ in self.param_traces:
                    buf.reset()

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
                log_dict = {
                    "train/loss":       loss,
                    "train/perplexity": compute_perplexity(loss),
                    "tokens":           tokens_seen,
                }
                if self.param_traces:
                    with torch.no_grad():
                        trace_norms = [buf.grad.abs().mean().item()
                                       for _, buf, _, _ in self.param_traces]
                    log_dict["eprop/trace_norm_mean"] = sum(trace_norms) / len(trace_norms)
                    log_dict["eprop/l_signal"]        = self._last_l_signal
                    log_dict["eprop/update_mag"]      = self._last_update_mag
                logger.log(log_dict, step=step)

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
    Eligibility traces: correct per-synapse pre/post identity (same as EpropTrainer).

    Use for quick sanity checks. Not faithful to the Bellec et al. formulation.
    """

    def train_step(self, x, y, state):
        x, y = x.to(self.device), y.to(self.device)
        batch, seq_len = x.shape
        if state is None:
            state = self.model.init_state(batch)
        total_loss = 0.0
        l_acc, update_acc, update_n = 0.0, 0.0, 0

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
            l_acc += L_scalar
            if self.normalize_l_signal:
                L_scalar = 1.0

            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            if not self.freeze_readout:
                self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
                if self.embed_optimizer:
                    self.embed_optimizer.step(); self.embed_optimizer.zero_grad()
            else:
                self.readout_optimizer.zero_grad()
                if self.embed_optimizer:
                    self.embed_optimizer.zero_grad()

            if not self.freeze_recurrent:
                self._update_traces(new_state)
                with torch.no_grad():
                    for param, buf, _, _ in self.param_traces:
                        g = (L_scalar * buf.grad).clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                        update_acc += g.abs().mean().item()
                        update_n   += 1
                self._maybe_enforce_dale()

            state = new_state

        self._last_l_signal   = l_acc / seq_len
        self._last_update_mag = update_acc / max(update_n, 1)
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
        l_acc, update_acc, update_n = 0.0, 0.0, 0

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
            l_acc += ri.grad.detach().abs().mean().item()  # log pre-average magnitude
            if self.normalize_l_signal:
                L_vec = L_vec / (L_vec.abs().mean() + 1e-8)

            nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
            if not self.freeze_readout:
                self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
                if self.embed_optimizer:
                    self.embed_optimizer.step(); self.embed_optimizer.zero_grad()
            else:
                self.readout_optimizer.zero_grad()
                if self.embed_optimizer:
                    self.embed_optimizer.zero_grad()

            if not self.freeze_recurrent:
                self._update_traces(new_state)
                with torch.no_grad():
                    # For L5E post-synaptic neurons, use the per-neuron vector learning
                    # signal L_j = ∂loss/∂z_j reshaped to [n_cols, n_l5e].
                    # For all other post populations, fall back to scalar approximation.
                    n_l5e  = getattr(self.model, "n_l5e", None)
                    n_cols = getattr(self.model, "n_columns", 1)
                    L_mat  = None
                    if n_l5e is not None and n_cols > 1:
                        try:
                            L_mat = L_vec.reshape(n_cols, n_l5e)   # [n_cols, n_l5e]
                        except RuntimeError:
                            pass

                    for param, buf, _, post_v_key in self.param_traces:
                        trace = buf.grad  # [n_post, n_pre] or [n_cols, n_post, n_pre]
                        if L_mat is not None and post_v_key == "l5_e_v" and buf.n_cols > 1:
                            # Per-column, per-neuron credit: [n_cols, n_l5e, n_pre]
                            g = torch.einsum("cn,cnp->cnp", L_mat, trace)
                        elif trace.dim() == 3:
                            # Batched non-L5E post: scalar approximation per column
                            g = L_vec.abs().mean() * trace
                        else:
                            # Non-batched fallback (original behaviour)
                            n_post = trace.shape[0]
                            L_post = L_vec[:n_post] if n_post <= L_vec.shape[0] else L_vec
                            g = L_post.unsqueeze(1) * trace
                        g = g.clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                        update_acc += g.abs().mean().item()
                        update_n   += 1
                self._maybe_enforce_dale()

            state = new_state

        self._last_l_signal   = l_acc / seq_len
        self._last_update_mag = update_acc / max(update_n, 1)
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

        # Adaptive BPTT: trigger consolidation based on plateau detection rather
        # than a fixed cycle.  Enabled when hybrid_adaptive=true.
        #   plateau_window  — steps over which to measure improvement (EMA half-life)
        #   plateau_thresh  — fractional improvement below which BPTT triggers
        #   plateau_cooldown — minimum steps between BPTT bursts
        self.adaptive          = lcfg.get("hybrid_adaptive", False)
        self.plateau_window    = lcfg.get("hybrid_plateau_window", 500)
        self.plateau_thresh    = lcfg.get("hybrid_plateau_thresh", 0.005)  # 0.5% improvement
        self.plateau_cooldown  = lcfg.get("hybrid_plateau_cooldown", 200)
        self._loss_ema         = None   # initialised on first step
        self._loss_ema_prev    = None   # EMA snapshot taken plateau_window steps ago
        self._steps_since_bptt = 0
        self._bptt_steps_remaining = 0  # countdown during an adaptive burst

        # Inner e-prop trainer (shares model + traces with this instance)
        if variant == "eprop_approx":
            self._eprop = EpropApproxTrainer.__new__(EpropApproxTrainer)
        else:
            self._eprop = EpropTrainer.__new__(EpropTrainer)
        # Share all state — don't re-init, just bind
        self._eprop.__dict__ = self.__dict__

        # Full-model optimizer for BPTT consolidation steps.
        # Uses a separate LR so consolidation doesn't overwrite e-prop's incremental
        # updates.  Default: same as e-prop LR; override with hybrid_bptt_lr.
        bptt_lr = lcfg.get("hybrid_bptt_lr", self.lr)
        self.full_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=bptt_lr,
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

    @property
    def _cycle_len(self):
        return self.eprop_steps + self.bptt_steps

    def _in_bptt_phase(self) -> bool:
        return (self._global_step % self._cycle_len) >= self.eprop_steps

    def _update_plateau_detection(self, loss: float) -> bool:
        """
        Update EMA and return True if a BPTT burst should be triggered.
        Uses two EMAs separated by plateau_window steps to measure improvement rate.
        """
        alpha = 2.0 / (self.plateau_window + 1)
        if self._loss_ema is None:
            self._loss_ema = loss
            self._loss_ema_prev = loss
            return False
        self._loss_ema = alpha * loss + (1 - alpha) * self._loss_ema

        # Snapshot the slow EMA every plateau_window steps for comparison
        if self._global_step % self.plateau_window == 0:
            improvement = (self._loss_ema_prev - self._loss_ema) / (self._loss_ema_prev + 1e-8)
            self._loss_ema_prev = self._loss_ema
            if improvement < self.plateau_thresh and self._steps_since_bptt >= self.plateau_cooldown:
                return True  # plateau detected
        return False

    def train_step(self, x, y, state):
        if self.adaptive:
            return self._adaptive_train_step(x, y, state)
        if self._in_bptt_phase():
            return self._bptt_consolidation_step(x, y, state)
        else:
            return self._eprop.train_step(x, y, state)

    def _adaptive_train_step(self, x, y, state):
        """Adaptive hybrid: run e-prop normally, trigger BPTT bursts on plateau."""
        if self._bptt_steps_remaining > 0:
            self._bptt_steps_remaining -= 1
            self._steps_since_bptt = 0
            loss, state = self._bptt_consolidation_step(x, y, state)
        else:
            loss, state = self._eprop.train_step(x, y, state)
            self._steps_since_bptt += 1
            if self._update_plateau_detection(loss):
                self._bptt_steps_remaining = self.bptt_steps  # start a burst
        return loss, state

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
            # Traces are now stale (weights changed) — reset so e-prop resumes cleanly.
            for _, buf, _, _ in self.param_traces:
                buf.reset()
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
            # Reset traces — readout shift changes the l_signal landscape so old
            # traces are mismatched to the new gradient direction.
            for _, buf, _, _ in self.param_traces:
                buf.reset()

            return loss.item(), cur_state
