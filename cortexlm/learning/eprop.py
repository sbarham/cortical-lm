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
from cortexlm.utils.metrics import compute_perplexity, compute_effective_timescales


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

        self.eprop_mode = config["learning"].get("eprop_mode", "sequential")

        self._setup_traces()
        self._global_step      = 0     # used by hybrid trainer for cycle tracking
        self._current_sgdr_cycle = 0   # which SGDR restart cycle we're in (0-indexed)
        self._sgdr_t0_steps    = None  # exposed so hybrid trainer can compute cycle idx

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

    def _collect_forward_activations(
        self, x: torch.Tensor, state: Optional[ModelState]
    ) -> Tuple[list, list, "ModelState"]:
        """
        Run the full sequence forward with no_grad, collecting per-timestep states
        and readout inputs.  Eliminates per-token CUDA kernel launch overhead from
        the training loop.

        Returns:
            states    — list of T ModelStates (one per timestep)
            ri_list   — list of T [B, rdim] readout-input tensors (detached clones)
            state     — final ModelState (for carrying across batches)
        """
        T = x.shape[1]
        states, ri_list = [], []
        with torch.no_grad():
            for t in range(T):
                state = state.detach()
                _, new_state = self.model.step(x[:, t], state)
                states.append(new_state)
                # clone: _last_readout_input is overwritten each step
                ri_list.append(self.model._last_readout_input.detach().clone())
                state = new_state
        return states, ri_list, state

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, state: Optional[ModelState]
    ) -> Tuple[float, ModelState]:
        raise NotImplementedError

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_hopfield_stats(self) -> dict:
        hpc = getattr(self.model, "hippocampus", None)
        if hpc is None:
            return {}
        weights = getattr(hpc, "_last_attn_weights", None)
        if weights is None:
            return {}
        eps = 1e-10
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean().item()
        attn_max = weights.max(dim=-1).values.mean().item()
        stats = {"hpc/attn_entropy": entropy, "hpc/attn_max": attn_max}
        surprise = getattr(self.model, "_last_hpc_surprise", None)
        if surprise is not None:
            stats["hpc/ca1_surprise"] = surprise
        return stats

    @torch.no_grad()
    def _collect_tau_stats(self, val_loader) -> dict:
        import numpy as np
        self.model.eval()
        try:
            x, _ = next(iter(val_loader))
        except StopIteration:
            return {}
        x = x[:4].to(self.device)
        batch, seq_len = x.shape
        state = self.model.init_state(batch)

        traces_l4e, traces_l23e, traces_l5e, traces_l6e = [], [], [], []
        for t in range(seq_len):
            _, state = self.model.step(x[:, t], state)
            cs = state.column_states
            if "r_l23e" in cs:
                traces_l4e.append( cs["r_l4e" ].mean(dim=(0, 1)).cpu().numpy())
                traces_l23e.append(cs["r_l23e"].mean(dim=(0, 1)).cpu().numpy())
                traces_l5e.append( cs["r_l5e" ].mean(dim=(0, 1)).cpu().numpy())
                traces_l6e.append( cs["r_l6e" ].mean(dim=(0, 1)).cpu().numpy())

        self.model.train()
        if not traces_l23e:
            return {}

        arr_l4e  = np.stack(traces_l4e)
        arr_l23e = np.stack(traces_l23e)
        arr_l5e  = np.stack(traces_l5e)
        arr_l6e  = np.stack(traces_l6e)

        stats = {}
        for key, arr in [("tau/l4e", arr_l4e), ("tau/l23e", arr_l23e),
                         ("tau/l5e", arr_l5e),  ("tau/l6e",  arr_l6e)]:
            taus = compute_effective_timescales(
                torch.from_numpy(arr), max_lag=min(50, seq_len // 4)
            )
            stats[f"{key}_mean"] = float(np.mean(taus))
            stats[f"{key}_std"]  = float(np.std(taus))
            stats[f"{key}_p25"]  = float(np.percentile(taus, 25))
            stats[f"{key}_p75"]  = float(np.percentile(taus, 75))
        return stats

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

    def train(self, train_loader, val_loader, logger=None, start_step: int = 0):
        tcfg   = self.config["training"]
        lcfg   = self.config["logging"]
        no_repeat  = tcfg.get("no_repeat", False)
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

        # SGDR setup: restarts every sgdr_restart_tokens tokens
        _sgdr_restart_tokens = self.config["learning"].get("sgdr_restart_tokens", None)
        _sgdr_t0_steps = None
        if _sgdr_restart_tokens is not None:
            _sgdr_t0_steps = max(1, _sgdr_restart_tokens // tokens_per_step)

        # Standalone phase trigger: advance DAWN phase schedule at a token interval
        # independently of the LR schedule (enables flat-LR DAWN ablation).
        _phase_trigger_tokens = self.config["learning"].get("hybrid_phase_trigger_tokens", None)
        _phase_t0_steps = (max(1, _phase_trigger_tokens // tokens_per_step)
                           if _phase_trigger_tokens is not None else None)

        # HPC beta annealing setup
        _hpc = getattr(self.model, "hippocampus", None)
        _hpc_beta_anneal = (
            _hpc is not None
            and hasattr(_hpc, "update_beta")
            and getattr(_hpc, "beta_init", None) is not None
        )

        import os
        os.makedirs(ckpt_dir, exist_ok=True)

        # Profiling: if profile_steps > 0, trace the first N steps and save to disk
        profile_steps = tcfg.get("profile_steps", 0)
        _prof = None
        if profile_steps > 0:
            prof_dir = os.path.join(ckpt_dir, "profile")
            os.makedirs(prof_dir, exist_ok=True)
            _prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_steps),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
                record_shapes=True,
                with_stack=False,
            )
            _prof.start()
            print(f"  profiling {profile_steps} steps → {prof_dir}")

        import time as _time
        _t_start = _time.time()

        state       = None
        step        = start_step
        tokens_seen = start_step * tokens_per_step
        _last_logged_sgdr_cycle   = (start_step // _sgdr_t0_steps) - 1 if _sgdr_t0_steps else -1
        _last_logged_phase_cycle  = (start_step // _phase_t0_steps) - 1 if _phase_t0_steps else -1
        train_iter  = iter(train_loader)
        # Expose iterator + loader so subclasses (e.g. hybrid BPTT phase) can pull
        # additional batches independently of the main e-prop loop.
        self._train_iter   = train_iter
        self._train_loader = train_loader

        # Fast-forward data iterator past already-seen steps.
        # Data order won't be identical (shuffler state is lost), but this ensures
        # the model doesn't re-train on the same epoch-start data every resume.
        if start_step > 0:
            print(f"  resuming from step {start_step:,} ({tokens_seen:,} tokens seen) — "
                  f"fast-forwarding data loader...")
            for _skip in range(start_step):
                try:
                    next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    self._train_iter = train_iter
                    next(train_iter)
            # Pre-initialize SGDR cycle and phase schedule so the first train_step
            # sees the right ratio without waiting for the loop to reach that step.
            _resume_t0 = _sgdr_t0_steps or _phase_t0_steps
            if _resume_t0 is not None:
                self._sgdr_t0_steps      = _resume_t0
                self._current_sgdr_cycle = start_step // _resume_t0
                if hasattr(self, "_update_phase_from_schedule"):
                    self._update_phase_from_schedule()
            print(f"  fast-forward complete"
                  + (f" — SGDR cycle {self._current_sgdr_cycle}" if _sgdr_t0_steps else "")
                  + (f", phase {self.eprop_steps}:{self.bptt_steps}"
                     if hasattr(self, "eprop_steps") else ""))

        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                if no_repeat:
                    print(f"\n  [no_repeat] dataset exhausted at step {step:,} "
                          f"({tokens_seen/1e6:.1f}M tokens) — stopping.")
                    break
                train_iter = iter(train_loader)
                self._train_iter = train_iter
                x, y = next(train_iter)

            if state is None or self.reset_state:
                state = self.model.init_state(x.shape[0])
                for _, buf, _, _ in self.param_traces:
                    buf.reset()

            self._global_step = step

            # HPC beta annealing
            if _hpc_beta_anneal:
                _hpc.update_beta(step, max_steps)

            # Standalone phase trigger (no SGDR): advance cycle index from token count
            # so _update_phase_from_schedule fires at the right moments even with flat LR.
            if _phase_t0_steps is not None and _sgdr_t0_steps is None:
                self._sgdr_t0_steps      = _phase_t0_steps
                _new_cycle = step // _phase_t0_steps
                self._current_sgdr_cycle = _new_cycle
                if _new_cycle > _last_logged_phase_cycle:
                    _last_logged_phase_cycle = _new_cycle
                    print(f"\n  [phase trigger] cycle {_new_cycle} at step {step:,} "
                          f"({tokens_seen / 1e6:.1f}M tokens)")

            # LR schedule: SGDR takes priority over cosine_decay
            if _sgdr_t0_steps is not None:
                self._sgdr_t0_steps    = _sgdr_t0_steps
                _new_cycle = step // _sgdr_t0_steps
                if _new_cycle > _last_logged_sgdr_cycle:
                    _last_logged_sgdr_cycle = _new_cycle
                    print(f"\n  [SGDR reset] cycle {_new_cycle} at step {step:,} "
                          f"({tokens_seen / 1e6:.1f}M tokens), lr → {self.lr:.2e}")
                self._current_sgdr_cycle = _new_cycle
                cycle_pos = step % _sgdr_t0_steps
                self._current_lr = self.lr * 0.5 * (
                    1.0 + math.cos(math.pi * cycle_pos / _sgdr_t0_steps)
                )
                for opt in (self.readout_optimizer, self.embed_optimizer,
                            getattr(self, "full_optimizer", None)):
                    if opt is not None:
                        for pg in opt.param_groups:
                            pg["lr"] = self._current_lr
            elif self.cosine_decay:
                self._current_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * step / max_steps))
                for opt in (self.readout_optimizer, self.embed_optimizer):
                    if opt is not None:
                        for pg in opt.param_groups:
                            pg["lr"] = self._current_lr

            loss, state = self.train_step(x, y, state)
            tokens_seen += x.shape[0] * seq_len

            if _prof is not None:
                _prof.step()
                if step >= profile_steps + 1:   # wait=1 warmup=1 active=N → done at 2+N
                    _prof.stop()
                    _prof = None
                    print(f"  profiling complete — view with: tensorboard --logdir {prof_dir}")

            if logger and step % log_interval == 0:
                log_dict = {
                    "train/loss":       loss,
                    "train/perplexity": compute_perplexity(loss),
                    "tokens":           tokens_seen,
                    "lr":               self._current_lr,
                    "elapsed_min":      (_time.time() - _t_start) / 60.0,
                }
                if self.param_traces:
                    with torch.no_grad():
                        trace_norms = [buf.grad.abs().mean().item()
                                       for _, buf, _, _ in self.param_traces]
                    log_dict["eprop/trace_norm_mean"] = sum(trace_norms) / len(trace_norms)
                    log_dict["eprop/l_signal"]        = self._last_l_signal
                    log_dict["eprop/update_mag"]      = self._last_update_mag
                if _hpc_beta_anneal:
                    log_dict["hpc/beta"] = _hpc._beta_current
                logger.log(log_dict, step=step)

            if step % eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                aux_stats = self._collect_hopfield_stats()
                aux_stats.update(self._collect_tau_stats(val_loader))
                if logger:
                    logger.log({
                        "val/loss":       val_loss,
                        "val/perplexity": compute_perplexity(val_loss),
                        "tokens":         tokens_seen + 1,
                        **aux_stats,
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
        if state is None:
            state = self.model.init_state(x.shape[0])
        if self.eprop_mode == "vectorized":
            return self._train_step_vectorized(x, y, state)
        return self._train_step_sequential(x, y, state)

    def _train_step_sequential(self, x, y, state):
        batch, seq_len = x.shape
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

    def _train_step_vectorized(self, x, y, state):
        B, T = x.shape

        # Phase 1: full forward pass — one Python loop but no readout/backward inside
        states, ri_list, state = self._collect_forward_activations(x, state)

        # Phase 2: single readout forward+backward over all B×T tokens
        # Stack as [B, T, rdim] so ri_flat[b*T+t] aligns with y.reshape(-1)[b*T+t]
        ri_flat = torch.stack(ri_list, dim=1).reshape(B * T, -1).requires_grad_(True)
        logits  = self.model.readout(ri_flat)
        loss    = F.cross_entropy(logits, y.reshape(-1))
        loss.backward()

        # Scale by T: cross_entropy averages over B*T here vs B in sequential path
        L_scalar = ri_flat.grad.abs().mean().item() * T
        l_acc    = L_scalar

        nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
        if not self.freeze_readout:
            self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.step(); self.embed_optimizer.zero_grad()
        else:
            self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.zero_grad()

        # Phase 3: sequential trace accumulation over pre-collected states
        # No CUDA kernel launches here — just cheap tensor indexing
        update_acc, update_n = 0.0, 0
        _L = 1.0 if self.normalize_l_signal else L_scalar
        if not self.freeze_recurrent:
            for t in range(T):
                self._update_traces(states[t])
                with torch.no_grad():
                    for param, buf, _, _ in self.param_traces:
                        g = (_L * buf.grad).clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                        update_acc += g.abs().mean().item()
                        update_n   += 1
                self._maybe_enforce_dale()

        self._last_l_signal   = l_acc
        self._last_update_mag = update_acc / max(update_n, 1)
        return loss.item(), state


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
        if state is None:
            state = self.model.init_state(x.shape[0])
        if self.eprop_mode == "vectorized":
            return self._train_step_vectorized(x, y, state)
        return self._train_step_sequential(x, y, state)

    def _train_step_sequential(self, x, y, state):
        batch, seq_len = x.shape
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
            L_vec = ri.grad.detach().mean(dim=0)
            l_acc += ri.grad.detach().abs().mean().item()
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
                    n_l5e  = getattr(self.model, "n_l5e", None)
                    n_cols = getattr(self.model, "n_columns", 1)
                    L_mat  = None
                    if n_l5e is not None and n_cols > 1:
                        try:
                            L_mat = L_vec.reshape(n_cols, n_l5e)
                        except RuntimeError:
                            pass

                    for param, buf, _, post_v_key in self.param_traces:
                        trace = buf.grad
                        if L_mat is not None and post_v_key == "l5_e_v" and buf.n_cols > 1:
                            g = torch.einsum("cn,cnp->cnp", L_mat, trace)
                        elif trace.dim() == 3:
                            g = L_vec.abs().mean() * trace
                        else:
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

    def _train_step_vectorized(self, x, y, state):
        B, T = x.shape

        # Phase 1: full forward pass — one Python loop, no readout/backward inside
        states, ri_list, state = self._collect_forward_activations(x, state)

        # Phase 2: single readout forward+backward over all B×T tokens
        # Stack as [B, T, rdim] → [B*T, rdim] so indices align with y.reshape(-1)
        ri_flat = torch.stack(ri_list, dim=1).reshape(B * T, -1).requires_grad_(True)
        logits  = self.model.readout(ri_flat)
        loss    = F.cross_entropy(logits, y.reshape(-1))
        loss.backward()

        # Per-timestep L_j: grad shape [B*T, rdim] → [B, T, rdim] → mean over B → [T, rdim]
        # Scale by T: cross_entropy averages over B*T here vs B in sequential path
        L_grads = ri_flat.grad.reshape(B, T, -1) * T
        l_acc   = L_grads.abs().mean().item()

        nn.utils.clip_grad_norm_(self.model.readout.parameters(), self.grad_clip)
        if not self.freeze_readout:
            self.readout_optimizer.step(); self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.step(); self.embed_optimizer.zero_grad()
        else:
            self.readout_optimizer.zero_grad()
            if self.embed_optimizer:
                self.embed_optimizer.zero_grad()

        # Phase 3: sequential trace accumulation over pre-collected states
        # No CUDA kernel launches — just cheap tensor indexing into pre-collected activations
        update_acc, update_n = 0.0, 0
        n_l5e  = getattr(self.model, "n_l5e", None)
        n_cols = getattr(self.model, "n_columns", 1)
        if not self.freeze_recurrent:
            for t in range(T):
                self._update_traces(states[t])
                L_vec = L_grads[:, t, :].mean(dim=0)   # [rdim]
                if self.normalize_l_signal:
                    L_vec = L_vec / (L_vec.abs().mean() + 1e-8)
                L_mat = None
                if n_l5e is not None and n_cols > 1:
                    try:
                        L_mat = L_vec.reshape(n_cols, n_l5e)
                    except RuntimeError:
                        pass
                with torch.no_grad():
                    for param, buf, _, post_v_key in self.param_traces:
                        trace = buf.grad
                        if L_mat is not None and post_v_key == "l5_e_v" and buf.n_cols > 1:
                            g = torch.einsum("cn,cnp->cnp", L_mat, trace)
                        elif trace.dim() == 3:
                            g = L_vec.abs().mean() * trace
                        else:
                            n_post = trace.shape[0]
                            L_post = L_vec[:n_post] if n_post <= L_vec.shape[0] else L_vec
                            g = L_post.unsqueeze(1) * trace
                        g = g.clamp(-self.grad_clip, self.grad_clip)
                        self._apply_recurrent_update(param, g)
                        update_acc += g.abs().mean().item()
                        update_n   += 1
                self._maybe_enforce_dale()

        self._last_l_signal   = l_acc
        self._last_update_mag = update_acc / max(update_n, 1)
        return loss.item(), state


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

        self.eprop_steps      = lcfg.get("hybrid_eprop_steps", 100)
        self.bptt_steps       = lcfg.get("hybrid_bptt_steps", 10)
        self.bptt_scope       = lcfg.get("hybrid_bptt_scope", "readout_only")
        # Separate batch size for the BPTT consolidation phase.
        # Allows small e-prop batches (better signal) with larger BPTT batches
        # (stable gradient estimates).  None = use the same batch as e-prop.
        self.bptt_batch_size  = lcfg.get("hybrid_bptt_batch_size", None)
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

        # Phase annealing schedule: list of (eprop_steps, bptt_steps) per SGDR cycle.
        # If set, overrides hybrid_eprop_steps/hybrid_bptt_steps after each SGDR restart.
        _e_sched = lcfg.get("hybrid_eprop_steps_schedule", None)
        _b_sched = lcfg.get("hybrid_bptt_steps_schedule", None)
        # Override system may deliver these as strings (e.g. "[20,10,0]") — parse if needed
        if isinstance(_e_sched, str):
            import json; _e_sched = json.loads(_e_sched)
        if isinstance(_b_sched, str):
            import json; _b_sched = json.loads(_b_sched)
        if _e_sched is not None and _b_sched is not None:
            self._phase_schedule = list(zip([int(x) for x in _e_sched],
                                            [int(x) for x in _b_sched]))
        else:
            self._phase_schedule = None
        self._last_logged_phase = -1   # suppress duplicate phase-transition prints

        # Full-model optimizer for BPTT consolidation steps.
        # Uses a separate LR so consolidation doesn't overwrite e-prop's incremental
        # updates.  Default: same as e-prop LR; override with hybrid_bptt_lr.
        bptt_lr = lcfg.get("hybrid_bptt_lr", self.lr)
        self.freeze_xi = lcfg.get("hybrid_freeze_xi", False)
        if self.freeze_xi and hasattr(self.model, "hippocampus") and hasattr(self.model.hippocampus, "Xi"):
            _xi_id = id(self.model.hippocampus.Xi)
            _bptt_params = [p for p in self.model.parameters() if id(p) != _xi_id]
        else:
            _bptt_params = list(self.model.parameters())
        self.full_optimizer = torch.optim.AdamW(
            _bptt_params, lr=bptt_lr,
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

    @property
    def _cycle_len(self):
        return self.eprop_steps + self.bptt_steps

    def _in_bptt_phase(self) -> bool:
        # eprop_steps == 0 → pure BPTT (modulo always >= 0 == eprop_steps)
        if self.eprop_steps == 0:
            return True
        return (self._global_step % self._cycle_len) >= self.eprop_steps

    def _update_phase_from_schedule(self):
        """Switch eprop/bptt step counts when a new cycle begins."""
        if self._phase_schedule is None or self._sgdr_t0_steps is None:
            return
        cycle_idx = min(self._current_sgdr_cycle, len(self._phase_schedule) - 1)
        if cycle_idx == self._last_logged_phase:
            return
        new_eprop, new_bptt = self._phase_schedule[cycle_idx]
        print(f"\n  [wake/sleep] cycle {cycle_idx}: "
              f"{self.eprop_steps}:{self.bptt_steps} → {new_eprop}:{new_bptt} "
              f"({'pure BPTT' if new_eprop == 0 else 'hybrid'})")
        self.eprop_steps = new_eprop
        self.bptt_steps  = new_bptt
        self._last_logged_phase = cycle_idx

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
        self._update_phase_from_schedule()
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
        # If a separate BPTT batch size is configured, pull a fresh batch of that
        # size from the training iterator, ignoring the small e-prop batch x, y.
        if self.bptt_batch_size is not None:
            xs, ys = [], []
            collected = 0
            while collected < self.bptt_batch_size:
                try:
                    xb, yb = next(self._train_iter)
                except StopIteration:
                    self._train_iter = iter(self._train_loader)
                    xb, yb = next(self._train_iter)
                xs.append(xb)
                ys.append(yb)
                collected += xb.shape[0]
            x = torch.cat(xs, dim=0)[:self.bptt_batch_size]
            y = torch.cat(ys, dim=0)[:self.bptt_batch_size]

        x, y = x.to(self.device), y.to(self.device)
        # If BPTT batch differs from e-prop batch, reinitialize state — the replay
        # batch is a fresh set of sequences so the e-prop state is mismatched anyway.
        state_batch = next(iter(state.column_states.values())).shape[0]
        if x.shape[0] != state_batch:
            state = self.model.init_state(x.shape[0])
        else:
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
