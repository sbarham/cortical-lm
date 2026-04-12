"""Standard BPTT trainer (full and truncated)."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR
from typing import Optional, Tuple

from cortexlm.model import CortexLM, ModelState
from cortexlm.utils.metrics import compute_perplexity, compute_bpt, compute_bpb, compute_effective_timescales



def _resolve_max_steps(config: dict) -> int:
    """Return max_steps, deriving from max_tokens if set (keeps total data constant across batch sizes)."""
    tcfg = config["training"]
    if "max_tokens" in tcfg:
        batch_size = tcfg.get("batch_size", 32)
        seq_len    = config.get("data", {}).get("seq_len", 128)
        steps = max(1, int(tcfg["max_tokens"]) // (batch_size * seq_len))
        print(f"  max_tokens={tcfg['max_tokens']:,} → max_steps={steps:,} "
              f"(batch={batch_size}, seq_len={seq_len})")
        return steps
    return tcfg.get("max_steps", 100_000)


def _resolve_interval(config: dict, key_tokens: str, key_steps: str, default_steps: int) -> int:
    """Return a step interval, deriving from a token count if the _tokens key is set.

    Keeps logging/eval/checkpoint frequency proportional to data seen rather than
    number of optimizer steps, so runs with different batch sizes produce the same
    number of log/eval/checkpoint events.
    """
    tcfg = config["training"]
    lcfg = config.get("logging", {})
    # Check training section first, then logging section
    token_val = tcfg.get(key_tokens) or lcfg.get(key_tokens)
    if token_val is not None:
        batch_size = tcfg.get("batch_size", 32)
        seq_len    = config.get("data", {}).get("seq_len", 128)
        return max(1, int(token_val) // (batch_size * seq_len))
    return tcfg.get(key_steps) or lcfg.get(key_steps, default_steps)


class BPTTTrainer:
    """
    Standard BPTT trainer.

    Supports:
    - Full BPTT (truncated_bptt_k=None)
    - Truncated BPTT in chunks of k tokens
    - State persistence across batches (detach but don't zero; biological continuity)
    - Re-applies Dale's Law after each optimizer step
    """

    def __init__(self, model: CortexLM, config: dict, device=None, tokenizer=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer  # optional; enables periodic text samples
        self._avg_bytes_per_token = (
            tokenizer.avg_bytes_per_token() if tokenizer is not None else 1.0
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        tcfg = config["training"]
        opt_name = tcfg.get("optimizer", "adamw")
        lr = tcfg.get("lr", 3e-4)
        wd = tcfg.get("weight_decay", 1e-4)

        xi_lr_mult = config.get("hippocampus", {}).get("xi_slow_lr_multiplier", 1.0)
        if xi_lr_mult != 1.0 and hasattr(model, "hippocampus") and hasattr(model.hippocampus, "Xi"):
            xi_params = [model.hippocampus.Xi]
            xi_ids = {id(p) for p in xi_params}
            other_params = [p for p in model.parameters() if id(p) not in xi_ids]
            param_groups = [
                {"params": other_params,  "lr": lr,              "initial_lr": lr},
                {"params": xi_params,     "lr": lr * xi_lr_mult, "initial_lr": lr * xi_lr_mult,
                 "name": "Xi_slow"},
            ]
            if opt_name == "adamw":
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
            else:
                self.optimizer = torch.optim.Adam(param_groups, weight_decay=wd)
        else:
            if opt_name == "adamw":
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.truncated_k = config["learning"].get("truncated_bptt_k", None)
        self.reset_state = config["learning"].get("reset_state_between_batches", False)

        # Scheduler
        max_steps = _resolve_max_steps(config)
        warmup = tcfg.get("warmup_steps", 1000)
        sched_name = tcfg.get("scheduler", "cosine")

        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps - warmup)
        elif sched_name == "sgdr":
            # Cosine annealing with warm restarts.
            # T_0: initial cycle length in steps.  Prefer sgdr_t0_tokens (token-based,
            #       scales with batch size) over sgdr_t0 (raw step count).
            # T_mult: cycle length multiplier per restart (1 = fixed-length cycles).
            if "sgdr_t0_tokens" in tcfg:
                _tps = tcfg.get("batch_size", 32) * config.get("data", {}).get("seq_len", 128)
                t0 = max(1, int(tcfg["sgdr_t0_tokens"]) // _tps)
            else:
                t0 = tcfg.get("sgdr_t0", max(50, (max_steps - warmup) // 10))
            t_mult = tcfg.get("sgdr_t_mult", 2)
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t0, T_mult=t_mult
            )
        elif sched_name == "linear":
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0,
                                      end_factor=0.0, total_iters=max_steps)
        else:
            self.scheduler = None

        self.warmup_steps = warmup
        self.step_count = 0

        self._persistent_state: Optional[ModelState] = None
        self._last_grad_norms: dict = {}

        # Training efficiency flags
        tcfg_full = config["training"]
        self.bf16 = (
            tcfg_full.get("bf16", True)
            and self.device.type == "cuda"
        )
        self.grad_accum_steps = max(1, tcfg_full.get("grad_accum_steps", 1))
        self._accum_count = 0

        # torch.compile (BPTT only — incompatible with e-prop register_forward_hook)
        if tcfg_full.get("compile", False):
            print("  torch.compile enabled (reduce-overhead mode)")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _warmup_lr(self):
        """Linear warmup."""
        if self.step_count < self.warmup_steps:
            factor = (self.step_count + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", self.config["training"]["lr"]) * factor

    def _enforce_dale(self):
        """Re-apply Dale's Law constraints after optimizer step."""
        for module in self.model.modules():
            if hasattr(module, "enforce_dale"):
                module.enforce_dale()

    def _normalize_xi(self):
        """Normalize Hopfield Xi rows to unit sphere if configured."""
        hpc = self.model.hippocampus
        if hasattr(hpc, "normalize_xi_rows"):
            hpc.normalize_xi_rows()

    def _collect_grad_norms(self) -> dict:
        """
        Collect gradient norms for key parameters along the credit-assignment path.
        Returns a dict of {label: norm} for logging.  Skips any parameter that
        doesn't exist or has no grad (e.g. simple_ei model vs layered, apical absent).
        """
        cols = self.model.columns
        apical = getattr(cols, "apical", None)

        candidates = {
            # ── Thalamic / feedforward input ──────────────────────────────────
            "grad/thal_input":      getattr(cols, "thal_proj_e_w", None),
            "grad/input_proj":      getattr(cols, "input_proj", None),   # simple_ei fallback
            # ── Inter-layer feedforward chain ─────────────────────────────────
            "grad/l4_to_l23":       getattr(getattr(cols, "syn_l4e_l23e",  None), "W_e_raw", None),
            "grad/l23_to_l5":       getattr(getattr(cols, "syn_l23e_l5e",  None), "W_e_raw", None),
            # ── Feedback loop: L5 → L6 → L4 ──────────────────────────────────
            "grad/l5_to_l6":        getattr(getattr(cols, "syn_l5e_l6e",   None), "W_e_raw", None),
            "grad/l6_to_l4":        getattr(getattr(cols, "syn_l6e_l4e",   None), "W_e_raw", None),
            "grad/l6_relay":        getattr(cols, "W_l6_relay", None),
            # ── Within-L4 recurrent ───────────────────────────────────────────
            "grad/l4_recurrent":    getattr(getattr(cols, "syn_l4_ee",     None), "W_e_raw", None),
            # ── Readout ───────────────────────────────────────────────────────
            "grad/readout":         next(iter(self.model.readout.parameters()), None),
            # ── Apical pathway (present only when apical_pathway != none) ─────
            "grad/apical_proj":     getattr(apical, "l5_proj",    None),
            "grad/apical_gate":     getattr(apical, "apical_gate", None),
            "grad/apical_l5_to_l23":getattr(apical, "l5_to_l23",  None),
        }
        norms = {}
        for label, param in candidates.items():
            if param is not None and param.grad is not None:
                norms[label] = param.grad.norm().item()

        # ── Apical forward stats (parameter reads, no backward needed) ────────
        if apical is not None and getattr(apical, "variant", "none") == "additive":
            gate_vals = torch.sigmoid(apical.apical_gate).detach()
            norms["apical/gate_mean"] = gate_vals.mean().item()
            norms["apical/gate_std"]  = gate_vals.std().item()

        return norms

    def train_step(
        self,
        x: torch.Tensor,      # [batch, seq_len]
        y: torch.Tensor,      # [batch, seq_len]
        model_state: Optional[ModelState] = None,
    ) -> Tuple[float, ModelState]:
        """
        One training step. Returns (loss_value, new_state).
        """
        x = x.to(self.device)
        y = y.to(self.device)
        batch, seq_len = x.shape

        if model_state is None:
            model_state = self.model.init_state(batch)

        if self.truncated_k is None:
            # Full BPTT
            loss, model_state = self._full_bptt(x, y, model_state)
        else:
            # Truncated BPTT
            loss, model_state = self._truncated_bptt(x, y, model_state)

        return loss, model_state

    def _full_bptt(self, x, y, state):
        is_first_accum = (self._accum_count == 0)
        is_last_accum  = (self._accum_count == self.grad_accum_steps - 1)

        if is_first_accum:
            self.optimizer.zero_grad()

        if self.bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, new_state = self.model(x, state)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                ) / self.grad_accum_steps
        else:
            logits, new_state = self.model(x, state)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ) / self.grad_accum_steps

        loss.backward()

        self._accum_count = (self._accum_count + 1) % self.grad_accum_steps

        if is_last_accum:
            self._last_grad_norms = self._collect_grad_norms()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._enforce_dale()
            self._normalize_xi()

        return loss.item() * self.grad_accum_steps, new_state.detach()

    def _truncated_bptt(self, x, y, state):
        k = self.truncated_k
        seq_len = x.shape[1]
        total_loss = 0.0
        n_chunks = 0

        for start in range(0, seq_len, k):
            x_chunk = x[:, start:start + k]
            y_chunk = y[:, start:start + k]

            self.optimizer.zero_grad()
            logits, state = self.model(x_chunk, state)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_chunk.reshape(-1),
            )
            loss.backward()
            # Collect grad norms on the last chunk (weights are still "fresh")
            if start + k >= seq_len:
                self._last_grad_norms = self._collect_grad_norms()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._enforce_dale()
            self._normalize_xi()

            state = state.detach()
            total_loss += loss.item()
            n_chunks += 1

        return total_loss / max(n_chunks, 1), state

    def train(self, train_loader, val_loader, logger=None, start_step: int = 0):
        """Full training loop."""
        from tqdm import tqdm
        tcfg = self.config["training"]
        no_repeat    = tcfg.get("no_repeat", False)
        max_steps    = _resolve_max_steps(self.config)
        eval_interval = _resolve_interval(self.config, "eval_tokens",       "eval_interval",       500)
        ckpt_interval = _resolve_interval(self.config, "checkpoint_tokens",  "checkpoint_interval", 5000)
        log_interval  = _resolve_interval(self.config, "log_tokens",         "log_interval",        100)
        sample_interval = _resolve_interval(self.config, "sample_tokens",    "sample_interval",     0)
        ckpt_dir = tcfg.get("checkpoint_dir", "checkpoints")

        import os
        os.makedirs(ckpt_dir, exist_ok=True)

        import time as _time_bptt
        _t_start = _time_bptt.time()

        self._persistent_state = None
        step        = start_step
        seq_len     = self.config["data"]["seq_len"]
        tokens_seen = start_step * self.config["training"]["batch_size"] * seq_len
        train_iter  = iter(train_loader)

        if start_step > 0:
            print(f"  resuming from step {start_step:,} ({tokens_seen/1e6:.1f}M tokens) — "
                  f"fast-forwarding data loader...")
            for _skip in range(start_step):
                try:
                    next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    next(train_iter)
            print("  fast-forward complete")

        # HPC beta annealing setup
        _hpc = getattr(self.model, "hippocampus", None)
        _hpc_beta_anneal = (
            _hpc is not None
            and hasattr(_hpc, "update_beta")
            and getattr(_hpc, "beta_init", None) is not None
        )

        # Tau snapshot setup
        lcfg = self.config.get("logging", {})
        _tau_snap_tokens = lcfg.get("tau_snapshot_tokens", 0)
        _tau_snap_dir = lcfg.get("tau_snapshot_dir",
                                  os.path.join(ckpt_dir, "tau_snapshots"))
        # On resume, pre-fill so we don't retake snapshots already saved
        _last_tau_snap = (tokens_seen // _tau_snap_tokens - 1
                          if _tau_snap_tokens > 0 else -1)

        pbar = tqdm(total=max_steps, unit="step", dynamic_ncols=True)
        pbar.set_description("training")

        while step < max_steps:
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                if no_repeat:
                    print(f"\n  [no_repeat] dataset exhausted at step {step:,} "
                          f"({tokens_seen/1e6:.1f}M tokens) — stopping.")
                    break
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            batch = x.shape[0]
            tokens_seen += batch * x.shape[1]

            # Disinhibition annealing: decay VIP→SST gain from 1→0 over anneal window
            cols = getattr(self.model, "columns", None)
            if cols is not None and getattr(cols, "disinhibition_anneal_tokens", 0) > 0:
                scale = max(0.0, 1.0 - tokens_seen / cols.disinhibition_anneal_tokens)
                cols.set_disinhibition_scale(scale)

            # HPC beta annealing: step beta_init → beta over first beta_anneal_frac of training
            if _hpc_beta_anneal:
                _hpc.update_beta(step, max_steps)

            # Initialize or reuse persistent state
            if self._persistent_state is None or self.reset_state:
                self._persistent_state = self.model.init_state(batch)

            loss, self._persistent_state = self.train_step(x, y, self._persistent_state)

            # LR management: once per training step (not per BPTT chunk)
            self._warmup_lr()
            if self.scheduler and self.step_count >= self.warmup_steps:
                self.scheduler.step()
            self.step_count += 1

            pbar.update(1)

            if step % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                ppl = compute_perplexity(loss)
                postfix = dict(loss=f"{loss:.3f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}")
                if torch.cuda.is_available():
                    used = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    postfix["vram"] = f"{used:.1f}/{total:.0f}GB"
                pbar.set_postfix(**postfix)
                if self._last_grad_norms:
                    parts = "  ".join(
                        f"{k.split('/')[1]}={v:.2e}" for k, v in self._last_grad_norms.items()
                    )
                    tqdm.write(f"  grad norms | {parts}")
                if logger:
                    import time as _time_bptt
                    log_dict = {
                        "train/loss":       loss,
                        "train/perplexity": ppl,
                        "lr":               lr,
                        "tokens":           tokens_seen,
                        "elapsed_min":      (_time_bptt.time() - _t_start) / 60.0,
                    }
                    log_dict.update(self._last_grad_norms)
                    _cols = getattr(self.model, "columns", None)
                    if _cols is not None and getattr(_cols, "disinhibition_anneal_tokens", 0) > 0:
                        log_dict["disinhibition/scale"] = _cols._disinhibition_scale
                    if _hpc_beta_anneal:
                        log_dict["hpc/beta"] = _hpc._beta_current
                    logger.log(log_dict, step=step)

            if step % eval_interval == 0:
                # Scale eval batches so total tokens ≈ constant (~640k) regardless of batch size
                target_tokens = 640_000
                seq_len = self.config["data"]["seq_len"]
                batch_size = self.config["training"]["batch_size"]
                max_eval_batches = max(4, min(50, target_tokens // (batch_size * seq_len)))
                val_loss, dist_stats = self.evaluate(val_loader, max_batches=max_eval_batches)
                val_ppl = compute_perplexity(val_loss)
                val_bpt = compute_bpt(val_loss)
                val_bpb = compute_bpb(val_loss, self._avg_bytes_per_token)
                tqdm.write(
                    f"step {step:6d} | val_loss={val_loss:.4f} "
                    f"val_ppl={val_ppl:.2f} val_bpt={val_bpt:.4f} val_bpb={val_bpb:.4f} "
                    f"H={dist_stats['val/output_entropy']:.3f} "
                    f"top5={dist_stats['val/top5_conc']:.3f}"
                )
                aux_stats = {**dist_stats}
                aux_stats.update(self._collect_hopfield_stats())
                aux_stats.update(self._collect_tau_stats(val_loader))
                if logger:
                    logger.log({
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/bpt": val_bpt,
                        "val/bpb": val_bpb,
                        "tokens": tokens_seen,
                        **aux_stats,
                    }, step=step)

            if step % ckpt_interval == 0 and step > 0:
                self._save_checkpoint(ckpt_dir, step)
                tqdm.write(f"  checkpoint saved → {ckpt_dir}/step_{step:07d}.pt")

            if _tau_snap_tokens > 0:
                _tau_snap_due = tokens_seen // _tau_snap_tokens
                if _tau_snap_due > _last_tau_snap:
                    _last_tau_snap = _tau_snap_due
                    self._save_tau_eff_snapshot(val_loader, _tau_snap_dir, tokens_seen)

            if sample_interval > 0 and self.tokenizer is not None \
                    and step % sample_interval == 0 and step > 0:
                self._generate_sample(step)

            step += 1

        pbar.close()

        # Always save a final checkpoint (unless the last step was already saved)
        final_step = step - 1
        if final_step % ckpt_interval != 0:
            self._save_checkpoint(ckpt_dir, final_step)
            tqdm.write(f"  final checkpoint saved → {ckpt_dir}/step_{final_step:07d}.pt")

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> tuple:
        """Returns (avg_loss, dist_stats).

        dist_stats contains output-distribution diagnostics averaged over batches:
          val/output_entropy  — mean H(p) in nats. Lower = sharper predictions.
          val/top5_conc       — mean fraction of probability mass in top-5 tokens.
          val/top10_conc      — mean fraction of probability mass in top-10 tokens.
          val/nll_entropy_gap — mean (NLL - H): large positive → model is confident
                                but on wrong tokens; near 0 → uncertainty is earned.
        """
        from tqdm import tqdm
        self.model.eval()
        total_loss = 0.0
        total_entropy = 0.0
        total_top5 = 0.0
        total_top10 = 0.0
        n = 0
        for i, (x, y) in enumerate(tqdm(val_loader, total=max_batches, desc="  evaluating",
                                         leave=False, unit="batch")):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            state = self.model.init_state(x.shape[0])
            logits, _ = self.model(x, state)             # [batch, seq, vocab]
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_y      = y.reshape(-1)

            loss = F.cross_entropy(flat_logits, flat_y)
            total_loss += loss.item()

            # Distribution stats — computed from softmax over vocab
            log_probs = F.log_softmax(flat_logits, dim=-1)   # [B*T, vocab]
            probs     = log_probs.exp()

            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            top5_conc  = probs.topk(5,  dim=-1).values.sum(dim=-1).mean().item()
            top10_conc = probs.topk(10, dim=-1).values.sum(dim=-1).mean().item()

            total_entropy += entropy
            total_top5    += top5_conc
            total_top10   += top10_conc
            n += 1

        self.model.train()
        denom = max(n, 1)
        avg_loss    = total_loss    / denom
        avg_entropy = total_entropy / denom
        dist_stats = {
            "val/output_entropy":  avg_entropy,
            "val/top5_conc":       total_top5  / denom,
            "val/top10_conc":      total_top10 / denom,
            "val/nll_entropy_gap": avg_loss - avg_entropy,
        }
        return avg_loss, dist_stats

    def _generate_sample(self, step: int):
        """Generate a short text sample and print it via tqdm.write."""
        from tqdm import tqdm
        from cortexlm.utils.sampling import generate

        lcfg = self.config.get("logging", {})
        prompt    = lcfg.get("sample_prompt", "")
        max_toks  = lcfg.get("sample_max_tokens", 150)
        top_p     = lcfg.get("sample_top_p", 0.9)
        temp      = lcfg.get("sample_temperature", 0.8)

        try:
            text = generate(
                self.model, self.tokenizer,
                prompt=prompt,
                max_new_tokens=max_toks,
                temperature=temp,
                top_p=top_p,
                device=self.device,
            )
            divider = "-" * 60
            prompt_label = f"prompt={prompt!r}" if prompt else "unconditional"
            tqdm.write(f"\n  [step {step:,} sample | {prompt_label} | temp={temp} top_p={top_p}]")
            tqdm.write(f"  {divider}")
            for line in text.splitlines():
                tqdm.write(f"  {line}")
            tqdm.write(f"  {divider}\n")
        except Exception as exc:
            tqdm.write(f"  [sample failed at step {step}: {exc}]")

    @torch.no_grad()
    def _collect_hopfield_stats(self) -> dict:
        """Attention entropy and sharpness over the Hopfield memory bank.

        hpc/attn_entropy  — mean Shannon entropy of the attention distribution
                            (nats; max = log(n_memories)).  High → diffuse/unused;
                            low → sharp retrieval of specific memories.
        hpc/attn_max      — mean of the max attention weight per batch item.
                            Complement to entropy: high → one memory dominates.
        """
        hpc = getattr(self.model, "hippocampus", None)
        if hpc is None:
            return {}
        weights = getattr(hpc, "_last_attn_weights", None)
        if weights is None:
            return {}
        # weights: [batch, n_memories]
        eps = 1e-10
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean().item()
        attn_max = weights.max(dim=-1).values.mean().item()
        stats = {
            "hpc/attn_entropy": entropy,
            "hpc/attn_max":     attn_max,
        }
        # CA1 surprise: mean mismatch norm from last training step
        surprise = getattr(self.model, "_last_hpc_surprise", None)
        if surprise is not None:
            stats["hpc/ca1_surprise"] = surprise
        return stats

    @torch.no_grad()
    def _collect_tau_stats(self, val_loader) -> dict:
        """Estimate effective neural timescales from autocorrelation of firing rates.

        Steps one batch through the model one token at a time, collects
        r_l23e and r_l5e traces, computes per-neuron tau_eff via log-linear
        ACF fit, and logs mean/std/p25/p75 for each layer.

        tau/l23e_mean, tau/l23e_std, tau/l23e_p25, tau/l23e_p75
        tau/l5e_mean,  tau/l5e_std,  tau/l5e_p25,  tau/l5e_p75
        """
        import numpy as np
        self.model.eval()
        try:
            x, _ = next(iter(val_loader))          # [batch, seq_len]
        except StopIteration:
            return {}
        x = x[:4].to(self.device)                  # keep batch small — 4 is enough for ACF
        batch, seq_len = x.shape
        state = self.model.init_state(batch)

        traces_l4e, traces_l23e, traces_l5e, traces_l6e = [], [], [], []
        for t in range(seq_len):
            _, state = self.model.step(x[:, t], state)
            col_state = state.column_states
            if "r_l23e" in col_state:
                # [batch, n_cols, n_neurons] → mean over batch & cols → [n_neurons]
                traces_l4e.append( col_state["r_l4e" ].mean(dim=(0, 1)).cpu().numpy())
                traces_l23e.append(col_state["r_l23e"].mean(dim=(0, 1)).cpu().numpy())
                traces_l5e.append( col_state["r_l5e" ].mean(dim=(0, 1)).cpu().numpy())
                traces_l6e.append( col_state["r_l6e" ].mean(dim=(0, 1)).cpu().numpy())

        self.model.train()
        if not traces_l23e:
            return {}

        # Stack to [T, n_neurons]
        arr_l4e  = np.stack(traces_l4e)    # [T, n_l4e]
        arr_l23e = np.stack(traces_l23e)   # [T, n_l23e]
        arr_l5e  = np.stack(traces_l5e)    # [T, n_l5e]
        arr_l6e  = np.stack(traces_l6e)    # [T, n_l6e]

        stats = {}
        for key, arr in [
            ("tau/l4e",  arr_l4e),
            ("tau/l23e", arr_l23e),
            ("tau/l5e",  arr_l5e),
            ("tau/l6e",  arr_l6e),
        ]:
            taus = compute_effective_timescales(
                torch.from_numpy(arr), max_lag=min(50, seq_len // 4)
            )
            stats[f"{key}_mean"] = float(np.mean(taus))
            stats[f"{key}_std"]  = float(np.std(taus))
            stats[f"{key}_p25"]  = float(np.percentile(taus, 25))
            stats[f"{key}_p75"]  = float(np.percentile(taus, 75))
        return stats

    @torch.no_grad()
    def _save_tau_eff_snapshot(self, val_loader, snap_dir: str, tokens_seen: int):
        """Estimate per-neuron effective timescales via ACF and save full distributions.

        Saves tau_eff arrays (one value per neuron) for L4, L2/3, L5, L6 excitatory
        populations.  Files: tau_<tokens:012d>.npz with keys l4e, l23e, l5e, l6e.
        """
        import os
        import numpy as np
        self.model.eval()
        try:
            x, _ = next(iter(val_loader))
        except StopIteration:
            return
        x = x[:4].to(self.device)
        seq_len = x.shape[1]
        state = self.model.init_state(x.shape[0])

        key_map = {"l4e": "r_l4e", "l23e": "r_l23e", "l5e": "r_l5e", "l6e": "r_l6e"}
        traces = {k: [] for k in key_map}
        for t in range(seq_len):
            _, state = self.model.step(x[:, t], state)
            col_state = state.column_states
            for k, rk in key_map.items():
                if rk in col_state:
                    traces[k].append(col_state[rk].mean(dim=(0, 1)).cpu().numpy())

        self.model.train()
        arrays = {}
        for k, trace_list in traces.items():
            if not trace_list:
                continue
            arr = np.stack(trace_list)   # [T, n_neurons]
            taus = compute_effective_timescales(
                torch.from_numpy(arr), max_lag=min(50, seq_len // 4)
            )
            arrays[k] = taus.astype(np.float32)
        if not arrays:
            return
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, f"tau_{tokens_seen:012d}.npz")
        np.savez(path, **arrays)
        l23_mean = arrays["l23e"].mean() if "l23e" in arrays else float("nan")
        print(f"  [tau_eff] {tokens_seen/1e6:.0f}M tokens — "
              f"L2/3 mean={l23_mean:.1f}ms → {path}", flush=True)

    def _save_checkpoint(self, ckpt_dir: str, step: int):
        import os
        path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
