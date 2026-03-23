"""Standard BPTT trainer (full and truncated)."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Optional, Tuple

from cortexlm.model import CortexLM, ModelState
from cortexlm.utils.metrics import compute_perplexity, compute_bpc


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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        tcfg = config["training"]
        opt_name = tcfg.get("optimizer", "adamw")
        lr = tcfg.get("lr", 3e-4)
        wd = tcfg.get("weight_decay", 1e-4)

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
        elif sched_name == "linear":
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0,
                                      end_factor=0.0, total_iters=max_steps)
        else:
            self.scheduler = None

        self.warmup_steps = warmup
        self.step_count = 0

        self._persistent_state: Optional[ModelState] = None
        self._last_grad_norms: dict = {}

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

    def _collect_grad_norms(self) -> dict:
        """
        Collect gradient norms for key parameters along the credit-assignment path.
        Returns a dict of {label: norm} for logging.  Skips any parameter that
        doesn't exist or has no grad (e.g. simple_ei model vs layered).
        """
        cols = self.model.columns
        candidates = {
            "grad/thal_input":  getattr(cols, "thal_proj_e_w", None),
            "grad/l4_to_l23":   getattr(getattr(cols, "syn_l4e_l23e", None), "W_e_raw", None),
            "grad/l23_to_l5":   getattr(getattr(cols, "syn_l23e_l5e", None), "W_e_raw", None),
            "grad/input_proj":  getattr(cols, "input_proj", None),   # simple_ei fallback
            "grad/readout":     next(iter(self.model.readout.parameters()), None),
        }
        norms = {}
        for label, param in candidates.items():
            if param is not None and param.grad is not None:
                norms[label] = param.grad.norm().item()
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
        self.optimizer.zero_grad()
        logits, new_state = self.model(x, state)
        # logits: [batch, seq_len, vocab_size]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        self._last_grad_norms = self._collect_grad_norms()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self._enforce_dale()
        self._warmup_lr()
        if self.scheduler and self.step_count >= self.warmup_steps:
            self.scheduler.step()
        self.step_count += 1
        return loss.item(), new_state.detach()

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
            self._warmup_lr()
            if self.scheduler and self.step_count >= self.warmup_steps:
                self.scheduler.step()
            self.step_count += 1

            state = state.detach()
            total_loss += loss.item()
            n_chunks += 1

        return total_loss / max(n_chunks, 1), state

    def train(self, train_loader, val_loader, logger=None):
        """Full training loop."""
        from tqdm import tqdm
        tcfg = self.config["training"]
        max_steps = _resolve_max_steps(self.config)
        eval_interval = tcfg.get("eval_interval", 500)
        ckpt_interval = tcfg.get("checkpoint_interval", 5000)
        ckpt_dir = tcfg.get("checkpoint_dir", "checkpoints")
        log_interval    = self.config["logging"].get("log_interval", 100)
        sample_interval = self.config["logging"].get("sample_interval", 0)

        import os
        os.makedirs(ckpt_dir, exist_ok=True)

        self._persistent_state = None
        step = 0
        tokens_seen = 0
        train_iter = iter(train_loader)

        pbar = tqdm(total=max_steps, unit="step", dynamic_ncols=True)
        pbar.set_description("training")

        while step < max_steps:
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            batch = x.shape[0]
            tokens_seen += batch * x.shape[1]

            # Initialize or reuse persistent state
            if self._persistent_state is None or self.reset_state:
                self._persistent_state = self.model.init_state(batch)

            loss, self._persistent_state = self.train_step(x, y, self._persistent_state)
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
                    log_dict = {
                        "train/loss": loss, "train/perplexity": ppl,
                        "lr": lr, "tokens": tokens_seen,
                    }
                    log_dict.update(self._last_grad_norms)
                    logger.log(log_dict, step=step)

            if step % eval_interval == 0:
                # Scale eval batches so total tokens ≈ constant (~640k) regardless of batch size
                target_tokens = 640_000
                seq_len = self.config["data"]["seq_len"]
                batch_size = self.config["training"]["batch_size"]
                max_eval_batches = max(4, min(50, target_tokens // (batch_size * seq_len)))
                val_loss = self.evaluate(val_loader, max_batches=max_eval_batches)
                val_ppl = compute_perplexity(val_loss)
                tqdm.write(
                    f"step {step:6d} | val_loss={val_loss:.4f} "
                    f"val_ppl={val_ppl:.2f} val_bpc={compute_bpc(val_loss):.4f}"
                )
                if logger:
                    logger.log({
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "val/bpc": compute_bpc(val_loss),
                        "tokens": tokens_seen,
                    }, step=step)

            if step % ckpt_interval == 0 and step > 0:
                self._save_checkpoint(ckpt_dir, step)
                tqdm.write(f"  checkpoint saved → {ckpt_dir}/step_{step:07d}.pt")

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

    def _generate_sample(self, step: int):
        """Generate a short text sample and print it via tqdm.write."""
        from tqdm import tqdm
        from cortexlm.utils.sampling import generate

        lcfg = self.config.get("logging", {})
        prompt    = lcfg.get("sample_prompt", "")
        max_toks  = lcfg.get("sample_tokens", 150)
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

    def _save_checkpoint(self, ckpt_dir: str, step: int):
        import os
        path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
