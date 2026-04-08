#!/usr/bin/env python3
"""
scripts/run_transformer_sgdr.py — Transformer baseline with parameterisable SGDR schedule.

Purpose: isolate whether the CortexLM sample-efficiency advantage comes from the SGDR
LR schedule vs the architecture.  Runs the matched-parameter transformer on wikitext103
with flat cosine decay or cosine-restart (SGDR) schedules, using exactly the same LR
formula as the e-prop hybrid trainer so the comparison is apples-to-apples.

Variants
--------
    flat      cosine decay to 0 over the full run  (standard baseline)
    sgdr12m   SGDR restarts every 12.5M tokens      (matches vc/vd)
    sgdr20m   SGDR restarts every 20M tokens         (matches ve/vf)
    sgdr25m   SGDR restarts every 25M tokens

LR formula (matches eprop trainer exactly):
    warmup  : lr = base_lr * step / warmup_steps           (first 5% of steps)
    SGDR    : lr = base_lr * 0.5 * (1 + cos(π * pos / T0)) per cycle
    flat    : lr = base_lr * 0.5 * (1 + cos(π * step / max_steps))

Usage
-----
# Most important comparison — match vc's SGDR cycle
python scripts/run_transformer_sgdr.py --runs sgdr12m --wandb --wandb-offline

# All variants
python scripts/run_transformer_sgdr.py --runs all --wandb --wandb-offline

# Specific token budget
python scripts/run_transformer_sgdr.py --runs flat sgdr12m --max-tokens 50000000 --wandb

# Dry run
python scripts/run_transformer_sgdr.py --runs all --dry-run
"""

import argparse
import math
import os
import sys
import time as _time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from cortexlm.baselines.transformer import TransformerBaseline
from cortexlm.model import CortexLM
from cortexlm.utils.config import get_config
from cortexlm.utils.logging import Logger, setup_logging
from cortexlm.utils.metrics import compute_perplexity
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer


CONFIG        = "configs/scale_transformer_baseline.yaml"
CORTEX_CONFIG = "configs/scale_5m_run5_combined.yaml"
TOKENIZER     = "tokenizers/wikitext103_bpe16k.pkl"
N_HEADS       = 4

# (id, label, sgdr_tokens)   sgdr_tokens=None → flat cosine decay
_VARIANTS = [
    ("flat",    "flat cosine decay (no SGDR)",         None),
    ("sgdr12m", "SGDR every 12.5M tokens  (matches vc/vd)", 12_500_000),
    ("sgdr20m", "SGDR every 20M tokens    (matches ve/vf)", 20_000_000),
    ("sgdr25m", "SGDR every 25M tokens",                   25_000_000),
]
VARIANTS = {v[0]: v for v in _VARIANTS}
VARIANT_IDS  = [v[0] for v in _VARIANTS]


# ── Model sizing ──────────────────────────────────────────────────────────────

def _count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_transformer(vocab_size: int, d_model: int, n_layers: int, seq_len: int):
    d_ff = d_model * 4
    return TransformerBaseline(vocab_size, d_model, n_layers, N_HEADS, d_ff, seq_len)


def match_transformer_size(target: int, vocab_size: int, n_layers: int, seq_len: int) -> int:
    """Binary search for d_model (multiple of n_heads) nearest to target params."""
    lo, hi = 16, 4096
    for _ in range(20):
        mid = ((lo + hi) // 2 // N_HEADS) * N_HEADS
        if mid <= lo:
            break
        try:
            p = _count(_make_transformer(vocab_size, mid, n_layers, seq_len))
            if p < target:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid
    # Pick whichever of lo/hi is closer to target
    p_lo = _count(_make_transformer(vocab_size, lo, n_layers, seq_len))
    p_hi = _count(_make_transformer(vocab_size, hi, n_layers, seq_len))
    return lo if abs(p_lo - target) <= abs(p_hi - target) else hi


# ── Training loop ─────────────────────────────────────────────────────────────

def train_transformer(variant_id: str, model, config: dict, train_loader, val_loader,
                      device, logger=None, wandb_offline: bool = False):
    vid, label, sgdr_tokens = VARIANTS[variant_id]

    tcfg          = config["training"]
    seq_len       = config["data"]["seq_len"]
    base_lr       = tcfg["lr"]
    tokens_per_step = tcfg["batch_size"] * seq_len

    if "max_tokens" in tcfg:
        max_steps = max(1, int(tcfg["max_tokens"]) // tokens_per_step)
    else:
        max_steps = tcfg.get("max_steps", 100_000)

    warmup_steps  = max(1, max_steps // 20)    # 5% warmup — matches eprop sweep
    sgdr_t0_steps = (max(1, sgdr_tokens // tokens_per_step)
                     if sgdr_tokens is not None else None)

    optimizer = AdamW(model.parameters(), lr=base_lr,
                      weight_decay=tcfg.get("weight_decay", 1e-4))
    grad_clip = tcfg.get("grad_clip", 1.0)

    eval_tokens = tcfg.get("eval_tokens", 500_000)
    log_tokens  = tcfg.get("log_tokens",  100_000)
    eval_interval = max(1, eval_tokens  // tokens_per_step)
    log_interval  = max(1, log_tokens   // tokens_per_step)

    def compute_lr(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        post = step - warmup_steps
        if sgdr_t0_steps is not None:
            cycle_pos = post % sgdr_t0_steps
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * cycle_pos / sgdr_t0_steps))
        else:
            T = max(max_steps - warmup_steps, 1)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * post / T))

    model = model.to(device)
    step        = 0
    tokens_seen = 0
    train_iter  = iter(train_loader)
    _t_start    = _time.time()

    print(f"\n  Training: {label}")
    if sgdr_t0_steps:
        n_cycles = max_steps // sgdr_t0_steps
        print(f"  SGDR T0={sgdr_t0_steps:,} steps ({sgdr_tokens/1e6:.1f}M tokens), "
              f"~{n_cycles} cycles over {max_steps:,} steps")
    print(f"  warmup={warmup_steps:,}  max_steps={max_steps:,}  "
          f"batch={tcfg['batch_size']}  lr={base_lr}")

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Set LR
        lr = compute_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens_seen += x.numel()

        if step % log_interval == 0 and logger:
            logger.log({
                "train/loss":       loss.item(),
                "train/perplexity": compute_perplexity(loss.item()),
                "lr":               lr,
                "tokens":           tokens_seen,
                "elapsed_min":      (_time.time() - _t_start) / 60.0,
            }, step=step)

        if step % eval_interval == 0:
            model.eval()
            val_loss, n = 0.0, 0
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_loader):
                    if i >= 50:
                        break
                    xv, yv = xv.to(device), yv.to(device)
                    lv, _ = model(xv)
                    val_loss += F.cross_entropy(
                        lv.reshape(-1, lv.size(-1)), yv.reshape(-1)
                    ).item()
                    n += 1
            val_loss /= max(n, 1)
            val_ppl   = compute_perplexity(val_loss)
            print(f"  step={step:6d} | tokens={tokens_seen/1e6:6.1f}M "
                  f"| val_ppl={val_ppl:.2f} | lr={lr:.2e}")
            if logger:
                logger.log({
                    "val/loss":       val_loss,
                    "val/perplexity": val_ppl,
                    "tokens":         tokens_seen,
                    "elapsed_min":    (_time.time() - _t_start) / 60.0,
                }, step=step)

        step += 1

    print(f"  [{vid}] done — {(_time.time() - _t_start)/60:.1f} min")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transformer baseline with parameterisable SGDR schedule"
    )
    parser.add_argument("--runs", nargs="+", default=["flat", "sgdr12m"],
                        help=f"Variant IDs or 'all'. Choices: {VARIANT_IDS}")
    parser.add_argument("--config", default=CONFIG,
                        help="Base config YAML (default: scale_transformer_baseline.yaml)")
    parser.add_argument("--cortex-config", default=CORTEX_CONFIG,
                        help="CortexLM config used only for parameter-count matching "
                             "(default: scale_5m_run5_combined.yaml)")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Token budget (default: from config, 100M)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Set WANDB_MODE=offline (log locally, sync later)")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"transformer-sgdr-{_time.strftime('%Y-%m-%d')}")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs == ["all"]:
        args.runs = VARIANT_IDS

    unknown = [r for r in args.runs if r not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}. Choices: {VARIANT_IDS}")

    config = get_config(args.config)
    if args.max_tokens:
        config["training"]["max_tokens"] = args.max_tokens
    setup_logging()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    import pickle
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = tokenizer.vocab_size
    config["data"]["vocab_size"] = vocab_size

    # Data
    train_ds, val_ds, _, _ = get_dataset(config, tokenizer)
    train_loader = make_dataloader(train_ds, config, shuffle=True)
    val_loader   = make_dataloader(val_ds,   config, shuffle=False)

    # Match parameter count to CortexLM — load the cortical config separately so
    # we get the real scale_5m architecture, not the transformer baseline defaults.
    cortex_config = get_config(args.cortex_config)
    cortex_config["data"]["vocab_size"] = vocab_size
    cortex = CortexLM(cortex_config, vocab_size)
    target_params = cortex.count_parameters()
    del cortex
    seq_len  = config["data"]["seq_len"]
    n_layers = config.get("baseline", {}).get("n_layers", 6)
    d_model  = match_transformer_size(target_params, vocab_size, n_layers, seq_len)
    n_params = _count(_make_transformer(vocab_size, d_model, n_layers, seq_len))
    print(f"CortexLM target: {target_params:,} params")
    print(f"Transformer:     {n_params:,} params  (d_model={d_model}, layers={n_layers})")

    print(f"\nTransformer SGDR sweep — {len(args.runs)} variant(s) × "
          f"{config['training'].get('max_tokens', 0)/1e6:.0f}M tokens")
    print(f"Variants: {args.runs}")
    if args.dry_run:
        print("*** DRY RUN — no training ***")
        for vid in args.runs:
            _, label, sgdr_tokens = VARIANTS[vid]
            print(f"  [{vid}]  {label}"
                  + (f"  (SGDR {sgdr_tokens/1e6:.1f}M)" if sgdr_tokens else ""))
        return

    passed, failed = [], []
    for vid in args.runs:
        _, label, _ = VARIANTS[vid]
        run_name = f"transformer-{vid}"
        run_config = {
            **config,
            "name": run_name,
            "training": {
                **config["training"],
                "checkpoint_dir": f"checkpoints/{run_name}",
            },
            "logging": {
                **config.get("logging", {}),
                "wandb":   args.wandb,
                "project": args.wandb_project,
                "group":   args.wandb_group,
            },
        }

        model = _make_transformer(vocab_size, d_model, n_layers, seq_len)
        logger = Logger(run_config) if args.wandb else None

        try:
            train_transformer(vid, model, run_config, train_loader, val_loader,
                              device, logger=logger)
            if logger:
                logger.finish()
            passed.append(vid)
        except Exception as e:
            print(f"\n  !! [{vid}] FAILED: {e}")
            failed.append(vid)
            if args.stop_on_failure:
                break

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Done: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
