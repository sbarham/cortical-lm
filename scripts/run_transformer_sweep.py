#!/usr/bin/env python3
"""
scripts/run_transformer_sweep.py — Strong-baseline transformer sweep.

Finds the best possible param-matched transformer for the TinyStories paper
experiments.  All variants are parameter-matched to phase1f_hopfield (best
CortexLM architecture) via binary search over d_model.

Sweep axes
----------
  depth       : n_layers  2 | 4 | 6 | 8
  position    : learned absolute PE  |  RoPE
  activation  : GELU  |  SwiGLU  (param-matched via 8/3 × d_ff)
  learning rate: 1e-4 | 3e-4 | 6e-4 | 1e-3
  LR schedule : cosine decay  |  SGDR 20M tokens (matches DAWN)
  n_heads     : 4  |  8

Usage
-----
# Dry-run: print all variants + predicted param counts
python scripts/run_transformer_sweep.py --dry-run

# Run all variants in series (just launch and watch)
python scripts/run_transformer_sweep.py

# Run with W&B logging
python scripts/run_transformer_sweep.py --wandb --wandb-offline

# Run a subset
python scripts/run_transformer_sweep.py --runs l4_rope_swiglu l4_rope_swiglu_lr6e4

# Print srun commands for parallel cluster submission
python scripts/run_transformer_sweep.py \\
    --srun-prefix 'srun --gres=gpu:1 -n1 --time=02:00:00' > sweep_jobs.sh
bash sweep_jobs.sh
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


# ── Constants ─────────────────────────────────────────────────────────────────

CORTEX_CONFIG = "configs/phase1f_hopfield.yaml"   # param-match target
DATA_CONFIG   = "configs/transformer_tinystories.yaml"
TOKENIZER     = "tokenizers/tinystories_bpe4096.pkl"
MAX_TOKENS    = 100_000_000
SGDR_TOKENS   = 20_000_000   # matches DAWN cycle length


# ── Sweep variants ────────────────────────────────────────────────────────────
#
# (id, n_layers, n_heads, pos_encoding, activation, lr, sgdr_tokens, label)
#
# Organised into tiers so the most informative axes are covered first.

_V = [
    # ── Tier 1: Depth × PE  (GELU, lr=3e-4, cosine) ─────────────────────────
    ("l2_learned",       2, 4, "learned", "gelu",   3e-4, None,        "L=2  learned PE  GELU"),
    ("l4_learned",       4, 4, "learned", "gelu",   3e-4, None,        "L=4  learned PE  GELU"),
    ("l6_learned",       6, 4, "learned", "gelu",   3e-4, None,        "L=6  learned PE  GELU"),
    ("l8_learned",       8, 4, "learned", "gelu",   3e-4, None,        "L=8  learned PE  GELU"),
    ("l2_rope",          2, 4, "rope",    "gelu",   3e-4, None,        "L=2  RoPE        GELU"),
    ("l4_rope",          4, 4, "rope",    "gelu",   3e-4, None,        "L=4  RoPE        GELU"),
    ("l6_rope",          6, 4, "rope",    "gelu",   3e-4, None,        "L=6  RoPE        GELU"),
    ("l8_rope",          8, 4, "rope",    "gelu",   3e-4, None,        "L=8  RoPE        GELU"),

    # ── Tier 2: SwiGLU (with RoPE, best depths) ──────────────────────────────
    ("l2_rope_swiglu",   2, 4, "rope",    "swiglu", 3e-4, None,        "L=2  RoPE  SwiGLU"),
    ("l4_rope_swiglu",   4, 4, "rope",    "swiglu", 3e-4, None,        "L=4  RoPE  SwiGLU"),
    ("l6_rope_swiglu",   6, 4, "rope",    "swiglu", 3e-4, None,        "L=6  RoPE  SwiGLU"),

    # ── Tier 3: LR sweep  (l4_rope_swiglu, cosine) ───────────────────────────
    ("l4_rope_swiglu_lr1e4", 4, 4, "rope", "swiglu", 1e-4, None,      "L=4  RoPE  SwiGLU  lr=1e-4"),
    ("l4_rope_swiglu_lr6e4", 4, 4, "rope", "swiglu", 6e-4, None,      "L=4  RoPE  SwiGLU  lr=6e-4"),
    ("l4_rope_swiglu_lr1e3", 4, 4, "rope", "swiglu", 1e-3, None,      "L=4  RoPE  SwiGLU  lr=1e-3"),

    # ── Tier 4: SGDR schedule (matches DAWN 20M cycle) ───────────────────────
    ("l4_rope_sgdr",         4, 4, "rope", "gelu",   3e-4, SGDR_TOKENS, "L=4  RoPE  GELU   SGDR"),
    ("l4_rope_swiglu_sgdr",  4, 4, "rope", "swiglu", 3e-4, SGDR_TOKENS, "L=4  RoPE  SwiGLU SGDR"),
    ("l4_rope_swiglu_lr6e4_sgdr", 4, 4, "rope", "swiglu", 6e-4, SGDR_TOKENS, "L=4  RoPE  SwiGLU lr=6e-4  SGDR"),
    ("l6_rope_swiglu_lr6e4_sgdr", 6, 4, "rope", "swiglu", 6e-4, SGDR_TOKENS, "L=6  RoPE  SwiGLU lr=6e-4  SGDR"),

    # ── Tier 5: n_heads  (l4_rope_swiglu, lr=3e-4) ───────────────────────────
    ("l4_rope_swiglu_h2",    4, 2, "rope", "swiglu", 3e-4, None,       "L=4  RoPE  SwiGLU  h=2"),
    ("l4_rope_swiglu_h8",    4, 8, "rope", "swiglu", 3e-4, None,       "L=4  RoPE  SwiGLU  h=8"),
]

VARIANTS    = {v[0]: v for v in _V}
VARIANT_IDS = [v[0] for v in _V]


# ── Model sizing ──────────────────────────────────────────────────────────────

def _count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make(vocab_size, d_model, n_layers, n_heads, seq_len, pos_encoding, activation):
    d_ff = d_model * 4
    return TransformerBaseline(
        vocab_size, d_model, n_layers, n_heads, d_ff, seq_len,
        pos_encoding=pos_encoding, activation=activation,
    )


def match_size(target, vocab_size, n_layers, n_heads, seq_len,
               pos_encoding, activation) -> tuple[int, int]:
    """Return (d_model, n_params) nearest to target, d_model a multiple of n_heads."""
    lo, hi = 8, 4096
    for _ in range(24):
        mid = ((lo + hi) // 2 // n_heads) * n_heads
        if mid <= lo:
            break
        try:
            p = _count(_make(vocab_size, mid, n_layers, n_heads, seq_len,
                             pos_encoding, activation))
            if p < target:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid
    p_lo = _count(_make(vocab_size, lo, n_layers, n_heads, seq_len, pos_encoding, activation))
    p_hi = _count(_make(vocab_size, hi, n_layers, n_heads, seq_len, pos_encoding, activation))
    d = lo if abs(p_lo - target) <= abs(p_hi - target) else hi
    return d, _count(_make(vocab_size, d, n_layers, n_heads, seq_len, pos_encoding, activation))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(vid, model, config, train_loader, val_loader, device, logger=None):
    _, n_layers, n_heads, pos_encoding, activation, base_lr, sgdr_tokens, label = VARIANTS[vid]

    tcfg          = config["training"]
    seq_len       = config["data"]["seq_len"]
    tokens_per_step = tcfg["batch_size"] * seq_len

    max_tokens  = tcfg.get("max_tokens", MAX_TOKENS)
    max_steps   = max(1, int(max_tokens) // tokens_per_step)
    warmup_steps = max(1, max_steps // 20)    # 5% warmup
    sgdr_t0_steps = (max(1, sgdr_tokens // tokens_per_step)
                     if sgdr_tokens is not None else None)

    optimizer = AdamW(model.parameters(), lr=base_lr,
                      weight_decay=tcfg.get("weight_decay", 1e-4))
    grad_clip = tcfg.get("grad_clip", 1.0)

    log_tokens  = tcfg.get("log_tokens",  200_000)
    eval_tokens = tcfg.get("eval_tokens", 1_000_000)
    log_interval  = max(1, log_tokens  // tokens_per_step)
    eval_interval = max(1, eval_tokens // tokens_per_step)

    def compute_lr(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        post = step - warmup_steps
        if sgdr_t0_steps is not None:
            cycle_pos = post % sgdr_t0_steps
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * cycle_pos / sgdr_t0_steps))
        T = max(max_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * post / T))

    model = model.to(device)
    step, tokens_seen = 0, 0
    train_iter = iter(train_loader)
    t0 = _time.time()

    print(f"\n  [{vid}]  {label}")
    if sgdr_t0_steps:
        print(f"  SGDR T0={sgdr_tokens/1e6:.0f}M tokens, "
              f"~{max_steps // sgdr_t0_steps} cycles")
    print(f"  warmup={warmup_steps:,}  max_steps={max_steps:,}  "
          f"batch={tcfg['batch_size']}  lr={base_lr:.0e}")
    sys.stdout.flush()

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
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
                "elapsed_min":      (_time.time() - t0) / 60.0,
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
            elapsed = (_time.time() - t0) / 60.0
            print(f"  step={step:6d} | tokens={tokens_seen/1e6:5.1f}M "
                  f"| val_ppl={compute_perplexity(val_loss):.2f} "
                  f"| lr={lr:.2e} | {elapsed:.1f}min")
            sys.stdout.flush()
            if logger:
                logger.log({
                    "val/loss":       val_loss,
                    "val/perplexity": compute_perplexity(val_loss),
                    "tokens":         tokens_seen,
                    "elapsed_min":    elapsed,
                }, step=step)

        step += 1

    elapsed = (_time.time() - t0) / 60.0
    print(f"  [{vid}] done in {elapsed:.1f} min")
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transformer hyperparameter sweep (param-matched to phase1f)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", default=VARIANT_IDS,
                        help="Variant IDs or 'all'")
    parser.add_argument("--config", default=DATA_CONFIG)
    parser.add_argument("--cortex-config", default=CORTEX_CONFIG,
                        help="CortexLM config for parameter-count matching")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"Token budget per run (default: {MAX_TOKENS//1_000_000}M)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default="transformer-sweep")
    parser.add_argument("--srun-prefix", default=None,
                        help="Print one 'srun-prefix python ... --runs <id>' per variant "
                             "instead of running. Example: "
                             "--srun-prefix 'srun --gres=gpu:1 -n1 --time=02:00:00'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs == ["all"]:
        args.runs = VARIANT_IDS
    unknown = [r for r in args.runs if r not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}")

    # ── srun mode: print one command per variant and exit ─────────────────────
    if args.srun_prefix is not None:
        base = [sys.executable, "scripts/run_transformer_sweep.py"]
        if args.wandb:          base += ["--wandb"]
        if args.wandb_offline:  base += ["--wandb-offline"]
        if args.max_tokens:     base += ["--max-tokens", str(args.max_tokens)]
        base += ["--wandb-project", args.wandb_project,
                 "--wandb-group",   args.wandb_group,
                 "--cortex-config", args.cortex_config,
                 "--config",        args.config]
        for vid in args.runs:
            print(f"{args.srun_prefix} {' '.join(base)} --runs {vid}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────────
    setup_logging()
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    config = get_config(args.config)
    if args.max_tokens:
        config["training"]["max_tokens"] = args.max_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import pickle
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = tokenizer.vocab_size
    config["data"]["vocab_size"] = vocab_size

    train_ds, val_ds, _, _ = get_dataset(config, tokenizer)
    train_loader = make_dataloader(train_ds, config, shuffle=True)
    val_loader   = make_dataloader(val_ds,   config, shuffle=False)

    # Parameter-match target
    cortex_cfg = get_config(args.cortex_config)
    cortex_cfg["data"]["vocab_size"] = vocab_size
    target_params = CortexLM(cortex_cfg, vocab_size).count_parameters()
    seq_len = config["data"]["seq_len"]
    print(f"\nParameter target (phase1f): {target_params:,}")

    # ── Dry-run: list all variants with predicted sizes ───────────────────────
    if args.dry_run:
        print(f"\n{'─'*72}")
        print(f"  {'ID':<32} {'layers':>6} {'heads':>5} {'pos':>8} {'act':>7} "
              f"{'lr':>7} {'sgdr':>8} {'d_model':>8} {'params':>10}")
        print(f"{'─'*72}")
        for vid in args.runs:
            _, nl, nh, pos, act, lr, sgdr, label = VARIANTS[vid]
            d, np_ = match_size(target_params, vocab_size, nl, nh, seq_len, pos, act)
            sgdr_str = f"{sgdr//1_000_000}M" if sgdr else "cosine"
            print(f"  {vid:<32} {nl:>6} {nh:>5} {pos:>8} {act:>7} "
                  f"{lr:>7.0e} {sgdr_str:>8} {d:>8} {np_:>10,}")
        print(f"{'─'*72}")
        print(f"  {len(args.runs)} variant(s) × "
              f"{config['training'].get('max_tokens', MAX_TOKENS)//1_000_000}M tokens")
        return

    # ── Run sweep ─────────────────────────────────────────────────────────────
    n_total = len(args.runs)
    budget_m = config["training"].get("max_tokens", MAX_TOKENS) // 1_000_000
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Transformer sweep — {n_total} variant(s) × {budget_m}M tokens each")
    print(f"  Parameter target (phase1f): {target_params:,}")
    print(f"  Device: {device}")
    print(f"{sep}")
    for i, vid in enumerate(args.runs, 1):
        _, nl, nh, pos, act, lr, sgdr, lbl = VARIANTS[vid]
        sgdr_str = f"SGDR {sgdr//1_000_000}M" if sgdr else "cosine"
        print(f"  {i:>2}/{n_total}  {vid:<36}  L={nl} {pos:<8} {act:<6} "
              f"lr={lr:.0e}  {sgdr_str}")
    print(f"{sep}\n")
    sys.stdout.flush()

    passed, failed = [], []

    for run_idx, vid in enumerate(args.runs, 1):
        _, nl, nh, pos, act, lr, sgdr, label = VARIANTS[vid]
        d_model, n_params = match_size(target_params, vocab_size, nl, nh,
                                       seq_len, pos, act)
        print(f"\n{sep}")
        print(f"  [{run_idx}/{n_total}]  {vid}  —  {label}")
        print(f"  d_model={d_model}  params={n_params:,}  "
              f"(target {target_params:,}, Δ={n_params-target_params:+,})")
        sys.stdout.flush()

        model = _make(vocab_size, d_model, nl, nh, seq_len, pos, act)

        run_cfg = {
            **config,
            "name": f"transformer-sweep-{vid}",
            "training": {
                **config["training"],
                "lr": lr,
                "checkpoint_dir": f"checkpoints/transformer-sweep/{vid}",
            },
            "logging": {
                **config.get("logging", {}),
                "wandb":   args.wandb,
                "project": args.wandb_project,
                "group":   args.wandb_group,
            },
        }

        logger = Logger(run_cfg) if args.wandb else None

        try:
            train_one(vid, model, run_cfg, train_loader, val_loader,
                      device, logger=logger)
            if logger:
                logger.finish()
            passed.append(vid)
        except Exception as e:
            import traceback
            print(f"\n  !! [{vid}] FAILED: {e}")
            traceback.print_exc()
            failed.append(vid)

    print(f"\n{sep}")
    print(f"  Sweep complete: {len(passed)}/{n_total} passed"
          + (f", {len(failed)} failed: {failed}" if failed else ""))
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
