#!/usr/bin/env python3
"""
scripts/run_transformer_10m_sweep.py — 10M parameter transformer sweep on WikiText-103.

Mirrors the 650K TinyStories sweep (run_transformer_sweep.py) at a larger scale,
and provides the true transformer baseline for comparing with CCLM at the WikiText-103
scale.  The current 181 ppl baseline may be drastically underestimating transformer
performance — this sweep finds the optimal configuration.

All variants are param-matched to ~10M via binary search over d_model.

Sweep axes
----------
  depth         : n_layers  2 | 4 | 6 | 8
  position      : learned absolute PE  |  RoPE
  activation    : GELU  |  SwiGLU  (param-matched via 2/3 × d_ff rule)
  learning rate : 1e-4 | 3e-4 | 6e-4 | 1e-3
  LR schedule   : cosine decay  |  SGDR 20M tokens (matches DAWN)
  n_heads       : 8  |  16

Usage
-----
# Dry-run: print all variants + predicted param counts
python scripts/run_transformer_10m_sweep.py --dry-run

# Run all variants in series
python scripts/run_transformer_10m_sweep.py --wandb --wandb-offline

# Run a subset
python scripts/run_transformer_10m_sweep.py --runs t01_baseline t04_rope_swiglu

# Print srun commands for parallel cluster dispatch
python scripts/run_transformer_10m_sweep.py \\
    --srun-prefix 'srun --gres=gpu:1 -n1 --time=04:00:00'
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
from cortexlm.utils.config import get_config
from cortexlm.utils.logging import Logger, setup_logging
from cortexlm.utils.metrics import compute_perplexity
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_CONFIG   = "configs/transformer_wikitext103.yaml"
MAX_TOKENS    = 100_000_000     # 100M tokens (~1× WT103 train)
SGDR_TOKENS   = 20_000_000     # matches DAWN cycle length
TARGET_PARAMS = 10_000_000     # ~10M ±5%


# ── Sweep variants ────────────────────────────────────────────────────────────
#
# (id, n_layers, n_heads, pos_encoding, activation, lr, sgdr_tokens, label)

_V = [
    # ── Tier 1: Baseline + depth × PE (GELU, lr=3e-4, cosine) ────────────────
    ("t01_baseline",         2, 8,  "learned", "gelu",   3e-4, None,        "L=2  learned PE  GELU  [baseline]"),
    ("t02_l4_learned",       4, 8,  "learned", "gelu",   3e-4, None,        "L=4  learned PE  GELU"),
    ("t03_l6_learned",       6, 8,  "learned", "gelu",   3e-4, None,        "L=6  learned PE  GELU"),
    ("t04_l4_rope",          4, 8,  "rope",    "gelu",   3e-4, None,        "L=4  RoPE        GELU"),
    ("t05_l6_rope",          6, 8,  "rope",    "gelu",   3e-4, None,        "L=6  RoPE        GELU"),
    ("t06_l8_rope",          8, 8,  "rope",    "gelu",   3e-4, None,        "L=8  RoPE        GELU"),

    # ── Tier 2: SwiGLU (with RoPE) ───────────────────────────────────────────
    ("t07_l4_rope_swiglu",   4, 8,  "rope",    "swiglu", 3e-4, None,        "L=4  RoPE  SwiGLU"),
    ("t08_l6_rope_swiglu",   6, 8,  "rope",    "swiglu", 3e-4, None,        "L=6  RoPE  SwiGLU"),
    ("t09_l8_rope_swiglu",   8, 8,  "rope",    "swiglu", 3e-4, None,        "L=8  RoPE  SwiGLU"),

    # ── Tier 3: LR sweep (l6_rope_swiglu, cosine) ────────────────────────────
    ("t10_l6_swiglu_lr1e4",  6, 8,  "rope",    "swiglu", 1e-4, None,        "L=6  RoPE  SwiGLU  lr=1e-4"),
    ("t11_l6_swiglu_lr6e4",  6, 8,  "rope",    "swiglu", 6e-4, None,        "L=6  RoPE  SwiGLU  lr=6e-4"),
    ("t12_l6_swiglu_lr1e3",  6, 8,  "rope",    "swiglu", 1e-3, None,        "L=6  RoPE  SwiGLU  lr=1e-3"),

    # ── Tier 4: SGDR schedule (matches DAWN 20M cycle) ───────────────────────
    ("t13_l4_rope_sgdr",         4, 8,  "rope", "gelu",   3e-4, SGDR_TOKENS, "L=4  RoPE  GELU   SGDR"),
    ("t14_l6_rope_swiglu_sgdr",  6, 8,  "rope", "swiglu", 3e-4, SGDR_TOKENS, "L=6  RoPE  SwiGLU SGDR"),
    ("t15_l6_swiglu_lr6e4_sgdr", 6, 8,  "rope", "swiglu", 6e-4, SGDR_TOKENS, "L=6  RoPE  SwiGLU lr=6e-4  SGDR"),
    ("t16_l8_swiglu_lr6e4_sgdr", 8, 8,  "rope", "swiglu", 6e-4, SGDR_TOKENS, "L=8  RoPE  SwiGLU lr=6e-4  SGDR"),

    # ── Tier 5: n_heads sweep (l6_rope_swiglu) ───────────────────────────────
    ("t17_l6_swiglu_h4",     6, 4,  "rope",    "swiglu", 3e-4, None,        "L=6  RoPE  SwiGLU  h=4"),
    ("t18_l6_swiglu_h16",    6, 16, "rope",    "swiglu", 3e-4, None,        "L=6  RoPE  SwiGLU  h=16"),

    # ── Tier 6: Best combo + SGDR (fill in after tiers 1–5) ──────────────────
    ("t19_l6_swiglu_lr6e4",  6, 8,  "rope",    "swiglu", 6e-4, None,        "L=6  RoPE  SwiGLU  lr=6e-4"),
    ("t20_l8_rope_swiglu",   8, 8,  "rope",    "swiglu", 6e-4, SGDR_TOKENS, "L=8  RoPE  SwiGLU  lr=6e-4  SGDR"),
]

VARIANTS    = {v[0]: v for v in _V}
VARIANT_IDS = [v[0] for v in _V]


# ── Model sizing ───────────────────────────────────────────────────────────────

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
    """Return (d_model, n_params) nearest to target.

    d_model is rounded to a multiple of 2*n_heads so that d_head = d_model/n_heads
    is always even — required by RoPE's rotate_half split.
    """
    step = 2 * n_heads
    lo, hi = step, 8192
    for _ in range(24):
        mid = ((lo + hi) // 2 // step) * step
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


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one(vid, model, config, train_loader, val_loader, device, logger=None):
    _, n_layers, n_heads, pos_encoding, activation, base_lr, sgdr_tokens, label = VARIANTS[vid]

    tcfg            = config["training"]
    seq_len         = config["data"]["seq_len"]
    tokens_per_step = tcfg["batch_size"] * seq_len

    max_tokens   = tcfg.get("max_tokens", MAX_TOKENS)
    max_steps    = max(1, int(max_tokens) // tokens_per_step)
    warmup_steps = max(1, max_steps // 20)    # 5% warmup
    sgdr_t0_steps = (max(1, sgdr_tokens // tokens_per_step)
                     if sgdr_tokens is not None else None)

    optimizer = AdamW(model.parameters(), lr=base_lr,
                      weight_decay=tcfg.get("weight_decay", 1e-4))
    grad_clip = tcfg.get("grad_clip", 1.0)

    log_tokens    = tcfg.get("log_tokens",  500_000)
    eval_tokens   = tcfg.get("eval_tokens", 5_000_000)
    log_interval  = max(1, log_tokens  // tokens_per_step)
    eval_interval = max(1, eval_tokens // tokens_per_step)

    vocab_size = config["data"]["vocab_size"]
    log2e      = math.log2(math.e)   # for bits-per-byte conversion

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
                "train/bpb":        loss.item() * log2e,
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
            val_ppl = compute_perplexity(val_loss)
            val_bpb = val_loss * log2e
            elapsed = (_time.time() - t0) / 60.0
            print(f"  step={step:6d} | tokens={tokens_seen/1e6:5.1f}M "
                  f"| val_ppl={val_ppl:.2f} | val_bpb={val_bpb:.4f} "
                  f"| lr={lr:.2e} | {elapsed:.1f}min")
            sys.stdout.flush()
            if logger:
                logger.log({
                    "val/loss":       val_loss,
                    "val/perplexity": val_ppl,
                    "val/bpb":        val_bpb,
                    "tokens":         tokens_seen,
                    "elapsed_min":    elapsed,
                }, step=step)

        step += 1

    elapsed = (_time.time() - t0) / 60.0
    print(f"  [{vid}] done in {elapsed:.1f} min")
    sys.stdout.flush()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="10M transformer hyperparameter sweep on WikiText-103.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", default=VARIANT_IDS,
                        help="Variant IDs or 'all'")
    parser.add_argument("--config",    default=DATA_CONFIG)
    parser.add_argument("--target",    type=int, default=TARGET_PARAMS,
                        help=f"Parameter target (default: {TARGET_PARAMS//1_000_000}M)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to a saved tokenizer .pkl — if not given, BPE is fit "
                             "on the WikiText-103 training split.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",   default="transformer-10m-sweep")
    parser.add_argument("--srun-prefix", default=None,
                        help="Print one srun command per variant instead of running.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs == ["all"]:
        args.runs = VARIANT_IDS
    unknown = [r for r in args.runs if r not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}")

    # ── srun mode ─────────────────────────────────────────────────────────
    if args.srun_prefix is not None:
        base = [sys.executable, "scripts/run_transformer_10m_sweep.py"]
        if args.wandb:          base += ["--wandb"]
        if args.wandb_offline:  base += ["--wandb-offline"]
        if args.max_tokens:     base += ["--max-tokens", str(args.max_tokens)]
        if args.tokenizer:      base += ["--tokenizer", args.tokenizer]
        base += ["--wandb-project", args.wandb_project,
                 "--wandb-group",   args.wandb_group,
                 "--config",        args.config,
                 "--target",        str(args.target)]
        for vid in args.runs:
            print(f"{args.srun_prefix} {' '.join(base)} --runs {vid}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    setup_logging()
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    config = get_config(args.config)
    if args.max_tokens:
        config["training"]["max_tokens"] = args.max_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    sys.stdout.flush()

    # Tokenizer: load from disk or fit BPE on WT103 training split
    if args.tokenizer:
        import pickle
        with open(args.tokenizer, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = build_tokenizer(config)

    vocab_size = tokenizer.vocab_size
    config["data"]["vocab_size"] = vocab_size

    train_ds, val_ds, _, _ = get_dataset(config, tokenizer)
    train_loader = make_dataloader(train_ds, config, shuffle=True)
    val_loader   = make_dataloader(val_ds,   config, shuffle=False)

    target_params = args.target
    seq_len       = config["data"]["seq_len"]
    print(f"\nParameter target: {target_params:,}  ({target_params/1e6:.1f}M)")
    print(f"Vocab size: {vocab_size:,}  |  seq_len: {seq_len}")

    # ── Dry-run ────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'─'*78}")
        print(f"  {'ID':<32} {'layers':>6} {'heads':>5} {'pos':>8} {'act':>7} "
              f"{'lr':>7} {'sgdr':>8} {'d_model':>8} {'params':>12}")
        print(f"{'─'*78}")
        for vid in args.runs:
            _, nl, nh, pos, act, lr, sgdr, label = VARIANTS[vid]
            d, np_ = match_size(target_params, vocab_size, nl, nh, seq_len, pos, act)
            sgdr_str = f"{sgdr//1_000_000}M" if sgdr else "cosine"
            pct = 100 * np_ / target_params
            print(f"  {vid:<32} {nl:>6} {nh:>5} {pos:>8} {act:>7} "
                  f"{lr:>7.0e} {sgdr_str:>8} {d:>8} {np_:>10,} ({pct:.1f}%)")
        print(f"{'─'*78}")
        print(f"  {len(args.runs)} variant(s) × "
              f"{config['training'].get('max_tokens', MAX_TOKENS)//1_000_000}M tokens")
        return

    # ── Run sweep ─────────────────────────────────────────────────────────
    n_total   = len(args.runs)
    budget_m  = config["training"].get("max_tokens", MAX_TOKENS) // 1_000_000
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Transformer 10M sweep — {n_total} variant(s) × {budget_m}M tokens each")
    print(f"  Parameter target: {target_params:,}  ({target_params/1e6:.1f}M)")
    print(f"  Device: {device}")
    print(f"{sep}")
    for i, vid in enumerate(args.runs, 1):
        _, nl, nh, pos, act, lr, sgdr, lbl = VARIANTS[vid]
        sgdr_str = f"SGDR {sgdr//1_000_000}M" if sgdr else "cosine"
        print(f"  {i:>2}/{n_total}  {vid:<36}  L={nl} h={nh} {pos:<8} {act:<6} "
              f"lr={lr:.0e}  {sgdr_str}")
    print(f"{sep}\n")
    sys.stdout.flush()

    passed, failed = [], []

    for run_idx, vid in enumerate(args.runs, 1):
        _, nl, nh, pos, act, lr, sgdr, label = VARIANTS[vid]
        d_model, n_params = match_size(target_params, vocab_size, nl, nh, seq_len, pos, act)

        print(f"\n{sep}")
        print(f"  [{run_idx}/{n_total}]  {vid}  —  {label}")
        print(f"  d_model={d_model}  params={n_params:,}  "
              f"(target {target_params:,}, Δ={n_params-target_params:+,}, "
              f"{100*n_params/target_params:.1f}%)")
        sys.stdout.flush()

        model = _make(vocab_size, d_model, nl, nh, seq_len, pos, act)

        run_cfg = {
            **config,
            "name": f"transformer-10m-{vid}",
            "training": {
                **config["training"],
                "lr": lr,
                "checkpoint_dir": f"checkpoints/transformer-10m-sweep/{vid}",
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
