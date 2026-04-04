#!/usr/bin/env python3
"""
scripts/run_wakesleep_sweep.py — Wake-sleep (e-prop hybrid) sweep for scale_5m architecture.

Sweeps the (eprop_steps, bptt_steps, eprop_batch, bptt_batch, freeze_xi) space
on the Run 5 combined architecture (L6 shortcut + full HPC fix).

All variants use:
    learning.rule = eprop_hybrid
    hybrid_bptt_scope = full
    reset_state_between_batches = true
    base config: configs/scale_5m_run5_combined.yaml

Variants
--------
v1   eprop=50  bptt=25  eb=16  bb=128  freeze_xi=false  ← START HERE (conservative)
v2   eprop=20  bptt=10  eb=8   bb=128  freeze_xi=false  ← TinyStories analog at scale
v3   eprop=100 bptt=50  eb=16  bb=128  freeze_xi=false  ← longer cycles
v4   eprop=50  bptt=25  eb=16  bb=64   freeze_xi=false  ← smaller bptt batch
v5   eprop=50  bptt=25  eb=16  bb=512  freeze_xi=false  ← large bptt batch
v6   eprop=50  bptt=25  eb=16  bb=128  freeze_xi=true   ← freeze Xi during BPTT
v7   eprop=50  bptt=25  eb=32  bb=128  freeze_xi=false  ← larger eprop batch
v8   eprop=100 bptt=25  eb=16  bb=128  freeze_xi=false  ← 4:1 ratio
v9   eprop=50  bptt=50  eb=16  bb=128  freeze_xi=false  ← 1:1 ratio
v10  eprop=50  bptt=25  eb=16  bb=512  freeze_xi=true   ← freeze Xi + big bptt

Diagnostic checkpoint (all variants): hpc/attn_max at step 100.
    > 0.003 and stable  → HPC staying active, promising config
    back to 0.0014      → BPTT washing out Xi, try v6 next

Usage
-----
# Run conservative baseline first (recommended)
python scripts/run_wakesleep_sweep.py --runs v1 --wandb

# Run specific variants
python scripts/run_wakesleep_sweep.py --runs v1 v6 --wandb

# Full sweep (sequential, ~3h each at 25M tokens)
python scripts/run_wakesleep_sweep.py --runs all --wandb

# Dry run (print commands without executing)
python scripts/run_wakesleep_sweep.py --runs all --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

SEQ_LEN = 256
BASE_CONFIG = "configs/scale_5m_run5_combined.yaml"
TOKENIZER   = "tokenizers/wikitext103_bpe16k.pkl"

# (id, label, eprop_steps, bptt_steps, eprop_batch, bptt_batch, freeze_xi)
_VARIANTS = [
    ("v1",  "conservative baseline — START HERE",    50,  25, 16, 128, False),
    ("v2",  "TinyStories analog (20/10, eb=8)",       20,  10,  8, 128, False),
    ("v3",  "longer cycles (100/50)",                100,  50, 16, 128, False),
    ("v4",  "smaller bptt batch (bb=64)",             50,  25, 16,  64, False),
    ("v5",  "large bptt batch (bb=512)",              50,  25, 16, 512, False),
    ("v6",  "freeze Xi during BPTT",                  50,  25, 16, 128, True),
    ("v7",  "larger eprop batch (eb=32)",             50,  25, 32, 128, False),
    ("v8",  "4:1 ratio (100 eprop / 25 bptt)",       100,  25, 16, 128, False),
    ("v9",  "1:1 ratio (50 eprop / 50 bptt)",         50,  50, 16, 128, False),
    ("v10", "freeze Xi + big bptt batch",             50,  25, 16, 512, True),
]

VARIANTS = {v[0]: v for v in _VARIANTS}


def build_command(variant_id: str, args: argparse.Namespace) -> list[str]:
    vid, label, eprop_steps, bptt_steps, eprop_batch, bptt_batch, freeze_xi = VARIANTS[variant_id]

    max_steps = args.max_tokens // (eprop_batch * SEQ_LEN)
    run_name  = f"ws-{vid}-e{eprop_steps}-b{bptt_steps}-eb{eprop_batch}-bb{bptt_batch}"
    if freeze_xi:
        run_name += "-fxi"
    ckpt_dir = f"checkpoints/{run_name}"

    # Token-based eval/log intervals scaled to eprop batch
    log_interval  = max(1, 100_000  // (eprop_batch * SEQ_LEN))
    eval_interval = max(1, 500_000  // (eprop_batch * SEQ_LEN))
    ckpt_interval = max(1, 10_000_000 // (eprop_batch * SEQ_LEN))
    warmup_steps  = max(1, max_steps // 20)  # 5% warmup

    overrides = [
        # Architecture stays as Run 5 — just override learning rule and schedule
        f"learning.rule=eprop_hybrid",
        f"learning.reset_state_between_batches=true",
        f"learning.hybrid_eprop_steps={eprop_steps}",
        f"learning.hybrid_bptt_steps={bptt_steps}",
        f"learning.hybrid_bptt_batch_size={bptt_batch}",
        f"learning.hybrid_bptt_scope=full",
        f"learning.hybrid_freeze_xi={'true' if freeze_xi else 'false'}",
        f"training.batch_size={eprop_batch}",
        f"training.max_steps={max_steps}",
        f"training.max_tokens={args.max_tokens}",
        f"training.warmup_steps={warmup_steps}",
        f"training.log_tokens=100000",
        f"training.eval_tokens=500000",
        f"training.checkpoint_interval={ckpt_interval}",
        f"training.checkpoint_dir={ckpt_dir}",
        f"data.tokenizer_path={TOKENIZER}",
        f"name={run_name}",
    ]
    if args.wandb:
        overrides += [
            f"logging.wandb=true",
            f"logging.project={args.wandb_project}",
            f"logging.group={args.wandb_group}",
        ]

    cmd = [sys.executable, "scripts/train.py",
           "--config", BASE_CONFIG,
           "--tokenizer", TOKENIZER,
           "--override"] + overrides
    return cmd


def run_variant(variant_id: str, cmd: list[str], dry_run: bool) -> bool:
    vid, label, *rest = VARIANTS[variant_id]
    eprop_steps, bptt_steps, eprop_batch, bptt_batch, freeze_xi = rest
    print(f"\n{'='*72}")
    print(f"  [{vid}]  {label}")
    print(f"  eprop={eprop_steps}  bptt={bptt_steps}  "
          f"eprop_batch={eprop_batch}  bptt_batch={bptt_batch}  "
          f"freeze_xi={freeze_xi}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*72}")

    if dry_run:
        print("  (dry run — skipping)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{vid}] FAILED (exit {result.returncode}) after {elapsed/60:.1f} min")
        return False

    print(f"\n  ✓  [{vid}] completed in {elapsed/60:.1f} min")
    return True


def main():
    all_ids = [v[0] for v in _VARIANTS]

    parser = argparse.ArgumentParser(description="Wake-sleep sweep for scale_5m Run 5 architecture")
    parser.add_argument("--runs", nargs="+", default=["v1"],
                        help=f"Variant IDs to run, or 'all'. Choices: {all_ids}")
    parser.add_argument("--max-tokens", type=int, default=25_000_000,
                        help="Token budget per variant (default: 25M for fast diagnostic)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default=f"wakesleep-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs == ["all"]:
        args.runs = all_ids

    unknown = [r for r in args.runs if r not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variant(s): {unknown}. Choices: {all_ids}")

    print(f"\nWake-sleep sweep — {len(args.runs)} variant(s) × {args.max_tokens/1e6:.0f}M tokens")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Variants: {args.runs}")

    passed, failed = [], []
    for vid in args.runs:
        cmd = build_command(vid, args)
        ok  = run_variant(vid, cmd, args.dry_run)
        (passed if ok else failed).append(vid)
        if not ok and args.stop_on_failure:
            break

    print(f"\n{'='*72}")
    print(f"  Done: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*72}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
