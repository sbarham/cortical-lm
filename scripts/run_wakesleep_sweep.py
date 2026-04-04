#!/usr/bin/env python3
"""
scripts/run_wakesleep_sweep.py — Wake-sleep (e-prop hybrid) sweep for scale_5m architecture.

2×3 factorial design: cycle length × SGDR.

    Rows    — wake:sleep cycle length (eprop_steps:bptt_steps, eprop_batch=16, bptt_batch=128)
                short   20:10
                mid     50:25  ← best from first sweep
                long   100:50

    Columns — LR schedule
                flat     no SGDR (cosine decay to 0)
                sgdr12   SGDR restart every 12.5M tokens (8 cycles over 100M)
                sgdr25   SGDR restart every 25M tokens   (4 cycles over 100M)

All variants use:
    learning.rule = eprop_hybrid
    hybrid_bptt_scope = full
    reset_state_between_batches = true
    eprop_batch = 16, bptt_batch = 128
    base config: configs/scale_5m_run5_combined.yaml

Grid (3×3)
----------
         flat      sgdr12     sgdr25
short    v1        v2         v3
mid      v4        v5         v6    ← v4 replicates winning ratio from prior sweep
long     v7        v8         v9

Diagnostic (early stopping check): hpc/attn_max at step 100
    > 0.003 and stable  → HPC active
    back to 0.0014      → HPC collapsed

Usage
-----
# Run recommended starting point (replicate + SGDR)
python scripts/run_wakesleep_sweep.py --runs v3 v4 --wandb

# Full 2×3 factorial
python scripts/run_wakesleep_sweep.py --runs all --wandb

# Dry run
python scripts/run_wakesleep_sweep.py --runs all --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

SEQ_LEN          = 256
BASE_CONFIG      = "configs/scale_5m_run5_combined.yaml"
TOKENIZER        = "tokenizers/wikitext103_bpe16k.pkl"
EPROP_BATCH      = 16
BPTT_BATCH       = 128
SGDR_TOKENS_12M  = 12_500_000   # restart every 12.5M tokens (8 restarts over 100M)
SGDR_TOKENS_25M  = 25_000_000   # restart every 25M tokens  (4 restarts over 100M)

# (id, label, eprop_steps, bptt_steps, sgdr_tokens)
# sgdr_tokens=None → flat cosine decay
_VARIANTS = [
    ("v1", "short  flat   (20:10, no SGDR)",          20,  10, None),
    ("v2", "short  sgdr12 (20:10, SGDR 12.5M)",       20,  10, SGDR_TOKENS_12M),
    ("v3", "short  sgdr25 (20:10, SGDR 25M)",         20,  10, SGDR_TOKENS_25M),
    ("v4", "mid    flat   (50:25, no SGDR)",           50,  25, None),
    ("v5", "mid    sgdr12 (50:25, SGDR 12.5M)",        50,  25, SGDR_TOKENS_12M),
    ("v6", "mid    sgdr25 (50:25, SGDR 25M)",          50,  25, SGDR_TOKENS_25M),
    ("v7", "long   flat   (100:50, no SGDR)",         100,  50, None),
    ("v8", "long   sgdr12 (100:50, SGDR 12.5M)",      100,  50, SGDR_TOKENS_12M),
    ("v9", "long   sgdr25 (100:50, SGDR 25M)",        100,  50, SGDR_TOKENS_25M),
]

VARIANTS = {v[0]: v for v in _VARIANTS}


def build_command(variant_id: str, args: argparse.Namespace) -> list[str]:
    vid, label, eprop_steps, bptt_steps, sgdr_tokens = VARIANTS[variant_id]

    max_steps    = args.max_tokens // (EPROP_BATCH * SEQ_LEN)
    sgdr_suffix  = f"-sgdr{sgdr_tokens // 1_000_000}m" if sgdr_tokens else ""
    run_name     = f"ws-{vid}-e{eprop_steps}-b{bptt_steps}{sgdr_suffix}"
    ckpt_dir     = f"checkpoints/{run_name}"
    warmup_steps = max(1, max_steps // 20)   # 5% warmup

    overrides = [
        f"learning.rule=eprop_hybrid",
        f"learning.reset_state_between_batches=true",
        f"learning.hybrid_eprop_steps={eprop_steps}",
        f"learning.hybrid_bptt_steps={bptt_steps}",
        f"learning.hybrid_bptt_batch_size={BPTT_BATCH}",
        f"learning.hybrid_bptt_scope=full",
        f"learning.hybrid_freeze_xi=false",
        f"training.batch_size={EPROP_BATCH}",
        f"training.max_steps={max_steps}",
        f"training.max_tokens={args.max_tokens}",
        f"training.warmup_steps={warmup_steps}",
        f"training.log_tokens=100000",
        f"training.eval_tokens=500000",
        f"training.checkpoint_dir={ckpt_dir}",
        f"data.tokenizer_path={TOKENIZER}",
        f"name={run_name}",
    ]
    if sgdr_tokens:
        overrides.append(f"learning.sgdr_restart_tokens={sgdr_tokens}")
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


def run_variant(variant_id: str, cmd: list[str], dry_run: bool, args: argparse.Namespace) -> bool:
    vid, label, eprop_steps, bptt_steps, sgdr_tokens = VARIANTS[variant_id]
    sgdr_str = f"{sgdr_tokens // 1_000_000}M" if sgdr_tokens else "none"
    print(f"\n{'='*72}")
    print(f"  [{vid}]  {label}")
    print(f"  eprop={eprop_steps}  bptt={bptt_steps}  "
          f"eprop_batch={EPROP_BATCH}  bptt_batch={BPTT_BATCH}  "
          f"sgdr={sgdr_str}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*72}")

    if dry_run:
        print("  (dry run — skipping)")
        return True

    env = os.environ.copy()
    if args.wandb_offline:
        env["WANDB_MODE"] = "offline"

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{vid}] FAILED (exit {result.returncode}) after {elapsed/60:.1f} min")
        return False

    print(f"\n  [{vid}] completed in {elapsed/60:.1f} min")
    return True


def main():
    all_ids = [v[0] for v in _VARIANTS]

    parser = argparse.ArgumentParser(
        description="Wake-sleep 3×3 factorial sweep (cycle length × SGDR)"
    )
    parser.add_argument("--runs", nargs="+", default=["v4", "v5", "v6"],
                        help=f"Variant IDs to run, or 'all'. Choices: {all_ids}")
    parser.add_argument("--max-tokens", type=int, default=100_000_000,
                        help="Token budget per variant (default: 100M)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Set WANDB_MODE=offline (log locally, sync later)")
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

    print(f"\nWake-sleep 3×3 sweep — {len(args.runs)} variant(s) × {args.max_tokens/1e6:.0f}M tokens")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Variants: {args.runs}")

    passed, failed = [], []
    for vid in args.runs:
        cmd = build_command(vid, args)
        ok  = run_variant(vid, cmd, args.dry_run, args)
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
