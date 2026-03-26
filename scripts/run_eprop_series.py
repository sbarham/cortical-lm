#!/usr/bin/env python3
"""
scripts/run_eprop_series.py — e-prop learning rule ablation series.

Runs four experiments at 150M tokens (3 passes through TinyStories) to answer:

    Q1. What is the cost of switching from BPTT to e-prop on the same architecture?
        → BPTT-1f vs eprop-1f

    Q2. Does the cortical structure help e-prop specifically?
        → eprop-1f vs eprop-1a (minimal, no layering / STP / AdEx)

    Q3. Does the Hopfield module survive e-prop?  Does it still contribute?
        → eprop-1d (AdEx, no Hopfield) vs eprop-1f (AdEx + Hopfield)

All runs log to the same W&B group so curves can be overlaid.

Notes
-----
- e-prop requires inter_column_stp: false (STP complicates eligibility traces).
  This is applied as an override automatically.
- e-prop trainer uses max_steps; this script converts max_tokens -> max_steps
  using batch_size * seq_len as tokens-per-step.
- LR is reduced relative to BPTT (e-prop updates are noisier); override with
  --lr if you want to sweep.

Usage
-----
# Default: 150M tokens, batch 1024, W&B disabled
python scripts/run_eprop_series.py --tokenizer checkpoints/tokenizer.pkl

# With W&B, custom group
python scripts/run_eprop_series.py --tokenizer checkpoints/tokenizer.pkl --wandb --wandb-project cortex-lm --wandb-group eprop-2026-03-25

# Specific experiments only
python scripts/run_eprop_series.py --tokenizer checkpoints/tokenizer.pkl --runs bptt_1f eprop_1f

# Dry run
python scripts/run_eprop_series.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

# ── Experiment definitions ────────────────────────────────────────────────────

# Base config, seq_len, and tokens_per_step are needed to compute max_steps.
SEQ_LEN = 128

EXPERIMENTS = [
    {
        "id":          "bptt_1f",
        "label":       "BPTT 1f (ceiling)",
        "config":      "configs/phase1f_hopfield.yaml",
        "extra":       ["learning.rule=bptt", "name=eprop-series-bptt-1f", "training.lr=3e-4"],
        "ckpt_dir":    "checkpoints/eprop-series-bptt-1f",
    },
    {
        "id":          "eprop_rough_cos_1f",
        "label":       "e-prop-rough 1f + cosine LR",
        "config":      "configs/phase1f_hopfield.yaml",
        "extra":       ["learning.rule=eprop_approx", "learning.cosine_decay=true",
                        "synapse.inter_column_stp=false", "name=eprop-rough-cos-1f"],
        "ckpt_dir":    "checkpoints/eprop-rough-cos-1f",
    },
    {
        "id":          "eprop_rough_1f",
        "label":       "e-prop-rough 1f (flat LR)",
        "config":      "configs/phase1f_hopfield.yaml",
        "extra":       ["learning.rule=eprop_approx", "synapse.inter_column_stp=false", "name=eprop-rough-1f"],
        "ckpt_dir":    "checkpoints/eprop-rough-1f",
    },
    {
        "id":          "eprop_1f",
        "label":       "e-prop 1f (proper, flat LR)",
        "config":      "configs/phase1f_hopfield.yaml",
        "extra":       ["learning.rule=eprop", "synapse.inter_column_stp=false", "name=eprop-1f"],
        "ckpt_dir":    "checkpoints/eprop-1f",
    },
    {
        "id":          "eprop_cos_1f",
        "label":       "e-prop 1f (proper, cosine LR)",
        "config":      "configs/phase1f_hopfield.yaml",
        "extra":       ["learning.rule=eprop", "learning.cosine_decay=true",
                        "synapse.inter_column_stp=false", "name=eprop-cos-1f"],
        "ckpt_dir":    "checkpoints/eprop-cos-1f",
    },
    {
        "id":          "eprop_1d",
        "label":       "e-prop-rough 1d (AdEx, no Hopfield — does HPC help e-prop?)",
        "config":      "configs/phase1d_adex.yaml",
        "extra":       ["learning.rule=eprop_approx", "synapse.inter_column_stp=false", "name=eprop-rough-1d"],
        "ckpt_dir":    "checkpoints/eprop-rough-1d",
    },
    {
        "id":          "eprop_1a",
        "label":       "e-prop-rough 1a (minimal — does cortical structure help e-prop?)",
        "config":      "configs/phase1a_minimal.yaml",
        "extra":       ["learning.rule=eprop_approx", "synapse.inter_column_stp=false", "name=eprop-rough-1a"],
        "ckpt_dir":    "checkpoints/eprop-rough-1a",
    },
]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    tokens_per_step = args.batch_size * SEQ_LEN
    max_steps = args.max_tokens // tokens_per_step

    overrides = [
        f"training.batch_size={args.batch_size}",
        f"training.max_steps={max_steps}",    # used by e-prop trainer
        f"training.max_tokens={args.max_tokens}",  # used by BPTT trainer
        f"training.lr={args.lr}",
        f"training.checkpoint_dir={exp['ckpt_dir']}",
    ]
    if args.tokenizer:
        overrides.append(f"data.tokenizer_path={args.tokenizer}")
    if args.wandb:
        overrides += [
            f"logging.project={args.wandb_project}",
            f"logging.group={args.wandb_group}",
        ]
    overrides += exp["extra"]

    cmd = [sys.executable, "scripts/train.py", "--config", exp["config"]]
    if args.wandb:
        cmd.append("--wandb")
    cmd += ["--override"] + overrides
    return cmd


def run_experiment(exp: dict, cmd: list[str], dry_run: bool) -> bool:
    print(f"\n{'='*70}")
    print(f"  [{exp['id']}]  {exp['label']}")
    print(f"  Config:  {exp['config']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  (dry run — skipping)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{exp['id']}] FAILED (exit {result.returncode}) after {elapsed/60:.1f} min")
        return False

    print(f"\n  ✓  [{exp['id']}] completed in {elapsed/60:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser(description="e-prop learning rule ablation series")

    parser.add_argument("--tokenizer", default=None,
                        help="Path to pre-trained tokenizer (recommended)")
    parser.add_argument("--max-tokens", type=int, default=150_000_000,
                        help="Token budget per run (default: 150M = ~3x TinyStories)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default 1e-4 — lower than BPTT for e-prop stability)")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Subset of experiment IDs to run (default: all). "
                             f"Choices: {[e['id'] for e in EXPERIMENTS]}")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default="eprop-2026-03-25")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.runs:
        ids = {e["id"] for e in EXPERIMENTS}
        for r in args.runs:
            if r not in ids:
                print(f"Unknown experiment id '{r}'. Choices: {sorted(ids)}")
                sys.exit(1)
        selected = [e for e in EXPERIMENTS if e["id"] in args.runs]

    tokens_per_step = args.batch_size * SEQ_LEN
    max_steps = args.max_tokens // tokens_per_step

    print(f"\ne-prop ablation series")
    print(f"  Experiments : {[e['id'] for e in selected]}")
    print(f"  Token budget: {args.max_tokens:,} tokens per run")
    print(f"  Steps       : {max_steps:,} steps  (batch={args.batch_size}, seq={SEQ_LEN})")
    print(f"  W&B         : {'enabled → ' + args.wandb_group if args.wandb else 'disabled'}")

    failed = []
    for exp in selected:
        cmd = build_command(exp, args)
        ok = run_experiment(exp, cmd, args.dry_run)
        if not ok:
            failed.append(exp["id"])

    print(f"\n{'='*70}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"  All {'(dry-run) ' if args.dry_run else ''}experiments completed.")


if __name__ == "__main__":
    main()
