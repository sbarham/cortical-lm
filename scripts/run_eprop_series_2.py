#!/usr/bin/env python3
"""
scripts/run_eprop_series_2.py — e-prop diagnostic and fix series.

Motivation
----------
Series 1 showed that both eprop_approx and eprop (proper) exhibit val ppl
*divergence* throughout training — recurrent weights are updated in a direction
that hurts generalisation even as training loss slowly improves.  The scalar vs
vector learning signal made no difference, ruling out signal directionality as
the cause.

This series systematically tests four hypotheses for why the recurrent updates
are harmful, each exposed as a config flag:

    freeze_recurrent  (diagnostic)
        Skip all recurrent weight updates; only train readout + embedding.
        If val divergence stops: the recurrent e-prop updates are definitively
        the cause.  If val still diverges: something upstream (e.g. the
        STP=false override changing the architecture) is responsible.

    adam_recurrent  (fix 1)
        Use Adam with moment estimates for recurrent weight updates instead of
        direct param.data -= lr * g.  The direct update is poorly scaled
        relative to the readout/embedding optimizers (which use Adam), and has
        no per-parameter adaptivity.

    dale_interval=10  (fix 2)
        Enforce Dale's law only every 10 inner timesteps instead of after every
        single token.  Frequent clipping may be immediately undoing each e-prop
        update before it can accumulate, driving oscillation.

    adam_recurrent + dale_interval=10  (fix 1+2 combined)
        Test whether the fixes are additive.

    tau_e=2  (fix 3 — short traces)
        Very short eligibility trace timescale: γ = exp(-1/2) ≈ 0.61.
        Tests whether the default trace timescale (geometric mean of tau_m range,
        ~8 ms) is too long, causing stale pre/post correlations to accumulate.

    tau_e=50  (fix 3 — long traces)
        Long trace timescale: γ = exp(-1/50) ≈ 0.98.
        Tests whether longer temporal credit assignment helps.

All runs use proper e-prop (learning.rule=eprop) as the base, since series 1
showed rough and proper behave identically — any fix that works will be
visible on proper e-prop.

Usage
-----
# Full series (6 experiments, ~6× 2 hours each)
python scripts/run_eprop_series_2.py \\
    --tokenizer checkpoints/tokenizer.pkl \\
    --wandb --wandb-project cortex-lm --wandb-group eprop2-YYYY-MM-DD

# Diagnostic first, then fixes if it confirms the recurrent updates are the cause
python scripts/run_eprop_series_2.py \\
    --tokenizer checkpoints/tokenizer.pkl \\
    --wandb --wandb-project cortex-lm --wandb-group eprop2-YYYY-MM-DD \\
    --runs freeze_1f adam_rec_1f dale_1f adam_dale_1f

# Dry run
python scripts/run_eprop_series_2.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

SEQ_LEN = 128

EXPERIMENTS = [
    {
        "id":       "freeze_readout_1f",
        "label":    "freeze readout+embed — diagnostic (only recurrent weights trained via e-prop)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.freeze_readout=true",
                     "synapse.inter_column_stp=false", "name=eprop-freeze-readout-1f"],
        "ckpt_dir": "checkpoints/eprop-freeze-readout-1f",
    },
    {
        "id":       "freeze_1f",
        "label":    "freeze recurrent — diagnostic (only readout+embed trained)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.freeze_recurrent=true",
                     "synapse.inter_column_stp=false", "name=eprop-freeze-1f"],
        "ckpt_dir": "checkpoints/eprop-freeze-1f",
    },
    {
        "id":       "adam_rec_1f",
        "label":    "Adam for recurrent updates (fix 1)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.adam_recurrent=true",
                     "synapse.inter_column_stp=false", "name=eprop-adam-rec-1f"],
        "ckpt_dir": "checkpoints/eprop-adam-rec-1f",
    },
    {
        "id":       "dale_1f",
        "label":    "Dale enforcement every 10 steps (fix 2)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.dale_interval=10",
                     "synapse.inter_column_stp=false", "name=eprop-dale10-1f"],
        "ckpt_dir": "checkpoints/eprop-dale10-1f",
    },
    {
        "id":       "adam_dale_1f",
        "label":    "Adam recurrent + Dale interval=10 (fix 1+2)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.adam_recurrent=true",
                     "learning.dale_interval=10",
                     "synapse.inter_column_stp=false", "name=eprop-adam-dale-1f"],
        "ckpt_dir": "checkpoints/eprop-adam-dale-1f",
    },
    {
        "id":       "short_tau_1f",
        "label":    "Short eligibility trace tau_e=2 (fix 3a)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.eprop_tau_e=2",
                     "synapse.inter_column_stp=false", "name=eprop-tau2-1f"],
        "ckpt_dir": "checkpoints/eprop-tau2-1f",
    },
    {
        "id":       "long_tau_1f",
        "label":    "Long eligibility trace tau_e=50 (fix 3b)",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["learning.rule=eprop", "learning.eprop_tau_e=50",
                     "synapse.inter_column_stp=false", "name=eprop-tau50-1f"],
        "ckpt_dir": "checkpoints/eprop-tau50-1f",
    },
]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    tokens_per_step = args.batch_size * SEQ_LEN
    max_steps = args.max_tokens // tokens_per_step

    overrides = [
        f"training.batch_size={args.batch_size}",
        f"training.max_steps={max_steps}",
        f"training.max_tokens={args.max_tokens}",
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
    parser = argparse.ArgumentParser(description="e-prop diagnostic and fix series (series 2)")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-tokens", type=int, default=150_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--runs", nargs="*", default=None,
                        help=f"Subset of IDs (default: all). "
                             f"Choices: {[e['id'] for e in EXPERIMENTS]}")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default=f"eprop2-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--stop-on-failure", action="store_true")
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

    print(f"\ne-prop series 2 — diagnostics and fixes")
    print(f"  Experiments : {[e['id'] for e in selected]}")
    print(f"  Token budget: {args.max_tokens:,} per run")
    print(f"  Steps       : {max_steps:,}  (batch={args.batch_size}, seq={SEQ_LEN})")
    print(f"  W&B         : {'enabled → ' + args.wandb_group if args.wandb else 'disabled'}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    failed = []
    for exp in selected:
        cmd = build_command(exp, args)
        ok = run_experiment(exp, cmd, args.dry_run)
        if not ok:
            failed.append(exp["id"])
            if args.stop_on_failure:
                print("  --stop-on-failure: aborting.")
                break

    print(f"\n{'='*70}")
    if failed:
        print(f"  FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"  All {'(dry-run) ' if args.dry_run else ''}experiments completed.")


if __name__ == "__main__":
    main()
