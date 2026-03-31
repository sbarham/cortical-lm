#!/usr/bin/env python3
"""
scripts/run_hopfield_apical_sweep.py — Hopfield + apical pathway sweep.

Motivation
----------
The new clean canonical series (1a–1i, no apical, single cosine LR) shows that
AdEx (1d) strongly dominates all Hopfield variants (1f–1i), which cluster
together and add little over AdEx alone.  This contradicts old canonical results
where Hopfield gave large gains — the key difference is that the old phase1f
config had `column.apical_pathway: additive` baked in.

This sweep tests whether the additive apical pathway is what unlocks the
Hopfield module's contribution, and whether CA1 adds anything on top.

Experiments
-----------
    1d_apical       AdEx + apical (reference: does apical help AdEx too?)
    1f_apical       Hopfield/CA3 + apical, cosine LR
    1i_apical       Hopfield + CA1 + apical, cosine LR
    1f_apical_sgdr  Hopfield/CA3 + apical, SGDR (warm restarts)
    1i_apical_sgdr  Hopfield + CA1 + apical, SGDR

The 1d_apical reference is critical: if Hopfield+apical > AdEx+apical, the
hippocampal module is genuinely contributing beyond what AdEx provides.  If
Hopfield+apical ≈ AdEx+apical, apical is doing the work regardless of HPC.

SGDR schedule: CosineAnnealingWarmRestarts, T_0 = ~10% of total steps,
T_mult = 2 (progressively longer cycles).  Configured via training.scheduler=sgdr.

Usage
-----
# Full sweep
python scripts/run_hopfield_apical_sweep.py \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb --wandb-project cortex-lm

# Cosine runs only first (then add SGDR if they look promising)
python scripts/run_hopfield_apical_sweep.py \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb --wandb-project cortex-lm \\
    --runs 1d_apical 1f_apical 1i_apical

# Dry run
python scripts/run_hopfield_apical_sweep.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

EXPERIMENTS = [
    {
        "id":       "1d_apical",
        "label":    "AdEx + apical (reference: does apical help AdEx?)",
        "config":   "configs/phase1d_adex.yaml",
        "extra":    ["column.apical_pathway=additive", "name=hopfield-apical-1d-cos"],
        "ckpt_dir": "checkpoints/hopfield-apical/1d-cos",
    },
    {
        "id":       "1f_apical",
        "label":    "Hopfield/CA3 + apical, cosine LR",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["column.apical_pathway=additive", "name=hopfield-apical-1f-cos"],
        "ckpt_dir": "checkpoints/hopfield-apical/1f-cos",
    },
    {
        "id":       "1i_apical",
        "label":    "Hopfield + CA1 + apical, cosine LR",
        "config":   "configs/phase1i_hopfield_ca1.yaml",
        "extra":    ["column.apical_pathway=additive", "name=hopfield-apical-1i-cos"],
        "ckpt_dir": "checkpoints/hopfield-apical/1i-cos",
    },
    {
        "id":       "1f_apical_sgdr",
        "label":    "Hopfield/CA3 + apical, SGDR",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["column.apical_pathway=additive", "training.scheduler=sgdr",
                     "name=hopfield-apical-1f-sgdr"],
        "ckpt_dir": "checkpoints/hopfield-apical/1f-sgdr",
    },
    {
        "id":       "1i_apical_sgdr",
        "label":    "Hopfield + CA1 + apical, SGDR",
        "config":   "configs/phase1i_hopfield_ca1.yaml",
        "extra":    ["column.apical_pathway=additive", "training.scheduler=sgdr",
                     "name=hopfield-apical-1i-sgdr"],
        "ckpt_dir": "checkpoints/hopfield-apical/1i-sgdr",
    },
]

EXP_IDS = [e["id"] for e in EXPERIMENTS]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--config", exp["config"], "--wandb"]

    overrides = [
        f"training.checkpoint_dir={exp['ckpt_dir']}",
        f"logging.project={args.wandb_project}",
        f"logging.group={args.wandb_group}",
        f"training.max_tokens={args.max_tokens}",
    ]
    if args.tokenizer:
        overrides.append(f"data.tokenizer_path={args.tokenizer}")
    if args.batch_size:
        overrides.append(f"training.batch_size={args.batch_size}")
    overrides += exp["extra"]

    for ov in (args.override or []):
        overrides.append(ov)

    cmd += ["--override"] + overrides
    return cmd


def run_experiment(exp: dict, cmd: list[str], dry_run: bool) -> bool:
    print(f"\n{'='*70}")
    print(f"  [{exp['id']}]  {exp['label']}")
    print(f"  Config : {exp['config']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  (dry run — skipping)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{exp['id']}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  ✓  [{exp['id']}] completed in {elapsed/3600:.1f}h")
    return True


def main():
    parser = argparse.ArgumentParser(description="Hopfield + apical pathway sweep")
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl")
    parser.add_argument("--max-tokens", type=int, default=150_000_000,
                        help="Token budget per run (default: 150M). Use 1_000_000_000 for full canonical.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from config, 512)")
    parser.add_argument("--runs", nargs="+", choices=EXP_IDS, default=None,
                        help=f"Subset of experiments (default: all). Choices: {EXP_IDS}")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"hopfield-apical-sweep-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [e for e in EXPERIMENTS if e["id"] in args.runs] if args.runs else EXPERIMENTS

    print(f"\nHopfield + apical sweep")
    print(f"  Experiments : {[e['id'] for e in selected]}")
    print(f"  W&B group   : {args.wandb_group}")
    print(f"  Tokenizer   : {args.tokenizer or '(from config)'}")
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
    print(f"  Summary: {len(selected)-len(failed)}/{len(selected)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
