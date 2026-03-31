#!/usr/bin/env python3
"""
run_apical_ablation.py — Sequential runner for the apical stream ablation series.

Runs all five apical_pathway variants (none, skip, additive, multiplicative,
corticortical) on the full Phase-1g architecture (AdEx + STP + annealed
disinhibition + Hopfield).  Each variant differs only in apical_pathway.

After all five complete, the winning variant is identified and a suggestion
printed for running the canonical 1a-1g series with that variant applied.

Usage
-----
# Run all five variants (uses tokenizer from 1a checkpoint if present)
python scripts/run_apical_ablation.py

# With W&B logging
python scripts/run_apical_ablation.py --wandb-project cortex-lm --wandb-group apical-ablation

# Re-use a tokenizer from an existing canonical run (recommended -- same vocab)
python scripts/run_apical_ablation.py --tokenizer checkpoints/phase1a/tokenizer.pkl

# Run only a subset
python scripts/run_apical_ablation.py --variants none additive multiplicative

# Dry run (print commands, no execution)
python scripts/run_apical_ablation.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = [
    ("none",           "configs/apical_none.yaml"),
    ("skip",           "configs/apical_skip.yaml"),
    ("additive",       "configs/apical_additive.yaml"),
    ("multiplicative", "configs/apical_multiplicative.yaml"),
    ("corticortical",  "configs/apical_corticortical.yaml"),
]

VARIANT_IDS = [v for v, _ in VARIANTS]


def parse_args():
    p = argparse.ArgumentParser(description="Run apical stream ablation series")
    p.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl",
                   help="Path to pre-trained tokenizer.pkl. Highly recommended to "
                        "reuse the same tokenizer as the canonical series for "
                        "comparable perplexities.")
    p.add_argument("--variants", nargs="+", choices=VARIANT_IDS, default=None,
                   help="Run only these variants (default: all five)")
    p.add_argument("--checkpoint-dir", default="checkpoints/apical",
                   help="Root checkpoint directory; each variant gets a subdirectory")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name (enables W&B logging)")
    p.add_argument("--wandb-group", default="apical-ablation",
                   help="W&B group name (default: apical-ablation)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size in all configs")
    p.add_argument("--override", nargs="*", default=[],
                   help="Extra key=value overrides forwarded to train.py (e.g. training.lr=1e-3)")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="Abort remaining variants if one fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them")
    return p.parse_args()


def build_command(variant_id: str, config_path: str, args, tokenizer_path: str | None) -> list[str]:
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config_path,
        "--override",
        f"training.checkpoint_dir=checkpoints/apical/{variant_id}",
    ]

    overrides = list(args.override)

    if tokenizer_path:
        overrides.append(f"data.tokenizer_path={tokenizer_path}")

    if args.batch_size:
        overrides.append(f"training.batch_size={args.batch_size}")

    if args.wandb_project:
        overrides.append(f"logging.wandb=true")
        overrides.append(f"logging.project={args.wandb_project}")
        overrides.append(f"logging.group={args.wandb_group}")

    if overrides:
        cmd += overrides

    return cmd


def main():
    args = parse_args()

    variants_to_run = [(v, c) for v, c in VARIANTS
                       if args.variants is None or v in args.variants]

    print(f"Apical ablation series: {len(variants_to_run)} variant(s)")
    print(f"  Variants : {[v for v, _ in variants_to_run]}")
    if args.tokenizer:
        print(f"  Tokenizer: {args.tokenizer}")
    if args.wandb_project:
        print(f"  W&B      : {args.wandb_project}/{args.wandb_group}")
    print()

    results: dict[str, dict] = {}
    tokenizer_path = args.tokenizer

    for variant_id, config_path in variants_to_run:
        print(f"{'='*60}")
        print(f"  Variant: {variant_id}  ({config_path})")
        print(f"{'='*60}")

        cmd = build_command(variant_id, config_path, args, tokenizer_path)

        if args.dry_run:
            print("  [DRY RUN] Would run:")
            print("  " + " ".join(cmd))
            results[variant_id] = {"status": "dry-run", "elapsed": 0.0}
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            results[variant_id] = {"status": "ok", "elapsed": elapsed}
            print(f"\n  [OK] {variant_id} finished in {elapsed/3600:.1f}h\n")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            results[variant_id] = {"status": "failed", "elapsed": elapsed, "returncode": e.returncode}
            print(f"\n  [FAILED] {variant_id} returned exit code {e.returncode}\n")
            if args.stop_on_failure:
                print("Aborting (--stop-on-failure)")
                break

        # After the first variant, look for its tokenizer for subsequent runs
        if tokenizer_path is None:
            ckpt = Path(f"checkpoints/apical/{variant_id}/tokenizer.pkl")
            if ckpt.exists():
                tokenizer_path = str(ckpt)
                print(f"  Reusing tokenizer from {tokenizer_path} for remaining variants\n")

    # Summary
    print(f"\n{'='*60}")
    print("  Apical Ablation — Summary")
    print(f"{'='*60}")
    for variant_id, info in results.items():
        status = info["status"]
        elapsed_h = info["elapsed"] / 3600
        print(f"  {variant_id:<20} {status:<10} {elapsed_h:.1f}h")

    ok_variants = [v for v, i in results.items() if i["status"] == "ok"]
    if ok_variants:
        print()
        print("  Next steps:")
        print("  1. Compare val/perplexity curves in W&B (group: apical-ablation)")
        print("  2. Choose winning variant (likely additive or multiplicative)")
        print("  3. Run canonical series 1a-1g with the winning apical variant:")
        print()
        print("     python scripts/run_canonical.py \\")
        print("       --wandb-project cortex-lm --wandb-group canonical-with-apical \\")
        print("       --override column.apical_pathway=<winner>")
        print()


if __name__ == "__main__":
    main()
