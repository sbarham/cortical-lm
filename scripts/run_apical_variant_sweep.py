#!/usr/bin/env python3
"""
scripts/run_apical_variant_sweep.py — Apical pathway variant sweep on a single architecture.

Purpose
-------
Given the winning architecture from run_apical_arch_sweep.py, sweep all five
apical pathway implementations to find the best variant:

    none           — no apical pathway (pure column feedforward)
    skip           — direct embedding → L5E skip connection (residual stream)
    additive       — gated additive (Larkum apical; gate learns to open)
    multiplicative — multiplicative gain (Larkum calcium spike)
    corticortical  — circular column→column feedback via apical dendrites

Each variant is applied by overriding `column.apical_pathway` on the base config,
so a single base config drives all five runs.

Usage
-----
# Sweep all five variants on the winning architecture
python scripts/run_apical_variant_sweep.py \\
    --base-config configs/phase1f_hopfield.yaml \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb-project cortex-lm

# Subset of variants
python scripts/run_apical_variant_sweep.py \\
    --base-config configs/phase1f_hopfield.yaml \\
    --variants none additive multiplicative \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl

# Dry run
python scripts/run_apical_variant_sweep.py \\
    --base-config configs/phase1f_hopfield.yaml \\
    --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

VARIANTS = ["none", "skip", "additive", "multiplicative", "corticortical"]


def build_command(variant: str, base_config: str, args, tokenizer: str | None) -> list[str]:
    # Derive a short name from the base config filename
    base_name = os.path.splitext(os.path.basename(base_config))[0]
    run_name = f"apical-variant-{base_name}-{variant}"
    ckpt_dir = f"checkpoints/apical-variant-sweep/{base_name}/{variant}"

    cmd = [sys.executable, "scripts/train.py", "--config", base_config, "--wandb"]

    overrides = [
        f"column.apical_pathway={variant}",
        f"training.checkpoint_dir={ckpt_dir}",
        f"name={run_name}",
        f"logging.project={args.wandb_project}",
        f"logging.group={args.wandb_group}",
    ]
    if tokenizer:
        overrides.append(f"data.tokenizer_path={tokenizer}")
    if args.batch_size:
        overrides.append(f"training.batch_size={args.batch_size}")
    for ov in (args.override or []):
        overrides.append(ov)

    cmd += ["--override"] + overrides
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Apical pathway variant sweep")
    parser.add_argument("--base-config", required=True,
                        help="Base config for the winning architecture "
                             "(e.g. configs/phase1f_hopfield.yaml)")
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl",
                        help="Pre-trained tokenizer.pkl")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None,
                        help=f"Variants to run (default: all). Choices: {VARIANTS}")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default=f"apical-variant-sweep-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.variants or VARIANTS

    base_name = os.path.splitext(os.path.basename(args.base_config))[0]
    print(f"\nApical variant sweep")
    print(f"  Base config : {args.base_config}  ({base_name})")
    print(f"  Variants    : {selected}")
    print(f"  W&B group   : {args.wandb_group}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    results = []

    for variant in selected:
        cmd = build_command(variant, args.base_config, args, args.tokenizer)
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Variant: {variant}  |  base: {base_name}")
        print(f"  {' '.join(cmd)}")
        print(f"{sep}\n", flush=True)

        if args.dry_run:
            results.append((variant, True, 0.0))
            continue

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        results.append((variant, ok, elapsed))

        if not ok:
            print(f"\n  !! Variant '{variant}' FAILED")
            if args.stop_on_failure:
                print("  --stop-on-failure: aborting.")
                break

    # Summary
    print(f"\n{'='*70}")
    print(f"  Apical variant sweep — summary  (base: {base_name})")
    print(f"{'='*70}")
    for variant, ok, elapsed in results:
        tag = "OK  " if ok else "FAIL"
        t = f"{elapsed/3600:.1f}h" if elapsed > 0 else "dry-run"
        print(f"  [{tag}]  {variant:<20} ({t})")

    ok_variants = [v for v, ok, _ in results if ok]
    if ok_variants:
        print(f"\n  Compare val/perplexity in W&B group '{args.wandb_group}'")
        print(f"  Then rerun full canonical 1a–1i with the winning variant:")
        print(f"    python scripts/run_canonical.py \\")
        print(f"      --tokenizer <tokenizer> \\")
        print(f"      --wandb-project {args.wandb_project} \\")
        print(f"      --override column.apical_pathway=<winner>")
    print()

    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
