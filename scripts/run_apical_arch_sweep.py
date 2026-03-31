#!/usr/bin/env python3
"""
scripts/run_apical_arch_sweep.py — Phase 1 architecture sweep with additive apical.

Purpose
-------
Run every canonical phase (1a–1i) with `column.apical_pathway=additive` applied
as an override.  This answers:

    "Which architectures benefit from / can train with a residual-equivalent
     apical pathway?"

Architectures that are too complex to train without the gradient highway from
the apical pathway will diverge or stall here — those are filtered out before
the variant sweep.  The best-performing architecture from this sweep becomes
the base config for run_apical_variant_sweep.py.

The additive apical pathway is chosen as the screening variant because:
    - Best gradient health in exploratory runs (l23_to_l5 healthy ~0.5–1.0)
    - Gated — gate learns to open gradually, preserving some column-internal
      credit assignment
    - Competitive early convergence without dominating the gradient path

After this sweep, pick the winning phase and run:
    python scripts/run_apical_variant_sweep.py \\
        --base-config configs/phase1X_<winner>.yaml \\
        --tokenizer tokenizers/tinystories_bpe4096.pkl \\
        --wandb-project cortex-lm --wandb-group apical-variant-sweep-<date>

Usage
-----
# Full sweep 1a–1i with additive apical
python scripts/run_apical_arch_sweep.py --tokenizer tokenizers/tinystories_bpe4096.pkl --wandb-project cortex-lm

# Partial sweep (e.g. just Hopfield variants)
python scripts/run_apical_arch_sweep.py --tokenizer tokenizers/tinystories_bpe4096.pkl --phases 1f 1g 1h 1i

# Override batch size
python scripts/run_apical_arch_sweep.py --tokenizer tokenizers/tinystories_bpe4096.pkl --batch-size 2048

# Dry run
python scripts/run_apical_arch_sweep.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

PHASES = [
    ("1a", "configs/phase1a_minimal.yaml"),
    ("1b", "configs/phase1b_layered.yaml"),
    ("1c", "configs/phase1c_stp.yaml"),
    ("1d", "configs/phase1d_adex.yaml"),
    ("1e", "configs/phase1e_disinhibition.yaml"),
    ("1f", "configs/phase1f_hopfield.yaml"),
    ("1g", "configs/phase1g_hopfield_disinhibition.yaml"),
    ("1h", "configs/phase1h_hopfield_annealed.yaml"),
    ("1i", "configs/phase1i_hopfield_ca1.yaml"),
]
PHASE_IDS = [p for p, _ in PHASES]

APICAL_VARIANT = "additive"


def build_command(phase: str, config: str, args, tokenizer: str | None) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--config", config, "--wandb"]

    overrides = [
        f"column.apical_pathway={APICAL_VARIANT}",
        f"training.checkpoint_dir=checkpoints/apical-arch-sweep/{phase}",
        f"name=apical-arch-sweep-{phase}",
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
    parser = argparse.ArgumentParser(description="Phase 1 architecture sweep with additive apical")
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl",
                        help="Pre-trained tokenizer.pkl (recommended — keeps vocab constant)")
    parser.add_argument("--phases", nargs="+", choices=PHASE_IDS, default=None,
                        help="Run only these phases (default: all 1a–1i)")
    parser.add_argument("--start-from", default="1a", choices=PHASE_IDS,
                        help="Start from this phase (default: 1a)")
    parser.add_argument("--end-at", default=None, choices=PHASE_IDS,
                        help="Stop after this phase inclusive (default: 1i)")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default=f"apical-arch-sweep-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.phases:
        selected = [(p, c) for p, c in PHASES if p in args.phases]
    else:
        start_idx = PHASE_IDS.index(args.start_from)
        end_idx = PHASE_IDS.index(args.end_at) + 1 if args.end_at else len(PHASES)
        selected = PHASES[start_idx:end_idx]

    print(f"\nApical architecture sweep  (variant={APICAL_VARIANT})")
    print(f"  Phases    : {[p for p, _ in selected]}")
    print(f"  W&B group : {args.wandb_group}")
    print(f"  Tokenizer : {args.tokenizer or '(from phase 1a checkpoint)'}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    tokenizer = args.tokenizer
    results = []

    for phase, config in selected:
        cmd = build_command(phase, config, args, tokenizer)
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Phase {phase} + apical={APICAL_VARIANT}  |  {config}")
        print(f"  {' '.join(cmd)}")
        print(f"{sep}\n", flush=True)

        if args.dry_run:
            results.append((phase, True, 0.0))
            continue

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        results.append((phase, ok, elapsed))

        if not ok:
            print(f"\n  !! Phase {phase} FAILED")
            if args.stop_on_failure:
                print("  --stop-on-failure: aborting.")
                break

        # Reuse tokenizer written by phase 1a for all subsequent phases
        if phase == "1a" and tokenizer is None:
            candidate = f"checkpoints/apical-arch-sweep/1a/tokenizer.pkl"
            if os.path.exists(candidate):
                tokenizer = candidate
                print(f"\n  Reusing tokenizer: {tokenizer}\n")

    # Summary
    print(f"\n{'='*70}")
    print("  Apical architecture sweep — summary")
    print(f"{'='*70}")
    for phase, ok, elapsed in results:
        tag = "OK  " if ok else "FAIL"
        t = f"{elapsed/3600:.1f}h" if elapsed > 0 else "dry-run"
        print(f"  [{tag}]  Phase {phase}  ({t})")

    print(f"\n  Next: compare val/perplexity curves in W&B group '{args.wandb_group}'")
    print(f"  Then run the variant sweep on the winning architecture:")
    print(f"    python scripts/run_apical_variant_sweep.py \\")
    print(f"      --base-config configs/phase1X_<winner>.yaml \\")
    print(f"      --tokenizer <tokenizer> \\")
    print(f"      --wandb-project {args.wandb_project}")
    print()

    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
