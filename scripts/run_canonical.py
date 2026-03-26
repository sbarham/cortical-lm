#!/usr/bin/env python3
"""
scripts/run_canonical.py — Sequential canonical Phase 1 ablation runs.

Each phase adds exactly one bio-plausible ingredient:
    1a  simple_ei baseline
    1b  + layered cortical columns (L4/L23/L5/L6)
    1c  + Tsodyks-Markram short-term plasticity
    1d  + AdEx adaptive neurons
    1e  + VIP→SST→PC disinhibition circuit
    1f  + Modern Hopfield hippocampal module  (full system)

All runs log to the same W&B project and group so ablation curves can be
overlaid on a shared tokens-seen axis. The tokenizer from Phase 1a is
reused for 1b–1f so tokenisation is held constant across the series.

Usage
-----
# Full run (trains tokenizer in 1a, reuses for 1b–1f)
python scripts/run_canonical.py --wandb-project cortex-lm-canonical

# Reuse a pre-trained tokenizer (recommended — skips BPE in every phase)
python scripts/run_canonical.py \\
    --tokenizer checkpoints/tokenizer.pkl \\
    --wandb-project cortex-lm-canonical

# Override batch size to saturate an H100
python scripts/run_canonical.py \\
    --tokenizer checkpoints/tokenizer.pkl \\
    --batch-size 2048 \\
    --wandb-project cortex-lm-canonical

# Resume from a specific phase after a partial run
python scripts/run_canonical.py \\
    --tokenizer checkpoints/tokenizer.pkl \\
    --start-from 1c \\
    --wandb-project cortex-lm-canonical

# Dry run — print commands without executing
python scripts/run_canonical.py --dry-run
"""

import argparse
import glob
import os
import subprocess
import sys
import time

# ── Phase definitions ────────────────────────────────────────────────────────
# Baselines are parameter-matched to the full system (phase 1f).
BASELINE_CONFIG = "configs/phase1f_hopfield.yaml"   # parameter-matched to 1f (no apical)
BASELINE_MODELS = ["rnn", "lstm", "lstm_attention", "transformer"]

PHASES = [
    ("1a", "configs/phase1a_minimal.yaml"),
    ("1b", "configs/phase1b_layered.yaml"),
    ("1c", "configs/phase1c_stp.yaml"),
    ("1d", "configs/phase1d_adex.yaml"),
    ("1e", "configs/phase1e_disinhibition.yaml"),       # + VIP->SST->PC disinhibition
    ("1f", "configs/phase1f_hopfield.yaml"),             # + Hopfield HPC (no disinhibition)
    ("1g", "configs/phase1g_hopfield_disinhibition.yaml"),  # 1f + always-on disinhibition
    ("1h", "configs/phase1h_hopfield_annealed.yaml"),   # 1f + annealed disinhibition
    ("1i", "configs/phase1i_hopfield_ca1.yaml"),        # 1f + CA1 write-gating
]

PHASE_IDS = [p for p, _ in PHASES]


def find_tokenizer_after(checkpoint_root: str) -> str | None:
    """Search for any tokenizer.pkl written under checkpoint_root."""
    matches = glob.glob(
        os.path.join(checkpoint_root, "**", "tokenizer.pkl"), recursive=True
    )
    return sorted(matches)[-1] if matches else None


def build_command(
    config: str,
    tokenizer: str | None,
    wandb_project: str,
    wandb_group: str,
    batch_size: int | None,
    extra_overrides: list[str],
) -> list[str]:
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config,
        "--wandb",
        "--override",
        f"logging.project={wandb_project}",
        f"logging.group={wandb_group}",
    ]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    if batch_size:
        cmd += ["--override", f"training.batch_size={batch_size}"]
    for ov in extra_overrides:
        cmd += ["--override", ov]
    return cmd


def run_phase(
    phase: str,
    config: str,
    tokenizer: str | None,
    wandb_project: str,
    wandb_group: str,
    batch_size: int | None,
    extra_overrides: list[str],
    dry_run: bool,
) -> tuple[bool, float]:
    """Run one phase. Returns (success, elapsed_seconds)."""
    cmd = build_command(
        config, tokenizer, wandb_project, wandb_group, batch_size, extra_overrides
    )

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Phase {phase}  |  {config}")
    print(f"  {' '.join(cmd)}")
    print(f"{sep}\n", flush=True)

    if dry_run:
        return True, 0.0

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical Phase 1 ablation series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to pre-trained tokenizer.pkl (skips BPE in all phases)",
    )
    parser.add_argument(
        "--start-from", default="1a", choices=PHASE_IDS, metavar="PHASE",
        help=f"Start from this phase, skipping earlier ones. Choices: {PHASE_IDS}",
    )
    parser.add_argument(
        "--phases", nargs="+", choices=PHASE_IDS, metavar="PHASE",
        help="Run only these specific phases (overrides --start-from)",
    )
    parser.add_argument(
        "--wandb-project", default="cortex-lm",
        help="W&B project name (default: cortex-lm)",
    )
    parser.add_argument(
        "--wandb-group", default=None,
        help="W&B run group. Default: canonical-YYYY-MM-DD",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size for all phases (total tokens held constant)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Extra config overrides applied to every phase (key=value format)",
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="After all phases, run parameter-matched baselines via run_baselines.py",
    )
    parser.add_argument(
        "--baseline-output", default="baseline_results.json",
        help="JSON output path for baseline results (default: baseline_results.json)",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort the series if any phase exits non-zero (default: continue)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    group = args.wandb_group or f"canonical-{time.strftime('%Y-%m-%d')}"
    tokenizer = args.tokenizer

    # Select phases
    if args.phases:
        phases_to_run = [(p, c) for p, c in PHASES if p in args.phases]
    else:
        start_idx = PHASE_IDS.index(args.start_from)
        phases_to_run = PHASES[start_idx:]

    # Header
    print("\nCortexLM — canonical Phase 1 ablation series")
    print(f"  W&B project : {args.wandb_project}")
    print(f"  W&B group   : {group}")
    print(f"  Phases      : {[p for p, _ in phases_to_run]}")
    print(f"  Batch size  : {args.batch_size or 'from config (512)'}")
    print(f"  Tokenizer   : {tokenizer or '(train fresh in phase 1a)'}")
    if args.baselines:
        print(f"  Baselines   : {BASELINE_MODELS} (param-matched to phase 1f)")
    if args.dry_run:
        print("  *** DRY RUN — commands will be printed but not executed ***")
    print()

    results: list[tuple[str, str, bool, float]] = []

    for phase, config in phases_to_run:
        ok, elapsed = run_phase(
            phase=phase,
            config=config,
            tokenizer=tokenizer,
            wandb_project=args.wandb_project,
            wandb_group=group,
            batch_size=args.batch_size,
            extra_overrides=args.override or [],
            dry_run=args.dry_run,
        )
        results.append((phase, config, ok, elapsed))

        # After phase 1a, locate the tokenizer it wrote so later phases reuse it
        if phase == "1a" and tokenizer is None and not args.dry_run:
            found = find_tokenizer_after("checkpoints")
            if found:
                tokenizer = found
                print(f"\n  [run_canonical] tokenizer for 1b–1f: {tokenizer}\n")
            else:
                print(
                    "\n  [run_canonical] WARNING: could not find tokenizer.pkl "
                    "after phase 1a — subsequent phases will retrain BPE\n"
                )

        if not ok:
            print(f"\n  [run_canonical] Phase {phase} FAILED")
            if args.stop_on_failure:
                print("  --stop-on-failure set: aborting series.")
                break

    # ── Baselines ────────────────────────────────────────────────────────────
    baseline_ok = True
    baseline_elapsed = 0.0
    if args.baselines:
        tok_arg = ["--tokenizer", tokenizer] if tokenizer else []
        baseline_cmd = [
            sys.executable, "scripts/run_baselines.py",
            "--config", BASELINE_CONFIG,
            "--models", *BASELINE_MODELS,
            "--output", args.baseline_output,
            "--wandb",
            "--wandb-project", args.wandb_project,
            "--wandb-group", group,
            *tok_arg,
        ]
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  Baselines  |  {BASELINE_CONFIG}")
        print(f"  {' '.join(baseline_cmd)}")
        print(f"{sep}\n", flush=True)
        if not args.dry_run:
            t0 = time.time()
            r = subprocess.run(baseline_cmd)
            baseline_elapsed = time.time() - t0
            baseline_ok = r.returncode == 0

    # ── Summary ─────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  CANONICAL RUN SUMMARY")
    print(sep)
    total_s = sum(e for *_, e in results)
    for phase, config, ok, elapsed in results:
        tag = "OK  " if ok else "FAIL"
        mins = f"{elapsed / 60:.1f} min" if elapsed > 0 else "dry-run"
        print(f"  [{tag}]  Phase {phase}  ({mins})  {os.path.basename(config)}")
    if args.baselines:
        tag = "OK  " if baseline_ok else "FAIL"
        mins = f"{baseline_elapsed / 60:.1f} min" if baseline_elapsed > 0 else "dry-run"
        print(f"  [{tag}]  Baselines  ({mins})  {', '.join(BASELINE_MODELS)}")
        total_s += baseline_elapsed
    print(f"\n  Total elapsed: {total_s / 3600:.2f} h")
    print(sep)

    all_ok = all(ok for *_, ok, _ in results) and baseline_ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
