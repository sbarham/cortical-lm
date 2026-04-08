#!/usr/bin/env python3
"""
scripts/run_wakesleep_canonical.py — Canonical ablation series with hybrid wake-sleep learning.

Runs phases 1a, 1b, 1c, 1d, 1f, 1i, 1k on TinyStories using the hybrid e-prop / BPTT
wake-sleep rule (eprop_hybrid) with a fixed 20:10 cycle ratio and no annealing (no DAWN).

Purpose: establish whether the wake-sleep hybrid gives a sample-efficiency win across each
ablation tier, independently of the scale_5m experiments.

Wake-sleep settings (fixed for all phases):
    learning.rule             = eprop_hybrid
    learning.eprop_mode       = vectorized
    learning.hybrid_eprop_steps     = 20
    learning.hybrid_bptt_steps      = 10
    learning.hybrid_bptt_batch_size = 128
    learning.hybrid_bptt_scope      = full
    learning.reset_state_between_batches = true
    training.batch_size       = 16   (e-prop batch; BPTT uses its own 128)
    column.apical_pathway     = additive

Phases included (most informative tiers):
    1a   Rate neurons + simple_ei column   (minimal baseline)
    1b   + Layered cortical columns
    1c   + Tsodyks-Markram STP
    1d   + AdEx adaptive neurons
    1f   + Modern Hopfield CA3
    1i   1f + CA1 write-gating             (best from original series)
    1k   1f + CA1 v2 (gradient-leak fix + Xi norm)  (best from v2 series)

Usage
-----
# Priority subset (most informative)
python scripts/run_wakesleep_canonical.py --runs 1f 1i 1k --wandb --wandb-offline

# Full set
python scripts/run_wakesleep_canonical.py --runs all --wandb --wandb-offline

# Dry run
python scripts/run_wakesleep_canonical.py --runs all --dry-run

# Custom token budget
python scripts/run_wakesleep_canonical.py --runs all --max-tokens 50000000 --wandb --wandb-offline
"""

import argparse
import os
import subprocess
import sys
import time

SEQ_LEN        = 256
EPROP_BATCH    = 16
BPTT_BATCH     = 128
EPROP_STEPS    = 20
BPTT_STEPS     = 10
TOKENIZER      = "tokenizers/tinystories_bpe4096.pkl"

EXPERIMENTS = [
    {
        "id":       "1a",
        "label":    "Phase 1a — rate neurons, simple_ei column",
        "config":   "configs/phase1a_minimal.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1a",
    },
    {
        "id":       "1b",
        "label":    "Phase 1b — layered cortical columns",
        "config":   "configs/phase1b_layered.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1b",
    },
    {
        "id":       "1c",
        "label":    "Phase 1c — STP synapses",
        "config":   "configs/phase1c_stp.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1c",
    },
    {
        "id":       "1d",
        "label":    "Phase 1d — AdEx adaptive neurons",
        "config":   "configs/phase1d_adex.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1d",
    },
    {
        "id":       "1f",
        "label":    "Phase 1f — Modern Hopfield CA3",
        "config":   "configs/phase1f_hopfield.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1f",
    },
    {
        "id":       "1i",
        "label":    "Phase 1i — CA3 + CA1 write-gating",
        "config":   "configs/phase1i_hopfield_ca1.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1i",
    },
    {
        "id":       "1k",
        "label":    "Phase 1k — CA3 + CA1 v2 (gradient-leak fix + Xi norm)",
        "config":   "configs/phase1k_hopfield_ca1_v2.yaml",
        "ckpt_dir": "checkpoints/wakesleep-canonical/1k",
    },
]

EXP_IDS = [e["id"] for e in EXPERIMENTS]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    max_tokens   = args.max_tokens
    max_steps    = max_tokens // (EPROP_BATCH * SEQ_LEN)
    warmup_steps = max(1, max_steps // 20)   # 5% warmup
    run_name     = f"ws-canonical-{exp['id']}"

    overrides = [
        # Wake-sleep learning rule
        f"learning.rule=eprop_hybrid",
        f"learning.eprop_mode=vectorized",
        f"learning.hybrid_eprop_steps={EPROP_STEPS}",
        f"learning.hybrid_bptt_steps={BPTT_STEPS}",
        f"learning.hybrid_bptt_batch_size={BPTT_BATCH}",
        f"learning.hybrid_bptt_scope=full",
        f"learning.reset_state_between_batches=true",
        # Training budget
        f"training.batch_size={EPROP_BATCH}",
        f"training.max_steps={max_steps}",
        f"training.max_tokens={max_tokens}",
        f"training.warmup_steps={warmup_steps}",
        f"training.log_tokens=100000",
        f"training.eval_tokens=500000",
        f"training.checkpoint_dir={exp['ckpt_dir']}",
        # Apical pathway (required for e-prop credit assignment)
        f"column.apical_pathway=additive",
        # Tokenizer + run identity
        f"data.tokenizer_path={args.tokenizer}",
        f"name={run_name}",
    ]

    if args.wandb:
        overrides += [
            f"logging.wandb=true",
            f"logging.project={args.wandb_project}",
            f"logging.group={args.wandb_group}",
        ]

    for ov in (args.override or []):
        overrides.append(ov)

    cmd = [sys.executable, "scripts/train.py",
           "--config", exp["config"]]
    if getattr(args, "resume", None):
        cmd += ["--resume", args.resume]
    cmd += ["--override"] + overrides
    return cmd


def run_experiment(exp: dict, cmd: list[str], dry_run: bool, args: argparse.Namespace) -> bool:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  [{exp['id']}]  {exp['label']}")
    print(f"  Config : {exp['config']}")
    print(f"  Command: {' '.join(cmd)}")
    print(sep)

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
        print(f"\n  !! [{exp['id']}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  [{exp['id']}] completed in {elapsed/3600:.1f}h")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Canonical ablation series 1a/1b/1c/1d/1f/1i/1k with hybrid wake-sleep learning"
    )
    parser.add_argument("--runs", nargs="+", default=["1f", "1i", "1k"],
                        help=f"Phase IDs to run, or 'all'. Choices: {EXP_IDS}")
    parser.add_argument("--max-tokens", type=int, default=150_000_000,
                        help="Token budget per phase (default: 150M).")
    parser.add_argument("--tokenizer", default=TOKENIZER,
                        help="Path to tokenizer.pkl.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Set WANDB_MODE=offline (log locally, sync later)")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"wakesleep-canonical-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run.")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint .pt file to resume from (single-phase runs only).")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs == ["all"]:
        args.runs = EXP_IDS

    unknown = [r for r in args.runs if r not in EXP_IDS]
    if unknown:
        parser.error(f"Unknown phase(s): {unknown}. Choices: {EXP_IDS}")

    exp_by_id = {e["id"]: e for e in EXPERIMENTS}
    selected  = [exp_by_id[r] for r in args.runs]

    max_steps = args.max_tokens // (EPROP_BATCH * SEQ_LEN)
    print(f"\nWake-sleep canonical ablation — {len(selected)} phase(s) × {args.max_tokens/1e6:.0f}M tokens")
    print(f"  Wake/sleep ratio : {EPROP_STEPS}:{BPTT_STEPS} (fixed, no annealing)")
    print(f"  E-prop batch     : {EPROP_BATCH}    BPTT batch: {BPTT_BATCH}")
    print(f"  Max steps        : {max_steps:,}    Warmup: {max(1, max_steps // 20):,} (5%)")
    print(f"  Phases           : {args.runs}")
    print(f"  W&B group        : {args.wandb_group}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    passed, failed = [], []
    for exp in selected:
        cmd = build_command(exp, args)
        ok  = run_experiment(exp, cmd, args.dry_run, args)
        (passed if ok else failed).append(exp["id"])
        if not ok and args.stop_on_failure:
            print("  --stop-on-failure: aborting.")
            break

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Done: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{sep}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
