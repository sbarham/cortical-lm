#!/usr/bin/env python3
"""
scripts/run_canonical_apical.py -- Canonical ablation series 1a-1i with additive apical pathway.

The original canonical BPTT series was run without the apical pathway, which is load-bearing
for e-prop and potentially for Hopfield CA3/CA1.  This script reruns 1a-1i with
column.apical_pathway=additive so every phase is evaluated under fair conditions.

Each run is otherwise identical to the canonical series: batch=512, cosine LR 3e-4,
1B-token budget.

Experiments
-----------
    1a   Rate neurons + simple_ei column + apical (baseline reference with apical)
    1b   + Layered cortical columns
    1c   + Tsodyks-Markram STP
    1d   + AdEx adaptive neurons
    1e   + VIP disinhibition
    1f   + Modern Hopfield CA3
    1g   1f + always-on disinhibition
    1h   1f + annealed disinhibition
    1i   1f + CA1 write-gating  <- first priority

Usage
-----
# Full sweep (1a-1i, 1B tokens each)
python scripts/run_canonical_apical.py \
    --tokenizer checkpoints/tokenizer.pkl \
    --wandb --wandb-project cortex-lm

# Run out of order -- 1i first
python scripts/run_canonical_apical.py \
    --tokenizer checkpoints/tokenizer.pkl \
    --wandb --wandb-project cortex-lm \
    --runs 1i 1f 1d 1a 1b 1c 1e 1g 1h

# SGDR variant (warm restarts instead of cosine decay)
python scripts/run_canonical_apical.py \
    --tokenizer checkpoints/tokenizer.pkl \
    --wandb --wandb-project cortex-lm \
    --override training.scheduler=sgdr \
    --wandb-group canonical-apical-sgdr

# Dry run
python scripts/run_canonical_apical.py --dry-run
"""

import argparse
import subprocess
import sys
import time

EXPERIMENTS = [
    {
        "id":       "1a",
        "label":    "Phase 1a -- rate neurons, simple_ei column + apical",
        "config":   "configs/phase1a_minimal.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1a"],
        "ckpt_dir": "checkpoints/canonical-apical/1a",
    },
    {
        "id":       "1b",
        "label":    "Phase 1b -- layered cortical columns + apical",
        "config":   "configs/phase1b_layered.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1b"],
        "ckpt_dir": "checkpoints/canonical-apical/1b",
    },
    {
        "id":       "1c",
        "label":    "Phase 1c -- STP + apical",
        "config":   "configs/phase1c_stp.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1c"],
        "ckpt_dir": "checkpoints/canonical-apical/1c",
    },
    {
        "id":       "1d",
        "label":    "Phase 1d -- AdEx neurons + apical",
        "config":   "configs/phase1d_adex.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1d"],
        "ckpt_dir": "checkpoints/canonical-apical/1d",
    },
    {
        "id":       "1e",
        "label":    "Phase 1e -- VIP disinhibition + apical",
        "config":   "configs/phase1e_disinhibition.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1e"],
        "ckpt_dir": "checkpoints/canonical-apical/1e",
    },
    {
        "id":       "1f",
        "label":    "Phase 1f -- Hopfield CA3 + apical",
        "config":   "configs/phase1f_hopfield.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1f"],
        "ckpt_dir": "checkpoints/canonical-apical/1f",
    },
    {
        "id":       "1g",
        "label":    "Phase 1g -- CA3 + always-on disinhibition + apical",
        "config":   "configs/phase1g_hopfield_disinhibition.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1g"],
        "ckpt_dir": "checkpoints/canonical-apical/1g",
    },
    {
        "id":       "1h",
        "label":    "Phase 1h -- CA3 + annealed disinhibition + apical",
        "config":   "configs/phase1h_hopfield_annealed.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1h"],
        "ckpt_dir": "checkpoints/canonical-apical/1h",
    },
    {
        "id":       "1i",
        "label":    "Phase 1i -- CA3 + CA1 write-gating + apical  [FIRST PRIORITY]",
        "config":   "configs/phase1i_hopfield_ca1.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1i"],
        "ckpt_dir": "checkpoints/canonical-apical/1i",
    },
    {
        "id":       "1j",
        "label":    "Phase 1j -- CA3 + Xi normalisation + apical (v2 CA3 baseline)",
        "config":   "configs/phase1j_hopfield_v2.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1j"],
        "ckpt_dir": "checkpoints/canonical-apical/1j",
    },
    {
        "id":       "1k",
        "label":    "Phase 1k -- CA3 + CA1 (gradient-leak fix + Xi norm) + apical",
        "config":   "configs/phase1k_hopfield_ca1_v2.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1k"],
        "ckpt_dir": "checkpoints/canonical-apical/1k",
    },
    {
        "id":       "1l",
        "label":    "Phase 1l -- CA3 + CA1 forward-gated error feedback + apical",
        "config":   "configs/phase1l_hopfield_ca1_fwdgate.yaml",
        "extra":    ["column.apical_pathway=additive", "name=canonical-apical-1l"],
        "ckpt_dir": "checkpoints/canonical-apical/1l",
    },
]

EXP_IDS = [e["id"] for e in EXPERIMENTS]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--config", exp["config"]]
    if args.wandb:
        cmd.append("--wandb")

    overrides = [
        f"training.checkpoint_dir={exp['ckpt_dir']}",
        f"training.max_tokens={args.max_tokens}",
    ]
    if args.wandb_project:
        overrides.append(f"logging.project={args.wandb_project}")
    if args.wandb_group:
        overrides.append(f"logging.group={args.wandb_group}")
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
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  [{exp['id']}]  {exp['label']}")
    print(f"  Config : {exp['config']}")
    print(f"  Command: {' '.join(cmd)}")
    print(sep)

    if dry_run:
        print("  (dry run -- skipping)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{exp['id']}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  [{exp['id']}] completed in {elapsed/3600:.1f}h")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Canonical ablation series 1a-1i with additive apical pathway"
    )
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.pkl (reuse from a prior run for comparable ppl).")
    parser.add_argument("--max-tokens", type=int, default=1_000_000_000,
                        help="Token budget per run (default: 1B for canonical).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from config, 512).")
    parser.add_argument("--runs", nargs="+", choices=EXP_IDS, default=None,
                        help=(
                            "Run a subset in the specified order (default: all 1a-1i). "
                            f"Choices: {EXP_IDS}.  Example: --runs 1i 1f 1d"
                        ))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"canonical-apical-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs:
        exp_by_id = {e["id"]: e for e in EXPERIMENTS}
        selected = [exp_by_id[r] for r in args.runs]
    else:
        selected = EXPERIMENTS

    sep = "=" * 70
    print(f"\nCanonical ablation 1a-1i -- with additive apical pathway")
    print(f"  Experiments : {[e['id'] for e in selected]}")
    print(f"  Max tokens  : {args.max_tokens:,}")
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

    print(f"\n{sep}")
    print(f"  Summary: {len(selected)-len(failed)}/{len(selected)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
