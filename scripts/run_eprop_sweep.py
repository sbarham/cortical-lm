#!/usr/bin/env python3
"""
scripts/run_eprop_sweep.py -- General-purpose e-prop sweep across architecture variants.

Covers all canonical architecture variants (1a-1l) with any e-prop learning rule
and hyperparameter configuration.  Designed to be the single script for all future
e-prop studies -- replace run_eprop_series.py and run_eprop_series_2.py.

Architecture variants
---------------------
    1a   Rate neurons, simple_ei column
    1b   + Layered cortical columns
    1c   + Tsodyks-Markram STP
    1d   + AdEx adaptive neurons
    1e   + VIP disinhibition
    1f   + Hopfield CA3 (no CA1)                        [e-prop series-3/4 baseline]
    1g   CA3 + always-on disinhibition
    1h   CA3 + annealed disinhibition
    1i   CA3 + CA1 (original -- gradient leak present)
    1j   CA3 + Xi normalisation, no CA1                 [v2 CA3 baseline]
    1k   CA3 + CA1, gradient-leak fix + Xi norm         [v2 CA1]
    1l   CA3 + CA1, forward-gated error feedback        [v2 CA1 + forward gate]

All variants run with column.apical_pathway=additive and
learning.reset_state_between_batches=true by default (both required for e-prop).

Learning rule flags (all exposed as CLI args with sensible defaults)
---------------------------------------------------------------------
    --rule          eprop | eprop_approx | eprop_hybrid   (default: eprop_hybrid)
    --eprop-steps   e-prop steps per awake phase           (default: 20)
    --bptt-steps    BPTT steps per consolidation phase     (default: 10)
    --bptt-lr       learning rate for BPTT consolidation   (default: 3e-4)
    --bptt-scope    full | readout_only                    (default: full)
    --tau-e         eligibility trace timescale            (default: from config)
    --batch-size    batch size                             (default: 32)
    --max-tokens    token budget per run                   (default: 50M)
    --lr            base learning rate                     (default: 1e-4)

Usage
-----
# Sweep 1i-1l with aggressive hybrid (the canonical next experiment)
python scripts/run_eprop_sweep.py \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb --wandb-project cortex-lm \\
    --runs 1i 1j 1k 1l

# Single run, custom hyperparameters
python scripts/run_eprop_sweep.py \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb --wandb-project cortex-lm \\
    --runs 1f --rule eprop --tau-e 128 --batch-size 8 --max-tokens 100000000

# Full architecture sweep with default aggressive hybrid
python scripts/run_eprop_sweep.py \\
    --tokenizer tokenizers/tinystories_bpe4096.pkl \\
    --wandb --wandb-project cortex-lm \\
    --runs 1a 1b 1c 1d 1f 1j

# Dry run
python scripts/run_eprop_sweep.py --dry-run --runs 1i 1j 1k 1l
"""

import argparse
import subprocess
import sys
import time

EXPERIMENTS = [
    {
        "id":       "1a",
        "label":    "Phase 1a -- rate neurons, simple_ei column",
        "config":   "configs/phase1a_minimal.yaml",
        "ckpt_tag": "1a",
    },
    {
        "id":       "1b",
        "label":    "Phase 1b -- layered cortical columns",
        "config":   "configs/phase1b_layered.yaml",
        "ckpt_tag": "1b",
    },
    {
        "id":       "1c",
        "label":    "Phase 1c -- STP",
        "config":   "configs/phase1c_stp.yaml",
        "ckpt_tag": "1c",
    },
    {
        "id":       "1d",
        "label":    "Phase 1d -- AdEx adaptive neurons",
        "config":   "configs/phase1d_adex.yaml",
        "ckpt_tag": "1d",
    },
    {
        "id":       "1e",
        "label":    "Phase 1e -- VIP disinhibition",
        "config":   "configs/phase1e_disinhibition.yaml",
        "ckpt_tag": "1e",
    },
    {
        "id":       "1f",
        "label":    "Phase 1f -- Hopfield CA3 (e-prop series-3/4 baseline)",
        "config":   "configs/phase1f_hopfield.yaml",
        "ckpt_tag": "1f",
    },
    {
        "id":       "1g",
        "label":    "Phase 1g -- CA3 + always-on disinhibition",
        "config":   "configs/phase1g_hopfield_disinhibition.yaml",
        "ckpt_tag": "1g",
    },
    {
        "id":       "1h",
        "label":    "Phase 1h -- CA3 + annealed disinhibition",
        "config":   "configs/phase1h_hopfield_annealed.yaml",
        "ckpt_tag": "1h",
    },
    {
        "id":       "1i",
        "label":    "Phase 1i -- CA3 + CA1 (original; gradient leak present)",
        "config":   "configs/phase1i_hopfield_ca1.yaml",
        "ckpt_tag": "1i",
    },
    {
        "id":       "1j",
        "label":    "Phase 1j -- CA3 + Xi normalisation, no CA1 (v2 baseline)",
        "config":   "configs/phase1j_hopfield_v2.yaml",
        "ckpt_tag": "1j",
    },
    {
        "id":       "1k",
        "label":    "Phase 1k -- CA3 + CA1, gradient-leak fix + Xi norm (v2 CA1)",
        "config":   "configs/phase1k_hopfield_ca1_v2.yaml",
        "ckpt_tag": "1k",
    },
    {
        "id":       "1l",
        "label":    "Phase 1l -- CA3 + CA1, forward-gated error feedback",
        "config":   "configs/phase1l_hopfield_ca1_fwdgate.yaml",
        "ckpt_tag": "1l",
    },
]

EXP_IDS = [e["id"] for e in EXPERIMENTS]


def build_command(exp: dict, args: argparse.Namespace) -> list[str]:
    rule = args.rule
    tag_parts = [f"eprop-{rule.replace('_', '-')}", exp["ckpt_tag"]]
    if args.tau_e is not None:
        tag_parts.append(f"tau{args.tau_e}")
    if args.batch_size != 32:
        tag_parts.append(f"bs{args.batch_size}")
    run_name = "-".join(tag_parts)

    ckpt_dir = f"checkpoints/eprop-sweep/{args.wandb_group}/{exp['ckpt_tag']}"

    cmd = [sys.executable, "scripts/train.py", "--config", exp["config"]]
    if args.wandb:
        cmd.append("--wandb")

    overrides = [
        # E-prop core
        f"learning.rule={rule}",
        "learning.reset_state_between_batches=true",
        "column.apical_pathway=additive",
        # Hybrid parameters (ignored by non-hybrid rules)
        f"learning.hybrid_eprop_steps={args.eprop_steps}",
        f"learning.hybrid_bptt_steps={args.bptt_steps}",
        f"learning.hybrid_bptt_lr={args.bptt_lr}",
        f"learning.hybrid_bptt_scope={args.bptt_scope}",
        # STP off -- complicates eligibility traces
        "synapse.inter_column_stp=false",
        # Training
        f"training.batch_size={args.batch_size}",
        f"training.max_tokens={args.max_tokens}",
        f"training.lr={args.lr}",
        f"training.checkpoint_dir={ckpt_dir}",
        # Logging
        f"logging.project={args.wandb_project}",
        f"logging.group={args.wandb_group}",
        f"name={run_name}",
    ]

    if args.tokenizer:
        overrides.append(f"data.tokenizer_path={args.tokenizer}")
    if args.tau_e is not None:
        overrides.append(f"learning.eprop_tau_e={args.tau_e}")

    # Log frequently enough for e-prop's slow throughput
    log_tokens = min(51_200, args.max_tokens // 20)
    overrides.append(f"training.log_tokens={log_tokens}")
    overrides.append(f"training.eval_tokens={log_tokens * 2}")

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
        description="General-purpose e-prop sweep across architecture variants 1a-1l"
    )

    # Experiment selection
    parser.add_argument("--runs", nargs="+", choices=EXP_IDS, default=None,
                        help=f"Variants to run in specified order. Choices: {EXP_IDS}")
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl",
                        help="Path to tokenizer.pkl (strongly recommended for comparable ppl).")

    # Learning rule
    parser.add_argument("--rule", default="eprop_hybrid",
                        choices=["eprop", "eprop_approx", "eprop_hybrid"],
                        help="E-prop learning rule variant (default: eprop_hybrid).")
    parser.add_argument("--eprop-steps", type=int, default=20,
                        help="E-prop steps per awake phase, hybrid only (default: 20).")
    parser.add_argument("--bptt-steps", type=int, default=10,
                        help="BPTT steps per consolidation phase, hybrid only (default: 10).")
    parser.add_argument("--bptt-lr", type=float, default=3e-4,
                        help="Learning rate for BPTT consolidation (default: 3e-4).")
    parser.add_argument("--bptt-scope", default="full",
                        choices=["full", "readout_only"],
                        help="BPTT scope: full network or readout only (default: full).")
    parser.add_argument("--tau-e", type=int, default=None,
                        help="Eligibility trace timescale override (default: from config).")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32; keep small to reduce sign cancellation).")
    parser.add_argument("--max-tokens", type=int, default=50_000_000,
                        help="Token budget per run (default: 50M).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate (default: 1e-4).")

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"eprop-sweep-{time.strftime('%Y-%m-%d')}")

    # Misc
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
    print(f"\nE-prop sweep -- {args.rule}")
    print(f"  Variants    : {[e['id'] for e in selected]}")
    print(f"  Rule        : {args.rule}")
    if args.rule == "eprop_hybrid":
        print(f"  Hybrid      : {args.eprop_steps} eprop + {args.bptt_steps} bptt "
              f"({args.bptt_scope}), bptt_lr={args.bptt_lr}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max tokens  : {args.max_tokens:,}")
    print(f"  LR          : {args.lr}")
    print(f"  W&B group   : {args.wandb_group}")
    print(f"  Tokenizer   : {args.tokenizer or '(from config)'}")
    if args.tau_e is not None:
        print(f"  tau_e       : {args.tau_e}")
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
