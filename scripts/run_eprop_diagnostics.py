#!/usr/bin/env python3
"""
scripts/run_eprop_diagnostics.py -- Focused diagnostic sweeps for e-prop hyperparameters.

Three independent sweep modes, each run against a single architecture variant:

    ratio     Wake/sleep ratio sweep around the 2:1 aggressive-hybrid baseline.
              BPTT steps fixed at 10; e-prop steps varied.
              Ratios: 3:1, 2:1(*), 3:2, 1:1, 1:2, 1:3   (* = already done)

    tau_e     Eligibility trace timescale.
              Default range: 16, 32, 64, 128, 256 (ms / steps).
              Motivation: longer tau_e credits slower-changing context; default
              auto (geometric mean of tau_m_range ~8) is likely too short for
              seq_len=128.

    batch     Batch size sweep for sign-cancellation diagnosis.
              Default range: 32(*), 16, 8, 4.
              Motivation: e-prop batch-averages L_vec before the weight update,
              causing sign cancellation ∝ 1/√batch.  Smaller batch = stronger
              signal, at the cost of slower wall-clock throughput.

Usage
-----
# Wake/sleep ratio sweep on 1k (default variant)
python scripts/run_eprop_diagnostics.py --sweep ratio --wandb

# tau_e sweep on 1f (for direct comparison with series-4)
python scripts/run_eprop_diagnostics.py --sweep tau_e --variant 1f --wandb

# Batch sweep, small budget (quick read on MPS/laptop)
python scripts/run_eprop_diagnostics.py --sweep batch --variant 1k \\
    --max-tokens 20000000 --device mps --wandb

# Dry run to inspect commands
python scripts/run_eprop_diagnostics.py --sweep ratio --dry-run

# Custom tau_e values (space-separated)
python scripts/run_eprop_diagnostics.py --sweep tau_e --tau-e-values 64 128 256 --wandb
"""

import argparse
import subprocess
import sys
import time

# ── Architecture registry (shared with run_eprop_sweep.py) ───────────────────

EXPERIMENTS = {
    "1a": {"config": "configs/phase1a_minimal.yaml",           "ckpt_tag": "1a"},
    "1b": {"config": "configs/phase1b_layered.yaml",           "ckpt_tag": "1b"},
    "1c": {"config": "configs/phase1c_stp.yaml",               "ckpt_tag": "1c"},
    "1d": {"config": "configs/phase1d_adex.yaml",              "ckpt_tag": "1d"},
    "1e": {"config": "configs/phase1e_disinhibition.yaml",     "ckpt_tag": "1e"},
    "1f": {"config": "configs/phase1f_hopfield.yaml",          "ckpt_tag": "1f"},
    "1g": {"config": "configs/phase1g_hopfield_disinhibition.yaml", "ckpt_tag": "1g"},
    "1h": {"config": "configs/phase1h_hopfield_annealed.yaml", "ckpt_tag": "1h"},
    "1i": {"config": "configs/phase1i_hopfield_ca1.yaml",      "ckpt_tag": "1i"},
    "1j": {"config": "configs/phase1j_hopfield_v2.yaml",       "ckpt_tag": "1j"},
    "1k": {"config": "configs/phase1k_hopfield_ca1_v2.yaml",   "ckpt_tag": "1k"},
    "1l": {"config": "configs/phase1l_hopfield_ca1_fwdgate.yaml", "ckpt_tag": "1l"},
}

# ── Sweep definitions ─────────────────────────────────────────────────────────

def ratio_sweep(bptt_lr: float) -> list[dict]:
    """Wake/sleep ratio sweep.  BPTT steps fixed at 10; e-prop steps varied."""
    return [
        {"id": "3:1",  "label": "e-prop:bptt = 3:1",  "eprop_steps": 30, "bptt_steps": 10, "bptt_lr": bptt_lr},
        # 2:1 is the existing baseline -- included for legend continuity but marked
        {"id": "2:1",  "label": "e-prop:bptt = 2:1 (baseline)", "eprop_steps": 20, "bptt_steps": 10, "bptt_lr": bptt_lr},
        {"id": "3:2",  "label": "e-prop:bptt = 3:2",  "eprop_steps": 15, "bptt_steps": 10, "bptt_lr": bptt_lr},
        {"id": "1:1",  "label": "e-prop:bptt = 1:1",  "eprop_steps": 10, "bptt_steps": 10, "bptt_lr": bptt_lr},
        {"id": "1:2",  "label": "e-prop:bptt = 1:2",  "eprop_steps":  5, "bptt_steps": 10, "bptt_lr": bptt_lr},
        {"id": "1:3",  "label": "e-prop:bptt = 1:3",  "eprop_steps":  3, "bptt_steps": 10, "bptt_lr": bptt_lr},
    ]


def tau_e_sweep(values: list[int], eprop_steps: int, bptt_steps: int, bptt_lr: float) -> list[dict]:
    return [
        {
            "id":          f"tau{v}",
            "label":       f"tau_e = {v}",
            "tau_e":       v,
            "eprop_steps": eprop_steps,
            "bptt_steps":  bptt_steps,
            "bptt_lr":     bptt_lr,
        }
        for v in values
    ]


def batch_sweep(values: list[int], eprop_steps: int, bptt_steps: int, bptt_lr: float) -> list[dict]:
    return [
        {
            "id":          f"bs{v}",
            "label":       f"batch_size = {v}",
            "batch_size":  v,
            "eprop_steps": eprop_steps,
            "bptt_steps":  bptt_steps,
            "bptt_lr":     bptt_lr,
        }
        for v in values
    ]


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(condition: dict, exp: dict, args: argparse.Namespace) -> list[str]:
    sweep      = args.sweep
    variant    = args.variant
    eprop_steps = condition["eprop_steps"]
    bptt_steps  = condition["bptt_steps"]
    bptt_lr     = condition["bptt_lr"]
    batch_size  = condition.get("batch_size", args.batch_size)
    tau_e       = condition.get("tau_e", args.tau_e)

    # Run name encodes sweep type + condition + variant
    run_name = f"diag-{sweep}-{condition['id']}-{variant}"

    ckpt_dir = f"checkpoints/eprop-diag/{args.wandb_group}/{sweep}/{condition['id']}/{variant}"

    cmd = [sys.executable, "scripts/train.py", "--config", exp["config"]]
    if args.wandb:
        cmd.append("--wandb")
    if args.device:
        cmd += ["--device", args.device]

    overrides = [
        "learning.rule=eprop_hybrid",
        "learning.reset_state_between_batches=true",
        "column.apical_pathway=additive",
        f"learning.hybrid_eprop_steps={eprop_steps}",
        f"learning.hybrid_bptt_steps={bptt_steps}",
        f"learning.hybrid_bptt_lr={bptt_lr}",
        f"learning.hybrid_bptt_scope={args.bptt_scope}",
        "synapse.inter_column_stp=false",
        f"training.batch_size={batch_size}",
        f"training.max_tokens={args.max_tokens}",
        f"training.lr={args.lr}",
        f"training.checkpoint_dir={ckpt_dir}",
        f"logging.project={args.wandb_project}",
        f"logging.group={args.wandb_group}",
        f"name={run_name}",
    ]

    if args.tokenizer:
        overrides.append(f"data.tokenizer_path={args.tokenizer}")
    if tau_e is not None:
        overrides.append(f"learning.eprop_tau_e={tau_e}")

    log_tokens = min(51_200, args.max_tokens // 20)
    overrides += [
        f"training.log_tokens={log_tokens}",
        f"training.eval_tokens={log_tokens * 2}",
    ]

    for ov in (args.override or []):
        overrides.append(ov)

    cmd += ["--override"] + overrides
    return cmd


# ── Runner ────────────────────────────────────────────────────────────────────

def run_condition(condition: dict, cmd: list[str], dry_run: bool) -> bool:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  [{condition['id']}]  {condition['label']}")
    print(f"  Command: {' '.join(cmd)}")
    print(sep)

    if dry_run:
        print("  (dry run -- skipping)")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{condition['id']}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  [{condition['id']}] completed in {elapsed/3600:.1f}h")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Focused diagnostic sweeps for e-prop hyperparameters"
    )

    # Sweep selection
    parser.add_argument("--sweep", required=True,
                        choices=["ratio", "tau_e", "batch"],
                        help="Which diagnostic sweep to run.")
    parser.add_argument("--variant", default="1k",
                        choices=list(EXPERIMENTS.keys()),
                        help="Architecture variant to run against (default: 1k).")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Condition IDs to skip (e.g. --skip 2:1 to skip the baseline).")

    # Hybrid defaults (shared across all sweeps; overridden per condition where relevant)
    parser.add_argument("--eprop-steps", type=int, default=20,
                        help="E-prop steps (used as default for tau_e and batch sweeps; default: 20).")
    parser.add_argument("--bptt-steps", type=int, default=10,
                        help="BPTT steps per cycle (default: 10).")
    parser.add_argument("--bptt-lr", type=float, default=3e-4,
                        help="BPTT consolidation LR (default: 3e-4).")
    parser.add_argument("--bptt-scope", default="full",
                        choices=["full", "readout_only"],
                        help="BPTT scope (default: full).")

    # Sweep-specific value lists
    parser.add_argument("--tau-e-values", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256],
                        help="tau_e values to sweep (default: 16 32 64 128 256).")
    parser.add_argument("--batch-values", type=int, nargs="+",
                        default=[32, 16, 8, 4],
                        help="Batch sizes to sweep (default: 32 16 8 4).")

    # Training
    parser.add_argument("--tau-e", type=int, default=None,
                        help="Fixed tau_e override for ratio/batch sweeps (default: from config).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Fixed batch size for ratio/tau_e sweeps (default: 32).")
    parser.add_argument("--max-tokens", type=int, default=50_000_000,
                        help="Token budget per condition (default: 50M).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate (default: 1e-4).")

    # Infrastructure
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl")
    parser.add_argument("--device", default=None,
                        help="Device override: cuda, mps, cpu (default: auto).")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",
                        default=f"eprop-diag-{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    exp = EXPERIMENTS[args.variant]

    # Build condition list for selected sweep
    if args.sweep == "ratio":
        conditions = ratio_sweep(bptt_lr=args.bptt_lr)
    elif args.sweep == "tau_e":
        conditions = tau_e_sweep(
            values=args.tau_e_values,
            eprop_steps=args.eprop_steps,
            bptt_steps=args.bptt_steps,
            bptt_lr=args.bptt_lr,
        )
    else:  # batch
        conditions = batch_sweep(
            values=args.batch_values,
            eprop_steps=args.eprop_steps,
            bptt_steps=args.bptt_steps,
            bptt_lr=args.bptt_lr,
        )

    if args.skip:
        conditions = [c for c in conditions if c["id"] not in args.skip]

    sep = "=" * 70
    print(f"\nE-prop diagnostic sweep -- {args.sweep}")
    print(f"  Variant     : {args.variant}  ({exp['config']})")
    print(f"  Conditions  : {[c['id'] for c in conditions]}")
    print(f"  Max tokens  : {args.max_tokens:,}")
    print(f"  BPTT LR     : {args.bptt_lr}")
    print(f"  W&B group   : {args.wandb_group}")
    print(f"  Tokenizer   : {args.tokenizer or '(from config)'}")
    if args.device:
        print(f"  Device      : {args.device}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    failed = []
    for condition in conditions:
        cmd = build_command(condition, exp, args)
        ok = run_condition(condition, cmd, args.dry_run)
        if not ok:
            failed.append(condition["id"])
            if args.stop_on_failure:
                print("  --stop-on-failure: aborting.")
                break

    print(f"\n{sep}")
    print(f"  Summary: {len(conditions)-len(failed)}/{len(conditions)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
