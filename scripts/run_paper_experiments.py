#!/usr/bin/env python3
"""
scripts/run_paper_experiments.py — Reproduce all experiments in the CortexLM paper.

All runs use TinyStories (full dataset, no repeated epochs), BPE vocab 4096.
Architecture: ~650K parameters.  Tokenizer: tokenizers/tinystories_bpe4096.pkl.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment groups
-----------------

  exp1_dawn         §3.1  Architecture ablation under DAWN learning (with apical)
                          E/I → layered → AdEx → STP → HPC → HPC+CA1
                          Establishes monotonically improving architecture series.

  exp1_no_apical    §3.3  Same series WITHOUT apical pathway.
                          Shows learning-signal collapse; motivates apical requirement.

  exp2_bptt         §4.1  Same architecture series under pure BPTT.
                          Non-monotonic; HPC hurts; key figure for learning rule section.

  exp2_learning_rule §4.2  Learning rule ablation on best architecture (1f):
                          eprop_only → bptt_only → hybrid_fixed → hybrid_sgdr → hybrid_dawn
                          Note: hybrid_dawn is identical to exp1_dawn/1k; that run is reused.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DAWN settings (Experiment 1 and hybrid_dawn in Experiment 2):
  learning.rule             = eprop_hybrid
  learning.eprop_mode       = vectorized
  Batch sizes               = 16 (e-prop) / 128 (BPTT consolidation)
  Wake/sleep ratio          = anneals 20:10 → 10:20 → 5:25 → 2:28 → 0:30
  SGDR cycle                = 65.1M tokens  (521_044_049 // 8, exactly 8 cycles over full TinyStories)
  column.apical_pathway     = additive  (exp1_dawn only; omitted in exp1_no_apical)

Usage
-----
# Dry run to see all commands
python scripts/run_paper_experiments.py --groups all --dry-run

# Run a single group
python scripts/run_paper_experiments.py --groups exp1_dawn --wandb --wandb-offline

# Run specific phases within a group
python scripts/run_paper_experiments.py --groups exp1_dawn --phases 1f 1k --wandb --wandb-offline

# Run multiple groups in sequence
python scripts/run_paper_experiments.py --groups exp1_dawn exp2_bptt --wandb --wandb-offline

# Full paper reproduction (long — run on cluster)
python scripts/run_paper_experiments.py --groups all --wandb --wandb-offline

# Override token budget (e.g. for a quick sanity check)
python scripts/run_paper_experiments.py --groups exp1_dawn --phases 1a \\
    --override training.max_tokens=5000000 --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

# ── Constants ─────────────────────────────────────────────────────────────────

TOKENIZER  = "tokenizers/tinystories_bpe4096.pkl"
SEQ_LEN    = 256

# SGDR cycle calibrated for full TinyStories (521,044,049 train tokens, 8 cycles).
# Exact: 521_044_049 // 8 = 65_130_506.
SGDR_TOKENS = 65_130_506   # exactly 8 cycles over full TinyStories train set

# Token budget: set very large; training stops at end-of-data via no_repeat=true.
# Acts as a safety ceiling only.
MAX_TOKENS_NO_REPEAT = 2_000_000_000   # 2B >> full TinyStories

# DAWN learning overrides — shared by exp1_dawn and exp2_learning_rule/hybrid_dawn
DAWN_OVERRIDES = [
    "learning.rule=eprop_hybrid",
    "learning.eprop_mode=vectorized",
    "learning.hybrid_eprop_steps=20",
    "learning.hybrid_bptt_steps=10",
    "learning.hybrid_bptt_batch_size=128",
    "learning.hybrid_bptt_scope=full",
    "learning.reset_state_between_batches=true",
    f"learning.sgdr_restart_tokens={SGDR_TOKENS}",
    "learning.hybrid_eprop_steps_schedule=[20,10,5,2,0]",
    "learning.hybrid_bptt_steps_schedule=[10,20,25,28,30]",
    "training.batch_size=16",
    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
    "training.no_repeat=true",
]

# ── Architecture ablation phases ──────────────────────────────────────────────
# Paper order: E/I → layered → AdEx → STP → HPC → HPC+CA1
# (Note: AdEx before STP differs from config numbering 1c/1d; IDs kept canonical.)

ARCH_PHASES = [
    {
        "id":     "1a",
        "label":  "E/I (rate neurons + simple_ei column)",
        "config": "configs/phase1a_minimal.yaml",
    },
    {
        "id":     "1b",
        "label":  "Layered cortical columns",
        "config": "configs/phase1b_layered.yaml",
    },
    {
        "id":     "1c",
        "label":  "Tsodyks-Markram STP",
        "config": "configs/phase1c_stp.yaml",
    },
    {
        "id":     "1d",
        "label":  "AdEx adaptive neurons",
        "config": "configs/phase1d_adex.yaml",
    },
    {
        "id":     "1f",
        "label":  "Modern Hopfield CA3  [best architecture]",
        "config": "configs/phase1f_hopfield.yaml",
    },
    {
        "id":     "1m",
        "label":  "Thalamic relay (thalamocortical projection)",
        "config": "configs/phase1m_thalamus.yaml",
    },
    {
        "id":     "1n",
        "label":  "Thalamic relay + L6 corticothalamic shortcut",
        "config": "configs/phase1n_thalamus_l6.yaml",
    },
]

ARCH_PHASE_IDS = [p["id"] for p in ARCH_PHASES]

# ── Group definitions ─────────────────────────────────────────────────────────

GROUPS = {
    # ── Experiment 1 ──────────────────────────────────────────────────────────
    "exp1_dawn": {
        "label":   "Exp 1 — Architecture ablation with DAWN (with apical)  [§3.1]",
        "section": "3.1",
        "wandb_group_default": "paper-exp1-dawn",
        "phases": [
            {**p,
             "run_prefix": "exp1-dawn",
             "extra": ["column.apical_pathway=additive"] + DAWN_OVERRIDES}
            for p in ARCH_PHASES
        ],
    },
    "exp1_no_apical": {
        "label":   "Exp 1 — Apical pathway finding: same series WITHOUT apical  [§3.3]",
        "section": "3.3",
        "wandb_group_default": "paper-exp1-no-apical",
        "phases": [
            {**p,
             "run_prefix": "exp1-noapical",
             "extra": DAWN_OVERRIDES}   # no apical override
            for p in ARCH_PHASES
        ],
    },

    # ── Experiment 2a ─────────────────────────────────────────────────────────
    "exp2_bptt": {
        "label":   "Exp 2a — Architecture ablation under pure BPTT (with apical)  [§4.1]",
        "section": "4.1",
        "wandb_group_default": "paper-exp2-bptt",
        "phases": [
            {**p,
             "run_prefix": "exp2-bptt",
             "extra": [
                 "column.apical_pathway=additive",
                 "learning.rule=bptt",
                 f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                 "training.no_repeat=true",
                 # batch_size from config (512); no override needed
             ]}
            for p in ARCH_PHASES
        ],
    },

    # ── Experiment 2b ─────────────────────────────────────────────────────────
    "exp2_learning_rule": {
        "label":   "Exp 2b — Learning rule ablation on best architecture (1f)  [§4.2]",
        "section": "4.2",
        "wandb_group_default": "paper-exp2-lr",
        "phases": [
            {
                "id":     "eprop_only",
                "label":  "e-prop only (vectorized, no consolidation)",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": [
                    "column.apical_pathway=additive",
                    "learning.rule=eprop",
                    "learning.eprop_mode=vectorized",
                    "training.batch_size=16",
                    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                    "training.no_repeat=true",
                ],
            },
            {
                "id":     "bptt_only",
                "label":  "BPTT only",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": [
                    "column.apical_pathway=additive",
                    "learning.rule=bptt",
                    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                    "training.no_repeat=true",
                ],
            },
            {
                "id":     "hybrid_fixed",
                "label":  "Hybrid 20:10 fixed (no SGDR, no annealing)",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": [
                    "column.apical_pathway=additive",
                    "learning.rule=eprop_hybrid",
                    "learning.eprop_mode=vectorized",
                    "learning.hybrid_eprop_steps=20",
                    "learning.hybrid_bptt_steps=10",
                    "learning.hybrid_bptt_batch_size=128",
                    "learning.hybrid_bptt_scope=full",
                    "learning.reset_state_between_batches=true",
                    "training.batch_size=16",
                    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                    "training.no_repeat=true",
                ],
            },
            {
                "id":     "hybrid_sgdr",
                "label":  "Hybrid 20:10 + SGDR (no annealing)",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": [
                    "column.apical_pathway=additive",
                    "learning.rule=eprop_hybrid",
                    "learning.eprop_mode=vectorized",
                    "learning.hybrid_eprop_steps=20",
                    "learning.hybrid_bptt_steps=10",
                    "learning.hybrid_bptt_batch_size=128",
                    "learning.hybrid_bptt_scope=full",
                    "learning.reset_state_between_batches=true",
                    f"learning.sgdr_restart_tokens={SGDR_TOKENS}",
                    "training.batch_size=16",
                    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                    "training.no_repeat=true",
                ],
            },
            {
                "id":     "hybrid_dawn_no_sgdr",
                "label":  "Hybrid + DAWN annealing, NO SGDR  (isolates annealing vs LR-restart contributions)",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": [
                    "column.apical_pathway=additive",
                    "learning.rule=eprop_hybrid",
                    "learning.eprop_mode=vectorized",
                    "learning.hybrid_eprop_steps=20",
                    "learning.hybrid_bptt_steps=10",
                    "learning.hybrid_bptt_batch_size=128",
                    "learning.hybrid_bptt_scope=full",
                    "learning.reset_state_between_batches=true",
                    # No sgdr_restart_tokens — flat cosine LR throughout
                    f"learning.hybrid_phase_trigger_tokens={SGDR_TOKENS}",  # phases fire at same intervals
                    "learning.hybrid_eprop_steps_schedule=[20,10,5,2,0]",
                    "learning.hybrid_bptt_steps_schedule=[10,20,25,28,30]",
                    "training.batch_size=16",
                    f"training.max_tokens={MAX_TOKENS_NO_REPEAT}",
                    "training.no_repeat=true",
                ],
            },
            {
                "id":     "hybrid_dawn",
                "label":  "Hybrid + SGDR + DAWN annealing  [= exp1_dawn/1f — reuse if already run]",
                "config": "configs/phase1f_hopfield.yaml",
                "run_prefix": "exp2-lr",
                "extra": ["column.apical_pathway=additive"] + DAWN_OVERRIDES,
            },
        ],
    },
}

GROUP_IDS = list(GROUPS.keys())


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(phase: dict, args: argparse.Namespace) -> list[str]:
    run_name = f"{phase['run_prefix']}-{phase['id']}"
    ckpt_dir = f"checkpoints/paper/{phase['run_prefix']}/{phase['id']}"

    overrides = [
        f"data.tokenizer_path={args.tokenizer}",
        f"training.checkpoint_dir={ckpt_dir}",
        f"training.log_tokens=200000",
        f"training.eval_tokens=1000000",
        f"name={run_name}",
    ]
    overrides += phase["extra"]

    if args.wandb:
        overrides += [
            "logging.wandb=true",
            f"logging.project={args.wandb_project}",
            f"logging.group={args.wandb_group}",
        ]

    for ov in (args.override or []):
        overrides.append(ov)

    cmd = [sys.executable, "scripts/train.py",
           "--config", phase["config"],
           "--override"] + overrides
    return cmd


# ── Runner ────────────────────────────────────────────────────────────────────

def run_phase(phase: dict, cmd: list[str], dry_run: bool, wandb_offline: bool) -> bool:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  [{phase['id']}]  {phase['label']}")
    print(f"  Config : {phase['config']}")
    print(f"  Command: {' '.join(cmd)}")
    print(sep)

    if dry_run:
        print("  (dry run — skipping)")
        return True

    env = os.environ.copy()
    if wandb_offline:
        env["WANDB_MODE"] = "offline"

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  !! [{phase['id']}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  [{phase['id']}] completed in {elapsed/3600:.1f}h")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all CortexLM paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {gid:<22} {GROUPS[gid]['label']}" for gid in GROUP_IDS
        ),
    )
    parser.add_argument("--groups", nargs="+", default=["exp1_dawn"],
                        help=f"Group(s) to run, or 'all'. Choices: {GROUP_IDS}")
    parser.add_argument("--phases", nargs="+", default=None,
                        help="Run only these phase IDs within the selected group(s). "
                             "Example: --phases 1f 1k")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Set WANDB_MODE=offline (log locally, sync later)")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group override (default: per-group value from script)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra key=value overrides forwarded to every run.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.groups == ["all"]:
        args.groups = GROUP_IDS

    unknown = [g for g in args.groups if g not in GROUPS]
    if unknown:
        parser.error(f"Unknown group(s): {unknown}. Choices: {GROUP_IDS}")

    # Count total phases to run
    total = sum(
        len([p for p in GROUPS[g]["phases"]
             if args.phases is None or p["id"] in args.phases])
        for g in args.groups
    )

    print(f"\nCortexLM paper experiments")
    print(f"  Groups   : {args.groups}")
    print(f"  Phases   : {args.phases or '(all)'}")
    print(f"  Total    : {total} run(s)")
    print(f"  no_repeat: true — training stops at end of TinyStories")
    print(f"  SGDR T0  : {SGDR_TOKENS/1e6:.1f}M tokens (~8 cycles over full dataset)")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print()

    passed, failed = [], []

    for group_id in args.groups:
        group = GROUPS[group_id]
        wandb_group = args.wandb_group or group["wandb_group_default"]

        # Temporarily set wandb_group for this group's runs
        _orig = args.wandb_group
        args.wandb_group = wandb_group

        print(f"\n{'━'*72}")
        print(f"  GROUP: {group['label']}")
        print(f"  W&B group: {wandb_group}")
        print(f"{'━'*72}")

        phases = group["phases"]
        if args.phases is not None:
            phases = [p for p in phases if p["id"] in args.phases]
            missing = [pid for pid in args.phases
                       if pid not in {p["id"] for p in group["phases"]}]
            if missing:
                print(f"  Warning: phase(s) {missing} not found in {group_id} — skipping")

        for phase in phases:
            cmd = build_command(phase, args)
            ok  = run_phase(phase, cmd, args.dry_run, args.wandb_offline)
            tag = f"{group_id}/{phase['id']}"
            (passed if ok else failed).append(tag)
            if not ok and args.stop_on_failure:
                print("  --stop-on-failure: aborting.")
                args.wandb_group = _orig
                _print_summary(passed, failed)
                sys.exit(1)

        args.wandb_group = _orig

    _print_summary(passed, failed)
    sys.exit(0 if not failed else 1)


def _print_summary(passed, failed):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Done: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
    if passed:
        print(f"  Passed: {passed}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
