#!/usr/bin/env python3
"""
scripts/run_cclm_1d_sweep.py — Hyperparameter sweep over the CCLM architecture 1d.

Architecture 1d = layered cortical columns + AdEx adaptive neurons + STP, NO hippocampus.
All variants use BPTT + SGDR (fast; ~45 min/run on H100).

Sweep organisation
------------------
  Session 1 (rdout*)  : Readout architecture — depth, width, activation, weight tying
  Session 2 (col*)    : Column structure — layer sizes, n_columns, L6
  Session 3 (con*)    : Connectivity — sigma, p_max, inter-column sparsity
  Session 4 (nrn*)    : Neuron dynamics — tau_w_range, tau_m_range, adaptation_a
  Session 5 (trn*)    : Training — lr, BPTT horizon, batch size, weight decay
  Session 6 (cmb*)    : Combinations (TBD after sessions 1–5 complete)

Parameter target: ~635K (620K–650K acceptable).
Base config: configs/phase1d_adex.yaml

Usage
-----
# Dry-run — show all commands
python scripts/run_cclm_1d_sweep.py --dry-run

# Run a full session
python scripts/run_cclm_1d_sweep.py --session 1 --wandb --wandb-offline

# Run a single variant
python scripts/run_cclm_1d_sweep.py --runs rdout01 --wandb --wandb-offline

# Run all sessions in series
python scripts/run_cclm_1d_sweep.py --session all --wandb --wandb-offline

# Print srun commands (one per variant) for parallel cluster dispatch
python scripts/run_cclm_1d_sweep.py --session 1 \\
    --srun-prefix 'srun --gres=gpu:1 -n1 --time=01:30:00'
"""

import argparse
import os
import subprocess
import sys
import time

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_CONFIG = "configs/phase1d_adex.yaml"
TOKENIZER   = "tokenizers/tinystories_bpe4096.pkl"
MAX_TOKENS  = 100_000_000          # 100M tokens — matches paper budget
SGDR_TOKENS = 20_000_000           # 20M per cycle × 5 = 100M (matches DAWN)

BPTT_SGDR_OVERRIDES = [
    "learning.rule=bptt",
    "training.scheduler=sgdr",
    f"training.sgdr_t0_tokens={SGDR_TOKENS}",
    "training.sgdr_t_mult=1",
    f"training.max_tokens={MAX_TOKENS}",
    "column.apical_pathway=additive",
]

# ── Variant definitions ───────────────────────────────────────────────────────
# Each variant is a dict with:
#   id       str   Short unique identifier
#   label    str   Human-readable description
#   session  int   Session number (1–6)
#   extra    list  Additional overrides on top of BPTT_SGDR_OVERRIDES
#
# Layer size overrides use the nested dot-notation:
#   column.layer_sizes.l23.n_e=64
# These are resolved via the YAML-aware _parse_val in train.py.

VARIANTS = [

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 1 — Readout architecture
    # Baseline readout: n_layers=1, hidden_dim=128 (from phase1d_adex.yaml)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id":      "rdout01",
        "session": 1,
        "label":   "Readout baseline (n_layers=1, hidden=128, relu) [base]",
        "extra":   [],
    },
    {
        "id":      "rdout02",
        "session": 1,
        "label":   "Readout depth-2 (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2"],
    },
    {
        "id":      "rdout03",
        "session": 1,
        "label":   "Readout depth-3 (n_layers=3, hidden=128)",
        "extra":   ["readout.n_layers=3"],
    },
    {
        "id":      "rdout04",
        "session": 1,
        "label":   "Readout wider (n_layers=1, hidden=256)",
        "extra":   ["readout.hidden_dim=256"],
    },
    {
        "id":      "rdout05",
        "session": 1,
        "label":   "Readout depth-2 wider (n_layers=2, hidden=256)",
        "extra":   ["readout.n_layers=2", "readout.hidden_dim=256"],
    },
    {
        "id":      "rdout06",
        "session": 1,
        "label":   "Readout GELU (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2", "readout.activation=gelu"],
    },
    {
        "id":      "rdout07",
        "session": 1,
        "label":   "Readout Swish/SiLU (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2", "readout.activation=swish"],
    },
    {
        "id":      "rdout08",
        "session": 1,
        "label":   "Readout SwiGLU (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2", "readout.activation=swiglu"],
    },
    {
        "id":      "rdout09",
        "session": 1,
        "label":   "Readout SwiGLU depth-3 (n_layers=3, hidden=128)",
        "extra":   ["readout.n_layers=3", "readout.activation=swiglu"],
    },
    {
        "id":      "rdout10",
        "session": 1,
        "label":   "Readout SwiGLU + residuals (n_layers=3, hidden=256)",
        "extra":   ["readout.n_layers=3", "readout.hidden_dim=256",
                    "readout.activation=swiglu", "readout.residual=true"],
    },
    {
        "id":      "rdout11",
        "session": 1,
        "label":   "Readout weight tying + depth-2 (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2", "readout.weight_tying=true"],
    },
    {
        "id":      "rdout12",
        "session": 1,
        "label":   "Readout SwiGLU + weight tying + depth-2 (n_layers=2, hidden=128)",
        "extra":   ["readout.n_layers=2", "readout.activation=swiglu",
                    "readout.weight_tying=true"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 2 — Column structure
    # Baseline: n_columns=8, L4(16e/4i), L23(32e/8i), L5(16e/4i), L6(12e/3i)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id":      "col01",
        "session": 2,
        "label":   "Column baseline [same as rdout01 — reference]",
        "extra":   [],
    },
    {
        "id":      "col02",
        "session": 2,
        "label":   "Wider L2/3 (n_l23e=48, n_l23i=12)",
        "extra":   ["column.layer_sizes.l23.n_e=48", "column.layer_sizes.l23.n_i=12"],
    },
    {
        "id":      "col03",
        "session": 2,
        "label":   "Wider L5 excitatory (n_l5e=24, n_l5i=6)",
        "extra":   ["column.layer_sizes.l5.n_e=24", "column.layer_sizes.l5.n_i=6"],
    },
    {
        "id":      "col04",
        "session": 2,
        "label":   "Wider L4 (n_l4e=24, n_l4i=6)",
        "extra":   ["column.layer_sizes.l4.n_e=24", "column.layer_sizes.l4.n_i=6"],
    },
    {
        "id":      "col05",
        "session": 2,
        "label":   "Deeper columns: all layers wider (L4×1.5, L23×1.5, L5×1.5, L6×1.5)",
        "extra":   [
            "column.layer_sizes.l4.n_e=24",  "column.layer_sizes.l4.n_i=6",
            "column.layer_sizes.l23.n_e=48", "column.layer_sizes.l23.n_i=12",
            "column.layer_sizes.l5.n_e=24",  "column.layer_sizes.l5.n_i=6",
            "column.layer_sizes.l6.n_e=18",  "column.layer_sizes.l6.n_i=4",
        ],
    },
    {
        "id":      "col06",
        "session": 2,
        "label":   "No L6 (n_l6e=0, n_l6i=0) — remove corticothalamic layer",
        "extra":   ["column.layer_sizes.l6.n_e=0", "column.layer_sizes.l6.n_i=0"],
    },
    {
        "id":      "col07",
        "session": 2,
        "label":   "More columns: n_columns=12",
        "extra":   ["column.n_columns=12"],
    },
    {
        "id":      "col08",
        "session": 2,
        "label":   "More columns: n_columns=16",
        "extra":   ["column.n_columns=16"],
    },
    {
        "id":      "col09",
        "session": 2,
        "label":   "Fewer columns: n_columns=4",
        "extra":   ["column.n_columns=4"],
    },
    {
        "id":      "col10",
        "session": 2,
        "label":   "Isolated columns (no inter-column connectivity)",
        "extra":   ["connectivity.type=none"],
    },
    {
        "id":      "col11",
        "session": 2,
        "label":   "E/I ratio: more inhibitory (L23: 32e/16i, L4: 16e/8i)",
        "extra":   ["column.layer_sizes.l4.n_i=8", "column.layer_sizes.l23.n_i=16"],
    },
    {
        "id":      "col12",
        "session": 2,
        "label":   "Disinhibition enabled (VIP circuit)",
        "extra":   ["column.disinhibition=true"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 3 — Connectivity
    # Baseline: gaussian_1d, p_max=0.7, sigma=2.0
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id":      "con01",
        "session": 3,
        "label":   "Connectivity baseline [reference]",
        "extra":   [],
    },
    {
        "id":      "con02",
        "session": 3,
        "label":   "Narrow spread sigma=1.0 (more local)",
        "extra":   ["connectivity.sigma=1.0"],
    },
    {
        "id":      "con03",
        "session": 3,
        "label":   "Wide spread sigma=3.0 (more global)",
        "extra":   ["connectivity.sigma=3.0"],
    },
    {
        "id":      "con04",
        "session": 3,
        "label":   "Very wide spread sigma=5.0",
        "extra":   ["connectivity.sigma=5.0"],
    },
    {
        "id":      "con05",
        "session": 3,
        "label":   "Sparse connectivity p_max=0.4",
        "extra":   ["connectivity.p_max=0.4"],
    },
    {
        "id":      "con06",
        "session": 3,
        "label":   "Dense connectivity p_max=0.9",
        "extra":   ["connectivity.p_max=0.9"],
    },
    {
        "id":      "con07",
        "session": 3,
        "label":   "Narrow + dense: sigma=1.0, p_max=0.9",
        "extra":   ["connectivity.sigma=1.0", "connectivity.p_max=0.9"],
    },
    {
        "id":      "con08",
        "session": 3,
        "label":   "Wide + sparse: sigma=4.0, p_max=0.4",
        "extra":   ["connectivity.sigma=4.0", "connectivity.p_max=0.4"],
    },
    {
        "id":      "con09",
        "session": 3,
        "label":   "More columns + wide spread: n_columns=16, sigma=3.0",
        "extra":   ["column.n_columns=16", "connectivity.sigma=3.0"],
    },
    {
        "id":      "con10",
        "session": 3,
        "label":   "Uniform random connectivity",
        "extra":   ["connectivity.type=random"],
    },
    {
        "id":      "con11",
        "session": 3,
        "label":   "Very narrow sigma=0.5 (nearly isolated)",
        "extra":   ["connectivity.sigma=0.5"],
    },
    {
        "id":      "con12",
        "session": 3,
        "label":   "Medium sigma=2.0, denser p_max=0.85",
        "extra":   ["connectivity.p_max=0.85"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 4 — Neuron dynamics (AdEx)
    # Baseline: tau_m_range=[2,30], tau_w_range=[30,500], adaptation_a=0.1
    # NOTE: AdEx tanh nonlinearity is NOT varied (biologically motivated).
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id":      "nrn01",
        "session": 4,
        "label":   "Neuron baseline [reference]",
        "extra":   [],
    },
    {
        "id":      "nrn02",
        "session": 4,
        "label":   "No adaptation (adaptation_a=0.0)",
        "extra":   ["neuron.adaptation_a=0.0"],
    },
    {
        "id":      "nrn03",
        "session": 4,
        "label":   "Weak adaptation (adaptation_a=0.05)",
        "extra":   ["neuron.adaptation_a=0.05"],
    },
    {
        "id":      "nrn04",
        "session": 4,
        "label":   "Strong adaptation (adaptation_a=0.3)",
        "extra":   ["neuron.adaptation_a=0.3"],
    },
    {
        "id":      "nrn05",
        "session": 4,
        "label":   "Very strong adaptation (adaptation_a=0.5)",
        "extra":   ["neuron.adaptation_a=0.5"],
    },
    {
        "id":      "nrn06",
        "session": 4,
        "label":   "Fast adaptation timescale tau_w=[10,100]ms",
        "extra":   ["neuron.tau_w_range=[10,100]"],
    },
    {
        "id":      "nrn07",
        "session": 4,
        "label":   "Slow adaptation timescale tau_w=[100,1000]ms",
        "extra":   ["neuron.tau_w_range=[100,1000]"],
    },
    {
        "id":      "nrn08",
        "session": 4,
        "label":   "Narrow tau_w range [50,200]ms (less diversity)",
        "extra":   ["neuron.tau_w_range=[50,200]"],
    },
    {
        "id":      "nrn09",
        "session": 4,
        "label":   "Wider tau_m range [2,100]ms",
        "extra":   ["neuron.tau_m_range=[2,100]"],
    },
    {
        "id":      "nrn10",
        "session": 4,
        "label":   "Narrow tau_m range [5,20]ms",
        "extra":   ["neuron.tau_m_range=[5,20]"],
    },
    {
        "id":      "nrn11",
        "session": 4,
        "label":   "Learned taus (learn_taus=true)",
        "extra":   ["neuron.learn_taus=true"],
    },
    {
        "id":      "nrn12",
        "session": 4,
        "label":   "Strong adaptation + slow tau_w: a=0.3, tau_w=[100,1000]",
        "extra":   ["neuron.adaptation_a=0.3", "neuron.tau_w_range=[100,1000]"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 5 — Training hyperparameters
    # Baseline: lr=3e-4, truncated_bptt_k=32, batch_size=512, weight_decay=1e-4
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id":      "trn01",
        "session": 5,
        "label":   "Training baseline [reference]",
        "extra":   [],
    },
    {
        "id":      "trn02",
        "session": 5,
        "label":   "Low LR (lr=1e-4)",
        "extra":   ["training.lr=1e-4"],
    },
    {
        "id":      "trn03",
        "session": 5,
        "label":   "High LR (lr=6e-4)",
        "extra":   ["training.lr=6e-4"],
    },
    {
        "id":      "trn04",
        "session": 5,
        "label":   "Very high LR (lr=1e-3)",
        "extra":   ["training.lr=1e-3"],
    },
    {
        "id":      "trn05",
        "session": 5,
        "label":   "Short BPTT horizon k=16",
        "extra":   ["learning.truncated_bptt_k=16"],
    },
    {
        "id":      "trn06",
        "session": 5,
        "label":   "Long BPTT horizon k=64",
        "extra":   ["learning.truncated_bptt_k=64"],
    },
    {
        "id":      "trn07",
        "session": 5,
        "label":   "Very long BPTT horizon k=128",
        "extra":   ["learning.truncated_bptt_k=128"],
    },
    {
        "id":      "trn08",
        "session": 5,
        "label":   "Smaller batch size (batch_size=256)",
        "extra":   ["training.batch_size=256"],
    },
    {
        "id":      "trn09",
        "session": 5,
        "label":   "Larger batch size (batch_size=1024)",
        "extra":   ["training.batch_size=1024"],
    },
    {
        "id":      "trn10",
        "session": 5,
        "label":   "More regularisation (weight_decay=0.01)",
        "extra":   ["training.weight_decay=0.01"],
    },
    {
        "id":      "trn11",
        "session": 5,
        "label":   "High regularisation (weight_decay=0.1)",
        "extra":   ["training.weight_decay=0.1"],
    },
    {
        "id":      "trn12",
        "session": 5,
        "label":   "Shorter SGDR cycle T0=10M (2× restarts)",
        "extra":   [f"training.sgdr_t0_tokens=10000000"],
    },
    {
        "id":      "trn13",
        "session": 5,
        "label":   "Very high LR (lr=2e-3) — motivated by transformer sweep",
        "extra":   ["training.lr=2e-3"],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 6 — Combinations (fill in after sessions 1–5 complete)
    # ═══════════════════════════════════════════════════════════════════════
    # TBD: inspect best variants from each session, then define combos here.
]

VARIANT_MAP = {v["id"]: v for v in VARIANTS}
SESSION_IDS = sorted({v["session"] for v in VARIANTS})


# ── Command builder ────────────────────────────────────────────────────────────

def build_command(variant: dict, args: argparse.Namespace) -> list[str]:
    vid      = variant["id"]
    ckpt_dir = f"checkpoints/cclm-1d-sweep/{vid}"
    run_name = f"cclm-1d-sweep-{vid}"

    overrides = [
        f"data.tokenizer_path={args.tokenizer}",
        f"training.checkpoint_dir={ckpt_dir}",
        "training.log_tokens=200000",
        "training.eval_tokens=1000000",
        f"name={run_name}",
    ] + BPTT_SGDR_OVERRIDES + variant["extra"]

    if args.wandb:
        group = args.wandb_group
        overrides += [
            "logging.wandb=true",
            f"logging.project={args.wandb_project}",
            f"logging.group={group}",
        ]

    cmd = [sys.executable, "scripts/train.py",
           "--config", args.config,
           "--override"] + overrides
    return cmd


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_variant(variant: dict, cmd: list[str], dry_run: bool, wandb_offline: bool) -> bool:
    sep = "=" * 72
    vid = variant["id"]
    print(f"\n{sep}")
    print(f"  [{vid}]  S{variant['session']}  {variant['label']}")
    print(f"  {' '.join(cmd)}")
    print(sep)
    sys.stdout.flush()

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
        print(f"\n  !! [{vid}] FAILED after {elapsed/3600:.1f}h")
        return False

    print(f"\n  [{vid}] completed in {elapsed/3600:.1f}h")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CCLM 1d hyperparameter sweep (BPTT+SGDR, TinyStories).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--session", default=None,
                        help="Session number(s) to run: 1 | 2 | ... | all. "
                             "Comma-separated for multiple: --session 1,2")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Specific variant IDs to run (overrides --session)")
    parser.add_argument("--config", default=BASE_CONFIG,
                        help=f"Base config (default: {BASE_CONFIG})")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-project", default="cortex-lm")
    parser.add_argument("--wandb-group",   default="cclm-1d-sweep")
    parser.add_argument("--srun-prefix", default=None,
                        help="Print 'srun-prefix python ... --runs <id>' per variant; don't run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Select variants ───────────────────────────────────────────────────
    if args.runs:
        unknown = [r for r in args.runs if r not in VARIANT_MAP]
        if unknown:
            parser.error(f"Unknown variant(s): {unknown}. "
                         f"Valid: {sorted(VARIANT_MAP)}")
        selected = [VARIANT_MAP[r] for r in args.runs]
    elif args.session:
        if args.session == "all":
            selected = VARIANTS
        else:
            sessions = {int(s) for s in args.session.split(",")}
            selected = [v for v in VARIANTS if v["session"] in sessions]
            if not selected:
                parser.error(f"No variants found for session(s) {sessions}")
    else:
        # Default: show help
        parser.print_help()
        print(f"\nAvailable sessions: {SESSION_IDS}")
        print(f"Total variants: {len(VARIANTS)}")
        print("\nVariant IDs:")
        for v in VARIANTS:
            print(f"  S{v['session']}  {v['id']:<12}  {v['label']}")
        return

    # ── srun mode ─────────────────────────────────────────────────────────
    if args.srun_prefix is not None:
        base = [sys.executable, "scripts/run_cclm_1d_sweep.py",
                "--config", args.config,
                "--tokenizer", args.tokenizer]
        if args.wandb:          base += ["--wandb"]
        if args.wandb_offline:  base += ["--wandb-offline"]
        base += ["--wandb-project", args.wandb_project,
                 "--wandb-group",   args.wandb_group]
        for v in selected:
            print(f"{args.srun_prefix} {' '.join(base)} --runs {v['id']}")
        return

    # ── Plan banner ───────────────────────────────────────────────────────
    n_total = len(selected)
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  CCLM 1d sweep — {n_total} variant(s) × "
          f"{MAX_TOKENS//1_000_000}M tokens each")
    print(f"  Base config : {args.config}")
    print(f"  Tokenizer   : {args.tokenizer}")
    print(f"{sep}")
    for i, v in enumerate(selected, 1):
        print(f"  {i:>2}/{n_total}  S{v['session']}  {v['id']:<12}  {v['label']}")
    print(f"{sep}\n")
    sys.stdout.flush()

    # ── Run ───────────────────────────────────────────────────────────────
    passed, failed = [], []
    for run_idx, v in enumerate(selected, 1):
        print(f"\n[{run_idx}/{n_total}]")
        cmd = build_command(v, args)
        ok  = run_variant(v, cmd, dry_run=args.dry_run,
                          wandb_offline=args.wandb_offline)
        (passed if ok else failed).append(v["id"])

    print(f"\n{sep}")
    print(f"  Sweep complete: {len(passed)}/{n_total} passed"
          + (f", {len(failed)} failed: {failed}" if failed else ""))
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
