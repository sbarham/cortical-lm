"""Main training entry point.

Usage:
    python scripts/train.py --config configs/standard.yaml [--resume path/to/checkpoint]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cortexlm.utils.config import get_config
from cortexlm.utils.logging import Logger, setup_logging
from cortexlm.model import CortexLM
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer
from cortexlm.learning import get_trainer


def _count_params_and_exit(config: dict):
    """Build the model using the configured vocab_size and print a parameter breakdown."""
    from cortexlm.model import CortexLM

    # Use configured vocab_size as an estimate — actual BPE vocab may differ by a few %.
    vocab_size = config["data"].get("vocab_size", 4096)
    print(f"Parameter count  (vocab_size={vocab_size:,} -- configured target)")
    print(f"  Note: actual BPE vocab may differ slightly; run without --count-params to see exact count.")
    print()

    model = CortexLM(config, vocab_size)

    # Per-component breakdown
    components = {
        "embedding":    model.embedding,
        "columns":      model.columns,
        "connectivity": model.connectivity,
        "hippocampus":  model.hippocampus,
        "readout":      model.readout,
        "hpc_proj":     model.hpc_input_proj,
    }

    total = 0
    rows = []
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        total += n
        rows.append((name, n))

    width = max(len(r[0]) for r in rows)
    for name, n in rows:
        bar = "#" * max(1, round(40 * n / max(total, 1)))
        print(f"  {name:<{width}}  {n:>10,}  {bar}")
    print(f"  {'TOTAL':<{width}}  {total:>10,}")

    # Column detail
    n_cols = model.n_columns
    col_params = sum(p.numel() for p in model.columns.parameters() if p.requires_grad)
    print(f"\n  Column params total: {col_params:,}  ({n_cols} columns, ~{col_params//n_cols:,} each)")

    sys.exit(0)


def _make_run_tag(config: dict) -> str:
    """
    Build a compact, human-readable tag from the most discriminating
    hyperparameters.  Used as the checkpoint subdirectory name so that
    multiple runs of the same base config (different sizes, LRs, etc.)
    don't collide.

    Example: cortex-lm-minimal_sei-rate_c8e40_bs32_lr3e-4
    """
    name = config.get("name", "run").replace(" ", "_")

    # Column model: shorten to avoid long paths
    col_model_map = {"simple_ei": "sei", "layered": "lyr"}
    col_model = col_model_map.get(config["column"]["model"], config["column"]["model"])

    # Neuron model
    neuron_map = {"rate": "rate", "rate_adex": "adex", "lif": "lif"}
    neuron = neuron_map.get(config["neuron"]["model"], config["neuron"]["model"])

    # Number of columns
    n_cols = config["column"]["n_columns"]

    # Neurons per column: for simple_ei use n_e; for layered use L5 n_e as proxy
    if config["column"]["model"] == "simple_ei":
        n_e = config["column"].get("n_e", "?")
    else:
        n_e = config["column"].get("layer_sizes", {}).get("l5", {}).get("n_e", "?")

    # Training
    bs = config["training"]["batch_size"]
    lr = config["training"]["lr"]

    # Compact LR: 0.0003 -> 3e-4, 0.001 -> 1e-3
    def fmt_lr(v):
        s = f"{v:.0e}"                  # e.g. '3e-04'
        s = s.replace("e-0", "e-").replace("e+0", "e")  # '3e-4'
        return s

    return f"{name}_{col_model}-{neuron}_c{n_cols}e{n_e}_bs{bs}_lr{fmt_lr(lr)}"


def main():
    parser = argparse.ArgumentParser(description="Train CortexLM")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as key=value pairs (e.g. training.lr=1e-3)")
    parser.add_argument("--count-params", action="store_true",
                        help="Print parameter count breakdown and exit (no training)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to a saved tokenizer.pkl (skips BPE retraining)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing token cache and re-tokenize from scratch")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (overrides config)")
    args = parser.parse_args()

    # Parse overrides — dotted keys like "training.batch_size=128" become nested dicts
    def _parse_val(v):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    def _nested_set(d, dotted_key, value):
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    overrides = {}
    for kv in (args.override or []):
        k, v = kv.split("=", 1)
        _nested_set(overrides, k, _parse_val(v))

    config = get_config(args.config, overrides if overrides else None)
    if args.wandb:
        config.setdefault("logging", {})["wandb"] = True
    setup_logging()

    # ── Auto-derive checkpoint directory from run name ──────────────────────
    # If the config still has the generic default, use checkpoints/<run-name>
    # so different experiments don't overwrite each other.
    # An explicit setting in the YAML or --override always takes precedence.
    if config["training"].get("checkpoint_dir", "checkpoints") == "checkpoints":
        config["training"]["checkpoint_dir"] = os.path.join("checkpoints", _make_run_tag(config))

    # ── Count-params dry run ────────────────────────────────────────────────
    if args.count_params:
        _count_params_and_exit(config)

    # Set seed
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"  GPU:  {props.name}  VRAM={total_gb:.0f}GB")
        print(f"  Tip:  if vram stays below ~60% during training, increase batch_size")
    print()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("[ 1/4 ] Building tokenizer")
    if args.tokenizer:
        import pickle
        print(f"  Loading tokenizer from {args.tokenizer}")
        with open(args.tokenizer, "rb") as _f:
            tokenizer = pickle.load(_f)
    else:
        tokenizer = build_tokenizer(config)
    vocab_size = tokenizer.vocab_size
    config["data"]["vocab_size"] = vocab_size

    # Persist tokenizer so post-training scripts don't need to rebuild it
    import pickle
    ckpt_dir = config["training"].get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tok_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    with open(tok_path, "wb") as _f:
        pickle.dump(tokenizer, _f)
    print(f"  Tokenizer saved → {tok_path}")
    print()

    # ── Datasets ───────────────────────────────────────────────────────────
    print("[ 2/4 ] Loading datasets")
    train_ds, val_ds, test_ds, tokenizer = get_dataset(
        config, tokenizer, force_rebuild=args.no_cache
    )
    train_loader = make_dataloader(train_ds, config, shuffle=True)
    val_loader   = make_dataloader(val_ds, config, shuffle=False)
    print()

    # ── Model ──────────────────────────────────────────────────────────────
    print("[ 3/4 ] Building model")
    model = CortexLM(config, vocab_size)
    n_params = model.count_parameters()
    print(f"  CortexLM | params={n_params:,} | columns={config['column']['n_columns']} "
          f"| neuron={config['neuron']['model']} | hpc={config['hippocampus']['model']}")
    print()

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # ── Training ───────────────────────────────────────────────────────────
    print("[ 4/4 ] Training")
    trainer = get_trainer(model, config, device, tokenizer=tokenizer)
    logger = Logger(config)

    tcfg = config["training"]
    from cortexlm.learning.bptt import _resolve_max_steps
    _max_steps = _resolve_max_steps(config)
    print(f"  rule={config['learning']['rule']} | steps={_max_steps:,} "
          f"| batch={tcfg['batch_size']} | lr={tcfg['lr']} | seq_len={config['data']['seq_len']}")
    print()

    trainer.train(train_loader, val_loader, logger=logger)

    logger.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
