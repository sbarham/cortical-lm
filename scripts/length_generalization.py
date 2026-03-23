"""Length generalisation experiment.

Tests whether CortexLM — or any checkpoint produced by this codebase — can
generalise beyond the sequence length it was trained on.

The core idea: a model with no positional encoding and purely stateful dynamics
(leaky integrators with heterogeneous τ_m) should degrade *gracefully* as
evaluation length grows beyond the training length.  A Transformer trained with
absolute positional encodings cannot generalise at all.  RNNs can in principle
but often fail in practice.

What the script measures
------------------------
For each evaluation length L in a configurable sweep (e.g. 128, 192, 256, 384,
512, 768, 1024):

  1. Collect N long raw-token sequences from the validation corpus, each at
     least L+1 tokens long.
  2. Feed each sequence token-by-token, carrying state forward continuously
     (no resets within a sequence).
  3. Record per-position cross-entropy averaged over the batch.
  4. Split positions into:
       • "in-distribution"  (positions 1–T_train, where T_train is the
                             seq_len the model was trained on)
       • "out-of-distribution" (positions T_train+1–L)
  5. Also compute position-averaged perplexity for each half separately.

Outputs
-------
  • A plot: per-position mean cross-entropy (solid line) ±1 std (shaded band),
    for each tested length.  A vertical dashed line marks T_train.
  • A summary table: in-dist ppl, OOD ppl, OOD/in-dist ratio for each L.
  • A JSON file with all raw numbers for further analysis.

Multiple checkpoints can be compared on the same axes (e.g. phase1b vs a
baseline LSTM trained with the same pipeline).

Usage
-----
  # Single model
  python scripts/length_generalization.py \\
      --checkpoint checkpoints/phase1b/step_0007999.pt \\
      --lengths 128 192 256 384 512 768 1024 \\
      --n-sequences 200 \\
      --output results/length_gen_phase1b

  # Compare multiple models (label:checkpoint pairs)
  python scripts/length_generalization.py \\
      --models "phase1b:checkpoints/phase1b/step_0007999.pt" \\
               "lstm:checkpoints/lstm/step_0007999.pt" \\
      --lengths 128 256 512 1024 \\
      --n-sequences 200 \\
      --output results/length_gen_comparison

  # Quick smoke test
  python scripts/length_generalization.py \\
      --checkpoint checkpoints/phase1b/step_0007999.pt \\
      --lengths 128 256 \\
      --n-sequences 20 \\
      --output /tmp/length_gen_test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortexlm.model import CortexLM
from cortexlm.utils.metrics import compute_perplexity


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_long_sequences(
    config: dict,
    tokenizer,
    min_length: int,
    n_sequences: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Gather `n_sequences` token sequences each at least `min_length+1` tokens
    from the validation split, returned as a padded tensor
    [n_sequences, min_length+1].

    Sequences are drawn contiguously from the validation corpus (no padding
    needed if min_length <= corpus chunk size; for longer lengths the corpus
    is tiled).
    """
    from cortexlm.data import get_dataset, build_tokenizer

    dataset_name = config["data"]["dataset"]

    # Rebuild the *raw* validation token list at the required length.
    # We temporarily override seq_len so the dataset returns chunks at least
    # as long as min_length.
    cfg = {**config, "data": {**config["data"], "seq_len": min_length}}
    _, val_ds, _, _ = get_dataset(cfg, tokenizer)

    collected: List[torch.Tensor] = []
    indices = list(range(len(val_ds)))

    # Shuffle deterministically so repeated calls are comparable
    rng = np.random.default_rng(seed=0)
    rng.shuffle(indices)

    for idx in indices:
        x, y = val_ds[idx]         # x: [min_length], y: [min_length]
        # Reconstruct the full chunk: tokens[0..min_length] (x) + tokens[min_length] (last of y)
        seq = torch.cat([x, y[-1:]])   # [min_length + 1]
        if seq.shape[0] >= min_length + 1:
            collected.append(seq[:min_length + 1])
        if len(collected) >= n_sequences:
            break

    if len(collected) < n_sequences:
        # Tile if not enough unique sequences (short corpus)
        while len(collected) < n_sequences:
            collected.extend(collected[:n_sequences - len(collected)])

    seqs = torch.stack(collected[:n_sequences], dim=0)  # [n_seq, min_length+1]
    return seqs.to(device)


# ── Per-position evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_position_losses(
    model: CortexLM,
    sequences: torch.Tensor,       # [n_seq, seq_len+1]
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run model token-by-token on `sequences`, returning per-position
    cross-entropy averaged over the batch.

    Returns: ce_per_position [seq_len]  (position i = loss predicting token i+1)
    """
    n_seq, total_len = sequences.shape
    seq_len = total_len - 1
    device = sequences.device

    # Accumulate per-position losses across all batches
    pos_loss_sum = np.zeros(seq_len, dtype=np.float64)
    pos_count    = 0

    model.eval()
    for start in range(0, n_seq, batch_size):
        batch_seqs = sequences[start: start + batch_size]   # [B, total_len]
        B = batch_seqs.shape[0]

        state = model.init_state(B)
        batch_pos_losses = np.zeros(seq_len, dtype=np.float64)

        for t in range(seq_len):
            x_t = batch_seqs[:, t]          # [B]
            y_t = batch_seqs[:, t + 1]      # [B] — next token

            logits, state = model.step(x_t, state)   # logits: [B, vocab]

            # Per-example loss at this position, then mean over batch
            loss_t = F.cross_entropy(logits, y_t, reduction="mean")
            batch_pos_losses[t] = loss_t.item()

        pos_loss_sum += batch_pos_losses
        pos_count    += 1

    return pos_loss_sum / max(pos_count, 1)   # [seq_len]


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(
    pos_losses: np.ndarray,   # [seq_len]
    t_train: int,
) -> Dict[str, float]:
    """
    Compute in-distribution and OOD perplexity from per-position losses.

    in-dist  = positions 0 .. t_train-1  (trained context length)
    OOD      = positions t_train .. end  (beyond training length)
    """
    in_dist = pos_losses[:t_train]
    ood     = pos_losses[t_train:]

    def _ppl(arr):
        if len(arr) == 0:
            return float("nan")
        return float(np.exp(arr.mean()))

    return {
        "in_dist_ppl":  _ppl(in_dist),
        "ood_ppl":      _ppl(ood),
        "ratio":        _ppl(ood) / _ppl(in_dist) if len(ood) > 0 else float("nan"),
        "in_dist_loss": float(in_dist.mean()),
        "ood_loss":     float(ood.mean()) if len(ood) > 0 else float("nan"),
        "seq_len":      len(pos_losses),
        "t_train":      t_train,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    results: Dict[str, Dict[int, np.ndarray]],   # {model_label: {seq_len: pos_losses}}
    t_train: int,
    output_path: str,
):
    """
    Two-panel figure:
      Left:  per-position cross-entropy for each (model, eval_length) combination.
             Smoothed with a short rolling window so the trend is readable.
             Vertical dashed line at t_train.
      Right: OOD ppl vs eval length, one line per model.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available — skipping plot (JSON results still saved)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_pos, ax_ood = axes

    model_labels = list(results.keys())
    colors = cm.tab10(np.linspace(0, 0.9, max(len(model_labels), 1)))

    # Rolling mean window for position plot
    window = 8

    def _smooth(x, w):
        if len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    for model_idx, (label, length_dict) in enumerate(results.items()):
        col = colors[model_idx]
        lengths = sorted(length_dict.keys())

        # Per-position plot — show each eval length as a progressively lighter shade
        for li, seq_len in enumerate(lengths):
            pos_losses = length_dict[seq_len]
            alpha = 0.4 + 0.6 * (li / max(len(lengths) - 1, 1))
            smoothed = _smooth(pos_losses, window)
            positions = np.arange(len(smoothed)) + window // 2
            ax_pos.plot(
                positions, smoothed,
                color=col, alpha=alpha,
                label=f"{label} L={seq_len}" if len(model_labels) > 1 else f"L={seq_len}",
                linewidth=1.2,
            )

        # OOD ppl plot
        ood_ppls  = []
        eval_lens = []
        for seq_len in lengths:
            s = summarise(length_dict[seq_len], t_train)
            if not np.isnan(s["ood_ppl"]):
                eval_lens.append(seq_len)
                ood_ppls.append(s["ood_ppl"])
        if eval_lens:
            ax_ood.plot(eval_lens, ood_ppls, "o-", color=col, label=label, linewidth=2)

    # Annotations on position plot
    ax_pos.axvline(t_train, color="black", linestyle="--", linewidth=1.5,
                   label=f"T_train = {t_train}")
    ax_pos.set_xlabel("Position in sequence")
    ax_pos.set_ylabel("Cross-entropy loss")
    ax_pos.set_title("Per-position loss (smoothed)")
    ax_pos.legend(fontsize=7, ncol=2)
    ax_pos.grid(True, alpha=0.3)

    # OOD ppl plot annotations
    ax_ood.axvline(t_train, color="black", linestyle="--", linewidth=1.5,
                   label=f"T_train = {t_train}")
    ax_ood.set_xlabel("Evaluation sequence length")
    ax_ood.set_ylabel("OOD perplexity (positions > T_train)")
    ax_ood.set_title("OOD perplexity vs evaluation length")
    ax_ood.legend(fontsize=9)
    ax_ood.grid(True, alpha=0.3)

    fig.suptitle(
        f"Length generalisation  (T_train = {t_train})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    plot_file = output_path + "_length_gen.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {plot_file}")
    plt.close(fig)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device,
):
    """Load a CortexLM checkpoint and its associated tokenizer."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Load tokenizer from the checkpoint directory
    ckpt_dir = os.path.dirname(checkpoint_path)
    tok_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        # Fallback: rebuild from config
        from cortexlm.data import build_tokenizer
        print(f"  tokenizer.pkl not found in {ckpt_dir}, rebuilding...")
        tokenizer = build_tokenizer(config)
    else:
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)

    vocab_size = tokenizer.vocab_size
    config["data"]["vocab_size"] = vocab_size

    model = CortexLM(config, vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    t_train = config["data"]["seq_len"]
    print(f"  Loaded: {os.path.basename(checkpoint_path)} | "
          f"params={model.count_parameters():,} | T_train={t_train}")

    return model, tokenizer, config, t_train


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Length generalisation experiment for CortexLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model specification — either a single checkpoint or multiple labelled ones
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", metavar="PATH",
        help="Single checkpoint to evaluate",
    )
    group.add_argument(
        "--models", nargs="+", metavar="LABEL:PATH",
        help='Multiple checkpoints as "label:path" pairs for comparison',
    )

    parser.add_argument(
        "--lengths", nargs="+", type=int,
        default=[128, 192, 256, 384, 512, 768, 1024],
        help="Evaluation sequence lengths to sweep (default: 128 192 256 384 512 768 1024)",
    )
    parser.add_argument(
        "--n-sequences", type=int, default=200,
        help="Number of validation sequences to average over (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--output", default="results/length_gen",
        help="Output path prefix for plot and JSON (default: results/length_gen)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if a GPU is available",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plot (always saves JSON)",
    )

    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model list
    if args.checkpoint:
        model_specs = [("model", args.checkpoint)]
    else:
        model_specs = []
        for spec in args.models:
            if ":" not in spec:
                parser.error(f'--models entries must be "label:path", got: {spec!r}')
            label, path = spec.split(":", 1)
            model_specs.append((label, path))

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Sort lengths so the sweep is monotonically increasing
    eval_lengths = sorted(set(args.lengths))
    max_length   = max(eval_lengths)

    # ── Run experiment ────────────────────────────────────────────────────────
    # {label: {seq_len: pos_losses[seq_len]}}
    all_results: Dict[str, Dict[int, np.ndarray]] = {}
    summary_rows: List[dict] = []

    for label, ckpt_path in model_specs:
        print(f"\n{'='*60}")
        print(f"Model: {label}  ({ckpt_path})")
        print(f"{'='*60}")

        model, tokenizer, config, t_train = load_model_and_tokenizer(ckpt_path, device)
        all_results[label] = {}

        # Load sequences once at the maximum required length
        print(f"\nLoading {args.n_sequences} sequences of length {max_length} "
              f"from validation set...")
        long_seqs = _load_long_sequences(
            config, tokenizer,
            min_length=max_length,
            n_sequences=args.n_sequences,
            device=device,
        )   # [n_seq, max_length+1]
        print(f"  Sequence tensor: {tuple(long_seqs.shape)}")

        for seq_len in tqdm(eval_lengths, desc=f"  Eval lengths ({label})"):
            # Slice to the required length (model sees tokens 0..seq_len-1,
            # predicts tokens 1..seq_len)
            seqs = long_seqs[:, :seq_len + 1]

            pos_losses = evaluate_position_losses(model, seqs, batch_size=args.batch_size)
            all_results[label][seq_len] = pos_losses

            row = summarise(pos_losses, t_train)
            row["model"] = label
            summary_rows.append(row)

            in_ppl = row["in_dist_ppl"]
            ood_ppl = row["ood_ppl"]
            ratio   = row["ratio"]
            ood_str = f"{ood_ppl:.2f} (×{ratio:.2f})" if not np.isnan(ood_ppl) else "n/a"
            print(
                f"    L={seq_len:5d} | in-dist ppl={in_ppl:.2f} | OOD ppl={ood_str}"
            )

        del model   # free VRAM before next model

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'L':>6} {'T_train':>8} {'in-dist ppl':>12} "
          f"{'OOD ppl':>10} {'ratio':>7}")
    print("-" * 70)
    for row in summary_rows:
        ood_str   = f"{row['ood_ppl']:.2f}"  if not np.isnan(row['ood_ppl'])  else "n/a"
        ratio_str = f"{row['ratio']:.3f}"    if not np.isnan(row['ratio'])    else "n/a"
        print(
            f"{row['model']:<20} {row['seq_len']:>6} {row['t_train']:>8} "
            f"{row['in_dist_ppl']:>12.2f} {ood_str:>10} {ratio_str:>7}"
        )
    print(f"{'='*70}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = args.output + "_results.json"
    save_data = {
        "summary": summary_rows,
        "per_position": {
            label: {
                str(seq_len): pos_losses.tolist()
                for seq_len, pos_losses in length_dict.items()
            }
            for label, length_dict in all_results.items()
        },
        "config": {
            "eval_lengths": eval_lengths,
            "n_sequences": args.n_sequences,
            "batch_size": args.batch_size,
        },
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        # Use the t_train from the first model (they should all share it for a
        # fair comparison; if they differ, the plot still works)
        t_train_plot = summary_rows[0]["t_train"] if summary_rows else 128
        plot_results(all_results, t_train_plot, args.output)


if __name__ == "__main__":
    main()
