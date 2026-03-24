"""Visualize a training run from its metrics.jsonl file.

Usage:
    python scripts/plot_run.py --run-dir checkpoints/
    python scripts/plot_run.py --run-dir checkpoints/ --output my_run.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Helpers ────────────────────────────────────────────────────────────────

def load_metrics(run_dir: str):
    """Read metrics.jsonl → two lists of dicts: train records, val records."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"No metrics.jsonl found in {run_dir}")
        sys.exit(1)

    train_records = []
    val_records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "val/loss" in rec:
                val_records.append(rec)
            elif "train/loss" in rec:
                train_records.append(rec)
    return train_records, val_records


def ema_smooth(values, alpha=0.1):
    """Exponential moving average smoothing."""
    if not values:
        return np.array([])
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot a CortexLM training run")
    parser.add_argument("--run-dir", required=True, help="Checkpoint directory containing metrics.jsonl")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <run-dir>/training_curves.png)")
    parser.add_argument("--smooth", type=float, default=0.05,
                        help="EMA smoothing factor for train curves (0=none, 1=max, default=0.05)")
    args = parser.parse_args()

    train_recs, val_recs = load_metrics(args.run_dir)
    print(f"Loaded {len(train_recs)} train records, {len(val_recs)} val records")

    if not train_recs:
        print("No training records found. Has training started?")
        sys.exit(1)

    # Extract series
    tr_steps = np.array([r["step"] for r in train_recs])
    tr_loss  = np.array([r["train/loss"] for r in train_recs])
    tr_ppl   = np.array([r.get("train/perplexity", np.exp(r["train/loss"])) for r in train_recs])
    tr_lr    = np.array([r.get("lr", float("nan")) for r in train_recs])

    val_steps = np.array([r["step"] for r in val_recs]) if val_recs else np.array([])
    val_loss  = np.array([r["val/loss"] for r in val_recs]) if val_recs else np.array([])
    val_ppl   = np.array([r.get("val/perplexity", np.exp(r["val/loss"])) for r in val_recs]) if val_recs else np.array([])
    val_bpb   = np.array([r.get("val/bpb", r.get("val/bpc", float("nan"))) for r in val_recs]) if val_recs else np.array([])

    # Smoothed train
    tr_loss_sm = ema_smooth(tr_loss.tolist(), alpha=args.smooth)
    tr_ppl_sm  = ema_smooth(tr_ppl.tolist(), alpha=args.smooth)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Training run: {os.path.abspath(args.run_dir)}", fontsize=10, color="gray")

    TRAIN_RAW   = dict(color="#aac4e8", alpha=0.4, linewidth=0.8)
    TRAIN_SMOOTH= dict(color="#1f6fb5", linewidth=1.8, label="train (smoothed)")
    VAL_STYLE   = dict(color="#e05c2a", linewidth=1.5, marker="o", markersize=4, label="val")

    # ── Panel 1: Loss ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(tr_steps, tr_loss, **TRAIN_RAW)
    ax.plot(tr_steps, tr_loss_sm, **TRAIN_SMOOTH)
    if val_recs:
        ax.plot(val_steps, val_loss, **VAL_STYLE)
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xlabel("Step")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate last val loss
    if val_recs:
        ax.annotate(f"{val_loss[-1]:.3f}",
                    xy=(val_steps[-1], val_loss[-1]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=7, color="#e05c2a")

    # ── Panel 2: Perplexity (log scale) ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(tr_steps, tr_ppl, **TRAIN_RAW)
    ax.plot(tr_steps, tr_ppl_sm, **TRAIN_SMOOTH)
    if val_recs:
        ax.plot(val_steps, val_ppl, **VAL_STYLE)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylabel("Perplexity")
    ax.set_xlabel("Step")
    ax.set_title("Perplexity (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Draw intuitive threshold lines
    random_ppl = None
    # Try to infer vocab size from a val record
    ax.axhline(y=500, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="ppl=500")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="ppl=100")
    ax.legend(fontsize=7)

    # ── Panel 3: Val BPB ──────────────────────────────────────────────────
    ax = axes[1, 0]
    if val_recs and not np.all(np.isnan(val_bpb)):
        ax.plot(val_steps, val_bpb, color="#2ca02c", linewidth=1.5, marker="s",
                markersize=4, label="val bpb")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="bpb=1 (near-optimal)")
        ax.set_ylabel("Bits per byte")
        ax.set_title("Val BPB")
        ax.legend(fontsize=8)
        ax.annotate(f"{val_bpb[-1]:.3f}",
                    xy=(val_steps[-1], val_bpb[-1]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=7, color="#2ca02c")
    else:
        ax.set_title("Val BPB (no data)")
        ax.text(0.5, 0.5, "No BPB data yet", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Learning rate ─────────────────────────────────────────────
    ax = axes[1, 1]
    if not np.all(np.isnan(tr_lr)):
        ax.plot(tr_steps, tr_lr, color="#9467bd", linewidth=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    else:
        ax.text(0.5, 0.5, "No LR data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Step")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3, which="both")

    # ── Summary box ───────────────────────────────────────────────────────
    if train_recs:
        current_step = int(tr_steps[-1])
        current_loss = float(tr_loss[-1])
        summary_lines = [
            f"Steps completed: {current_step:,}",
            f"Last train loss: {current_loss:.4f}",
        ]
        if val_recs:
            summary_lines += [
                f"Last val loss:   {val_loss[-1]:.4f}",
                f"Last val ppl:    {val_ppl[-1]:.1f}",
            ]
            if not np.all(np.isnan(val_bpc)):
                summary_lines.append(f"Last val BPC:    {val_bpc[-1]:.4f}")
        fig.text(0.01, 0.01, "  ".join(summary_lines),
                 fontsize=7, color="gray", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = args.output or os.path.join(args.run_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Also print a quick text summary
    if train_recs:
        print(f"\nRun summary")
        print(f"  Steps:      {int(tr_steps[-1]):,}")
        print(f"  Train loss: {float(tr_loss[-1]):.4f}  (ppl {float(tr_ppl[-1]):.1f})")
        if val_recs:
            print(f"  Val loss:   {float(val_loss[-1]):.4f}  (ppl {float(val_ppl[-1]):.1f})", end="")
            if not np.all(np.isnan(val_bpb)):
                print(f"  bpb {float(val_bpb[-1]):.4f}", end="")
            print()


if __name__ == "__main__":
    main()
