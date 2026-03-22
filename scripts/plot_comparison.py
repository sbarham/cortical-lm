"""Side-by-side comparison plot: CortexLM variants and/or baselines.

Reads metrics.jsonl from one or more CortexLM run directories and/or a
baseline_results.json from run_baselines.py, then plots val perplexity
(and optionally val loss) vs. tokens seen on a shared axis.

Usage:
    # CortexLM variants only:
    python scripts/plot_comparison.py \\
        --cortex "simple_ei:checkpoints/minimal" "layered:checkpoints/layered"

    # CortexLM + baselines:
    python scripts/plot_comparison.py \\
        --cortex "simple_ei:checkpoints/minimal" "layered:checkpoints/layered" \\
        --baselines baseline_results.json

    # Single CortexLM vs baselines (the typical Phase 3 comparison):
    python scripts/plot_comparison.py \\
        --cortex "CortexLM (standard):checkpoints/standard" \\
        --baselines baseline_results.json \\
        --output comparison_phase3.png
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


# ── Colour palette ─────────────────────────────────────────────────────────
# CortexLM variants get blues/teals; baselines get warm colours.
_CORTEX_COLORS = ["#1f77b4", "#17becf", "#2ca02c", "#9467bd", "#8c564b"]
_BASELINE_COLORS = {
    "transformer":    "#d62728",
    "lstm_attention": "#ff7f0e",
    "lstm":           "#e377c2",
    "rnn_attention":  "#bcbd22",
    "rnn":            "#7f7f7f",
}
_BASELINE_STYLE = {"linestyle": "--", "linewidth": 1.4, "marker": "o", "markersize": 3}
_CORTEX_STYLE   = {"linewidth": 2.0, "marker": "s", "markersize": 4}


# ── Loaders ────────────────────────────────────────────────────────────────

def load_cortex_run(label: str, run_dir: str):
    """
    Load a CortexLM run from metrics.jsonl.

    Returns:
        dict with keys: label, val_tokens, val_ppl, val_loss, val_bpc,
                        train_tokens, train_ppl, params (if inferable)
    """
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {label!r}")
        return None

    val_tokens, val_ppl, val_loss, val_bpc = [], [], [], []
    train_tokens, train_ppl = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("tokens", None)
            if "val/loss" in r:
                val_tokens.append(t)
                val_ppl.append(r.get("val/perplexity", np.exp(r["val/loss"])))
                val_loss.append(r["val/loss"])
                val_bpc.append(r.get("val/bpc", float("nan")))
            elif "train/loss" in r:
                train_tokens.append(t)
                train_ppl.append(r.get("train/perplexity", np.exp(r["train/loss"])))

    if not val_ppl:
        print(f"  WARNING: no val records in {path} — skipping {label!r}")
        return None

    # Infer param count from any checkpoint in run_dir
    params = _infer_params(run_dir)

    return dict(
        label=label,
        val_tokens=np.array([t if t is not None else i for i, t in enumerate(val_tokens)]),
        val_ppl=np.array(val_ppl),
        val_loss=np.array(val_loss),
        val_bpc=np.array(val_bpc),
        train_tokens=np.array([t if t is not None else i for i, t in enumerate(train_tokens)]),
        train_ppl=np.array(train_ppl),
        params=params,
        tokens_are_real=any(t is not None for t in val_tokens),
    )


def _infer_params(run_dir: str):
    """Try to read param count from a checkpoint in run_dir."""
    import torch
    for fname in sorted(os.listdir(run_dir)):
        if fname.endswith(".pt"):
            try:
                ckpt = torch.load(os.path.join(run_dir, fname),
                                  map_location="cpu", weights_only=False)
                cfg = ckpt.get("config")
                if cfg:
                    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from cortexlm.model import CortexLM
                    import pickle
                    tok_pkl = os.path.join(run_dir, "tokenizer.pkl")
                    if os.path.exists(tok_pkl):
                        with open(tok_pkl, "rb") as f:
                            tok = pickle.load(f)
                        vs = tok.vocab_size
                    else:
                        vs = cfg["data"].get("vocab_size", 4096)
                    import torch as _torch
                    _torch.manual_seed(cfg["training"].get("seed", 42))
                    m = CortexLM(cfg, vs)
                    return m.count_parameters()
            except Exception:
                pass
    return None


def load_baselines(path: str):
    """
    Load baseline_results.json written by run_baselines.py.

    Returns list of dicts: {label, tokens, ppl, params, hidden_size}
    """
    with open(path) as f:
        data = json.load(f)

    results = []
    for name, info in data.get("baselines", {}).items():
        log = info.get("log", [])
        if not log:
            continue
        tokens = np.array([r["tokens"] for r in log])
        ppl    = np.array([r["perplexity"] for r in log])
        results.append(dict(
            label=name.replace("_", " "),
            name=name,
            tokens=tokens,
            ppl=ppl,
            params=info.get("params"),
            hidden_size=info.get("hidden_size"),
        ))
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def _millions(x, pos=None):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return str(int(x))


def plot_comparison(cortex_runs, baseline_runs, args):
    n_panels = 1 + (1 if args.show_loss else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax_ppl  = axes[0]
    ax_loss = axes[1] if n_panels > 1 else None

    legend_entries = []

    # ── CortexLM variants ─────────────────────────────────────────────────
    for i, run in enumerate(cortex_runs):
        color = _CORTEX_COLORS[i % len(_CORTEX_COLORS)]
        x = run["val_tokens"]
        label = run["label"]
        if run.get("params"):
            label += f"  ({run['params']/1e3:.0f}K params)"
        if not run["tokens_are_real"]:
            label += "  [x=step, not tokens]"

        kw = dict(**_CORTEX_STYLE, color=color, label=label)
        ax_ppl.plot(x, run["val_ppl"], **kw)
        if ax_loss:
            ax_loss.plot(x, run["val_loss"], **kw)
        legend_entries.append(label)

    # ── Baselines ─────────────────────────────────────────────────────────
    for bl in baseline_runs:
        name = bl.get("name", bl["label"])
        color = _BASELINE_COLORS.get(name, "#888888")
        label = bl["label"]
        if bl.get("params"):
            label += f"  ({bl['params']/1e3:.0f}K params)"

        kw = dict(**_BASELINE_STYLE, color=color, label=label)
        ax_ppl.plot(bl["tokens"], bl["ppl"], **kw)
        if ax_loss:
            # baselines don't log loss directly — skip
            pass
        legend_entries.append(label)

    # ── Threshold lines ───────────────────────────────────────────────────
    for ppl_thresh, style, lbl in [
        (500, dict(color="gray", linestyle=":", linewidth=0.8), "ppl=500"),
        (100, dict(color="gray", linestyle="--", linewidth=0.8), "ppl=100"),
    ]:
        ax_ppl.axhline(y=ppl_thresh, **style, label=lbl, alpha=0.5)

    # ── Formatting ────────────────────────────────────────────────────────
    for ax, ylabel, title in [
        (ax_ppl, "Val perplexity (log)", "Validation perplexity vs. tokens"),
        (ax_loss, "Val loss", "Validation loss vs. tokens"),
    ]:
        if ax is None:
            continue
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Tokens seen")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_millions))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    # Summary table below the plot
    rows = []
    for run in cortex_runs:
        if run["val_ppl"].size:
            rows.append(f"{run['label']}: final val ppl = {run['val_ppl'][-1]:.1f}")
    for bl in baseline_runs:
        rows.append(f"{bl['label']}: final val ppl = {bl['ppl'][-1]:.1f}")
    if rows:
        fig.text(0.01, 0.01, "   ".join(rows), fontsize=7, color="gray", va="bottom")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = args.output
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print text table
    print("\nFinal val perplexity comparison:")
    all_rows = (
        [(r["label"], r["val_ppl"][-1] if r["val_ppl"].size else float("nan"),
          r.get("params")) for r in cortex_runs] +
        [(b["label"], b["ppl"][-1], b.get("params")) for b in baseline_runs]
    )
    all_rows.sort(key=lambda x: x[1])
    for label, ppl, params in all_rows:
        param_str = f"  ({params/1e3:.0f}K params)" if params else ""
        print(f"  {ppl:7.1f}  {label}{param_str}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare CortexLM variants and baselines")
    parser.add_argument("--cortex", nargs="*", default=[],
                        metavar="LABEL:RUN_DIR",
                        help='CortexLM runs to include, e.g. "simple_ei:checkpoints/minimal"')
    parser.add_argument("--baselines", default=None, metavar="PATH",
                        help="Path to baseline_results.json from run_baselines.py")
    parser.add_argument("--output", default="comparison.png",
                        help="Output PNG path (default: comparison.png)")
    parser.add_argument("--show-loss", action="store_true",
                        help="Add a second panel showing val loss")
    args = parser.parse_args()

    if not args.cortex and not args.baselines:
        parser.error("Provide at least one --cortex run or --baselines file.")

    # Parse "label:path" entries
    cortex_runs = []
    for entry in (args.cortex or []):
        if ":" not in entry:
            parser.error(f"--cortex entries must be in LABEL:RUN_DIR format, got: {entry!r}")
        label, run_dir = entry.split(":", 1)
        print(f"Loading CortexLM run: {label!r} from {run_dir}")
        run = load_cortex_run(label, run_dir)
        if run:
            cortex_runs.append(run)

    baseline_runs = []
    if args.baselines:
        if not os.path.exists(args.baselines):
            print(f"Baselines file not found: {args.baselines}")
            sys.exit(1)
        print(f"Loading baselines from {args.baselines}")
        baseline_runs = load_baselines(args.baselines)
        print(f"  Found baselines: {[b['label'] for b in baseline_runs]}")

    if not cortex_runs and not baseline_runs:
        print("No data loaded. Nothing to plot.")
        sys.exit(1)

    plot_comparison(cortex_runs, baseline_runs, args)


if __name__ == "__main__":
    main()
