"""
plot_tau_distributions.py — Four-panel KDE figure of tau_eff distributions
from a tau snapshot .npz file (produced by the tau_eff recording runs).

Each .npz contains per-neuron ACF-estimated effective timescales for L4, L2/3,
L5, L6 excitatory populations at a specific point in training.

Usage:
    # Single snapshot (e.g. final):
    python scripts/plot_tau_distributions.py tau_000100000000.npz

    # Snapshot directory — uses the last snapshot:
    python scripts/plot_tau_distributions.py checkpoints/.../tau_snapshots/

    # Overlay two snapshots (e.g. DAWN vs BPTT at end of training):
    python scripts/plot_tau_distributions.py run_a/tau_snapshots/ \\
        --compare run_b/tau_snapshots/ --label-main DAWN --label-compare BPTT

Output:
    tau_distributions.pdf and tau_distributions.png in --out-dir.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Publication-quality rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LAYER_KEYS   = ["l4e", "l23e", "l5e", "l6e"]
LAYER_LABELS = {
    "l4e":  "L4 excitatory",
    "l23e": "L2/3 excitatory",
    "l5e":  "L5 excitatory",
    "l6e":  "L6 excitatory",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_npz(path: Path) -> Path:
    """If path is a directory, return the last tau_*.npz inside it."""
    if path.is_dir():
        snaps = sorted(path.glob("tau_*.npz"))
        if not snaps:
            raise FileNotFoundError(f"No tau_*.npz files found in {path}")
        return snaps[-1]
    return path


def _load_snapshot(path: Path) -> dict:
    """Load a tau snapshot .npz and return {layer_key: 1-D float32 array}."""
    data = np.load(str(path))
    return {k: data[k].flatten() for k in LAYER_KEYS if k in data}


def _tokens_from_path(path: Path) -> int:
    """Parse tokens_seen from filename, e.g. tau_000100000000.npz -> 100_000_000."""
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# KDE helper
# ---------------------------------------------------------------------------

def _kde(data, n_points=512):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.array([]), np.array([])
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), n_points)
        return xs, kde(xs)
    except ImportError:
        warnings.warn("scipy not available; using histogram for KDE approximation.")
        counts, edges = np.histogram(data, bins=50, density=True)
        return 0.5 * (edges[:-1] + edges[1:]), counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_layer(ax, key, data_main, label_main, data_compare=None, label_compare=None):
    n = len(data_main) if data_main is not None else 0

    if data_main is not None and len(data_main) > 0:
        xs, dens = _kde(data_main)
        if len(xs) > 0:
            ax.plot(xs, dens, color="black", linestyle="-", linewidth=1.2,
                    label=label_main)
            mean_val = float(np.mean(data_main))
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=0.8,
                       label=f"mean {mean_val:.1f} ms")

    if data_compare is not None and len(data_compare) > 0:
        xs2, dens2 = _kde(data_compare)
        if len(xs2) > 0:
            ax.plot(xs2, dens2, color="0.50", linestyle="--", linewidth=1.0,
                    label=label_compare)
            mean2 = float(np.mean(data_compare))
            ax.axvline(mean2, color="0.50", linestyle=":", linewidth=0.8,
                       label=f"mean {mean2:.1f} ms")

    ax.set_title(f"{LAYER_LABELS[key]} (N={n})")
    ax.set_xlabel(r"$\tau_\mathrm{eff}$ (ms)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)


def make_figure(snap_main, snap_compare=None,
                label_main="Model", label_compare="Compare",
                title=None):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, key in enumerate(LAYER_KEYS):
        ax = axes.flat[i]
        dm = snap_main.get(key)
        dc = snap_compare.get(key) if snap_compare else None
        _plot_layer(ax, key, dm, label_main, dc, label_compare)
    suptitle = title or r"$\tau_\mathrm{eff}$ Distributions"
    fig.suptitle(suptitle, fontsize=10, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot tau_eff distributions from tau snapshot .npz files."
    )
    parser.add_argument(
        "snapshot",
        help="Path to a tau_*.npz file, or a tau_snapshots/ directory "
             "(uses the last snapshot in the directory).",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Optional second .npz file or directory to overlay.",
    )
    parser.add_argument("--label-main",    default=None,
                        help="Legend label for main snapshot (default: filename)")
    parser.add_argument("--label-compare", default=None,
                        help="Legend label for compare snapshot (default: filename)")
    parser.add_argument("--out-dir",  default=None,
                        help="Output directory (default: same as snapshot)")
    parser.add_argument("--title",    default=None, help="Optional figure title")
    args = parser.parse_args()

    snap_path = _resolve_npz(Path(args.snapshot))
    if not snap_path.exists():
        print(f"ERROR: snapshot not found: {snap_path}", file=sys.stderr)
        sys.exit(1)

    tokens = _tokens_from_path(snap_path)
    label_main    = args.label_main    or snap_path.stem
    label_compare = args.label_compare or ""

    out_dir = Path(args.out_dir) if args.out_dir else snap_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshot: {snap_path}  ({tokens/1e6:.0f}M tokens)")
    snap_main = _load_snapshot(snap_path)
    for k, v in snap_main.items():
        print(f"  {LAYER_LABELS[k]}: N={len(v)}  mean={v.mean():.1f}  "
              f"std={v.std():.1f}  p25={np.percentile(v,25):.1f}  "
              f"p75={np.percentile(v,75):.1f} ms")

    snap_compare = None
    if args.compare:
        compare_path = _resolve_npz(Path(args.compare))
        if not compare_path.exists():
            warnings.warn(f"Compare snapshot not found: {compare_path}; skipping.")
        else:
            print(f"Loading compare: {compare_path}")
            snap_compare = _load_snapshot(compare_path)
            label_compare = args.label_compare or compare_path.stem

    title = args.title or (
        rf"$\tau_\mathrm{{eff}}$ Distributions — {tokens/1e6:.0f}M tokens"
    )
    fig = make_figure(snap_main, snap_compare,
                      label_main=label_main, label_compare=label_compare,
                      title=title)

    for ext in ("pdf", "png"):
        out_path = out_dir / f"tau_distributions.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
