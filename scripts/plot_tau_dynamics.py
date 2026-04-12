"""
plot_tau_dynamics.py — Plot mean ± std of tau_m over training from .npz snapshot files.

Usage:
    python scripts/plot_tau_dynamics.py tau_snapshots/ [--out-dir /path/to/out]
                                        [--title "My Title"]

Expects tau_*.npz files in snap_dir, each containing arrays:
    l4e, l23e, l5e, l6e  (tau_m values for that snapshot)
Filenames encode tokens_seen as a 12-digit zero-padded integer after the first underscore,
e.g.  tau_000500000000.npz

Output:
    tau_dynamics.pdf and tau_dynamics.png in --out-dir.
"""

import argparse
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
    "figure.figsize": (7, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Layer configuration: (npz_key, display_label, linestyle, ribbon_gray)
LAYERS = [
    ("l4e",  "L4",    "-",    "0.75"),
    ("l23e", "L2/3",  "--",   "0.60"),
    ("l5e",  "L5",    ":",    "0.45"),
    ("l6e",  "L6",    "-.",   "0.30"),
]


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

def _parse_tokens_from_stem(stem):
    """
    Parse tokens from a filename stem like 'tau_000500000000'.
    Returns int or None.
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def load_snapshots(snap_dir):
    """
    Load all tau_*.npz files from snap_dir, sorted by encoded token count.
    Returns:
        tokens_list: list of int (tokens_seen per snapshot)
        data: dict layer_key -> list of 1-D numpy arrays (one per snapshot; may be None)
    """
    snap_path = Path(snap_dir)
    if not snap_path.exists():
        print(f"ERROR: snap_dir does not exist: {snap_path}", file=sys.stderr)
        sys.exit(1)

    files = sorted(snap_path.glob("tau_*.npz"), key=lambda p: p.stem)
    if not files:
        print(f"ERROR: No tau_*.npz files found in {snap_path}", file=sys.stderr)
        sys.exit(1)

    tokens_list = []
    raw = {key: [] for key, *_ in LAYERS}

    for f in files:
        tokens = _parse_tokens_from_stem(f.stem)
        if tokens is None:
            warnings.warn(f"Could not parse token count from {f.name}; skipping.")
            continue

        try:
            npz = np.load(str(f))
        except Exception as e:
            warnings.warn(f"Error loading {f}: {e}; skipping.")
            continue

        tokens_list.append(tokens)
        for key, *_ in LAYERS:
            if key in npz:
                raw[key].append(npz[key].flatten())
            else:
                warnings.warn(f"Key '{key}' missing in {f.name}.")
                raw[key].append(None)

    return tokens_list, raw


# ---------------------------------------------------------------------------
# Compute statistics
# ---------------------------------------------------------------------------

def compute_stats(tokens_list, raw):
    """
    For each layer, compute mean and std at each snapshot step.
    Returns dict: layer_key -> (means array, stds array), with NaN for missing steps.
    """
    stats = {}
    for key, *_ in LAYERS:
        means = []
        stds = []
        for arr in raw[key]:
            if arr is None or len(arr) == 0:
                means.append(float("nan"))
                stds.append(float("nan"))
            else:
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr)))
        stats[key] = (np.array(means), np.array(stds))
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(tokens_list, stats, title=None):
    tokens_m = np.array(tokens_list) / 1e6

    fig, ax = plt.subplots(figsize=(7, 4))

    for key, label, ls, ribbon_gray in LAYERS:
        means, stds = stats[key]
        valid = np.isfinite(means)
        if not np.any(valid):
            warnings.warn(f"No valid data for layer {key}; skipping.")
            continue

        tx = tokens_m[valid]
        my = means[valid]
        sy = stds[valid]

        line_color = "black"
        # Offset slightly for visual separation using shading intensity
        fill_color = ribbon_gray

        ax.plot(
            tx, my,
            linestyle=ls,
            color=line_color,
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=label,
        )
        ax.fill_between(
            tx,
            my - sy,
            my + sy,
            alpha=0.18,
            color=fill_color,
            linewidth=0,
        )

    ax.set_xlabel("Tokens Seen (millions)")
    ax.set_ylabel(r"$\tau_m$ (ms)")
    ax.set_title(title or r"$\tau_m$ Dynamics Over Training")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot tau_m dynamics over training from .npz snapshot files."
    )
    parser.add_argument("snap_dir", help="Path to tau_snapshots/ directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same as snap_dir)")
    parser.add_argument("--title", default=None, help="Optional figure title")
    args = parser.parse_args()

    snap_path = Path(args.snap_dir)
    out_dir = Path(args.out_dir) if args.out_dir else snap_path
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshots from: {snap_path}")
    tokens_list, raw = load_snapshots(snap_path)
    print(f"Loaded {len(tokens_list)} snapshot(s).")

    stats = compute_stats(tokens_list, raw)
    fig = make_figure(tokens_list, stats, title=args.title)

    for ext in ("pdf", "png"):
        out_path = out_dir / f"tau_dynamics.{ext}"
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
