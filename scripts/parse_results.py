"""
parse_results.py — Read W&B offline run directories, extract training curves,
and generate paper numbers and figures.

Usage:
    python scripts/parse_results.py [--wandb-dir wandb/] [--group GROUP]
                                    [--runs run1,run2] [--out-dir results/]

Outputs (all written to --out-dir):
    results_table.csv         — per-run metrics
    fig_learning_curves.pdf/png
    fig_efficiency_profile.pdf/png
    fig_aulc.pdf/png
"""

import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not installed; config.yaml parsing unavailable.")

# ---------------------------------------------------------------------------
# Publication-quality rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MARKERS = ["o", "s", "^", "D", "v", "*", "p", "h", "X", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    """Load a .jsonl file into a list of dicts; return [] on failure."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as e:
        warnings.warn(f"Cannot read {path}: {e}")
    return records


def _load_config(run_dir):
    """Return (name, group) from config.yaml or wandb-metadata.json."""
    config_yaml = run_dir / "files" / "config.yaml"
    meta_json = run_dir / "files" / "wandb-metadata.json"

    name, group = None, None

    if HAS_YAML and config_yaml.exists():
        try:
            with open(config_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                name_entry = cfg.get("name", {})
                group_entry = cfg.get("group", {})
                if isinstance(name_entry, dict):
                    name = name_entry.get("value")
                elif isinstance(name_entry, str):
                    name = name_entry
                if isinstance(group_entry, dict):
                    group = group_entry.get("value")
                elif isinstance(group_entry, str):
                    group = group_entry
        except Exception as e:
            warnings.warn(f"Error reading {config_yaml}: {e}")

    if (name is None or group is None) and meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if name is None:
                name = meta.get("name")
            if group is None:
                group = meta.get("group")
        except Exception as e:
            warnings.warn(f"Error reading {meta_json}: {e}")

    # Fall back to directory name
    if name is None:
        name = run_dir.name
    return name, group


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _tokens_to_threshold(tokens, ppls, thresholds):
    """For each threshold, return the first tokens value where ppl < threshold."""
    result = {}
    for thr in thresholds:
        key = f"tokens_to_{thr}ppl"
        result[key] = None
        for t, p in zip(tokens, ppls):
            if p is not None and p < thr:
                result[key] = t
                break
    return result


def _aulc(tokens, ppls):
    """Compute AULC (token-weighted mean ppl) for the full curve and early/late 25%."""
    # Filter out None values
    valid = [(t, p) for t, p in zip(tokens, ppls) if p is not None and t is not None]
    if len(valid) < 2:
        return None, None, None

    ts = [v[0] for v in valid]
    ps = [v[1] for v in valid]

    total = ts[-1] - ts[0]
    if total <= 0:
        return None, None, None

    t_25 = ts[0] + 0.25 * total
    t_75 = ts[0] + 0.75 * total

    aulc_full = 0.0
    aulc_early_num, aulc_early_den = 0.0, 0.0
    aulc_late_num, aulc_late_den = 0.0, 0.0

    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        aulc_full += ps[i] * dt

        if ts[i] <= t_25:
            aulc_early_num += ps[i] * dt
            aulc_early_den += dt
        if ts[i] >= t_75:
            aulc_late_num += ps[i] * dt
            aulc_late_den += dt

    aulc_full /= total
    aulc_early = aulc_early_num / aulc_early_den if aulc_early_den > 0 else None
    aulc_late = aulc_late_num / aulc_late_den if aulc_late_den > 0 else None

    return aulc_full, aulc_early, aulc_late


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(wandb_dir, group_filter=None, run_filter=None):
    """
    Scan wandb_dir for offline-run-* subdirectories.
    Returns list of dicts with keys: run_name, group, tokens, ppls.
    """
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        warnings.warn(f"wandb dir does not exist: {wandb_path}")
        return []

    run_dirs = sorted(wandb_path.glob("offline-run-*"))
    if not run_dirs:
        warnings.warn(f"No offline-run-* directories found in {wandb_path}")
        return []

    runs = []
    for run_dir in run_dirs:
        name, group = _load_config(run_dir)

        if run_filter and name not in run_filter:
            continue
        if group_filter and group != group_filter:
            continue

        history_path = run_dir / "files" / "wandb-history.jsonl"
        records = _load_jsonl(history_path)
        if not records:
            warnings.warn(f"No history records in {history_path}; skipping {name}")
            continue

        tokens = []
        ppls = []
        for rec in records:
            t = rec.get("tokens")
            p = rec.get("val/perplexity")
            if t is not None:
                tokens.append(t)
                ppls.append(p)

        if not tokens:
            warnings.warn(f"No token data in {name}; skipping")
            continue

        # Sort by tokens
        paired = sorted(zip(tokens, ppls), key=lambda x: x[0])
        tokens = [x[0] for x in paired]
        ppls = [x[1] for x in paired]

        runs.append({
            "run_name": name,
            "group": group or "",
            "tokens": tokens,
            "ppls": ppls,
        })

    return runs


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save(fig, out_dir, stem):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}")


def plot_learning_curves(runs, out_dir, group_label="all"):
    fig, ax = plt.subplots()
    for i, run in enumerate(runs):
        tokens_m = [t / 1e6 for t in run["tokens"]]
        ppls = run["ppls"]
        valid = [(t, p) for t, p in zip(tokens_m, ppls) if p is not None]
        if not valid:
            continue
        tx, py = zip(*valid)
        final = py[-1]
        label = f"{run['run_name']} (final: {final:.2f})"
        ax.semilogy(
            tx, py,
            label=label,
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(tx) // 10),
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            linewidth=1.2,
            markersize=4,
        )

    ax.set_xlabel("Tokens (millions)")
    ax.set_ylabel("Val Perplexity (log scale)")
    ax.set_title(f"Learning Curves — {group_label}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.8)
    plt.tight_layout()
    _save(fig, out_dir, "fig_learning_curves")
    plt.close(fig)


def plot_efficiency_profile(runs, out_dir):
    thresholds = [100, 75, 50, 25]

    # Identify transformer baseline runs
    transformer_runs = [r for r in runs if "transformer" in r["run_name"].lower()]

    fig, ax = plt.subplots()
    for i, run in enumerate(runs):
        thr_tokens = _tokens_to_threshold(run["tokens"], run["ppls"], thresholds)
        xs, ys = [], []
        dnr_xs = []
        for thr in thresholds:
            val = thr_tokens.get(f"tokens_to_{thr}ppl")
            if val is not None:
                xs.append(thr)
                ys.append(val / 1e6)
            else:
                dnr_xs.append(thr)

        marker = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        color_val = 0.15 + 0.7 * (i / max(len(runs) - 1, 1))
        color = str(color_val)

        if xs:
            ax.semilogy(
                xs, ys,
                label=run["run_name"],
                marker=marker,
                linestyle=ls,
                linewidth=1.2,
                markersize=5,
                color=color,
            )
        # DNR markers
        for dx in dnr_xs:
            # Estimate a y position at top of current axis for annotation
            ax.scatter(
                [dx], [ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e3],
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=40,
                zorder=5,
            )
            ax.annotate(
                "DNR",
                xy=(dx, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e3),
                fontsize=7,
                ha="center",
                va="bottom",
                color=color,
            )

    # Transformer baseline horizontal dashed lines
    for tr in transformer_runs:
        thr_tokens = _tokens_to_threshold(tr["tokens"], tr["ppls"], thresholds)
        for thr in thresholds:
            val = thr_tokens.get(f"tokens_to_{thr}ppl")
            if val is not None:
                ax.axhline(val / 1e6, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(thresholds)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Perplexity Threshold")
    ax.set_ylabel("Tokens to Threshold (millions, log scale)")
    ax.set_title("Efficiency Profile")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.8)
    plt.tight_layout()
    _save(fig, out_dir, "fig_efficiency_profile")
    plt.close(fig)


def plot_aulc(runs, out_dir):
    names = [r["run_name"] for r in runs]
    aulcs_full, aulcs_early, aulcs_late = [], [], []

    for run in runs:
        f, e, l = _aulc(run["tokens"], run["ppls"])
        aulcs_full.append(f if f is not None else float("nan"))
        aulcs_early.append(e if e is not None else float("nan"))
        aulcs_late.append(l if l is not None else float("nan"))

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots()
    bars_full = ax.bar(x - width, aulcs_full, width, label="Full", color="0.2")
    bars_early = ax.bar(x, aulcs_early, width, label="Early 25%", color="0.55")
    bars_late = ax.bar(x + width, aulcs_late, width, label="Late 25%", color="0.80")

    # Transformer AULC_full dashed line
    transformer_runs = [r for r in runs if "transformer" in r["run_name"].lower()]
    for tr in transformer_runs:
        tf, _, _ = _aulc(tr["tokens"], tr["ppls"])
        if tf is not None:
            ax.axhline(tf, color="black", linestyle="--", linewidth=1.0,
                       label=f"Transformer AULC_full ({tf:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Run")
    ax.set_ylabel("AULC (lower = better)")
    ax.set_title("AULC Comparison (lower = better)")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    _save(fig, out_dir, "fig_aulc")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(runs, out_dir):
    out_path = Path(out_dir) / "results_table.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    thresholds = [100, 75, 50, 25]
    fieldnames = [
        "run_name", "group", "final_ppl",
        "aulc_full", "aulc_early", "aulc_late",
        "tokens_to_100ppl", "tokens_to_75ppl", "tokens_to_50ppl", "tokens_to_25ppl",
    ]
    rows = []
    for run in runs:
        ppls_valid = [p for p in run["ppls"] if p is not None]
        final_ppl = ppls_valid[-1] if ppls_valid else None
        aulc_full, aulc_early, aulc_late = _aulc(run["tokens"], run["ppls"])
        thr = _tokens_to_threshold(run["tokens"], run["ppls"], thresholds)
        rows.append({
            "run_name": run["run_name"],
            "group": run["group"],
            "final_ppl": f"{final_ppl:.4f}" if final_ppl is not None else "",
            "aulc_full": f"{aulc_full:.4f}" if aulc_full is not None else "",
            "aulc_early": f"{aulc_early:.4f}" if aulc_early is not None else "",
            "aulc_late": f"{aulc_late:.4f}" if aulc_late is not None else "",
            "tokens_to_100ppl": thr.get("tokens_to_100ppl") or "",
            "tokens_to_75ppl": thr.get("tokens_to_75ppl") or "",
            "tokens_to_50ppl": thr.get("tokens_to_50ppl") or "",
            "tokens_to_25ppl": thr.get("tokens_to_25ppl") or "",
        })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return rows, fieldnames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse W&B offline runs and generate paper figures."
    )
    parser.add_argument("--wandb-dir", default="wandb/", help="Path to wandb directory")
    parser.add_argument("--group", default=None, help="Filter by W&B group name")
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated list of run names to include (overrides --group filter)"
    )
    parser.add_argument("--out-dir", default="results/", help="Output directory")
    args = parser.parse_args()

    run_filter = None
    if args.runs:
        run_filter = set(r.strip() for r in args.runs.split(","))

    runs = discover_runs(args.wandb_dir, group_filter=args.group, run_filter=run_filter)

    if not runs:
        print("No runs found. Check --wandb-dir and filter arguments.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s).")

    # Write CSV and collect rows for stdout summary
    rows, fieldnames = write_csv(runs, args.out_dir)

    # Print summary table to stdout
    col_widths = {f: max(len(f), max((len(str(r[f])) for r in rows), default=0)) for f in fieldnames}
    header = "  ".join(f.ljust(col_widths[f]) for f in fieldnames)
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[f]).ljust(col_widths[f]) for f in fieldnames))

    # Figures
    group_label = args.group or "all"
    print("\nGenerating figures...")
    plot_learning_curves(runs, args.out_dir, group_label=group_label)
    print(f"  Saved fig_learning_curves.pdf/png to {args.out_dir}")
    plot_efficiency_profile(runs, args.out_dir)
    print(f"  Saved fig_efficiency_profile.pdf/png to {args.out_dir}")
    plot_aulc(runs, args.out_dir)
    print(f"  Saved fig_aulc.pdf/png to {args.out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
