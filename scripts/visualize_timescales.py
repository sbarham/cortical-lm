"""Visualize empirical timescales of cortical neurons.

Usage:
    python scripts/visualize_timescales.py --checkpoint path/to/checkpoint
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from cortexlm.model import CortexLM
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer
from cortexlm.utils.metrics import compute_effective_timescales


def collect_activation_traces(model, loader, n_batches=5, device=None):
    """
    Run the model on validation batches and collect per-neuron activation traces.

    Returns:
        traces: dict of layer_name -> Tensor [T_total, n_neurons]
    """
    model.eval()
    traces = {}

    total_tokens = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            batch, seq_len = x.shape

            state = model.init_state(batch)
            for t in range(seq_len):
                _, state = model.step(x[:, t], state)
                # Collect L5 activations from each column
                for col_i, cs in enumerate(state.column_states):
                    key_e = "r_l5e" if "r_l5e" in cs else "r_e"
                    act = cs[key_e].mean(dim=0).cpu()  # mean over batch: [n_l5e]
                    layer_key = f"col{col_i}_l5e"
                    if layer_key not in traces:
                        traces[layer_key] = []
                    traces[layer_key].append(act)
                total_tokens += batch

    # Stack time axis
    traces = {k: torch.stack(v, dim=0) for k, v in traces.items()}
    print(f"Collected traces: {total_tokens} token-steps across {len(traces)} layers")
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--output-dir", default="timescale_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Load pre-saved tokenizer if available; fall back to rebuilding
    import pickle
    tok_pkl = os.path.join(os.path.dirname(args.checkpoint), "tokenizer.pkl")
    if os.path.exists(tok_pkl):
        with open(tok_pkl, "rb") as _f:
            tokenizer = pickle.load(_f)
        print(f"Loaded tokenizer from {tok_pkl}")
    else:
        tokenizer = build_tokenizer(config)
    config["data"]["vocab_size"] = tokenizer.vocab_size
    _, val_ds, _, _ = get_dataset(config, tokenizer)
    val_loader = make_dataloader(val_ds, config, shuffle=False)

    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    model = CortexLM(config, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Collect traces
    traces = collect_activation_traces(model, val_loader, args.n_batches, device)

    # Compute effective timescales
    all_taus = []
    all_cols = []
    for layer_key, trace_tensor in traces.items():
        taus = compute_effective_timescales(trace_tensor)
        all_taus.extend(taus.tolist())
        col_idx = int(layer_key.split("col")[1].split("_")[0])
        all_cols.extend([col_idx] * len(taus))

    all_taus = np.array(all_taus)
    all_cols = np.array(all_cols)

    # Plot 1: Histogram of tau_eff
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(np.log10(all_taus + 1e-3), bins=40, edgecolor="k")
    axes[0].set_xlabel("log10(τ_eff) [timesteps]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Effective Timescales")

    # Plot 2: Scatter τ_eff vs column index
    scatter = axes[1].scatter(all_cols, all_taus, alpha=0.3, s=10, c=all_taus,
                               cmap="viridis", norm=plt.Normalize(0, np.percentile(all_taus, 95)))
    plt.colorbar(scatter, ax=axes[1], label="τ_eff")
    axes[1].set_xlabel("Column index")
    axes[1].set_ylabel("τ_eff [timesteps]")
    axes[1].set_title("Effective Timescale vs Column Position")

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "timescales.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # Print summary statistics
    print(f"\nTimescale statistics:")
    print(f"  Median τ_eff: {np.median(all_taus):.1f} timesteps")
    print(f"  5th  pct:     {np.percentile(all_taus, 5):.1f}")
    print(f"  95th pct:     {np.percentile(all_taus, 95):.1f}")


if __name__ == "__main__":
    main()
