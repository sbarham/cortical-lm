"""Visualize the token vocabulary: lengths, examples, and embedding space.

Usage:
    # Just tokenizer stats (no model required):
    python scripts/viz_vocab.py --run-dir checkpoints/

    # Include embedding PCA (requires a model checkpoint):
    python scripts/viz_vocab.py --run-dir checkpoints/ --checkpoint checkpoints/step_0005000.pt
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── Load tokenizer ─────────────────────────────────────────────────────────

def load_tokenizer(run_dir: str):
    tok_path = os.path.join(run_dir, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        print(f"No tokenizer.pkl in {run_dir}")
        print("Run train.py first, or point --run-dir at your checkpoint directory.")
        sys.exit(1)
    with open(tok_path, "rb") as f:
        tok = pickle.load(f)
    return tok


def get_vocab_dict(tokenizer) -> dict:
    """
    Return {token_string: token_id} for whatever tokenizer type we have.
    Returns empty dict if the tokenizer has no string-level vocab (bytes, tiktoken).
    """
    cls = type(tokenizer).__name__

    if cls == "BPETokenizer" and tokenizer._tokenizer is not None:
        return tokenizer._tokenizer.get_vocab()  # {str: int}

    if cls == "CharTokenizer":
        return dict(tokenizer._char2id)  # {str: int}

    if cls == "BytePatchTokenizer":
        vocab = {}
        for patch_bytes, idx in tokenizer._patch2id.items():
            if isinstance(patch_bytes, bytes):
                try:
                    vocab[patch_bytes.decode("utf-8", errors="replace")] = idx
                except Exception:
                    vocab[repr(patch_bytes)] = idx
        return vocab

    if cls == "BytesTokenizer":
        return {chr(i): i for i in range(256)}

    if cls == "TiktokenTokenizer":
        # tiktoken exposes a limited decode-by-id API; just sample
        vocab = {}
        enc = tokenizer._enc
        for i in range(min(tokenizer.vocab_size, 10000)):
            try:
                vocab[enc.decode([i])] = i
            except Exception:
                pass
        return vocab

    return {}


# ── Panels ─────────────────────────────────────────────────────────────────

def plot_length_distribution(ax, vocab: dict, tokenizer_name: str):
    """Histogram of token string lengths."""
    if not vocab:
        ax.text(0.5, 0.5, "No string vocab available", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_title("Token length distribution")
        return

    lengths = [len(s) for s in vocab]
    bins = range(0, max(lengths) + 2)
    ax.hist(lengths, bins=bins, edgecolor="white", color="#4c72b0", linewidth=0.3)
    ax.set_xlabel("Token length (chars)")
    ax.set_ylabel("Count")
    ax.set_title(f"Token length distribution  [{tokenizer_name}, vocab={len(vocab):,}]")
    ax.axvline(np.mean(lengths), color="#dd4444", linestyle="--", linewidth=1,
               label=f"mean={np.mean(lengths):.1f}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def plot_token_grid(ax, vocab: dict, n_show: int = 120):
    """
    Display a grid of token strings, colored by token ID rank.
    Shows the first n_show tokens (roughly lowest IDs, which for BPE
    are often the most frequent / shortest).
    """
    ax.set_axis_off()
    if not vocab:
        ax.text(0.5, 0.5, "No string vocab", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_title("Token grid (n/a)")
        return

    # Sort by ID, take first n_show
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])[:n_show]
    tokens = [tok for tok, _ in by_id]
    ids    = [idx for _, idx in by_id]

    cols = 12
    rows = (len(tokens) + cols - 1) // cols

    norm = Normalize(vmin=0, vmax=max(ids) if ids else 1)
    cmap = plt.cm.viridis

    for k, (tok, tok_id) in enumerate(zip(tokens, ids)):
        row, col = divmod(k, cols)
        y = 1.0 - (row + 0.5) / rows
        x = (col + 0.5) / cols

        color = cmap(norm(tok_id))
        # Show repr for whitespace/control chars
        display = tok.replace("\n", "↵").replace("\t", "→").replace(" ", "·")
        display = display[:8] + "…" if len(display) > 8 else display

        ax.text(x, y, display, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                          alpha=0.6, edgecolor="none"))

    ax.set_title(f"First {len(tokens)} tokens by ID  (color = token rank)")


def plot_length_vs_id(ax, vocab: dict):
    """Scatter: token ID vs token string length, to show BPE compression structure."""
    if not vocab:
        ax.text(0.5, 0.5, "No string vocab", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_title("ID vs length (n/a)")
        return

    ids     = np.array(list(vocab.values()))
    lengths = np.array([len(k) for k in vocab.keys()])

    ax.scatter(ids, lengths, alpha=0.15, s=6, c=lengths, cmap="plasma", rasterized=True)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Token length (chars)")
    ax.set_title("Token ID vs string length")

    # Overlay mean length per ID-bin
    if len(ids) > 50:
        bin_edges = np.linspace(ids.min(), ids.max(), 30)
        bin_means = []
        bin_centers = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (ids >= lo) & (ids < hi)
            if mask.sum() > 0:
                bin_means.append(lengths[mask].mean())
                bin_centers.append((lo + hi) / 2)
        ax.plot(bin_centers, bin_means, color="#dd4444", linewidth=1.5, label="bin mean")
        ax.legend(fontsize=8)

    ax.grid(True, alpha=0.3)


def plot_embedding_pca(ax, checkpoint_path: str, vocab: dict, tokenizer):
    """PCA of the token embedding matrix from a saved checkpoint."""
    if checkpoint_path is None:
        ax.text(0.5, 0.5, "Pass --checkpoint to see\nembedding PCA",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title("Embedding PCA (no checkpoint)")
        return

    try:
        import torch
        from sklearn.decomposition import PCA
    except ImportError:
        ax.text(0.5, 0.5, "scikit-learn required\npip install scikit-learn",
                transform=ax.transAxes, ha="center", va="center", color="red")
        ax.set_title("Embedding PCA")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    emb_key = next((k for k in sd if "embedding.weight" in k), None)
    if emb_key is None:
        ax.text(0.5, 0.5, "No embedding.weight in checkpoint",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title("Embedding PCA (not found)")
        return

    W = sd[emb_key].float().numpy()  # [vocab_size, embed_dim]
    vocab_size = W.shape[0]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(W)  # [vocab_size, 2]

    # Color by token ID
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=np.arange(vocab_size), cmap="viridis",
                    alpha=0.4, s=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Token ID", fraction=0.046, pad=0.04)

    # Label a handful of interesting tokens
    if vocab:
        by_id = sorted(vocab.items(), key=lambda kv: kv[1])
        label_targets = (
            by_id[:5] +                             # very first tokens
            by_id[vocab_size // 4: vocab_size // 4 + 3] +  # quarter
            by_id[vocab_size // 2: vocab_size // 2 + 3] +  # midpoint
            by_id[-5:]                              # last tokens
        )
        for tok_str, tok_id in label_targets:
            if tok_id < vocab_size:
                x, y = coords[tok_id]
                display = tok_str.replace("\n", "↵").replace(" ", "·")[:6]
                ax.annotate(display, (x, y), fontsize=5, fontfamily="monospace",
                            xytext=(3, 3), textcoords="offset points", alpha=0.8)

    pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({pct[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({pct[1]:.1f}%)")
    ax.set_title(f"Token embedding PCA  [step {ckpt.get('step', '?'):,}]"
                 if isinstance(ckpt.get("step"), int) else "Token embedding PCA")
    ax.grid(True, alpha=0.2)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize the CortexLM token vocabulary")
    parser.add_argument("--run-dir", required=True,
                        help="Checkpoint directory containing tokenizer.pkl")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint (.pt) for embedding PCA")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <run-dir>/vocab.png)")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.run_dir)
    cls_name = type(tokenizer).__name__
    vocab = get_vocab_dict(tokenizer)

    print(f"Tokenizer: {cls_name}  vocab_size={tokenizer.vocab_size:,}")
    if vocab:
        lengths = [len(s) for s in vocab]
        print(f"  Token lengths: min={min(lengths)} mean={np.mean(lengths):.1f} max={max(lengths)}")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    plot_length_distribution(fig.add_subplot(gs[0, 0]), vocab, cls_name)
    plot_token_grid(fig.add_subplot(gs[0, 1]), vocab)
    plot_length_vs_id(fig.add_subplot(gs[1, 0]), vocab)
    plot_embedding_pca(fig.add_subplot(gs[1, 1]), args.checkpoint, vocab, tokenizer)

    out_path = args.output or os.path.join(args.run_dir, "vocab.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
