"""Train all baseline models at matched parameter counts.

Usage:
    python scripts/run_baselines.py --config configs/standard.yaml \
        --models rnn lstm lstm_attention transformer
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cortexlm.utils.config import get_config
from cortexlm.utils.metrics import compute_perplexity
from cortexlm.model import CortexLM
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer
from cortexlm.baselines import get_baseline


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_hidden_size(target_params: int, baseline_name: str, vocab_size: int,
                      embed_dim: int, n_layers: int, seq_len: int) -> int:
    """Binary search for hidden_size that gives ~target_params."""
    lo, hi = 16, 4096
    for _ in range(20):
        mid = (lo + hi) // 2
        try:
            m = _make_baseline(baseline_name, vocab_size, embed_dim, mid, n_layers, seq_len)
            p = count_params(m)
            if p < target_params:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid
    return lo


def _make_baseline(name, vocab_size, embed_dim, hidden_size, n_layers, seq_len):
    from cortexlm.baselines.rnn import VanillaRNN
    from cortexlm.baselines.lstm import LSTMBaseline
    from cortexlm.baselines.rnn_attention import RNNWithAttention
    from cortexlm.baselines.lstm_attention import LSTMWithAttention
    from cortexlm.baselines.transformer import TransformerBaseline

    if name == "rnn":
        return VanillaRNN(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "lstm":
        return LSTMBaseline(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "rnn_attention":
        return RNNWithAttention(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "lstm_attention":
        return LSTMWithAttention(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "transformer":
        n_heads = 4
        d_ff = hidden_size * 4
        return TransformerBaseline(vocab_size, hidden_size, n_layers, n_heads, d_ff, seq_len)
    raise ValueError(name)


def train_baseline(model, train_loader, val_loader, config, device, results_log):
    """Train one baseline model and log perplexity vs tokens."""
    model = model.to(device)
    tcfg = config["training"]
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["max_steps"])

    perp_log = []
    tokens_seen = 0
    step = 0
    train_iter = iter(train_loader)
    eval_interval = tcfg["eval_interval"]

    while step < tcfg["max_steps"]:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        tokens_seen += x.numel()

        model.train()
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_loader):
                    if i >= 20:
                        break
                    xv, yv = xv.to(device), yv.to(device)
                    lv, _ = model(xv)
                    val_loss += F.cross_entropy(
                        lv.reshape(-1, lv.size(-1)), yv.reshape(-1)
                    ).item()
                    n += 1
            val_loss /= max(n, 1)
            perp = compute_perplexity(val_loss)
            perp_log.append({"tokens": tokens_seen, "step": step, "perplexity": perp})
            print(f"  step={step:6d} tokens={tokens_seen:10d} perplexity={perp:.2f}")

        step += 1

    results_log.extend(perp_log)
    return perp_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+",
                        default=["rnn", "lstm", "lstm_attention", "transformer"])
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()

    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config)
    config["data"]["vocab_size"] = tokenizer.vocab_size
    train_ds, val_ds, _, _ = get_dataset(config, tokenizer)
    train_loader = make_dataloader(train_ds, config, shuffle=True)
    val_loader   = make_dataloader(val_ds, config, shuffle=False)

    # Get target param count from cortex-lm
    cortex_model = CortexLM(config, tokenizer.vocab_size)
    target_params = cortex_model.count_parameters()
    print(f"CortexLM parameter count: {target_params:,}")
    del cortex_model

    embed_dim = config["embedding"]["dim"]
    seq_len   = config["data"]["seq_len"]
    n_layers  = config.get("baseline", {}).get("n_layers", 2)

    all_results = {"cortex_lm_params": target_params, "baselines": {}}

    for name in args.models:
        print(f"\n=== Training baseline: {name} ===")
        hidden = match_hidden_size(
            target_params, name, tokenizer.vocab_size, embed_dim, n_layers, seq_len
        )
        model = _make_baseline(name, tokenizer.vocab_size, embed_dim, hidden, n_layers, seq_len)
        n = count_params(model)
        print(f"  hidden_size={hidden}, params={n:,}")

        results_log = []
        perp_log = train_baseline(model, train_loader, val_loader, config, device, results_log)
        all_results["baselines"][name] = {"params": n, "hidden_size": hidden, "log": perp_log}

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
