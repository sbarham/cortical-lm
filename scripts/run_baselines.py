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
from cortexlm.utils.logging import Logger, setup_logging
from cortexlm.model import CortexLM
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer
from cortexlm.baselines import get_baseline


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_hidden_size(target_params: int, baseline_name: str, vocab_size: int,
                      embed_dim: int, n_layers: int, seq_len: int) -> int:
    """Binary search for hidden_size that gives ~target_params."""
    # Transformer requires hidden_size divisible by n_heads=4; snap every
    # candidate to the nearest multiple of 4 so the exception branch is
    # never triggered by a divisibility error.
    n_heads = 4
    lo, hi = 16, 4096
    for _ in range(20):
        mid = ((lo + hi) // 2 // n_heads) * n_heads
        if mid <= lo:
            break
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


def _resolve_interval(config: dict, key_tokens: str, key_steps: str, default_steps: int) -> int:
    tcfg = config["training"]
    lcfg = config.get("logging", {})
    token_val = tcfg.get(key_tokens) or lcfg.get(key_tokens)
    if token_val is not None:
        batch_size = tcfg.get("batch_size", 32)
        seq_len    = config.get("data", {}).get("seq_len", 128)
        return max(1, int(token_val) // (batch_size * seq_len))
    return tcfg.get(key_steps) or lcfg.get(key_steps, default_steps)


def train_baseline(model, name, train_loader, val_loader, config, device, results_log, logger=None):
    """Train one baseline model and log perplexity vs tokens."""
    model = model.to(device)
    tcfg = config["training"]
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    warmup = tcfg.get("warmup_steps", 100)

    if "max_tokens" in tcfg:
        batch_size = tcfg.get("batch_size", 32)
        seq_len = config.get("data", {}).get("seq_len", 128)
        max_steps = max(1, int(tcfg["max_tokens"]) // (batch_size * seq_len))
    else:
        max_steps = tcfg.get("max_steps", 100_000)

    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup, 1))

    perp_log = []
    tokens_seen = 0
    step = 0
    train_iter = iter(train_loader)
    eval_interval = _resolve_interval(config, "eval_tokens", "eval_interval", 500)
    log_interval  = _resolve_interval(config, "log_tokens",  "log_interval",  100)

    while step < max_steps:
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

        # Linear warmup then cosine decay
        if step < warmup:
            factor = (step + 1) / warmup
            for pg in optimizer.param_groups:
                pg["lr"] = tcfg["lr"] * factor
        else:
            scheduler.step()

        if step % log_interval == 0:
            train_ppl = compute_perplexity(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            if logger:
                logger.log({
                    "train/perplexity": train_ppl,
                    "train/loss": loss.item(),
                    "lr": lr,
                    "tokens": tokens_seen,
                }, step=step)

        if step % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_loader):
                    if i >= 50:
                        break
                    xv, yv = xv.to(device), yv.to(device)
                    lv, _ = model(xv)
                    val_loss += F.cross_entropy(
                        lv.reshape(-1, lv.size(-1)), yv.reshape(-1)
                    ).item()
                    n += 1
            val_loss /= max(n, 1)
            train_ppl = compute_perplexity(loss.item())
            val_ppl   = compute_perplexity(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            perp_log.append({
                "tokens": tokens_seen, "step": step,
                "val_perplexity": val_ppl, "train_perplexity": train_ppl,
            })
            print(f"  step={step:6d} | tokens={tokens_seen:10d} "
                  f"| train_ppl={train_ppl:.2f} | val_ppl={val_ppl:.2f}")
            if logger:
                logger.log({
                    "val/perplexity": val_ppl,
                    "val/loss": val_loss,
                    "train/perplexity": train_ppl,
                    "train/loss": loss.item(),
                    "lr": lr,
                    "tokens": tokens_seen,
                }, step=step)

        step += 1

    results_log.extend(perp_log)
    return perp_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+",
                        default=["rnn", "lstm", "lstm_attention", "transformer"])
    parser.add_argument("--output", default="baseline_results.json")
    parser.add_argument("--tokenizer", default="tokenizers/tinystories_bpe4096.pkl",
                        help="Path to a saved tokenizer.pkl (skips BPE retraining)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (overrides config)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (overrides config)")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B run group — use same group as canonical runs to overlay curves")
    args = parser.parse_args()

    config = get_config(args.config)
    if args.wandb:
        config.setdefault("logging", {})["wandb"] = True
    if args.wandb_project:
        config.setdefault("logging", {})["project"] = args.wandb_project
    if args.wandb_group:
        config.setdefault("logging", {})["group"] = args.wandb_group
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tokenizer:
        import pickle
        print(f"Loading tokenizer from {args.tokenizer}")
        with open(args.tokenizer, "rb") as f:
            tokenizer = pickle.load(f)
    else:
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

        baseline_config = {
            **config,
            "name": f"baseline-{name}",
            "training": {**config["training"],
                         "checkpoint_dir": f"checkpoints/baseline_{name}"},
        }
        logger = Logger(baseline_config)
        results_log = []
        perp_log = train_baseline(model, name, train_loader, val_loader, config, device, results_log, logger)
        logger.finish()
        all_results["baselines"][name] = {"params": n, "hidden_size": hidden, "log": perp_log}

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
