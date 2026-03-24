"""Evaluation script.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint --split val|test
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from cortexlm.utils.metrics import compute_perplexity, compute_bpt, compute_bpb, compute_effective_timescales
from cortexlm.model import CortexLM
from cortexlm.data import get_dataset, make_dataloader, build_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate CortexLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max-batches", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    print("Building tokenizer and data...")
    tokenizer = build_tokenizer(config)
    config["data"]["vocab_size"] = tokenizer.vocab_size
    _, val_ds, test_ds, _ = get_dataset(config, tokenizer)
    ds = val_ds if args.split == "val" else test_ds
    loader = make_dataloader(ds, config, shuffle=False)

    model = CortexLM(config, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            state = model.init_state(x.shape[0])
            logits, _ = model(x, state)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    tokenizer = build_tokenizer(config)
    avg_bpt = tokenizer.avg_bytes_per_token()
    print(f"Split:      {args.split}")
    print(f"Loss:       {avg_loss:.4f}")
    print(f"Perplexity: {compute_perplexity(avg_loss):.2f}")
    print(f"BPT:        {compute_bpt(avg_loss):.4f}  (bits per token)")
    print(f"BPB:        {compute_bpb(avg_loss, avg_bpt):.4f}  (bits per byte, avg {avg_bpt:.2f} bytes/token)")


if __name__ == "__main__":
    main()
