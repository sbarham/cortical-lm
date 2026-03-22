"""Sample text from a trained CortexLM checkpoint using top-p (nucleus) sampling.

Usage:
    # Unconditional samples (random start):
    python scripts/sample.py --run-dir checkpoints/ --checkpoint checkpoints/step_0005000.pt

    # With a prompt:
    python scripts/sample.py --run-dir checkpoints/ --checkpoint checkpoints/step_0005000.pt \\
        --prompt "Once upon a time there was a little girl"

    # Greedy decoding (temperature=0):
    python scripts/sample.py --run-dir checkpoints/ --checkpoint checkpoints/step_0005000.pt \\
        --temperature 0 --n-samples 1

    # Multiple samples, higher randomness:
    python scripts/sample.py --run-dir checkpoints/ --checkpoint checkpoints/step_0005000.pt \\
        --n-samples 5 --temperature 1.1 --top-p 0.92 --max-tokens 300
"""

import argparse
import os
import pickle
import sys

import torch

# cortexlm must be on path before importing — load_run handles this,
# but we need it here too for the shared sampling utility.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortexlm.utils.sampling import generate


# ── Checkpoint loading ────────────────────────────────────────────────────

def load_run(run_dir: str, checkpoint_path: str, device: torch.device):
    # Ensure cortexlm is importable before unpickling the tokenizer
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Tokenizer
    tok_path = os.path.join(run_dir, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        print(f"tokenizer.pkl not found in {run_dir}. Run train.py first.")
        sys.exit(1)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    from cortexlm.model import CortexLM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    config["data"]["vocab_size"] = tokenizer.vocab_size

    # Reproduce the same random connectivity structure used during training.
    # The training seed is set before CortexLM() with no torch random ops in between,
    # so re-seeding here gives identical synapse key names in the state_dict.
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)

    model = CortexLM(config, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    step = ckpt.get("step", "?")
    return model, tokenizer, config, step


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sample text from a CortexLM checkpoint")
    parser.add_argument("--run-dir", required=True,
                        help="Checkpoint directory (contains tokenizer.pkl)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--prompt", default="",
                        help="Prompt string to condition on (empty = unconditional)")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of independent samples to generate (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens per sample (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Softmax temperature. 0 = greedy (default: 0.9)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus probability mass (default: 0.9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config, step = load_run(args.run_dir, args.checkpoint, device)
    n_params = model.count_parameters()

    print(f"  Model:       CortexLM  params={n_params:,}  step={step}")
    print(f"  Tokenizer:   {type(tokenizer).__name__}  vocab={tokenizer.vocab_size:,}")
    print(f"  Sampling:    temperature={args.temperature}  top_p={args.top_p}")
    if args.prompt:
        print(f"  Prompt:      {args.prompt!r}")
    else:
        print(f"  Prompt:      (unconditional — random start token)")
    print()

    DIVIDER = "-" * 72

    for i in range(args.n_samples):
        print(f"[ Sample {i + 1}/{args.n_samples} ]")
        print(DIVIDER)
        text = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        # Clean up null bytes and control chars for display
        text = text.replace("\x00", "").replace("\r", "")
        print(text)
        print(DIVIDER)
        print()


if __name__ == "__main__":
    main()
