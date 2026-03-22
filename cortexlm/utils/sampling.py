"""Shared top-p (nucleus) sampling utilities used by trainers and scripts."""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from typing import Optional


def top_p_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    """
    Nucleus (top-p) sampling from a logit vector.

    Args:
        logits:      [vocab_size] raw (un-normalised) scores
        top_p:       nucleus probability mass (0–1). 0 = greedy.
        temperature: softmax temperature. <1 = sharper, >1 = flatter.

    Returns: sampled token id (int)
    """
    if temperature <= 0.0 or top_p <= 0.0:
        return int(logits.argmax().item())

    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep only the smallest set summing to at least top_p
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum()

    rank = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_indices[rank].item())


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 150,
    temperature: float = 0.9,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text token-by-token from a CortexLM model.

    If prompt is empty, starts from a randomly chosen token in the first
    quarter of the vocabulary (avoids rare/sentinel tokens).

    Returns the full generated string (prompt included).
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    # Encode prompt
    if prompt:
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
    else:
        ids = [random.randint(0, max(1, tokenizer.vocab_size // 4))]

    # Prime hidden state from prompt tokens
    state = model.init_state(1)
    for tok_id in ids[:-1]:
        tok = torch.tensor([tok_id], dtype=torch.long, device=device)
        _, state = model.step(tok, state)

    current = torch.tensor([ids[-1]], dtype=torch.long, device=device)
    generated = list(ids)

    for _ in range(max_new_tokens):
        logits, state = model.step(current, state)  # logits: [1, vocab_size]
        next_id = top_p_sample(logits[0], top_p=top_p, temperature=temperature)
        generated.append(next_id)
        current = torch.tensor([next_id], dtype=torch.long, device=device)

    if was_training:
        model.train()

    text = tokenizer.decode(generated)
    return text.replace("\x00", "").replace("\r", "")
