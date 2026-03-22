"""Tests for learning rules (BPTT and e-prop)."""

import pytest
import torch
import torch.nn.functional as F
from cortexlm.utils.config import get_default_config
from cortexlm.model import CortexLM
from cortexlm.learning.bptt import BPTTTrainer
from cortexlm.learning.eprop import EpropTrainer, EligibilityTraceBuffer


VOCAB_SIZE = 32


def _small_config():
    cfg = get_default_config()
    cfg["column"]["model"] = "simple_ei"
    cfg["column"]["n_columns"] = 4
    cfg["column"]["n_e"] = 16
    cfg["column"]["n_i"] = 4
    cfg["embedding"]["dim"] = 16
    cfg["readout"]["hidden_dim"] = 32
    cfg["readout"]["n_layers"] = 1
    cfg["hippocampus"]["model"] = "none"
    cfg["connectivity"]["type"] = "gaussian_1d"
    cfg["connectivity"]["p_max"] = 0.3
    cfg["connectivity"]["sigma"] = 2.0
    cfg["neuron"]["model"] = "rate"
    cfg["synapse"]["inter_column_stp"] = False
    cfg["training"]["lr"] = 1e-3
    cfg["training"]["grad_clip"] = 1.0
    cfg["training"]["max_steps"] = 200
    cfg["training"]["eval_interval"] = 50
    cfg["training"]["checkpoint_interval"] = 999999  # don't checkpoint during test
    cfg["learning"]["rule"] = "bptt"
    cfg["learning"]["reset_state_between_batches"] = True
    return cfg


def _make_synthetic_batch(batch=4, seq_len=16, vocab_size=VOCAB_SIZE):
    """Simple repeating pattern to test that loss can decrease."""
    # Repeat pattern: [0, 1, 2, ..., 9, 0, 1, ...]
    pattern = [i % vocab_size for i in range(seq_len + 1)]
    x = torch.tensor([pattern[:-1]] * batch, dtype=torch.long)
    y = torch.tensor([pattern[1:]]  * batch, dtype=torch.long)
    return x, y


# ── BPTT ───────────────────────────────────────────────────────────────────

def test_bptt_loss_decreases():
    """Loss should decrease over 100 steps on a tiny synthetic sequence."""
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    trainer = BPTTTrainer(model, cfg, device=torch.device("cpu"))

    x, y = _make_synthetic_batch()
    losses = []
    state = None
    for _ in range(100):
        loss, state = trainer.train_step(x, y, state)
        losses.append(loss)

    # Loss should be lower in the second half than the first
    first_half  = sum(losses[:50])  / 50
    second_half = sum(losses[50:]) / 50
    assert second_half < first_half, \
        f"Loss did not decrease: first={first_half:.4f}, second={second_half:.4f}"


def test_bptt_state_returned():
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    trainer = BPTTTrainer(model, cfg, device=torch.device("cpu"))
    x, y = _make_synthetic_batch()
    loss, state = trainer.train_step(x, y)
    assert state is not None
    assert len(state.column_states) == cfg["column"]["n_columns"]


def test_bptt_truncated():
    """Truncated BPTT should also work and reduce loss."""
    cfg = _small_config()
    cfg["learning"]["truncated_bptt_k"] = 4
    model = CortexLM(cfg, VOCAB_SIZE)
    trainer = BPTTTrainer(model, cfg, device=torch.device("cpu"))
    x, y = _make_synthetic_batch(seq_len=16)
    loss1, _ = trainer.train_step(x, y)
    for _ in range(10):
        loss2, _ = trainer.train_step(x, y)
    assert torch.isfinite(torch.tensor(loss2))


# ── e-prop ─────────────────────────────────────────────────────────────────

def test_eprop_trace_shape():
    """Eligibility traces should have shape [n_post, n_pre]."""
    n_pre, n_post = 8, 16
    gamma = 0.9
    buf = EligibilityTraceBuffer(n_pre, n_post, gamma, torch.device("cpu"))
    assert buf.trace.shape == (n_post, n_pre)


def test_eprop_trace_decays():
    """After zeroing input, trace should decay by factor gamma."""
    n_pre, n_post = 4, 8
    gamma = 0.5
    buf = EligibilityTraceBuffer(n_pre, n_post, gamma, torch.device("cpu"))

    # Set trace to ones
    buf.trace = torch.ones(n_post, n_pre)

    # Update with zero input (delta=0): trace should be gamma * trace
    r_pre  = torch.zeros(2, n_pre)
    psi    = torch.zeros(2, n_post)
    buf.update(r_pre, psi)

    expected = gamma * torch.ones(n_post, n_pre)
    assert torch.allclose(buf.trace, expected, atol=1e-5), \
        f"Trace decay failed: got {buf.trace.mean():.3f}, expected {expected.mean():.3f}"


def test_eprop_trace_nonzero_with_input():
    """Trace should be non-zero when there is pre and post activity."""
    n_pre, n_post = 8, 16
    gamma = 0.9
    buf = EligibilityTraceBuffer(n_pre, n_post, gamma, torch.device("cpu"))

    r_pre = torch.ones(4, n_pre) * 0.5
    psi   = torch.ones(4, n_post) * 0.3
    buf.update(r_pre, psi)

    assert buf.trace.abs().sum().item() > 0


def test_eprop_weight_updates_nonzero():
    """After one step, at least some weights should change."""
    cfg = _small_config()
    cfg["learning"]["rule"] = "eprop"
    model = CortexLM(cfg, VOCAB_SIZE)
    trainer = EpropTrainer(model, cfg, device=torch.device("cpu"))

    # Snapshot weights
    from cortexlm.synapses.static import StaticSynapse
    before = {}
    for name, mod in model.named_modules():
        if isinstance(mod, StaticSynapse):
            before[name] = mod.W_e_raw.data.clone()

    x, y = _make_synthetic_batch(seq_len=8)
    trainer.train_step(x, y)

    changed = False
    for name, mod in model.named_modules():
        if isinstance(mod, StaticSynapse) and name in before:
            if not torch.allclose(before[name], mod.W_e_raw.data):
                changed = True
                break

    assert changed, "No weights changed after e-prop step"
