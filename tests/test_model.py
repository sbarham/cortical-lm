"""Tests for the top-level CortexLM model."""

import pytest
import torch
import torch.nn.functional as F
from cortexlm.utils.config import get_default_config
from cortexlm.model import CortexLM, ModelState


def _small_config(col_model="simple_ei", hpc="none"):
    cfg = get_default_config()
    cfg["column"]["model"] = col_model
    cfg["column"]["n_columns"] = 4
    cfg["column"]["n_e"] = 20
    cfg["column"]["n_i"] = 5
    cfg["column"]["layer_sizes"] = {
        "l4":  {"n_e": 20, "n_i": 5},
        "l23": {"n_e": 40, "n_i": 10},
        "l5":  {"n_e": 20, "n_i": 5},
        "l6":  {"n_e": 20, "n_i": 5},
    }
    cfg["embedding"]["dim"] = 16
    cfg["readout"]["hidden_dim"] = 32
    cfg["readout"]["n_layers"] = 1
    cfg["hippocampus"]["model"] = hpc
    if hpc == "modern_hopfield":
        cfg["hippocampus"]["n_memories"] = 16
        cfg["hippocampus"]["d_model"] = 32
        cfg["hippocampus"]["ca1"] = False
    cfg["connectivity"]["type"] = "gaussian_1d"
    cfg["connectivity"]["p_max"] = 0.5
    cfg["connectivity"]["sigma"] = 2.0
    cfg["neuron"]["model"] = "rate"
    cfg["synapse"]["inter_column_stp"] = False
    return cfg


VOCAB_SIZE = 50


def test_forward_logit_shape():
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    batch, seq_len = 2, 8
    tokens = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    logits, _ = model(tokens)
    assert logits.shape == (batch, seq_len, VOCAB_SIZE)


def test_loss_finite_positive():
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    batch, seq_len = 2, 8
    tokens = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    targets = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    logits, _ = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_gradients_flow_to_all_parameters():
    """
    Gradients must flow to all *active* parameters.
    Zero-dim synapse weights (W_e_raw with n_pre_e=0, or W_i_raw with n_pre_i=0)
    are structural placeholders and are intentionally not used in the forward pass.
    """
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    batch, seq_len = 2, 4
    tokens = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    targets = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    logits, _ = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    loss.backward()

    # Identify structurally dead params: zero-dim sides in connectivity StaticSynapse
    # (BatchedStaticSynapse in columns never creates W_*_raw for zero-dim sides)
    from cortexlm.synapses.static import StaticSynapse
    dead_params = set()
    for mname, mod in model.named_modules():
        if isinstance(mod, StaticSynapse):
            if mod.n_pre_e == 0:
                dead_params.add(f"{mname}.W_e_raw")
            if mod.n_pre_i == 0:
                dead_params.add(f"{mname}.W_i_raw")

    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None and name not in dead_params:
            no_grad_params.append(name)

    assert len(no_grad_params) == 0, \
        f"Active parameters with no gradient: {no_grad_params}"


def test_state_changes_across_steps():
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    batch = 2
    state0 = model.init_state(batch)
    tok = torch.randint(0, VOCAB_SIZE, (batch,))
    _, state1 = model.step(tok, state0)

    # At least one state tensor should have changed
    changed = any(
        not torch.allclose(state0.column_states[k], state1.column_states[k])
        for k in state0.column_states
    )
    assert changed, "State did not change after a step with non-trivial input"


def test_state_persistence():
    """State at t=1 should differ from initial state."""
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    batch = 2
    state = model.init_state(batch)
    tokens = torch.randint(0, VOCAB_SIZE, (batch, 5))
    _, final_state = model(tokens, state)

    # final state should differ from initial — pick any state tensor
    cs0, cs1 = state.column_states, final_state.column_states
    key = next(k for k in cs0 if cs0[k].numel() > 0)
    assert not torch.allclose(cs0[key], cs1[key])


def test_model_with_hippocampus():
    cfg = _small_config(hpc="modern_hopfield")
    model = CortexLM(cfg, VOCAB_SIZE)
    batch, seq_len = 2, 4
    tokens = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    logits, _ = model(tokens)
    assert logits.shape == (batch, seq_len, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_model_parameter_count():
    cfg = _small_config()
    model = CortexLM(cfg, VOCAB_SIZE)
    n_params = model.count_parameters()
    assert n_params > 0
