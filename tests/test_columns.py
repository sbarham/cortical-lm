"""Tests for cortical column models."""

import pytest
import torch
from cortexlm.utils.config import get_default_config
from cortexlm.columns.simple_ei import SimpleEIColumn
from cortexlm.columns.layered import LayeredColumn


def _small_config(col_model="simple_ei"):
    cfg = get_default_config()
    cfg["column"]["model"] = col_model
    cfg["column"]["n_e"] = 20
    cfg["column"]["n_i"] = 5
    cfg["column"]["layer_sizes"] = {
        "l4":  {"n_e": 20, "n_i": 5},
        "l23": {"n_e": 40, "n_i": 10},
        "l5":  {"n_e": 20, "n_i": 5},
        "l6":  {"n_e": 20, "n_i": 5},
    }
    cfg["embedding"]["dim"] = 32
    return cfg


# ── SimpleEIColumn ─────────────────────────────────────────────────────────

def test_simple_ei_output_shape():
    cfg = _small_config("simple_ei")
    col = SimpleEIColumn(cfg)
    batch = 4
    state = col.init_state(batch)
    inputs = {"thalamic_input": torch.randn(batch, 32)}
    outputs, new_state = col(inputs, state)
    assert "e_out" in outputs
    assert outputs["e_out"].shape == (batch, cfg["column"]["n_e"])


def test_simple_ei_state_keys():
    cfg = _small_config("simple_ei")
    col = SimpleEIColumn(cfg)
    state = col.init_state(2)
    assert "r_e" in state
    assert "r_i" in state


def test_simple_ei_state_evolves():
    cfg = _small_config("simple_ei")
    col = SimpleEIColumn(cfg)
    batch = 2
    state = col.init_state(batch)
    inputs = {"thalamic_input": torch.randn(batch, 32)}
    _, state2 = col(inputs, state)
    assert not torch.allclose(state["r_e"], state2["r_e"])


def test_simple_ei_interface():
    cfg = _small_config("simple_ei")
    col = SimpleEIColumn(cfg)
    assert "thalamic_input" in col.input_keys()
    assert "e_out" in col.output_keys()


# ── LayeredColumn ─────────────────────────────────────────────────────────

def test_layered_output_keys():
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    batch = 2
    state = col.init_state(batch)
    inputs = {
        "thalamic_input": torch.randn(batch, 32),
        "l23_feedback": None,
    }
    outputs, _ = col(inputs, state)
    assert set(outputs.keys()) == {"l23_out", "l5_out", "l6_out"}


def test_layered_output_shapes():
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    batch = 3
    state = col.init_state(batch)
    inputs = {"thalamic_input": torch.randn(batch, 32)}
    outputs, _ = col(inputs, state)
    ls = cfg["column"]["layer_sizes"]
    assert outputs["l23_out"].shape == (batch, ls["l23"]["n_e"])
    assert outputs["l5_out"].shape  == (batch, ls["l5"]["n_e"])
    assert outputs["l6_out"].shape  == (batch, ls["l6"]["n_e"])


def test_layered_state_evolves():
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    batch = 2
    state = col.init_state(batch)
    x = torch.randn(batch, 32)
    inputs = {"thalamic_input": x}
    _, state2 = col(inputs, state)
    assert not torch.allclose(state["r_l4e"], state2["r_l4e"])


def test_layered_state_has_all_layers():
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    state = col.init_state(2)
    for layer in ["l4", "l23", "l5", "l6"]:
        assert f"r_{layer}e" in state
        assert f"r_{layer}i" in state


def test_layered_output_in_range():
    """All layer outputs should be in [0, 1] (sigmoid activations)."""
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    batch = 4
    state = col.init_state(batch)
    inputs = {"thalamic_input": torch.randn(batch, 32) * 5}
    outputs, _ = col(inputs, state)
    for key, val in outputs.items():
        assert val.min().item() >= 0.0 - 1e-5, f"{key} below 0"
        assert val.max().item() <= 1.0 + 1e-5, f"{key} above 1"


def test_layered_excitatory_weights_nonneg():
    """All excitatory weights in layered column must be >= 0 (Dale's Law)."""
    cfg = _small_config("layered")
    col = LayeredColumn(cfg)
    from cortexlm.synapses.static import StaticSynapse
    for name, mod in col.named_modules():
        if isinstance(mod, StaticSynapse):
            assert (mod.W_e >= 0).all(), f"{name}.W_e has negative values"
            assert (mod.W_i <= 0).all(), f"{name}.W_i has positive values"
