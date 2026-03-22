"""Tests for synapse models."""

import pytest
import torch
from cortexlm.utils.config import get_default_config
from cortexlm.synapses.static import StaticSynapse
from cortexlm.synapses.stp import STPSynapse


@pytest.fixture
def config():
    return get_default_config()


# ── Dale's Law ─────────────────────────────────────────────────────────────

def test_static_dale_law_excitatory(config):
    """Excitatory weights must be >= 0."""
    syn = StaticSynapse(16, 8, 32)
    W_e = syn.W_e
    assert (W_e >= 0).all(), f"W_e has negative values: min={W_e.min().item()}"


def test_static_dale_law_inhibitory(config):
    """Inhibitory weights must be <= 0."""
    syn = StaticSynapse(16, 8, 32)
    W_i = syn.W_i
    assert (W_i <= 0).all(), f"W_i has positive values: max={W_i.max().item()}"


def test_static_dale_law_after_simulated_update(config):
    """Dale's Law must hold after parameter perturbation (softplus is always non-negative)."""
    syn = StaticSynapse(8, 4, 16)
    # Simulate an optimizer step that changes raw weights
    with torch.no_grad():
        syn.W_e_raw.data += torch.randn_like(syn.W_e_raw) * 5
        syn.W_i_raw.data += torch.randn_like(syn.W_i_raw) * 5
    syn.enforce_dale()
    assert (syn.W_e >= 0).all()
    assert (syn.W_i <= 0).all()


def test_static_synapse_output_shape(config):
    syn = StaticSynapse(16, 8, 32)
    batch = 4
    r_e = torch.rand(batch, 16)
    r_i = torch.rand(batch, 8)
    out = syn(r_e, r_i)
    assert out.shape == (batch, 32)


# ── STP dynamics ───────────────────────────────────────────────────────────

def test_stp_x_decreases_under_high_rate(config):
    """Depression variable x should decrease under sustained high firing."""
    syn = STPSynapse(8, 4, 16, config)
    batch = 2
    state = syn.init_state(batch)

    # High excitatory firing rate
    r_e_high = torch.ones(batch, 8) * 0.9
    r_i = torch.zeros(batch, 4)

    x_init = state["x_e"].mean().item()
    for _ in range(30):
        _, state = syn(r_e_high, r_i, state)
    x_final = state["x_e"].mean().item()

    assert x_final < x_init, f"x_e did not decrease: init={x_init:.3f}, final={x_final:.3f}"


def test_stp_u_increases_under_sustained_activity(config):
    """Facilitation variable u should increase under sustained activity."""
    syn = STPSynapse(8, 4, 16, config)
    batch = 2
    state = syn.init_state(batch)

    r_e = torch.ones(batch, 8) * 0.5
    r_i = torch.ones(batch, 4) * 0.5

    u_init = state["u_e"].mean().item()
    for _ in range(20):
        _, state = syn(r_e, r_i, state)
    u_final = state["u_e"].mean().item()

    assert u_final > u_init, f"u_e did not increase: init={u_init:.3f}, final={u_final:.3f}"


def test_stp_state_recovers_after_silence(config):
    """After sustained activity, silence should let x recover toward 1."""
    syn = STPSynapse(8, 4, 16, config)
    batch = 2
    state = syn.init_state(batch)

    r_e = torch.ones(batch, 8) * 0.9
    r_i = torch.zeros(batch, 4)

    # Drive depression
    for _ in range(50):
        _, state = syn(r_e, r_i, state)
    x_depressed = state["x_e"].mean().item()

    # Silence
    r_e_zero = torch.zeros(batch, 8)
    for _ in range(200):
        _, state = syn(r_e_zero, r_i, state)
    x_recovered = state["x_e"].mean().item()

    assert x_recovered > x_depressed, \
        f"x did not recover: depressed={x_depressed:.3f}, recovered={x_recovered:.3f}"


def test_stp_state_keys(config):
    syn = STPSynapse(8, 4, 16, config)
    state = syn.init_state(2)
    assert set(state.keys()) == {"u_e", "x_e", "u_i", "x_i"}


def test_stp_x_u_range(config):
    """x must be in [0, 1] and u in [U0, 1] throughout."""
    syn = STPSynapse(8, 4, 16, config)
    batch = 2
    state = syn.init_state(batch)
    r_e = torch.rand(batch, 8)
    r_i = torch.rand(batch, 4)

    for _ in range(50):
        _, state = syn(r_e, r_i, state)

    assert (state["x_e"] >= 0).all() and (state["x_e"] <= 1).all()
    assert (state["x_i"] >= 0).all() and (state["x_i"] <= 1).all()
    assert (state["u_e"] >= 0).all() and (state["u_e"] <= 1).all()
    assert (state["u_i"] >= 0).all() and (state["u_i"] <= 1).all()
