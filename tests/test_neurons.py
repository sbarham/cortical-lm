"""Tests for neuron population models."""

import math
import pytest
import torch
from cortexlm.utils.config import get_default_config
from cortexlm.neurons.rate import RateNeurons
from cortexlm.neurons.rate_adex import RateAdExNeurons
from cortexlm.neurons.utils import init_lognormal_taus


@pytest.fixture
def config():
    cfg = get_default_config()
    cfg["neuron"]["tau_m_range"] = [2.0, 30.0]
    cfg["neuron"]["tau_w_range"] = [30.0, 500.0]
    cfg["neuron"]["learn_taus"] = False
    return cfg


# ── Rate neurons ───────────────────────────────────────────────────────────

def test_rate_output_in_01(config):
    """Rate-coded output must be in [0, 1] for any input."""
    n, batch = 64, 8
    pop = RateNeurons(n, config)
    state = pop.init_state(batch)
    for inp in [torch.zeros(batch, n), torch.randn(batch, n) * 10, -torch.ones(batch, n) * 100]:
        out, _ = pop(inp, state)
        assert out.shape == (batch, n)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6


def test_rate_state_evolves(config):
    """State should change under non-zero input."""
    n, batch = 32, 4
    pop = RateNeurons(n, config)
    state = pop.init_state(batch)
    x = torch.ones(batch, n)
    _, state2 = pop(x, state)
    assert not torch.allclose(state["v"], state2["v"])


def test_rate_state_keys(config):
    pop = RateNeurons(32, config)
    assert "v" in pop.state_keys()


def test_rate_init_state_shape(config):
    n, batch = 64, 8
    pop = RateNeurons(n, config)
    state = pop.init_state(batch)
    assert state["v"].shape == (batch, n)


# ── AdEx neurons ───────────────────────────────────────────────────────────

def test_adex_output_in_01(config):
    """AdEx output must be in [0, 1]."""
    n, batch = 64, 8
    pop = RateAdExNeurons(n, config)
    state = pop.init_state(batch)
    for inp in [torch.zeros(batch, n), torch.randn(batch, n) * 5]:
        out, _ = pop(inp, state)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6


def test_adex_constant_input_convergence(config):
    """Under constant input, v should move toward a fixed point."""
    n, batch = 16, 2
    pop = RateAdExNeurons(n, config)
    state = pop.init_state(batch)
    x = torch.ones(batch, n) * 0.5

    v_history = []
    for _ in range(50):
        out, state = pop(x, state)
        v_history.append(state["v"].mean().item())

    # v should settle (difference between last 10 steps should be small)
    diffs = [abs(v_history[i+1] - v_history[i]) for i in range(40, 49)]
    assert max(diffs) < 0.1, f"v did not converge: diffs={diffs}"


def test_adex_adaptation_grows_with_sustained_input(config):
    """Adaptation variable w should increase under sustained excitatory input."""
    n, batch = 16, 2
    pop = RateAdExNeurons(n, config)
    state = pop.init_state(batch)
    x = torch.ones(batch, n) * 2.0

    w_init = state["w"].mean().item()
    for _ in range(30):
        _, state = pop(x, state)
    w_final = state["w"].mean().item()

    assert w_final > w_init, f"w did not grow: init={w_init}, final={w_final}"


def test_adex_adaptation_decays_after_silence(config):
    """After driving w up, removing input should let w decay."""
    n, batch = 16, 2
    pop = RateAdExNeurons(n, config)
    state = pop.init_state(batch)

    # Drive up w
    x_on = torch.ones(batch, n) * 3.0
    for _ in range(50):
        _, state = pop(x_on, state)
    w_peak = state["w"].mean().item()

    # Silence
    x_off = torch.zeros(batch, n)
    for _ in range(100):
        _, state = pop(x_off, state)
    w_after = state["w"].mean().item()

    assert w_after < w_peak, f"w did not decay: peak={w_peak}, after={w_after}"


def test_adex_state_keys(config):
    pop = RateAdExNeurons(32, config)
    assert set(pop.state_keys()) == {"v", "w"}


# ── Log-normal tau initialization ─────────────────────────────────────────

def test_lognormal_tau_distribution():
    """Check that drawn taus roughly span the specified range."""
    n = 10_000
    lo, hi = 2.0, 30.0
    taus = init_lognormal_taus(n, lo, hi)

    assert taus.shape == (n,)
    assert taus.min().item() >= lo - 1e-6
    assert taus.max().item() <= hi + 1e-6

    # Check log-normal: ~5th percentile ≈ lo, ~95th percentile ≈ hi
    log_taus = torch.log(taus)
    p5  = torch.quantile(log_taus, 0.05).item()
    p95 = torch.quantile(log_taus, 0.95).item()
    expected_mu = (math.log(lo) + math.log(hi)) / 2

    assert abs(log_taus.mean().item() - expected_mu) < 0.5, \
        f"Mean log-tau {log_taus.mean():.2f} far from expected {expected_mu:.2f}"


def test_lognormal_taus_are_positive():
    taus = init_lognormal_taus(1000, 1.0, 100.0)
    assert (taus > 0).all()


def test_tau_buffer_not_parameter(config):
    """Taus should be buffers by default, not parameters."""
    config["neuron"]["learn_taus"] = False
    pop = RateAdExNeurons(32, config)
    param_names = [n for n, _ in pop.named_parameters()]
    assert "tau_m" not in param_names
    assert "tau_w" not in param_names


def test_tau_learnable_when_configured(config):
    """Taus should be parameters when learn_taus=True."""
    config["neuron"]["learn_taus"] = True
    pop = RateAdExNeurons(32, config)
    param_names = [n for n, _ in pop.named_parameters()]
    assert "tau_m" in param_names
    assert "tau_w" in param_names
