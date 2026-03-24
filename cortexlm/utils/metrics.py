"""Evaluation metrics: perplexity, BPC, firing rate stats, timescale estimation."""

import math
import numpy as np
import torch


def compute_perplexity(loss: float) -> float:
    return math.exp(loss)


def compute_bpt(loss: float) -> float:
    """Bits per token: loss (nats/token) converted to bits/token."""
    return loss / math.log(2)


def compute_bpb(loss: float, avg_bytes_per_token: float) -> float:
    """Bits per byte: bpt divided by average token length in bytes."""
    if avg_bytes_per_token <= 0:
        return float("nan")
    return compute_bpt(loss) / avg_bytes_per_token


# Legacy alias
compute_bpc = compute_bpt


def mean_firing_rates(model_outputs: dict) -> dict:
    """
    Given a dict of layer_name -> activation tensor [batch, seq, n_neurons],
    return mean firing rate per layer.
    """
    rates = {}
    for name, acts in model_outputs.items():
        rates[name] = acts.mean().item()
    return rates


def estimate_autocorrelation(trace: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """
    Estimate normalized autocorrelation of a 1D trace up to max_lag.
    trace: [T]
    Returns: [max_lag+1] autocorrelation values, normalized so [0]=1.
    """
    T = len(trace)
    trace = trace - trace.mean()
    var = (trace ** 2).mean()
    if var < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if T - lag > 0:
            acf[lag] = (trace[:T - lag] * trace[lag:]).mean() / var
    return acf


def fit_exponential_timescale(acf: np.ndarray, dt: float = 1.0) -> float:
    """
    Fit tau to acf[lag] ~ exp(-lag / tau) using log-linear regression.
    Returns tau in timesteps (multiply by dt for seconds).
    """
    lags = np.arange(len(acf))
    # Only fit positive portion
    pos_mask = acf > 0.01
    if pos_mask.sum() < 3:
        return 1.0
    log_acf = np.log(np.clip(acf[pos_mask], 1e-10, None))
    lags_fit = lags[pos_mask].astype(float)
    # Linear regression: log_acf = -lags / tau + const
    A = np.column_stack([lags_fit, np.ones_like(lags_fit)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_acf, rcond=None)
    slope = coeffs[0]
    if slope >= 0:
        return float(len(acf))  # can't estimate; return max
    tau = -1.0 / slope
    return float(tau * dt)


def compute_effective_timescales(
    activation_traces: torch.Tensor,  # [T, n_neurons]
    max_lag: int = 200,
    dt: float = 1.0,
) -> np.ndarray:
    """
    For each neuron, compute effective timescale from autocorrelation.
    Returns: [n_neurons] array of tau_eff values.
    """
    traces = activation_traces.detach().cpu().numpy()
    T, N = traces.shape
    taus = np.zeros(N)
    for i in range(N):
        acf = estimate_autocorrelation(traces[:, i], min(max_lag, T // 4))
        taus[i] = fit_exponential_timescale(acf, dt)
    return taus
