"""Config loading, validation, and defaults."""

import copy
import math
import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    "name": "cortex-lm",
    "version": "0.1",

    "data": {
        "dataset": "tinystories",      # tinystories | wikitext2 | wikitext103 | openwebtext | ptb
        "tokenizer": "bpe",            # char | bpe | bytes | byte_patch | tiktoken
        "vocab_size": 4096,            # null = infer from data (char/bytes); int = target vocab size
        "seq_len": 256,
        "train_split": 0.9,
        "bpe_train_sample": 100_000,   # chars to train BPE on (subset for speed)
        "byte_patch_size": 4,          # 2 | 4 | 8 (for byte_patch tokenizer)
        "tiktoken_encoding": "cl100k_base",
        "openwebtext_token_limit": 10_000_000,  # cap for streaming OWT
        "num_workers": 2,
        "cache_dir": "data/cache",     # pre-tokenized cache (built once, reused across runs)
        "streaming": False,            # use streaming for openwebtext automatically
    },

    "neuron": {
        "model": "rate_adex",          # rate | rate_adex | lif
        "nonlinearity": "tanh",        # tanh | sigmoid | relu
        "tau_m_dist": "lognormal",     # fixed | lognormal | uniform
        "tau_m_range": [2, 30],
        "tau_w_dist": "lognormal",
        "tau_w_range": [30, 500],
        "adaptation_a": 0.1,
        "learn_taus": False,
    },

    "synapse": {
        "inter_column_stp": True,
        "tau_f_range": [50, 300],
        "tau_d_range": [100, 800],
        "e_synapse_type": "depressing",
        "i_synapse_type": "facilitating",
        "U0_e": 0.2,
        "U0_i": 0.5,
    },

    "column": {
        "model": "layered",            # simple_ei | layered
        "n_columns": 16,
        "disinhibition": False,
        "layer_sizes": {
            "l4":  {"n_e": 80,  "n_i": 20},
            "l23": {"n_e": 160, "n_i": 40},
            "l5":  {"n_e": 80,  "n_i": 20},
            "l6":  {"n_e": 80,  "n_i": 20},
        },
        # For simple_ei
        "n_e": 80,
        "n_i": 20,
    },

    "connectivity": {
        "type": "gaussian_1d",         # gaussian_1d | small_world | random_sparse
        "p_max": 0.7,
        "sigma": 3.0,
        "k": 4,
        "beta": 0.1,
    },

    "hippocampus": {
        "model": "none",               # none | modern_hopfield
        "n_memories": 512,
        "d_model": 256,
        "beta": 1.0,
        "ca1": False,
    },

    "readout": {
        "source": "l5",                # l5 | l23 | all_layers
        "n_layers": 2,
        "hidden_dim": 256,
    },

    "embedding": {
        "dim": 64,
    },

    "learning": {
        "rule": "bptt",                # bptt | eprop
        "truncated_bptt_k": None,
        "reset_state_between_batches": False,
        "eprop_tau_e": None,           # null = use mean tau_m
    },

    "training": {
        "batch_size": 32,
        "max_steps": 100_000,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "optimizer": "adamw",          # adam | adamw
        "scheduler": "cosine",         # none | cosine | linear
        "warmup_steps": 1000,
        "eval_interval": 500,
        "checkpoint_interval": 5000,
        "checkpoint_dir": "checkpoints",
        "seed": 42,
    },

    "simulation": {
        "dt": 1.0,
    },

    "logging": {
        "wandb": False,
        "project": "cortex-lm",
        "log_interval": 100,
        "log_weight_stats": True,
        "log_tau_stats": True,
        "log_spike_rates": True,
        # Periodic text samples during training (0 = disabled)
        "sample_interval": 2000,
        "sample_max_tokens": 150,
        "sample_top_p": 0.9,
        "sample_temperature": 0.8,
        "sample_prompt": "",      # empty = random start token from vocab
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def get_default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def get_config(path: str, overrides: dict = None) -> dict:
    """Load YAML config from path, merge with defaults, apply overrides."""
    config = get_default_config()

    with open(path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_merge(config, user_config)

    if overrides:
        config = _deep_merge(config, overrides)

    _validate_config(config)
    return config


def _validate_config(cfg: dict):
    valid_datasets = {"tinystories", "wikitext2", "wikitext103", "openwebtext", "ptb"}
    valid_tokenizers = {"char", "bpe", "bytes", "byte_patch", "tiktoken"}
    valid_neuron_models = {"rate", "rate_adex", "lif"}
    valid_column_models = {"simple_ei", "layered"}
    valid_conn_types = {"gaussian_1d", "small_world", "random_sparse"}
    valid_hpc_models = {"none", "modern_hopfield"}
    valid_learning_rules = {"bptt", "eprop"}

    assert cfg["data"]["dataset"] in valid_datasets, f"Unknown dataset: {cfg['data']['dataset']}"
    assert cfg["data"]["tokenizer"] in valid_tokenizers, f"Unknown tokenizer: {cfg['data']['tokenizer']}"
    assert cfg["neuron"]["model"] in valid_neuron_models
    assert cfg["column"]["model"] in valid_column_models
    assert cfg["connectivity"]["type"] in valid_conn_types
    assert cfg["hippocampus"]["model"] in valid_hpc_models
    assert cfg["learning"]["rule"] in valid_learning_rules

    tau_m = cfg["neuron"]["tau_m_range"]
    tau_w = cfg["neuron"]["tau_w_range"]
    assert tau_m[0] > 0 and tau_m[1] > tau_m[0], "tau_m_range must be (lo, hi) with lo > 0"
    assert tau_w[0] > 0 and tau_w[1] > tau_w[0], "tau_w_range must be (lo, hi) with lo > 0"
