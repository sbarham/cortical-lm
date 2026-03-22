from .base import BaselineModel
from .rnn import VanillaRNN
from .lstm import LSTMBaseline
from .rnn_attention import RNNWithAttention
from .lstm_attention import LSTMWithAttention
from .transformer import TransformerBaseline


def get_baseline(name: str, vocab_size: int, config: dict) -> BaselineModel:
    """Factory: instantiate baseline model by name."""
    bcfg = config.get("baseline", {})
    hidden_size = bcfg.get("hidden_size", 256)
    n_layers    = bcfg.get("n_layers", 2)
    embed_dim   = config["embedding"]["dim"]
    n_heads     = bcfg.get("n_heads", 4)
    d_ff        = bcfg.get("d_ff", 512)

    if name == "rnn":
        return VanillaRNN(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "lstm":
        return LSTMBaseline(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "rnn_attention":
        return RNNWithAttention(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "lstm_attention":
        return LSTMWithAttention(vocab_size, embed_dim, hidden_size, n_layers)
    elif name == "transformer":
        seq_len = config["data"]["seq_len"]
        return TransformerBaseline(vocab_size, embed_dim, n_layers, n_heads, d_ff, seq_len)
    else:
        raise ValueError(f"Unknown baseline: {name}")
