from .base import CorticalColumn
from .simple_ei import SimpleEIColumn, BatchedSimpleEIColumns
from .layered import LayeredColumn, BatchedLayeredColumns


def get_column(config: dict):
    """Factory: create a single CorticalColumn from config (used in tests)."""
    model = config["column"]["model"]
    if model == "simple_ei":
        return SimpleEIColumn(config)
    elif model == "layered":
        return LayeredColumn(config)
    else:
        raise ValueError(f"Unknown column model: {model}")


def get_batched_columns(config: dict, n_cols: int):
    """Factory: create a single batched module for all columns."""
    model = config["column"]["model"]
    if model == "simple_ei":
        return BatchedSimpleEIColumns(config, n_cols)
    elif model == "layered":
        return BatchedLayeredColumns(config, n_cols)
    else:
        raise ValueError(f"Unknown column model: {model}")
