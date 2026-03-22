from .base import HippocampalModule
from .none import NullHippocampus
from .modern_hopfield import ModernHopfieldHippocampus


def get_hippocampus(config: dict, n_columns: int, n_l5e: int):
    model = config["hippocampus"]["model"]
    if model == "none":
        return NullHippocampus(config, n_columns, n_l5e)
    elif model == "modern_hopfield":
        return ModernHopfieldHippocampus(config, n_columns, n_l5e)
    else:
        raise ValueError(f"Unknown hippocampus model: {model}")
