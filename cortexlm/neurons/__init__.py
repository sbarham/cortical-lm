from .base import NeuronPopulation
from .rate import RateNeurons
from .rate_adex import RateAdExNeurons
from .batched import BatchedNeuronPop
from .utils import init_lognormal_taus


def get_neuron_population(config: dict, n_neurons: int, device=None):
    """Factory: instantiate neuron population from config."""
    model = config["neuron"]["model"]
    if model == "rate":
        return RateNeurons(n_neurons, config)
    elif model == "rate_adex":
        return RateAdExNeurons(n_neurons, config)
    elif model == "lif":
        from .lif import LIFNeurons
        return LIFNeurons(n_neurons, config)
    else:
        raise ValueError(f"Unknown neuron model: {model}")
