from .bptt import BPTTTrainer
from .eprop import EpropApproxTrainer, EpropTrainer, EpropHybridTrainer


def get_trainer(model, config, device=None, tokenizer=None):
    rule = config["learning"]["rule"]
    if rule == "bptt":
        return BPTTTrainer(model, config, device, tokenizer=tokenizer)
    elif rule == "eprop_approx":
        return EpropApproxTrainer(model, config, device)
    elif rule == "eprop":
        return EpropTrainer(model, config, device)
    elif rule == "eprop_hybrid":
        return EpropHybridTrainer(model, config, device)
    else:
        raise ValueError(f"Unknown learning rule: {rule}")
