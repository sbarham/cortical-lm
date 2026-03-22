from .bptt import BPTTTrainer
from .eprop import EpropTrainer


def get_trainer(model, config, device=None, tokenizer=None):
    rule = config["learning"]["rule"]
    if rule == "bptt":
        return BPTTTrainer(model, config, device, tokenizer=tokenizer)
    elif rule == "eprop":
        return EpropTrainer(model, config, device)
    else:
        raise ValueError(f"Unknown learning rule: {rule}")
