from .static import StaticSynapse
from .stp import STPSynapse


def get_synapse(n_pre_e: int, n_pre_i: int, n_post: int, config: dict, use_stp: bool = False):
    if use_stp:
        return STPSynapse(n_pre_e, n_pre_i, n_post, config)
    return StaticSynapse(n_pre_e, n_pre_i, n_post)
