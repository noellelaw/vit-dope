import warnings

import mmcv
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed
    seed = np.random.randint(2**31)
    return seed


def train_model(model,
                dataset,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.
    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """

    # TODO:
    # - Get dataset
    # - Get data loaders
    # - Finish training code
    optimizer = clip_grad_norm_( 
        max_norm = 1.,
        norm_type = 2
        )