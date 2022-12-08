# Copyright (c) Open-MMLab. All rights reserved.
import io
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import torch
import torchvision
from torch.optim import Optimizer
from torch.utils import model_zoo
from torch.nn import functional as F

from scipy import interpolate
import numpy as np
import math

def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None,
                    patch_padding='pad',
                    ):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        patch_padding (str): 'pad' or 'bilinear' or 'bicubic', used for interpolate patch embed from 14x14 to 16x16

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if 'patch_embed.proj.weight' in state_dict:
        proj_weight = state_dict['patch_embed.proj.weight']
        orig_size = proj_weight.shape[2:]
        current_size = model.patch_embed.proj.weight.shape[2:]
        padding_size = current_size[0] - orig_size[0]
        padding_l = padding_size // 2
        padding_r = padding_size - padding_l
        if orig_size != current_size:
            if 'pad' in patch_padding:
                proj_weight = torch.nn.functional.pad(proj_weight, (padding_l, padding_r, padding_l, padding_r))
            elif 'bilinear' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bilinear', align_corners=False)
            elif 'bicubic' in patch_padding:
                proj_weight = torch.nn.functional.interpolate(proj_weight, size=current_size, mode='bicubic', align_corners=False)
            state_dict['patch_embed.proj.weight'] = proj_weight

    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

