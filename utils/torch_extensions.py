import torch
import numpy as np


def get_model_params_size(model: torch.nn.Module):
    """
    Get torch nn.Module and return number of parameters and size in MBs
    :param model: torch model (should derivate from torch.nn.Module)
    :return: params amount, size in MBs
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params, params * 32 / 1024 / 1024
