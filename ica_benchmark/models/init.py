from torch import nn
import torch


def glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.
    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def temporal_filter_init(weight):
    with torch.no_grad():
        new_weights = torch.zeros_like(weight)
        new_weights[..., -1] = 1
        weight.copy_(new_weights)


def spatial_filter_init(weight):
    with torch.no_grad():
        n_channels = weight.size(2)
        new_weights = torch.ones_like(weight) / n_channels
        weight.copy_(new_weights)
