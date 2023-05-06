import torch


def norm(filter_weights, dim=2):
    norms = torch.norm(filter_weights, dim=dim, p=2, keepdim=True)
    return norms.mean()


def apply_max_norm(weights, max_value=1., eps=1e-5, dim=2):
    norms = norm(weights, dim=dim)
    desired = torch.clamp(norms, 0, max_value)
    new_weights = weights * (desired / (eps + norms))
    weights.copy_(new_weights)
