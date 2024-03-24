from collections import namedtuple


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def cast_or_default(value, base_cls, default):
    if value is None:
        return default

    if isinstance(value, base_cls):
        return value

    return base_cls(value)


PoolParams = namedtuple("PoolParams", ["kernel_size", "stride"])
