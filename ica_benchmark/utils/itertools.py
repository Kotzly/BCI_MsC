from itertools import product
from types import GeneratorType
from collections.abc import Iterable
import pandas as pd
import numpy as np


# ITERABLES_TYPES = (list, tuple, GeneratorType, product, chain)
# [TODO] str is iterable, but not one we want here
ITERABLES_TYPES = Iterable


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def insideout_group_iterator(split_kwargs_dicts, d=None, level=None):

    level = level or len(split_kwargs_dicts) - 1

    for inner_d in product_dict(**split_kwargs_dicts[level]):
        if level == 0:
            yield {**d, **inner_d}
        else:
            yield group_iterator(split_kwargs_dicts, d={**d, **inner_d}, level=level - 1)


def group_iterator(split_kwargs_dicts, d=None, level=None):

    level = level or 0
    d = d or dict()

    for inner_d in product_dict(**split_kwargs_dicts[level]): 
        if level == (len(split_kwargs_dicts) - 1):
            yield {**d, **inner_d}
        else:
            yield group_iterator(split_kwargs_dicts, d={**d, **inner_d}, level=level + 1)


def constrained_group_iterator(split_kwargs_dicts, d=None, level=None, constraining_function=None, level_idx_dict=None):
    constraining_function = constraining_function or (lambda l, i, kwargs: (kwargs, True))
    level = level or 0
    level_idx_dict = level_idx_dict or dict()
    d = d or dict()

    for idx, inner_d in enumerate(product_dict(**split_kwargs_dicts[level])):
        level_idx_dict[level] = idx

        kwargs = {**d, **inner_d}
        kwargs, valid = constraining_function(level, level_idx_dict, kwargs)
        if not valid:
            continue

        if level == (len(split_kwargs_dicts) - 1):
            yield kwargs
        else:
            yield constrained_group_iterator(
                split_kwargs_dicts,
                d=kwargs,
                level=level + 1,
                constraining_function=constraining_function,
                level_idx_dict=level_idx_dict
            )


def apply_to_iterator(iterator, fn):
    for item in iterator:
        if isinstance(item, ITERABLES_TYPES):
            yield apply_to_iterator(item, fn)
        else:
            yield fn(item)


def unpack_deep_iterable(deep_iterable):
    # Keep levels as a nested list
    if isinstance(deep_iterable, (GeneratorType, tuple, list)):
        # If deep_iterable is iterable, just make sure that if it is a generator that it is iterated
        deep_iterable = list(deep_iterable)
        return [
            unpack_deep_iterable(shallow_iterable)
            for shallow_iterable in deep_iterable
        ]
    else:
        return deep_iterable


def flatten_deep_iterable(deep_iterable):
    # Returns a flat iterator of all items that are not in ITERABLES_TYPES inside deep_iterable
    for item in deep_iterable:
        if isinstance(item, ITERABLES_TYPES):
            for nested_item in flatten_deep_iterable(item):
                yield nested_item
        else:
            yield item


def __create_splitter_constraint_fn(splitter, uids):
    # [TODO] Change this to be generic
    # Creates a dataframe with columns fold, uid and group (train or test)
    # It is only used to later check if volunteer with uid is in train or test for each fold.
    uids = np.array(uids)
    splits_dfs = list()
    for fold, splits_idx in enumerate(splitter.split(uids)):
        for group_i, group_uid_idx in enumerate(splits_idx):
            split_uids = uids[group_uid_idx]
            split_df = pd.DataFrame()
            split_df["uid"] = split_uids
            split_df["fold"] = fold
            split_df["group"] = group_i
            splits_dfs.append(split_df)

    split_df = pd.concat(splits_dfs, axis=0)

    def my_constraint_fn(level, level_idx_dict, kwargs):
        kwargs = {**kwargs}

        if ("group" not in kwargs) or ("uid" not in kwargs):
            return kwargs, True

        r = (
            kwargs["uid"] in
            split_df[(split_df.group == kwargs["group"]) & (split_df.fold == kwargs["fold"])].uid.to_numpy()
        )
        kwargs.pop("group")
        kwargs.pop("fold")

        return kwargs, r

    return my_constraint_fn
