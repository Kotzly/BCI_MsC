import pandas as pd
from sklearn.model_selection import KFold
from warnings import warn, filterwarnings
from itertools import product, chain
import numpy as np
from mne import Epochs
import mne
from types import GeneratorType
from ica_benchmark.io.load import OpenBMI_Dataset
from collections.abc import Iterable
from pathlib import Path


# ITERABLES_TYPES = (list, tuple, GeneratorType, product, chain)
# [TODO] str is iterable, but not one we want here
ITERABLES_TYPES = Iterable

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

def apply_to_iterator(iterator, fn):
    for item in iterator:
        if isinstance(item, ITERABLES_TYPES):
            yield apply_to_iterator(item, fn)
        else:
            yield fn(item)


def make_epochs_splits_indexes(arr, n=None, n_splits=2, sizes=None, shuffle=False, seed=1):
    if not isinstance(arr, (Epochs,)):
        arr = np.array(arr)

    np.random.seed(seed)
    if n is None:
        if isinstance(arr, Epochs):
            n = len(arr.events)
        else:
            n = len(arr)

    sizes = sizes or [1 / n_splits] * n_splits

    assert np.sum(sizes) == 1.

    sizes = np.cumsum(
        [0] + [int(size * n) for size in sizes]
    )

    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    slices = [slice(start, end) for start, end in zip(sizes[:-1], sizes[1:])]
    indexes = [idx[s] for s in slices]
    return indexes


def make_epochs_splits(arr, n=None, n_splits=2, sizes=None, shuffle=False, seed=1):
    indexes = make_epochs_splits_indexes(arr, n=n, n_splits=n_splits, sizes=sizes, shuffle=shuffle, seed=seed)
    arrs = [arr[idx] for idx in indexes]
    return arrs


def remove_key(d, k):
    return {
        key: value
        for key, value in d.items()
        if key != k
    }


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


def splits_from_group_iterator(group_iterator_instance):

    for iteration_splits_kwargs in group_iterator_instance:
        yield [
            Split(
                [
                    dict(
                        **split_kwargs
                    )
                    for split_kwargs in splits_kwargs_list
                ]
            )
            for splits_kwargs_list in iteration_splits_kwargs
        ]


def split_group_iterator(split_kwargs_dicts):
    return splits_from_group_iterator(
        group_iterator(split_kwargs_dicts)
    )


def create_split_group_iterator(outer_split_kwargs=None, inner_split_kwargs=None, merge_kwargs=None):
    outer_split_kwargs = outer_split_kwargs or dict()
    inner_split_kwargs = inner_split_kwargs or dict()
    merge_kwargs = merge_kwargs or dict()

    return split_group_iterator(
        [
            outer_split_kwargs,
            inner_split_kwargs,
            merge_kwargs,
        ]
    )


class Split():

    def __init__(self, kwarg_dict_list):
        self.kwargs_list = kwarg_dict_list

    def to_dataframe(self):
        return pd.DataFrame.from_records(self.kwargs_list)

    def __repr__(self):
        dict_reps = [str(d) for d in self.kwargs_list]
        return "Split({})".format(",".join(dict_reps))

    def load_epochs(self, dataset, **load_kwargs, concatenate=True):

        epochs = [
            dataset.load_subject(kwargs["uid"], **remove_key(kwargs, "uid"), **load_kwargs)[0]
            for kwargs in self.kwargs_list
        ]
        if concatenate:
            epochs = mne.concatenate_epochs(epochs)

        return epochs


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


def constrained_split_group_iterator(split_kwargs_dicts):

    for iteration_splits_kwargs in group_iterator(split_kwargs_dicts):
        yield [
            Split(
                [
                    dict(
                        **split_kwargs
                    )
                    for split_kwargs in splits_kwargs_list
                ]
            )
            for splits_kwargs_list in iteration_splits_kwargs
        ]


def create_splitter_constraint_fn(splitter, uids):
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


def kfold_split_group_iterator(splitter, uids, n_groups=2):
    
    kfold_iterable = constrained_group_iterator(
        [
            dict(fold=np.arange(splitter.get_n_splits())),
            dict(group=np.arange(n_groups)),
            dict(uid=uids),
        ],
        constraining_function=create_splitter_constraint_fn(splitter, uids)
    )
    for iteration_splits_kwargs in kfold_iterable:
        yield [
            Split(
                [
                    dict(
                        **split_kwargs
                    )
                    for split_kwargs in splits_kwargs_list
                ]
            )
            for splits_kwargs_list in iteration_splits_kwargs
        ]


class Splitter():

    SESSION_KWARGS = dict(intra=dict(), inter=dict())

    def default_splitter(self):
        splitter = KFold(4)
        warn("Using default splitter: " + str(splitter))
        return splitter

    def __init__(self, dataset, uids, sessions, runs, load_kwargs=None, splitter=None, unsafe=False, intra_session_shuffle=False, fold_sizes=None):
        self.dataset = dataset
        self.uids = uids
        self.sessions = sessions
        self.runs = runs
        self.load_kwargs = load_kwargs or load_kwargs
        self.splitter = splitter or self.default_splitter()
        self.intra_session_shuffle = intra_session_shuffle
        self.fold_sizes = fold_sizes

    def validate_config(self, mode):
        valid_modes = [
            "inter_subject",
            "inter_session",
            "intra_session_intra_run",
            "intra_session_inter_run",
            "intra_session_intra_run_merged"
        ]
        fold_sizes = self.fold_sizes
        assert mode in valid_modes, "Please choose one mode among the following: {}".format(", ".join(valid_modes))
        if mode == "inter_subject":
            if fold_sizes is not None:
                warn("You are using the inter_subject mode, so the fold_sizes argument will not be used")
        elif mode == "inter_session":
            if len(self.runs) > 1:
                warn("You are using inter session protocol with more than one run. All runs from each session will be concatenated and yielded in different steps.")
        elif (mode == "intra_session_inter_run"):
            if (len(self.runs) == 1):
                warn("You are using an intra session protocol, splitting by run, but only passed one run. The splitter can only yield one epoch at time (from the only run you passed as argument)")
        elif mode in ("intra_session_intra_run", "intra_session_intra_run_merged"):
            if fold_sizes is None:
                warn("You are using intra session intra run protocol with no fold sizes. The splitter will only yield one epoch at time")

    def inter_subject(self, splitter=None, n_groups=2):
        splitter = splitter or self.splitter
        fold_iterable = constrained_group_iterator(
            [
                dict(fold=np.arange(splitter.get_n_splits())),
                dict(group=np.arange(n_groups)),
                dict(uid=self.uids, session=self.sessions, run=self.runs),
            ],
            constraining_function=create_splitter_constraint_fn(splitter, self.uids)
        )

        return splits_from_group_iterator(fold_iterable)

    def inter_session(self):
        inter_session_iterator = create_split_group_iterator(
            dict(uid=self.uids),
            dict(session=self.sessions),
            dict(run=self.runs),
        )
        return inter_session_iterator

    def intra_session_inter_run(self):
        intra_session_inter_run_iterator = create_split_group_iterator(
            dict(uid=self.uids, session=self.sessions),
            dict(run=self.runs),
            dict(),
        )
        return intra_session_inter_run_iterator

    def intra_session_intra_run(self, merge=False):
        # intra_session_intra_run_merge
        # Duas runs mergidas, precisa separar por pct
        if merge:
            return self.intra_session_intra_run_merged()
        else:
            # Cada run em seu experiment, mas ainda precisa separar por pct
            intra_run_iterator = create_split_group_iterator(
                dict(uid=self.uids, session=self.sessions, run=self.runs),
                dict(),
                dict(),
            )
            return intra_run_iterator

    def intra_session_intra_run_merged(self, merge=False):
        intra_session_iterator = create_split_group_iterator(
            dict(uid=self.uids, session=self.sessions),
            dict(),
            dict(run=self.runs),
        )

        return intra_session_iterator

    def yield_splits_epochs(self, mode):

        split_fn_dict = dict(
            # Intra subject, inter session
            inter_session=self.inter_session,
            # Inter subject, will concatenate all sessions and runs
            inter_subject=self.inter_subject,
            # Intra subject, intra_session, inter run (will split runs)
            intra_session_inter_run=self.intra_session_inter_run,
            # Intra subject, intra_session, intra run (will split using fold sizes)
            intra_session_intra_run=self.intra_session_intra_run,
            # Intra subject, intra_session, intra run (will merge all runs and split using fold sizes)
            intra_session_intra_run_merged=self.intra_session_intra_run_merged,
        )

        split_fn = split_fn_dict[mode]
        for fold_splits in split_fn():

            # if (len(fold_splits) == 1) and (fold_sizes is None):
            #     warn("This splitter return only one split and you passed no fold sizes for intra splitting. Is this what you want?")
                
            yield fold_splits

    def load_from_split(self, splits, fold_sizes=None):
        fold_sizes = fold_sizes or self.fold_sizes
        splits_epochs = [
            split.load_epochs(self.dataset, **self.load_kwargs)
            for split in splits
        ]
        if fold_sizes is not None:
            assert len(splits_epochs) == 1, "You passed fold_sizes={} but there in more than one split".format(fold_sizes)
            splits_epochs = make_epochs_splits(
                splits_epochs[0],
                sizes=fold_sizes,
                shuffle=self.intra_session_shuffle
            )
        return splits_epochs


if __name__ == "__main__":

    mne.set_log_level(False)
    filterwarnings("ignore", category=RuntimeWarning)

    openbmi_dataset_folderpath = Path('/home/paulo/Documents/datasets/OpenBMI/edf/')
    dataset = OpenBMI_Dataset(openbmi_dataset_folderpath)
    fold_sizes = None
    splitter = Splitter(
        dataset,
        uids=dataset.list_uids()[:4],
        sessions=dataset.SESSIONS,
        runs=dataset.RUNS,
        load_kwargs=dict(
            reject=False,
            tmin=0,
            tmax=1
        ),
        splitter=KFold(4),
        intra_session_shuffle=False,
        fold_sizes=fold_sizes
    )

    splits_iterable = splitter.yield_splits_epochs(mode="inter_subject")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_split(fold_splits, fold_sizes=fold_sizes)
        print(f"\tEpochs {epochs}")
        print()
        del epochs
    exit()

    print("Changing splitter uids to ['1']")
    splitter.uids = ["1"]

    splits_iterable = splitter.yield_splits_epochs(mode="inter_session")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_split(fold_splits, fold_sizes=None)
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(mode="intra_session_inter_run")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_split(fold_splits, fold_sizes=None)
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(mode="intra_session_intra_run")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_split(fold_splits, fold_sizes=[.75, .25])
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(mode="intra_session_intra_run_merged")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_split(fold_splits, fold_sizes=[.75, .25])
        print(f"\tEpochs {epochs}")
        print()
        del epochs
