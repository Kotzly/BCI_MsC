import pandas as pd
from sklearn.model_selection import KFold
from warnings import warn, filterwarnings
import numpy as np
from mne import Epochs
import mne
from ica_benchmark.io.load import Dataset, OpenBMI_Dataset
from pathlib import Path
from ica_benchmark.utils.itertools import group_iterator, constrained_group_iterator
from typing import Union, List, Dict, Any, Optional
from sklearn.model_selection import BaseCrossValidator
from copy import copy


class Split:
    def __init__(self, kwarg_dict_list):
        self.kwargs_list = kwarg_dict_list

    def to_dataframe(self):
        return pd.DataFrame.from_records(self.kwargs_list)

    def __repr__(self):
        dict_reps = [str(d) for d in self.kwargs_list]
        return "Split({})".format(",".join(dict_reps))

    def __getitem__(self, k):
        values = [kwargs[k] for kwargs in self.kwargs_list]
        return values

    def load_epochs(self, dataset, concatenate=True, **load_kwargs):

        epochs = [
            dataset.load_subject(
                kwargs["uid"], **remove_key(kwargs, "uid"), **load_kwargs
            )[0]
            for kwargs in self.kwargs_list
        ]
        if concatenate:
            epochs = mne.concatenate_epochs(epochs)

        return epochs


def remove_key(d, k):
    """
    Removes a key from a dictionary and returns a copy of the modified dictionary.

    Parameters
    ----------
    d : dict
        The dictionary from which the key will be removed.
    k : object
        The key to be removed.

    Returns
    -------
    d : dict
        Copy of the input dictionary with the specified key removed.
    """
    d = copy(d)
    d.pop(k)
    return d


def make_epochs_splits_indexes(
    arr: Union[np.ndarray, Epochs],
    n_splits: int = 2,
    sizes: List[float] = None,
    shuffle: bool = False,
    seed: int = None,
):
    """Given an array, return the indexes fot splitting it into smaller sizes.
    For instance `make_epochs_splits_indexes(rand(100), sizes=[.2, .3, .5])` returns three
    indexes lists, with 20%, 30% and 50% of all indexes.
    """
    assert np.sum(sizes) == 1.0, "sizes must sum to 1!"

    if not isinstance(arr, (Epochs,)):
        arr = np.array(arr)

    if seed:
        np.random.seed(seed)

    n = len(arr.events) if isinstance(arr, Epochs) else len(arr)

    sizes = sizes or [1 / n_splits] * n_splits

    sizes = np.cumsum([int(size * n) for size in sizes[:-1]])
    idx = np.arange(n)

    if shuffle:
        np.random.shuffle(idx)

    indexes = np.split(idx, sizes)
    return indexes


def make_epochs_splits(
    arr: Union[np.ndarray, Epochs],
    n_splits: int = 2,
    sizes: Optional[List[float]] = None,
    shuffle: bool = False,
    seed: int = None,
):
    """
    Splits an array an Epochs object smaller sized objects.

    Parameters
    ----------
    arr : List[Union[np.ndarray, Epochs]]
        Array or list of `np.ndarray` or `Epochs` instances to be split.
    n_splits : int, optional
        Number of splits. Default is 2.
    sizes : Optional[List[int]], optional
        Sizes of each split.
    shuffle : bool, optional
        Whether to shuffle the input before splitting. Default is False.
    seed : Optional[int], optional
        Seed for shuffling the input.

    Returns
    -------
    arrs : List[Union[np.ndarray, Epochs]]
        List of arrays or lists resulting from the splitting of the input `arr`.
    """
    indexes = make_epochs_splits_indexes(
        arr, n_splits=n_splits, sizes=sizes, shuffle=shuffle, seed=seed
    )
    arrs = [arr[idx] for idx in indexes]
    return arrs


def splits_from_group_iterator(group_iterator_instance):
    """
    Generate Splits from a group iterator instance

    Parameters
    ----------
    group_iterator_instance : Iterator
        Instance of the iterator that contains the splits information.
        Preferably created with group_iterator`

    Returns
    -------
    splits : List[Split]
        List of generated splits
    """
    for iteration_splits_kwargs in group_iterator_instance:
        yield [
            Split([dict(**split_kwargs) for split_kwargs in splits_kwargs_list])
            for splits_kwargs_list in iteration_splits_kwargs
        ]


def split_group_iterator(split_kwargs_dicts: List[Dict[str, List[Any]]]):
    """
    Generate Splits from a list of kwargs.

    Parameters
    ----------
    split_kwargs_dicts : List[Dict[str, List[Any]]]]
        List of kwargs dictionaries.

    Returns
    -------
    splits_iterator : iterator
        Iterator for splits. Is a three-depth list.
    """
    splits_iterator = splits_from_group_iterator(group_iterator(split_kwargs_dicts))
    return splits_iterator


def create_split_group_iterator(
    outer_split_kwargs=None, inner_split_kwargs=None, merge_kwargs=None
):
    """Create a 3-level (in depth) iterator.
    Outer split kwargs refers to kwargs common in the outmost level.
    Inner split kwargs refers to kwargs common in the mid-level, and can be interpreted as the train and test sets.
    Merge kwargs refers to kwargs common for a set.

    Example

    create_split_group_iterator(
        dict(uid=[1, 2]),
        dict(session=[1, 2]),
        dict(run=[1, 2]
    )
    returns
    [
        [
            "Split({'uid': 1, 'session': 1, 'run': 1},{'uid': 1, 'session': 1, 'run': 2})",
            "Split({'uid': 1, 'session': 2, 'run': 1},{'uid': 1, 'session': 2, 'run': 2})"
        ],
        [
            "Split({'uid': 2, 'session': 1, 'run': 1},{'uid': 2, 'session': 1, 'run': 2})",
            "Split({'uid': 2, 'session': 2, 'run': 1},{'uid': 2, 'session': 2, 'run': 2})"
        ]
    ]
    """
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


def create_splitter_constraint_fn(splitter: BaseCrossValidator, uids: List[str]):
    """
    Creates a constraint function that checks if a volunteer with a specific UID is in the training or testing group for a particular fold.

    Parameters
    ----------
    splitter : object
        An object with a `split` method that returns indices for splitting the data into folds.
    uids : list
        List of unique IDs for each volunteer.

    Returns
    -------
    my_constraint_fn : function
        Constraint function that takes in level, level_idx_dict, kwargs and returns kwargs, r. `r` is True if the UID specified in `kwargs` is in the training or testing group for the fold specified in `kwargs`.
    """

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
            kwargs["uid"]
            in split_df[
                (split_df.group == kwargs["group"]) & (split_df.fold == kwargs["fold"])
            ].uid.to_numpy()
        )
        kwargs.pop("group")
        kwargs.pop("fold")

        return kwargs, r

    return my_constraint_fn


def kfold_split_group_iterator(
    splitter: BaseCrossValidator, uids: List[str], n_groups: int = 2
):
    """Util function for creating a split iterator of a KFold splitter."""

    kfold_iterable = constrained_group_iterator(
        [
            dict(fold=np.arange(splitter.get_n_splits())),
            dict(group=np.arange(n_groups)),
            dict(uid=uids),
        ],
        constraining_function=create_splitter_constraint_fn(splitter, uids),
    )
    for iteration_splits_kwargs in kfold_iterable:
        yield [
            Split([dict(**split_kwargs) for split_kwargs in splits_kwargs_list])
            for splits_kwargs_list in iteration_splits_kwargs
        ]


class Splitter:
    """
    This class is responsible to return a fold iterator for a dataset in many ways:
        - inter_subject
        - inter_session
        - intra_session_intra_run
        - intra_session_inter_run
        - intra_session_intra_run_merged

    [TODO] Make intra run protocols use "splitter" for splitting train and test

    Parameters
    ----------
    dataset : Dataset
        The dataset to be used for splitting.
    uids : List[str]
        List of unique IDs of the samples in the dataset.
    sessions : List[int]
        List of session IDs corresponding to each sample.
    runs : List[int]
        List of run IDs corresponding to each sample.
    load_kwargs : Dict, optional
        Keyword arguments for loading the dataset, by default None.
    splitter : Optional[BaseCrossValidator], optional
        Object responsible for splitting the dataset for the inter subject protocol, by default None.
    intra_session_shuffle : bool, optional
        Whether to shuffle the samples within each session, by default False.
    fold_sizes : List[float], optional
        List of float values representing the size of each fold, by default None.
    seed : Optional[int], optional
        Seed for random number generator, by default None.
    """

    # [TODO] Make intra run protocols use "splitter" for splitting train and test

    SESSION_KWARGS = dict(intra=dict(), inter=dict())
    VALID_MODES = [
        "inter_subject",
        "inter_session",
        "intra_session_intra_run",
        "intra_session_inter_run",
        "intra_session_intra_run_merged",
    ]

    def default_splitter(self):
        splitter = KFold(4)
        warn("Using default splitter: " + str(splitter))
        return splitter

    def __init__(
        self,
        dataset: Dataset,
        uids: List[str],
        sessions: List[int],
        runs: List[int],
        load_kwargs: Dict = None,
        splitter: Optional[BaseCrossValidator] = None,
        intra_session_shuffle: bool = False,
        fold_sizes: List[float] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.uids = uids
        self.sessions = sessions
        self.runs = runs
        self.load_kwargs = load_kwargs or load_kwargs
        self.splitter = splitter or self.default_splitter()
        self.intra_session_shuffle = intra_session_shuffle
        self.fold_sizes = fold_sizes
        self.seed = seed

    def validate_config(self, mode):

        fold_sizes = self.fold_sizes
        assert (
            mode in self.VALID_MODES
        ), "Please choose one mode among the following: {}".format(
            ", ".join(self.VALID_MODES)
        )
        if mode == "inter_subject":
            if fold_sizes is not None:
                warn(
                    "You are using the inter_subject mode, so the fold_sizes argument will not be used"
                )
        elif mode == "inter_session":
            if len(self.runs) > 1:
                warn(
                    "You are using inter session protocol with more than one run. All runs from each session will be concatenated and yielded in different steps."
                )
        elif mode == "intra_session_inter_run":
            if len(self.runs) == 1:
                warn(
                    "You are using an intra session protocol, splitting by run, but only passed one run. The splitter can only yield one epoch at time (from the only run you passed as argument)"
                )
        elif mode in ("intra_session_intra_run", "intra_session_intra_run_merged"):
            if fold_sizes is None:
                warn(
                    "You are using intra session intra run protocol with no fold sizes. The splitter will only yield one epoch at time"
                )

    def inter_subject(self, splitter=None, n_groups=2):
        splitter = splitter or self.splitter
        fold_iterable = constrained_group_iterator(
            [
                dict(fold=np.arange(splitter.get_n_splits())),
                dict(group=np.arange(n_groups)),
                dict(uid=self.uids, session=self.sessions, run=self.runs),
            ],
            constraining_function=create_splitter_constraint_fn(splitter, self.uids),
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
        """Yields the fold splits and epochs for the specified mode.

        Parameters
        ----------
        mode : str
            The mode to split the data into folds.

        Returns
        -------
        Generator
            Yields the fold splits and epochs for the specified mode.
        """
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
        return split_fn()

    def load_from_splits(self, splits: Split, fold_sizes=None):
        fold_sizes = fold_sizes or self.fold_sizes
        splits_epochs = [
            split.load_epochs(self.dataset, **self.load_kwargs) for split in splits
        ]
        if fold_sizes is not None:
            assert (
                len(splits_epochs) == 1
            ), "You passed fold_sizes={} but there in more than one split".format(
                fold_sizes
            )
            splits_epochs = make_epochs_splits(
                splits_epochs[0],
                sizes=fold_sizes,
                shuffle=self.intra_session_shuffle,
                seed=self.seed,
            )
        else:
            if len(splits_epochs) == 1:
                warn(
                    "You are using folds with only one split, and yet you did not use the fold_sizes argument."
                )

        return splits_epochs


if __name__ == "__main__":

    mne.set_log_level(False)
    filterwarnings("ignore", category=RuntimeWarning)

    openbmi_dataset_folderpath = Path("/home/paulo/Documents/datasets/OpenBMI/edf/")
    dataset = OpenBMI_Dataset(openbmi_dataset_folderpath)
    fold_sizes = None
    splitter = Splitter(
        dataset,
        uids=dataset.list_uids()[:4],
        sessions=dataset.SESSIONS,
        runs=dataset.RUNS,
        load_kwargs=dict(reject=False, tmin=0, tmax=1),
        splitter=KFold(4),
        intra_session_shuffle=False,
        fold_sizes=fold_sizes,
    )

    splits_iterable = splitter.yield_splits_epochs(mode="inter_subject")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_splits(fold_splits, fold_sizes=fold_sizes)
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    print("Changing splitter uids to ['1']")
    splitter.uids = ["1"]

    splits_iterable = splitter.yield_splits_epochs(mode="inter_session")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_splits(fold_splits, fold_sizes=None)
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(mode="intra_session_inter_run")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_splits(fold_splits, fold_sizes=None)
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(mode="intra_session_intra_run")
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_splits(fold_splits, fold_sizes=[0.75, 0.25])
        print(f"\tEpochs {epochs}")
        print()
        del epochs

    splits_iterable = splitter.yield_splits_epochs(
        mode="intra_session_intra_run_merged"
    )
    for i, fold_splits in enumerate(splits_iterable):
        print(f"Fold {i}")
        print(f"\tSplits {fold_splits}")
        epochs = splitter.load_from_splits(fold_splits, fold_sizes=[0.75, 0.25])
        print(f"\tEpochs {epochs}")
        print()
        del epochs
