import torch


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def to_tensor(*args, device="cpu"):
    return (
        torch.unsqueeze(
            torch.from_numpy(arg).to(device),
            1
        )
        for arg
        in args
    )


# https://github.com/TNTLFreiburg/braindecode/blob/d9feb5c6cfcd203fa8daa79ccd3217712714f330/braindecode/mne_ext/signalproc.py#L75
def apply_raw(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.

    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.
    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.
    """
    new_data = func(raw.get_data())
    new_raw = copy_as_raw(raw, new_data)
    return new_raw


def copy_as_raw(raw, data):
    """Aux method."""
    # merge copied instance and picked data with sources
    data_ = data
#     assert data_.shape[1] == stop - start

    preloaded = raw.preload
    if preloaded:
        # get data and temporarily delete
        data = raw._data
        raw.preload = False
        del raw._data
    # copy and crop here so that things like annotations are adjusted
    out = raw.copy()

    out._data = data_
    out._first_samps = [out.first_samp]
    out._last_samps = [out.last_samp]
    out._filenames = [None]
    out.preload = True
    out._projector = None
    return out
