# https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py#L127-L154
import scipy.signal
import numpy as np


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.

    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.
    Returns
    -------
    is_stable: bool
        Filter is stable or not.
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.

    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)


def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass"
    )
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def bandpass_cnt(
    data, low_cut_hz=4., high_cut_hz=38., fs=250., filt_order=3, axis=1, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Channels x times
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter
    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    n_channels, n_times = data.shape
    assert n_channels < n_times

    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz is None or high_cut_hz == fs / 2.0
    ):
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz is None:
        return lowpass_cnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz is None or high_cut_hz >= (fs / 2.0):
        return highpass_cnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed
