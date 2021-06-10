import numpy as np
from mne.time_frequency import tfr_array_multitaper, psd_array_multitaper

DEFAULT_FREQUENCIES = np.linspace(3, 30, 10)

DEFAULT_TRF_KWARGS = dict(
    sfreq=250.0,
    freqs=DEFAULT_FREQUENCIES, 
    n_cycles=7.0,
    zero_mean=True,
    time_bandwidth=4,
    decim=1,
    output='power',
    n_jobs=4,
    verbose="CRITICAL",
)

DEFAULT_PSD_KWARGS = dict(
    sfreq=250.0,
    fmin=0,
    fmax=np.inf,
    bandwidth=None,
    n_jobs=4,
    verbose="CRITICAL",
)

def psd_multitaper(
    arr,
    **mne_kwargs
    ):

    if not mne_kwargs:
        mne_kwargs = DEFAULT_PSD_KWARGS

    psd, freqs = psd_array_multitaper(
        arr.T,
        **mne_kwargs
    )
    psd = np.expand_dims(psd.T, axis=0)

    return psd, freqs


def psd_feature_transform(x, freqs=DEFAULT_FREQUENCIES, bandwidth=3, mne_kwargs=None):
    if mne_kwargs is None:
        mne_kwargs = DEFAULT_PSD_KWARGS
    psd, psd_freqs = psd_multitaper(x, **mne_kwargs)
    feature_vector = list()
    for freq in freqs:
        top_freq = freq + bandwidth / 2
        bot_freq = freq - bandwidth / 2
        selected_freqs = np.bitwise_and(psd_freqs >= bot_freq, psd_freqs <= top_freq)
        feature = psd[:, selected_freqs, :].mean(axis=1)
        feature_vector.append(feature)
    
    features = np.concatenate(feature_vector, axis=0).flatten()
    return features
        
def tfr_feature_transform(x, mne_kwargs=None):
    if mne_kwargs is None:
        mne_kwargs = DEFAULT_TRF_KWARGS
    return tfr_multitaper(
        x,
        epochs_mode=False,
        feature_format=None,
        **mne_kwargs
    )

def tfr_multitaper(
    arr,
    epochs_mode=False,
    feature_format=None,
    **mne_kwargs
):
    """
    Extract the Multitaper Time-Frequency spectogram of an array.
    
    Parameters
    ----------
    arr : np.ndarray
        An (n_times, n_channels) shaped array of EEG data. If epochs_mode is True, the shape must be (n_epochs, n_times, n_channels).
    feature_format : 
    epochs_mode : bool
        Whether the input array has the epochs dimension.
    **mne_kwargs
        Other keyword arguments that will be passed to mne.time_frequency.tfr_array_multitaper
    
    Returns
    -------
    tfr_psd : np.ndarray
        The Time-frequency spectogram of `arr`.
        If feature_format is True, the shape is (n_epochs, n_times, n_chans * n_freqs).
        If feature_format is False, the shape is (n_epochs, n_times, n_chans, n_freqs).

    """
    temp_mne_kwargs = DEFAULT_TRF_KWARGS.copy()
    temp_mne_kwargs.update(mne_kwargs)
    mne_kwargs = temp_mne_kwargs

    # arr is (n_times, n_channels)
    if not epochs_mode:
        assert arr.ndim == 2, "The input array must be of shape (n_times, n_channels)"
        # to (n_channels, n_times)
        arr = np.expand_dims(arr.T, axis=0)
    else:
        assert arr.ndim == 3, "The input array must be of shape (n_epochs, n_times, n_channels)"
        # to (n_epochs, n_channels, n_times)
        arr = arr.transpose(0, 2, 1)
    
    # input (n_epochs, n_channels, n_times)
    tfr_psd = tfr_array_multitaper(
        arr,
        **mne_kwargs
    )
    # output = (n_epochs, n_chans, n_freqs, n_times)
    
    if feature_format is None:
        return tfr_psd
    
    n_epochs, n_chans, n_freqs, n_times = tfr_psd.shape
    if feature_format:
        #(n_epochs, n_chans, n_freqs, n_times) -> (size, features)
        tfr_psd = tfr_psd\
            .transpose(0, 3, 1, 2)\
            .reshape(n_epochs, n_times, n_chans * n_freqs)[:, -n_times:, :]
            
    else:
        tfr_psd = tfr_psd.transpose(0, 3, 1, 2)[:, :, :, -n_times:]
        
    tfr_psd = tfr_psd.squeeze() if (n_epochs == 1) else tfr_psd
    
    return tfr_psd
