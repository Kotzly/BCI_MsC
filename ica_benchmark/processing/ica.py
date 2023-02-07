import mne
from mne.preprocessing.ica import (
    ICA,
    _exp_var_ncomp,
    check_random_state,
    _PCA,
    _ensure_int,
    logger,
    infomax,
    warn,
)
import numpy as np

from ica_benchmark.processing.jade import JadeICA
from ica_benchmark.processing.sobi import SOBI
from ica_benchmark.processing.whitening import Whitening
from sklearn.decomposition import PCA
from coroica import UwedgeICA, CoroICA
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level("ERROR")


def _get_kwargs(m, is_extended=None):
    if is_extended is None:
        return dict(method=m)
    return dict(method=m, fit_params=dict(extended=is_extended))


_ica_kwargs_dict = {
    "fastica": _get_kwargs("fastica"),
    "infomax": _get_kwargs("infomax"),
    "picard": _get_kwargs("picard", is_extended=False),
    "ext_infomax": _get_kwargs("infomax", is_extended=True),
    "picard_o": dict(method="picard", fit_params=dict(extended=True, ortho=True)),
    "whitening": _get_kwargs("whitening"),
    # "pca": _get_kwargs("pca")  # Disabled
}


_coro_kwargs_dict = {
    "sobi_coro": dict(partitionsize=int(10 ** 6), timelags=list(range(1, 101))),  # [TODO] resolve nan issue
    "choi_var": dict(),
    "choi_vartd": dict(timelags=[1, 2, 3, 4, 5]),  # [TODO] resolve nan issue
    "choi_td": dict(instantcov=False, timelags=[1, 2, 3, 4, 5]),  # [TODO] resolve nan issue
    "coro": dict(),  # [TODO] resolve nan issue
}


_jade_kwargs_dict = {"jade": dict()}


_sobi_kwargs_dict = {"sobi": dict(lags=100)}


_all_methods = list(
    {
        **_ica_kwargs_dict,
        **_coro_kwargs_dict,
        **_jade_kwargs_dict,
        **_sobi_kwargs_dict
    }
)


def create_gdf_obj(arr):
    if isinstance(arr, mne.io.Raw):
        return arr

    n_channels = arr.shape[1]
    info = mne.create_info(
        ch_names=["C" + str(x + 1) for x in range(n_channels)],
        ch_types=["eeg"] * n_channels,
        sfreq=SAMPLING_FREQ,
    )
    return mne.io.RawArray(arr.T, info)


class CustomICA(ICA):

    def transform(self, X, copy=True):
        if copy:
            X = X.copy()
        return self.get_sources(X)

    def _compute_pre_whitener(self, data):
        """Aux function."""
        data = self._do_proj(data, log_suffix='(pre-whitener computation)')
        if self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            info = self.info
            pre_whitener = np.empty([len(data), 1])
            for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
                pre_whitener[picks_] = np.std(data[picks_])
            if _contains_ch_type(info, "ref_meg"):
                picks_ = pick_types(info, ref_meg=True, exclude=[])
                pre_whitener[picks_] = np.std(data[picks_])
            if _contains_ch_type(info, "eog"):
                picks_ = pick_types(info, eog=True, exclude=[])
                pre_whitener[picks_] = np.std(data[picks_])
        else:
            pre_whitener, _ = compute_whitener(
                self.noise_cov, self.info, picks=self.info.ch_names)
            assert data.shape[0] == pre_whitener.shape[1]
        self.pre_whitener_ = pre_whitener

    def _fit(self, data, fit_type):
        """Aux function."""
        random_state = check_random_state(self.random_state)
        n_channels, n_samples = data.shape
        self._compute_pre_whitener(data)
        data = self._pre_whiten(data)

        # [TODO] Remove the PCA. The whitening step is important, but the PCA is not. To only do the ICA (plus the whitening), it is maybe necessary to remove the PCA
        # The code highly utilizes the PCA parameters, so it may be hard to remove it using the MNE code. One option could be sting the pca components to the identity matrix.

        # the things to store for PCA
        self.pca_mean_ = None
        self.pca_components_ = np.eye(n_channels)
        self.pca_explained_variance_ = np.ones(n_channels)

        if self.n_components is None:
            self.n_components_ = n_channels
        else:
            self.n_components_ = _ensure_int(self.n_components)

        # update number of components
        self._update_ica_names()

        sel = slice(0, self.n_components_)
        if self.method == "fastica":
            from sklearn.decomposition import FastICA
            ica = FastICA(whiten=False, random_state=random_state, **self.fit_params)
            ica.fit(data[:, sel])
            self.unmixing_matrix_ = ica.components_
            self.n_iter_ = ica.n_iter_
        elif self.method == "infomax":
            unmixing_matrix, n_iter = infomax(
                data[:, sel],
                random_state=random_state,
                return_n_iter=True,
                **self.fit_params,
            )
            self.unmixing_matrix_ = unmixing_matrix
            self.n_iter_ = n_iter
            del unmixing_matrix, n_iter
        elif self.method == "picard":
            from picard import picard
            _, W, _, n_iter = picard(
                data[:, sel].T,
                whiten=False,
                # MNE already centers data using the PCA
                centering=False,
                return_n_iter=True,
                random_state=random_state,
                **self.fit_params,
            )
            self.unmixing_matrix_ = W
            self.n_iter_ = n_iter + 1  # picard() starts counting at 0
            del _, n_iter
        elif self.method == "whitening":
            # [TODO] this doesn't seem right
            self.unmixing_matrix_ = np.eye(n_channels)
            self.n_iter_ = 1
        elif self.method == "pca":
            # [TODO] Disabled by now, enable in the future
            pca = PCA(self.n_components_)
            pca.fit(data.T)
            self.unmixing_matrix_ = pca
            self.n_iter_ = 1
        elif self.method in _coro_kwargs_dict:
            kwargs = _coro_kwargs_dict[self.method]
            coroica_constructor = UwedgeICA if self.method != "coro" else CoroICA
            coroica = coroica_constructor(n_components=self.n_components, **kwargs)
            coroica.fit(data[:, sel])
            self.unmixing_matrix_ = coroica.V_
            self.n_iter_ = coroica.n_iter_ + 1

        elif self.method in _jade_kwargs_dict:
            kwargs = _jade_kwargs_dict[self.method]
            jade_ica = JadeICA(self.n_components, **kwargs)
            jade_ica.fit(data[:, sel].T)
            self.unmixing_matrix_ = jade_ica.B
            self.n_iter_ = jade_ica.n_iter + 1

        elif self.method in _sobi_kwargs_dict:
            sobi_ica = SOBI(**_sobi_kwargs_dict[self.method])
            sobi_ica.fit(data[:, sel])
            self.unmixing_matrix_ = sobi_ica.W
            self.n_iter_ = sobi_ica.counter + 1

        assert self.unmixing_matrix_.shape == (self.n_components_,) * 2

        self._update_mixing_matrix()
        self.current_fit = fit_type


SAMPLING_FREQ = 250.0


def get_all_methods():
    return _all_methods


def get_ica_instance(method, n_components=None, **kwargs):
    if method in _ica_kwargs_dict:
        return CustomICA(n_components=n_components, **_ica_kwargs_dict[method], **kwargs)
    return CustomICA(n_components, method=method, **kwargs)


mne.preprocessing.ica._KNOWN_ICA_METHODS = list(
    set(mne.preprocessing.ica._KNOWN_ICA_METHODS) | set(get_all_methods())
)
