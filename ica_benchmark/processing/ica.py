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
from coroica import UwedgeICA, CoroICA
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level("ERROR")


def _get_kwargs(m, is_extended=False):
    if is_extended:
        return dict(method=m, fit_params=dict(extended=True))
    return dict(method=m)


_ica_kwargs_dict = {
    "fastica": _get_kwargs("fastica"),
    "infomax": _get_kwargs("infomax"),
    "picard": _get_kwargs("picard"),
    "ext_infomax": _get_kwargs("infomax", is_extended=True),
    "ext_picard": _get_kwargs("picard", is_extended=True),
}


_coro_kwargs_dict = {
    "sobi": dict(partitionsize=int(10 ** 6), timelags=list(range(1, 101))),  # [TODO] resolve nan issue
    "choi_var": dict(),
    "choi_vartd": dict(timelags=[1, 2, 3, 4, 5]),  # [TODO] resolve nan issue 
    "choi_td": dict(instantcov=False, timelags=[1, 2, 3, 4, 5]),  # [TODO] resolve nan issue
    "coro": dict(),  # [TODO] resolve nan issue
}


_jade_kwargs_dict = {"jade": dict()}


_all_methods = (
    list(_ica_kwargs_dict) + list(_coro_kwargs_dict) + list(_jade_kwargs_dict)
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

    def _fit(self, data, fit_type):
        """Aux function."""
        random_state = check_random_state(self.random_state)
        n_channels, n_samples = data.shape
        self._compute_pre_whitener(data)
        data = self._pre_whiten(data)

        pca = _PCA(n_components=self._max_pca_components, whiten=True)
        data = pca.fit_transform(data.T)
        use_ev = pca.explained_variance_ratio_
        n_pca = self.n_pca_components
        if isinstance(n_pca, float):
            n_pca = int(_exp_var_ncomp(use_ev, n_pca)[0])
        elif n_pca is None:
            n_pca = len(use_ev)
        assert isinstance(n_pca, (int, np.int_))

        # If user passed a float, select the PCA components explaining the
        # given cumulative variance. This information will later be used to
        # only submit the corresponding parts of the data to ICA.
        if self.n_components is None:
            # None case: check if n_pca_components or 0.999999 yields smaller
            msg = "Selecting by non-zero PCA components"
            self.n_components_ = min(n_pca, _exp_var_ncomp(use_ev, 0.999999)[0])
        elif isinstance(self.n_components, float):
            self.n_components_, ev = _exp_var_ncomp(use_ev, self.n_components)
            if self.n_components_ == 1:
                raise RuntimeError(
                    "One PCA component captures most of the "
                    f"explained variance ({100 * ev}%), your threshold "
                    "results in 1 component. You should select "
                    "a higher value."
                )
            msg = "Selecting by explained variance"
        else:
            msg = "Selecting by number"
            self.n_components_ = _ensure_int(self.n_components)
        # check to make sure something okay happened
        if self.n_components_ > n_pca:
            ev = np.cumsum(use_ev)
            ev /= ev[-1]
            evs = 100 * ev[[self.n_components_ - 1, n_pca - 1]]
            raise RuntimeError(
                f"n_components={self.n_components} requires "
                f"{self.n_components_} PCA values (EV={evs[0]:0.1f}%) but "
                f"n_pca_components ({self.n_pca_components}) results in "
                f"only {n_pca} components (EV={evs[1]:0.1f}%)"
            )
        logger.info("%s: %s components" % (msg, self.n_components_))

        # the things to store for PCA
        self.pca_mean_ = pca.mean_
        self.pca_components_ = pca.components_
        self.pca_explained_variance_ = pca.explained_variance_
        del pca
        # update number of components
        self._update_ica_names()
        if self.n_pca_components is not None and self.n_pca_components > len(
            self.pca_components_
        ):
            raise ValueError(
                f"n_pca_components ({self.n_pca_components}) is greater than "
                f"the number of PCA components ({len(self.pca_components_)})"
            )

        # take care of ICA
        sel = slice(0, self.n_components_)
        if self.method == "fastica":
            from sklearn.decomposition import FastICA

            ica = FastICA(whiten=False, random_state=random_state, **self.fit_params)
            ica.fit(data[:, sel])
            self.unmixing_matrix_ = ica.components_
            self.n_iter_ = ica.n_iter_
        elif self.method in ("infomax", "extended-infomax"):
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
                return_n_iter=True,
                random_state=random_state,
                **self.fit_params,
            )
            self.unmixing_matrix_ = W
            self.n_iter_ = n_iter + 1  # picard() starts counting at 0
            del _, n_iter
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

        assert self.unmixing_matrix_.shape == (self.n_components_,) * 2
        norms = self.pca_explained_variance_
        stable = norms / norms[0] > 1e-6  # to be stable during pinv
        norms = norms[: self.n_components_]
        if not stable[self.n_components_ - 1]:
            max_int = np.where(stable)[0][-1] + 1
            warn(
                f"Using n_components={self.n_components} (resulting in "
                f"n_components_={self.n_components_}) may lead to an "
                f"unstable mixing matrix estimation because the ratio "
                f"between the largest ({norms[0]:0.2g}) and smallest "
                f"({norms[-1]:0.2g}) variances is too large (> 1e6); "
                f"consider setting n_components=0.999999 or an "
                f"integer <= {max_int}"
            )
        norms = np.sqrt(norms)
        norms[norms == 0] = 1.0
        self.unmixing_matrix_ /= norms  # whitening
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
    set(
        mne.preprocessing.ica._KNOWN_ICA_METHODS
    ) | 
    set(get_all_methods())
)
