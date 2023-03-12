from mne import BaseEpochs
from sklearn.base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import namedtuple
import matplotlib
from scipy.stats import pearsonr as pcc
from mne.io.constants import FIFF


adaptative_constants = namedtuple("adaptative_constants", ["a", "b", "sigmarn"])


def match_corr(eeg, eog):
    corr_m = list()
    for eeg_ch in eeg:
        line = list()
        for eog_ch in eog:
            line.append(
                pcc(eeg_ch, eog_ch)[0]
            )
        corr_m.append(line)
    corr_m = np.array(corr_m)
    max_corr_ch = np.argmax(np.abs(corr_m), axis=0)
    max_corr_m = corr_m[max_corr_ch, np.arange(len(eog))]
    return max_corr_ch, max_corr_m, corr_m


def windowed_correlation(x, y, window_size=32):
    i = 0
    correlations = list()
    n_ch, n_p = x.shape
    matched_channels, _, _ = match_corr(x, y)
    while i + 1 < n_p:
        correlations.append(
            [
                pcc(
                    x[ch, i:i + window_size],
                    y[idx, i:i + window_size]
                )[0]
                for idx, ch
                in enumerate(matched_channels)
            ]
        )
        i += window_size
    correlations = np.abs(np.array(correlations))
    return correlations


def match_corr_epochs(eeg, eog):
    assert eeg.shape[0] == eog.shape[0], (eeg.shape, eog.shape)
    assert eeg.shape[2] == eog.shape[2], (eeg.shape, eog.shape)

    eeg = eeg.transpose(1, 0, 2).reshape(eeg.shape[1], -1)
    eog = eog.transpose(1, 0, 2).reshape(eog.shape[1], -1)
    max_corr_ch, max_corr_m, corr_m = match_corr(eeg, eog)
    return max_corr_ch, max_corr_m, corr_m


def orthogonalize(w):
    d, v = np.linalg.eigh(w @ w.T)
    d = np.diag(d)
#     w = matrix_div(v, np.sqrt(d)) @ np.linalg.inv(v) @ w
    w = v @ np.sqrt(np.linalg.inv(d)) @ np.linalg.inv(v) @ w
    return w


def nonlinearity(x, n_sub=1):
    n_e = len(x)
    assert n_e >= n_sub
    x = x.copy()

    for i in range(n_sub):
        # x[i] = 2 * np.tanh(x[i])
        x[i] = np.tanh(x[i]) - x[i]
    for i in range(n_sub, n_e):
        x[i] = -2 * np.tanh(x[i])
    return x


def matrix_div(a, b):
    # x / y
    # (inverse (y') * x')'
    return (np.linalg.inv(b.T) @ a.T).T


def update_m(x, m, lambdas=.005):
    # Natural gradient-based recursive least-squares
    # algorithm for adaptive blind source separation

    n_ch, n_p = x.shape
    lambdas = lambdas[-n_p:]
    v = m @ x
    # [TODO] Check if it is possible to no use the median value, and use the full array
    # i.e. `l = 1 - lambdas`

    l_avg = 1 - lambdas[len(lambdas) // 2 - 1]  # In code
    l = l_avg  # 1 - lambdas
    # l = 1 - lambdas

    Q = l / (1 - l) + np.trace(v.T @ v) / n_p  # In code
    # Q = (l / (1 - l) + np.diag(v.T @ v)).sum() / n_p

    # Division by n_p to normalize the block-wide v @ v.T multiplication]
    # v @ v.T = sum_{i=0}^{n_p} v[:, [i]] @ v.T[[i], :]
    m = (1 / l_avg) * (m - v @ v.T @ m / Q / n_p)
    # m /= np.linalg.norm(m, "fro")

    return m


def update_w_block(v, w, lambdas=None, n_sub=1):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal
    # https://github.com/goodshawn12/orica/blob/master/orica.m
    n_ch, n_p = v.shape
    if isinstance(lambdas, list):
        lambdas = np.array(lambdas)
    elif isinstance(lambdas, float):
        lambdas = np.array([lambdas] * n_p)

    assert len(lambdas) == n_p, (len(lambdas), n_p)
    # lambdas = lambdas[-n_p:]
    # assert len(lambdas) == n_p
    y = w @ v
    fy = nonlinearity(y, n_sub=n_sub)

    lambda_prod = np.product(1 / (1 - lambdas))
    # dot(f, y, 1) = (fy * y).sum(axis=0)
    Q = 1 + lambdas * ((fy * y).sum(axis=0) - 1)

    w = lambda_prod * (w - y @ np.diag(lambdas / Q) @ fy.T @ w)
    w = orthogonalize(w)
    return w


def update_w_block_paper(v, w, lambdas=None, n_sub=1):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal
    # https://github.com/goodshawn12/orica/blob/master/orica.m
    n_ch, n_p = v.shape
    if isinstance(lambdas, list):
        lambdas = np.array(lambdas)
    elif isinstance(lambdas, float):
        lambdas = np.array([lambdas] * n_p)

    lambdas = lambdas[-n_p:]
    # assert len(lambdas) == n_p
    y = w @ v
    fy = nonlinearity(y, n_sub=n_sub)

    lambda_prod = np.product(1 / (1 - lambdas))
    # dot(f, y, 1) = (fy * y).sum(axis=0)
    Q = (1 - lambdas) / lambdas + np.diag(fy.T @ y)
    # Q = (1 - lambdas) / lambdas + (fy.T * y).sum(axis=1)
    I = np.eye(n_ch)
    D = np.stack(
        [
            y[:, [i]] * fy[:, [i]].T
            for i in range(n_p)

        ]
    )
    Q = Q.reshape(n_p, 1, 1)
    # w = lambda_prod * (w - y @ np.diag(lambdas / Q) @ fy.T @ w)
    w = lambda_prod * (I - (D / Q).sum(axis=0)) @ w
    w = orthogonalize(w)

    return w


class ORICA(BaseEstimator):

    SUPPORTED_MODES = (
        "constant",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        "adaptative",
        # "adaptative_exp",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        "decay",
        # "pid"
    )

    def __init__(self, n_channels=6, size_block=4, block_update=False, stride=1, mode=None, lm_0=None, lw_0=None, gamma=None, adaptative_constants=None, n_passes=1, adap_exp_norm=1e4, tau_const=np.inf, n_sub=1):
        self.mode = mode
        self.lm_0 = lm_0
        self.lw_0 = lw_0
        self.n_channels = n_channels
        self.gamma = gamma
        self.n_passes = n_passes
        self.size_block = size_block
        self.adaptative_constants = adaptative_constants
        self.adap_exp_norm = adap_exp_norm
        self.block_update = block_update
        self.stride = stride
        self.Rn = None
        self.n_sub = n_sub
        self.tau_const = tau_const
        self.I = np.eye(n_channels)
        self.init()

    def init(self):
        if self.mode is None:
            self.mode = "constant"

        assert self.mode in self.SUPPORTED_MODES
        self.lambda_const = 1 - np.exp(-1 / (self.tau_const))
        if self.gamma is None:
            self.gamma = 0.6

        if self.lm_0 is None:
            self.lm_0 = .995  # np.eye(self.n_channels)

        if self.lw_0 is None:
            self.lw_0 = .995  # np.eye(self.n_channels)

        if self.adaptative_constants is None:
            # ORIGINAL
            # self.adaptative_constants = adaptative_constants(.03, .012, 0.05)
            # MYCHANGE
            self.adaptative_constants = adaptative_constants(.03, .012, 0.05)

        self.w_0 = self.m_0 = self.I
        self.w = None
        self.m = None
        self.iteration = None

    def ANSI(self, y):
        # Adaptative Non stationary Index
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        _, size_block = y.shape
        sigmarn = self.adaptative_constants.sigmarn
        R = y @ nonlinearity(y, n_sub=self.n_sub).T / size_block
        if self.Rn is None:
            # code uses(self.I + y @ nonlinearity(y, n_sub=self.n_sub).T / self.size_block
            self.Rn = self.I - R
        else:
            # Code in octave uses sigma * (self.I + y @ nonlinearity(y, n_sub=self.n_sub).T / self.size_block)
            self.Rn = (
                (1 - sigmarn) * self.Rn +
                sigmarn * (self.I - R)
            )
        z = np.linalg.norm(
            self.Rn,
            "fro"
        )
        return z

    def NSI(self, y):
        # Non stationary Index
        # Real-time Adaptive EEG Source Separation using Online Recursive Independent Component Analysis
        _, size_block = y.shape
        sigma = np.linalg.norm(
            y @ nonlinearity(y, n_sub=self.n_sub).T / size_block - self.I,
            "fro"
        )
        return sigma

    def G(self, z, z_min=None, c=5, b=1.5, eps=1):
        if z_min is None:
            z_min = z
        return .5 * (
            1 + np.tanh(
                (
                    z / max(z_min, eps) - c
                ) / b
            )
        )

    def get_lambdas(self, x, iteration, m=None, w=None):
        if m is None:
            m = self.m
        if w is None:
            w = self.w
        y = w @ m @ x

        n_ch, n_p = x.shape

        def sigmoid(x):
            return (1 / (1 + np.exp(- x)))

        if self.mode == "constant":
            sigma = self.NSI(y)
            lm = np.repeat(self.lm_0, n_p)
            lw = np.repeat(self.lw_0, n_p)

        elif self.mode == "decay":
            sigma = self.NSI(y)
            # MATLAB code does state.counter + dataRange
            # So it basically uses 2 times the iteration count, so we do that here
            # [TODO] should we try parameterizing this 2?
            div = iteration ** self.gamma
            lm, lw = self.lm_0 / div, self.lw_0 / div
            # log((lm, iteration))
            # print(iteration - n_p + 1, lm[0], np.product(1 / (1 - lm[-8:])))
            lm = np.where(lm < self.lambda_const, self.lambda_const, lm)
            lw = np.where(lw < self.lambda_const, self.lambda_const, lw)

        elif self.mode == "adaptative":
            alpha, beta, _ = self.adaptative_constants
            sigma = self.ANSI(y)

            if not hasattr(self, "lm"):
                self.lm = np.repeat([self.lm_0], n_p)
                self.lw = np.repeat([self.lw_0], n_p)
            
            if not hasattr(self, "z_min"):
                if hasattr(self, "sigmas") and (len(self.sigmas) > 0):
                    self.z_min = min(self.sigmas)
                else:
                    self.z_min = sigma

            # Code updates z_min, although paper doesnt say to do that
            self.z_min = min(sigma, self.z_min)

            G = self.G(sigma, z_min=self.z_min)
            lm = self.lm - alpha * self.lm ** 2 + beta * G * self.lm
            lw = self.lw - alpha * self.lw ** 2 + beta * G * self.lw
            self.lw, self.lm = lw, lm

        elif self.mode == "adaptative_exp":
            sigma = self.NSI(y)
            ll = 1 - np.exp(- sigma / self.adap_exp_norm)
            lm, lw = ll, ll

        assert not any(np.isnan(lm))
        assert not any(np.isnan(lw))

        return lm, lw, sigma

    def fit(self, x, y=None):
        pass
        # self.base_sigma = self.NSI(self.w_0 @ self.m_0 @ x)

    def transform_(self, X, warm_start=False):
        X_filtered = X.copy()

        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
        else:
            assert not ((self.w is None) or (self.m is None)), "You need to call transform once"

        self.lambdas = [self.lw_0]
        self.sigmas = list()
        for i in tqdm(range(self.size_block, len(X.T), self.stride)):
            X_i = X[:, i - self.size_block:i]
            for n_pass in range(self.n_passes):

                lm, lw, sigma = self.get_lambdas(X_i, i, self.m, self.w)
                print(lm)
                self.m = update_m(X_i, self.m, lambdas=lm)
                self.w = update_w_block(self.m @ X_i, self.w, lambdas=lw, n_sub=self.n_sub)

                y = self.w @ self.m @ X_i
                X_filtered[:, i - self.size_block:i] = y

                self.lambdas.append(lw.mean())
                self.sigmas.append(sigma)

        return X_filtered

    def transform(self, X, warm_start=False, scaling=1, save=False):
        orig_mpl_backend = matplotlib.get_backend()
        matplotlib.use("agg")

        X = X * scaling
        # X = X / X.transpose(1, 0, 2).reshape(X.shape[1], -1).std(axis=1)[np.newaxis, :, np.newaxis]
        n_channels, n_times = X.shape
        X_filtered = X.copy()
        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
            counter = 0
        else:
            assert not ((self.w is None) or (self.m is None)), "You need to call transform once"
            counter = self.iteration[-self.size_block:]

        self.lambdas = [self.lw_0]
        self.sigmas = list()
        im_cnt = 0
        for i in tqdm(range(self.size_block, n_times + self.stride, self.stride), leave=False):
            X_i = X[:, i - self.size_block:i]
            n_p = X_i.shape[1]
            for n_pass in range(self.n_passes):
                self.iteration = np.arange(i - self.size_block, i) + 1 + counter
                lm, lw, sigma = self.get_lambdas(X_i, self.iteration, self.m, self.w)
                # print(e, n_times, e * n_times + i, max(lm), min(lm))
                # [TODO use new m for w update]
                m = update_m(X_i, self.m, lambdas=lm[-n_p:])
                w = update_w_block(m @ X_i, self.w, lambdas=lw[-n_p:], n_sub=self.n_sub)

                self.m, self.w = m, w

                y = self.w @ self.m @ X_i
                X_filtered[:, i - self.size_block:i] = y

                self.lambdas.append(lw.mean())
                self.sigmas.append(sigma)

                counter += self.stride

            if ((i % 100) < self.stride) and save:
                fig = plt.figure(figsize=(9, 9))
                plt.imshow(self.m)
                plt.colorbar()
                fig.savefig("./m/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                plt.close(fig)
                del fig

                fig = plt.figure(figsize=(9, 9))
                plt.imshow(self.w)
                plt.colorbar()
                fig.savefig("./w/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                plt.close(fig)
                del fig

                fig = plt.figure(figsize=(9, 9))
                plt.imshow(np.corrcoef(self.m @ X_i))
                plt.colorbar()
                fig.savefig("./c/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                plt.close(fig)
                del fig

                plt.close("all")
                im_cnt += 1

        matplotlib.use(orig_mpl_backend)
        return X_filtered

    def transform_epochs(self, X, warm_start=False, scaling=1, save=False, verbose=True):
        orig_mpl_backend = matplotlib.get_backend()
        matplotlib.use("agg")

        X = X.get_data() * scaling
        # X = X / X.transpose(1, 0, 2).reshape(X.shape[1], -1).std(axis=1)[np.newaxis, :, np.newaxis]
        n_epochs, n_channels, n_times = X.shape
        X_filtered = X.copy()
        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
            counter = 0
        else:
            assert not ((self.w is None) or (self.m is None)), "You need to call transform once"
            counter = self.iteration[-self.size_block:]

        self.lambdas = [self.lw_0]
        self.sigmas = list()
        im_cnt = 0

        for e in tqdm(range(n_epochs)):
            window_iter = range(self.size_block, n_times + self.stride, self.stride)
            for i in tqdm(window_iter, leave=False):
                X_i = X[e, :, i - self.size_block:i]
                n_p = X_i.shape[1]
                for n_pass in range(self.n_passes):
                    self.iteration = np.arange(e * n_times + i - self.size_block, e * n_times + i) + 1 + counter

                    lm, lw, sigma = self.get_lambdas(X_i, self.iteration, self.m, self.w)
                    # print("{}-{}:{}-{}".format(e * n_times + i - n_p + 1, e * n_times + i + 1, lm[0], lm[-1]))
                    m = update_m(X_i, self.m, lambdas=lm[-n_p:])
                    w = update_w_block(m @ X_i, self.w, lambdas=lw[-n_p:], n_sub=self.n_sub)
                    # print(e * n_times + i, m.max())
                    self.m, self.w = m, w

                    y = self.w @ self.m @ X_i
                    X_filtered[e, :, i - self.size_block:i] = y

                    self.lambdas.append(lw.mean())
                    self.sigmas.append(sigma)

                    counter += self.stride

                if ((i % 100) < self.stride) and save:
                    fig = plt.figure(figsize=(9, 9))
                    plt.imshow(self.m)
                    plt.colorbar()
                    fig.savefig("./m/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                    plt.close(fig)
                    del fig

                    fig = plt.figure(figsize=(9, 9))
                    plt.imshow(self.w)
                    plt.colorbar()
                    fig.savefig("./w/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                    plt.close(fig)
                    del fig

                    fig = plt.figure(figsize=(9, 9))
                    plt.imshow(np.corrcoef(self.m @ X_i))
                    plt.colorbar()
                    fig.savefig("./c/img_{}.png".format(str(im_cnt).rjust(5, "0")))
                    plt.close(fig)
                    del fig

                    plt.close("all")
                    im_cnt += 1

        matplotlib.use(orig_mpl_backend)
        return X_filtered


class CBEB_ORICA(BaseEstimator):

    def __init__(self, n_components=None, n_sub=0, tran_lm=.995, tran_lw=0.995, gamma=.6, ss_lm=1e-3, ss_lw=1e-3, scaling=1e6, size_block=8, stride=8):
        self.n_channels = n_components
        self.ss_lm = ss_lm
        self.ss_lw = ss_lw
        self.n_sub = n_sub
        self.tran_lm = tran_lm
        self.tran_lw = tran_lw
        self.gamma = gamma
        self.size_block = size_block
        self.stride = stride
        self.scaling = scaling

        self.epochs_input = False

    def _create_ica_names(self):
        self._ica_names = [
            "ICA{}".format(
                str(i).rjust(3, "0")
                for i in range(self.n_channels)
            )
        ]

    def epochs_to_np(self, epochs):
        if isinstance(epochs, BaseEpochs):
            epochs.load_data()
            return epochs.get_data()
        return epochs

    def fit(self, x, y=None, return_filtered=False):

        if isinstance(x, BaseEpochs):
            self.epochs_input = True
            x = x.get_data()

        self.n_epochs, n_channels, self.n_times = x.shape

        if self.n_channels is None:
            self.n_channels = n_channels
        self._create_ica_names()
        self.ica = ORICA(
            mode="decay",
            n_channels=self.n_channels,
            block_update=True,
            size_block=self.size_block,
            stride=self.stride,
            lm_0=self.tran_lm,
            lw_0=self.tran_lw,
            gamma=self.gamma,
            n_sub=self.n_sub,
        )

        self.ica.fit(x)
        assert n_channels == self.n_channels

        x = x.transpose(1, 0, 2).reshape(n_channels, -1)
        x = (
            self.ica
            .transform(
                x,
                scaling=self.scaling,
                save=False,
            )
            .reshape(n_channels, self.n_epochs, self.n_times)
            .transpose(1, 0, 2)
        )

        if return_filtered:
            return x
        return self

    def transform(self, x, y=None, as_epochs=True):

        x_copy = None
        if isinstance(x, BaseEpochs):
            x_copy = x.copy()
            x = x.get_data()

        self.ica.mode = "constant"
        self.ica.lm_0, self.ica.lw_0 = self.ss_lm, self.ss_lw

        n_epochs, n_channels, n_times = x.shape
        assert n_channels == self.n_channels

        x = x.transpose(1, 0, 2).reshape(n_channels, -1)
        x = (
            self.ica
            .transform(
                x,
                scaling=self.scaling,
                save=False,
                warm_start=True
            )
            .reshape(n_channels, n_epochs, n_times)
            .transpose(1, 0, 2)
        )

        if as_epochs and (x_copy is not None):
            return self.epochs_from_array(x, x_copy)

        return x

    def fit_transform(self, x, y=None, as_epochs=True):
        if as_epochs and isinstance(x, BaseEpochs):
            x_copy = x.copy()

        x_filtered = self.fit(x, y=y, return_filtered=True)

        if as_epochs:
            return self.epochs_from_array(x_filtered, x_copy)
        return x_filtered

    def _export_info(self, info, container, add_channels):
        # Adapted from
        # https://github.com/mne-tools/mne-python/blob/96a4bc2e928043a16ab23682fc818cf0a3e78aef/mne/preprocessing/ica.py#L1221
        """Aux method."""
        # set channel names and info
        ch_names = []
        ch_info = []
        for ii, name in enumerate(self._ica_names):
            ch_names.append(name)
            ch_info.append(dict(
                ch_name=name, cal=1, logno=ii + 1,
                coil_type=FIFF.FIFFV_COIL_NONE,
                kind=FIFF.FIFFV_MISC_CH,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                unit=FIFF.FIFF_UNIT_NONE,
                loc=np.zeros(12, dtype='f4'),
                range=1.0, scanno=ii + 1, unit_mul=0))

        if add_channels is not None:
            # re-append additionally picked ch_names
            ch_names += add_channels
            # re-append additionally picked ch_info
            ch_info += [k for k in container.info['chs'] if k['ch_name'] in
                        add_channels]
        with info._unlock(update_redundant=True, check_after=True):
            info['chs'] = ch_info
            info['bads'] = []
            info['projs'] = []  # make sure projections are removed.

    def epochs_from_array(self, arr, epochs):
        # Method copied on how MNE creates a new epoch in the ICA class
        # https://github.com/mne-tools/mne-python/blob/96a4bc2e928043a16ab23682fc818cf0a3e78aef/mne/preprocessing/ica.py#L1185
        """Aux method."""
        out = epochs.copy()

        assert arr.ndim == 3
        assert arr.shape[1] == self.n_channels
        out._data = arr

        self._export_info(out.info, epochs, None)
        out.preload = True
        out._raw = None
        out._projector = None
        return out


def plot_various(x, n=6, d=1, figsize=(20, 5), ax=None, show=True, title=""):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.plot(x.T + d * np.array(range(n)).reshape(1, -1), alpha=.4)
    ax.set_title(title)
    if show:
        plt.show()


if __name__ == "__main__":
    from ica_benchmark.sample.ica import sample_ica_data

    X, sources, W = sample_ica_data(N=1000, seed=100)

    ica = ORICA(mode="decay", block_update=True, size_block=8, stride=8, gamma=0.6, lm_0=.995, lw_0=.995)

    X_filtered = ica.transform(X)

    plot_various(X, d=1, figsize=(20, 4), title="Raw")

    plot_various(np.linalg.pinv(W) @ X, d=1, figsize=(20, 4), n=4, title="Estimated sources from W real")

    plot_various(X_filtered, d=10, figsize=(20, 4), title="Filtered")

    plot_various(ica.w @ ica.m @ X, d=10, figsize=(20, 4), title="Estimated sources with final ICA")

    plt.figure(figsize=(20, 4))
    plt.plot(np.log(ica.sigmas))
    plt.title("Sigmas")
    plt.show()

    plt.figure(figsize=(20, 4))
    plt.plot(np.log(ica.lambdas))
    plt.title("Lambdas")
    plt.show()

