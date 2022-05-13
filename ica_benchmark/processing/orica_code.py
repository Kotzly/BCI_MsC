from sklearn.base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import namedtuple

adaptative_constants = namedtuple("adaptative_constants", ["a", "b", "sigmarn"])

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
        x[i] = -2 * np.tanh(x[i])
    for i in range(n_sub, n_e):
        x[i] = 2 * np.tanh(x[i])
    return x


def matrix_div(a, b):
    # x / y
    # (inverse (y') * x')'
    return (np.linalg.inv(b.T) @ a.T).T


def update_m(x, m, lambdas=.005):
    # Natural gradient-based recursive least-squares
    # algorithm for adaptive blind source separation

    n_ch, n_p = x.shape
    v = m @ x
    l = 1 - lambdas[len(lambdas) // 2]

    Q = l / (1 - l) + np.trace(v.T @ v) / n_p

    # Division by n_p to normalize the block-wide v @ v.T multiplication]
    # v @ v.T = sum_{i=0}^{n_p} v[:, [i]] @ v.T[[i], :]
    m = (1 / l) * (m - v @ v.T @ m / Q / n_p)
    #m /= np.linalg.norm(m, "fro")

    return m




def update_w_block(v, w, lambdas=None, n_sub=1):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal
    # https://github.com/goodshawn12/orica/blob/master/orica.m
    n_ch, n_p = v.shape
    if isinstance(lambdas, list):
        lambdas = np.array(lambdas)
    elif isinstance(lambdas, float):
        lambdas = np.array([lambdas] * n_p)

    assert len(lambdas) == n_p

    y = w @ v
    fy = nonlinearity(y, n_sub=n_sub)

    lambda_prod = np.product(1 / (1 - lambdas))
    # dot(f, y, 1) = (fy * y).sum(axis=0)
    Q = 1 + lambdas * ((fy * y).sum(axis=0) - 1)
    I = np.eye(n_ch)

    w = lambda_prod * (w - y @ np.diag(lambdas / Q) @ fy.T @ w)
    w = orthogonalize(w)
    
    return w


class ORICA(BaseEstimator):

    SUPPORTED_MODES = (
        "constant",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        #"adaptative",
        #"adaptative_exp",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        "decay",
        #"pid"
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
        self.lambda_const = 1-np.exp(-1/(self.tau_const))
        if self.gamma is None:
            self.gamma = 0.6

        if self.lm_0 is None:
            self.lm_0 = .995  # np.eye(self.n_channels)

        if self.lw_0 is None:
            self.lw_0 = .995  # np.eye(self.n_channels)

        if self.adaptative_constants is None:
            # MYCHANGE
            # self.adaptative_constants = adaptative_constants(.03, .012, 0.05)
            self.adaptative_constants = adaptative_constants(.1, .012, 0.05)
        
        self.w_0 = self.m_0 = self.I
        self.w = None
        self.m = None

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
            div = np.arange(iteration - n_p + 1, iteration + 1) ** self.gamma
            lm, lw = self.lm_0 / div, self.lw_0 / div
            lm = np.where(lm < self.lambda_const, self.lambda_const, lm)
            lw = np.where(lw < self.lambda_const, self.lambda_const, lw)
            

        elif self.mode == "adaptative":
            alpha, beta, _ = self.adaptative_constants
            sigma = self.ANSI(y)

            if not hasattr(self, "lm"):
                self.lm = self.lm_0
                self.lw = self.lw_0
                self.z_min = sigma

            # Code updates z_min, although paper doesnt say to do that
            # self.z_min = min(sigma, self.z_min)
            
            G = self.G(sigma, z_min=self.z_min)
            lm = self.lm - alpha * self.lm ** 2 + beta * G * self.lm
            lw = self.lw - alpha * self.lw ** 2 + beta * G * self.lw
            self.lw, self.lm = lw, lm            

        elif self.mode == "adaptative_exp":
            sigma = self.NSI(y)
            l = 1 - np.exp(- sigma / self.adap_exp_norm)
            lm, lw = l, l

        return lm , lw, sigma

    def fit(self, x, y=None):
        self.base_sigma = self.NSI(self.w_0 @ self.m_0 @ x)

    def transform(self, X, warm_start=False):
        X_filtered = X.copy()
        

        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
        else:
            assert not ((self.w is None) or (self.m is None)), "You need to call transform once"

        lm, lw, _ = self.get_lambdas(X, self.size_block, self.m, self.w)
        self.lambdas = [lw.mean()]
        self.sigmas = list()
        for i in tqdm(range(self.size_block, len(X.T), self.stride)):
            X_i = X[:, i - self.size_block:i]
            for n_pass in range(self.n_passes):

                self.m = update_m(X_i, self.m, lambdas=lm)
                self.w = update_w_block(self.m @ X_i, self.w, lambdas=lw)

                y = self.w @ self.m @ X_i
                X_filtered[:, i - self.size_block:i] = y

                lm, lw, sigma = self.get_lambdas(X_i, i, self.m, self.w)
                self.lambdas.append(lw.mean())
                self.sigmas.append(sigma)
        
        return X_filtered

    def transform_epochs(self, X, warm_start=False):
        X = X.get_data()
        n_epochs, n_channels, n_times = X.shape
        X_filtered = X.copy()
        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
        lm, lw, _ = self.get_lambdas(X[0], self.size_block, self.m, self.w)
        self.lambdas = [lw.mean()]
        self.sigmas = list()

        for e in tqdm(range(n_epochs)):
            for i in tqdm(range(self.size_block, n_times + self.stride, self.stride), leave=False):
                X_i = X[e, :, i - self.size_block:i]
                n_p = X_i.shape[1]
                for n_pass in range(self.n_passes):

                    lm, lw, sigma = self.get_lambdas(X_i, e * n_times + i, self.m, self.w)
#                    print(e, n_times, e * n_times + i, max(lm), min(lm))
                    self.m = update_m(X_i, self.m, lambdas=lm[-n_p:])
                    self.w = update_w_block(self.m @ X_i, self.w, lambdas=lw[-n_p:])

                    y = self.w @ self.m @ X_i
                    X_filtered[e, :, i - self.size_block:i] = y

                    self.lambdas.append(lw.mean())
                    self.sigmas.append(sigma)
        
        return X_filtered


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
    
    ica.fit(X)
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

