from sklearn.base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from zmq import has


def nonlinearity(x, n_sub=1):
    n_e = len(x)
    assert n_e >= n_sub
    x = x.copy()

    for i in range(n_sub):
        x[i] = -2 * np.tanh(x[i])
    for i in range(n_sub, n_e):
        # x[i] = 2 * np.tanh(x[i])
        x[i] = np.tanh(x[i]) - x[i]
    return x


def matrix_div(a, b):
    # x / y
    # (inverse (y') * x')'
    return (np.linalg.inv(b.T) @ a.T).T


def update_m(x, m, l=.005):
    # Natural gradient-based recursive least-squares
    # algorithm for adaptive blind source separation

    n_ch, n_p = x.shape
    v = m @ x
    beta = l / (1 - l)
    # The second part of the sum is equivalent to v.T @ v
    # In matlab this is done with trace(v.T @ v) / n_p but this way
    # We avoid a O(n²) matrix multiplication and do a O(n) matrix multp
    Q = beta + (v ** 2).sum(axis=0).mean()
    # Division by n_p to normalize the block-wide v @ v.T multiplication]
    # v @ v.T = sum_{i=0}^{n_p} v[:, [i]] @ v.T[[i], :]
    m_new = (1 / l) * (m - v @ v.T @ m / Q / n_p)
    return m_new


def update_w_block(v, w, lambdas=None):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal
    # https://github.com/goodshawn12/orica/blob/master/orica.m
    n_ch, n_p = v.shape
    if isinstance(lambdas, list):
        lambdas = np.array(lambdas)
    elif isinstance(lambdas, float):
        lambdas = np.array([lambdas] * n_p)

    assert len(lambdas) == n_p

    y = w @ v
    fy = nonlinearity(y)

    lambda_prod = np.product(1 / (1 - lambdas))

    I = np.eye(n_ch)

    F = 0
    for i in range(n_p):
        beta = (1 - lambdas[i]) / lambdas[i]
        y_ = y[:, [i]]
        fy_ = fy[:, [i]]
        Q = beta + fy_.T @ y_
        F += y_ @ fy_.T / Q

    w = lambda_prod * (I - F) @ w
    try:
        d, v = np.linalg.eigh(w @ w.T)
    except np.linalg.LinAlgError as e:
        print(
            lambda_prod,
            1 / (1 - lambdas),
            np.linalg.norm(F),
            np.linalg.norm(w),
            np.linalg.norm(v),
        )
        raise e
        
    d = np.diag(d)
#     w = matrix_div(v, np.sqrt(d)) @ np.linalg.inv(v) @ w
    w = v @ np.sqrt(np.linalg.inv(d)) @ np.linalg.inv(v) @ w
    return w


def update_w(v, w, l=.995):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal
    # https://github.com/goodshawn12/orica/blob/master/orica.m
    n_ch, n_p = v.shape
    y = w @ v
    fy = nonlinearity(y)
    # l = min(l, 1 - 1e-8)
    scalar = l / (1 - l)
    I = np.eye(n_ch)

#     Q = (1 + l * (fy.T @ y - 1))
    Q = 1 + l * ((fy * y).sum(axis=0).mean() - 1)
    inv_iter = I - y @ fy.T / Q
    w = w + scalar * inv_iter @ w

    d, v = np.linalg.eigh(w @ w.T)
    d = np.diag(d)
#     w = matrix_div(v, np.sqrt(d)) @ np.linalg.inv(v) @ w
    w = v @ np.sqrt(np.linalg.inv(d)) @ np.linalg.inv(v) @ w

    return w


def update_w_2(v, w, l=.995):
    # Independent Component Analysis Using an Extended Infomax
    # Algorithm for Mixed Subgaussian and Supergaussian Sources
    n_ch, n_p = v.shape

    y = w @ v
    fy = nonlinearity(y)

    beta = l / (1 - l)
#     Q = (beta + v.T @ v) # diagonal(v.T @ v).mean()

    Q = beta + (fy * y).sum(axis=0).mean()

    w = (1 / l) * (w - y @ fy.T @ w / Q)

    d, v = np.linalg.eigh(w @ w.T)
    d = np.diag(d)
    w = matrix_div(v, np.sqrt(d)) @ np.linalg.inv(v) @ w

    return w


class ORICA(BaseEstimator):

    SUPPORTED_MODES = (
        "constant",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        "adaptative",
        "adaptative_exp",
        # Tracking Non-stationary EEG Sources using Adaptive Online Recursive Independent Component Analysis
        "decay",
        "pid"
    )

    def __init__(self, n_channels=6, size_block=4, block_update=False, stride=1, mode=None, lm_0=None, lw_0=None, gamma=None, adaptative_constants=None, n_passes=1, adap_exp_norm=1e4):
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
        self.I = np.eye(n_channels)
        self.init()
    
    def init(self):
        if self.mode is None:
            self.mode = "constant"

        assert self.mode in self.SUPPORTED_MODES

        if self.gamma is None:
            self.gamma = 0.6

        if self.lm_0 is None:
            self.lm_0 = .995  # np.eye(self.n_channels)

        if self.lw_0 is None:
            self.lw_0 = .995  # np.eye(self.n_channels)

        if self.adaptative_constants is None:
            self.adaptative_constants = (.03, .012, 0.05)
        
        self.w_0 = self.m_0 = self.I

    def ANSI(self, x, m, w, R=None, sigma=.1):
        y = w @ m @ x
        if R is None:
            R = self.NSI(y)
        R = (1 - sigma) * R - sigma * (self.I - y @ nonlinearity(y).T)
        return R

    def NSI(self, y):
        # Non stationarity index
        _, size_block = y.shape
        sigma = np.linalg.norm(
            y @ nonlinearity(y).T / size_block - self.I,
            "fro"
        )
        return sigma

    def G(self, z, z_min=None, c=5, b=1.5, eps=1e-6):
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
        sigma = self.NSI(y)

        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        if self.mode == "constant":
            lm, lw = self.lm_0, self.lw_0

        elif self.mode == "decay":
            div = iteration ** self.gamma
            lm, lw = self.lm_0 / div, self.lw_0 / div

        elif self.mode == "adaptative":
            a, b, _ = self.adaptative_constants
            if not hasattr(self, "lm"):
                self.lm = 0.1
                self.lw = 0.1
                self.z_min = sigma
                
            lm = self.lm - a * self.lm ** 2 + b * self.G(self.NSI(y), z_min=self.z_min)
            lw = self.lw - a * self.lw ** 2 + b * self.G(self.NSI(y), z_min=self.z_min)
            self.lw, self.lm = lw, lm

        elif self.mode == "adaptative_exp":
            l = 1 - np.exp(- sigma / self.adap_exp_norm)
            lm, lw = l, l

        return lm , lw, sigma

    def fit(self, x, y=None):
        self.base_sigma = self.NSI(self.w_0 @ self.m_0 @ x)

    def transform(self, X):
        X_filtered = X.copy()
        lm = self.lm_0
        lw = self.lw_0
        m = self.m_0
        w = self.w_0
        lambdas = [lw]
        sigmas = list()
        for i in tqdm(range(self.size_block, len(X.T), self.stride)):
        # for i in range(self.size_block, len(X.T), self.stride):
            for n_pass in range(self.n_passes):
                # l = l_0 / (i - size_block + 1) ** gamma
                m = update_m(X[:, i - self.size_block: i], m, l=1 - lm)
                if self.block_update:
                    lambdas_ = [self.lw_0] * (self.size_block - len(lambdas)) + lambdas[-self.size_block:]
                    w = update_w_block(m @ X[:, i - self.size_block:i], w, lambdas=lambdas_)
                else:
                    w = update_w(m @ X[:, i - self.size_block:i], w, l=lw)

                y = w @ m @ X[:, i - self.size_block:i]
                X_filtered[:, i - self.size_block:i] = y

                lm, lw, sigma = self.get_lambdas(X, i, m, w)
                lambdas.append(lw)
                sigmas.append(sigma)
        
        self.w = w
        self.m = m
        self.sigmas = sigmas
        return X_filtered


def plot_various(x, n=6, d=1, figsize=(20, 5), ax=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.plot(x.T + d * np.array(range(n)).reshape(1, -1), alpha=.4)
    if show:
        plt.show()

if __name__ == "__main__":
    from ica_benchmark.sample.ica import sample_ica_data

    X, sources, W = sample_ica_data()
#    ica = ORICA(mode="adaptative_exp", block_update=False, size_block=16, stride=8, adap_exp_norm=1e5)
    ica = ORICA(mode="adaptative_exp", block_update=True, size_block=16, stride=16, adap_exp_norm=1e4)
#    ica = ORICA(mode="constant", block_update=True, size_block=8, stride=8)
    ica.fit(X)
    X_filtered = ica.transform(X)
    
    plot_various(X, d=1, figsize=(20, 4))

    plot_various(np.linalg.pinv(W) @ X, d=1, figsize=(20, 4), n=4)

    plot_various(X_filtered, d=10, figsize=(20, 4))

    plot_various(ica.w @ ica.m @ X, d=10, figsize=(20, 4))

    plt.figure(figsize=(20, 4))
    plt.plot(np.log(ica.sigmas))
    plt.show()

    y = ica.w @ ica.m @ X
    plt.imshow(
        y @ nonlinearity(y).T / ica.size_block
    )
    plt.colorbar()
    plt.show()
    print(ica.NSI(y))