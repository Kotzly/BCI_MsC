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
    # EDIT: trace solution proved to be a little faster
    #Q = beta + (v ** 2).sum(axis=0).mean()
    Q = beta + np.trace(v.T @ v) / n_p

    # Division by n_p to normalize the block-wide v @ v.T multiplication]
    # v @ v.T = sum_{i=0}^{n_p} v[:, [i]] @ v.T[[i], :]
    m = (1 / l) * (m - v @ v.T @ m / Q / n_p)
    #m /= np.linalg.norm(m, "fro")

    return m


def update_m_orica(x, m, l=.005):
    # Online Recursive ICA Algorithm Used for Motor Imagery EEG Signal

    n_ch, n_p = x.shape
    v = m @ x
    beta = l / (1 - l)

    Q = 1 + l * (np.trace(v.T @ v) / n_p - 1)
    I = np.eye(n_ch)
    m = m + beta * (I - v @ v.T / n_p / Q) @ m

    return m



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
    w = orthogonalize(w)
    
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

    w = orthogonalize(w)

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

    w = orthogonalize(w)

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
        self.Rn = None
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
        R = y @ nonlinearity(y).T / size_block
        if self.Rn is None:
            # code uses(self.I + y @ nonlinearity(y).T / self.size_block
            self.Rn = self.I - R
        else:
            # Code in octave uses sigma * (self.I + y @ nonlinearity(y).T / self.size_block)
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
            y @ nonlinearity(y).T / size_block - self.I,
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

        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        if self.mode == "constant":
            sigma = self.NSI(y)
            lm, lw = self.lm_0, self.lw_0

        elif self.mode == "decay":
            sigma = self.NSI(y)
            div = iteration ** self.gamma
            lm, lw = self.lm_0 / div, self.lw_0 / div

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
        lm = self.lm_0
        lw = self.lw_0
        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
        else:
            assert not ((self.w is None) or (self.m is None)), "You need to call transform once"
        self.lambdas = [lw]
        self.sigmas = list()
        for i in tqdm(range(self.size_block, len(X.T), self.stride)):
        #for i in range(self.size_block, len(X.T), self.stride):
            for n_pass in range(self.n_passes):
                X_i = X[:, i - self.size_block: i]
                self.m = update_m(X[:, i - self.size_block: i], self.m, l=1 - lm)
                #self.m = update_m_orica(X[:, i - self.size_block: i], self.m, l=1 - lm)
                if self.block_update:
                    lambdas_ = [self.lw_0] * (self.size_block - len(self.lambdas)) + self.lambdas[-self.size_block:]
                    self.w = update_w_block(self.m @ X_i, self.w, lambdas=lambdas_)
                else:
                    self.w = update_w(self.m @ X_i, self.w, l=lw)

                y = self.w @ self.m @ X_i
                X_filtered[:, i - self.size_block:i] = y

                lm, lw, sigma = self.get_lambdas(X_i, i, self.m, self.w)
                self.lambdas.append(lw)
                self.sigmas.append(sigma)
        
        return X_filtered

    def transform_epochs(self, X, warm_start=False):
        X = X.get_data()
        n_epochs, n_channels, n_times = X.shape
        X_filtered = X.copy()
        lm = self.lm_0
        lw = self.lw_0
        if not warm_start:
            self.m = self.m_0
            self.w = self.w_0
        self.lambdas = [lw]
        self.sigmas = list()
        iter_n = 0
        for e in tqdm(range(n_epochs)):
            for i in tqdm(range(self.size_block, n_times, self.stride), leave=False):
            #for i in range(self.size_block, len(X.T), self.stride):
                for n_pass in range(self.n_passes):
                    iter_n += 1
                    X_i =  X[e, :, i - self.size_block:i]
                    # l = l_0 / (i - size_block + 1) ** gamma
                    #self.m = update_m(X[e, :, i - self.size_block: i], self.m, l=1 - lm)
                    self.m = update_m_orica(X[e, :, i - self.size_block: i], self.m, l=1 - lm)
                    
                    if self.block_update:
                        lambdas_ = [self.lw_0] * (self.size_block - len(self.lambdas)) + self.lambdas[-self.size_block:]
                        self.w = update_w_block(self.m @ X_i, self.w, lambdas=lambdas_)
                    else:
                        self.w = update_w(self.m @ X_i, self.w, l=lw)

                    y = self.w @ self.m @ X_i
                    X_filtered[e, :, i - self.size_block:i] = y

                    lm, lw, sigma = self.get_lambdas(X_i, iter_n, self.m, self.w)
                    self.lambdas.append(lw)
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

    X, sources, W = sample_ica_data(N=50000, seed=100)
    
    #ica = ORICA(mode="constant", block_update=True, size_block=8, stride=8, lw_0=.0078, lm_0=0.0078)
    #ica = ORICA(mode="decay", block_update=False, size_block=16, stride=4, gamma=0.6, lm_0=.995, lw_0=.995)
    #ica = ORICA(mode="adaptative", block_update=True, size_block=8, stride=8, lm_0=.1, lw_0=.1)
    #ica = ORICA(mode="adaptative_exp", block_update=False, size_block=64, stride=2, adap_exp_norm=1e4)
    #ica = ORICA(mode="decay", block_update=False, size_block=32, stride=8, gamma=0.6, lm_0=.995, lw_0=.995)
    ica = ORICA(mode="adaptative", block_update=True, size_block=32, stride=32, lm_0=.1, lw_0=.1)
    
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

