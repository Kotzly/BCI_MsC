import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator
from ica_benchmark.data.utils import is_notebook


def whitening_multivar_matrix(X):
    cov = X @ X.T / len(X.T)
    d, E = np.linalg.eigh(cov)

    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    B = E @ D_inv @ E.T
    return B


def whitening_multivar(X):
    X = X.copy()
    B = whitening_multivar_matrix(X)
    X -= X.mean(axis=1, keepdims=True)
    return B @ X


whitening = whitening_multivar


def exponential_standardize(data, start=1000, eps=1e-4, alpha=1e-3):
    # https://arxiv.org/pdf/1703.05051.pdf
    # data: n_channels, n_times
    # Make sure data is sampled at 250Hz! Default values of alpha 1e-3 and start of 1000 expect it!

    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    mean = data[:, :start].mean(axis=1, keepdims=True)
    var = data[:, :start].var(axis=1, keepdims=True)

    N = data.shape[1]
    new_data = data.copy()
    new_data[:, :start] -= (new_data[:, :start] - mean) / np.maximum(np.sqrt(var), eps)

    for idx in tqdm(range(start, N)):
        x_t = new_data[:, [idx]]
        mean = alpha * x_t + (1 - alpha) * mean
        var = alpha * (x_t - mean) ** 2 + (1 - alpha) * var
        new_data[:, [idx]] = (x_t - mean) / np.maximum(np.sqrt(var), eps)
    return new_data


def _whitening(X):

    """
    Function that withens the data.

    Attributes:
        * X: time series data (channels, times)

    Returns:
        * Whitened data
        * Singular vectors
        * singular values

    """

    X = X - X.mean(axis=1, keepdims=True)
    U, d, _ = np.linalg.svd(X, full_matrices=False)
    U_d = (U / d).T
    X_whiten = np.dot(U_d, X) * np.sqrt(X.shape[1])

    return X_whiten, U, d


class Whitening(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X):
        """Whitening

        Args:
            X (np.ndarray): (n_times, n_channels)
        """
        self.W = whitening_multivar(X.T)
        self.is_fitted_ = True

    def transform(self, X):

        check_is_fitted(self, 'is_fitted_')
        return X @ self.W.T

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
