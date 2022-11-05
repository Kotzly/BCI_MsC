import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator


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

        self.W = whitening_multivar(X.T)
        self.is_fitted_ = True

    def transform(self, X):

        check_is_fitted(self, 'is_fitted_')
        return X @ self.W.T

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
