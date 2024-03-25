# Code from https://github.com/edouardpineau/Time-Series-ICA-with-SOBI-Jacobi

import numpy as np
import itertools

from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator


def center(X, mean=None):
    """
    Function to center the data using empirical mean to have each variable with zero mean.

    Attributes:
        * X: data to center
        * mean: if the true mean is known, can be used (default=None)

    Returns:
        * Centered data
    """

    if mean is None:
        return X - X.mean(axis=1, keepdims=True)
    else:
        return X - mean


def time_lagged_autocov(X, lags):
    """
    Computes the auto-covariance tensor, containing all lagged-autocovariance with lag from 0 (covariance) to lags

    Attributes:
        * X: time series data (dimension: variables x time)
        * lags: number of lags to consider

    Returns:
        * Autocovariance tensor

    """

    lags = lags + 1
    n, cols = X.shape
    L = cols - lags
    R = np.empty([lags, n, n])

    X0 = center(X[:, :L])

    for k in range(lags):
        Xk = center(X[:, k : k + L])
        R[k] = (1.0 / L) * (X0.dot(Xk.T))
        R[k] = 0.5 * (R[k] + R[k].T)

    return R


def whitening(X):
    """
    Function that withens the data.

    Attributes:
        * X: time series data

    Returns:
        * Whitened data
        * Singular vectors
        * singular values

    """

    X = center(X)
    U, d, _ = np.linalg.svd(X, full_matrices=False)
    U_d = (U / d).T
    X_whiten = np.dot(U_d, X) * np.sqrt(X.shape[1])

    return X_whiten, U, d


def off_frobenius(M):
    """
    Computes the square Frobenius norm of the matrix M-diag(M)

    Attributes:
        * M: square matrix

    Returns:
        * Off-diagonal Frobenius norm


    """

    return (
        np.linalg.norm(np.tril(M, k=-1), ord="fro") ** 2
        + np.linalg.norm(np.triu(M, k=1), ord="fro") ** 2
    )


def rotation(M):
    """
    This function infers Jacobi rotation matrix R used in the joint diagonalization of a set of matrices

    See: https://en.wikipedia.org/wiki/Jacobi_rotation

    Attributes:
        * M: matrix to be rotated

    Returns:
        * Rotation matrix


    """

    h = np.array(
        [
            M[:, 0, 0] - M[:, 1, 1],
            M[:, 1, 0] + M[:, 0, 1],
            1j * (M[:, 1, 0] - M[:, 0, 1]),
        ]
    ).T
    G = np.real(h.T.dot(h))
    [eigvals, v] = np.linalg.eigh(G)
    [x, y, z] = np.sign(v[0, -1]) * v[:, -1]

    r = np.sqrt(x**2 + y**2 + z**2)
    c = np.sqrt((x + r) / (2 * r))
    s = (y - 1j * z) / np.sqrt(2 * r * (x + r))

    R = np.array([[c, np.conjugate(s)], [-s, np.conjugate(c)]])

    return R


def joint_diagonalization(C, V=None, eps=1e-3, max_iter=1000, verbose=-1):
    """
    Joint diagonalization of a set of matrices C

    Attributes:
        * C: set of symmetric matrices
        * V:  a priori eigan-vectors of the matrices C (default=None)
        * eps: tolerance for stopping criteria (default=1e-3)
        * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

    Returns:
        * V: a posteriori eigen-vectors of the matrices C
        * C: diagonalized matrices

    """

    d = C.shape[1]
    list_pairs = list(itertools.combinations(range(d), 2))

    if V is None:
        V = np.eye(d) + 1j * np.zeros((d, d))

    O_cs = np.sum([off_frobenius(c) for c in C])
    counter = 0

    if verbose > 0:
        print("Iter: {:.0f}, Diagonalization: {:.2f}".format(counter, O_cs))

    diff = np.inf

    while (diff > eps) and (counter < max_iter):
        counter += 1
        for i, j in list_pairs:
            V_ = np.eye(d) + 1j * np.zeros((d, d))
            idx = (slice(None),) + np.ix_([i, j], [i, j])
            R = rotation(C[idx])
            V_[np.ix_([i, j], [i, j])] = V_[np.ix_([i, j], [i, j])].dot(R)
            V = V.dot(V_.T)
            C = np.matmul(np.matmul(V_, C), V_.T)

        O_cs_new = np.sum([off_frobenius(c) for c in C])
        diff = np.abs(O_cs - O_cs_new)

        if verbose > 0:
            print("Iter: {:.0f}, Diagonalization: {:.2f}".format(counter, O_cs))
        O_cs = O_cs_new

    # Only for real signals, unmixing matrix is real
    V = np.real(V)
    return V, C, counter


class SOBI(TransformerMixin, BaseEstimator):
    """

    Linear ICA for time series data using joint diagonalization of the lagged-autocovariance matrices

    """

    def __init__(self, lags=1, eps=1e-3, max_iter=1000):

        self.lags = lags
        self.eps = eps
        self.max_iter = max_iter
        self.is_fitted_ = False
        self.counter = 0

    def fit(self, X):
        """

        Attributes:
            * X: time series data (dimension: time x variables)
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        """

        X_white, U, d = whitening(X.T)
        C = time_lagged_autocov(X_white, self.lags)
        C = C + 1j * np.zeros_like(C)
        V, C, self.counter = joint_diagonalization(
            C, eps=self.eps, max_iter=self.max_iter
        )
        self.W = (V.T).dot((U / d).T)

        self.is_fitted_ = True

    def transform(self, X):
        """

        Attributes:
            * X: time series data (dimension: time x variables)

        Returns:
            * Estimated sources

        """

        check_is_fitted(self, "is_fitted_")
        return X.dot(self.W.T)

    def fit_transform(self, X):
        """

        Attributes:
            * X: time series data (dimension: time x variables)
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        Returns:
            * Estimated sources

        """

        self.fit(X)
        return self.transform(X)
