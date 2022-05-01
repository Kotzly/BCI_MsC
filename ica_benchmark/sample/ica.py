import numpy as np
from scipy.stats import norm


def sample_ica_data(N=20000, n_electrodes=6, seed=42):

    np.random.seed(seed)

    rv = norm()
    x = np.linspace(norm.ppf(1e-15), norm.ppf(1 - 1e-3), N)
    x2 = np.linspace(norm.ppf(1e-3), norm.ppf(1 - 1e-15), N)
    dist = (rv.pdf(x))** 6 * 80

    artifact = dist

    sources = np.random.normal(0, 0.025, size=(1, N))
    sources = np.concatenate(
        [
            sources,
            artifact.reshape(1, N) * np.sin(np.linspace(0, 500, N)).reshape(1, N),
            np.sin(np.linspace(0, 400, N)).reshape(1, N) / 5,
            np.sin(np.linspace(0, 10, N) ** 2).reshape(1, N) / 5,
        ],
        axis=0
    )
    N_sources = 4
    W = np.random.normal(0, .5, size=(n_electrodes, N_sources))
    W += .1 * np.sign(W)
    X = W @ sources
    noise = np.random.normal(0, 0.01, size=X.shape)
    X += noise

    return X, sources, W
