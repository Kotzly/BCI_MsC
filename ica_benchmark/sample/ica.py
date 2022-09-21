import numpy as np
from scipy.stats import norm

def sample_ica_data(N=20000, n_electrodes=6, seed=42):

    np.random.seed(seed)

    rv = norm()
    x = np.linspace(norm.ppf(1e-15), norm.ppf(1 - 1e-3), N)
    x2 = np.linspace(norm.ppf(1e-3), norm.ppf(1 - 1e-15), N)
    dist = (rv.pdf(x)) ** 6 * 80

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


def sample_sines_data(N=20000, n_electrodes=22, seed=42):

    np.random.seed(seed)

    frequencies = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    sources = np.concatenate(
        [
            np.sin(
                np.linspace(0, 5 * i * 10, N).reshape(1, N) +
                np.random.rand() * 2 * np.pi
            ) / 5
            for i
            in frequencies
        ],
        axis=0
    )
    W = np.random.normal(0, .5, size=(n_electrodes, 10))
    W += .1 * np.sign(W)
    X = W @ sources
    noise = np.random.normal(0, 0.05, size=X.shape)
    X += noise

    return X, sources, W

def sample_iva_data(K, M, N, seed=42):
    # K number of datasets
    # M number of SVC (electrodes)
    # N number of samples
    np.random.seed(seed)

    X, S, A = list(), list(), list()
    SCVs = list()
    for m in range (M):
        cov = np.random.normal(size=(K, K))
        cov = cov @ cov.T
        SCV = np.random.multivariate_normal(np.zeros(K), cov, size=N).T
        SCVs.append(SCV)
    S = np.stack(SCVs).transpose(1, 0, 2)
    assert S.shape == (K, M, N)        
        
    A = np.random.normal(0, 1, size=(K, M, M))
    X = np.stack([a @ s for a, s in zip(A, S)])    
    
    return X, S, A
