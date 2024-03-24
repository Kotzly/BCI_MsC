import numpy as np
from scipy.stats import norm


def saw_tooth_wave(N=20000, n_periods=10, seed=42):
    # Create a saw tooth wave with n_periods peaks
    # and N samples

    np.random.seed(seed)
    wave = np.linspace(0, n_periods, N) % 1
    wave -= wave.mean()
    wave = wave.reshape(1, N)

    return wave


def sample_ica_data(N=20000, n_electrodes=6, seed=42):

    np.random.seed(seed)

    rv = norm()
    x = np.linspace(norm.ppf(1e-15), norm.ppf(1 - 1e-3), N)
    dist = (rv.pdf(x)) ** 6 * 80

    artifact = dist

    noise = np.random.normal(0, 0.025, size=(1, N))

    sources = np.concatenate(
        [
            noise,
            # saw_tooth = filtered_saw_tooth_wave(N=N, n_periods=10, seed=seed),
            artifact.reshape(1, N) * np.sin(np.linspace(0, 500, N)).reshape(1, N),
            np.sin(np.linspace(0, 400, N)).reshape(1, N) / 5,
            np.sin(np.linspace(0, 10, N) ** 2).reshape(1, N) / 5,
        ],
        axis=0,
    )
    N_sources = len(sources)
    W = np.random.normal(0, 0.5, size=(n_electrodes, N_sources))
    W += 0.1 * np.sign(W)
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
                np.linspace(0, 5 * i * 10, N).reshape(1, N)
                + np.random.rand() * 2 * np.pi
            )
            / 5
            for i in frequencies
        ],
        axis=0,
    )
    W = np.random.normal(0, 0.5, size=(n_electrodes, 10))
    W += 0.1 * np.sign(W)
    X = W @ sources
    noise = np.random.normal(0, 0.05, size=X.shape)
    X += noise

    return X, sources, W
