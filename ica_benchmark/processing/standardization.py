from ica_benchmark.data.utils import is_notebook
import numpy as np


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
