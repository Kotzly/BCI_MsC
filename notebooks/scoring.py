import numpy as np
from scipy.stats import chi2_contingency
import numpy as np
from scipy.signal import coherence as coherence_
from multiprocessing import Pool

def square(x):
    return x ** 2

# def calc_MI(x, y, bins):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
#     mi = 0.5 * g / c_xy.sum()
#     return mi

def mutual_information(X, Y, bins=100):

    minX, maxX = X.min(), X.max()
    minY, maxY = X.min(), X.max()
    range1D = (min(minX, minY), max(maxX, maxY))
    range2D = (range1D, range1D)

    c_XY = np.histogram2d(X,Y,bins, range=range2D)[0]
    c_X = np.histogram(X, bins, range=range1D)[0]
    c_Y = np.histogram(Y, bins, range=range1D)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY

    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = - sum(c_normalized * np.log2(c_normalized))  
    return H

def correntropy(x, y, sigma=1):
    s = np.exp((- (x - y) ** 2 )/ (2 * sigma ** 2)) / ( sigma * np.sqrt(np.pi * 2))
    return s.mean()

def coherence(x, y):
    return coherence_(x, y, fs=250.0, window='hann', nperseg=30 * 10, noverlap=30 * 5, nfft=None)

def apply_fn(args):
    x, y, func = args
    return func(x, y)

def apply_pairwise_parallel(arr, func=mutual_information):
    n = arr.shape[1]
    res_arr = []
    args = []
    for i0 in range(n):
        for i1 in range(n):
            if i0 >= i1: continue
            args.append((arr[:, i0], arr[:, i1], func))
            
    with Pool(3) as pool:
        res_arr = pool.map(apply_fn, args)

    return np.array(res_arr).mean()

def apply_pairwise(arr, func=mutual_information):
    n = arr.shape[1]
    res_arr = []
    for i0 in range(n):
        for i1 in range(n):
            if i0 >= i1: continue
            res_arr.append(func(arr[:, i0], arr[:, i1]))
    return np.array(res_arr).mean()
