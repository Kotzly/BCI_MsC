from pathlib import Path
import mne
import numpy as np
from mne.decoding import CSP
from mne.time_frequency import psd_array_welch, psd_welch
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

TEST_LABEL_FOLDER = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/")
ALL_CLF_METHODS = [
    "lda",
    "svm_rbf",
    "svm_linear",
    "svm_poly",
    "svm_sigmoid",
    "knn",
    "random_forest",
    "extra_trees",
    "gaussian_nb",
    "mlp",
    "logistic",
]


def cue_name(cue):
    return {
        0: "Left hand",
        1: "Right hand",
        2: "Foot",
        3: "Tongue",
    }[cue]


class ConcatenateChannelsPSD(BaseEstimator):
    def __init__(self):
        super(ConcatenateChannelsPSD).__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        n = len(x)
        return x.reshape(n, -1)


class PSD(BaseEstimator):
    # [TODO] Mantain feature names (e.g. '{channel}_{band}')
    BANDS_DICT = {
        "mu": (8, 13),
        "beta": (13, 25),
    }

    def __init__(self, channels, **kwargs):
        super(PSD).__init__()
        self.channels = channels
        self.kwargs = kwargs

    def set_params(self, **params):
        for param in params:
            assert params in ["picks", "n_fft", "n_overlap", "n_per_seg"]
        self.kwargs.update(params)

    def get_params(self, *args, **kwargs):
        return self.kwargs

    def fit(self, x, y=None):
        self.feature_names = [f"{channel}_{band}" for channel in self.channels for band in self.BANDS_DICT]
        return self

    def transform(self, x, y=None):
        if isinstance(x, list):
            x = mne.concatenate_epochs(x)
            psds, freqs = psd_welch(x, **self.kwargs)
        if isinstance(x, mne.Epochs):
            psds, freqs = psd_welch(x, **self.kwargs)
        if isinstance(x, np.ndarray):
            psds, freqs = psd_array_welch(x, **self.kwargs)
        if ("average" in self.kwargs) and (self.kwargs["average"] is None):
            psds = psds.sum(axis=3)
        self.freqs = freqs

        band_spectras = list()
        for band, (lfreq, hfreq) in self.BANDS_DICT.items():
            band_spectra = psds[:, :, (freqs >= lfreq) & (freqs < hfreq)]
            band_spectras.append(band_spectra.sum(axis=2, keepdims=True))

        band_spectras = np.concatenate(band_spectras, axis=2)
        features = pd.DataFrame(
            band_spectras.reshape(len(band_spectras), -1),
            columns=self.feature_names
        )

        return features


def load_subject_epochs(dataset, uid, load_kwargs):
    train_epochs, train_labels = dataset.load_subject(
        uid, **load_kwargs, run=1, session=1
    )
    test_epochs, test_labels = dataset.load_subject(
        uid, **load_kwargs, run=1, session=2
    )

    return (train_epochs, train_labels), (test_epochs, test_labels)


# Other
def get_corr_features(x):
    features = list()
    for i in range(len(x)):
        corr_matrix = np.corrcoef(x[i])
        idx1, idx2 = np.triu_indices(corr_matrix.shape[0], 1)
        features.append(corr_matrix[idx1, idx2].reshape(1, -1))
    return np.vstack(features)


class CSPWrapper(BaseEstimator):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.csp = None

    def fit(self, x, y):
        self.csp = CSP(n_components=self.n_components)
        self.csp.fit(x, y)
        return self

    def transform(self, x):
        return self.csp.transform(x)

    def set_params(self, **params):
        self.n_components = params["n_components"]
        self.csp = CSP(n_components=self.n_components)
        return self

    def get_params(self, deep=True):
        return dict(n_components=self.n_components)


def alg_rename(alg_name):
    alg_rename_dict = {
        "ext_infomax": "Ext. Infomax",
        "infomax": "Infomax",
        "none": "None",
        "picard": "Picard",
        "orica 0": "ORICA (0)",
        "orica 1": "ORICA (1)",
        "Orica (0)": "ORICA (0)",
        "Orica (1)": "ORICA (1)",
        "jade": "JADE",
        "sobi": "SOBI",
        "fastica": "FastICA",
    }
    return alg_rename_dict.get(alg_name, alg_name)


def get_n(x):
    return eval(str(x))["sequentialfeatureselector__n_features_to_select"]


def get_classifier(clf_method, random_state=1):
    if clf_method == "lda":
        clf = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        param_grid = dict(n_components=[1, 2, 3])
    elif clf_method == "svm_rbf":
        clf = SVC(kernel="rbf", random_state=random_state)
        param_grid = dict(C=np.logspace(-1, 3, 6), gamma=np.logspace(-1, -3, 5))
    elif clf_method == "svm_linear":
        clf = SVC(kernel="linear", random_state=random_state)
        param_grid = dict(C=np.logspace(-2, 3, 10))
    elif clf_method == "svm_poly":
        clf = SVC(kernel="poly", random_state=random_state)
        param_grid = dict(
            C=np.logspace(-1, 3, 6), gamma=np.logspace(-1, -3, 4), degree=[2, 3]
        )
    elif clf_method == "svm_sigmoid":
        clf = SVC(kernel="sigmoid", C=1, random_state=random_state)
        param_grid = dict(C=np.logspace(-1, 3, 6), gamma=np.logspace(-1, -3, 6))
    elif clf_method == "knn":
        clf = KNeighborsClassifier()
        param_grid = dict(
            n_neighbors=[2, 4, 6, 8, 10], weights=["uniform", "distance11"]
        )
    elif clf_method == "random_forest":
        clf = RandomForestClassifier(random_state=random_state)
        param_grid = dict(
            n_estimators=[10, 15, 25, 40],
            max_features=["sqrt", "log2"],
            max_depth=[2, 3],
        )
    elif clf_method == "extra_trees":
        clf = ExtraTreesClassifier(random_state=random_state)
        param_grid = dict(n_estimators=[10, 20, 30, 40], max_depth=[2, 3, 5])
    elif clf_method == "gaussian_nb":
        clf = GaussianNB()
        param_grid = dict()
    elif clf_method == "mlp":
        clf = MLPClassifier(
            random_state=random_state, max_iter=5000, validation_fraction=0.25
        )
        param_grid = dict(
            hidden_layer_sizes=[
                (),
                (10,),
                (6,),
                (4,),
                (10, 6),
                (6, 4),
                (4, 4),
            ],
            activation=["relu", "logistic"],
            # solver=["adam", "sgd"],
        )
    elif clf_method == "logistic_l2":
        clf = LogisticRegression(random_state=random_state, max_iter=3000)
        param_grid = dict(
            C=np.logspace(-2, 3, 20), penalty=["l2"], fit_intercept=[True]
        )
    elif clf_method == "logistic_l1":
        clf = LogisticRegression(random_state=random_state, max_iter=3000)
        param_grid = dict(
            C=np.logspace(-2, 3, 20),
            penalty=["l1"],
            solver=["liblinear"],
            fit_intercept=[True],
        )
    elif clf_method == "logistic":
        clf = LogisticRegression(random_state=random_state, max_iter=3000)
        param_grid = dict(penalty=["none"], fit_intercept=[True])
    else:
        raise Exception("Unknown classifier method: {}".format(clf_method))

    return clf, param_grid
