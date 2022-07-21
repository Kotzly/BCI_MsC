from pathlib import Path

import mne
import numpy as np
import pandas as pd
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from mne.decoding import CSP
from mne.time_frequency import psd_array_welch, psd_welch
from numpy.linalg import eig
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


TEST_LABEL_FOLDER = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/")
ALL_CLF_METHODS = ["lda", "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid", "knn", "random_forest", "extra_trees", "gaussian_nb", "mlp", "logistic"]


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
    BANDS_DICT = {
        "mu": (8, 13),
        "beta": (13, 25),
    }

    def __init__(self, **kwargs):
        super(PSD).__init__()
        self.kwargs = kwargs

    def set_params(self, **params):
        for param in params:
            assert params in ["picks", "n_fft", "n_overlap", "n_per_seg"]
        self.kwargs.update(params)

    def get_params(self, *args, **kwargs):
        return self.kwargs

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if isinstance(x, list):
            x = mne.concatenate_epochs(x)
            psds, freqs = psd_welch(x, ** self.kwargs)
        if isinstance(x, mne.Epochs):
            psds, freqs = psd_welch(x, ** self.kwargs)
        if isinstance(x, np.ndarray):
            psds, freqs = psd_array_welch(x, ** self.kwargs)
        if ("average" in self.kwargs) and (self.kwargs["average"] is None):
            psds = psds.sum(axis=3)
        self.freqs = freqs

        band_spectras = list()
        for band, (lfreq, hfreq) in self.BANDS_DICT.items():
            band_spectra = psds[:, :, (freqs >= lfreq) & (freqs < hfreq)]
            band_spectras.append(
                band_spectra.sum(axis=2, keepdims=True)
            )

        band_spectras = np.concatenate(band_spectras, axis=2)

        return band_spectras


def whitening(X, B=None):
    is_epochs = isinstance(X, mne.Epochs)
    if is_epochs:
        epochs = X.copy()
        X = X._data
        n_epochs, n_channels, n_times = X.shape
        X = X.transpose(1, 0, 2).reshape(n_channels, -1)

    if B is None:

        n, T = X.shape

        X -= X.mean(axis=1, keepdims=1)

        [D, U] = eig((X @ X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]

        # The m most significant princip. comp. by decreasing variance
        PCs = np.arange(n - 1, - 1, -1)

        # PCA
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]].T

        # --- Scaling ---------------------------------
        # The scales of the principal components
        scales = np.sqrt(Ds[PCs])
        B = np.diag(1.0 / scales) * B
        # Sphering

    X = B @ X

    if is_epochs:
        X = X.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
        epochs._data = X
        X = epochs

    return X, B


# LOADING
def preprocess_epochs(epochs, car_filter=True):

    epochs = epochs.copy()
    epochs.load_data()

    if car_filter:
        epochs.set_eeg_reference("average")

    return epochs


def load_subject_epochs(root, subject_number, test_label_folder=TEST_LABEL_FOLDER):
    train_file_path = root / "A{}T.gdf".format(str(subject_number).rjust(2, "0"))
    test_file_path = root / "A{}E.gdf".format(str(subject_number).rjust(2, "0"))
    test_label_file_path = test_label_folder / "A{}E.csv".format(str(subject_number).rjust(2, "0"))
    train_epochs = BCI_IV_Comp_Dataset.load_dataset(
        [train_file_path],
        reject=False,
        as_epochs=True,
        concatenate=False,
        drop_bad=False,
        return_metadata=False,
        tmin=0.,
        tmax=4.,
    )[0]
    test_epochs = BCI_IV_Comp_Dataset.load_dataset(
        [test_file_path],
        reject=False,
        as_epochs=True,
        concatenate=False,
        drop_bad=False,
        return_metadata=False,
        tmin=0.,
        # The last timestamp does not exist, so MNE will ignore the last epoch because it will not end in 6s
        # So here we use 5.5 seconds because there will always be 5.5 seconds after a event
        tmax=4.,
        has_labels=False
    )[0]

    train_epochs = preprocess_epochs(train_epochs, car_filter=False)
    test_epochs = preprocess_epochs(test_epochs, car_filter=False)

    train_labels = train_epochs.events[:, 2]

    test_labels = pd.read_csv(test_label_file_path, header=None).to_numpy().flatten() - 1
    assert len(test_labels) == len(test_epochs), "{} epochs | {} labels".format(len(test_epochs), len(test_labels))

    test_epochs.events[:, 2] = test_labels
    test_labels = test_epochs.events[:, 2]

    return (train_epochs, train_labels), (test_epochs, test_labels)


# Other
def get_corr_features(x):
    features = list()
    for i in range(len(x)):
        corr_matrix = np.corrcoef(x[i])
        idx1, idx2 = np.triu_indices(corr_matrix.shape[0], 1)
        features.append(
            corr_matrix[idx1, idx2].reshape(1, -1)
        )
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
        "fastica": "FastICA"
    }
    return alg_rename_dict.get(alg_name, alg_name)


def get_n(x):
    return eval(str(x))["sequentialfeatureselector__n_features_to_select"]


def extract_subject_id(s):
    return int(s[1:-1])


def get_classifier(clf_method, n_inputs=10, random_state=1):
    if clf_method == "lda":
        clf = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        param_grid = dict()
    elif clf_method == "svm_rbf":
        clf = SVC(kernel="rbf", random_state=random_state)
        param_grid = dict(
            C=np.logspace(-1, 3, 5),
            gamma=np.logspace(-1, -3, 4)
        )
    elif clf_method == "svm_linear":
        clf = SVC(kernel="linear", random_state=random_state)
        param_grid = dict(
            C=np.logspace(-1, 3, 10)
        )
    elif clf_method == "svm_poly":
        clf = SVC(kernel="poly", random_state=random_state)
        param_grid = dict(
            C=np.logspace(-1, 3, 5),
            gamma=np.logspace(-1, -3, 4),
            degree=[2, 3]
        )
    elif clf_method == "svm_sigmoid":
        clf = SVC(kernel="sigmoid", C=1, random_state=random_state)
        param_grid = dict(
            C=np.logspace(-1, 3, 5),
            gamma=np.logspace(-1, -3, 5)
        )
    elif clf_method == "knn":
        clf = KNeighborsClassifier()
        param_grid = dict(
            n_neighbors=[2, 4, 6, 8, 10],
            weights=["uniform", "distance"]
        )
    elif clf_method == "random_forest":
        clf = RandomForestClassifier(random_state=random_state)
        param_grid = dict(
            n_estimators=[5, 10, 20, 25, 30, 40],
            max_features=["auto", "sqrt", "log2"],
        )
    elif clf_method == "extra_trees":
        clf = ExtraTreesClassifier(random_state=random_state)
        param_grid = dict(
            n_estimators=[5, 10, 20, 25, 30, 40]
        )
    elif clf_method == "gaussian_nb":
        clf = GaussianNB()
        param_grid = dict()
    elif clf_method == "mlp":
        clf = MLPClassifier(random_state=random_state, max_iter=5000)
        param_grid = dict(
            hidden_layer_sizes=[
                (),
                (n_inputs,),
                (n_inputs // 2,),
                (n_inputs // 4),
                (n_inputs // 2, n_inputs // 2),
                (n_inputs // 4, n_inputs // 4),
                (n_inputs // 2, n_inputs // 2, n_inputs // 2),
            ],
            activation=["relu", "logistic"]
        )
    elif clf_method == "logistic":
        clf = LogisticRegression(random_state=random_state, max_iter=2000)
        param_grid = dict(
            C=np.logspace(0, 3, 10),
            penalty=["l2", "none"],
            fit_intercept=[True, False]
        )
    else:
        raise Exception("Unknown classifier method: {}".format(clf_method))

    return clf, param_grid
