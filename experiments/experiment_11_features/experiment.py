import numpy as np
from pathlib import Path
import time

from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from ica_benchmark.scoring import SCORING_FN_DICT, apply_pairwise_parallel
from ica_benchmark.processing.ica import get_all_methods, get_ica_instance

from ica_benchmark.io.load import BCI_IV_Comp_Dataset

from sacred.observers import MongoObserver, FileStorageObserver
from sacred import Experiment

import json

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from mne.time_frequency import psd_multitaper


ex = Experiment("experiment")
ex.observers.append(
    FileStorageObserver('my_runs')
)
ex.observers.append(
    MongoObserver(
        url='mongodb://admin:admin@localhost:27017',
        db_name='sacred',
    )
)


filepaths = list(Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf").glob("*T.gdf"))


ICA_N_COMPONENTS = None
TRAIN_PERCENTAGE = .75
TEST_PERCENTAGE = 1 - TRAIN_PERCENTAGE
# PICKS = ["ICA{}".format(str(i).rjust(3, "0")) for i in range(22 if ICA_N_COMPONENTS is None else ICA_N_COMPONENTS)]
# PICKS = ["EEG-C3", "EEG-C4", "EEG-Fz", "EEG-Cz", "EEG-Pz"]
PICKS = ["EEG-C3", "EEG-C4", "EEG-Cz"]

PSD_PICKS = ["ICA{}".format(str(i).rjust(3, "0")) for i in range(len(PICKS) if ICA_N_COMPONENTS is None else ICA_N_COMPONENTS)]
SEED = 42


class AverageFrequencyBand(BaseEstimator):
    DEFAULT_BANDS = [
        (8, 13),
        (13, 25)
    ]

    def __init__(self, bands=None, **kwargs):
        super(AverageFrequencyBand).__init__()
        if bands is None:
            bands = self.DEFAULT_BANDS

        self.kwargs = kwargs
        self.bands = bands
        self.check_bands()

    def check_bands(self):
        for lf, hf in self.bands:
            assert lf <= hf, "The bands are not valid: {}".format(self.bands)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        psds, freqs = psd_multitaper(x, ** self.kwargs)
        features = list()
        for lf, hf in self.bands:
            if hf > freqs.max():
                continue
            feature = psds[:, :, np.bitwise_and(freqs >= lf, freqs <= hf)].sum(axis=2)
            features.append(feature)

        features = np.concatenate(features, axis=1)
        return features


def preprocess_epochs(epochs, picks=PICKS, inplace=False):
    if not inplace:
        epochs = epochs.copy()

    return epochs.load_data().pick(picks).filter(l_freq=8, h_freq=25).resample(100)


@ex.capture
def run_ica_experiment(_run, method_idx):

    # filepaths = Path(r"C:\Users\paull\Documents\GIT\BCI_MsC\notebooks\BCI_Comp_IV_2a\BCICIV_2a_gdf/").glob("*T.gdf")

    dataset, metadata = BCI_IV_Comp_Dataset.load_dataset(
        filepaths,
        as_epochs=True,
        concatenate=False,
        drop_bad=True,
        return_metadata=True,
        tmin=-1.,
        tmax=3.
    )

    all_methods = get_all_methods()
    methods = all_methods if method_idx is None else [all_methods[method_idx]]
    name = "" if method_idx is None else "_{}".format(all_methods[method_idx])
    print("Using methods", methods)

    results = dict()
    for method in methods:
        print("Running for method", method)
        clf = make_pipeline(
            AverageFrequencyBand(fmin=0.1, fmax=30, picks=PSD_PICKS),
            StandardScaler(),
            SVC(
                C=4
            )
        )
        results[method] = list()
        for i, (epochs, mdata) in enumerate(zip(dataset, metadata)):
            print("\t", i, mdata["id"])
            ICA = get_ica_instance(method, n_components=ICA_N_COMPONENTS, random_state=SEED, max_iter=5000)
            start = time.time()

            epochs = preprocess_epochs(epochs)

            n_epochs = len(epochs)
            epochs_idx = list(range(n_epochs))
            train_epochs_idx, test_epochs_idx = train_test_split(epochs_idx, test_size=TEST_PERCENTAGE, random_state=SEED)

            train_epochs = epochs.copy().drop(test_epochs_idx)
            test_epochs = epochs.copy().drop(train_epochs_idx)

            ICA.fit(train_epochs)
            transformed_train_epochs = ICA.get_sources(train_epochs)
            transformed_test_epochs = ICA.get_sources(test_epochs)

            duration = time.time() - start

            scores = dict()
            signal = np.hstack(transformed_test_epochs.get_data())
            for fn_name in SCORING_FN_DICT:
                score = apply_pairwise_parallel(signal, SCORING_FN_DICT[fn_name])
                scores[fn_name] = score

            X, Y = transformed_train_epochs, transformed_train_epochs.events[:, 2]
            X_test, Y_test = transformed_test_epochs, transformed_test_epochs.events[:, 2]

            try:
                clf.fit(X, Y)
            except Exception as e:
                print("\t\tFailed during fit:", str(e))
                results[method].append(
                    {
                        "id": mdata["id"],
                        "score": None,
                        "bas": None,
                        "duration": duration
                    }
                )
                continue

            pred = clf.predict(X_test)
            bas = balanced_accuracy_score(Y_test, pred)
            results[method].append(
                {
                    "id": mdata["id"],
                    "score": scores,
                    "bas": bas,
                    "duration": duration
                }
            )

            del epochs, transformed_train_epochs, transformed_test_epochs

    results_filepath = f"./results{name}.json"
    with open(results_filepath, "w") as json_file:
        json.dump(results, json_file, indent=4)

    _run.add_artifact(results_filepath, content_type="json")


@ex.config
def cfg():
    method_idx = None


@ex.automain
def main(_run):
    run_ica_experiment(_run)
