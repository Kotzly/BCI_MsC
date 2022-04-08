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

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

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


train_filepaths = list(Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf").glob("*0[1,3,4,7,8]T.gdf"))
test_filepaths = list(Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf").glob("*0[2,9,6,5]T.gdf"))


ICA_N_COMPONENTS = None
CSP_N_COMPONENTS = 12
PICKS = ["ICA{}".format(str(i).rjust(3, "0")) for i in range(22 if ICA_N_COMPONENTS is None else ICA_N_COMPONENTS)]


class Averager(BaseEstimator):
    def __init__(self):
        super(Averager).__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x.mean(axis=1)


class PSD(BaseEstimator):
    def __init__(self, **kwargs):
        super(PSD).__init__()
        self.kwargs = kwargs

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        psds, freqs = psd_multitaper(x, ** self.kwargs)
        self.freqs = freqs
        return psds


@ex.capture
def run_ica_experiment(_run, method_idx):

    train_epochs = BCI_IV_Comp_Dataset.load_dataset(
        train_filepaths,
        as_epochs=True,
        concatenate=True,
        drop_bad=True,
        return_metadata=False,
        tmin=-1.,
        tmax=3.
    )
    train_epochs.load_data().filter(l_freq=None, h_freq=40)
    test_epochs_list, test_metadata_list = BCI_IV_Comp_Dataset.load_dataset(
        test_filepaths,
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

    print("Loaded test files:", len(test_epochs_list))
    results = dict()
    for method in methods:
        print("Running for method", method)
        results[method] = list()
        clf = make_pipeline(
            PSD(fmin=0.1, fmax=40, picks=PICKS),
            Averager(),
            MinMaxScaler(),
            SVC(
                C=4
            )
        )

        start = time.time()
        ICA = get_ica_instance(method, n_components=ICA_N_COMPONENTS)
        ICA.fit(train_epochs)
        print("\tICA fitted!")
        duration = time.time() - start
        transformed_train_epochs = ICA.get_sources(train_epochs)

        for i, (epochs, mdata) in enumerate(zip(test_epochs_list, test_metadata_list)):
            print("\t", i, mdata["id"])

            epochs = epochs.copy().load_data().filter(l_freq=None, h_freq=40).resample(90.)
            transformed_test_epochs = ICA.get_sources(epochs)

            scores = dict()
            signal = np.hstack(transformed_test_epochs.get_data())
            for fn_name in SCORING_FN_DICT:
                score = apply_pairwise_parallel(signal, SCORING_FN_DICT[fn_name])
                scores[fn_name] = score

            X_train, Y_train = transformed_train_epochs, transformed_train_epochs.events[:, 2]
            X_test, Y_test = transformed_test_epochs, transformed_test_epochs.events[:, 2]

            try:
                clf.fit(X_train, Y_train)
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
