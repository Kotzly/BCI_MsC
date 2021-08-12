import numpy as np
from pathlib import Path
import time

from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from ica_benchmark.scoring import SCORING_FN_DICT, apply_pairwise_parallel
from ica_benchmark.processing.ica import get_all_methods, get_ica_instance

from ica_benchmark.io.load import BCI_IV_Comp_Dataset

from sacred.observers import MongoObserver, FileStorageObserver
from sacred import Experiment

import json

from sklearn.preprocessing import MinMaxScaler

from mne.decoding import Vectorizer, CSP

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


train_filepaths = list(Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf").glob("*0[1-5]T.gdf"))
test_filepaths = list(Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf").glob("*0[6-9]T.gdf"))


def run_ica_experiment(_run):

    # filepaths = Path(r"C:\Users\paull\Documents\GIT\BCI_MsC\notebooks\BCI_Comp_IV_2a\BCICIV_2a_gdf/").glob("*T.gdf")

    train_epochs = BCI_IV_Comp_Dataset.load_dataset(
        train_filepaths,
        as_epochs=True,
        concatenate=True,
        drop_bad=True,
        return_metadata=False,
        tmin=-1.,
        tmax=3.
    )

    test_epochs_list, test_metadata_list = BCI_IV_Comp_Dataset.load_dataset(
        test_filepaths,
        as_epochs=True,
        concatenate=False,
        drop_bad=True,
        return_metadata=True,
        tmin=-1.,
        tmax=3.
    )
    print("Loaded test files:", len(test_epochs_list))
    results = dict()
    for method in get_all_methods():
        print("Running for method", method)
        results[method] = list()
        clf = make_pipeline(
            CSP(n_components=12),
            Vectorizer(),
            MinMaxScaler(),
            LogisticRegression(
                penalty='l2',
                multi_class='auto'
            )
        )

        start = time.time()
        ICA = get_ica_instance(method)
        ICA.fit(train_epochs)
        duration = time.time() - start

        for i, (epochs, mdata) in enumerate(zip(test_epochs_list, test_metadata_list)):
            transformed_train_epochs = ICA.get_sources(train_epochs)
            transformed_test_epochs = ICA.get_sources(epochs)
            print(transformed_test_epochs)

            scores = dict()
            print(transformed_train_epochs.get_data().shape)
            signal = np.hstack(transformed_test_epochs.get_data())
            for fn_name in SCORING_FN_DICT:
                score = apply_pairwise_parallel(signal, SCORING_FN_DICT[fn_name])
                scores[fn_name] = score

            X_train, Y_train = transformed_train_epochs.get_data(), transformed_train_epochs.events[:, 2]
            X_test, Y_test = transformed_test_epochs.get_data(), transformed_test_epochs.events[:, 2]
            clf.fit(X_train, Y_train)
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

    with open("./results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    _run.add_artifact("./results.json", content_type="json")


@ex.automain
def main(_run):
    run_ica_experiment(_run)
