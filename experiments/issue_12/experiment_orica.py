import numpy as np
from pathlib import Path
import time

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score
from sklearn.pipeline import make_pipeline
from ica_benchmark.processing.ica import get_ica_instance
from ica_benchmark.processing.orica_code import ORICA

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SequentialFeatureSelector
from itertools import product
import random

from utils import get_classifier, load_subject_epochs, whitening, PSD, ConcatenateChannelsPSD
from utils import alg_rename, extract_subject_id, get_n, ALL_CLF_METHODS


def run(filepath, ica_methods=None, clf_methods=None, channels=None, n_runs=10, random_state=1):
    np.random.seed(random_state)
    random.seed(random_state)

    ica_methods = [None] if ica_methods is None else ica_methods
    clf_methods = ["lda"] if clf_methods is None else clf_methods

    subject_number = int(filepath.name[1:3])
    (train_epochs, train_labels), (test_epochs, test_labels) = load_subject_epochs(filepath.parent, subject_number)

    selected_channels = train_epochs.ch_names if channels is None else channels

    train_epochs.pick(selected_channels)
    test_epochs.pick(selected_channels)

    results = list()
    print("[{}]".format(filepath.name))
    orica_processed_dict = dict()
    for n_run, ica_method, clf_method in product(range(n_runs), ica_methods, clf_methods):
        print("[{}/{}] Method: {}, {}".format(n_run + 1, n_runs, ica_method, clf_method), end="")
        x_train, y_train = train_epochs.copy(), train_labels
        x_test, y_test = test_epochs.copy(), test_labels

        len_size = 250

        # if ica_method in ["sobi", "jade", "picard"]:
        # JADE and Picard needs whitening too, but
        # Used jade already whitens the signals
        # MNE whitens the signal when using PICARD
        if ica_method in ["sobi"]:
            x_train, B = whitening(x_train)
            x_test, _ = whitening(x_test, B)

        if ica_method in ["none", None]:
            x_train = x_train.get_data()
            x_test = x_test.get_data()
            psd = PSD(
                sfreq=train_epochs.info["sfreq"],
                n_fft=1 * len_size,
                n_overlap=len_size // 4,
                n_per_seg=1 * len_size,
                average="mean",
                window="hamming",
            )
        elif "orica" in ica_method:
            if ica_method in orica_processed_dict:
                (x_train, x_test) = orica_processed_dict[ica_method]
            else:
                x_train = x_train.get_data()
                x_test = x_test.get_data()
                n_sub = int(ica_method.split(" ")[-1])

                n_channels = len(selected_channels)
                ICA = ORICA(
                    mode="decay",
                    n_channels=n_channels,
                    block_update=True,
                    size_block=8,
                    stride=8,
                    lm_0=.995,
                    lw_0=.995,
                    gamma=.6,
                    n_sub=n_sub,
                )
                ICA.fit(x_train)

                n_epochs, n_channels, n_times = x_train.shape
                x = x_train.transpose(1, 0, 2).reshape(n_channels, -1)
                x_train = (
                    ICA
                    .transform(
                        x,
                        scaling=1e6,
                        save=False,
                    )
                    .reshape(n_channels, n_epochs, n_times)
                    .transpose(1, 0, 2)
                )

                #################################################################################
                ICA.mode = "constant"
                ICA.lm_0, ICA.lw_0 = 0.001, 0.001
                #################################################################################

                n_epochs, n_channels, n_times = x_test.shape
                x = x_test.transpose(1, 0, 2).reshape(n_channels, -1)
                x_test = (
                    ICA
                    .transform(
                        x,
                        scaling=1e6,
                        save=False,
                        warm_start=True
                    )
                    .reshape(n_channels, n_epochs, n_times)
                    .transpose(1, 0, 2)
                )
                orica_processed_dict[ica_method] = (x_train, x_test)

            psd = PSD(
                sfreq=train_epochs.info["sfreq"],
                n_fft=1 * len_size,
                n_overlap=len_size // 4,
                n_per_seg=1 * len_size,
                average="mean",
                window="hamming",
            )
        else:
            ICA = get_ica_instance(ica_method, random_state=random_state + n_run)

            ICA.fit(x_train)
            x_train = ICA.transform(x_train)
            x_test = ICA.transform(x_test)
            psd = PSD(
                picks=x_train.ch_names,
                n_fft=1 * len_size,
                n_overlap=len_size // 4,
                n_per_seg=1 * len_size,
                average="mean",
                window="hamming",
                proj=False
            )

        classifier, clf_param_grid = get_classifier(clf_method, random_state=random_state + n_run, n_inputs=20)

        SFS = SequentialFeatureSelector(
            LDA(n_components=1, solver="svd"),
            direction='forward',
            cv=4,
            scoring=make_scorer(cohen_kappa_score, greater_is_better=True)
        )
        param_grid = dict(
            # sequentialfeatureselector__n_features_to_select=[12],
        )
        for key, value in clf_param_grid.items():
            clf_name = classifier.__class__.__name__.lower()
            param_grid["{}__".format(clf_name) + key] = value

        psd.fit(x_train)
        x_train = psd.transform(x_train)
        x_test = psd.transform(x_test)

        clf = make_pipeline(
            ConcatenateChannelsPSD(),
            StandardScaler(),
            # SFS,
            classifier
        )

        gs_cv = GridSearchCV(
            clf,
            param_grid=param_grid,
            cv=4,
            scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
            error_score=-1,
            refit=True,
            n_jobs=4,
            verbose=0
        )

        start = time.time()
        gs_cv.fit(x_train, y_train)
        end = time.time()

        pred = gs_cv.predict(x_test)
        bas = balanced_accuracy_score(y_test, pred)
        acc = accuracy_score(y_test, pred)
        kappa = cohen_kappa_score(y_test, pred)

        results.append(
            [
                n_run,
                ica_method,
                clf_method,
                filepath.name.split(".")[0],
                acc,
                bas,
                kappa,
                end - start,
                gs_cv.best_params_
            ]
        )
        print(", Took {}s".format(end - start))
        print("Kappa", kappa)


    columns = ["run", "algorithm", "classifier", "uid", "acc", "bas", "kappa", "ica_fit_time", "best_params"]
    results = pd.DataFrame(results, columns=columns)
    return results


if __name__ == "__main__":
    TEST_LABEL_FOLDER = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/")
    root = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/")
    selected_channels = ["EEG-Fz", "EEG-C3", "EEG-C4", "EEG-Cz"]
    clf_methods = ["lda", "mlp", "svm_rbf", "svm_linear"]
    filepaths = list(sorted(root.glob("A*T.gdf")))
    N_RUNS = 8

    deterministic_methods = ["none", "orica 0", "orica 1", "ext_infomax"]
    deterministic_results = dict()

    for filepath in filepaths:
        print(filepath.name)
        subject_results_dict = run(filepath, ica_methods=deterministic_methods, clf_methods=clf_methods, n_runs=N_RUNS)
        deterministic_results[filepath.name] = subject_results_dict

    results_df = pd.concat(
        list(deterministic_results.values())
    )

    results_df.rename(
        columns={
            "kappa": "Kappa",
            "acc": "Acc.",
            "bas": "Bal. Acc.",
        },
        inplace=True
    )
    results_df.algorithm = results_df.algorithm.apply(alg_rename)
    results_df.uid = results_df.uid.apply(extract_subject_id)
    results_df["hyperparameters"] = results_df.best_params.apply(str)
    #results_df.best_params = results_df.best_params.apply(get_n)

    results_df.to_csv("results.csv", index=False)

