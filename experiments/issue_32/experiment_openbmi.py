import numpy as np
from pathlib import Path
import time
import os

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score
from sklearn.pipeline import make_pipeline
from ica_benchmark.processing.ica import get_ica_instance
from ica_benchmark.processing.orica_code import ORICA
from ica_benchmark.io.load import OpenBMI_Dataset

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from mne import Epochs, concatenate_epochs, BaseEpochs

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SequentialFeatureSelector
import random

from utils import get_classifier, load_subject_epochs, PSD
from utils import alg_rename

from colorama import Fore, Style
from itertools import cycle
import pickle

COLORS = [
    Fore.BLUE,
    Fore.CYAN,
    Fore.GREEN,
    Fore.MAGENTA,
    Fore.RED,
    Fore.BLACK,
    Fore.WHITE,
    Fore.YELLOW,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTRED_EX,
    Fore.LIGHTBLACK_EX,
    Fore.LIGHTWHITE_EX,
    Fore.LIGHTYELLOW_EX,
]


def is_ica_stochastic(ica_method):
    is_orica = "orica" in ica_method
    is_det_prep = ica_method in ("sobi", "jade", "none", None, "pca", "whitening")
    return not (is_orica or is_det_prep)


def is_stochastic(ica_method, clf_method):
    if not is_ica_stochastic(ica_method):
        if clf_method in ("lda", "knn", "gaussian_nb"):
            return False
    return True


def repeat_deterministic(results_df, times=1):
    original_df = results_df.copy()
    results_df = results_df.copy()
    results_df["duplicated"] = 0
    keys = ["algorithm", "classifier"]

    non_repeated = original_df.groupby(keys, as_index=False).nunique().query("run == 1")
    for _, (alg, clf) in non_repeated[keys].iterrows():
        for i in range(times):
            sliced_df = original_df.query(
                "(algorithm == @alg) & (classifier == @clf)"
            ).copy()
            if i == sliced_df.run.unique().item():
                continue
            sliced_df.loc[:, ["run"]] = i
            sliced_df.loc[:, ["duplicated"]] = 1
            results_df = pd.concat([results_df, sliced_df], axis=0)
    return results_df.reset_index(drop=True)


def get_PSD(data, channels, sfreq=250, len_size=250):
    if isinstance(data, BaseEpochs):
        psd = PSD(
            channels=channels,
            picks=data.ch_names,
            n_fft=1 * len_size,
            n_overlap=len_size // 4,
            n_per_seg=1 * len_size,
            average="mean",
            window="hamming",
            proj=False,
        )
    elif isinstance(data, np.ndarray):
        psd = PSD(
            channels=channels,
            sfreq=sfreq,
            n_fft=1 * len_size,
            n_overlap=len_size // 4,
            n_per_seg=1 * len_size,
            average="mean",
            window="hamming",
        )
    else:
        raise Exception("Data is neither Epochs or ndarray instance")
    return psd


def save_artifact(artifact, name, save_folder, uid, ica_method, clf_method, n_run):
    save_folder = save_folder / f"{uid}"

    for subfolder in (ica_method, clf_method):
        if subfolder is not None:
            save_folder = save_folder / subfolder

    save_folder.mkdir(parents=True, exist_ok=True)

    artifact_path = save_folder / f"{name}_{n_run}.pkl"

    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)


def load_subject_epochs(dataset, uid, load_kwargs):
    train_epochs_1, train_labels_1 = dataset.load_subject(
        uid, **load_kwargs, run=1, session=1
    )
    train_epochs_2, train_labels_2 = dataset.load_subject(
        uid, **load_kwargs, run=2, session=1
    )

    test_epochs_1, test_labels_1 = dataset.load_subject(
        uid, **load_kwargs, run=1, session=2
    )
    test_epochs_2, test_labels_2 = dataset.load_subject(
        uid, **load_kwargs, run=2, session=2
    )

    train_epochs = concatenate_epochs([train_epochs_1, train_epochs_2])
    test_epochs = concatenate_epochs([test_epochs_1, test_epochs_2])

    train_labels = np.concatenate([train_labels_1, train_labels_2])
    test_labels = np.concatenate([test_labels_1, test_labels_2])

    filter_kwargs = dict(method="iir", iir_params=dict(order=5, ftype="butter"))

    # Filtering
    train_epochs = train_epochs.filter(8, 30, **filter_kwargs)
    test_epochs = test_epochs.filter(8, 30, **filter_kwargs)

    return (train_epochs, train_labels), (test_epochs, test_labels)


def run(
    dataset,
    uid,
    save_folder,
    ica_methods=None,
    clf_methods=None,
    channels=None,
    n_runs=10,
    random_state=1,
):
    np.random.seed(random_state)
    random.seed(random_state)

    ica_method_color_dict = {
        ica_method: color for ica_method, color in zip(ica_methods, cycle(COLORS))
    }
    clf_method_color_dict = {
        clf_method: color for clf_method, color in zip(clf_methods, cycle(COLORS))
    }

    ica_methods = ica_methods or [None]
    clf_methods = clf_methods or ["lda"]

    load_kwargs = dict(reject=False, tmin=1.0, tmax=3.5)
    (train_epochs, train_labels), (test_epochs, test_labels) = load_subject_epochs(
        dataset, uid, load_kwargs=load_kwargs
    )

    n_channels = len(train_epochs.ch_names)

    results = list()

    len_size = int(train_epochs.info["sfreq"])

    for ica_method in ica_methods:
        for n_run in range(n_runs):
            run_seed = random_state + n_run

            # Every first run, run the ICA
            # If the ICA is non-deterministic, run it
            refit_ica = is_ica_stochastic(ica_method) or (n_run == 0)

            pp_end = pp_start = psd_end = psd_start = 0
            if refit_ica:
                x_train, y_train = train_epochs.copy(), train_labels
                x_test, y_test = test_epochs.copy(), test_labels

                if ica_method in ("none", None):
                    x_train = x_train.get_data()
                    x_test = x_test.get_data()
                elif "orica" in ica_method:
                    x_train = x_train.get_data()
                    x_test = x_test.get_data()
                    n_sub = int(ica_method.split(" ")[-1])

                    ICA = ORICA(
                        mode="decay",
                        n_channels=n_channels,
                        block_update=True,
                        size_block=8,
                        stride=8,
                        lm_0=0.995,
                        lw_0=0.995,
                        gamma=0.6,
                        n_sub=n_sub,
                    )

                    ICA.fit(x_train)
                    save_artifact(ICA, "ica", save_folder, uid, ica_method, None, n_run)

                    n_epochs, n_channels, n_times = x_train.shape
                    x = x_train.transpose(1, 0, 2).reshape(n_channels, -1)
                    x_train = (
                        ICA.transform(
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
                        ICA.transform(x, scaling=1e6, save=False, warm_start=True)
                        .reshape(n_channels, n_epochs, n_times)
                        .transpose(1, 0, 2)
                    )
                    save_artifact(
                        ICA, "ica_end", save_folder, uid, ica_method, None, n_run
                    )

                else:
                    ICA = get_ica_instance(ica_method, random_state=run_seed)
                    ICA.fit(x_train)
                    x_train = ICA.transform(x_train)
                    x_test = ICA.transform(x_test)
                    save_artifact(ICA, "ica", save_folder, uid, ica_method, None, n_run)

                psd = get_PSD(
                    x_train,
                    channels=train_epochs.ch_names,
                    sfreq=train_epochs.info["sfreq"],
                    len_size=len_size,
                )

                # Feature extraction
                psd_start = time.time()
                psd.fit(x_train)
                x_train = psd.transform(x_train)
                x_test = psd.transform(x_test)

                features_names = x_train.columns.tolist()
                psd_end = time.time()

                # Feature scaling
                scaling = make_pipeline(
                    StandardScaler(),
                )
                x_train = scaling.fit_transform(x_train, y_train)
                x_test = scaling.transform(x_test)

                x_train = pd.DataFrame(x_train, columns=features_names)
                x_test = pd.DataFrame(x_test, columns=features_names)

                # Feature selection
                pp_start = time.time()
                SFS = SequentialFeatureSelector(
                    LogisticRegression(),
                    direction="forward",
                    cv=4,
                    scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
                    n_features_to_select="auto",
                    tol=0.01,
                )

                SFS.fit(x_train, y_train)

                save_artifact(SFS, "sfs", save_folder, uid, ica_method, None, n_run)
                x_train = SFS.transform(x_train)
                x_test = SFS.transform(x_test)
                pp_end = time.time()

            for clf_method in clf_methods:
                if not is_stochastic(ica_method, clf_method) and n_run > 0:
                    # Save time
                    continue

                print(
                    "[{}|{} : {}/{}]{}({})".format(
                        "{}{}{}".format(
                            ica_method_color_dict[ica_method],
                            ica_method,
                            Style.RESET_ALL,
                        ),
                        "{}{}{}".format(
                            clf_method_color_dict[clf_method],
                            clf_method,
                            Style.RESET_ALL,
                        ),
                        n_run + 1,
                        n_runs,
                        f" {Fore.RED}-NEW-{Style.RESET_ALL} " if refit_ica else "",
                        "ND" if is_stochastic(ica_method, clf_method) else "D",
                    ),
                    end="",
                )

                classifier, clf_param_grid = get_classifier(
                    clf_method, random_state=run_seed
                )

                # Classifier hyperparam opt
                gs_cv = GridSearchCV(
                    classifier,
                    param_grid=clf_param_grid,
                    cv=4,
                    scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
                    error_score=-1,
                    refit=True,
                    n_jobs=os.cpu_count() - 1,
                    verbose=0,
                )
                fit_start = time.time()
                gs_cv.fit(x_train, y_train)
                fit_end = time.time()

                pred = gs_cv.predict(x_test)

                results.append(
                    [
                        n_run,
                        ica_method,
                        clf_method,
                        uid,
                        accuracy_score(y_test, pred),
                        balanced_accuracy_score(y_test, pred),
                        cohen_kappa_score(y_test, pred),
                        fit_end - fit_start,
                        pp_end - pp_start,
                        str(
                            {
                                **gs_cv.best_params_,
                                "selected_features": SFS.feature_names_in_.tolist(),
                            }
                        ),
                    ]
                )
                n_selected_features = SFS.support_.sum()
                print(
                    ", Took {:.2f}s/{:.2f}s/{:.2f}s, selected {}".format(
                        psd_end - psd_start,
                        pp_end - pp_start,
                        fit_end - fit_start,
                        n_selected_features,
                    )
                )
                print("\tKappa", "{:.3f}".format(cohen_kappa_score(y_test, pred)))

    columns = [
        "run",
        "algorithm",
        "classifier",
        "uid",
        "acc",
        "bas",
        "kappa",
        "clf_fit_time",
        "preprocess_fit_time",
        "selected_features",
    ]
    results_df = pd.DataFrame(results, columns=columns)
    return results_df


if __name__ == "__main__":
    openbmi_dataset_folderpath = Path("/home/paulo/Documents/datasets/OpenBMI/edf/")
    dataset = OpenBMI_Dataset(openbmi_dataset_folderpath)

    clf_methods = [
        "mlp",
        "random_forest",
        "extra_trees",
        "gaussian_nb",
        "lda",
        "svm_sigmoid",
        "svm_poly",
        "svm_linear",
        "svm_rbf",
        "logistic_l2",
        "logistic_l1",
        "logistic",
    ]

    N_RUNS = 10

    ica_methods = [
        "none",
        "orica 0",
        "orica 1",
        "ext_infomax",
        "infomax",
        "sobi",
        "jade",
        "picard",
        "fastica",
        "picard_o",
        # "whitening",
        # "pca",
    ]

    deterministic_results = dict()
    save_folder = Path("results_openbmi")
    results_filepath = save_folder / "results.csv"

    for uid in list(dataset.list_uids()):
        if (save_folder / uid).exists():
            print(uid, "already exists")
            continue

        subject_results_dict = run(
            dataset,
            uid,
            save_folder,
            ica_methods=ica_methods,
            clf_methods=clf_methods,
            n_runs=N_RUNS,
        )
        deterministic_results[uid] = subject_results_dict

        results_df = pd.concat(list(deterministic_results.values()), axis=0)
        results_df = repeat_deterministic(results_df, times=N_RUNS)

        results_df.rename(
            columns={
                "kappa": "Kappa",
                "acc": "Acc.",
                "bas": "Bal. Acc.",
            },
            inplace=True,
        )
        results_df.algorithm = results_df.algorithm.apply(alg_rename)

        results_df.to_csv(results_filepath, index=False)
