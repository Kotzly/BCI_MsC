import numpy as np
from pathlib import Path
import time
import pandas as pd
import random

from colorama import Fore, Style
from itertools import cycle
from warnings import filterwarnings

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SequentialFeatureSelector

from ica_benchmark.processing.ica import get_ica_instance
from ica_benchmark.processing.orica_code import ORICA
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from mne import EpochsArray, create_info

from utils import get_classifier, ConcatenateChannelsPSD
from utils import alg_rename, get_PSD, is_ica_stochastic, is_stochastic, repeat_deterministic


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


def run(dataset, uid, load_kwargs, ica_methods=None, clf_methods=None, n_runs=10, random_state=1):
    np.random.seed(random_state)
    random.seed(random_state)

    # Color dict for ICAs and Classifiers
    ica_method_color_dict = {
        ica_method: color
        for ica_method, color
        in zip(ica_methods, cycle(COLORS))
    }
    clf_method_color_dict = {
        clf_method: color
        for clf_method, color
        in zip(clf_methods, cycle(COLORS))
    }

    ica_methods = ica_methods or [None]
    clf_methods = clf_methods or ["lda"]

    subject_number = int(uid)

    # Load epochs
    train_epochs = dataset.load_subject(uid, session=1, run=1, **load_kwargs)[0]
    inter_test_epochs = dataset.load_subject(uid, session=2, run=1, **load_kwargs)[0]

    train_epochs.load_data()
    inter_test_epochs.load_data()

    # Select only Left and Right hand imagery
    train_labels, inter_test_labels = train_epochs.events[:, 2], inter_test_epochs.events[:, 2]

    train_epochs = train_epochs[(train_labels == 0) | (train_labels == 1)]
    inter_test_epochs = inter_test_epochs[(inter_test_labels == 0) | (inter_test_labels == 1)]

    # Select the same data size for intra and inter protocols
    n = len(train_epochs.events[:, 2])
    intra_test_epochs = train_epochs[n // 2:]
    train_epochs = train_epochs[:n // 2]

    # Get labels
    train_labels = train_epochs.events[:, 2]
    intra_test_labels = intra_test_epochs.events[:, 2]
    inter_test_labels = inter_test_epochs.events[:, 2]

    filter_kwargs = dict(
        method="iir",
        iir_params=dict(
            order=5,
            ftype="butter"
        )
    )

    # Filtering
    train_epochs = train_epochs.filter(8, 30, **filter_kwargs)
    intra_test_epochs = intra_test_epochs.filter(8, 30, **filter_kwargs)
    inter_test_epochs = inter_test_epochs.filter(8, 30, **filter_kwargs)
    results_train = list()
    results_inter = list()
    results_intra = list()
    print(
        "[{}]: {} for train, {} for intra protocol, {} for inter protocol".format(
            subject_number,
            len(train_labels),
            len(intra_test_labels),
            len(inter_test_labels)
        )
    )

    len_size = int(train_epochs.info["sfreq"])
    sfreq = int(train_epochs.info["sfreq"])

    for ica_method in ica_methods:
        for n_run in range(n_runs):
            run_seed = random_state + n_run

            # Every first run, run the ICA
            # If the ICA is non-deterministic, run it
            refit_ica = is_ica_stochastic(ica_method) or (n_run == 0)

            pp_end = pp_start = psd_end = psd_start = 0
            if refit_ica:
                x_train, y_train = train_epochs.copy(), train_labels
                x_intra_test, y_intra_test = intra_test_epochs.copy(), intra_test_labels
                x_inter_test, y_inter_test = inter_test_epochs.copy(), inter_test_labels

                if ica_method in ("none", None):
                    pass
                elif "orica" in ica_method:

                    n_sub = int(ica_method.split(" ")[-1])

                    ICA = ORICA(
                        mode="decay",
                        n_channels=len(x_train.ch_names),
                        block_update=True,
                        size_block=8,
                        stride=8,
                        lm_0=.995,
                        lw_0=.995,
                        gamma=.6,
                        n_sub=n_sub,
                    )
                    ICA.fit(x_train)

                    n_epochs, n_channels, n_times = x_train.get_data().shape
                    x = x_train.get_data().transpose(1, 0, 2).reshape(n_channels, -1)
                    arr_train = (
                        ICA
                        .transform(
                            x,
                            scaling=1e6,
                            save=False,
                        )
                        .reshape(n_channels, n_epochs, n_times)
                        .transpose(1, 0, 2)
                    )
                    W = ICA.w.copy()
                    M = ICA.m.copy()
                    #################################################################################
                    ICA.mode = "constant"
                    ICA.lm_0, ICA.lw_0 = 0.001, 0.001
                    #################################################################################

                    ICA.m = M.copy()
                    ICA.w = W.copy()
                    n_epochs, n_channels, n_times = x_intra_test.get_data().shape
                    x = x_intra_test.get_data().transpose(1, 0, 2).reshape(n_channels, -1)
                    arr_intra_test = (
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

                    ICA.m = M.copy()
                    ICA.w = W.copy()
                    n_epochs, n_channels, n_times = x_inter_test.get_data().shape
                    x = x_inter_test.get_data().transpose(1, 0, 2).reshape(n_channels, -1)
                    arr_inter_test = (
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

                    x_train = EpochsArray(
                        arr_train,
                        create_info(
                            ["ICA{}".format(str(i).rjust(3, "0")) for i in range(arr_train.shape[1])],
                            x_train.info["sfreq"],
                            verbose=False
                        ),
                        events=x_train.events,
                        reject=None
                    )
                    x_intra_test = EpochsArray(
                        arr_intra_test,
                        create_info(
                            ["ICA{}".format(str(i).rjust(3, "0")) for i in range(arr_intra_test.shape[1])],
                            x_intra_test.info["sfreq"],
                            verbose=False
                        ),
                        events=x_intra_test.events,
                        reject=None
                    )
                    x_inter_test = EpochsArray(
                        arr_inter_test,
                        create_info(
                            ["ICA{}".format(str(i).rjust(3, "0")) for i in range(arr_inter_test.shape[1])],
                            x_inter_test.info["sfreq"],
                            verbose=False
                        ),
                        events=x_inter_test.events,
                        reject=None
                    )
                    del arr_train, arr_intra_test, arr_inter_test
                else:
                    ICA = get_ica_instance(ica_method, random_state=run_seed)
                    x_train = ICA.fit_transform(x_train)
                    x_intra_test = ICA.transform(x_intra_test)
                    x_inter_test = ICA.transform(x_inter_test)

                psd = get_PSD(
                    x_train,
                    sfreq=sfreq,
                    len_size=len_size
                )

                # Feature extraction
                psd_start = time.time()
                psd.fit(x_train)
                x_train = psd.transform(x_train)
                x_intra_test = psd.transform(x_intra_test)
                x_inter_test = psd.transform(x_inter_test)

                psd_end = time.time()

                # Feature scaling
                scaling = make_pipeline(
                    ConcatenateChannelsPSD(),
                    StandardScaler(),
                )

                x_train = scaling.fit_transform(x_train)
                x_intra_test = scaling.transform(x_intra_test)
                x_inter_test = scaling.transform(x_inter_test)

                # Feature selection
                pp_start = time.time()
                SFS = SequentialFeatureSelector(
                    LDA(),
                    direction='forward',
                    cv=4,
                    scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
                    n_features_to_select="auto",
                    tol=0.001
                )
                preprocessing = make_pipeline(SFS)
                preprocessing.fit(x_train, y_train)
                x_train = preprocessing.transform(x_train)
                x_intra_test = preprocessing.transform(x_intra_test)
                x_inter_test = preprocessing.transform(x_inter_test)
                pp_end = time.time()

            for clf_method in clf_methods:

                if not is_stochastic(ica_method, clf_method) and n_run > 0:
                    # Save time
                    continue

                print(
                    "[{}|{} : {}/{}]{}({})".format(
                        "{}{}{}".format(ica_method_color_dict[ica_method], ica_method, Style.RESET_ALL),
                        "{}{}{}".format(clf_method_color_dict[clf_method], clf_method, Style.RESET_ALL),
                        n_run + 1,
                        n_runs,
                        f" {Fore.RED}-NEW-{Style.RESET_ALL} " if refit_ica else "",
                        "ND" if is_stochastic(ica_method, clf_method) else "D"
                    ),
                    end=""
                )

                classifier, clf_param_grid = get_classifier(clf_method, random_state=run_seed)

                # Classifier hyperparam opt
                gs_cv = GridSearchCV(
                    classifier,
                    param_grid=clf_param_grid,
                    cv=4,
                    scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
                    error_score=-1,
                    refit=True,
                    n_jobs=3,
                    verbose=0
                )
                fit_start = time.time()
                gs_cv.fit(x_train, y_train)
                fit_end = time.time()

                train_pred = gs_cv.predict(x_train)
                intra_pred = gs_cv.predict(x_intra_test)
                inter_pred = gs_cv.predict(x_inter_test)

                results_train.append(
                    [
                        n_run,
                        ica_method,
                        clf_method,
                        subject_number,
                        accuracy_score(y_train, train_pred),
                        balanced_accuracy_score(y_train, train_pred),
                        cohen_kappa_score(y_train, train_pred),
                        fit_end - fit_start,
                        pp_end - pp_start,
                        str({**gs_cv.best_params_, "selected_features": list(SFS.support_.astype(int))})
                    ]
                )
                results_intra.append(
                    [
                        n_run,
                        ica_method,
                        clf_method,
                        subject_number,
                        accuracy_score(y_intra_test, intra_pred),
                        balanced_accuracy_score(y_intra_test, intra_pred),
                        cohen_kappa_score(y_intra_test, intra_pred),
                    ]
                )
                results_inter.append(
                    [
                        n_run,
                        ica_method,
                        clf_method,
                        subject_number,
                        accuracy_score(y_inter_test, inter_pred),
                        balanced_accuracy_score(y_inter_test, inter_pred),
                        cohen_kappa_score(y_inter_test, inter_pred),
                    ]
                )
                n_selected_features = SFS.support_.sum()
                print(", Took {:.2f}s/{:.2f}s/{:.2f}s, selected {}".format(psd_end - psd_start, pp_end - pp_start, fit_end - fit_start, n_selected_features))
                print(
                    "\tKappa:", "Intra:{:.3f} | Inter:{:.3f}".format(
                        results_intra[-1][-1],
                        results_inter[-1][-1]
                    )
                )

    columns = ["run", "algorithm", "classifier", "uid", "acc", "bas", "kappa", "clf_fit_time", "preprocess_fit_time", "hyperparameters"]
    train_results_df = pd.DataFrame(results_train, columns=columns)
    intra_results_df = pd.DataFrame(results_intra, columns=columns[:-3])
    inter_results_df = pd.DataFrame(results_inter, columns=columns[:-3])

    return train_results_df, intra_results_df, inter_results_df


if __name__ == "__main__":

    filterwarnings("ignore", category=RuntimeWarning)

    bci_dataset_folderpath = Path('/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/')
    bci_test_dataset_folderpath = Path('/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/')
    save_path = Path("./bcicomp_fix")
    save_path.mkdir(exist_ok=True)

    dataset = BCI_IV_Comp_Dataset(bci_dataset_folderpath, test_folder=bci_test_dataset_folderpath)
    load_kwargs = dict(
        tmin=0,
        tmax=4,
        reject=False
    )

    N_RUNS = 3
    clf_methods = ["lda"]
    ica_methods = ["orica 0", "orica 1"]
    results_lists = [list(), list(), list()]
    for uid in dataset.list_uids():

        results_dfs = run(dataset, uid, load_kwargs, ica_methods=ica_methods, clf_methods=clf_methods, n_runs=N_RUNS)
        dfs_names = ["train", "intra", "inter"]
        for name, results_df, results_list in zip(dfs_names, results_dfs, results_lists):
            results_df = repeat_deterministic(results_df, times=N_RUNS)

            results_list.append(results_df)

            results_df = pd.concat(
                results_list,
                axis=0
            )

            results_df = results_df.rename(
                columns={
                    "kappa": "Kappa",
                    "acc": "Acc.",
                    "bas": "Bal. Acc.",
                }
            )
            results_df.algorithm = results_df.algorithm.apply(alg_rename)

            results_df.to_csv(save_path / f"{name}.csv", index=False)
