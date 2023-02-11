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
from sklearn.model_selection import KFold

from ica_benchmark.processing.ica import get_ica_instance
from ica_benchmark.processing.orica_code import ORICA
from ica_benchmark.io.load import OpenBMI_Dataset
from ica_benchmark.split.split import Splitter

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


def run(splits, uid, ica_methods=None, clf_methods=None, channels=None, n_runs=10, random_state=1):
    np.random.seed(random_state)
    random.seed(random_state)

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

    filter_kwargs = dict(
        method="iir",
        iir_params=dict(
            order=5,
            ftype="butter"
        )
    )
    channels = ["FC" + str(s) for s in [5, 3, 1, 2, 4, 6]]
    channels += ["C" + str(s) for s in [5, 3, 1, 2, 4, 6]]
    channels += ["CP" + str(s) for s in [5, 3, 1, 2, 4, 6]]

    subject_number = int(uid)
    train_epochs, test_epochs = splitter.load_from_splits(splits)
    train_labels, test_labels = train_epochs.events[:, 2], test_epochs.events[:, 2]

    train_epochs = train_epochs.pick(channels).filter(8, 30, **filter_kwargs)
    test_epochs = test_epochs.pick(channels).filter(8, 30, **filter_kwargs)

    n_channels = len(channels)

    results = list()
    print("[{}]".format(subject_number))

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

                else:
                    ICA = get_ica_instance(ica_method, random_state=run_seed)
                    ICA.fit(x_train)
                    x_train = ICA.transform(x_train)
                    x_test = ICA.transform(x_test)

                psd = get_PSD(
                    x_train,
                    sfreq=train_epochs.info["sfreq"],
                    len_size=len_size
                )

                # Feature extraction
                psd_start = time.time()
                psd.fit(x_train)
                x_train = psd.transform(x_train)
                x_test = psd.transform(x_test)
                psd_end = time.time()

                # Feature scaling
                scaling = make_pipeline(
                    ConcatenateChannelsPSD(),
                    StandardScaler(),
                )
                x_train = scaling.fit_transform(x_train)
                x_test = scaling.transform(x_test)

                # Feature selection
                pp_start = time.time()
                SFS = SequentialFeatureSelector(
                    LDA(),
                    direction='forward',
                    cv=4,
                    scoring=make_scorer(cohen_kappa_score, greater_is_better=True),
                    n_features_to_select="auto",
                    tol=0.01
                )
                preprocessing = make_pipeline(SFS)
                preprocessing.fit(x_train, y_train)
                x_train = preprocessing.transform(x_train)
                x_test = preprocessing.transform(x_test)
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

                pred = gs_cv.predict(x_test)

                results.append(
                    [
                        n_run,
                        ica_method,
                        clf_method,
                        subject_number,
                        accuracy_score(y_test, pred),
                        balanced_accuracy_score(y_test, pred),
                        cohen_kappa_score(y_test, pred),
                        fit_end - fit_start,
                        pp_end - pp_start,
                        str({**gs_cv.best_params_, "selected_features": list(SFS.support_.astype(int))})
                    ]
                )
                n_selected_features = SFS.support_.sum()
                print(", Took {:.2f}s/{:.2f}s/{:.2f}s, selected {}".format(psd_end - psd_start, pp_end - pp_start, fit_end - fit_start, n_selected_features))
                print("\tKappa", "{:.3f}".format(results[-1][-4]))

    columns = ["run", "algorithm", "classifier", "uid", "acc", "bas", "kappa", "clf_fit_time", "preprocess_fit_time", "hyperparameters"]
    results_df = pd.DataFrame(results, columns=columns)
    return results_df


if __name__ == "__main__":

    filterwarnings("ignore", category=RuntimeWarning)

    openbmi_dataset_folderpath = Path('/home/paulo/Documents/datasets/OpenBMI/edf/')
    dataset = OpenBMI_Dataset(openbmi_dataset_folderpath)
    save_path = Path("./openbmi_intra")
    save_path.mkdir(exist_ok=True)

    splitter = Splitter(
        dataset,
        uids=dataset.list_uids(),
        sessions=[1],
        runs=[1, 2],
        load_kwargs=dict(
            reject=False,
            tmin=1,
            tmax=3.5
        ),
        splitter=KFold(3),
        intra_session_shuffle=False,
        fold_sizes=None
    )

    # clf_methods = ["mlp", "random_forest", "extra_trees", "gaussian_nb", "lda", "svm_sigmoid", "svm_poly", "svm_linear", "svm_rbf", "logistic_l2", "logistic_l1", "logistic"]
    # clf_methods = ["random_forest", "extra_trees", "gaussian_nb", "lda", "svm_sigmoid", "svm_poly", "svm_linear", "svm_rbf", "logistic_l2", "logistic_l1", "logistic"]
    clf_methods = ["random_forest", "gaussian_nb", "lda", "logistic"]

    N_RUNS = 3

    # ica_methods = ["none", "orica 0", "orica 1", "ext_infomax", "sobi", "jade", "fastica", "picard_o"]
    ica_methods = ["none", "orica 0", "orica 1", "ext_infomax", "sobi", "fastica", "picard"]

    results_list = list()
    mode = "intra_session_intra_run_merged"
    print(mode.upper())
    splitter.fold_sizes = [0.5, 0.5]
    splits_iterable = splitter.yield_splits_epochs(mode=mode)
    for i, splits in enumerate(splits_iterable):
        train_session = np.unique(splits[0]["session"]).item()

        uid = splits[0]["uid"][0]
        subject_results_df = run(splits, uid, ica_methods=ica_methods, clf_methods=clf_methods, n_runs=N_RUNS)
        subject_results_df = repeat_deterministic(subject_results_df, times=N_RUNS)
        subject_results_df["train_session"] = train_session

        results_list.append(subject_results_df)
        results_df = pd.concat(
            results_list,
            axis=0
        )

        # results_df["train_run"] = train_run
        # results_df["test_run"] = test_run

        results_df.rename(
            columns={
                "kappa": "Kappa",
                "acc": "Acc.",
                "bas": "Bal. Acc.",
            },
            inplace=True
        )
        results_df.algorithm = results_df.algorithm.apply(alg_rename)

        results_df.to_csv(save_path / f"results_splitting_{mode}.csv", index=False)
