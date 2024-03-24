import os
import pickle
import random
import time
from itertools import cycle
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from mne import BaseEpochs, Epochs, concatenate_epochs
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils import PSD, alg_rename, get_classifier, load_subject_epochs

from ica_benchmark.data.utils import apply_raw, to_tensor
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from ica_benchmark.models import EEGNet
from ica_benchmark.processing.filter import bandpass_cnt
from ica_benchmark.processing.ica import get_ica_instance
from ica_benchmark.processing.orica_code import ORICA
from ica_benchmark.processing.standardization import exponential_standardize

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
    keys = ["algorithm"]

    non_repeated = original_df.groupby(keys, as_index=False).nunique().query("run == 1")
    for _, (alg,) in non_repeated[keys].iterrows():
        for i in range(times):
            sliced_df = original_df.query("(algorithm == @alg)").copy()
            if i == sliced_df.run.unique().item():
                continue
            sliced_df.loc[:, ["run"]] = i
            sliced_df.loc[:, ["duplicated"]] = 1
            results_df = pd.concat([results_df, sliced_df], axis=0)
    return results_df.reset_index(drop=True)


def process_raw_fn(data):
    # https://arxiv.org/pdf/1703.05051.pdf
    # First the bandpass, them the exp stand

    data = bandpass_cnt(
        data, low_cut_hz=4.0, high_cut_hz=38.0, fs=128.0, filtfilt=False
    )
    data = exponential_standardize(data)

    return data


def raw_fn(raw):
    return apply_raw(process_raw_fn, raw.copy())


def save_artifact(artifact, name, save_folder, uid, ica_method, clf_method, n_run):
    save_folder = save_folder / f"{uid}"

    for subfolder in (ica_method, clf_method):
        if subfolder is not None:
            save_folder = save_folder / subfolder

    save_folder.mkdir(parents=True, exist_ok=True)

    artifact_path = save_folder / f"{name}_{n_run}.pkl"

    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)


device = "cpu"
n_channels = 22
f1 = 4
d = 2
f2 = d * f1

length = 256
load_kwargs = dict(tmin=0.5, tmax=2.6, reject=False, raw_fn=raw_fn)


def run(
    dataset,
    uid,
    save_folder,
    ica_methods=None,
    channels=None,
    n_runs=10,
    random_state=1,
):
    np.random.seed(random_state)
    random.seed(random_state)

    ica_methods = ica_methods or [None]

    (train_epochs_orig, train_labels), (
        test_epochs_orig,
        test_labels,
    ) = load_subject_epochs(dataset, uid, load_kwargs=load_kwargs)

    train_epochs_orig.load_data().resample(128)
    test_epochs_orig.load_data().resample(128)

    results_list = list()

    for ica_method in ica_methods:
        for n_run in range(n_runs):
            print(f"Run {n_run} of {n_runs} for {ica_method} on {uid}")
            run_seed = random_state + n_run

            seed_everything(run_seed)

            train_epochs = train_epochs_orig.copy()
            test_epochs = test_epochs_orig.copy()

            # train_test_split melts epochs, so we need to concatenate them again
            train_epochs_list, val_epochs_list = train_test_split(
                train_epochs, random_state=run_seed, stratify=train_epochs.events[:, 2]
            )
            train_epochs = concatenate_epochs(train_epochs_list)
            val_epochs = concatenate_epochs(val_epochs_list)

            n_channels = len(train_epochs.ch_names)
            # Every first run, run the ICA
            # If the ICA is non-deterministic, run it
            refit_ica = is_ica_stochastic(ica_method) or (n_run == 0)

            if refit_ica:
                x_train, y_train = train_epochs.copy(), train_epochs.events[:, 2]
                x_val, y_val = val_epochs.copy(), val_epochs.events[:, 2]
                x_test, y_test = test_epochs.copy(), test_epochs.events[:, 2]

                if ica_method in ("none", None):
                    x_train = x_train.get_data()
                    x_val = x_val.get_data()
                    x_test = x_test.get_data()

                elif "orica" in ica_method:
                    x_train = x_train.get_data()
                    x_val = x_val.get_data()
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
                            scaling=1,
                            save=False,
                        )
                        .reshape(n_channels, n_epochs, n_times)
                        .transpose(1, 0, 2)
                    )

                    #################################################################################
                    ICA.mode = "constant"
                    ICA.lm_0, ICA.lw_0 = 0.001, 0.001
                    #################################################################################

                    n_epochs, n_channels, n_times = x_val.shape
                    x = x_val.transpose(1, 0, 2).reshape(n_channels, -1)
                    x_val = (
                        ICA.transform(x, scaling=1, save=False, warm_start=True)
                        .reshape(n_channels, n_epochs, n_times)
                        .transpose(1, 0, 2)
                    )

                    n_epochs, n_channels, n_times = x_test.shape
                    x = x_test.transpose(1, 0, 2).reshape(n_channels, -1)
                    x_test = (
                        ICA.transform(x, scaling=1, save=False, warm_start=True)
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
                    x_val = ICA.transform(x_val)
                    x_test = ICA.transform(x_test)
                    save_artifact(ICA, "ica", save_folder, uid, ica_method, None, n_run)

            if isinstance(x_train, BaseEpochs):
                x_train, y_train = x_train.get_data(), x_train.events[:, 2]
                x_val, y_val = x_val.get_data(), x_val.events[:, 2]
                x_test, y_test = x_test.get_data(), x_test.events[:, 2]

            x_train = x_train[:, :, :length]
            x_val = x_val[:, :, :length]
            x_test = x_test[:, :, :length]

            (
                train_data,
                train_labels,
                val_data,
                val_labels,
                test_data,
                test_labels,
            ) = to_tensor(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                device=device,
            )

            train_dataloader = DataLoader(
                TensorDataset(train_data.float(), train_labels.long().flatten()),
                batch_size=32,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                TensorDataset(val_data.float(), val_labels.long().flatten()),
                batch_size=len(val_data),
                shuffle=False,
            )
            test_dataloader = DataLoader(
                TensorDataset(test_data.float(), test_labels.long().flatten()),
                batch_size=len(test_data),
                shuffle=False,
            )

            torch.autograd.set_detect_anomaly(True)

            model = EEGNet(n_channels, 4, length, f1=f1, d=d, f2=f2).to(device).float()
            check_val_every_n_epoch = 5
            patience = 100
            checkpoint_dirpath = (
                save_folder / "checkpoints" / f"{uid}_{ica_method}_{n_run}"
            )
            trainer = pl.Trainer(
                callbacks=[
                    ModelCheckpoint(
                        dirpath=checkpoint_dirpath,
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                    ),
                    EarlyStopping(
                        monitor="val_loss",
                        min_delta=1e-3,
                        patience=patience // check_val_every_n_epoch,
                        verbose=False,
                        mode="min",
                    ),
                ],
                deterministic=True,
                check_val_every_n_epoch=check_val_every_n_epoch,
                accelerator="cpu",
                logger=pl.loggers.CSVLogger(
                    save_folder / f"./logs_{ica_method}",
                    name=f"subject_{uid}",
                    version=f"trial_{n_run}",
                ),
                max_epochs=10000,
            )

            model.set_trainer(trainer)
            model.fit(train_dataloader, val_dataloader)

            result = model.trainer.test(model, test_dataloader, ckpt_path="best")[0]

            result.update(
                dict(
                    uid=uid,
                    run=n_run,
                    algorithm=ica_method,
                )
            )
            results_list.append(result)

            # Create a dataframe from results_list, a list of dicts
            pd.DataFrame.from_records(results_list).to_csv(
                save_folder / "results.csv", index=False
            )

    results_df = pd.DataFrame.from_records(results_list)
    return results_df


if __name__ == "__main__":
    bci_dataset_folderpath = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/")
    bci_test_dataset_folderpath = Path(
        "/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/"
    )
    dataset = BCI_IV_Comp_Dataset(
        bci_dataset_folderpath, test_folder=bci_test_dataset_folderpath
    )

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
    save_folder = Path("./results")
    for uid in dataset.list_uids():
        subject_results_dict = run(
            dataset,
            uid,
            save_folder,
            ica_methods=ica_methods,
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

        results_df.to_csv(save_folder / "results_final.csv", index=False)
