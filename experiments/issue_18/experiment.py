from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from ica_benchmark.data.utils import apply_raw, to_tensor
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from ica_benchmark.models import EEGNet
from ica_benchmark.processing.filter import bandpass_cnt
from ica_benchmark.processing.standardization import exponential_standardize
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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


bci_dataset_folderpath = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/")
bci_test_dataset_folderpath = Path(
    "/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/"
)

dataset = BCI_IV_Comp_Dataset(
    bci_dataset_folderpath, test_folder=bci_test_dataset_folderpath
)
device = "cpu"
n_channels = 22
f1 = 4
d = 2
f2 = d * f1


load_kwargs = dict(tmin=0.5, tmax=2.6, reject=False, raw_fn=raw_fn)

results_list = list()

for uid in dataset.list_uids():
    train_epochs, train_labels_np = dataset.load_subject(
        uid, run=1, session=1, **load_kwargs
    )
    test_epochs, test_labels_np = dataset.load_subject(
        uid, run=1, session=2, **load_kwargs
    )

    train_data_np = train_epochs.load_data().resample(128).get_data()
    test_data_np = test_epochs.load_data().resample(128).get_data()

    length = 256

    for trial_number in range(10):
        seed_everything(trial_number)

        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data_np,
            train_labels_np,
            test_size=0.3,
            stratify=train_labels_np,
            random_state=trial_number,
        )
        test_data = test_data_np.copy()
        test_labels = test_labels_np.copy()

        train_data = train_data[:, :, :length]
        val_data = val_data[:, :, :length]
        test_data = test_data[:, :, :length]

        (
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        ) = to_tensor(
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
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
        trainer = pl.Trainer(
            callbacks=[
                ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
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
                "./logs", name=f"subject_{uid}", version=f"trial_{trial_number}"
            ),
            max_epochs=10000,
        )

        model.set_trainer(trainer)
        model.fit(train_dataloader, val_dataloader)

        result = model.trainer.test(model, test_dataloader, ckpt_path="best")[0]
        result.update(dict(uid=uid, run=trial_number))
        results_list.append(result)

        pd.DataFrame(results_list).to_csv("results.csv", index=False)
