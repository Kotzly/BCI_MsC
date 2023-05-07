from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from pathlib import Path
from ica_benchmark.models import EEGNet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from ica_benchmark.data.utils import apply_raw
from ica_benchmark.processing.filter import bandpass_cnt
from ica_benchmark.processing.standardization import exponential_standardize
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
from lightning.pytorch import seed_everything


def process_raw_fn(data):
    # https://arxiv.org/pdf/1703.05051.pdf
    # First the bandpass, them the exp stand
    data = bandpass_cnt(data, low_cut_hz=4., high_cut_hz=38., fs=128., filtfilt=False)
    data = exponential_standardize(data)

    return data


def raw_fn(raw):
    return apply_raw(
        process_raw_fn,
        raw.copy().resample(128)
    )


def to_tensor(*args, device="cpu"):
    return (
        torch.unsqueeze(
            torch.from_numpy(arg).to(device),
            1
        )
        for arg
        in args
    )


bci_dataset_folderpath = Path('/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/')
bci_test_dataset_folderpath = Path('/home/paulo/Documents/datasets/BCI_Comp_IV_2a/true_labels/')

dataset = BCI_IV_Comp_Dataset(bci_dataset_folderpath, test_folder=bci_test_dataset_folderpath)
device = "cpu"
n_channels = 22
f1 = 4
d = 2
f2 = d * f1


load_kwargs = dict(
    tmin=.5, tmax=2.6, reject=False, raw_fn=raw_fn
)

results_list = list()

for uid_number in range(1, 10):
    for trial_number in range(10):
        seed_everything(trial_number)

        uid = str(uid_number)

        train_epochs, train_labels = dataset.load_subject(uid, run=1, session=1, **load_kwargs)
        test_epochs, test_labels = dataset.load_subject(uid, run=1, session=2, **load_kwargs)

        train_epochs.load_data()
        test_epochs.load_data()

        train_data = train_epochs.get_data()
        test_data = test_epochs.get_data()

        N = len(train_data)
        n_train = (3 * N) // 5
        print(n_train)

        val_data = train_data[-n_train:]
        val_labels = train_labels[-n_train:]
        train_data = train_data[:n_train]
        train_labels = train_labels[:n_train]

        length = 256
        train_data = train_data[:, :, :length]
        val_data = val_data[:, :, :length]
        test_data = test_data[:, :, :length]

        train_data, train_labels, val_data, val_labels, test_data, test_labels = to_tensor(
            train_data, train_labels,
            val_data, val_labels,
            test_data, test_labels,
            device=device
        )

        train_dataloader = DataLoader(
            TensorDataset(
                train_data.float(),
                train_labels.long().flatten()
            ),
            batch_size=32,
            shuffle=True
        )
        val_dataloader = DataLoader(
            TensorDataset(
                val_data.float(),
                val_labels.long().flatten()
            ),
            batch_size=len(val_data),
            shuffle=False
        )
        test_dataloader = DataLoader(
            TensorDataset(
                test_data.float(),
                test_labels.long().flatten()
            ),
            batch_size=len(test_data),
            shuffle=False
        )

        torch.autograd.set_detect_anomaly(True)

        model = EEGNet(n_channels, 4, length, f1=f1, d=d, f2=f2).to(device).float()
        trainer = pl.Trainer(
            default_root_dir="./training",
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    every_n_epochs=1,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-3,
                    patience=100,
                    verbose=False,
                    mode="min"
                )
            ],
            deterministic=True,
            check_val_every_n_epoch=5,
            accelerator="cpu",
            logger=pl.loggers.CSVLogger(
                "./logs",
                name=f"subject_{uid}",
                version=f"trial_{trial_number}"
            ),
            max_epochs=10000
        )

        model.set_trainer(trainer)
        model.fit(train_dataloader, val_dataloader)

        result = model.trainer.test(model, test_dataloader, ckpt_path="best")
        result[0].update(
            dict(
                uid=uid,
                trial=trial_number
            )
        )
        results_list.append(result)

pd.DataFrame.from_record(results_list).to_csv("result.csv")
