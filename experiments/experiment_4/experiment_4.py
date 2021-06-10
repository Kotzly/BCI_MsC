from mne.io import read_raw_gdf
import numpy as np
import mne
from pathlib import Path
import numpy as np
import time

from ica_benchmark.scoring import mutual_information, coherence, correntropy, apply_pairwise, apply_pairwise_parallel, SCORING_FN_DICT
from ica_benchmark.processing.ica import get_ica_transformers
from ica_benchmark.io.load import join_gdfs_to_numpy, load_subjects_data
from ica_benchmark.processing.dataset import WindowTransformerDataset
from ica_benchmark.processing.feature import psd_feature_transform
from ica_benchmark.processing.label import softmax_label_transform, sigmoid_label_transform
from ica_benchmark.io.load import load_subjects_data

import copy

from mne.io import read_raw_gdf
from mne import Epochs, read_events, find_events, events_from_annotations
from mne.time_frequency import psd_array_multitaper

from torch_optimizer.radam import RAdam

from sklearn.preprocessing import RobustScaler

from sacred.observers import MongoObserver, FileStorageObserver
from sacred import Experiment

from torch.optim import Adam
from torch.nn import Module, Linear, Softmax
from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch.functional as F
from torch import nn
import torch
import json

from functools import partial
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score


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

ex.add_config("dataset_dict.json")
ex.add_config("config.json")
ex.add_config("tags.json")


class SimpleModel(Module):
    def __init__(self, feature_size, n_classes):
        super().__init__()
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.build()
        
    def build(self):
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(self.feature_size),
            nn.Linear(
                self.feature_size,
                self.n_classes
            ),
            nn.Softmax(dim=1)
        ) 
    
    def forward(self, x):
        return self.model(x)


def extract(filepaths, ica=None, fit_ica=True):
    if ica is None:
        ica = get_ica_transformers("infomax")
    
    epochs, labels = list(), list()
    for filepath in filepaths:
        print("loading", filepath)
        try:
            gdf = read_raw_gdf(
                filepath,
                eog=["EOG-left", "EOG-central", "EOG-right"],
                exclude=["EOG-left", "EOG-central", "EOG-right"]
            )
            events = events_from_annotations(gdf, event_id={"769": 0, "770": 1, "771": 2, "772": 3})
            epoch = Epochs(gdf, events[0], event_repeated="drop", reject_by_annotation=True, tmin=-.3, tmax=.7, reject=dict(eeg=1e-4))
            epoch.drop_bad()
        except ValueError:
            print("Error in", filepath)
            continue
        epochs.append(epoch)
        labels.append(epoch.events[:, 2])
    
    labels = np.concatenate(labels)
    n_epochs, n_channels, n_times = epochs[0].get_data().shape
    ica_vec = [epoch.get_data().transpose(1, 0, 2).reshape(n_channels, -1).T for epoch in epochs]
    ica_vec = np.concatenate(ica_vec, axis=0)
    
    if fit_ica:
        ica.fit(ica_vec)

    transformed = ica.transform(ica_vec)
    transformed = ica_vec

    transformed = transformed.T.reshape(n_channels, -1, n_times).transpose(1, 0, 2)
    
    features, freqs = psd_array_multitaper(transformed, 250., fmin=0, fmax=20, bandwidth=2)

    n_epochs, _, _ = features.shape
#    features = features.reshape(n_epochs, -1)
    features = features.mean(axis=2)
    labels_placeholder = np.zeros((len(labels), 4))

    for i, l in enumerate(labels):
        labels_placeholder[i, l] = 1

    labels = labels_placeholder
    return features, labels, ica

def run_experiment(_run):
    root = Path("/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/")
    train_files = [
        "A01T.gdf",
        "A02T.gdf",
        "A03T.gdf",
        "A04T.gdf",
        "A05T.gdf",
    ]
    train_filepaths = [root / filename for filename in train_files]

    val_files = [
        "A06T.gdf",
        "A07T.gdf",
    ]
    val_filepaths = [root / filename for filename in val_files]

    test_files = [
        "A08T.gdf",
        "A09T.gdf",
    ]
    test_filepaths = [root / filename for filename in test_files]

    x_train, y_train, ica = extract(train_filepaths, fit_ica=True)
    x_val, y_val, _ = extract(val_filepaths, ica=ica, fit_ica=False)
    x_test, y_test, _ = extract(test_filepaths, ica=ica, fit_ica=False)

    scaler = RobustScaler()

    x_train, x_val, x_test = scaler.fit_transform(x_train), scaler.transform(x_val), scaler.transform(x_test)
    train_dataloader = list(zip(x_train, y_train))
    val_dataloader = list(zip(x_val, y_val))
    test_dataloader = list(zip(x_test, y_test))

    datasets_sizes = {
        "train": len(train_dataloader),
        "val": len(val_dataloader),
        "test": len(test_dataloader),
    }

    print(datasets_sizes)
    
    model = SimpleModel(x_train.shape[1], 4).cuda()
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
#    optimizer = RAdam(model.parameters(), lr=0.0001, weight_decay=1e-6)

    dataloader_kwargs = dict(
        batch_size=16,
        shuffle=True,
        num_workers=4, 
        drop_last=True,
        prefetch_factor=2,
    )

    dataloaders = {
        "train": DataLoader(train_dataloader, **dataloader_kwargs),
        "val": DataLoader(val_dataloader, **dataloader_kwargs),
        "test": DataLoader(test_dataloader, **dataloader_kwargs),
    }
    losses = dict(train=list(), val=list())
    accuracies = dict(train=list(), val=list())
    
    best_model_sd = model.state_dict()
    best_metric = 10
    patience = 0
    max_patience = 50
    for epoch in range(1000):
        for phase in "train", "val":
            epoch_losses = []
            corrects = 0
            for x, y in dataloaders[phase]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                x = x.float().cuda()
                y = y.float().cuda()
                y_p = model(x)
                loss = loss_fn(y_p, y)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                
                corrects += torch.sum(y_p.argmax(axis=1) == y.argmax(axis=1)).item()

                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
            
            epoch_loss = np.mean(epoch_losses)
            losses[phase].append(epoch_loss)
            accuracies[phase].append(corrects / datasets_sizes[phase])
            _run.log_scalar(f"{phase}_loss", epoch_loss, epoch)
            _run.log_scalar(f"{phase}_acc", corrects / datasets_sizes[phase], epoch)

        
        print("{}: {:.4f} / {:.4f}".format(epoch, losses["train"][-1], losses["val"][-1]))

        if epoch > 1:
            if losses["val"][-1] < best_metric:
                best_metric = losses["val"][-1]
                best_model_sd = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            
        if patience >= max_patience:
            break

    print("Loading model with loss", best_metric)
    model.load_state_dict(best_model_sd)

    acc = 0
    for x, y in dataloaders["test"]:
        pred = model(x.float().cuda()).argmax(axis=1)
        y_true = y.argmax(axis=1).float().cuda()
        acc += torch.sum(pred == y_true)
    acc = acc.cpu().numpy() / len(test_dataloader)
    print("Test acc:", acc)

@ex.automain
def main(_run):
    run_experiment(_run)
