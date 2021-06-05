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

def dataset_to_np(data):
    arr = np.concatenate(
        [data[sub]["gdf"]._data.T for sub in data],
        axis=0
    )
    labels = np.concatenate(
        [data[sub]["labels"] for sub in data],
        axis=0
    )
    return arr, labels

def eval_dataset(m, loader, run=None, epoch=None, name="D"):
    y_t = []
    y_p = []
    for x, y in loader:
        y_p_value = m(x.float().cuda()).cpu().detach().numpy()
        y_p.append(y_p_value)
        y_t.append(y.cpu().detach().numpy())

    y_p_multi = np.concatenate(y_p, axis=0)
    y_t_multi = np.concatenate(y_t, axis=0)
    y_p = y_p_multi.argmax(axis=1)
    y_t = y_t_multi.argmax(axis=1)

    
    b_acc = balanced_accuracy_score(y_t, y_p)
    kappa = cohen_kappa_score(y_t, y_p)
    precision = precision_score(y_t, y_p, average="macro")
    recall = recall_score(y_t, y_p, average="macro")
    auc = roc_auc_score(y_t_multi, y_p_multi, multi_class="ovr")
    
    run.log_scalar(f"{name}_balanced_accuracy", b_acc, epoch)
    run.log_scalar(f"{name}_precision", precision, epoch)
    run.log_scalar(f"{name}_kappa", kappa, epoch)
    run.log_scalar(f"{name}_recall", recall, epoch)
    run.log_scalar(f"{name}_auc", auc, epoch)

    return kappa

def run_experiment(_run, root, dataset_dict, ica_method, n_components):

    dataset_data_dict = {}
    root = Path(root)

    for fold in ("train", "validation", "test"):
        data = load_subjects_data(root, subjects=dataset_dict[fold], mode="both")
        dataset_data_dict[fold] = dataset_to_np(data)
    
    ######## RIGHT
    x_train, y_train = dataset_data_dict["train"]
    x_val, y_val = dataset_data_dict["validation"]
    x_test, y_test = dataset_data_dict["test"]
    y_train, y_val, y_test = y_train[:, 3:7], y_val[:, 3:7], y_test[:, 3:7], 
    ######## Testing
    #from sklearn.model_selection import train_test_split
    #X = np.concatenate([x_train, x_val, x_test], axis=0)
    #Y = np.concatenate([y_train, y_val, y_test], axis=0)
    #x_train, X, y_train, Y = train_test_split(X, Y, test_size=0.70, random_state=42)
    #x_val, x_test, y_val, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    ########
    print(x_train.shape, y_train.shape)

    ica_transform = get_ica_transformers(method=ica_method, n_components=n_components)
    ica_transform.fit(x_train)

    x_train = ica_transform.transform(x_train)
    x_val = ica_transform.transform(x_val)
    x_test = ica_transform.transform(x_test)

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    #############################################################################

    freqs = np.array([4, 7, 10, 13, 16, 19, 21])
    n_features = len(freqs) * 22

    feature_transform = partial(psd_feature_transform, freqs=freqs, bandwidth=2)

    dataset_kwargs = dict(
        feature_transform_fn=feature_transform,
        label_transform_fn=softmax_label_transform,
        window_size=250,
        stride=125
    )

    dataloader_kwargs = dict(
        batch_size=32,
        shuffle=True,
        num_workers=4, 
        drop_last=True,
        prefetch_factor=2,
    )

    dataset = WindowTransformerDataset(x_train, y_train, **dataset_kwargs)
    train_dataloader = DataLoader(dataset, **dataloader_kwargs)

    dataloader_kwargs.update(dict(batch_size=128))
    dataset = WindowTransformerDataset(x_val, y_val, **dataset_kwargs)
    val_dataloader = DataLoader(dataset, **dataloader_kwargs)

    dataset = WindowTransformerDataset(x_test, y_test, **dataset_kwargs)
    test_dataloader = DataLoader(dataset, **dataloader_kwargs)


    model = SimpleModel(n_features, 5).cuda()
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    loss_fn = nn.BCELoss()        
    
    best_metric = 0
    patience = 0
    for epoch in range(30):
        epoch_losses = []
        for x, y in train_dataloader:
            x = x.float().cuda()
            y = y.float().cuda()
            y_p = model(x)
            loss = loss_fn(y_p, y)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        val_metric = eval_dataset(model, val_dataloader, run=_run, epoch=epoch, name="val")
        _run.log_scalar("train_loss", epoch_loss, epoch)

        print(f"{epoch}: {epoch_loss:.4f} / {val_metric:.4f}")

        if val_metric >= (best_metric + 0.001):
            best_metric = val_metric
            patience = 0
        elif epoch >= 5:
            patience += 1
        
        if patience == 3:
            break

    eval_dataset(model, test_dataloader, run=_run, epoch=epoch, name="test")

@ex.automain
def main(
    _run,
    root,
    dataset_dict,
    ica_method,
    n_components,
):
    run_experiment(_run, root, dataset_dict, ica_method, n_components)