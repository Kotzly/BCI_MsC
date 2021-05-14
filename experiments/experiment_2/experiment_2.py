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
from sklearn.metrics import balanced_accuracy_score


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
            nn.Linear(
                self.feature_size,
                100
            ),
            nn.ReLU(),
            nn.Linear(
                100,
                50
            ),
            nn.ReLU(),
            nn.Linear(
                50,
                self.n_classes
            ),
            #nn.Softmax(dim=1)
            
            # Using sigmoid because there can be a label with 0,0,0,0
            nn.Sigmoid(),
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

    ica_transform = get_ica_transformers(n_components=n_components)[ica_method]
    ica_transform.fit(x_train)

    x_train = ica_transform.transform(x_train)
    x_val = ica_transform.transform(x_val)
    x_test = ica_transform.transform(x_test)

    #############################################################################

    freqs = np.array([4, 7, 10, 13, 16, 19, 21])
    n_features = len(freqs) * 22

    feature_transform = partial(psd_feature_transform, freqs=freqs, bandwidth=2)

    dataset_kwargs = dict(
        feature_transform_fn=feature_transform,
#        label_transform_fn=softmax_label_transform,
        label_transform_fn=sigmoid_label_transform,
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


    model = SimpleModel(n_features, 4).cuda()
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    loss_fn = nn.BCELoss()


    def eval_dataset(m, loader):
        acc_arr = []
        for x, y in loader:
            y_p = m(x.float().cuda()).cpu().detach().numpy()
            acc = balanced_accuracy_score(y.argmax(axis=1), y_p.argmax(axis=1))
            acc_arr.append(acc)
        return np.mean(acc_arr)
    
    batch_i = 0
    best_acc = 0
    patience = 0
    for epoch in range(30):
        epoch_losses = []
        for x, y in train_dataloader:
            batch_i += 1
            x = x.float().cuda()
            y = y.float().cuda()
            y_p = model(x)
            loss = loss_fn(y_p, y)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        val_acc = eval_dataset(model, val_dataloader)
        _run.log_scalar("val_acc", val_acc, epoch)
        _run.log_scalar("train_loss", epoch_loss, epoch)

        print(f"{epoch}: {epoch_loss:.4f} / {val_acc:.4f}")

        if val_acc >= (best_acc + 0.001):
            best_acc = val_acc
            patience = 0
        elif epoch >= 5:
            patience += 1
        
        if patience == 3:
            break
    
@ex.automain
def main(
    _run,
    root,
    dataset_dict,
    ica_method,
    n_components,
):
    run_experiment(_run, root, dataset_dict, ica_method, n_components)