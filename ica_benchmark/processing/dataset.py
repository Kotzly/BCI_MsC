from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch
import numpy as np
from ica_benchmark.processing.feature import psd_feature_transform, tfr_feature_transform
from ica_benchmark.processing.label import label_transform

def with_default(value, default):
    return value if value is not None else default


class WindowTransformer():

    def __init__(
        self,
        feature_transform_fn=psd_feature_transform,
        label_transform_fn=label_transform,
        window_size=250,
        stride=125,
        iterator_mode=False,
        ):

        self.feature_transform_fn = feature_transform_fn
        self.label_transform_fn = label_transform_fn
        self.window_size = window_size
        self.stride = stride
        self.iterator_mode = iterator_mode

    def transform(self, x, y=None, start=None, end=None):
        
        size = len(x)
        start, end = with_default(start, 0), with_default(end, size)

        if y is not None:
            assert len(x) == len(y), "X and Y must have same sizes"

        if self.iterator_mode:
            return self._transform_iter(x, y=y, start=start, end=end)
        else:
            return self._transform_list(x, y=y, start=start, end=end)

    def _transform_list(self, x, y=None, start=None, end=None):
        
        with_y = y is not None

        output_x, output_y = list(), list()

        for step in range(start, end, self.stride):

            if step + self.window_size > end:
                break

            item_x = self.feature_transform_fn(x[step : step + self.window_size])
            if with_y:
                item_y = self.label_transform_fn(y[step : step + self.window_size])
            else:
                item_y = None

            return_items = item_x if with_y else (item_x, item_y)

            output_x.append(item_x)
            output_y.append(item_y)
        
        output_x = np.concatenate(output_x, axis=0)

        if with_y:
            output_y = np.array(output_y)

        return (output_x, output_y) if with_y else output_x

    def _transform_iter(self, x, y=None, start=None, end=None):
        
        with_y = y is not None
        
        if y is not None:
            assert len(x) == len(y), "X and Y must have same sizes"

        output_x, output_y = list(), list()

        for step in range(start, end, self.stride):

            if step + self.window_size > end:
                break

            item_x = self.feature_transform_fn(x[step : step + self.window_size])
            if with_y:
                item_y = self.label_transform_fn(y[step : step + self.window_size])
            else:
                item_y = None

            return_items = (item_x, item_y) if with_y else item_x

            yield return_items


class WindowTransformerDataset(Dataset):

    def __init__(
        self,
        X,
        Y,
        feature_transform_fn=psd_feature_transform,
        label_transform_fn=label_transform,
        window_size=500,
        stride=250,
        start=None,
        end=None
        ):
        super(WindowTransformerDataset).__init__()

        self.feature_transform_fn = feature_transform_fn
        self.label_transform_fn = label_transform_fn
        self.window_size = window_size
        self.stride = stride
        
        assert len(X) == len(Y), "X and Y must have same sizes."
        assert len(X) >= window_size, "Window size must be smaller than the array size."
        self.X, self.Y = X, Y
        
        self.start = with_default(start, 0)
        self.end = with_default(end, len(X))
        
    def __len__(self):
        return (len(self.X) - self.window_size) // self.stride - 2
    
    def __getitem__(self, i):
        
        idx = i * self.stride

        x = self.feature_transform_fn(self.X[idx : idx + self.window_size])
        y = self.label_transform_fn(self.Y[idx : idx + self.window_size])

        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
