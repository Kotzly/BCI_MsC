from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch
from statistics import mode
import numpy as np

def with_default(value, default):
    return value if value is not None else default


class WindowTransformer():

    def __init__(
        self,
        feature_transform_fn,
        label_transform_fn=mode,
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


class IterDataset(IterableDataset):
    def __init__(self, x, y, transformer_instance):
        super(IterDataset).__init__()
        self.transformer_instance = transformer_instance
        self.start = 0
        assert len(x) == len(y), "Lengths must be equal"
        self.end = len(y)
        self.x, self.y = x, y
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start, iter_end = self.start, self.end
        else:  # in a worker process
            # split workload
            per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        
        for x, y in self.transformer_instance.transform(self.x, self.y, start=iter_start, end=iter_end):
            yield x, y


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
