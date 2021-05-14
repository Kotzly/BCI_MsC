import numpy as np
from ica_benchmark.dictionary import annotation_encode_dict
from statistics import mode

def get_annotations_from_gdf(gdf_obj):
    data = gdf_obj
    sr = data.info["sfreq"]
    n_samples = data._raw_extras[0]["nsamples"]

    onsets = np.trunc(data.annotations.onset * sr).astype(np.uint32, casting="unsafe")
    durations = np.trunc(data.annotations.duration * sr).astype(np.uint32, casting="unsafe")
    
    desc = data.annotations.description.astype(np.uint32)
    labels_codes = np.vectorize(annotation_encode_dict.get)(desc)
    
    n_codes = len(annotation_encode_dict)
    labels = np.zeros((n_samples, n_codes))
    
    for code, onset, duration in zip(labels_codes, onsets, durations):
        labels[onset:onset+duration, code] = 1
    
    return labels

def create_null_class(x):
    x = np.concatenate(
        [np.zeros((x.shape[0], 1)), x.copy()],
        axis=1
    )
    x[:, 0] = np.bitwise_not(
        np.any(x[:, 1:], axis=1)
    ).astype(np.uint32)
    return x

def sigmoid_label_transform(x):
    l = create_null_class(x)
    label = mode(l.argmax(axis=1))
    sparse_label = np.zeros(x.shape[1])
    if label > 0:
        sparse_label[label-1] = 1
    return sparse_label

def softmax_label_transform(x):
    l = create_null_class(x)
    label = mode(l.argmax(axis=1))
    sparse_label = np.zeros(l.shape[1])
    sparse_label[label] = 1
    return sparse_label
