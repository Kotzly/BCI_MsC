import numpy as np
from ica_benchmark.dictionary import annotation_encode_dict


def get_annotations(gdf_obj):
    data = gdf_obj
    sr = data.info["sfreq"]
    n_samples = data._raw_extras[0]["n_records"]

    onsets = np.trunc(data.annotations.onset * sr).astype(np.uint32, casting="unsafe")
    durations = np.trunc(data.annotations.duration * sr).astype(np.uint32, casting="unsafe")
    
    desc = data.annotations.description.astype(np.uint32)
    labels_codes = np.vectorize(annotation_encode_dict.get)(desc)
    
    n_codes = len(annotation_encode_dict)
    labels = np.zeros((n_samples, n_codes))
    
    for code, onset, duration in zip(labels_codes, onsets, durations):
        labels[onset:onset+duration, code] = 1
    
    return labels