from mne.io import read_raw_gdf
from pathlib import Path
import numpy as np
from ica_benchmark.processing.label import get_annotations

PRELOAD = False

def join_gdfs_to_numpy(gdfs):
    
    all_labels = []
    for gdf in gdfs:
        all_labels.append(
            get_annotations(gdf)
        )

    labels = np.concatenate(all_labels, axis=0)
    gdf_base = gdfs[0].copy()
    for gdf in gdfs[1:]:
        gdf_base._data = np.concatenate(
            [
                gdf_base._data,
                gdf._data
            ],
            axis=1
        )
    
    data = gdf_base._data.T
    
    return data, labels

def load_gdf_file(filepath):
    gdf_data = read_raw_gdf(filepath, preload=PRELOAD)

    chs = gdf_data.ch_names

    gdf_data = read_raw_gdf(
        filepath,
        preload=True,
        eog=["EOG-left", "EOG-central", "EOG-right"],
        exclude=[x for x in chs if "EOG" in x]
    )
    ch_names = gdf_data.ch_names
    info = parse_info(
        gdf_data._raw_extras[0]["subject_info"]
    )
    
    labels = get_annotations(gdf_data)
    
    return gdf_data, labels, ch_names, info

def parse_info(info_dict):
    cols = ['id', 'smoking', 'alcohol_abuse', 'drug_abuse', 'medication', 'weight', 'height', 'sex', 'handedness', 'age']
    parsed_info = {k:v for k, v in info_dict.items() if k in cols}
    return parsed_info

def load_subject_data(root, subject, filepaths=None, return_as_gdf=True):
    
    if filepaths is None:
        filepaths = root.glob("{subject}..gdf")

    gdf_all_data =[]
    labels_all_data = []

    ch_names_t, info_t = None, None

    for filepath in filepaths:
        gdf_data, labels, ch_names, info = load_gdf_file(filepath)
        gdf_all_data.append(gdf_data.get_data())
        labels_all_data.append(labels)

        assert np.all(ch_names_t == ch_names) or ch_names_t is None
        assert np.all(info_t == info) or info_t is None
        ch_names_t = ch_names
        info_t = info

    gdf_data._data = np.concatenate(
        gdf_all_data,
        axis=1
    )
    
    labels = np.concatenate(
        labels_all_data,
        axis=0
    )
    
    if not return_as_gdf:
        gdf_data = gdf_data._data.T
        assert len(gdf_data) == len(labels)
    else:
        assert len(gdf_data._data.T) == len(labels)
    
    return gdf_data, labels, ch_names, info

def id_from_filepath(filepath):
    return filepath.name[:3]

def load_subjects_data(root, subjects=None, mode=None, return_as_gdf=True):
    data_dict = {}
    if subjects is None:
        subjects = dict()
        if mode is None:
            filepaths = root.glob("*.gdf")
        elif mode.lower() in ("t", "train"):
            filepaths = root.glob("*T.gdf")
        elif mode.lower() in ("v", "test", "validation", "val", "e", "evaluation"):
            filepaths = root.glob("*E.gdf")
            
        for filepath in filepaths:
            subject_id = id_from_filepath(filepath)
            if subject_id in subjects:
                subjects[subject_id].append(filepath)
            else:
                subjects[subject_id] = [filepath]

        subjects[subject_id] = sorted(subjects[subject_id])

    chs_ = None
    for subject_id, filepaths in subjects.items():
    
#        data_dict[subject_id] =  load_subject_data(root, subject_id, filepaths)
        gdf, labels, chs, info =  load_subject_data(root, subject_id, filepaths, return_as_gdf=return_as_gdf)
        if chs_ is None:
            chs_ = chs
        else:
            assert chs_ == chs
        data_dict[subject_id] = {
            "gdf": gdf,
            "chs": chs,
            "info": info,
            "labels": labels
        }    
    
    return data_dict

def join_gdfs_to_numpy(gdfs):
    
    all_labels = []
    for gdf in gdfs:
        all_labels.append(
            get_annotations(gdf)
        )

    labels = np.concatenate(all_labels, axis=0)
    gdf_base = gdfs[0].copy()
    for gdf in gdfs[1:]:
        gdf_base._data = np.concatenate(
            [
                gdf_base._data,
                gdf._data
            ],
            axis=1
        )
    
    data = gdf_base._data.T
    
    return data, labels

def load_dataset(dataset, mode=None):
    dataset_dict = dict()
    for dataset_name in dataset:
        dataset_dict[dataset_name] = load_subjects_data(root, dataset[dataset_name], mode=None)
    return dataset_dict