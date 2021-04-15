from mne.io import read_raw_gdf
from pathlib import Path
import numpy as np

PRELOAD = False

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

def load_subject_data(root, subject, mode=None):
    if mode is None:
        mode = "train"
    
    if mode == "train":
        filepath = root / f"{subject}T.gdf"
        gdf_data, labels, ch_names, info = load_gdf_file(filepath)
    elif mode == "test":
        filepath = root / f"{subject}E.gdf"
        gdf_data, labels, ch_names, info = load_gdf_file(filepath)
    elif mode == "both":
        filepath_t = root / f"{subject}T.gdf"
        filepath_e = root / f"{subject}E.gdf"
        gdf_data_t, labels_t, ch_names_t, info_t = load_gdf_file(filepath_t)
        gdf_data_e, labels_e, ch_names, info = load_gdf_file(filepath_e)
        
        
        assert np.all(ch_names_t == ch_names)
        assert np.all(info_t == info)
        
        gdf_data = gdf_data_t.copy()
        gdf_data._data = np.concatenate(
            [
                gdf_data_t._data,
                gdf_data_e._data,
            ],
            axis=1
        )
        
        labels = np.concatenate(
            [
                labels_t,
                labels_e
            ],
            axis=0
        )
    
    return gdf_data, labels, ch_names, info

def load_subjects_data(root, datasets=None, mode="train"):
    if datasets is None:
#         data_dict = {
#             "all": {
#                 filepath.name[:3]: None for filepath in root.glob("*T.gdf")
#             }
#         }
        data_dict = {
            filepath.name[:3]: {
                filepath.name[:3]: None
            } for filepath in root.glob("*T.gdf")
        }
    else:
        data_dict = {
            dataset: {
                subject_id: {} for subject_id in datasets[dataset]
            } for dataset in datasets
        }
    
    chs_ = None
    for dataset in data_dict:
        for subject_id in data_dict[dataset]:
            gdf, labels, chs, info = load_subject_data(root, subject_id, mode=mode)
            if chs_ is None:
                chs_ = chs
            else:
                assert chs_ == chs
            data_dict[dataset][subject_id] = {
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
        