from mne.io import read_raw_gdf
import numpy as np
import mne
from pathlib import Path
import numpy as np
import time

from ica_benchmark.scoring import mutual_information, coherence, correntropy, apply_pairwise, apply_pairwise_parallel, SCORING_FN_DICT
from ica_benchmark.processing.ica import get_ica_transformers
from ica_benchmark.io.load import join_gdfs_to_numpy, load_subjects_data

from sacred.observers import MongoObserver, FileStorageObserver
from sacred import Experiment

import json

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

def run_ica_experiment(_run, root, dataset_dict, n_components, ica_method, fn_name):

    dataset_data_dict = {}
    for fold in ("train", "validation", "test"):
        data = load_subjects_data(root, subjects=dataset_dict[fold], mode="both")
        dataset_dict[fold] = data

    root = Path(root)

    gdfs = [datasets["train"][subject]["gdf"] for subject in datasets["train"]]
    
    joined_dataset = join_gdfs_to_numpy(gdfs)

    joined_eeg, labels = join_gdfs_to_numpy(gdfs)
    ica_transform = get_ica_transformers()[ica_method]
    ica_transform.fit(joined_eeg)

    results = {}

    for dataset_name in datasets:
        dataset_fold = datasets[dataset_name]
        if not dataset_name in results:
            results[dataset_name] = {}
        for subject_id in dataset_fold:
            gdf_data = dataset_fold[subject_id]["gdf"]   
            data = gdf_data.get_data().T

            data_arr = ica_transform.transform(data)

            start = time.time()
            score = apply_pairwise_parallel(data_arr, SCORING_FN_DICT[fn_name])
            duration = time.time() - start
            
            results[dataset_name][subject_id] = {
                "score": score,
                "duration": duration
            }
    with open("results.json") as json_file:
        json.dump(results, json_file)

    _run.add_artifact("results.json", content_type="json")

@ex.automain
def main(
    _run,
    root,
    dataset_dict,
    n_components,
    ica_method,
    scoring_fn_name
):
    run_ica_experiment(_run, root, dataset_dict, n_components, ica_method, scoring_fn_name)