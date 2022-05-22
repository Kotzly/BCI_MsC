from hashlib import algorithms_available
from ica_benchmark.processing.orica_code import ORICA, plot_various, match_corr_epochs, windowed_correlation
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
from pathlib import Path
import json
from mne import concatenate_epochs

if __name__ == "__main__":

    uids = [f"A0{i}" for i in range(1, 9 + 1)]
    results = dict()
    for uid in uids:
        results[uid] = list()
        filepaths = [
            Path(f"/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/{uid}E.gdf"),
            # Path(f"/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/{uid}E.gdf"),
        ]
        # BCI_IV_Comp_Dataset.EVENT_MAP_DICT = {
        #     "276": 0,
        #     "277": 1,
        #     "1072": 2
        # }
        BCI_IV_Comp_Dataset.EVENT_MAP_DICT = {
            "277": 0,
            "276": 1,
            "1072": 2
        }
        epochs_list = [
            BCI_IV_Comp_Dataset.load_as_epochs(filepaths[0], load_eog=True, tmin=4., tmax=60., reject=False).load_data(),
            # BCI_IV_Comp_Dataset.load_as_epochs(filepaths[1], load_eog=True, tmin=4., tmax=60., reject=False, has_labels=False).load_data(),
        ]
        epochs = concatenate_epochs(epochs_list)

        # eeg_epochs = epochs.copy().pick("eeg").pick(['EEG-Fz', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Pz'])
        # eog_epochs = epochs.copy().pick("eog").pick(['EOG-left', 'EOG-right'])
        eeg_epochs = epochs.copy().pick("eeg")
        eog_epochs = epochs.copy().pick("eog").pick(['EOG-right'])

        eog_data = eog_epochs.get_data()
        eeg_data = eeg_epochs.get_data()
        
        n_channels = eeg_data.shape[1]
        
        single_epoch_eeg = eeg_epochs.get_data().transpose(1, 0, 2).reshape(n_channels, -1)
        pd.DataFrame(single_epoch_eeg).to_csv(f"{uid}T.csv", header=False, index=False)
        
        ch, corr, _ = match_corr_epochs(eeg_data, eog_data)
        best_eog_idx = np.argmax(corr)
        best_eeg_idx = ch[best_eog_idx]
        print("UID:", uid)
        print("\tNo filtering:", corr)

        correlation_array_dict = dict()
        algorithms = ["FastICA", "ORICA"]
        for algorithm in algorithms:
            if algorithm == "FastICA":
            ##################################################################################################
                fica = FastICA(n_channels, max_iter=100)
                n_epochs, n_channels, n_times = eeg_data.shape
                eeg_sources_data = fica.fit_transform(
                    eeg_data.transpose(1, 0, 2).reshape(n_channels, -1).T
                ).T.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
            ##################################################################################################
            elif algorithm == "ORICA":
                ica = ORICA(mode="decay", n_channels=n_channels, block_update=True, size_block=8, stride=8, lm_0=.995, lw_0=.995, gamma=.6, n_sub=1)
                # ica = ORICA(mode="constant", n_channels=n_channels, block_update=True, size_block=8, stride=8, lm_0=.005, lw_0=.005, n_sub=2)
                eeg_sources_data = ica.transform_epochs(eeg_epochs, scaling=1e6, verbose=False)
            else:
                raise Exception()
            ##################################################################################################
            # eeg_sources_data = eeg_data


            n_eeg_channels = eeg_sources_data.shape[1]
            n_eog_channels = eog_data.shape[1]

            assert len(eeg_sources_data) == len(eog_data)
            n_epochs = len(eeg_sources_data)

            correlation_array_dict[algorithm] = windowed_correlation(
                eeg_sources_data[-1],
                eog_data[-1],
                window_size=250
            )
            full_corr = match_corr_epochs(eeg_sources_data, eog_data)[1]
            le_corr = match_corr_epochs(eeg_sources_data[[-1]], eog_data[[-1]])[1]

            print(f"\t[{algorithm}] ICA on EEG:", full_corr)
            print(f"\t[{algorithm}] ICA on EEG (last epoch):", le_corr)

            if algorithm == "ORICA":
                print("\t[CORRELATIONS] FINAL W:", match_corr_epochs((ica.w @ ica.m @ eeg_sources_data[-1])[np.newaxis, :, :], eog_data[[-1]])[1])

            r = dict(
                algorithm=algorithm,
                full_correlation=list(full_corr),
                last_epoch_correlation=list(le_corr),
                # correlation_array=np.round(correlation_array_dict[algorithm], decimals=4).T.tolist()
            )
            results[uid].append(r)
        n_algorithms = len(algorithms)
        # fig, axes = plt.subplots(2, 1, figsize=(20, 1 + 3 * n_algorithms))
        # for ax, (algorithm, c_arr) in zip(axes, correlation_array_dict.items()):
        #     ax.plot(c_arr)
        #     ax.set_title(algorithm)
        #     ax.set_ylim((0, 1))
        #     ax.grid()
        # plt.show()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    for algorithm in algorithms:
        avg_corr = np.mean(
            [
                results[uid][algorithms.index(algorithm)]["last_epoch_correlation"]
                for uid in uids
            ]
        )
        print(
            f"[{algorithm}] Mean last epoch Corr.:",
            avg_corr
        )