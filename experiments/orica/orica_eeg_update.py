from ica_benchmark.processing.orica_code import ORICA, plot_various, match_corr_epochs, windowed_correlation
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    uid = "A07"
    filepath = Path(f"/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/{uid}T.gdf")
    BCI_IV_Comp_Dataset.EVENT_MAP_DICT = {
        "277": 0,
        "276": 1,
        "1072": 2
    }
    epochs = BCI_IV_Comp_Dataset.load_as_epochs(filepath, load_eog=True, tmin=4., tmax=60., reject=False).load_data()
    exclude_channels = ["EEG-Fz"]
    eeg_epochs = epochs.copy().pick("eeg")#.pick([ch for ch in epochs.ch_names if not ch in exclude_channels])
    # eeg_epochs = epochs.copy().pick("eeg").pick(['EEG-Fz', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Pz'])
    # eog_epochs = epochs.copy().pick("eog").pick(['EOG-left', 'EOG-central', 'EOG-right'])
    # eeg_epochs = epochs.copy().pick("eeg").pick(['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Pz'])
    # eog_epochs = epochs.copy().pick("eog").pick(['EOG-left'])
    eog_epochs = epochs.copy().pick("eog").pick(['EOG-central'])
    print(eog_epochs.ch_names)
    print(eeg_epochs.ch_names)
    # exit()
    n_channels = len(eeg_epochs.ch_names)
    print("NCHANNELS:", n_channels, eeg_epochs.ch_names)
    eog_data = eog_epochs.get_data()
    eeg_data = eeg_epochs.get_data()
    
    print("EEG:", eeg_data.shape)
    single_epoch_eeg = eeg_epochs.get_data().transpose(1, 0, 2).reshape(n_channels, -1)
    pd.DataFrame(single_epoch_eeg).to_csv(f"{uid}T.csv", header=False, index=False)
    # ica = ORICA(mode="constant", n_channels=n_channels, block_update=True, size_block=8, stride=8, lw_0=.0078, lm_0=0.0078)
    ica = ORICA(mode="decay", n_channels=n_channels, block_update=True, size_block=8, stride=8, lm_0=.995, lw_0=.995, gamma=0.6, n_sub=1)
    # ica = ORICA(mode="adaptative", n_channels=n_channels, block_update=True, size_block=8, stride=8, lm_0=.15, lw_0=.15, n_sub=3)
    # print("PEARSON", pearsonr(eog_data.transpose(1, 0, 2).reshape(2, -1)[0], eog_data.transpose(1, 0, 2).reshape(2, -1)[1]))
    
    # MATLAB
    # matlab_filtered = pd.read_csv("/home/paulo/Documents/GIT/orica/filtered.csv", index_col=None, header=None).to_numpy().T
    # assert matlab_filtered.shape == single_epoch_eeg.shape, (matlab_filtered.shape, single_epoch_eeg.shape)
    # n_epochs, _, n_times = eeg_data.shape
    # n_channels = n_channels
    # matlab_filtered = matlab_filtered.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    # eeg_sources_data = matlab_filtered
    ##################################################################################################
    # fica = FastICA(n_channels)
    # n_epochs, n_channels, n_times = eeg_data.shape
    # eeg_sources_data = fica.fit_transform(
    #     eeg_data.transpose(1, 0, 2).reshape(n_channels, -1).T
    # ).T.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    ##################################################################################################
    eeg_sources_data = ica.transform_epochs(eeg_epochs, scaling=1e6)
    ##################################################################################################
    # eeg_sources_data = eeg_data

    n_eeg_channels = eeg_sources_data.shape[1]
    n_eog_channels = eog_data.shape[1]

    assert len(eeg_sources_data) == len(eog_data)
    n_epochs = len(eeg_sources_data)

    ch, corr, _ = match_corr_epochs(eeg_data, eog_data)
    best_eog_idx = np.argmax(corr)
    best_eeg_idx = ch[best_eog_idx]

    print("[CORRELATIONS] No filtering:", corr)
    print("[CORRELATIONS] ICA on EEG:", match_corr_epochs(eeg_sources_data, eog_data)[1])
    print("[CORRELATIONS] ICA on EEG (last epoch):", match_corr_epochs(eeg_sources_data[[-1]], eog_data[[-1]])[1])

    fig, axes = plt.subplots(2, 1, figsize=(20, 4))
    axes[0].plot(
        windowed_correlation(
            eeg_sources_data[-1],
            eog_data[-1],
            window_size=250
        )
    )
    axes[0].set_ylim((0, 1.))
    axes[1].plot(eog_data[-1].T)
    for ax in axes:
        ax.grid()
    plt.show()

    if ica.w is not None:
        print("[CORRELATIONS] FINAL W:", match_corr_epochs((ica.w @ ica.m @ eeg_sources_data[-1])[np.newaxis, :, :], eog_data[[-1]])[1])

    if hasattr(ica, "sigmas"):
        plt.figure(figsize=(20, 4))
        plt.plot(np.log(ica.sigmas))
        plt.show()

    # plot_various(
    #     eeg_data[-1],
    #     n=n_channels,
    #     d=4e-5,
    #     title="EEG"
    # )
    # plot_various(
    #     eeg_sources_data[-1],
    #     n=n_channels,
    #     d=4,
    #     title="ICA EEG",
    #     figsize=(20, 11 * 1.5)
    # )
    # plot_various(
    #     eog_data[-1],
    #     n=3,
    #     d=1e-4,
    #     title="ICA EOG",
    #     figsize=(20, 3 * 1.5)
    # )

    print(match_corr_epochs(eeg_sources_data, eog_data)[1])
    for eeg_channel in range(n_eeg_channels):
        # fig, axes = plt.subplots(3, 1, figsize=(20, 3 * 4))
        for eog_channel in range(n_eog_channels):
            pcc_list = list()
            for epoch in range(n_epochs):
                pcc = pearsonr(
                    eeg_sources_data[epoch, eeg_channel, :],
                    eog_data[epoch, eog_channel, :],
                )[0]

                pcc_list.append(pcc)
                if np.abs(pcc) > 0.5:
                    plt.figure(figsize=(18, 4))
                    plt.plot(eeg_sources_data[epoch, eeg_channel, :] * np.sign(pcc), label="EEG", c="k")
                    plt.twinx().plot(eog_data[epoch, eog_channel, :], label="EOG", c="g")
                    plt.legend()
                    plt.title(f"EOG:{eog_channel} EEG:{eeg_channel} EPOCH:{epoch}")
                    plt.show()
            # ax = axes[eog_channel]
            # ax.axline((0, .5), (1, 0.5), c="r")
            # ax.axline((0, .0), (1, 0.), c="g")
            # ax.plot(pcc_list)
            # ax.scatter(range(len(pcc_list)), pcc_list)
            # ax.set_title("EEG {} - EOG {} | {:.3f}".format(eeg_channel, eog_channel, pcc))
            # ax.set_ylim(-1, 1)
            # ax.grid()
            ind = "\t\t<--------" if any(np.array(np.abs(pcc_list)) > .5) else ""
            print("EEG {} | EOG {}: {} {}".format(eeg_channel, eog_channel, pcc_list, ind))

