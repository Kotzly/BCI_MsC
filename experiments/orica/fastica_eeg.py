from ica_benchmark.processing.orica_code import ORICA
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA

if __name__ == "__main__":

    filepath = "/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/A01T.gdf"

    BCI_IV_Comp_Dataset.EVENT_MAP_DICT = {
        "276": 0,
        "277": 1,
        "1072": 2
    }
    epochs = BCI_IV_Comp_Dataset.load_as_epochs(filepath, load_eog=True, tmin=0., tmax=60., reject=False).load_data().filter(.05, 30.)
    eeg_epochs = epochs.copy().pick("eeg")
    eog_epochs = epochs.copy().pick("eog")
    eog_data = eog_epochs.get_data()
    eeg_data = eeg_epochs.get_data()
    print("EEG:", eeg_epochs.get_data().shape)

    ica = FastICA(22)
    n_epochs, n_channels, n_times = eeg_data.shape
    eeg_sources_data = ica.fit_transform(
        eeg_data.transpose(1, 0, 2).reshape(n_channels, -1).T
    ).T.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    
    ica_eog = FastICA(3)
    n_epochs, n_channels, n_times = eog_data.shape
    eog_data = ica_eog.fit_transform(
        eog_data.transpose(1, 0, 2).reshape(n_channels, -1).T
    ).T.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)

    n_eeg_channels = eeg_sources_data.shape[1]
    n_eog_channels = eog_data.shape[1]

    assert len(eeg_sources_data) == len(eog_data)
    n_epochs = len(eeg_sources_data)
    
    for eeg_channel in range(n_eeg_channels):
        fig, axes = plt.subplots(3, 1, figsize=(20, 3 * 4))
        for eog_channel in range(n_eog_channels):
            pcc_list = list()
            for epoch in range(n_epochs):
                pcc = pearsonr(
                    eeg_sources_data[epoch, eeg_channel, :],
                    eog_data[epoch, eog_channel, :],
                )[0]
                pcc_list.append(pcc)
                # print(
                #    "EEG: {} - EOG: {} -> {:.3f}".format(
                #        eeg_channel,
                #        eog_channel,
                #        pcc
                #    )
                #)
                if np.abs(pcc) > .8:
                    fig_, ax = plt.subplots(1, 1, figsize=(20, 4))
                    ax.plot(
                        eeg_sources_data[epoch, eeg_channel, :],
                    )
                    ax.plot(
                        eog_data[epoch, eog_channel, :],
                    )
                    fig_.show()
            ax = axes[eog_channel]
            ax.axline((0, .5), (1, 0.5), c="r")
            ax.axline((0, .0), (1, 0.), c="g")
            ax.plot(pcc_list)
            ax.set_title("EEG {} - EOG {} | {:.3f}".format(eeg_channel, eog_channel, pcc))
            ax.set_ylim(-1, 1)
            ax.grid()
        plt.show()
