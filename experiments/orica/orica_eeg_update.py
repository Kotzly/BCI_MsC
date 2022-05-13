from ica_benchmark.processing.orica_code import ORICA, plot_various
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
    epochs = BCI_IV_Comp_Dataset.load_as_epochs(filepath, load_eog=True, tmin=4., tmax=60., reject=False).load_data().filter(0.05, 30.)
    eeg_epochs = epochs.copy().pick("eeg")
    eog_epochs = epochs.copy().pick("eog")
    eog_data = eog_epochs.get_data()
    print("EEG:", eeg_epochs.get_data().shape)
    #ica = ORICA(mode="adaptative", block_update=True, size_block=8, stride=8, lm_0=.1, lw_0=.1)
    #ica = ORICA(n_channels=22, mode="decay", block_update=False, size_block=32, stride=4, gamma=0.6, lm_0=.995, lw_0=.995)
    #ica = ORICA(mode="decay", n_channels=22, block_update=True, size_block=4, stride=4, gamma=.8, lm_0=.95, lw_0=.95)
    #ica = ORICA(mode="constant", n_channels=22, block_update=True, size_block=16, stride=16, lw_0=.01, lm_0=0.01)
    ica = ORICA(mode="decay", n_channels=22, block_update=True, size_block=4, stride=4, lm_0=.995, lw_0=.995, gamma=0.6, n_sub=3)
    eeg_sources_data = ica.transform_epochs(eeg_epochs)
    #eeg_sources_data = np.stack([ica.w @ eeg_epoch_data for eeg_epoch_data in eeg_sources_data])
    
    
    
    print("EOG")
    ica_eog = FastICA(3)
    n_epochs, n_channels, n_times = eog_data.shape
    eog_data = ica_eog.fit_transform(
        eog_data.transpose(1, 0, 2).reshape(n_channels, -1).T
    ).T.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)

    #ica_eog = ORICA(mode="decay", n_channels=3, block_update=True, size_block=8, stride=8, lm_0=.995, lw_0=.995, gamma=0.4, n_sub=2)
    #eog_data = ica_eog.transform_epochs(eog_epochs)
    #eog_data = np.stack([ica_eog.w @ eog_epoch_data for eog_epoch_data in eog_data])

    n_eeg_channels = eeg_sources_data.shape[1]
    n_eog_channels = eog_data.shape[1]

    assert len(eeg_sources_data) == len(eog_data)
    n_epochs = len(eeg_sources_data)

    plt.figure(figsize=(20, 4))
    plt.plot(np.log(ica.sigmas))
    plt.show()

    plot_various(
        eeg_epochs.get_data()[1],
        n=22,
        d=4e-5,
        title="EEG"
    )
    plot_various(
        eeg_sources_data[1],
        n=22,
        d=4,
        title="ICA EEG"
    )
    plot_various(
        eog_data[-1],
        n=3,
        d=1e-4,
        title="ICA EOG"
    )

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
            ax = axes[eog_channel]
            ax.axline((0, .5), (1, 0.5), c="r")
            ax.axline((0, .0), (1, 0.), c="g")
            ax.plot(pcc_list)
            ax.scatter(range(len(pcc_list)), pcc_list)
            ax.set_title("EEG {} - EOG {} | {:.3f}".format(eeg_channel, eog_channel, pcc))
            ax.set_ylim(-1, 1)
            ax.grid()
        plt.show()

    plt.imshow(ica.w)
    plt.title("W")
    plt.colorbar()
    plt.show()

    plt.imshow(ica.m)
    plt.title("M")
    plt.colorbar()
    plt.show()
    