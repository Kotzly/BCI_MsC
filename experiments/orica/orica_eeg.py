from ica_benchmark.processing.orica import ORICA
from ica_benchmark.io.load import BCI_IV_Comp_Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    filepath = "/home/paulo/Documents/datasets/BCI_Comp_IV_2a/gdf/A01T.gdf"

    epochs = BCI_IV_Comp_Dataset.load_as_epochs(filepath, load_eog=True, tmin=0., tmax=6.).load_data().filter(0., 30.)
    eeg_epochs = epochs.copy().pick("eeg")
    eog_epochs = epochs.copy().pick("eog")
    eog_data = eog_epochs.get_data()

    #ica = ORICA(mode="adaptative", block_update=True, size_block=8, stride=8, lm_0=.1, lw_0=.1)
    #ica = ORICA(n_channels=22, mode="decay", block_update=False, size_block=32, stride=4, gamma=0.6, lm_0=.995, lw_0=.995)
    #ica = ORICA(mode="decay", n_channels=22, block_update=True, size_block=4, stride=4, gamma=.8, lm_0=.95, lw_0=.95)
    #ica = ORICA(mode="constant", n_channels=22, block_update=True, size_block=8, stride=8, lw_0=.078, lm_0=0.078)
    ica = ORICA(mode="constant", n_channels=22, block_update=True, size_block=8, stride=8, lw_0=.05, lm_0=0.005)
    
    #ica = ORICA(mode="adaptative", block_update=True, size_block=32, stride=32, lm_0=.1, lw_0=.1)
    eeg_sources_data = ica.transform_epochs(eeg_epochs)

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
                )
                pcc_list.append(pcc)
                # print(
                #    "EEG: {} - EOG: {} -> {:.3f}".format(
                #        eeg_channel,
                #        eog_channel,
                #        pcc
                #    )
                #)
            ax = axes[eog_channel]
            ax.plot(pcc_list)

            ax.set_ylim(-1, 1)
            ax.grid()
        plt.show()
            