import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


for uid in map(str, range(1, 10)):
    for trial_number in range(10):

        folderpath = Path(f"./logs/subject_{uid}/trial_{trial_number}")

        result_df = pd.read_csv(folderpath / "metrics.csv")

        train_df = result_df.dropna(subset=["train_cohen_kappa_score"])
        val_df = result_df.dropna(subset=["val_cohen_kappa_score"])

        metrics = [
            "cohen_kappa_score",
            "loss"
        ]

        for metric in metrics:

            x_train = train_df.epoch
            y_train = train_df[f"train_{metric}"]

            x_val = val_df.epoch
            y_val = val_df[f"val_{metric}"]

            plt.clf()
            plt.figure(figsize=(15, 6))
            plt.plot(x_train, y_train, label="Train")
            plt.plot(x_val, y_val, label="Validation")
            plt.legend()
            plt.title(metric)
            # plt.savefig(folderpath / f"{metric}.png")
            plt.grid()
            plt.savefig(f"./{uid}_{trial_number}_{metric}.png")
