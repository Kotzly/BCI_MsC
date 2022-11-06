import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import bootstrap, wilcoxon
from pathlib import Path

N_BOOT = 100


def annotate_bars(ax, labels):
    for i, (bar, err_line, label) in enumerate(zip(ax.patches, ax.get_lines(), labels)):
        _x = bar.get_x() + bar.get_width() / 2
        # _y = (p.get_y() + p.get_height() - len(label) / 50) / 2
        _y = bar.get_y() + err_line.get_ydata()[1] + (err_line.get_ydata()[1] - bar.get_height())
        ax.text(_x, _y, label, ha="center", rotation=90)


def ranked_barplot(results_df, grouping_col="algorithm", x_col="uid", val_col="kappa", figsize=None, x_label=None, save_filepath=None):
    x_label = x_label if x_label is not None else x_col
    
    # Select best algorithm per subject
    highest_df = (
        results_df
        .groupby([x_col, grouping_col], as_index=False)
        .mean()
        .sort_values(by=val_col, ascending=False)
        .drop_duplicates(subset=x_col)
    )

    # Filter original results_df to only include best algorithm per subject 
    df = (
        results_df
        .merge(highest_df[[x_col, grouping_col]], on=[x_col, grouping_col], how="inner")
        .sort_values(by=x_col, ascending=True)
    )
    label_df = highest_df[[x_col, grouping_col]].sort_values(by=x_col, ascending=True)

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    ax.grid()
    sns.barplot(x=x_col, y=val_col, data=df, ax=ax, n_boot=N_BOOT, order=label_df[x_col])

    annotate_bars(ax, label_df[grouping_col])
    ax.tick_params(labelrotation=45)
    ax.set_ylim((0, 1))
    ax.set_xlabel(x_label)
    ax.set_title(f"Best {val_col} per {x_label}", fontsize=12)

    if save_filepath is not None:
        fig.savefig(save_filepath)
    

def detailed_barplot(results_df, x_col="uid", hue_col="algorithm", val_col="Kappa", order_col="run", save_filepath=None, w=5, cmap="nipy_spectral", x_label=None):
    if x_label is None:
        x_label = x_col

    fig = plt.figure(figsize=(20, 6), dpi=120)
    ax = plt.gca()

    x_c = w / 2
    cmap = plt.get_cmap(cmap)

    hues = results_df[hue_col].unique()
    n_hues = len(hues)
    hue_color_dict = {
        hue: cmap(i / n_hues)
        for i, hue
        in enumerate(hues)
    }
    legends = [
        Patch(
            facecolor=hue_color_dict[hue],
            edgecolor=None,
            label=hue
        )
        for hue in hues
    ]

    x_values = results_df[x_col].unique()

    for x in x_values:
        x_df = results_df[results_df[x_col] == x]
        x_list = list()
        # Sort algorithms by mean Kappa value
        ordered_hue = (
            x_df
            .groupby(hue_col, as_index=False)
            .mean()
            .sort_values(by=val_col)[hue_col]
            .to_numpy()
        )
        best_hue = ordered_hue[-1]
        best_hue_df = x_df[x_df[hue_col] == best_hue]

        for hue in ordered_hue:
            hue_df = x_df[x_df[hue_col] == hue]
            if hue_df[val_col].nunique() > 1:
                res = bootstrap((hue_df[val_col], ), np.mean, n_resamples=100)
                low = res.confidence_interval.low
                high = res.confidence_interval.high
            else:
                low, high = 0, 0

            avg = hue_df[val_col].mean()
            low, high = avg - low, high - avg

            x_list.append(x_c)
            ax.bar(x_c, avg, width=w, color=hue_color_dict[hue], yerr=([low], [high]))

            if (hue != best_hue):

                pvalue = wilcoxon(hue_df[val_col], best_hue_df[val_col], alternative="less", zero_method="zsplit").pvalue
                if (pvalue < 0.05):
                    ax.text(x_c, -0.03 if avg > 0 else 0.03, "*", ha="center", va="center", fontsize=20, color="r")
            x_c += w
        x_c += w * 2

        mid = np.mean(x_list)
        ax.text(mid, -0.1, x, horizontalalignment="center", va="center_baseline", fontsize=15, rotation=-45)
    ax.set_xlabel(x_label, fontsize=20)

    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid()
    ax.legend(handles=legends, loc=(1, .2), fontsize=15)
    ax.set_ylabel(val_col, fontsize=20)
    ax.set_title(f"{val_col} per {x_col}, per {hue_col}", fontsize=20)

    # Ensure figure doesnt get cropped during save
    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath)


def average_barplot(results_df, x_col="algorithm", grouping_col="uid", val_col="Kappa", order_col="run", save_filepath=None, w=5, cmap="nipy_spectral", x_label=None, n_boots=N_BOOT):
    if x_label is None:
        x_label = x_col

    fig = plt.figure(figsize=(20, 6), dpi=120)
    ax = plt.gca()

    x_c = w / 2
    cmap = plt.get_cmap(cmap)
    
    hues = results_df[x_col].unique()
    n_hues = len(hues)
    hue_color_dict = {
        hue: cmap(i / n_hues)
        for i, hue
        in enumerate(hues)
    }
    legends = [
        Patch(
            facecolor=hue_color_dict[hue],
            edgecolor=None,
            label=hue
        )
        for hue in hues
    ]

    ordered_x_df = results_df.groupby([x_col, grouping_col]).mean().groupby(x_col).mean().sort_values(by=val_col).reset_index()
    best_x = ordered_x_df[x_col].to_numpy()[-1]
    best_x_df = results_df[results_df[x_col] == best_x]
    x_c = 0
    for x in ordered_x_df[x_col]:
        x_c += 1.5 * w
        
        x_df = results_df[results_df[x_col] == x]
        res = bootstrap((x_df[val_col], ), np.mean, n_resamples=n_boots)
        avg = x_df[val_col].mean()

        ax.bar(x_c, avg, w, yerr=([avg - res.confidence_interval.low], [res.confidence_interval.high - avg]), color=hue_color_dict[x])
        ax.set_xticks([])
        ax.text(x_c, -0.025, x, horizontalalignment="center", fontsize=15)
        ax.text(x_c, -0.05, r"$\bar\rho={:.3f}$".format(avg), horizontalalignment="center", fontsize=15, usetex=True)
        if (x != best_x):
            pvalue = wilcoxon(
                x_df.sort_values(by=[grouping_col])[val_col],
                best_x_df.sort_values(by=[grouping_col])[val_col],
                alternative="less",
                zero_method="zsplit"
            ).pvalue
            
            if (pvalue < 0.05):
                ax.text(x_c, res.confidence_interval.high, "*", horizontalalignment="center", fontsize=25, color="r")
    #ax.legend(handles=legends, loc=(1, .2), fontsize=15)
    ax.set_ylabel(val_col, fontsize=20)
    ax.grid()
    ax.set_title(f"Average {val_col} for each {x_col}", fontsize=20)

    if save_filepath is not None:
        fig.savefig(save_filepath)


def boxplot_algorithms(results_df, metric="kappa", save_filepath=None):
    algorithms = results_df.groupby(["uid", "algorithm"]).mean().groupby("algorithm").mean().sort_values(by=metric).reset_index().algorithm
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    sns.boxplot(
        x="Algorithm",
        y=metric,
        data=results_df.rename(
            columns=dict(algorithm="Algorithm")
        ),
        order=algorithms,
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)

    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=15)
    ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=15)

    if save_filepath is not None:
        fig.savefig(save_filepath)


def boxplot_subjects(results_df, metric="kappa", save_filepath=None):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    sns.boxplot(
        x="uid",
        y=metric,
        hue="Algorithm",
        data=results_df.rename(
            columns=dict(algorithm="Algorithm")
        ),
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)

    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=15)
    ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=15)

    if save_filepath is not None:
        fig.savefig(save_filepath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--p",
        dest="path",
        default="/home/paulo/Documents/GIT/BCI_MsC/experiments/issue_12/results.csv",
        type=Path,
        help="Path to results.csv"
    )
    parser.add_argument(
        "-save_folder",
        "--s",
        dest="save_folder",
        default="./plots",
        type=Path,
        help="folder_to_save"
    )
    args = parser.parse_args()

    results_df = pd.read_csv(args.path)

    classifiers = results_df.classifier.unique()
    algorithms = results_df.algorithm.unique()
    results_folder = args.save_folder
    results_folder.mkdir(exist_ok=True)

    for classifier in classifiers:

        ranked_barplot(
            results_df.query("classifier == @classifier"),
            x_col="uid",
            val_col="Kappa",
            grouping_col="algorithm",
            save_filepath=results_folder / f"{classifier}.png"
        )

        detailed_barplot(
            results_df.query("classifier == @classifier"),
            x_col="uid",
            hue_col="algorithm",
            val_col="Kappa",
            order_col="run",
            save_filepath=results_folder / f"detailed_{classifier}.png",
            w=5,
            cmap="nipy_spectral",
            x_label=None
        )

    for algorithm in algorithms:
        ranked_barplot(
            results_df.query("algorithm == @algorithm"),
            x_col="uid",
            val_col="Kappa",
            grouping_col="classifier",
            save_filepath=results_folder / f"{algorithm}.png"
        )

        detailed_barplot(
            results_df.query("algorithm == @algorithm"),
            x_col="uid",
            hue_col="classifier",
            val_col="Kappa",
            order_col="run",
            save_filepath=results_folder / f"detailed_{algorithm}.png",
            w=5,
            cmap="nipy_spectral",
            x_label=None
        )

    for algorithm in algorithms:
        
        detailed_barplot(
            results_df.query("algorithm == @algorithm"),
            x_col="classifier",
            hue_col="uid",
            val_col="Kappa",
            order_col="run",
            save_filepath=results_folder / f"classifier_comparison_{algorithm}.png",
            w=5,
            cmap="nipy_spectral",
            x_label=None
        )