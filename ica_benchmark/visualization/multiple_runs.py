import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import bootstrap, wilcoxon
from pathlib import Path

N_BOOT = 100


def annotate_bars(ax, labels):
    bottom_lim, top_lim = ax.get_ylim()
    mid = (top_lim - bottom_lim) / 2
    for bar, err_line, label in zip(ax.patches, ax.get_lines(), labels):
        _x = bar.get_x() + bar.get_width() / 2
        bar_height = bar.get_height()
        if bar_height > mid:
            _y = (bar.get_y() + err_line.get_ydata()[0]) / 2
        else:
            _y = (top_lim + err_line.get_ydata()[1]) / 2
        # err_line.get_ydata() -> (bottom error, top error)
        ax.text(_x, _y, label, ha="center", rotation=90)


def best_per_group_barplot(
    results_df,
    grouping_cols="algorithm",
    x_col="uid",
    val_col="kappa",
    figsize=None,
    x_label=None,
    save_filepath=None,
    ylim=None,
    title=None,
):
    x_label = x_label if x_label is not None else x_col
    ylim = ylim if ylim is not None else (0, 1)
    title = title if title is not None else f"Best {val_col} per {x_label}"
    grouping_cols = (
        grouping_cols if isinstance(grouping_cols, list) else [grouping_cols]
    )
    # Select best algorithm per subject
    merge_cols = [x_col] + grouping_cols
    highest_df = (
        results_df.groupby(merge_cols, as_index=False)
        .mean()
        .sort_values(by=val_col, ascending=False)
        .drop_duplicates(subset=x_col)
    )

    # Filter original results_df to only include best algorithm per subject
    df = results_df.merge(highest_df[merge_cols], how="inner").sort_values(
        by=x_col, ascending=True
    )
    label_df = highest_df[merge_cols].sort_values(by=x_col, ascending=True)

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    ax.grid()
    sns.barplot(
        x=x_col, y=val_col, data=df, ax=ax, n_boot=N_BOOT, order=label_df[x_col]
    )

    bar_labels = label_df[grouping_cols].apply(lambda x: "\n".join(map(str, x)), axis=1)
    annotate_bars(ax, bar_labels)
    ax.tick_params(labelrotation=60)
    ax.set_ylim(ylim)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath)


def detailed_barplot(
    results_df,
    x_col="uid",
    hue_col="algorithm",
    val_col="Kappa",
    key_cols="run",
    save_filepath=None,
    w=5,
    cmap="nipy_spectral",
    x_label=None,
    y_label=None,
    title=None,
    ylim=None,
    figsize=None,
    labelpad=10,
    tick_pad=-0.1,
    ast_loc=(-0.03, 0.03),
):
    x_label = x_label if x_label is not None else x_col
    y_label = y_label if y_label is not None else val_col
    title = title if title is not None else f"{val_col} per {x_label}, per {hue_col}"
    ylim = ylim if ylim is not None else (0, 1)
    figsize = figsize if figsize is not None else (16, 6)
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = plt.gca()

    x_c = w / 2
    cmap = plt.get_cmap(cmap)

    hues = results_df[hue_col].unique()
    n_hues = len(hues)
    hue_color_dict = {hue: cmap(i / n_hues) for i, hue in enumerate(hues)}
    legends = [
        Patch(facecolor=hue_color_dict[hue], edgecolor=None, label=hue) for hue in hues
    ]

    x_values = results_df[x_col].unique()

    for x in x_values:
        x_df = results_df[results_df[x_col] == x]
        x_list = list()
        # Sort algorithms by mean Kappa value
        ordered_hue = (
            x_df.groupby(hue_col, as_index=False)
            .mean()
            .sort_values(by=val_col)[hue_col]
            .to_numpy()
        )
        best_hue = ordered_hue[-1]
        best_hue_df = x_df[x_df[hue_col] == best_hue]

        for hue in ordered_hue:
            hue_df = x_df[x_df[hue_col] == hue]
            if hue_df[val_col].nunique() > 1:
                res = bootstrap((hue_df[val_col],), np.mean, n_resamples=100)
                low = res.confidence_interval.low
                high = res.confidence_interval.high
            else:
                low, high = 0, 0

            avg = hue_df[val_col].mean()
            low, high = avg - low, high - avg

            x_list.append(x_c)
            ax.bar(x_c, avg, width=w, color=hue_color_dict[hue], yerr=([low], [high]))

            if hue != best_hue:
                pvalue = wilcoxon(
                    hue_df[val_col],
                    best_hue_df[val_col],
                    alternative="less",
                    zero_method="zsplit",
                ).pvalue
                if pvalue < 0.05:
                    ax.text(
                        x_c,
                        ast_loc[0] if avg > 0 else ast_loc[1],
                        "*",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="r",
                    )
            x_c += w
        x_c += w * 2

        mid = np.mean(x_list)
        ax.text(
            mid,
            tick_pad,
            x,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=15,
            rotation=-60,
        )

    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=12)
    tick_sep = (ylim[1] - ylim[0]) / 10
    ax.set_yticks(np.arange(ylim[0], ylim[1] + tick_sep / 10, tick_sep))
    ax.grid()
    ax.legend(handles=legends, loc=(1, 0.2), fontsize=15)
    ax.set_xlabel(x_label, fontsize=20, labelpad=labelpad)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(ylim)

    # Ensure figure doesnt get cropped during save
    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath)


def average_barplot(
    results_df,
    x_col="algorithm",
    grouping_cols="uid",
    val_col="Kappa",
    key_cols="run",
    save_filepath=None,
    w=5,
    cmap="nipy_spectral",
    x_label=None,
    y_label=None,
    n_boots=N_BOOT,
    title=None,
    label_rotation=90,
    figsize=None,
    labelpad=10,
    legend_in=None,
    ylim=None,
):
    x_label = x_col if x_label is None else x_label
    y_label = y_label if y_label is not None else val_col
    figsize = figsize if figsize is not None else (16, 6)
    key_cols = [key_cols] if isinstance(key_cols, str) else key_cols
    title = title if title is not None else f"Average {val_col} for each {x_col}"
    grouping_cols = [grouping_cols] if isinstance(grouping_cols, str) else grouping_cols

    assert legend_in in (
        "legend",
        "xlabel",
        None,
    ), "legend_in must be 'legend' or 'xlabel' or None"

    fig = plt.figure(figsize=figsize, dpi=120)
    ax = plt.gca()

    x_c = w / 2
    cmap = plt.get_cmap(cmap)

    hues = results_df[x_col].unique()
    n_hues = len(hues)
    hue_color_dict = {hue: cmap(i / n_hues) for i, hue in enumerate(hues)}
    legends = [
        Patch(facecolor=hue_color_dict[hue], edgecolor=None, label=hue) for hue in hues
    ]

    ordered_x_df = (
        results_df.groupby([x_col, *grouping_cols])
        .mean()
        .groupby(x_col)
        .mean()
        .sort_values(by=val_col)
        .reset_index()
    )
    best_x = ordered_x_df[x_col].to_numpy()[-1]
    best_x_df = results_df[results_df[x_col] == best_x]
    x_c = 0
    for x in ordered_x_df[x_col]:
        x_c += 1.5 * w

        x_df = results_df[results_df[x_col] == x]
        res = bootstrap((x_df[val_col],), np.mean, n_resamples=n_boots)
        avg = x_df[val_col].mean()

        ax.bar(
            x_c,
            avg,
            w,
            yerr=(
                [avg - res.confidence_interval.low],
                [res.confidence_interval.high - avg],
            ),
            color=hue_color_dict[x],
            label=(
                "{}\n$\\bar\\rho={:.3f}$".format(x, avg)
                if legend_in == "legend"
                else None
            ),
        )
        ax.set_xticks([])
        ax.text(
            x_c,
            -0.025,
            "{}\n$\\bar\\rho={:.3f}$".format(x, avg) if legend_in == "xlabel" else x,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=15,
            rotation=label_rotation,
        )
        # ax.text(x_c, -0.05, r"$\bar\rho={:.3f}$".format(avg), horizontalalignment="center", fontsize=15, usetex=True, rotation=60)
        if x != best_x:
            merge_cols = [*grouping_cols, *key_cols]
            best_metric_arr = best_x_df.merge(x_df[merge_cols], on=merge_cols)[val_col]
            try:
                pvalue = wilcoxon(
                    x_df[val_col],
                    best_metric_arr,
                    alternative="less",
                    zero_method="zsplit",
                ).pvalue
            except ValueError as value_exception:
                raise ValueError(
                    f"Could not calculate pvalue for by mering on {merge_cols}. Most likely you have not specified all key_cols.\n"
                    + str(value_exception)
                ) from value_exception

            if pvalue < 0.05:
                ax.text(
                    x_c,
                    res.confidence_interval.high,
                    "*",
                    horizontalalignment="center",
                    fontsize=25,
                    color="r",
                )

    ax.set_xlabel(x_label, fontsize=20, labelpad=labelpad)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title, fontsize=20)
    if legend_in == "legend":
        ax.legend(loc=(1.0, 0.0))
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath)


def boxplot_algorithms(results_df, metric="kappa", save_filepath=None):
    algorithms = (
        results_df.groupby(["uid", "algorithm"])
        .mean()
        .groupby("algorithm")
        .mean()
        .sort_values(by=metric)
        .reset_index()
        .algorithm
    )
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    sns.boxplot(
        x="Algorithm",
        y=metric,
        data=results_df.rename(columns=dict(algorithm="Algorithm")),
        order=algorithms,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=12)
    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)

    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=15, labelpad=10)
    ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=15)

    fig.tight_layout()
    if save_filepath is not None:
        fig.savefig(save_filepath)


def boxplot_subjects(results_df, metric="kappa", save_filepath=None):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    sns.boxplot(
        x="uid",
        y=metric,
        hue="Algorithm",
        data=results_df.rename(columns=dict(algorithm="Algorithm")),
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=12)
    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)

    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=15, labelpad=10)
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
        help="Path to results.csv",
    )
    parser.add_argument(
        "-save_folder",
        "--s",
        dest="save_folder",
        default="./plots",
        type=Path,
        help="folder_to_save",
    )
    args = parser.parse_args()

    results_df = pd.read_csv(args.path)

    classifiers = results_df.classifier.unique()
    algorithms = results_df.algorithm.unique()
    results_folder = args.save_folder
    results_folder.mkdir(exist_ok=True)

    for classifier in classifiers:
        classifier_df = results_df[results_df.classifier == classifier]
        best_per_group_barplot(
            classifier_df,
            x_col="uid",
            val_col="Kappa",
            grouping_cols="algorithm",
            save_filepath=results_folder
            / f"best_algorithm_per_subject_for_{classifier}.png",
            x_label="Subject",
        )

        detailed_barplot(
            classifier_df,
            x_col="uid",
            hue_col="algorithm",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"detailed_{classifier}.png",
            x_label="Subject",
        )
        detailed_barplot(
            classifier_df,
            x_col="algorithm",
            hue_col="uid",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"algorithm_comparison_for_{classifier}.png",
            x_label="Subject",
        )
        average_barplot(
            classifier_df,
            x_col="algorithm",
            grouping_cols="uid",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder
            / f"average_per_algorithm_for_{classifier}.png",
            n_boots=N_BOOT,
        )
        average_barplot(
            classifier_df,
            x_col="uid",
            grouping_cols="algorithm",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"average_per_subject_for_{classifier}.png",
            n_boots=N_BOOT,
        )
    for algorithm in algorithms:
        algorithm_df = results_df[results_df.algorithm == algorithm]
        best_per_group_barplot(
            algorithm_df,
            x_col="uid",
            val_col="Kappa",
            grouping_cols="classifier",
            save_filepath=results_folder
            / f"best_classifier_per_subject_for_{algorithm}.png",
            x_label="Subject",
        )
        detailed_barplot(
            algorithm_df,
            x_col="uid",
            hue_col="classifier",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"detailed_{algorithm}.png",
            x_label="Subject",
        )
        detailed_barplot(
            algorithm_df,
            x_col="classifier",
            hue_col="uid",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"classifier_comparison_for_{algorithm}.png",
            x_label="Subject",
        )
        average_barplot(
            algorithm_df,
            x_col="classifier",
            grouping_cols="uid",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder
            / f"average_per_classifier_for_{algorithm}.png",
            n_boots=N_BOOT,
        )
        average_barplot(
            algorithm_df,
            x_col="uid",
            grouping_cols="classifier",
            val_col="Kappa",
            key_cols="run",
            save_filepath=results_folder / f"average_per_subject_for_{algorithm}.png",
            n_boots=N_BOOT,
        )

    best_per_group_barplot(
        results_df,
        grouping_cols=["algorithm"],
        x_col="classifier",
        val_col="Kappa",
        figsize=None,
        x_label="Classifier",
        save_filepath=results_folder / "classifier_scores.png",
    )

    best_per_group_barplot(
        results_df,
        grouping_cols=["classifier"],
        x_col="algorithm",
        val_col="Kappa",
        figsize=None,
        x_label="Algorithm",
        save_filepath=results_folder / "algorithm_scores.png",
    )

    average_barplot(
        results_df,
        x_col="uid",
        grouping_cols=["algorithm", "classifier"],
        val_col="Kappa",
        key_cols="run",
        save_filepath=results_folder / "average_per_subject.png",
        n_boots=N_BOOT,
    )

    average_barplot(
        results_df,
        x_col="algorithm",
        grouping_cols=["uid", "classifier"],
        val_col="Kappa",
        key_cols="run",
        save_filepath=results_folder / "average_per_algorithm.png",
        n_boots=N_BOOT,
    )

    average_barplot(
        results_df,
        x_col="classifier",
        grouping_cols=["uid", "algorithm"],
        val_col="Kappa",
        key_cols="run",
        save_filepath=results_folder / "average_per_classifier.png",
        n_boots=N_BOOT,
    )
