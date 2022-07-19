import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import bootstrap, wilcoxon

N_BOOT = 100

metrics = ["bas", "kappa", "acc"]


def flatten_results(results_dict):
    lines = list()
    columns = ["uid", "algorithm", "metric", "value"]
    for uid, alg_metrics in results_dict.items():
        for alg_name, metrics in alg_metrics.items():
            for metric_name, value in metrics.items():
                lines.append(
                    [uid, alg_name, metric_name, value]
                )
    results_df = pd.DataFrame(lines, columns=columns)
    return results_df


def plot_sub_barplot(results_df, metric="kappa", algorithms=None, figsize=None, save_filepath=None):

    if figsize is None:
        figsize = (12, 5)
    if algorithms is None:
        algorithms = results_df.algorithm.unique()

    df = results_df.query("algorithm in @algorithms")

    fig = plt.figure(figsize=figsize)
    plt.grid()
    sns.barplot(x="uid", y=metric, data=df, hue="algorithm", ax=plt.gca(), n_boot=N_BOOT)

    if save_filepath is not None:
        fig.savefig(save_filepath)

    plt.show()


def rename_uid(df):
    return df.rename(columns=dict(uid="Subject"))


def single(ax, labels):
    for i, (p, label) in enumerate(zip(ax.patches, labels)):
        _x = p.get_x() + p.get_width() / 2 - (p.get_width() / 3 if i == 1 else 0)
        _y = (p.get_y() + p.get_height() - len(label) / 50) / 2
        ax.text(_x, _y, label.upper(), ha="center", rotation=90)


def plot_best_algorithm(results_df, metric="kappa", algorithms=None, figsize=None, save_filepath=None):
    if algorithms is None:
        algorithms = results_df.algorithm.unique()

    df = results_df.query("(algorithm in @algorithms)").groupby(["uid", "algorithm"], as_index=False).mean()
    df = df.sort_values(by=metric, ascending=False).drop_duplicates(subset="uid")
    df = results_df.query("(algorithm in @algorithms)").merge(df[["uid", "algorithm"]], on=["uid", "algorithm"], how="inner").sort_values(by="uid", ascending=True)
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()
    ax.grid()
    sns.barplot(x="Subject", y=metric, data=rename_uid(df), label="algorithm", ax=ax, n_boot=N_BOOT)
    single(ax, df.drop_duplicates(subset=["uid"]).algorithm.to_numpy())
    ax.tick_params(labelrotation=45)
    ax.set_ylim((0, 1))
    ax.set_xlabel("Subject")
    ax.set_title(f"Best {metric} per Subject", fontsize=12)

    if save_filepath is not None:
        fig.savefig(save_filepath)

    plt.show()


def get_wilcoxon_algorithms(results_df, uid, alg1, alg2, alternative="greater", metric="kappa"):
    a = results_df.query("(uid == @uid) and (algorithm == @alg1)").sort_values(by="run")[metric]
    b = results_df.query("(uid == @uid) and (algorithm == @alg2)").sort_values(by="run")[metric]

    return wilcoxon(a, b, alternative=alternative, zero_method="zsplit")


def get_wilcoxon(results_df, alg1, alg2, alternative="greater", metric="kappa"):
    r1 = results_df.query("(algorithm == @alg1)").sort_values(by=["uid", "run"])[metric]
    r2 = results_df.query("(algorithm == @alg2)").sort_values(by=["uid", "run"])[metric]
    return wilcoxon(r1, r2, alternative=alternative, zero_method="zsplit")


def sortedgroupedbar(results_df, metric="kappa", save_filepath=None):
    w = 5
    fig = plt.figure(figsize=(20, 6), dpi=120)
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    algs = results_df.algorithm.unique()
    n_algs = len(algs)
    algorithm_color_dict = {
        alg: cmap(i / n_algs)
        for i, alg
        in enumerate(algs)
    }

    x_c = w / 2
    legends = [
        Patch(
            facecolor=algorithm_color_dict[alg],
            edgecolor=None,
            label=alg
        )
        for alg in algs
    ]
    algorithms = results_df.algorithm.unique()
    uids = results_df.uid.unique()

    for uid in uids:
        x_list = list()
        algorithms = results_df.query("uid == @uid").groupby("algorithm", as_index=False).mean().sort_values(by=metric).algorithm.to_numpy()
        best_algorithm = algorithms[-1]
        for algorithm in algorithms:
            df = results_df.query("(uid == @uid) and (algorithm == @algorithm)")
            if df[metric].nunique() > 1:
                res = bootstrap((df[metric], ), np.mean, n_resamples=100)
                low = res.confidence_interval.low
                high = res.confidence_interval.high
            else:
                low, high = 0, 0
            avg = df[metric].mean()
            low, high = avg - low, high - avg
            x_list.append(x_c)
            ax.bar(x_c, avg, width=w, color=algorithm_color_dict[algorithm], yerr=([low], [high]))
            if (algorithm != best_algorithm):
                pvalue = get_wilcoxon_algorithms(results_df, uid, best_algorithm, algorithm, metric=metric).pvalue
                if (pvalue < 0.05):
                    ax.text(x_c, -0.03 if avg > 0 else 0.03, "*", ha="center", va="center", fontsize=20, color="r")
            x_c += w
        x_c += w * 2

        mid = np.mean(x_list)
        ax.text(mid, -0.1, uid, horizontalalignment="center", fontsize=15)
    ax.set_xlabel("Subject", fontsize=20)

    for loc in ["right", "left", "top", "bottom"]:
        ax.spines[loc].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid()
    ax.legend(handles=legends, loc=(1, .2), fontsize=15)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_title(f"{metric} per Subject, per ICA method", fontsize=20)

    if save_filepath is not None:
        fig.savefig(save_filepath)

    plt.show()


def plot_average_algorithm_metric(results_df, metric="kappa", save_filepath=None):
    w = 5
    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    algs = results_df.algorithm.unique()
    n_algs = len(algs)
    algorithm_color_dict = {
        alg: cmap(i / n_algs)
        for i, alg
        in enumerate(algs)
    }

    x_c = w / 2
    legends = [
        Patch(
            facecolor=algorithm_color_dict[alg],
            edgecolor=None,
            label=alg
        )
        for alg in algs
    ]
    sorted_df = results_df.groupby(["uid", "algorithm"]).mean().groupby("algorithm").mean().sort_values(by=metric).reset_index()
    x_c = 0
    best_algorithm = sorted_df.algorithm.to_numpy()[-1]
    for algorithm in sorted_df.algorithm:
        df = results_df.query("algorithm == @algorithm")
        res = bootstrap((df[metric], ), np.mean, n_resamples=100)
        x_c += 1.5 * w
        avg = df[metric].mean()
        ax.bar(x_c, avg, w, yerr=([avg - res.confidence_interval.low], [res.confidence_interval.high - avg]))
        ax.set_xticks([])
        ax.text(x_c, -0.03, algorithm.upper(), horizontalalignment="center", fontsize=15)
        ax.text(x_c, -0.06, r"$\bar\rho={:.3f}$".format(avg), horizontalalignment="center", fontsize=15, usetex=True)
        if (best_algorithm != algorithm):
            pvalue = get_wilcoxon(results_df, best_algorithm, algorithm, metric=metric).pvalue
            if (pvalue < 0.05):
                ax.text(x_c, res.confidence_interval.high, "*", horizontalalignment="center", fontsize=25, color="r")
    ax.legend(handles=legends, loc=(1, .2), fontsize=15)
    ax.set_ylabel(metric, fontsize=20)
    ax.grid()
    ax.set_title(f"Average {metric} for each ICA algorithm", fontsize=20)

    if save_filepath is not None:
        fig.savefig(save_filepath)

    plt.show()


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
