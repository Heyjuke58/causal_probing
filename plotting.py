import argparse
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Tuple

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axis import Axis

from src.hyperparameter import AMOUNT_LAYERS
from src.probing_config import ProbingTask

LAYERS = [x for x in range(AMOUNT_LAYERS)]
SMALL_FONT_SIZE = 12
params = {
    "legend.fontsize": "large",
    "legend.title_fontsize": "large",
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)
MARKER_STYLES = ["s", "X", "d"]

REG_TASKS = {ProbingTask.BM25, ProbingTask.SEM, ProbingTask.AVG_TI, ProbingTask.TI}


class ReconstructionPlot:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __enter__(self):
        self.name = self.kwargs.get("name", "default_name")
        self.legend_title = self.kwargs.get("legend_title", "encodings trained on")
        self.fig, self.ax = plt.subplots()
        plt.ylabel(self.kwargs.get("ylabel", "Accuracy"))
        plt.xlabel(self.kwargs.get("xlabel", "Layer"))
        plt.title("\n".join(wrap(self.kwargs.get("title", "Title not given"), 60)))
        plt.xticks(LAYERS)
        plt.tight_layout()
        plt.grid()
        return self

    def __exit__(self, type, value, traceback):
        self.ax.legend(title=self.legend_title)
        self.fig.savefig("./plots/" + self.name + ".png")
        plt.clf()


class ReconstructionPlot1by2:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ax: Axis

    def __enter__(self):
        self.ax = self.kwargs.get("ax", None)
        self.legend_title = self.kwargs.get("legend_title", "embeddings")
        self.ax.set(xticks=LAYERS)
        self.ax.set(title="\n".join(wrap(self.kwargs.get("title", ""), 60)))
        self.ax.grid()
        return self

    def __exit__(self, type, value, traceback):
        self.ax.set(ylabel=self.kwargs.get("ylabel", "Accuracy"))
        self.ax.set(xlabel=self.kwargs.get("xlabel", "Layer"))
        plt.subplots_adjust(bottom=0.15, left=0.05, top=0.9)
        self.ax.legend(title=self.legend_title, loc="lower left")


class ReconstructionPlot2by2:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ax: Axis

    def __enter__(self):
        self.ax = self.kwargs.get("ax", None)
        self.legend_title = self.kwargs.get("legend_title", "embeddings")
        self.ax.set(xticks=LAYERS)
        self.ax.set(title="\n".join(wrap(self.kwargs.get("title", ""), 60)))
        self.ax.grid()
        return self

    def __exit__(self, type, value, traceback):
        self.ax.set(ylabel=self.kwargs.get("ylabel", "Accuracy"))
        self.ax.set(xlabel=self.kwargs.get("xlabel", "Layer"))
        plt.subplots_adjust(bottom=0.07, left=0.05, top=0.95)
        self.ax.legend(title=self.legend_title, loc="lower left")


def _reorganize_df(df, score_variant: str):
    assert score_variant in {"acc", "r2"}, f"Wrong score variant {score_variant} not one of acc, r2."
    df_orig = df[[f"{score_variant}_orig", "layer"]].rename(columns={f"{score_variant}_orig": "y"})
    df_orig["type"] = "original"
    df_probed = df[[f"{score_variant}_probed", "layer"]].rename(columns={f"{score_variant}_probed": "y"})
    df_probed["type"] = "probed"
    df_control = df[[f"{score_variant}_control", "layer"]].rename(columns={f"{score_variant}_control": "y"})
    df_control["type"] = "control"
    combined_df = pd.concat([df_orig, df_probed, df_control])
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def plot_reconstruction_2(df, task, baseline: dict):
    standard_kwargs_sns_lineplot = {
        "x": "layer",
        "y": "y",
        "hue": "type",
        "style": "type",
        "dashes": False,
        "markers": MARKER_STYLES,
        "markersize": 10,
        "palette": "colorblind",
    }
    if task in REG_TASKS:
        y_label = "R²"
        score_variant = "r2"
    else:
        y_label = "Accuracy"
        score_variant = "acc"
    args_baseline_linear = [LAYERS, [baseline["linear"] for _ in range(AMOUNT_LAYERS)]]
    args_baseline_mlp = [LAYERS, [baseline["mlp"] for _ in range(AMOUNT_LAYERS)]]
    kwargs_baseline = {
        "label": "random",
        "color": "grey",
        "linestyle": "dashed",
    }
    fig, axs = plt.subplots(1, 2, sharex="row", sharey="row")
    fig.set_size_inches(12, 3.5)
    fig.tight_layout()

    with ReconstructionPlot1by2(ax=axs[0], title=f"Linear probe model", ylabel=y_label) as plot:
        plot.ax.plot(*args_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], score_variant)
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot1by2(ax=axs[1], title=f"MLP probe model", ylabel=y_label) as plot:
        plot.ax.plot(*args_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], score_variant)
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    fig.savefig(f"./plots/{task}_reconstruction.png", dpi=300)


def plot_reconstruction(df, task, reg_baselines: dict, clf_baslines: dict):
    standard_kwargs_sns2 = {}
    standard_kwargs_sns_scatterplot = {
        "x": "layer",
        "y": "y",
        "hue": "type",
        "style": "type",
        "markers": MARKER_STYLES,
        "s": 100,
        "palette": "colorblind",
    }
    standard_kwargs_sns_lineplot = {
        "x": "layer",
        "y": "y",
        "hue": "type",
        "style": "type",
        "dashes": False,
        "markers": MARKER_STYLES,
        "markersize": 10,
        "palette": "colorblind",
    }
    args_clf_baseline_linear = [LAYERS, [clf_baslines["linear"] for _ in range(AMOUNT_LAYERS)]]
    args_clf_baseline_mlp = [LAYERS, [clf_baslines["mlp"] for _ in range(AMOUNT_LAYERS)]]
    args_reg_baseline_linear = [LAYERS, [reg_baselines["linear"] for _ in range(AMOUNT_LAYERS)]]
    args_reg_baseline_mlp = [LAYERS, [reg_baselines["mlp"] for _ in range(AMOUNT_LAYERS)]]
    kwargs_baseline = {
        "label": "random",
        "color": "grey",
        "linestyle": "dashed",
    }

    fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
    fig.set_size_inches(12, 7)
    fig.tight_layout()

    with ReconstructionPlot2by2(ax=axs[0, 0], title=f"Linear probe model", ylabel="Accuracy", xlabel=None) as plot:
        plot.ax.plot(*args_clf_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "acc")
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[0, 1], title=f"MLP probe model", ylabel="Accuracy", xlabel=None) as plot:
        plot.ax.plot(*args_clf_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "acc")
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[1, 0], ylabel="R²") as plot:
        plot.ax.plot(*args_reg_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "r2")
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[1, 1], ylabel="R²") as plot:
        plot.ax.plot(*args_reg_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "r2")
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    fig.savefig(f"./plots/{task}_reconstruction.png", dpi=300)
    plt.clf()

    fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
    fig.set_size_inches(12, 7)
    fig.tight_layout()

    with ReconstructionPlot2by2(ax=axs[0, 0], title=f"Linear probe model", ylabel="Accuracy", xlabel=None) as plot:
        plot.ax.plot(*args_clf_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "acc")
        sns.scatterplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_scatterplot)

    with ReconstructionPlot2by2(ax=axs[0, 1], title=f"MLP probe model", ylabel="Accuracy", xlabel=None) as plot:
        plot.ax.plot(*args_clf_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "acc")
        sns.scatterplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_scatterplot)

    with ReconstructionPlot2by2(ax=axs[1, 0], ylabel="R²") as plot:
        plot.ax.plot(*args_reg_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "r2")
        sns.scatterplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_scatterplot)

    with ReconstructionPlot2by2(ax=axs[1, 1], ylabel="R²") as plot:
        plot.ax.plot(*args_reg_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "r2")
        sns.scatterplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_scatterplot)

    fig.savefig(f"./plots/{task}_reconstruction_scatterplot.png", dpi=300)


def plot_behavior_heatmap(df, task: ProbingTask):
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    pivot_df = df.pivot(index="exp", columns="layer", values="ndcg_10")
    pivot_df.index = pd.CategoricalIndex(
        pivot_df.index,
        categories=["intervention\non document", "intervention\non query\n& document", "control", "baseline"],
    )
    pivot_df.sort_index(level=0, inplace=True)
    sns.heatmap(data=pivot_df, annot=True, cmap="crest", xticklabels=True)
    ax.yaxis.set_label_position("right")
    plt.ylabel("NDCG@10")
    plt.yticks(rotation=0)
    # plt.xticks(LAYERS)
    plt.xlabel("Layer")
    plt.subplots_adjust(bottom=0.1, left=0.15)
    fig.savefig(f"./plots/{task}_behaviour_heatmap.png", dpi=300)
    pass


def plot_behavior(df, baseline: float, task: ProbingTask):
    standard_kwargs_sns = {
        "markers": MARKER_STYLES,
        "markersize": 10,
        "style": "exp",
        "dashes": False,
        "palette": "colorblind",
    }
    standard_kwargs_sns_scatter = {
        "markers": MARKER_STYLES,
        "s": 100,
        "style": "exp",
        "alpha": 0.8,
        "palette": "colorblind",
    }
    kwargs_baseline = {
        "label": "no intervention",
        "color": "grey",
        "linestyle": "dashed",
    }
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    plt.xticks(LAYERS)
    plt.ylabel("NDCG@10")
    plt.xlabel("Layer")
    plt.grid()
    plt.ylim(0.2, 0.75)
    # plt.subplots_adjust(bottom=0.17)

    # sns.lineplot(data=df, x="layer", y="ndcg_10", hue="exp", **standard_kwargs_sns)
    sns.scatterplot(data=df, x="layer", y="ndcg_10", hue="exp", **standard_kwargs_sns_scatter)
    # sns.barplot(data=df, x="layer", y="ndcg_10", hue="exp", width=0.7)
    plt.plot(LAYERS, [baseline for _ in range(AMOUNT_LAYERS)], **kwargs_baseline)
    plt.legend(title="experiment", loc="lower left")
    fig.savefig(f"./plots/{task}_behaviour_scatterplot.png", dpi=300)
    pass


def prepare_behavior_plot(task: ProbingTask, viz_type: str):
    df_dict = defaultdict(list)
    for file in Path("./logs/results/behavior/").rglob(f"trec_eval*{task}*layer*.tsv"):
        with open(file, "r") as f:
            for line in f.readlines():
                if line[0] == "M":
                    continue
                l = line[:-1].split("\t")
                ndcg_10 = float(l[1])
                layer = int(file.name.split("_")[6])
                exp = "intervention\non document"
                if "control" in file.name:
                    exp = "control"
                elif "altered_query_embeddings" in file.name:
                    exp = "intervention\non query\n& document"
                df_dict["layer"].append(layer)
                df_dict["ndcg_10"].append(ndcg_10)
                df_dict["exp"].append(exp)

    baseline_ndcg_10 = 0
    with open("./logs/reproduction/results/trec_eval_2023_02_23-09-48-36.tsv", "r") as f:  # file path to baseline values
        for line in f.readlines():
            if line[0] == "M":
                continue
            l = line[:-1].split("\t")
            baseline_ndcg_10 = float(l[1])

    if viz_type == "heatmap":
        for i in range(AMOUNT_LAYERS):
            df_dict["layer"].append(i)
            df_dict["ndcg_10"].append(baseline_ndcg_10)
            df_dict["exp"].append("baseline")

    return pd.DataFrame.from_dict(df_dict), baseline_ndcg_10


def prepare_reconstruction_plot(task: ProbingTask) -> Tuple[pd.DataFrame, Dict, Dict]:
    df_dict = defaultdict(list)
    for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*reconstruction.log"):
        try:
            task_str, merging, _, layer, _ = file.name.split("_")
        except:
            task_str, merging, suf, _, layer, _ = file.name.split("_")
            merging = merging + "_" + suf

        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                if line[0] != "m" and line[0] != "r":
                    l = line[:-1].split("\t")
                    l = [float(x) if not x in ["linear", "mlp"] else x for x in l]
                    r2_orig, r2_probed, r2_control, r2_diff, acc_orig, acc_probed, acc_control, acc_diff, acc_maj, model_type = l

                    # df_dict["task"].append(task_str)
                    df_dict["merging"].append(merging)
                    df_dict["model_type"].append(model_type)
                    df_dict["layer"].append(int(layer))
                    df_dict["r2_orig"].append(r2_orig)
                    df_dict["r2_probed"].append(r2_probed)
                    df_dict["r2_control"].append(r2_control)
                    df_dict["r2_diff"].append(r2_diff)
                    df_dict["acc_orig"].append(acc_orig)
                    df_dict["acc_probed"].append(acc_probed)
                    df_dict["acc_control"].append(acc_control)
                    df_dict["acc_diff"].append(acc_diff)

    df = pd.DataFrame.from_dict(df_dict)

    model_types = ["linear", "mlp"]
    reg_baselines = {}
    clf_baselines = {}

    for m in model_types:
        with open(f"./logs/results/ablation/{task}_regressor_{m}_baseline.log", "r") as f:
            reg_baselines[m] = float(f.read())
        with open(f"./logs/results/ablation/{task}_classification_{m}_baseline.log", "r") as f:
            clf_baselines[m] = float(f.read())

    return df, reg_baselines, clf_baselines


def prepare_reconstruction_plot_2(task: ProbingTask, normalized_target: bool) -> Tuple[pd.DataFrame, Dict]:
    df_dict = defaultdict(list)
    reg = True if task in REG_TASKS else False
    if normalized_target:
        path = f"{task}*normalized_target*reconstruction_{'reg' if reg else 'clf'}.log"
        files = [file for file in Path(f"./logs/results/ablation/{task}/").rglob(path)]
    else:
        path = f"{task}*reconstruction_{'reg' if reg else 'clf'}.log"
        files = [file for file in list(Path(f"./logs/results/ablation/{task}/").rglob(path)) if not "normalized_target" in file.name]
    for file in files:
        s = file.name.split("_")
        layer = s[-3] if not "normalized_target" in file.name else s[-5]

        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                if line[0] != "a" and line[0] != "r":
                    l = line[:-1].split("\t")
                    l = [float(x) if not x in ["linear", "mlp"] else x for x in l]
                    if reg:
                        r2_orig, r2_probed, r2_control, model_type = l
                        df_dict["r2_orig"].append(r2_orig)
                        df_dict["r2_probed"].append(r2_probed)
                        df_dict["r2_control"].append(r2_control)

                    else:
                        acc_orig, acc_probed, acc_control, model_type = l
                        df_dict["acc_orig"].append(acc_orig)
                        df_dict["acc_probed"].append(acc_probed)
                        df_dict["acc_control"].append(acc_control)

                    df_dict["model_type"].append(model_type)
                    df_dict["layer"].append(int(layer))

    df = pd.DataFrame.from_dict(df_dict)

    model_types = ["linear", "mlp"]
    baselines = {}
    normlize_str = "_normalized_target" if normalized_target else ""

    for m in model_types:
        with open(
            f"./logs/results/ablation/{task}/{task}_{'regressor' if reg else 'classification'}_{m}{normlize_str}_baseline.log", "r"
        ) as f:
            baselines[m] = float(f.read())

    return df, baselines


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reconstruction", dest="reconstruction", action="store_true")
    parser.add_argument("-p", "--bucketized_and_reg", dest="bucketized_and_reg", action="store_true")
    parser.add_argument("-n", dest="normalized_target", action="store_true")
    parser.add_argument("-b", "--behavior", dest="behavior", action="store_true")
    parser.add_argument(
        "-t",
        "--task",
        type=ProbingTask,
        dest="task",
        default=ProbingTask.BM25,
        choices=list(ProbingTask),
        help="Task to plot.",
    )
    parser.add_argument("-v", "--viz_type", dest="viz_type", default="lineplot", choices=["lineplot", "heatmap"])

    args = parser.parse_args()

    return args


def main(reconstruction: bool, behavior: bool, task: ProbingTask, viz_type: str, normalized_target: bool, bucketized_and_reg: bool):
    if reconstruction:
        if bucketized_and_reg:
            df, reg_baselines, clf_baselines = prepare_reconstruction_plot(task)
            plot_reconstruction(df, task, reg_baselines, clf_baselines)
        else:
            df, baselines = prepare_reconstruction_plot_2(task, normalized_target)
            plot_reconstruction_2(df, task, baselines)
    if behavior:
        df, baseline = prepare_behavior_plot(task, viz_type)
        if viz_type == "heatmap":
            plot_behavior_heatmap(df, task)
        elif viz_type == "lineplot":
            plot_behavior(df, baseline, task)


if __name__ == "__main__":
    args = parse_arguments()
    args = vars(args)
    main(**args)
