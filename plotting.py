import argparse
import itertools
import logging
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Tuple

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axis import Axis

from src.hyperparameter import AMOUNT_LAYERS, LAST_LAYER_IDX
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


class ReconstructionPlot1by2:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ax: Axis

    def __enter__(self):
        self.ax = self.kwargs.get("ax", None)
        # self.legend_title = self.kwargs.get("legend_title", "representations")
        self.ax.set(xticks=LAYERS)
        self.ax.set(title="\n".join(wrap(self.kwargs.get("title", ""), 60)))
        self.ax.grid()
        return self

    def __exit__(self, type, value, traceback):
        self.ax.set(ylabel=self.kwargs.get("ylabel", "Accuracy"))
        self.ax.set(xlabel=self.kwargs.get("xlabel", "Layer"))
        self.ax.tick_params(left=self.kwargs.get("y_ticks", True))
        plt.subplots_adjust(bottom=0.17, left=0.05, top=0.92, right=0.85)
        if self.kwargs.get("legend", False):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width, box.height])
            self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # self.ax.legend(loc="center", fancybox=True, framealpha=0.5)
        else:
            self.ax.get_legend().remove()


class ReconstructionPlot2by2:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ax: Axis

    def __enter__(self):
        self.ax = self.kwargs.get("ax", None)
        # self.legend_title = self.kwargs.get("legend_title", "representations")
        self.ax.set(xticks=LAYERS)
        self.ax.set(title="\n".join(wrap(self.kwargs.get("title", ""), 60)))
        self.ax.grid()
        return self

    def __exit__(self, type, value, traceback):
        self.ax.set(ylabel=self.kwargs.get("ylabel", "Accuracy"))
        self.ax.set(xlabel=self.kwargs.get("xlabel", "Layer"))
        self.ax.tick_params(left=self.kwargs.get("y_ticks", True))
        plt.subplots_adjust(bottom=0.09, left=0.055, top=0.95, right=0.85)
        if self.kwargs.get("legend", False):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width, box.height])
            self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # self.ax.legend(loc="center", fancybox=True, framealpha=0.5)
        else:
            self.ax.get_legend().remove()


class SubspacePlot1by2:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ax: Axis

    def __enter__(self):
        self.ax = self.kwargs.get("ax", None)
        self.ax.set(xticks=LAYERS)
        self.ax.set(title="\n".join(wrap(self.kwargs.get("title", ""), 60)))
        return self

    def __exit__(self, type, value, traceback):
        self.ax.set(xlabel=self.kwargs.get("xlabel", "Layer"))
        self.ax.tick_params(left=self.kwargs.get("y_ticks", True))
        self.ax.set(ylabel=self.kwargs.get("y_label", "Rank of eliminated subspace"))
        # plt.subplots_adjust(bottom=0.15, left=0.05, top=0.9, right=0.9)


def _reorganize_df(df, score_variant: str):
    assert score_variant in {"acc", "r2"}, f"Wrong score variant {score_variant} not one of acc, r2."
    df_orig = df[[f"{score_variant}_orig", "layer"]].rename(columns={f"{score_variant}_orig": "y"})
    df_orig["type"] = "original"
    df_probed = df[[f"{score_variant}_probed", "layer"]].rename(columns={f"{score_variant}_probed": "y"})
    df_probed["type"] = "counterfactual"
    df_control = df[[f"{score_variant}_control", "layer"]].rename(columns={f"{score_variant}_control": "y"})
    df_control["type"] = "control"
    combined_df = pd.concat([df_orig, df_probed, df_control])
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def plot_reconstruction_2(df, task, maj_acc: float):
    standard_kwargs_sns_lineplot = {
        "x": "layer",
        "y": "y",
        "hue": "type",
        "style": "type",
        "err_kws": {"linestyle": "--", "hatch": "///"},
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
    args_maj_acc = [LAYERS, [maj_acc for _ in range(AMOUNT_LAYERS)]]
    kwargs_baseline = {
        "label": "majority\naccuracy",
        "color": "grey",
        "linestyle": "dashed",
    }
    fig, axs = plt.subplots(1, 2, sharex="row", sharey="row")
    fig.set_size_inches(13, 3)
    fig.tight_layout()

    with ReconstructionPlot1by2(ax=axs[0], title=f"Linear probe", ylabel=y_label) as plot:
        plot.ax.plot(*args_maj_acc, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], score_variant)
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot1by2(ax=axs[1], title=f"MLP probe", ylabel=y_label, legend=True, y_ticks=False) as plot:
        plot.ax.plot(*args_maj_acc, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], score_variant)
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    fig.suptitle(
        f"{str(task).upper().replace('_', ' ') if task != ProbingTask.QC_COARSE else 'QC'}", y=0.9, x=0.925, fontweight="bold", fontsize=18
    )
    fig.savefig(f"./plots/{task}_reconstruction.png", dpi=300)


def plot_reconstruction(df, task, maj_acc: float):
    standard_kwargs_sns_lineplot = {
        "x": "layer",
        "y": "y",
        "hue": "type",
        "style": "type",
        "err_kws": {"linestyle": "--", "hatch": "///"},
        "dashes": False,
        "markers": MARKER_STYLES,
        "markersize": 10,
        "palette": "colorblind",
    }
    args_maj_acc = [LAYERS, [maj_acc for _ in range(AMOUNT_LAYERS)]]
    kwargs_baseline = {
        "label": "majority\naccuracy",
        "color": "grey",
        "linestyle": "dashed",
    }

    fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
    fig.set_size_inches(13, 6)
    fig.tight_layout()

    with ReconstructionPlot2by2(ax=axs[0, 0], title=f"Linear probe", ylabel="Accuracy", xlabel=None) as plot:
        plot.ax.plot(*args_maj_acc, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "acc")
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[0, 1], title=f"MLP probe", ylabel="Accuracy", xlabel=None, legend=True, y_ticks=False) as plot:
        plot.ax.plot(*args_maj_acc, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "acc")
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[1, 0], ylabel="R²") as plot:
        # plot.ax.plot(*args_reg_baseline_linear, **kwargs_baseline)
        df_linear = _reorganize_df(df[df.model_type == "linear"], "r2")
        sns.lineplot(data=df_linear, ax=plot.ax, **standard_kwargs_sns_lineplot)

    with ReconstructionPlot2by2(ax=axs[1, 1], ylabel="R²", legend=True, y_ticks=False) as plot:
        # plot.ax.plot(*args_reg_baseline_mlp, **kwargs_baseline)
        df_mlp = _reorganize_df(df[df.model_type == "mlp"], "r2")
        sns.lineplot(data=df_mlp, ax=plot.ax, **standard_kwargs_sns_lineplot)

    fig.suptitle(f"{str(task).upper().replace('_', ' ')}", y=0.93, x=0.925, fontweight="bold", fontsize=18)
    fig.savefig(f"./plots/{task}_reconstruction_both.png", dpi=300)
    plt.clf()


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
    sns.heatmap(data=pivot_df, annot=False, cmap="crest", xticklabels=True)
    ax.yaxis.set_label_position("right")
    plt.ylabel("NDCG@10")
    plt.yticks(rotation=0)
    # plt.xticks(LAYERS)
    plt.xlabel("Layer")
    plt.subplots_adjust(bottom=0.1, left=0.15)
    fig.savefig(f"./plots/{task}_behaviour_heatmap.png", dpi=300)


def plot_all_behavior_heatmap(df, metric: str):
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(10, 4.5)
    fig.tight_layout()
    df["task"] = df["task"].astype("category")
    df["task"] = df["task"].cat.rename_categories(
        {
            "bm25": "BM25",
            "sem": "SEM",
            # "avg_ti": "AVG TI",
            "ti": "TI",
            "ner": "NER",
            "coref": "COREF",
            "qc_coarse": "QC",
            # "control (1)": "control (1)",
            # "control (4)": "control (4)",
            # "control (8)": "control (8)",
            "baseline": "baseline",
        }
    )
    metrics = {"ndcg_10": "NDCG@10", "map": "MAP", "p_10": "P@10", "mrr": "MRR", "recall_1000": "Recall@1000"}
    # df["task"].cat.set_categories(
    #     ["BM25", "SEM", "AVG TI", "TI", "NER", "COREF", "QC", "control (1)", "control (4)", "control (8)", "baseline"], inplace=True
    # )
    df["task"] = df["task"].cat.set_categories(["BM25", "SEM", "TI", "NER", "COREF", "QC", "baseline"])
    pivot_df = df.pivot(index="task", columns="layer", values=metric)
    pivot_df.sort_index(level=0, inplace=True)
    sns.heatmap(data=pivot_df, annot=False, cmap="crest", xticklabels=True)
    ax.yaxis.set_label_position("right")
    plt.ylabel(metrics[metric])
    plt.yticks(rotation=0)
    # plt.xticks(LAYERS)
    plt.xlabel("Layer")

    plt.subplots_adjust(bottom=0.11, left=0.12, right=1.05)
    fig.savefig(f"./plots/all_behaviour_heatmap_{metric}.png", dpi=300)


def plot_single_behavior_bar(df, task: ProbingTask, metric: str):
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(8, 3.5)
    fig.tight_layout()
    control_dict = {
        ProbingTask.BM25: 1,
        ProbingTask.SEM: 1,
        ProbingTask.AVG_TI: 1,
        ProbingTask.TI: 1,
        ProbingTask.COREF: 1,
        ProbingTask.NER: 8,
        ProbingTask.QC_COARSE: 4,
    }
    names = {
        "bm25": "BM25",
        "sem": "SEM",
        "avg_ti": "AVG TI",
        "ti": "TI",
        "ner": "NER",
        "coref": "COREF",
        "qc_coarse": "QC",
    }
    df = df.replace(names)
    metrics = {"ndcg_10": "NDCG@10", "map": "MAP", "p_10": "P@10", "mrr": "MRR", "recall_1000": "Recall@1000"}
    kwargs_baseline = {
        "label": "baseline",
        "color": "grey",
        "linestyle": "dashed",
    }
    sns_plot = sns.barplot(
        data=df[(df["task"] == names[task.value]) | (df["task"] == f"control ({control_dict[task]})")],
        x="layer",
        y=metric,
        hue="task",
        palette="colorblind",
    )
    ax.axhline(y=df[df["task"] == "baseline"][metric].values[0], **kwargs_baseline)
    hatches = itertools.cycle(["//", "-", "x", "\\", "*", "o", "O", "."])
    for i, bar in enumerate(sns_plot.patches):
        if i % 13 == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)
    plt.ylabel(metrics[metric])
    plt.yticks(rotation=0)
    plt.xlabel("Layer")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(bottom=0.14, left=0.075, right=0.8)
    fig.savefig(f"./plots/{task}_behaviour_barplot_{metric}.png", dpi=300)


# def plot_behavior(df, baseline: float, task: ProbingTask):
#     standard_kwargs_sns = {
#         "markers": MARKER_STYLES,
#         "markersize": 10,
#         "style": "exp",
#         "dashes": False,
#         "palette": "colorblind",
#     }
#     standard_kwargs_sns_scatter = {
#         "markers": MARKER_STYLES,
#         "s": 100,
#         "style": "exp",
#         "alpha": 0.8,
#         "palette": "colorblind",
#     }
#     kwargs_baseline = {
#         "label": "no intervention",
#         "color": "grey",
#         "linestyle": "dashed",
#     }
#     fig = plt.gcf()
#     fig.set_size_inches(8, 5)
#     fig.tight_layout()
#     plt.xticks(LAYERS)
#     plt.ylabel("NDCG@10")
#     plt.xlabel("Layer")
#     plt.grid()
#     plt.ylim(0.2, 0.75)
#     # plt.subplots_adjust(bottom=0.17)

#     # sns.lineplot(data=df, x="layer", y="ndcg_10", hue="exp", **standard_kwargs_sns)
#     sns.scatterplot(data=df, x="layer", y="ndcg_10", hue="exp", **standard_kwargs_sns_scatter)
#     # sns.barplot(data=df, x="layer", y="ndcg_10", hue="exp", width=0.7)
#     plt.plot(LAYERS, [baseline for _ in range(AMOUNT_LAYERS)], **kwargs_baseline)
#     plt.legend(title="experiment", loc="lower left")
#     fig.savefig(f"./plots/{task}_behaviour_scatterplot.png", dpi=300)


def plot_subspace(df, maj_acc, task: ProbingTask):
    fig, axs = plt.subplots(1, 2, sharex="row", sharey="row")
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.5])
    fig.set_size_inches(10, 3.5)

    df["frac"] = df["score"] / df["orig_acc"]
    v_min = df.frac.min()
    v_max = df.frac.max()

    df_probed = df[df.exp == "probed"]
    df_control = df[df.exp == "control"]

    pivot_df_probed_score = df_probed.pivot(index="rank", columns="layer", values="score")
    pivot_df_probed_frac = df_probed.pivot(index="rank", columns="layer", values="frac")
    pivot_df_control_score = df_control.pivot(index="rank", columns="layer", values="score")
    pivot_df_control_frac = df_control.pivot(index="rank", columns="layer", values="frac")

    pivot_df_probed_score.sort_index(level=0, ascending=False, inplace=True)
    pivot_df_probed_frac.sort_index(level=0, ascending=False, inplace=True)
    pivot_df_control_score.sort_index(level=0, ascending=False, inplace=True)
    pivot_df_control_frac.sort_index(level=0, ascending=False, inplace=True)

    with SubspacePlot1by2(ax=axs[0], title="probed") as plot:
        sns.heatmap(
            data=pivot_df_probed_frac,
            ax=plot.ax,
            vmin=v_min,
            vmax=v_max,
            cmap="crest",
            xticklabels=True,
            yticklabels=True,
            cbar=False,
            norm="linear",
        )
        zm = np.ma.masked_greater_equal(pivot_df_probed_score.values, maj_acc)
        x = np.arange(len(pivot_df_probed_score.columns) + 1)
        y = np.arange(len(pivot_df_probed_score.index) + 1)
        plot.ax.pcolor(x, y, zm, hatch="//", alpha=0.0)
    with SubspacePlot1by2(ax=axs[1], title="control", y_ticks=False, y_label=None) as plot:
        sns.heatmap(
            data=pivot_df_control_frac,
            ax=plot.ax,
            vmin=v_min,
            vmax=v_max,
            cmap="crest",
            xticklabels=True,
            yticklabels=True,
            cbar=True,
            cbar_ax=cbar_ax,
            norm="linear",
        )
        zm = np.ma.masked_greater_equal(pivot_df_control_score.values, maj_acc)
        x = np.arange(len(pivot_df_control_score.columns) + 1)
        y = np.arange(len(pivot_df_control_score.index) + 1)
        plot.ax.pcolor(x, y, zm, hatch="//", alpha=0.0)
    plt.ylabel("Fraction from original accuracy")
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(bottom=0.15, right=0.9)
    fig.savefig(f"./plots/{task}_subspace_heatmap.png", dpi=300)


def prepare_behavior_plot(task: ProbingTask, viz_type: str):
    df_dict = defaultdict(list)
    for file in Path("./logs/results/behavior/").rglob(f"trec_eval*{task}*layer*.tsv"):
        with open(file, "r") as f:
            for line in f.readlines():
                if line[0] == "M":
                    continue
                l = line[:-1].split("\t")
                ndcg_10 = float(l[1])
                layer = [int(x) for x in file.name.split("_")[:10] if x.isdigit()][0]
                exp = "intervention\non document"
                if "control" in file.name:
                    exp = "control"
                elif "altered_query_embeddings" in file.name:
                    exp = "intervention\non query\n& document"

                if "qc" in file.name.split("_")[:10]:
                    exp = "intervention\non query"
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


def prepare_all_behavior_plot(full_baseline: bool = False):
    df_dict = defaultdict(list)
    # tasks = [ProbingTask.BM25, ProbingTask.SEM, ProbingTask.TI, ProbingTask.AVG_TI, ProbingTask.NER, ProbingTask.COREF, ProbingTask.QC_COARSE]
    tasks = [ProbingTask.BM25, ProbingTask.SEM, ProbingTask.TI, ProbingTask.NER, ProbingTask.COREF, ProbingTask.QC_COARSE]
    for task in tasks:
        for file in Path(f"./logs/results/behavior/{task}/").rglob(f"trec_eval*{task}_layer*None_altered_query*.tsv"):
            # skip control for now, can later run control again for 1, 4, and 8 rank subspace
            if "control" in file.name:
                continue
            layer = [int(x) for x in file.name.split("_")[:10] if x.isdigit()][0]
            with open(file, "r") as f:
                for line in f.readlines():
                    if line[0] == "M":
                        continue
                    l = line[:-1].split("\t")
                    df_dict["layer"].append(layer)
                    df_dict["ndcg_10"].append(float(l[1]))
                    df_dict["map"].append(float(l[0]))
                    df_dict["mrr"].append(float(l[5]))
                    df_dict["p_10"].append(float(l[3]))
                    df_dict["recall_1000"].append(float(l[6]))
                    # if "control" in file.name:
                    # df_dict["task"].append("control")
                    # else:
                    df_dict["task"].append(task.value)

    # for layer in LAYERS:
    #     for file in Path(f"./logs/results/behavior/control/").rglob(f"trec_eval_control_{layer}_*.tsv"):
    #         control_subspace = file.name.split("_")[7]
    #         with open(file, "r") as f:
    #             for line in f.readlines():
    #                 if line[0] == "M":
    #                     continue
    #                 l = line[:-1].split("\t")
    #                 df_dict["layer"].append(layer)
    #                 df_dict["ndcg_10"].append(float(l[1]))
    #                 df_dict["map"].append(float(l[0]))
    #                 df_dict["mrr"].append(float(l[5]))
    #                 df_dict["p_10"].append(float(l[3]))
    #                 df_dict["recall_1000"].append(float(l[6]))
    #                 df_dict["task"].append(f"control ({control_subspace})")

    with open("./logs/reproduction/results/trec_eval_2023_02_23-09-48-36.tsv", "r") as f:  # file path to baseline values
        for line in f.readlines():
            if line[0] == "M":
                continue
            l = line[:-1].split("\t")
            for i in LAYERS:
                df_dict["layer"].append(i)
                df_dict["task"].append("baseline")
                if i == LAST_LAYER_IDX or full_baseline:
                    df_dict["ndcg_10"].append(float(l[1]))
                    df_dict["map"].append(float(l[0]))
                    df_dict["mrr"].append(float(l[5]))
                    df_dict["p_10"].append(float(l[3]))
                    df_dict["recall_1000"].append(float(l[6]))
                else:
                    df_dict["ndcg_10"].append(np.nan)
                    df_dict["map"].append(np.nan)
                    df_dict["mrr"].append(np.nan)
                    df_dict["p_10"].append(np.nan)
                    df_dict["recall_1000"].append(np.nan)

    return pd.DataFrame.from_dict(df_dict)


def prepare_reconstruction_plot(task: ProbingTask) -> Tuple[pd.DataFrame, float]:
    df_dict = defaultdict(list)
    for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*reconstruction_both.log"):
        try:
            layer = [int(x) for x in file.name.split("_") if x.isdigit()][0]
            merging = "placeholder"
        except:
            raise Exception(f"Could not parse layer from {file.name}")

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

    # model_types = ["linear", "mlp"]
    # reg_baselines = {}
    # clf_baselines = {}

    # for m in model_types:
    #     with open(f"./logs/results/ablation/{task}/{task}_regressor_{m}_baseline.log", "r") as f:
    #         reg_baselines[m] = float(f.read())
    #     with open(f"./logs/results/ablation/{task}/{task}_classification_{m}_baseline.log", "r") as f:
    #         clf_baselines[m] = float(f.read())

    maj_acc_file = [file for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*majority_accuracy.log")][0]
    with open(maj_acc_file, "r") as f:
        maj_acc = float(f.read())

    return df, maj_acc


def prepare_reconstruction_plot_2(task: ProbingTask, normalized_target: bool) -> Tuple[pd.DataFrame, float]:
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

    maj_acc_file = [file for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*majority_accuracy.log")][0]
    with open(maj_acc_file, "r") as f:
        maj_acc = float(f.read())

    # model_types = ["linear", "mlp"]
    # baselines = {}
    # normlize_str = "_normalized_target" if normalized_target else ""
    # for m in model_types:
    #     with open(
    #         f"./logs/results/ablation/{task}/{task}_{'regressor' if reg else 'classification'}_{m}{normlize_str}_baseline.log", "r"
    #     ) as f:
    #         baselines[m] = float(f.read())

    return df, maj_acc


def prepare_subspace_plot(task: ProbingTask) -> pd.DataFrame:
    layer_baseline = defaultdict(list)
    reconstruction_files = [file for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*reconstruction*.log")]
    for file in reconstruction_files:
        s = file.name.split("_")
        layer = [int(x) for x in s if x.isdigit()][0]

        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                line_content = line[:-1].split("\t")
                if line_content[-1] == "linear":
                    layer_baseline[layer].append(float(line_content[0]))
    layer_baseline = {k: np.mean(v) for k, v in layer_baseline.items()}

    df_dict = defaultdict(list)
    files = [file for file in Path(f"./logs/results/ablation/subspace/").rglob(f"{task}*subspace*.log")]

    for file in files:
        s = file.name.split("_")
        layer = [int(x) for x in s if x.isdigit()][0]

        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                line_content = line.split("\t")
                exp = line_content[-2]
                values = [float(x) for x in line_content[0:10]]
                ranks = list(np.logspace(0, 2.8, num=10, dtype=int))
                for value, rank in zip(values, ranks):
                    df_dict["score"].append(value)
                    df_dict["layer"].append(layer)
                    df_dict["exp"].append(exp)
                    df_dict["rank"].append(rank)
                    df_dict["orig_acc"].append(layer_baseline[layer])

    maj_acc_file = [file for file in Path(f"./logs/results/ablation/{task}/").rglob(f"{task}*majority_accuracy.log")][0]
    with open(maj_acc_file, "r") as f:
        maj_acc = float(f.read())

    df1 = pd.DataFrame.from_dict(df_dict)

    return df1, maj_acc


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reconstruction", dest="reconstruction", action="store_true")
    parser.add_argument("-p", "--bucketized_and_reg", dest="bucketized_and_reg", action="store_true")
    parser.add_argument("-n", dest="normalized_target", action="store_true")
    parser.add_argument("-b", "--behavior", dest="behavior", action="store_true")
    parser.add_argument("-a", "--all_behavior", dest="all_behavior", action="store_true")
    parser.add_argument("-s", "--subspace", dest="subspace", action="store_true")
    parser.add_argument(
        "-m",
        "--metric",
        dest="metric",
        type=str,
        default="ndcg_10",
        choices=["ndcg_10", "map", "mrr", "p_10", "recall_1000"],
        help="metric to plot in behavior heatmap",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=ProbingTask,
        dest="task",
        default=ProbingTask.BM25,
        choices=list(ProbingTask),
        help="Task to plot.",
    )
    parser.add_argument("-v", "--viz_type", dest="viz_type", default="heatmap", choices=["lineplot", "heatmap"])

    args = parser.parse_args()

    return args


def main(
    reconstruction: bool,
    behavior: bool,
    subspace: bool,
    task: ProbingTask,
    viz_type: str,
    normalized_target: bool,
    bucketized_and_reg: bool,
    all_behavior: bool,
    metric: str,
):
    if reconstruction:
        if bucketized_and_reg:
            df, maj_acc = prepare_reconstruction_plot(task)
            plot_reconstruction(df, task, maj_acc)
        else:
            df, maj_acc = prepare_reconstruction_plot_2(task, normalized_target)
            plot_reconstruction_2(df, task, maj_acc)
    if behavior:
        if all_behavior:
            df = prepare_all_behavior_plot()
            plot_all_behavior_heatmap(df, metric)
        else:
            df = prepare_all_behavior_plot(full_baseline=True)
            # df, baseline = prepare_behavior_plot(task, viz_type)
            plot_single_behavior_bar(df, task, metric)
            # if viz_type == "heatmap":
            #     plot_behavior_heatmap(df, task)

    if subspace:
        df1, maj_acc = prepare_subspace_plot(task)
        plot_subspace(df1, maj_acc, task)


if __name__ == "__main__":
    args = parse_arguments()
    args = vars(args)
    main(**args)
