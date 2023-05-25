import argparse

from src.probing_config import ProbeModelType, ProbingTask


def parse_arguments_intervention():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--device_cpu", dest="device_cpu", action="store_true")
    parser.add_argument("--alter_query_embedding", dest="alter_query_embedding", action="store_true")
    parser.add_argument("--reconstruction_both", dest="reconstruction_both", action="store_true")
    parser.add_argument("--control_only", dest="control_only", action="store_true")
    parser.add_argument("--multiple_runs", dest="multiple_runs", action="store_true")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_choice",
        default="tct_colbert",
        help="Model to run.",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        dest="layer",
        default=None,
        help="On which layer to intervene",
    )
    parser.add_argument(
        "-r",
        "--eliminated_subspace_rank",
        type=int,
        dest="eliminated_subspace_rank",
        default=1,
        help="rank of the eliminated subspace by rlace",
    )
    parser.add_argument(
        "-a",
        "--ablation",
        type=str,
        dest="ablation",
        default=None,
        help="Whether to apply an ablation. Choices: token_wise, reconstruct_property, control",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=ProbingTask,
        dest="probing_task",
        default=ProbingTask.BM25,
        choices=list(ProbingTask),
        help="Probing task to perform. In other words: Which property to remove.",
    )
    parser.add_argument(
        "--probe_model_type",
        type=ProbeModelType,
        dest="probe_model_type",
        default=ProbeModelType.LINEAR,
        choices=list(ProbeModelType),
        help="Which type of probe model to use.",
    )

    args = parser.parse_args()

    return args


def parse_arguments_reproducer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--device_cpu", dest="device_cpu", action="store_true")
    parser.add_argument("--reindex", dest="reindex", action="store_true")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_choice",
        default="tct_colbert",
        help="Model to run.",
    )

    args = parser.parse_args()

    return args
