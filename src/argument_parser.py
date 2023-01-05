import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ro", "--reindex_original", dest="reindex_original", action="store_true")
    parser.add_argument("-rt", "--reindex_task", dest="reindex_task", action="store_true")
    parser.add_argument("-p", "--prepend_token", dest="prepend_token", action="store_true")
    parser.add_argument("-d", "--device_cpu", dest="device_cpu", action="store_true")
    parser.add_argument("-c", "--chunked_read_in", dest="chunked_read_in", action="store_true")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model_choice",
        default="tct_colbert",
        help="Model to run.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        dest="probing_task",
        default="bm25",
        help="Task to run. Possible tasks are: bm25, to be continued.",
    )

    args = parser.parse_args()

    # assert sum(list(map(int, args.split.split(',')))) == 100, "Not a valid train/val/test split. Must add up to 100 like 70,15,15."
    # assert sum(list(map(int, args.neg_sample_ratio.split(',')))) == 100, "Not a valid negative sampling ratio of easy and hard examples. Must add up to 100 like 50,50."

    return args
