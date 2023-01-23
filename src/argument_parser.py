import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ro", "--reindex_original", dest="reindex_original", action="store_true")
    parser.add_argument("-rt", "--reindex_task", dest="reindex_task", action="store_true")
    parser.add_argument("--do_not_prepend_token", dest="prepend_token", action="store_false")
    parser.add_argument("-c", "--chunked_read_in", dest="chunked_read_in", action="store_true")
    parser.add_argument("--generate_emb_chunks", dest="generate_emb_chunks", action="store_true")
    parser.add_argument("-es", "--init_elastic_search", dest="init_elastic_search", action="store_true")
    parser.add_argument("--all_layers", dest="all_layers", action="store_true")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--device_cpu", dest="device_cpu", action="store_true")
    parser.add_argument("--ablation_last_layer", dest="ablation_last_layer", action="store_true")
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
    parser.add_argument(
        "-i",
        "--index_str",
        type=str,
        dest="faiss_index_factory_str",
        default="IVF30000,Flat",
        help="Faiss index factory string. See https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index for more details.",
    )
    parser.add_argument(
        "-f",
        "--doc_store_framework",
        type=str,
        dest="doc_store_framework",
        default="haystack",
        help="Framewokr to build document store.",
    )

    args = parser.parse_args()

    # assert sum(list(map(int, args.split.split(',')))) == 100, "Not a valid train/val/test split. Must add up to 100 like 70,15,15."
    # assert sum(list(map(int, args.neg_sample_ratio.split(',')))) == 100, "Not a valid negative sampling ratio of easy and hard examples. Must add up to 100 like 50,50."

    return args
