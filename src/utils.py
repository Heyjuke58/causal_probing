from collections import defaultdict
from pathlib import Path
import pandas as pd
import logging
import ftfy

def get_corpus(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    corpus_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pid", "passage"],
        encoding="utf8",
        dtype={"pid": "int64", "passage": "string"},
    )
    # fix unicode errors
    if fix_unicode_errors:
        corpus_df["passage"] = corpus_df["passage"].apply(ftfy.fix_text)
    logging.info("Corpus preprocessed.")

    return corpus_df

def get_queries(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"], encoding="utf8")
    # fix unicode errors
    if fix_unicode_errors:
        queries_df["query"] = queries_df["query"].apply(ftfy.fix_text)
    logging.info("Queries preprocessed.")

    return queries_df

def get_qrels(path: Path) -> pd.DataFrame:
    qrels_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "x", "pid", "y"], encoding="utf8")
    qrels_df = qrels_df.drop(["x", "y"], axis=1)
    logging.info("Qrels loaded.")

    return qrels_df

def get_top_1000_passages(path: Path) -> dict[int, list[int]]:
    # q_id -> [p_id1, p_id2, .. , p_id1000]
    q_p_top1000_dict: dict[int, list[int]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            q_id, p_id = tuple(line.split(sep="\t")[:2])
            q_p_top1000_dict[int(q_id)].append(int(p_id))
    logging.info("Top 1000 passages per query parsed.")

    return q_p_top1000_dict