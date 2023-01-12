from collections import defaultdict
from pathlib import Path
import pandas as pd
import logging
import ftfy
import time
import math

import pickle
import os

def get_timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H-%M-%S")

def get_batch_amount(size, batch_size) -> int:
    return (
        math.floor(size / batch_size) + 1
        if not (size / batch_size).is_integer()
        else int(size / batch_size)
    )

def fuse_chunks(file_prefix: str, amount: int):
    # fuse chunked files
    logging.info(f"Fuse {file_prefix} chunks...")
    elements = []
    for i in range(amount):
        chunk_str = f"./{file_prefix}_chunk_{i}.pickle"
        try:
            with open(chunk_str, "rb") as docs_chunk:
                elements.extend(pickle.load(docs_chunk))
        except FileNotFoundError:
            raise FileNotFoundError("Error while loading chunked file.")
        try:
            os.remove(chunk_str)
        except FileNotFoundError:
            raise FileNotFoundError("Error while cleaning up chunked files.") 
    with open(f"./{file_prefix}.pickle", "wb+") as fused_file:
        pickle.dump(elements, fused_file)
    logging.info(f"Chunked {file_prefix} file has been fused.")

def get_corpus(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    logging.info("Loading corpus ...")
    corpus_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pid", "passage"],
        encoding="utf8",
        dtype={"pid": "int64", "passage": "string"},
    )
    logging.info("Corpus loaded. Preprocessing it ...")
    # fix unicode errors
    if fix_unicode_errors:
        corpus_df["passage"] = corpus_df["passage"].apply(ftfy.fix_text)
    logging.info("Corpus preprocessed.")

    return corpus_df

def get_queries(path: Path, fix_unicode_errors: bool = True) -> pd.DataFrame:
    logging.info("Loading queries ...")
    queries_df = pd.read_csv(path, sep="\t", header=None, names=["qid", "query"], encoding="utf8")
    # fix unicode errors
    logging.info("Queries loaded. Preprocessing them ...")
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