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