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