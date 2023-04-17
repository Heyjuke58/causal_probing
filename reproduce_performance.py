import sys

sys.path.insert(1, "./ranking-utils")
import logging
import os
from typing import Optional, Union

import faiss
import numpy as np
import pandas as pd
import torch
from ranking_utils import write_trec_eval_file

from src.argument_parser import parse_arguments_reproducer
from src.file_locations import *
from src.hyperparameter import BATCH_SIZE_LM_MODEL, EMBEDDING_SIZE, MODEL_CHOICES
from src.model import ModelWrapper
from src.trec_evaluation import trec_evaluation
from src.utils import get_batch_amount, get_corpus, get_device, get_queries, get_timestamp


class Reproducer:
    def __init__(
        self,
        model_choice: str = "tct_colbert",
        device_cpu: bool = False,
        debug: bool = False,
        reindex: bool = False,
    ) -> None:
        self.device = get_device(device_cpu)

        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_choice = model_choice
        self.debug = debug
        self.reindex = reindex
        self.model_wrapper = ModelWrapper(self.model_choice, self.device)
        self.corpus: pd.DataFrame

        if debug:
            global MSMARCO_CORPUS_PATH
            global MSMARCO_TREC_2019_TEST_QUERIES_PATH
            MSMARCO_CORPUS_PATH = MSMARCO_TOY_CORPUS_PATH
            # MSMARCO_TREC_2019_TEST_QUERIES_PATH = MSMARCO_TOY_QUERIES_PATH

    def run(self):
        self.timestamp = get_timestamp()
        if self.reindex:
            self._init_corpus()
            index = self._make_index()
        else:
            index = self._reload_index_from_file()
        self._evaluate(index)

    def _make_index(self, cache_index: bool = True):
        logging.info(f"Making index.")
        self._init_corpus()
        corpus_size = len(self.corpus)
        index = faiss.IndexIDMap2(faiss.index_factory(EMBEDDING_SIZE, "Flat", faiss.METRIC_INNER_PRODUCT))
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        passages = self.corpus["passage"].tolist()
        pids = self.corpus["pid"].to_numpy()

        for i in range(batches):
            start = BATCH_SIZE_LM_MODEL * i
            end = min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)
            embs = self.model_wrapper.get_passage_embeddings_pyserini(passages[start:end])
            index.add_with_ids(embs, pids[start:end])  # type: ignore
        logging.info(f"Index made.")
        if cache_index:
            logging.info(f"Saving to file...")
            faiss.write_index(index, "./cache/reproduction/faiss_index.bin")
            logging.info(f"Index saved to file.")

        return index

    def _reload_index_from_file(self):
        logging.info(f"Loading Index from file...")
        index = faiss.read_index("./cache/reproduction/faiss_index.bin")
        logging.info(f"Index load from file.")
        return index

    def _evaluate(
        self,
        faiss_index: faiss.Index,
        recall_at: int = 1000,
    ):
        logging.info(f"Evaluating performance of reproduction.")
        queries = get_queries(MSMARCO_TREC_2019_TEST_QUERIES_PATH)
        predictions: dict[str, dict[str, float]] = {}  # Query IDs mapped to document IDs mapped to scores.

        for i, row in queries.iterrows():
            qid = row[0]
            query = row[1]

            q_emb_np = self.model_wrapper.get_query_embedding_pyserini(query)
            q_emb_np = q_emb_np.reshape(1, q_emb_np.shape[0])

            scores, ids = faiss_index.search(q_emb_np, recall_at)

            docs_dict = {id: score for score, id in zip(scores[0].tolist(), ids[0].tolist())}
            predictions[str(qid)] = docs_dict  # type: ignore

        self._trec_eval(predictions)

    def _trec_eval(self, predictions):
        logging.info(f"Starting official TREC evaluation of reproduction.")
        out_file_str = f"./logs/reproduction/results/trec_eval_{self.timestamp}.tsv"
        eval_file_str = f"./logs/reproduction//trec_eval.tsv"
        out_file = Path(out_file_str)
        write_trec_eval_file(Path(eval_file_str), predictions, "reproduction")
        trec_evaluation(out_file, self.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, eval_file_str, 0)
        if Path(eval_file_str).is_file():
            os.remove(eval_file_str)
        logging.info(f"TREC evaluation of reproduction done. Logged results at time {self.timestamp}.")

    def _init_corpus(self):
        if not isinstance(self.corpus, pd.DataFrame):
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)


if __name__ == "__main__":
    args = parse_arguments_reproducer()

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logging_level = logging.DEBUG
    if not args.debug:
        logging_level = logging.INFO
        file_handler = logging.FileHandler(f"./logs/reproduction/console/{args.model_choice}_{get_timestamp()}.log")
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging_level)
        root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging_level)
    root.addHandler(console_handler)
    root.setLevel(logging_level)

    args = vars(args)
    r = Reproducer(**args)
    r.run()
