import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import torch
from ranking_utils import write_trec_eval_file

from src.file_locations import MSMARCO_QREL_2019_PATH, MSMARCO_TREC_2019_TEST_QUERIES_PATH, TREC_EVAL
from src.model import ModelWrapper
from src.probing_config import ProbingTask
from src.trec_evaluation import trec_evaluation
from src.utils import get_queries


def evaluate(
    model_wrapper: ModelWrapper,
    faiss_index: faiss.Index,
    timestamp: str,
    layer: Optional[int] = None,
    probing_task: Optional[ProbingTask] = None,
    eval_str: str = "",
    recall_at: int = 1000,
    projection: Optional[torch.Tensor] = None,
):
    logging.info(f"Evaluating performance{' ' + eval_str}.")
    queries = get_queries(MSMARCO_TREC_2019_TEST_QUERIES_PATH)
    predictions: dict[str, dict[str, float]] = {}  # Query IDs mapped to document IDs mapped to scores.

    # TODO: maybe remove iteration and do it with one faiss search
    for i, row in queries.iterrows():
        qid = row[0]
        query = row[1]
        q_emb_np = model_wrapper.get_query_embeddings_pyserini_with_intervention_at_layer([query], projection, layer)
        scores, ids = faiss_index.search(q_emb_np, recall_at)

        docs_dict = {id: score for score, id in zip(scores[0].tolist(), ids[0].tolist())}
        predictions[str(qid)] = docs_dict  # type: ignore

    logging.info(f"Starting official TREC evaluation of {eval_str}.")
    out_file_str = f"./logs/results/behavior{'/' + str(probing_task) if probing_task else '/control'}/trec_eval_{eval_str}_{timestamp}.tsv"
    eval_file_str = f"./logs/trec/trec_eval.tsv"
    out_file = Path(out_file_str)
    write_trec_eval_file(Path(eval_file_str), predictions, eval_str)
    trec_evaluation(out_file, model_wrapper.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, eval_file_str, 0)
    if Path(eval_file_str).is_file():
        os.remove(eval_file_str)
    logging.info(f"TREC evaluation of {eval_str} @ {timestamp} done. Logged results.")
