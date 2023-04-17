from pathlib import Path

from src.probing_config import ProbingTask

## DATA
DATASET_PATHS = {
    ProbingTask.BM25: Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json"),
    ProbingTask.BM25_BUCKETIZED: Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json"),
    ProbingTask.SEM: Path("./datasets/msmarco_sem_sim_60000_10_2022_03_21-17-06-14.json"),
    ProbingTask.SEM_BUCKETIZED: Path("./datasets/msmarco_sem_sim_60000_10_2022_03_21-17-06-14.json"),
    ProbingTask.AVG_TI: Path("./datasets/msmarco_avg_term_importance_60000_10_2023_03_15-05-14-29.json"),
    ProbingTask.AVG_TI_BUCKETIZED: Path("./datasets/msmarco_avg_term_importance_60000_10_2023_03_15-05-14-29.json"),
    ProbingTask.TI: Path("./datasets/msmarco_term_importance_60000_10_2023_03_24-10-08-18.json"),
    ProbingTask.TI_BUCKETIZED: Path("./datasets/msmarco_term_importance_60000_10_2023_03_24-10-08-18.json"),
    ProbingTask.COREF: Path("./datasets/msmarco_coref_res_60000_10_2023_04_12-12-24-34.json"),
    ProbingTask.NER: Path("./datasets/msmarco_ner_60000_10_2023_04_13-16-13-46.json"),
}
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
MSMARCO_DEV_QUERIES_PATH = Path("./assets/msmarco/queries.dev.tsv")
MSMARCO_DEV_QRELS_PATH = Path("./assets/msmarco/qrels.dev.tsv")
MSMARCO_DEV_TOP_1000_PATH = Path("./assets/msmarco/top1000.dev")
MSMARCO_TEST_43_TOP_1000_PATH = Path("./assets/msmarco/trec_43_test_top1000.tsv")
MSMARCO_TREC_2019_TEST_QUERIES_PATH = Path("./assets/msmarco/trec_43_test_queries.tsv")
MSMARCO_QREL_2019_PATH = "/home/hinrichs/causal_probing/assets/msmarco/2019-qrels-pass.txt"

## TREC EVAL PATH
TREC_EVAL = "/home/hinrichs/causal_probing/trec_eval/trec_eval"

## TOY DATA
MSMARCO_TOY_CORPUS_PATH = Path("./assets/msmarco/toy_collection.tsv")
MSMARCO_TOY_QUERIES_PATH = Path("./assets/msmarco/toy_query.tsv")
