from pathlib import Path

## DATA
DATASET_PATHS = {
    "bm25": Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json"),
    "sem_sim": Path("./datasets/msmarco_semantic_similarity.json"),
    "something_else": Path("./datasets/msmarco_.json"),
    
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