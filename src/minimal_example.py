from pathlib import Path
import logging
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import math
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document

from ranking_utils import write_trec_eval_file
from trec_evaluation import trec_evaluation
from utils import get_corpus, get_qrels, get_queries, get_top_1000_passages, fuse_chunks, get_timestamp
from argument_parser import parse_arguments

from src.elasticsearch_bm25 import ElasticSearchBM25
from src.rlace import solve_adv_game

## DATA
DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
# MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection_sample_100.tsv")
MSMARCO_DEV_QUERIES_PATH = Path("./assets/msmarco/queries.dev.tsv")
MSMARCO_DEV_QRELS_PATH = Path("./assets/msmarco/qrels.dev.tsv")
MSMARCO_DEV_TOP_1000_PATH = Path("./assets/msmarco/top1000.dev")
MSMARCO_TEST_43_TOP_1000_PATH = Path("./assets/msmarco/trec_43_test_top1000.tsv")
MSMARCO_TEST_43_QUERIES_PATH = Path("./assets/msmarco/trec_43_test_queries.tsv")
MSMARCO_QREL_2019_PATH = "/home/hinrichs/causal_probing/assets/msmarco/2019-qrels-pass.txt"

## TREC EVAL
TREC_EVAL = Path("/home/hinrichs/causal_probing/trec_eval")

## MODELS
MODEL_CHOICES = {"tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco"}

## HYPERPARAMETER
SEED = 12
BATCH_SIZE_FP = 10
EMBEDDING_SIZE = 768
CHUNK_SIZE = int(1e6)
CHUNK_AMOUNT = math.ceil(int(8.8e6) / CHUNK_SIZE) # 8.8M is the corpus size


class MinimalExample:
    def __init__(
        self, model_choice: str = "tct_colbert", probing_task: str = "bm25", reindex: bool = False
    ) -> None:
        self.TASKS = {"bm25": self._bm25_probing_task}
        self.model_choice_hf_str = MODEL_CHOICES[model_choice]
        self.model_choice = model_choice
        self.probing_task = probing_task
        self.dataset = pd.read_json(DATASET_PATH, orient="records")

        # index params
        self.faiss_index_factory_str = "IVF30000,Flat"
        self.index_name = f"{self.model_choice}_{self.faiss_index_factory_str.replace(',', '_')}"
        self.index_name_probing = f"{self.model_choice}_{self.probing_task}_{self.faiss_index_factory_str.replace(',', '_')}"

        # Evaluation files
        self.trec_eval_file = f"./logs/trec/trec_eval_{self.index_name}.tsv"
        self.trec_eval_file_probing = f"./logs/trec/trec_eval_{self.index_name_probing}.tsv"

        self.train, self.val, self.test = self._train_val_test_split()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_choice_hf_str)
        self.model = AutoModel.from_pretrained(self.model_choice_hf_str).to(self.device)
        # run probing task
        self.projection = self.TASKS[self.probing_task]()  # projection to remove a concept
        # init document store
        self.doc_store, self.es_bm25 = self._init_faiss_doc_store_and_es_bm25(
            reindex=reindex, batch_size=BATCH_SIZE_FP
        )
        # add index for embeddings after projection of probing task is applied
        self.probing_doc_store = self._add_index_to_doc_store(
            self.index_name_probing, self.projection, reindex=reindex
        )

        self._evaluate_performance()

    def _get_X_probing_task(
        self, X_query: np.ndarray, X_passage: np.ndarray, batch_size: int, merging: str
    ):
        merging_options = {"concat", "multiply_elementwise"}
        if merging not in merging_options:
            raise NotImplementedError(f"Merging option must be one of {merging_options}")
        X_file_str = f"./cache/X_{self.probing_task}_{merging}.pt"
        if not Path(X_file_str).is_file():
            logging.info(
                f"Getting embeddings of query-document pairs of probing task {self.probing_task} dataset ..."
            )
            q_embs = torch.empty((len(X_query), EMBEDDING_SIZE))
            p_embs = torch.empty((len(X_passage), EMBEDDING_SIZE))

            batches = (
                math.floor(len(X_query) / batch_size) + 1
                if not (len(X_query) / batch_size).is_integer()
                else int(len(X_query) / batch_size)
            )

            for i in tqdm(range(batches)):
                X_query_tokenized = self.tokenizer(
                    X_query.tolist()[batch_size * i : min(batch_size * (i + 1), len(X_query))],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                X_passage_tokenized = self.tokenizer(
                    X_passage.tolist()[batch_size * i : min(batch_size * (i + 1), len(X_query))],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                q_emb = self.model(**X_query_tokenized).pooler_output.detach().cpu()
                p_emb = self.model(**X_passage_tokenized).pooler_output.detach().cpu()
                q_embs[batch_size * i : min(batch_size * (i + 1), len(X_query)), :] = q_emb
                p_embs[batch_size * i : min(batch_size * (i + 1), len(X_query)), :] = p_emb

            # What needs to be done here?
            if merging == "concat":
                # concat embeddings to get a tensor of len(query) x EMBEDDING_SIZE * 2
                X = torch.cat((q_embs, p_embs), 1)
            elif merging == "multiply_elementwise":
                # multiply embeddings elementwise
                X = q_embs * p_embs
            else:
                X = torch.tensor([])
            torch.save(X, X_file_str)
            logging.info(
                f"All query-document pairs processed. Embeddings saved to file {X_file_str}."
            )
        else:
            X = torch.load(X_file_str)
            logging.info(f"Saved embeddings found locally. Restored embeddings from {X_file_str}.")
        return X

    def _train_val_test_split(self, ratio_train: float = 0.6, ratio_val_and_test: float = 0.2):
        train, val, test = np.split(
            self.dataset.sample(frac=1, random_state=SEED),
            [
                int(ratio_train * len(self.dataset)),
                int((ratio_train + ratio_val_and_test) * len(self.dataset)),
            ],
        )
        return train, val, test

    @staticmethod
    def rlace_linear_regression_closed_form(X: torch.Tensor, y: torch.Tensor):
        logging.info("Applying RLACE linear regression (closed form).")
        return torch.eye(X.shape[1], X.shape[1]) - (
            (X.t().mm(y)).mm(y.t().mm(X)) / ((y.t().mm(X)).mm(X.t().mm(y)))
        )

    def _bm25_probing_task(self):
        X_query = np.array([x["query"] for x in self.train["input"].values]) # type: ignore
        X_passage = np.array([x["passage"] for x in self.train["input"].values]) # type: ignore
        X = self._get_X_probing_task(X_query, X_passage, BATCH_SIZE_FP, "multiply_elementwise")
        y = torch.from_numpy(self.train["targets"].apply(lambda x: x[0]["label"]).to_numpy()) # type: ignore
        y = y.to(torch.float32).unsqueeze(1)

        # P can be calculated with a closed form, since we are dealing with a linear regression
        # (when we try to linearly predict the BM25 score from the models representation)
        P = self.rlace_linear_regression_closed_form(X, y)
        return P

    def _init_faiss_doc_store_and_es_bm25(self, reindex=False, batch_size: int = 20):
        if reindex:
            corpus_df = get_corpus(MSMARCO_CORPUS_PATH)
            es_bm25 = self._init_elasticsearch_bm25_index(corpus_df["passage"].to_dict())
            doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name}.db",
                faiss_index_factory_str=self.faiss_index_factory_str,
                index=self.index_name,
                duplicate_documents="skip",
            )
            passages = corpus_df["passage"].tolist()
            pids = corpus_df["pid"].tolist()
            docs = []

            batches = (
                math.floor(len(corpus_df) / batch_size) + 1
                if not (len(corpus_df) / batch_size).is_integer()
                else int(len(corpus_df) / batch_size)
            )
            chunk_counter = 0

            for i in tqdm(range(batches)):
                X_passage_tokenized = self.tokenizer(
                    passages[batch_size * i : min(batch_size * (i + 1), len(corpus_df))],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                embs = (
                    self.model(**X_passage_tokenized).pooler_output.detach().cpu()
                )  # maybe prepend '[D]'?
                del X_passage_tokenized  # free memory
                for j in range(embs.shape[0]):
                    emb = embs[j, :]
                    docs.append(
                        Document(
                            content=passages[i * batch_size + j],
                            embedding=emb,
                            meta={"pid": pids[i * batch_size + j]},
                        )
                    )
                del embs  # free memory
                if len(docs) > CHUNK_SIZE:
                    doc_store.write_documents(
                        documents=docs, duplicate_documents="skip", index=self.index_name
                    )
                    chunk = torch.zeros((min(CHUNK_SIZE, corpus_df.shape[0] - chunk_counter * CHUNK_SIZE), 768))
                    chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{chunk_counter}.pt"
                    for i, doc in enumerate(docs):
                        emb = torch.tensor(doc.embedding).unsqueeze(0)
                        chunk[i % CHUNK_SIZE] = emb
                    torch.save(chunk, chunk_str)
                    del chunk # free memory
                    chunk_counter += 1
                    docs = []
            logging.info("Vanilla document embeddings added to document store.")
            # save faiss index and embeddings
            with open(f"./cache/{self.index_name}.pickle", "wb+") as faiss_index_file:
                pickle.dump(doc_store.faiss_indexes[self.index_name], faiss_index_file)
            logging.info("Saved Faiss index file.")

            # fusing chunks not needed anymore, since embeddings are read chunked
            # fuse_chunks(f"./cache/{self.index_name}_embs", chunk_counter)

        else:
            # es_bm25 = self._init_elasticsearch_bm25_index(
            #     {}
            # )  # init ES with empty pool, since it has already been indexed.
            es_bm25 = None
            logging.info("Restoring document store with Faiss index...")
            doc_store = None
            with open(f"./cache/{self.index_name}.pickle", "rb") as faiss_index_file:
                faiss_index = pickle.load(faiss_index_file)
                doc_store = FAISSDocumentStore(
                    sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name}.db",
                    faiss_index_factory_str=self.faiss_index_factory_str,
                    index=self.index_name,
                    duplicate_documents="skip",
                    faiss_index=faiss_index,
                )
            logging.info("Document store with Faiss index restored.")

        return doc_store, es_bm25

    def _add_index_to_doc_store(self, index_name_probing, projection, reindex=False):
        faiss_index_file_str = f"./cache/{self.index_name_probing}.pickle"
        if reindex:
            # TODO: checken ob alle embeddings in RAM passen, sonst -> chunked einlesen -> chunk fuse obsolet
            logging.info(
                f"Adding new index {self.probing_task} of altered embeddings (RLACE/INLP) to second document store. "
                f"Loading altered embeddings from local chunks ..."
            )
            doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name_probing}.db",
                faiss_index_factory_str=self.faiss_index_factory_str,
                index=index_name_probing,
                duplicate_documents="skip",
            )

            for i in range(CHUNK_AMOUNT):
                logging.info(f"Processing chunk {i + 1}/{CHUNK_AMOUNT}")
                chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{i}.pt"
                emb_docs_chunk = torch.load(chunk_str)
                emb_docs_chunk_altered = torch.einsum('bc,cd->bd', emb_docs_chunk, projection)
                docs = [Document(id=str(i * CHUNK_SIZE + j), content="", embedding=emb_docs_chunk_altered[j,:].squeeze(0).numpy()) for j in range(emb_docs_chunk_altered.shape[0])]
                doc_store.write_documents(
                    documents=docs, duplicate_documents="skip", index=self.index_name_probing
                )
            # with open(f"./cache/{self.index_name}_embs.pickle", "rb") as docs_file:
            #     docs: list[Document] = pickle.load(docs_file)
            #     # TODO: check time of this loop maybe multiprocess it
            #     for doc in tqdm(docs, desc="Altering document embeddings"):
            #         emb = torch.tensor(doc.embedding).unsqueeze(0)
            #         doc.embedding = emb.mm(projection).squeeze(0).numpy()
            #     doc_store.write_documents(
            #         documents=docs, duplicate_documents="skip", index=self.index_name_probing
            #     )
            #     del docs # free memory
            # save faiss index
            with open(faiss_index_file_str, "wb+") as faiss_index_file:
                pickle.dump(doc_store.faiss_indexes[self.index_name_probing], faiss_index_file)

        else:
            logging.info(f"Restoring document store with Faiss index of altered embeddings (task: {self.probing_task}) ...")
            if Path(faiss_index_file_str).is_file():
                with open(faiss_index_file_str, "rb") as faiss_index_file:
                    faiss_index = pickle.load(faiss_index_file)
                    doc_store = FAISSDocumentStore(
                        sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name_probing}.db",
                        faiss_index_factory_str=self.faiss_index_factory_str,
                        index=index_name_probing,
                        duplicate_documents="skip",
                        faiss_index=faiss_index,
                    )
            else:
                raise FileNotFoundError(
                    f"Cannot load index {self.index_name_probing} from file. You need to reindex."
                )
        logging.info(
            f"Initialized second docuemnt store with altered embeddings of task {self.probing_task}. "
            f"Saved Faiss index file to {faiss_index_file_str}."
        )
        return doc_store

    def _init_elasticsearch_bm25_index(self, pool):
        bm25 = ElasticSearchBM25(
            pool,
            index_name="msmarco3",  # TODO: if more than msmarco should be used this has to be edited
            service_type="docker",
            max_waiting=100,
            port_http="12375",
            port_tcp="12376",
            es_version="7.16.2",
            reindexing=False,
        )

        return bm25

    def _evaluate_performance(self, mrr_at: int = 10, recall_at: int = 1000):
        logging.info("Evaluating performance ...")
        queries = get_queries(MSMARCO_TEST_43_QUERIES_PATH)
        top1000 = get_top_1000_passages(MSMARCO_TEST_43_TOP_1000_PATH)

        # qrels = get_qrels(MSMARCO_DEV_QRELS_PATH)
        reciprocal_ranks, recalls = [], []
        reciprocal_ranks_probing, recalls_probing = [], []
        predictions: dict[str, dict[str, float]] = {} # Query IDs mapped to document IDs mapped to scores.
        predictions_probing: dict[str, dict[str, float]] = {} # Query IDs mapped to document IDs mapped to scores.

        def reciprocal_rank(rel_docs: list[Document], relevant_pid: int):
            for j, doc in enumerate(rel_docs):
                if (
                    doc.meta["vector_id"] == relevant_pid
                ):  # vectore_id is a counter which corresponds to pid,
                    # however it would be better to have the meta data given the documents
                    return (j + 1) / len(relevant_docs)
            return 0

        def recall(relevant_docs: list[Document], relevant_pids: list[int]):
            # assert len(relevant_docs) == len(
            #     relevant_pids
            # ), "Number of relevant docs returned by the doc store and ground truth docs does not match"
            rel_docs_counter = 0
            for i, doc in enumerate(relevant_docs):
                # return early if there are not 1000 associated relevant pids
                if (i >= len(relevant_pids)):
                    return rel_docs_counter / len(relevant_pids)
                if (
                    doc.meta["vector_id"] in relevant_pids
                ):  # vectore_id is a counter which corresponds to pid,
                    # however it would be better to have the meta data given the documents
                    rel_docs_counter += 1
            return rel_docs_counter / len(relevant_docs)

        # get query embedding
        for i, row in tqdm(queries.iterrows(), desc="Evaluating"):
            qid = row[0]
            query = row[1]
            try:
                ## GETTING GROUND TRUTH BUT HOW?

                # 1st
                # get top1000 docs with ES BM25
                # documents_ranked = self.es_bm25.query(row[1][1], topk=recall_at)
                # top1_relevant_pid = int(list(documents_ranked.keys())[0])
                # top1000_relevant_pids = [int(x) for x in list(documents_ranked.keys())]

                # 2nd
                # get top 1 with qrels
                # top1_relevant_pid = qrels[qrels["qid"] == row[1][0]]["pid"].values[0]

                # 3rd
                # get top1000 docs with top1000 (not complete!)
                top1_relevant_pid = top1000[qid][0]
                top1000_relevant_pids = top1000[qid]

                pass
            except:
                continue
            X_query_tokenized = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            # maybe prepend '[Q]'?
            q_emb = self.model(**X_query_tokenized).pooler_output.detach().cpu()
            q_emb_probing = q_emb.mm(self.projection)
            q_emb = q_emb.squeeze(0).numpy()
            q_emb_probing = q_emb_probing.squeeze(0).numpy()
            relevant_docs = self.doc_store.query_by_embedding(
                q_emb, index=self.index_name, top_k=recall_at
            )
            relevant_docs_probing = self.probing_doc_store.query_by_embedding(
                q_emb_probing, index=self.index_name_probing, top_k=recall_at
            )
            docs_dict = {doc.id: doc.score for doc in relevant_docs}
            predictions[str(qid)] = docs_dict # type: ignore
            docs_dict = {doc.id: doc.score for doc in relevant_docs_probing}
            predictions_probing[str(qid)] = docs_dict # type: ignore

            # calculate MRR@10
            reciprocal_ranks.append(reciprocal_rank(relevant_docs[:mrr_at], top1_relevant_pid))
            reciprocal_ranks_probing.append(
                reciprocal_rank(relevant_docs_probing[:mrr_at], top1_relevant_pid)
            )

            # calculate Recall@1000
            recalls.append(recall(relevant_docs, top1000_relevant_pids))
            recalls_probing.append(recall(relevant_docs_probing, top1000_relevant_pids))

        mrr = np.mean(reciprocal_ranks)
        mrr_probing = np.mean(reciprocal_ranks_probing)
        r = np.mean(recalls)
        r_probing = np.mean(recalls_probing)

        logging.info(
            f"""   
        Model/Probing Task    | MRR@10 | R@1000
        {self.index_name}    | {mrr:.3f} | {r:.3f}
        {self.index_name_probing}    | {mrr_probing:.3f} | {r_probing:.3f}                
        """
        )
        logging.info(f"Starting official TREC evaluation.")
        write_trec_eval_file(Path(self.trec_eval_file), predictions, self.probing_task)
        write_trec_eval_file(Path(self.trec_eval_file_probing), predictions_probing, self.probing_task)

        # For trec evaluation
        out_file = Path(f"./logs/results/trec_eval_{self.index_name}.tsv")
        out_file_probing = Path(f"./logs/results/trec_eval_{self.index_name_probing}.tsv")
        trec_evaluation(out_file, self.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, self.trec_eval_file, 0)
        trec_evaluation(out_file_probing, self.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, self.trec_eval_file_probing, 0)

        return mrr, mrr_probing, r, r_probing


def main(args, model_choice, probing_task):
    min_ex = MinimalExample(model_choice, probing_task, args.reindex)
    pass

    # Ps_rlace, accs_rlace = {}, {}
    # optimizer_class = torch.optim.SGD
    # optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}
    # optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}

    # output = solve_adv_game(
    #     X,
    #     y,
    #     X,
    #     y,
    #     rank=1,
    #     device="cpu",
    #     out_iters=50000,
    #     optimizer_class=optimizer_class,
    #     optimizer_params_P=optimizer_params_P,
    #     optimizer_params_predictor=optimizer_params_predictor,
    #     epsilon=0.002,
    #     batch_size=128,
    # )


if __name__ == "__main__":
    args = parse_arguments()
    model_choice = args.models.split(",")[0]
    probing_task = args.tasks.split(",")[0]

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(f"./logs/{model_choice}_{probing_task}_{get_timestamp()}.log")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)

    main(args, model_choice, probing_task)
