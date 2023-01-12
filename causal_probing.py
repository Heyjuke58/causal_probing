import sys

sys.path.insert(1, "./ranking-utils")
sys.path.insert(1, "./haystack")
from pathlib import Path
from typing import Union, Optional
import logging
import torch
import pandas as pd
import numpy as np
import pickle
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.schema import Document
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from ranking_utils import write_trec_eval_file
from src.trec_evaluation import trec_evaluation
from src.utils import (
    get_corpus,
    get_qrels,
    get_queries,
    get_top_1000_passages,
    get_timestamp,
    get_batch_amount,
)
from src.argument_parser import parse_arguments

from src.elasticsearch_bm25 import ElasticSearchBM25
from src.rlace import solve_adv_game

## DATA
DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
MSMARCO_DEV_QUERIES_PATH = Path("./assets/msmarco/queries.dev.tsv")
MSMARCO_DEV_QRELS_PATH = Path("./assets/msmarco/qrels.dev.tsv")
MSMARCO_DEV_TOP_1000_PATH = Path("./assets/msmarco/top1000.dev")
MSMARCO_TEST_43_TOP_1000_PATH = Path("./assets/msmarco/trec_43_test_top1000.tsv")
MSMARCO_TEST_43_QUERIES_PATH = Path("./assets/msmarco/trec_43_test_queries.tsv")
MSMARCO_QREL_2019_PATH = "/home/hinrichs/causal_probing/assets/msmarco/2019-qrels-pass.txt"

## TREC EVAL PATH
TREC_EVAL = "/home/hinrichs/causal_probing/trec_eval"

## MODELS
MODEL_CHOICES = {
    "tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco",
    "tct_colbert_msmarcov2": "castorini/tct_colbert-v2-hnp-msmarco-r2",
    "tct_colbert_v1": "castorini/tct_colbert-msmarco",
}

## HYPERPARAMETER
SEED = 12
BATCH_SIZE_PROBING_MODEL = 10
BATCH_SIZE_LM_MODEL = 24
EMBEDDING_SIZE = 768
CHUNK_SIZE = int(1e6)
CHUNK_AMOUNT = math.ceil(int(8.8e6) / CHUNK_SIZE)  # 8.8M is the corpus size
INDEX_TRAINING_SAMPLE_SIZE = int(1.5e6)


class CausalProber:
    def __init__(
        self,
        model_choice: str = "tct_colbert",
        probing_task: str = "bm25",
        all_layers: bool = False,
        reindex_original: bool = False,
        reindex_task: bool = False,
        faiss_index_factory_str: str = "IVF30000,Flat",
        prepend_token: bool = True,
        device_cpu: bool = False,
        chunked_read_in: bool = False,  # whether to create doc store from chunked embeddings instead of saved index file
        init_elastic_search: bool = False,
        doc_store_framework: str = "haystack",
        debug: bool = False,
    ) -> None:
        self.TASKS = {"bm25": self._bm25_probing_task}
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_choice = model_choice
        self.probing_task = probing_task
        self.all_layers = all_layers
        self.dataset = pd.read_json(DATASET_PATH, orient="records")
        self.corpus = {}  # empty dict or dataframe
        self.prepend_token = (
            prepend_token  # whether to prepend '[Q]' and '[D]' to query and document text
            # results in tokenized list: 101, 1031 ([), xxxx (Q/D), 1033 (]), ...
        )

        if debug:
            global MSMARCO_CORPUS_PATH
            global MSMARCO_TEST_43_QUERIES_PATH
            MSMARCO_CORPUS_PATH = Path("./assets/msmarco/toy_collection.tsv")
            MSMARCO_TEST_43_QUERIES_PATH = Path("./assets/msmarco/toy_query.tsv")
            global CHUNK_SIZE
            global CHUNK_AMOUNT
            CHUNK_SIZE = 6000
            CHUNK_AMOUNT = 1

        # index params
        self.faiss_index_factory_str = faiss_index_factory_str
        self.index_name = f"{self.model_choice}_{self.faiss_index_factory_str.replace(',', '_')}{'_debug' if debug else ''}"
        self.index_name_probing = f"{self.model_choice}_{self.probing_task}_{self.faiss_index_factory_str.replace(',', '_')}{'_debug' if debug else ''}"
        self.doc_store: Optional[
            Union[FAISSDocumentStore, FaissSearcher]
        ]  # haystack or pyserini doc store

        # Evaluation files
        self.trec_eval_file = f"./logs/trec/trec_eval_{self.index_name}.tsv"
        self.trec_eval_file_probing = f"./logs/trec/trec_eval_{self.index_name_probing}.tsv"

        self.train, self.val, self.test = self._train_val_test_split()

        if device_cpu:
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available, but torch device is not set to cpu.")
            self.device = torch.device("cuda")

        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_huggingface_str)
        self.model = AutoModel.from_pretrained(self.model_huggingface_str).to(self.device)
        self.query_encoder = TctColBertQueryEncoder(self.model_huggingface_str)

        # run probing task
        self.projection = self.TASKS[self.probing_task]()  # projection to remove a concept

        if doc_store_framework == "haystack":

            # init document store
            if reindex_original:
                if chunked_read_in:
                    self._chunked_index_faiss_doc_store()
                else:
                    self._index_faiss_doc_store()
            else:
                self._init_indexed_faiss_doc_store()

            # init second document store with index of embeddings after projection of probing task is applied
            if reindex_task:
                self._index_probing_faiss_doc_store()
            else:
                self._init_indexed_probing_faiss_doc_store()
        else:
            self._index_faiss_doc_store_pyserini()

        # if init_elastic_search:
        #     self.es_service = self._init_elastic_search_service()
        # else:
        #     del self.corpus

        if debug:
            self._debug()

        self._evaluate_performance()

    def _get_X_probing_task(self, X_query: np.ndarray, X_passage: np.ndarray, merging: str):
        merging_options = {"concat", "multiply_elementwise"}
        if merging not in merging_options:
            raise NotImplementedError(f"Merging option must be one of {merging_options}")
        X_file_str = f"./cache/X_{self.probing_task}_{merging}.pt"
        if not Path(X_file_str).is_file():
            logging.info(
                f"Getting embeddings of query-document pairs of probing task {self.probing_task} dataset ..."
            )
            assert len(X_query) == len(X_passage)
            X_size = len(X_query)
            q_embs = torch.empty((X_size, EMBEDDING_SIZE))
            p_embs = torch.empty((X_size, EMBEDDING_SIZE))
            batches = get_batch_amount(X_size, BATCH_SIZE_LM_MODEL)

            for i in tqdm(range(batches)):
                q_emb = self._get_batch_embeddings_by_forward_pass(
                    X_query.tolist()[
                        BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)
                    ],
                    return_numpy=False,
                )
                p_emb = self._get_batch_embeddings_by_forward_pass(
                    X_passage.tolist()[
                        BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)
                    ],
                    return_numpy=False,
                )
                q_embs[
                    BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), X_size), :
                ] = q_emb
                p_embs[
                    BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), X_size), :
                ] = p_emb

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
        X_query = np.array(["[Q] " + x["query"] if self.prepend_token else x["query"] for x in self.train["input"].values])  # type: ignore
        X_passage = np.array(["[D] " + x["passage"] if self.prepend_token else x["passage"] for x in self.train["input"].values])  # type: ignore
        X = self._get_X_probing_task(X_query, X_passage, "multiply_elementwise")
        y = torch.from_numpy(self.train["targets"].apply(lambda x: x[0]["label"]).to_numpy())  # type: ignore
        y = y.to(torch.float32).unsqueeze(1)

        # P can be calculated with a closed form, since we are dealing with a linear regression
        # (when we try to linearly predict the BM25 score from the models representation)
        P = self.rlace_linear_regression_closed_form(X, y)
        return P

    def _index_faiss_doc_store_pyserini(self):
        self.doc_store = FaissSearcher.from_prebuilt_index(
            "msmarco-passage-tct_colbert-v2-hnp-bf", self.query_encoder
        )

    @staticmethod
    def _cleanup_before_indexing(index_name: str):
        if Path(f"./cache/faiss_doc_store_{index_name}.db").is_file():
            os.remove(f"./cache/faiss_doc_store_{index_name}.db")
        if Path(f"./cache/{index_name}.pickle").is_file():
            os.remove(f"./cache/{index_name}.pickle")

    def _train_index(self, probing_task: bool = False):
        if self.faiss_index_factory_str == "Flat":
            logging.info("Flat index does not need any training.")
            return
        logging.info("Training index...")
        if not isinstance(self.corpus, pd.DataFrame):
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
        corpus_sample = self.corpus.sample(n=INDEX_TRAINING_SAMPLE_SIZE, random_state=SEED)
        corpus_sample_size = len(corpus_sample)
        batches = get_batch_amount(corpus_sample_size, BATCH_SIZE_LM_MODEL)
        p_embs = np.zeros(
            (corpus_sample_size, EMBEDDING_SIZE), dtype="float32"
        )  # must to be float32
        passages = (
            corpus_sample["passage"]
            .apply(lambda x: "[D] " + x if self.prepend_token else x)
            .tolist()
        )

        for i in range(batches):
            embs = self._get_batch_embeddings_by_forward_pass(
                passages[
                    BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_sample_size)
                ]
            )
            p_embs[
                BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_sample_size), :
            ] = embs

        if probing_task:
            p_embs = torch.einsum("bc,cd->bd", torch.from_numpy(p_embs), self.projection).numpy()
            self.probing_doc_store.train_index(
                documents=None, embeddings=p_embs, index=self.index_name_probing
            )
        else:
            self.doc_store.train_index(documents=None, embeddings=p_embs, index=self.index_name)

        logging.info("Training index complete.")

    def _index_faiss_doc_store(self):
        self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
        corpus_size = len(self.corpus)
        self._cleanup_before_indexing(self.index_name)

        self.doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name}.db",
            faiss_index_factory_str=self.faiss_index_factory_str,
            index=self.index_name,
            duplicate_documents="skip",
            # similarity="cosine",
        )
        if self.faiss_index_factory_str != "Flat":
            self._train_index()
        passages = (
            self.corpus["passage"].apply(lambda x: "[D] " + x if self.prepend_token else x).tolist()
        )
        pids = self.corpus["pid"].tolist()
        docs = []
        chunk_counter = 0
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)

        for i in range(batches):
            embs = self._get_batch_embeddings_by_forward_pass(
                passages[BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)]
            )
            for j in range(embs.shape[0]):
                emb = embs[j, :]
                docs.append(
                    # TODO: set id and remove content if not used anymore
                    Document(
                        content=passages[i * BATCH_SIZE_LM_MODEL + j],
                        embedding=emb,  # type: ignore
                        meta={"pid": pids[i * BATCH_SIZE_LM_MODEL + j]},
                    )
                )
            del embs  # free memory
            if len(docs) >= CHUNK_SIZE:
                clipped_docs = docs[:CHUNK_SIZE]
                rest_docs = docs[CHUNK_SIZE:]
                self.doc_store.write_documents(
                    documents=clipped_docs, duplicate_documents="skip", index=self.index_name
                )
                chunk = torch.zeros((CHUNK_SIZE, 768))
                chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{chunk_counter}.pt"
                for i, doc in enumerate(clipped_docs):
                    emb = torch.tensor(doc.embedding).unsqueeze(0)
                    chunk[i] = emb
                torch.save(chunk, chunk_str)
                # free memory
                del chunk
                del docs
                chunk_counter += 1
                docs = rest_docs
        # last chunk which is not the usual chunk size
        chunk = torch.zeros((corpus_size - chunk_counter * CHUNK_SIZE, 768))
        chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{chunk_counter}.pt"
        for i, doc in enumerate(docs):
            emb = torch.tensor(doc.embedding).unsqueeze(0)
            chunk[i] = emb
        torch.save(chunk, chunk_str)
        # free memory
        del chunk
        del docs
        logging.info("Vanilla document embeddings added to document store.")
        # save faiss index and embeddings
        with open(f"./cache/{self.index_name}.pickle", "wb+") as faiss_index_file:
            pickle.dump(self.doc_store.faiss_indexes[self.index_name], faiss_index_file)
        logging.info("Saved Faiss index file.")

    def _chunked_index_faiss_doc_store(self):
        logging.info("Restoring document store with chunked embeddings...")
        self._cleanup_before_indexing(self.index_name)

        self.doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name}.db",
            faiss_index_factory_str=self.faiss_index_factory_str,
            index=self.index_name,
            duplicate_documents="skip",
        )
        if self.faiss_index_factory_str != "Flat":
            self._train_index()
        for i in range(CHUNK_AMOUNT):
            logging.info(f"Processing chunk {i + 1}/{CHUNK_AMOUNT}")
            chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{i}.pt"
            emb_docs_chunk = torch.load(chunk_str)
            docs = [
                Document(
                    id=str(i * CHUNK_SIZE + j),
                    content="",
                    embedding=emb_docs_chunk[j, :].squeeze(0).numpy(),
                )
                for j in range(emb_docs_chunk.shape[0])
            ]
            self.doc_store.write_documents(
                documents=docs, duplicate_documents="skip", index=self.index_name
            )
        logging.info("Vanilla document embeddings added to document store.")
        # save faiss index
        with open(f"./cache/{self.index_name}.pickle", "wb+") as faiss_index_file:
            pickle.dump(self.doc_store.faiss_indexes[self.index_name], faiss_index_file)
        logging.info("Saved Faiss index file.")

    def _init_indexed_faiss_doc_store(self):
        logging.info("Restoring document store with Faiss index...")
        with open(f"./cache/{self.index_name}.pickle", "rb") as faiss_index_file:
            faiss_index = pickle.load(faiss_index_file)
            self.doc_store = FAISSDocumentStore(
                sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name}.db",
                faiss_index_factory_str=self.faiss_index_factory_str,
                index=self.index_name,
                duplicate_documents="skip",
                faiss_index=faiss_index,
            )
        logging.info("Document store with Faiss index restored.")

    def _index_probing_faiss_doc_store(self):
        logging.info(
            f"Adding new index {self.probing_task} of altered embeddings (RLACE/INLP) to second document store. "
            f"Loading altered embeddings from local chunks ..."
        )
        self._cleanup_before_indexing(self.index_name_probing)
        self.probing_doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name_probing}.db",
            faiss_index_factory_str=self.faiss_index_factory_str,
            index=self.index_name_probing,
            duplicate_documents="skip",
            # similarity="cosine", # default: dotproduct
        )
        if self.faiss_index_factory_str != "Flat":
            self._train_index(probing_task=True)

        for i in range(CHUNK_AMOUNT):
            logging.info(f"Processing chunk {i + 1}/{CHUNK_AMOUNT}")
            chunk_str = f"./cache/emb_chunks/{self.index_name}_embs_chunk_{i}.pt"
            emb_docs_chunk = torch.load(chunk_str)
            emb_docs_chunk_altered = torch.einsum("bc,cd->bd", emb_docs_chunk, self.projection)
            docs = [
                Document(
                    id=str(i * CHUNK_SIZE + j),
                    content="",
                    embedding=emb_docs_chunk_altered[j, :].squeeze(0).numpy(),
                )
                for j in range(emb_docs_chunk_altered.shape[0])
            ]
            self.probing_doc_store.write_documents(
                documents=docs, duplicate_documents="skip", index=self.index_name_probing
            )

        # save faiss index
        faiss_index_file_str = f"./cache/{self.index_name_probing}.pickle"
        with open(faiss_index_file_str, "wb+") as faiss_index_file:
            pickle.dump(
                self.probing_doc_store.faiss_indexes[self.index_name_probing], faiss_index_file
            )
        logging.info(
            f"Initialized second docuemnt store with altered embeddings of task {self.probing_task}. "
            f"Saved Faiss index file to {faiss_index_file_str}."
        )

    def _init_indexed_probing_faiss_doc_store(self):
        logging.info(
            f"Restoring document store with Faiss index of altered embeddings (task: {self.probing_task}) ..."
        )
        faiss_index_file_str = f"./cache/{self.index_name_probing}.pickle"
        if Path(faiss_index_file_str).is_file():
            with open(faiss_index_file_str, "rb") as faiss_index_file:
                faiss_index = pickle.load(faiss_index_file)
                self.probing_doc_store = FAISSDocumentStore(
                    sql_url=f"sqlite:///cache/faiss_doc_store_{self.index_name_probing}.db",
                    faiss_index_factory_str=self.faiss_index_factory_str,
                    index=self.index_name_probing,
                    duplicate_documents="skip",
                    faiss_index=faiss_index,
                )
        else:
            raise FileNotFoundError(
                f"Cannot load index {self.index_name_probing} from file. You need to reindex."
            )
        logging.info(
            f"Initialized second docuemnt store with altered embeddings of task {self.probing_task}."
        )

    def _init_elastic_search_service(self):
        if not self.corpus:
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
        pool = self.corpus["passage"].to_dict()
        es_service = ElasticSearchBM25(
            pool,  # type: ignore
            index_name="msmarco3",  # TODO: if more than msmarco should be used this has to be edited
            service_type="docker",
            max_waiting=100,
            port_http="12375",
            port_tcp="12376",
            es_version="7.16.2",
            reindexing=False,
        )

        return es_service

    def _get_embedding_by_forward_pass(self, sequence: str) -> np.ndarray:
        X_sequence_tokenized = self.tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=512,  # in pyserini 36 for queries!
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_sequence_tokenized)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2).flatten()

    def _get_batch_embeddings_by_forward_pass(
        self, batch: list, return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        X_batch_tokenized = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_batch_tokenized)
        embeddings = outputs.last_hidden_state.detach().cpu()
        if return_numpy:
            embeddings = embeddings.numpy()
            return np.average(embeddings[:, 4:, :], axis=-2)
        else:
            return torch.mean(embeddings[:, 4:, :], dim=-2)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        query = "[Q] " + query if self.prepend_token else query
        return self._get_embedding_by_forward_pass(query)

    def _get_passage_embedding(self, passage: str) -> np.ndarray:
        passage = "[D] " + passage if self.prepend_token else passage
        return self._get_embedding_by_forward_pass(passage)

    def _debug(self):
        queries = get_queries(MSMARCO_TEST_43_QUERIES_PATH)
        top1000 = get_top_1000_passages(MSMARCO_TEST_43_TOP_1000_PATH)
        if not isinstance(self.corpus, pd.DataFrame):
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)

        for i, row in queries.iterrows():
            qid, query = row[0], row[1]
            try:
                top1_relevant_pid = top1000[qid][0]
                top1000_relevant_pids = top1000[qid]
                pass
            except:
                continue
            pass
            # pyserini
            if isinstance(self.doc_store, FaissSearcher):
                relevant_docs = self.doc_store.search(row[1])
                pass
            elif isinstance(self.doc_store, FAISSDocumentStore):
                q_emb_np = self._get_query_embedding(query)
                q_emb_pt = torch.from_numpy(q_emb_np).unsqueeze(0)
                q_emb_probing_pt = q_emb_pt.mm(self.projection)
                q_emb_probing_np = q_emb_probing_pt.squeeze(0).numpy()

                relevant_docs = self.doc_store.query_by_embedding(
                    q_emb_np, index=self.index_name, top_k=1000, return_embedding=True
                )
                relevant_docs_probing = self.probing_doc_store.query_by_embedding(
                    q_emb_probing_np, index=self.index_name_probing, top_k=10, return_embedding=True
                )

                top1_relevant_doc = self.corpus[self.corpus["pid"] == top1_relevant_pid]
                passage = top1_relevant_doc["passage"].tolist()[0]
                p_emb_np = self._get_passage_embedding(passage)
                p_emb_pt = torch.from_numpy(p_emb_np)
                for j, doc in enumerate(relevant_docs):
                    if doc.meta["pid"] == top1_relevant_pid or doc.meta["pid"] == 8434617:
                        emb_from_doc_store_np = doc.embedding
                        emb_from_doc_store_pt = torch.from_numpy(emb_from_doc_store_np)
                        all_close = torch.allclose(p_emb_pt, emb_from_doc_store_pt, rtol=1e-04)
                        pass

    def _evaluate_performance(self, mrr_at: int = 10, recall_at: int = 1000):
        logging.info("Evaluating performance ...")
        queries = get_queries(MSMARCO_TEST_43_QUERIES_PATH)

        predictions: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.
        predictions_probing: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.

        for i, row in tqdm(queries.iterrows(), desc="Evaluating"):
            qid = row[0]
            query = row[1]

            q_emb_np = self._get_query_embedding(query)
            q_emb_pt = torch.from_numpy(q_emb_np).unsqueeze(0)
            q_emb_probing_pt = q_emb_pt.mm(self.projection)
            q_emb_probing_np = q_emb_probing_pt.squeeze(0).numpy()
            relevant_docs = self.doc_store.query_by_embedding(
                q_emb_np, index=self.index_name, top_k=recall_at, return_embedding=True
            )
            relevant_docs_probing = self.probing_doc_store.query_by_embedding(
                q_emb_probing_np,
                index=self.index_name_probing,
                top_k=recall_at,
                return_embedding=True,
            )
            docs_dict = {doc.id: doc.score for doc in relevant_docs}
            predictions[str(qid)] = docs_dict  # type: ignore
            docs_dict = {doc.id: doc.score for doc in relevant_docs_probing}
            predictions_probing[str(qid)] = docs_dict  # type: ignore

        logging.info(f"Starting official TREC evaluation.")
        write_trec_eval_file(Path(self.trec_eval_file), predictions, self.probing_task)
        write_trec_eval_file(
            Path(self.trec_eval_file_probing), predictions_probing, self.probing_task
        )

        # For trec evaluation
        timestamp = get_timestamp()
        out_file = Path(f"./logs/results/trec_eval_{self.index_name}_{timestamp}.tsv")
        out_file_probing = Path(
            f"./logs/results/trec_eval_{self.index_name_probing}_{timestamp}.tsv"
        )
        trec_evaluation(
            out_file, self.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, self.trec_eval_file, 0
        )
        trec_evaluation(
            out_file_probing,
            self.model_choice,
            MSMARCO_QREL_2019_PATH,
            TREC_EVAL,
            self.trec_eval_file_probing,
            0,
        )
        logging.info(
            f"Evaluation done."
            f"Logged results to './logs/results/trec_eval_{self.index_name_probing}_{timestamp}.tsv' and to non task results, respectively."
        )


# Adversarial R-LACE
#
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

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logging_level = logging.DEBUG
    if not args.debug:
        logging_level = logging.INFO
        file_handler = logging.FileHandler(
            f"./logs/{args.model_choice}_{args.probing_task}_{get_timestamp()}.log"
        )
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging_level)
        root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging_level)
    root.addHandler(console_handler)
    root.setLevel(logging_level)

    args = vars(args)
    # del args["debug"]
    CausalProber(**args)
