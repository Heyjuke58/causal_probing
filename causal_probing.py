import sys

sys.path.insert(1, "./ranking-utils")
sys.path.insert(1, "./haystack")
from pathlib import Path
from typing import Union, Optional, Generator
import logging
import torch
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
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
    get_model_config,
    CustomBertEncoder,
)
from src.file_locations import *
from src.hyperparameter import *
from src.argument_parser import parse_arguments

from src.elasticsearch_bm25 import ElasticSearchBM25
from src.rlace import solve_adv_game


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
        generate_emb_chunks: bool = False,
        init_elastic_search: bool = False,
        doc_store_framework: str = "haystack",
        ablation_last_layer: bool = False,
        debug: bool = False,
    ) -> None:
        # Running options
        self.TASKS = {"bm25": self._init_probing_task, "sem_sim": self._init_probing_task}
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_choice = model_choice
        self.probing_task = probing_task
        self.all_layers = all_layers
        self.prepend_token = (
            prepend_token  # whether to prepend '[Q]' and '[D]' to query and document text
            # results in tokenized list: 101, 1031 ([), xxxx (Q/D), 1033 (]), ...
        )
        self.generate_emb_chunks = generate_emb_chunks
        self.reindex_original = reindex_original
        self.reindex_task = reindex_task
        self.chunked_read_in = chunked_read_in
        self.ablation_last_layer = ablation_last_layer
        self.debug = debug
        self.doc_store_framework = doc_store_framework

        self.dataset = pd.read_json(DATASET_PATHS[probing_task], orient="records")
        self.corpus = {}  # empty dict or dataframe

        if debug:
            global MSMARCO_CORPUS_PATH
            global MSMARCO_TREC_2019_TEST_QUERIES_PATH
            MSMARCO_CORPUS_PATH = MSMARCO_TOY_CORPUS_PATH
            # MSMARCO_TREC_2019_TEST_QUERIES_PATH = MSMARCO_TOY_QUERIES_PATH
            global CHUNK_SIZE
            global CHUNK_AMOUNT
            CHUNK_SIZE = 3500
            CHUNK_AMOUNT = 2
            global INDEX_TRAINING_SAMPLE_SIZE
            INDEX_TRAINING_SAMPLE_SIZE = 1500

        if CHUNK_SIZE % BATCH_SIZE_LM_MODEL != 0:
            raise ValueError(
                "CHUNK SIZE is not divisable by BATCH SIZE! This will most likely result in an index error."
            )

        # index params
        self.faiss_index_factory_str = faiss_index_factory_str
        self.index_name = f"{self.model_choice}_{self.faiss_index_factory_str.replace(',', '_')}{'_debug' if debug else ''}_fixed_average"  # TODO: remove suffix!
        self.index_name_probing = f"{self.model_choice}_{self.probing_task}_{self.faiss_index_factory_str.replace(',', '_')}{'_debug' if debug else ''} "
        self.chunk_name = f"{self.model_choice}{'_debug' if debug else ''}_embs_chunk"
        self.doc_store: FAISSDocumentStore
        self.doc_store_pyserini: FaissSearcher
        self.probing_doc_stores: list[FAISSDocumentStore]  # one for each BERT layer
        self.probing_doc_store: FAISSDocumentStore
        self.probing_doc_store_average: FAISSDocumentStore  # for ablation test
        self.probing_doc_store_token_wise: FAISSDocumentStore  # for ablation test

        self.train, self.val, self.test = self._train_val_test_split()

        if device_cpu:
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available, but torch device is not set to cpu.")
            self.device = torch.device("cuda")

        # init model
        model_cofig = get_model_config(self.model_huggingface_str)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_huggingface_str)
        self.model = BertModel.from_pretrained(self.model_huggingface_str, config=model_cofig).to(
            self.device
        )
        if doc_store_framework == "pyserini":
            self.query_encoder = TctColBertQueryEncoder(self.model_huggingface_str)

    def run(self):
        self.timestamp = get_timestamp()
        # run probing task
        self.projection = self.TASKS[self.probing_task]()  # projection to remove a concept

        # if self.doc_store_framework == "pyserini":
        #     self._index_faiss_doc_store_pyserini()

        if self.generate_emb_chunks:
            self._generate_embeddings_and_save_them_chunked()

        # init document store
        if self.reindex_original:
            if self.chunked_read_in:
                self._chunked_index_faiss_doc_store()
            else:
                self._index_faiss_doc_store_no_chunk_save()
        else:
            self._init_indexed_faiss_doc_store()

        if self.debug:
            # self._debug()
            self._debug_embedding_score()
        else:
            self._evaluate(self.doc_store, self.index_name)

        # # init second document store with index of embeddings after projection of probing task is applied
        # if self.reindex_task and not self.all_layers and not self.ablation_last_layer:
        #     self._index_probing_faiss_doc_store()
        #     self._evaluate(self.probing_doc_store, self.index_name_probing, alter_query_emb=True)
        # elif self.reindex_task and self.all_layers and not self.ablation_last_layer:
        #     self._index_probing_faiss_doc_stores_all_layers()
        # # elif not reindex_task and all_layers:
        # #     self._init_indexed_probing_faiss_doc_stores_all_layers()
        # elif self.ablation_last_layer:
        #     self._init_probing_doc_stores_ablation_last_layer_non_parallel()
        #     self._evaluate(
        #         self.probing_doc_store_average,
        #         self.index_name_probing,
        #         alter_query_emb=True,
        #         index_suffix="average",
        #     )
        #     self._evaluate(
        #         self.probing_doc_store_token_wise,
        #         self.index_name_probing,
        #         alter_query_emb=True,
        #         index_suffix="token_wise",
        #     )
        # else:
        #     self._init_indexed_probing_faiss_doc_store()

        # if init_elastic_search:
        #     self.es_service = self._init_elastic_search_service()

    def _init_corpus(self):
        if not isinstance(self.corpus, pd.DataFrame):
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)

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
                ] = q_emb  # type: ignore
                p_embs[
                    BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), X_size), :
                ] = p_emb  # type: ignore

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

    def _init_probing_task(self):
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
        doc_store = FaissSearcher.from_prebuilt_index(
            "msmarco-passage-tct_colbert-v2-hnp-bf", self.query_encoder
        )
        if doc_store:
            self.doc_store_pyserini = doc_store

    @staticmethod
    def _cleanup_before_indexing(
        index_name: str,
        layer: Optional[Union[int, bool]] = None,
        index_suffixes: Optional[list[str]] = None,
    ):
        def _cleanup(layer: Optional[int] = None, index_suffix: Optional[str] = None):
            layer = layer
            combined_str = f"{index_name}{f'_layer_{layer}' if isinstance(layer, int) else ''}{f'_{index_suffix}' if index_suffix else ''}"
            if Path(f"./cache/faiss_doc_store_{combined_str}.db").is_file():
                os.remove(f"./cache/faiss_doc_store_{combined_str}.db")
            if Path(f"./cache/{combined_str}.pickle").is_file():
                os.remove(f"./cache/{combined_str}.pickle")

        if isinstance(layer, bool):
            for layer in range(AMOUNT_LAYERS):
                _cleanup(layer=layer)
        elif isinstance(layer, int):
            _cleanup(layer=layer)
        elif isinstance(index_suffixes, list):
            for suf in index_suffixes:
                _cleanup(index_suffix=suf)

        else:
            _cleanup()

    def _get_doc_store(
        self,
        index_name,
        layer: Optional[int] = None,
        load_from_saved_index: bool = False,
        index_suffix: Optional[str] = None,
    ) -> FAISSDocumentStore:
        faiss_index = None
        combined_str = f"{index_name}{f'_layer_{layer}' if isinstance(layer, int) else ''}{f'_{index_suffix}' if index_suffix else ''}"
        if load_from_saved_index:
            if Path(f"./cache/{combined_str}.pickle").is_file():
                with open(f"./cache/{combined_str}.pickle", "rb") as faiss_index_file:
                    faiss_index = pickle.load(faiss_index_file)
            else:
                raise FileNotFoundError(
                    f"Cannot load index {index_name} from file. You need to reindex."
                )
        doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{combined_str}.db",
            faiss_index_factory_str=self.faiss_index_factory_str,
            index=index_name,
            duplicate_documents="skip",
            faiss_index=faiss_index,
        )
        return doc_store

    @staticmethod
    def _pickle_dump_index(
        doc_store: FAISSDocumentStore, index_name: str, index_suffix: Optional[str] = None
    ):
        logging.info(f"Saving faiss index {index_name}...")
        combined_str = f"{index_name}{f'_{index_suffix}' if index_suffix else ''}"
        with open(f"./cache/{combined_str}.pickle", "wb+") as faiss_index_file:
            pickle.dump(doc_store.faiss_indexes[index_name], faiss_index_file)
        logging.info(f"Saved faiss index {index_name} to file: {combined_str}.pickle.")

    # TODO: maybe remove later when not needed anymore
    def _get_passage_embeddings_for_index_training_all_layers(self) -> np.ndarray:
        self._init_corpus()
        corpus_sample = self.corpus.sample(n=INDEX_TRAINING_SAMPLE_SIZE, random_state=SEED)  # type: ignore
        batches = get_batch_amount(INDEX_TRAINING_SAMPLE_SIZE, BATCH_SIZE_LM_MODEL)
        passages = (
            "[D] " + corpus_sample["passage"] if self.prepend_token else corpus_sample["passage"]
        ).tolist()
        p_embs = np.zeros(
            (AMOUNT_LAYERS, INDEX_TRAINING_SAMPLE_SIZE, EMBEDDING_SIZE), dtype="float32"
        )
        for i in range(batches):
            embs = self._get_batch_embeddings_all_layers_by_forward_pass(
                passages[
                    BATCH_SIZE_LM_MODEL
                    * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE)
                ],
                return_numpy=False,
            )
            for layer in range(AMOUNT_LAYERS):
                p_embs[
                    layer,
                    BATCH_SIZE_LM_MODEL
                    * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE),
                    :,
                ] = torch.einsum("bc,cd->bd", embs[layer], self.projection).numpy()

        return p_embs

    def _train_index(self, probing_task: bool = False):
        if self.faiss_index_factory_str == "Flat":
            logging.info("Flat index does not need any training.")
            return
        logging.info("Training index...")
        self._init_corpus()
        corpus_sample = self.corpus.sample(n=INDEX_TRAINING_SAMPLE_SIZE, random_state=SEED)  # type: ignore
        batches = get_batch_amount(INDEX_TRAINING_SAMPLE_SIZE, BATCH_SIZE_LM_MODEL)
        passages = (
            "[D] " + corpus_sample["passage"] if self.prepend_token else corpus_sample["passage"]
        ).tolist()
        # TODO: remove all_layers option when multiprocessing works
        if probing_task and self.all_layers:
            p_embs = np.zeros(
                (AMOUNT_LAYERS, INDEX_TRAINING_SAMPLE_SIZE, EMBEDDING_SIZE), dtype="float32"
            )
            for i in range(batches):
                embs = self._get_batch_embeddings_all_layers_by_forward_pass(
                    passages[
                        BATCH_SIZE_LM_MODEL
                        * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE)
                    ],
                    return_numpy=False,
                )
                for layer in range(AMOUNT_LAYERS):
                    p_embs[
                        layer,
                        BATCH_SIZE_LM_MODEL
                        * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE),
                        :,
                    ] = torch.einsum("bc,cd->bd", embs[layer], self.projection).numpy()
            for layer in range(AMOUNT_LAYERS):
                self.probing_doc_stores[layer].train_index(
                    documents=None, embeddings=p_embs[layer, :, :], index=self.index_name_probing
                )
        else:
            p_embs = np.zeros(
                (INDEX_TRAINING_SAMPLE_SIZE, EMBEDDING_SIZE), dtype="float32"
            )  # must to be float32
            for i in range(batches):
                embs = self._get_batch_embeddings_by_forward_pass(
                    passages[
                        BATCH_SIZE_LM_MODEL
                        * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE)
                    ]
                )
                p_embs[
                    BATCH_SIZE_LM_MODEL
                    * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE),
                    :,
                ] = embs
            if probing_task:
                p_embs = torch.einsum(
                    "bc,cd->bd", torch.from_numpy(p_embs), self.projection
                ).numpy()
                self.probing_doc_store.train_index(
                    documents=None, embeddings=p_embs, index=self.index_name_probing
                )
            else:
                self.doc_store.train_index(documents=None, embeddings=p_embs, index=self.index_name)

        logging.info("Training index complete.")

    # TODO: generation might not be necessary at least for all layers
    def _generate_embeddings_and_save_them_chunked(self):
        self._init_corpus()
        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()
        corpus_size = len(passages)
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        chunk_counter = 0
        embs_all_layers = [torch.zeros((CHUNK_SIZE, EMBEDDING_SIZE)) for _ in range(AMOUNT_LAYERS)]
        last_chunk = False

        for i in range(batches):
            # handle different size of last chunk and set tensor sizes accordingly
            if chunk_counter + 1 == CHUNK_AMOUNT and not last_chunk:
                embs_all_layers = [
                    torch.zeros((corpus_size - chunk_counter * CHUNK_SIZE, EMBEDDING_SIZE))
                    for _ in range(AMOUNT_LAYERS)
                ]
                last_chunk = True

            embs_batch = self._get_batch_embeddings_all_layers_by_forward_pass(
                passages[BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)],
                return_numpy=False,
            )
            for layer, embs in enumerate(embs_batch):
                j = (i + 1) * BATCH_SIZE_LM_MODEL % CHUNK_SIZE
                embs_all_layers[layer][i * BATCH_SIZE_LM_MODEL % CHUNK_SIZE : CHUNK_SIZE if j == 0 else j, :] = embs  # type: ignore
            if (
                i + 1 - (chunk_counter * CHUNK_SIZE / BATCH_SIZE_LM_MODEL)
            ) * BATCH_SIZE_LM_MODEL >= CHUNK_SIZE:
                # this condition is not triggered for the last chunk
                for layer in range(AMOUNT_LAYERS):
                    self.save_embeddings_chunk(embs_all_layers[layer], chunk_counter, layer)
                chunk_counter += 1
                embs_all_layers = [
                    torch.zeros((CHUNK_SIZE, EMBEDDING_SIZE)) for _ in range(AMOUNT_LAYERS)
                ]
        for layer in range(AMOUNT_LAYERS):
            self.save_embeddings_chunk(embs_all_layers[layer], chunk_counter, layer)

    def save_embeddings_chunk(
        self,
        embeddings: torch.Tensor,
        chunk_counter: int,
        layer: Optional[int] = None,
    ):
        chunk_str = f"./cache/emb_chunks/{self.chunk_name}_{chunk_counter}{f'_layer_{layer}' if isinstance(layer, int) else ''}.pt"
        torch.save(embeddings, chunk_str)

    def load_embeddings_chunk(self, chunk_counter, layer: Optional[int] = None) -> torch.Tensor:
        chunk_str = f"./cache/emb_chunks/{self.chunk_name}_{chunk_counter}{f'_layer_{layer}' if isinstance(layer, int) else ''}.pt"
        return torch.load(chunk_str)

    def _index_faiss_doc_store_all_layers(self):
        self._init_corpus()
        corpus_size = len(self.corpus)
        self._cleanup_before_indexing(self.index_name, layer=True)
        self.doc_store = self._get_doc_store(self.index_name)

        if self.faiss_index_factory_str != "Flat":
            self._train_index()

        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()
        pids = self.corpus["pid"].tolist()
        docs = []
        chunk_counter = 0
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)

        for i in range(batches):
            embs_all_layers = self._get_batch_embeddings_all_layers_by_forward_pass(
                passages[BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)],
                return_numpy=False,
            )
            for layer, embs in enumerate(embs_all_layers):
                for j in range(embs.shape[0]):
                    emb = embs[j, :]
                    docs[layer].append(
                        Document(
                            content="",
                            embedding=emb,  # type: ignore
                            meta={
                                "pid": pids[i * BATCH_SIZE_LM_MODEL + j],
                                "vector_id": pids[i * BATCH_SIZE_LM_MODEL + j],
                            },
                        )
                    )
                del embs
                if len(docs[0]) >= CHUNK_SIZE:
                    pass
            embs = self._get_batch_embeddings_by_forward_pass(
                passages[BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)],
            )
            for j in range(embs.shape[0]):
                emb = embs[j, :]
                docs.append(
                    # TODO: set id and remove content if not used anymore
                    Document(
                        content=passages[i * BATCH_SIZE_LM_MODEL + j],
                        embedding=emb,  # type: ignore
                        meta={
                            "pid": pids[i * BATCH_SIZE_LM_MODEL + j],
                            "vector_id": pids[i * BATCH_SIZE_LM_MODEL + j],
                        },
                    )
                )
            del embs  # free memory
            if len(docs) >= CHUNK_SIZE:
                clipped_docs = docs[:CHUNK_SIZE]
                rest_docs = docs[CHUNK_SIZE:]
                self.doc_store.write_documents(
                    documents=clipped_docs, duplicate_documents="skip", index=self.index_name
                )
                # TODO: remove generation of chunks, since it is done by other function
                chunk = torch.zeros((CHUNK_SIZE, EMBEDDING_SIZE))
                for i, doc in enumerate(clipped_docs):
                    emb = torch.tensor(doc.embedding).unsqueeze(0)
                    chunk[i] = emb
                chunk_str = f"./cache/emb_chunks/{self.chunk_name}_{chunk_counter}.pt"
                torch.save(chunk, chunk_str)
                # free memory
                del chunk
                del docs
                chunk_counter += 1
                docs = rest_docs
        # last chunk which is not the usual chunk size
        chunk = torch.zeros((corpus_size - chunk_counter * CHUNK_SIZE, 768))
        chunk_str = f"./cache/emb_chunks/{self.chunk_name}_{chunk_counter}.pt"
        for i, doc in enumerate(docs):
            emb = torch.tensor(doc.embedding).unsqueeze(0)
            chunk[i] = emb
        torch.save(chunk, chunk_str)
        # free memory
        del chunk
        del docs
        logging.info("Vanilla document embeddings added to document store.")
        self._pickle_dump_index(self.doc_store, self.index_name)

    def _index_faiss_doc_store_no_chunk_save(self):
        self._init_corpus()
        corpus_size = len(self.corpus)
        self._cleanup_before_indexing(self.index_name)
        self.doc_store = self._get_doc_store(self.index_name)

        if self.faiss_index_factory_str != "Flat":
            self._train_index()
            self.doc_store.faiss_indexes[self.index_name].make_direct_map()
        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()
        pids = self.corpus["pid"].tolist()
        docs = []
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)

        for i in range(batches):
            embs: np.ndarray = self._get_batch_embeddings_by_forward_pass(
                passages[BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)],
            )  # type: ignore
            for j in range(embs.shape[0]):
                emb = embs[j, :]
                docs.append(
                    Document(
                        id=str(i * BATCH_SIZE_LM_MODEL + j),
                        content="",
                        embedding=emb,  # type: ignore
                        meta={
                            "pid": pids[i * BATCH_SIZE_LM_MODEL + j],
                            "vector_id": pids[i * BATCH_SIZE_LM_MODEL + j],
                        },
                    )
                )
            del embs  # free memory
            if len(docs) >= CHUNK_SIZE:
                self.doc_store.write_documents(
                    documents=docs, duplicate_documents="skip", index=self.index_name
                )
                docs = []
        self.doc_store.write_documents(
            documents=docs, duplicate_documents="skip", index=self.index_name
        )
        del docs
        logging.info("Vanilla document embeddings added to document store.")
        self._pickle_dump_index(self.doc_store, self.index_name)

    def _chunked_index_faiss_doc_store(self):
        logging.info("Restoring document store with chunked embeddings...")
        self._cleanup_before_indexing(self.index_name)
        self.doc_store = self._get_doc_store(self.index_name)
        if self.faiss_index_factory_str != "Flat":
            self._train_index()
        for i in range(CHUNK_AMOUNT):
            logging.info(f"Processing chunk {i + 1}/{CHUNK_AMOUNT}")
            emb_docs_chunk = self.load_embeddings_chunk(i, LAST_LAYER_IDX)
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
        self._pickle_dump_index(self.doc_store, self.index_name)

    def _init_indexed_faiss_doc_store(self):
        logging.info("Restoring document store with Faiss index...")
        self.doc_store = self._get_doc_store(self.index_name, load_from_saved_index=True)
        if "ivf" in self.faiss_index_factory_str.lower():
            self.doc_store.faiss_indexes[self.index_name].make_direct_map()
        logging.info("Document store with Faiss index restored.")

    def _index_probing_faiss_doc_store(self):
        logging.info(
            f"Adding new index {self.probing_task} of altered embeddings (RLACE/INLP) to second document store. "
            f"Loading altered embeddings from local chunks ..."
        )
        self._cleanup_before_indexing(self.index_name_probing)
        self.probing_doc_store = self._get_doc_store(self.index_name_probing)
        if self.faiss_index_factory_str != "Flat":
            self._train_index(probing_task=True)

        for i in range(CHUNK_AMOUNT):
            logging.info(f"Processing chunk {i + 1}/{CHUNK_AMOUNT}")
            chunk_str = f"./cache/emb_chunks/{self.chunk_name}_{i}_layer_{LAST_LAYER_IDX}.pt"
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
        logging.info(
            f"Initialized second docuemnt store with altered embeddings of task {self.probing_task}."
        )
        self._pickle_dump_index(self.probing_doc_store, self.index_name_probing)

    @staticmethod
    def _parallel_indexing_probing_doc_store(
        layer: int,
        faiss_index_factory_str: str,
        chunk_name: str,
        index_name_probing: str,
        projection: torch.Tensor,
        passage_embeddings: Optional[np.ndarray],
    ):
        logging.info(f"Started initilization of doc store for layer {layer}.")
        doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{index_name_probing}_layer_{layer}.db",
            faiss_index_factory_str=faiss_index_factory_str,
            index=index_name_probing,
            duplicate_documents="skip",
        )
        if isinstance(passage_embeddings, np.ndarray):
            doc_store.train_index(
                documents=None, embeddings=passage_embeddings, index=index_name_probing
            )

        for i in range(CHUNK_AMOUNT):
            chunk_str = f"./cache/emb_chunks/{chunk_name}_{i}_layer_{layer}.pt"
            emb_docs_chunk = torch.load(chunk_str)
            emb_docs_chunk_altered = torch.einsum("bc,cd->bd", emb_docs_chunk, projection)
            docs = [
                Document(
                    id=str(i * CHUNK_SIZE + j),
                    content="",
                    embedding=emb_docs_chunk_altered[j, :].squeeze(0).numpy(),
                )
                for j in range(emb_docs_chunk_altered.shape[0])
            ]
            doc_store.write_documents(
                documents=docs, duplicate_documents="skip", index=index_name_probing
            )
        logging.info(f"Finished initilization of doc store of layer {layer}.")
        faiss_index_file_str = f"./cache/{index_name_probing}_layer_{layer}.pickle"
        with open(faiss_index_file_str, "wb+") as faiss_index_file:
            pickle.dump(
                doc_store.faiss_indexes[index_name_probing],
                faiss_index_file,
            )

    def _index_probing_faiss_doc_stores_all_layers(self):
        logging.info(
            f"Adding new indices {self.probing_task} of altered embeddings (RLACE/INLP) to a document store for each layer. "
            f"Loading altered embeddings from local chunks ..."
        )
        self._cleanup_before_indexing(self.index_name_probing)

        if self.faiss_index_factory_str != "Flat":
            p_embs = self._get_passage_embeddings_for_index_training_all_layers()
            iter_params = zip(
                range(AMOUNT_LAYERS),
                [self.faiss_index_factory_str] * AMOUNT_LAYERS,
                [self.chunk_name] * AMOUNT_LAYERS,
                [self.index_name_probing] * AMOUNT_LAYERS,
                [self.projection] * AMOUNT_LAYERS,
                [p_embs[layer, :, :] for layer in range(AMOUNT_LAYERS)],
            )
        else:
            iter_params = zip(
                range(AMOUNT_LAYERS),
                [self.faiss_index_factory_str] * AMOUNT_LAYERS,
                [self.chunk_name] * AMOUNT_LAYERS,
                [self.index_name_probing] * AMOUNT_LAYERS,
                [self.projection] * AMOUNT_LAYERS,
                [None] * AMOUNT_LAYERS,
            )

        with Pool(CPU_CORES) as pool:
            pool.starmap(self._parallel_indexing_probing_doc_store, iter_params)

        # reload doc stores from faiss indexes
        self.probing_doc_stores = []
        for layer in range(AMOUNT_LAYERS):
            doc_store = self._get_doc_store(
                self.index_name_probing,
                layer,
                load_from_saved_index=True,
            )
            self.probing_doc_stores.append(doc_store)
        logging.info(
            f"Initialized document stores with altered embeddings of task {self.probing_task} for each layer."
        )

    @staticmethod
    def _parallel_indexing_probing_doc_store_2(
        layer: int,
        faiss_index_factory_str: str,
        model_str: str,
        index_name_probing: str,
        projection: torch.Tensor,
        corpus_size: int,
        device,
        emb_batch_generator: Generator,
    ):
        logging.info(f"Started initilization of doc store for layer {layer}.")
        doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{index_name_probing}_layer_{layer}.db",
            faiss_index_factory_str=faiss_index_factory_str,
            index=index_name_probing,
            duplicate_documents="skip",
        )
        if faiss_index_factory_str != "Flat":
            # TODO: training!
            pass

        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        custom_encoder = CustomBertEncoder(get_model_config(model_str), layer).to(device)
        docs = []

        for i in range(batches):
            embs = next(emb_batch_generator)
            embs_altered = torch.einsum("bac,cd->bad", embs, projection)
            if layer != AMOUNT_LAYERS - 1:
                # last layer output does not need to be fed into model again
                embs_altered.to(device)
                last_hidden = custom_encoder(embs_altered).last_hidden_state.detach().cpu()
                resulting_embs = torch.mean(last_hidden[:, 4:, :], dim=-2)
            else:
                resulting_embs = torch.mean(embs_altered[:, 4:, :], dim=-2)
            docs.extend(
                [
                    Document(
                        id=str(i * BATCH_SIZE_LM_MODEL + j),
                        content="",
                        embedding=resulting_embs[j, :].squeeze(0).numpy(),
                    )
                    for j in range(resulting_embs.shape[0])
                ]
            )
            if len(docs) >= CHUNK_SIZE:
                doc_store.write_documents(
                    documents=docs, duplicate_documents="skip", index=index_name_probing
                )
                docs = []

        doc_store.write_documents(
            documents=docs, duplicate_documents="skip", index=index_name_probing
        )
        del docs
        logging.info(f"Finished initilization of doc store of layer {layer}.")
        faiss_index_file_str = f"./cache/{index_name_probing}_layer_{layer}.pickle"
        with open(faiss_index_file_str, "wb+") as faiss_index_file:
            pickle.dump(
                doc_store.faiss_indexes[index_name_probing],
                faiss_index_file,
            )

    def _index_probing_faiss_doc_stores_all_layers_2(self):
        logging.info(
            f"Adding new indices {self.probing_task} of altered embeddings (RLACE/INLP) to a document store for each layer. "
        )
        self._cleanup_before_indexing(self.index_name_probing)
        self._init_corpus()
        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()

        iter_params = zip(
            range(AMOUNT_LAYERS),
            [self.faiss_index_factory_str] * AMOUNT_LAYERS,
            [self.model_choice] * AMOUNT_LAYERS,
            [self.index_name_probing] * AMOUNT_LAYERS,
            [self.projection] * AMOUNT_LAYERS,
            [len(passages)] * AMOUNT_LAYERS,
            [self.device] * AMOUNT_LAYERS,
            [
                self._iter_corpus_and_get_embeddings_for_layer(BATCH_SIZE_LM_MODEL, passages, layer)
                for layer in range(AMOUNT_LAYERS)
            ],
        )

        with Pool(CPU_CORES) as pool:
            pool.starmap(self._parallel_indexing_probing_doc_store_2, iter_params)

        # reload doc stores from faiss indexes
        self.probing_doc_stores = []
        for layer in range(AMOUNT_LAYERS):
            doc_store = self._get_doc_store(
                self.index_name_probing,
                layer,
                load_from_saved_index=True,
            )
            self.probing_doc_stores.append(doc_store)
        logging.info(
            f"Initialized document stores with altered embeddings of task {self.probing_task} for each layer."
        )

    @staticmethod
    def _parallel_indexing_probing_doc_store_ablation_last_layer(
        index_str: str,
        faiss_index_factory_str: str,
        index_name_probing: str,
        projection: torch.Tensor,
        emb_batches: list,
        emb_batches_train_index: list,
    ):
        logging.info(f"Started initilization of doc store for ablation: {index_str}.")
        doc_store = FAISSDocumentStore(
            sql_url=f"sqlite:///cache/faiss_doc_store_{index_name_probing}_{index_str}.db",
            faiss_index_factory_str=faiss_index_factory_str,
            index=index_name_probing,
            duplicate_documents="skip",
        )

        def get_altered_embs(embs):
            if index_str == "average":
                embs_avg = torch.mean(embs[:, 4:, :], dim=-2)
                return torch.einsum("bc,cd->bd", embs_avg, projection)
            else:
                embs_altered = torch.einsum("bac,cd->bad", embs, projection)
                return torch.mean(embs_altered[:, 4:, :], dim=-2)

        if faiss_index_factory_str != "Flat":
            logging.info(f"Training index of doc store for ablation: {index_str}")
            p_embs = np.zeros((INDEX_TRAINING_SAMPLE_SIZE, EMBEDDING_SIZE), dtype="float32")
            for i, embs in enumerate(emb_batches_train_index):
                embs_altered = get_altered_embs(embs)
                p_embs[
                    BATCH_SIZE_LM_MODEL
                    * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE),
                    :,
                ] = embs_altered
            doc_store.train_index(documents=None, embeddings=p_embs, index=index_name_probing)
            logging.info(f"Training index of doc store for ablation: {index_str} finished.")

        docs = []

        for i, embs in enumerate(emb_batches):
            embs_altered = get_altered_embs(embs)
            docs.extend(
                [
                    Document(
                        id=str(i * BATCH_SIZE_LM_MODEL + j),
                        content="",
                        embedding=embs_altered[j, :].squeeze(0).numpy(),
                    )
                    for j in range(embs_altered.shape[0])
                ]
            )
            if len(docs) >= CHUNK_SIZE:
                doc_store.write_documents(
                    documents=docs, duplicate_documents="skip", index=index_name_probing
                )
                docs = []

        doc_store.write_documents(
            documents=docs, duplicate_documents="skip", index=index_name_probing
        )
        del docs
        logging.info(f"Finished initilization of doc store of {index_str}.")
        faiss_index_file_str = f"./cache/{index_name_probing}_{index_str}.pickle"
        with open(faiss_index_file_str, "wb+") as faiss_index_file:
            pickle.dump(
                doc_store.faiss_indexes[index_name_probing],
                faiss_index_file,
            )

    def _init_probing_doc_stores_ablation_last_layer(self):
        logging.info(
            f"Adding new indices {self.probing_task} of altered embeddings (RLACE/INLP) to 2 document store for the last layer (average and token wise). "
        )
        index_strs = ["average, token_wise"]
        self._cleanup_before_indexing(self.index_name_probing, index_suffixes=index_strs)

        self._init_corpus()
        corpus_sample = self.corpus.sample(n=INDEX_TRAINING_SAMPLE_SIZE, random_state=SEED)  # type: ignore
        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()
        # for training index
        passages_sample = (
            "[D] " + corpus_sample["passage"] if self.prepend_token else corpus_sample["passage"]
        ).tolist()

        iter_params = zip(
            index_strs,
            [self.faiss_index_factory_str] * 2,
            [self.index_name_probing] * 2,
            [self.projection] * 2,
            # TODO: might be too memory expensive:
            # alternative 1 save embeddings in chunks (only last layer) -> average over embeddings and token wise
            # alternative 2 pass corpus so that parallel function can iterate over it
            [
                list(
                    self._iter_corpus_and_get_embeddings_for_layer(
                        BATCH_SIZE_LM_MODEL, passages, LAST_LAYER_IDX, return_numpy=False
                    )
                )
            ]
            * 2,
            [
                list(
                    self._iter_corpus_and_get_embeddings_for_layer(
                        BATCH_SIZE_LM_MODEL, passages_sample, LAST_LAYER_IDX, return_numpy=False
                    )
                )
            ]
            * 2,
        )

        with Pool(CPU_CORES) as pool:
            pool.starmap(self._parallel_indexing_probing_doc_store_ablation_last_layer, iter_params)

        logging.info(f"Reloading doc stores...")
        # reload doc stores from faiss indexes
        self.probing_doc_store_average = self._get_doc_store(
            self.index_name_probing,
            load_from_saved_index=True,
            index_suffix="average",
        )
        self.probing_doc_store_token_wise = self._get_doc_store(
            self.index_name_probing,
            load_from_saved_index=True,
            index_suffix="token_wise",
        )
        logging.info(
            f"Initialized document stores with altered embeddings of task {self.probing_task} for last layer ablation (average, token_wise)."
        )

    def _init_probing_doc_stores_ablation_last_layer_non_parallel(self):
        logging.info(
            f"Adding new indices {self.probing_task} of altered embeddings (RLACE/INLP) to 2 document store for the last layer (average and token wise). "
        )
        index_strs = ["average", "token_wise"]
        self._cleanup_before_indexing(self.index_name_probing, index_suffixes=index_strs)

        def get_altered_embs(embs, index_str):
            if index_str == "average":
                embs_avg = torch.mean(embs[:, 4:, :], dim=-2)
                return torch.einsum("bc,cd->bd", embs_avg, self.projection)
            else:
                embs_altered = torch.einsum("bac,cd->bad", embs, self.projection)
                return torch.mean(embs_altered[:, 4:, :], dim=-2)

        self._init_corpus()
        pids = self.corpus["pid"].tolist()
        corpus_sample = self.corpus.sample(n=INDEX_TRAINING_SAMPLE_SIZE, random_state=SEED)  # type: ignore
        passages = (
            "[D] " + self.corpus["passage"] if self.prepend_token else self.corpus["passage"]
        ).tolist()
        corpus_size = len(passages)
        # for training index
        passages_sample = (
            "[D] " + corpus_sample["passage"] if self.prepend_token else corpus_sample["passage"]
        ).tolist()

        self.probing_doc_store_average = self._get_doc_store(
            self.index_name_probing,
            index_suffix="average",
        )
        self.probing_doc_store_token_wise = self._get_doc_store(
            self.index_name_probing,
            index_suffix="token_wise",
        )
        if self.faiss_index_factory_str != "Flat":
            logging.info(f"Training index of doc store for ablation")
            p_embs = {
                suf: np.zeros((INDEX_TRAINING_SAMPLE_SIZE, EMBEDDING_SIZE), dtype="float32")
                for suf in index_strs
            }
            batches = get_batch_amount(INDEX_TRAINING_SAMPLE_SIZE, BATCH_SIZE_LM_MODEL)
            for i in range(batches):
                batch = passages_sample[
                    BATCH_SIZE_LM_MODEL
                    * i : min(BATCH_SIZE_LM_MODEL * (i + 1), len(passages_sample))
                ]
                # TODO: fix average computation over zero embeddings
                embs = self._get_batch_embeddings_by_forward_pass_for_layer(
                    batch, LAST_LAYER_IDX, return_numpy=False
                )
                for suf in index_strs:
                    embs_altered = get_altered_embs(embs, suf)
                    p_embs[suf][
                        BATCH_SIZE_LM_MODEL
                        * i : min(BATCH_SIZE_LM_MODEL * (i + 1), INDEX_TRAINING_SAMPLE_SIZE),
                        :,
                    ] = embs_altered
            self.probing_doc_store_average.train_index(
                documents=None, embeddings=p_embs["average"], index=self.index_name_probing
            )
            self.probing_doc_store_token_wise.train_index(
                documents=None, embeddings=p_embs["token_wise"], index=self.index_name_probing
            )
            logging.info(f"Training index of doc store for ablation doc stores finished.")

        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        docs = {suf: [] for suf in index_strs}
        for i in range(batches):
            batch = passages[
                BATCH_SIZE_LM_MODEL * i : min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)
            ]
            # TODO: fix average computation over zero embeddings
            embs = self._get_batch_embeddings_by_forward_pass_for_layer(
                batch, LAST_LAYER_IDX, return_numpy=False
            )
            for suf in index_strs:
                embs_altered = get_altered_embs(embs, suf)
                docs[suf].extend(
                    [
                        Document(
                            id=str(i * BATCH_SIZE_LM_MODEL + j),
                            content="",
                            embedding=embs_altered[j, :].squeeze(0).numpy(),
                            meta={
                                "pid": pids[i * BATCH_SIZE_LM_MODEL + j],
                                "vector_id": pids[i * BATCH_SIZE_LM_MODEL + j],
                            },
                        )
                        for j in range(embs_altered.shape[0])
                    ]
                )
            if len(docs[index_strs[0]]) >= CHUNK_SIZE:
                self.probing_doc_store_average.write_documents(
                    documents=docs["average"],
                    duplicate_documents="skip",
                    index=self.index_name_probing,
                )
                self.probing_doc_store_token_wise.write_documents(
                    documents=docs["token_wise"],
                    duplicate_documents="skip",
                    index=self.index_name_probing,
                )
                docs = {suf: [] for suf in index_strs}

        self.probing_doc_store_average.write_documents(
            documents=docs["average"], duplicate_documents="skip", index=self.index_name_probing
        )
        self.probing_doc_store_token_wise.write_documents(
            documents=docs["token_wise"], duplicate_documents="skip", index=self.index_name_probing
        )
        del docs

        self._pickle_dump_index(self.probing_doc_store_average, self.index_name_probing, "average")
        self._pickle_dump_index(
            self.probing_doc_store_token_wise, self.index_name_probing, "token_wise"
        )

        logging.info(
            f"Initialized document stores with altered embeddings of task {self.probing_task} for last layer ablation (average, token_wise)."
        )

    def _init_indexed_probing_faiss_doc_store(self):
        logging.info(
            f"Restoring document store with Faiss index of altered embeddings (task: {self.probing_task}) ..."
        )
        self.probing_doc_store = self._get_doc_store(
            self.index_name_probing,
            load_from_saved_index=True,
        )
        logging.info(
            f"Initialized second docuemnt store with altered embeddings of task {self.probing_task}."
        )

    def _init_elastic_search_service(self):
        self._init_corpus()
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

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_embedding_by_forward_pass(self, sequence: str) -> np.ndarray:
        X_sequence_tokenized = self.tokenizer(
            sequence,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_MODEL_INPUT,  # in pyserini 36 for queries!
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_sequence_tokenized)
        embeddings = self._mean_pooling(
            outputs["last_hidden_state"][:, 4:, :], X_sequence_tokenized["attention_mask"][:, 4:]
        )
        return embeddings.detach().cpu().numpy()

    def _get_query_embedding(self, query: str) -> np.ndarray:
        query = "[Q] " + query if self.prepend_token else query
        return self._get_embedding_by_forward_pass(query)

    def _get_query_embedding_pyserini(self, query: str) -> np.ndarray:
        max_length = 36  # hardcode for now
        inputs = self.tokenizer(
            "[CLS] [Q] " + query + "[MASK]" * max_length,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2).flatten()

    def _get_passage_embedding(self, passage: str) -> np.ndarray:
        passage = "[D] " + passage if self.prepend_token else passage
        return self._get_embedding_by_forward_pass(passage)

    def _get_batch_embeddings_by_forward_pass(
        self, batch: list, layer: Optional[Union[int, bool]] = False, return_numpy: bool = True
    ) -> Union[list[np.ndarray], list[torch.Tensor], np.ndarray, torch.Tensor]:
        X_batch_tokenized = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_MODEL_INPUT,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_batch_tokenized)
        if isinstance(layer, int):
            embeddings = self._mean_pooling(
                outputs.hidden_states[layer][:, 4:, :], X_batch_tokenized["attention_mask"][:, 4:]
            )
        elif isinstance(layer, bool) and layer:
            ret = []
            for layer_nr in range(AMOUNT_LAYERS):
                embeddings = self._mean_pooling(
                    outputs.hidden_states[layer_nr][:, 4:, :],
                    X_batch_tokenized["attention_mask"][:, 4:],
                )
                if return_numpy:
                    ret.append(embeddings.detach().cpu().numpy())
                else:
                    ret.append(embeddings.detach().cpu())
            return ret
        else:
            embeddings = self._mean_pooling(
                outputs.last_hidden_state[:, 4:, :], X_batch_tokenized["attention_mask"][:, 4:]
            )
            if return_numpy:
                return embeddings.detach().cpu().numpy()
        return embeddings.detach().cpu()

    def _get_batch_embeddings_all_layers_by_forward_pass(
        self, batch: list, return_numpy: bool = True
    ) -> Union[list[np.ndarray], list[torch.Tensor]]:
        X_batch_tokenized = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_MODEL_INPUT,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_batch_tokenized)
        ret = []
        for layer in range(12):
            if return_numpy:
                embeddings = outputs.hidden_states[layer].detach().cpu().numpy()
                ret.append(np.average(embeddings[:, 4:, :], axis=-2))
            else:
                embeddings = outputs.hidden_states[layer].detach().cpu()
                ret.append(torch.mean(embeddings[:, 4:, :], dim=-2))
        return ret

    def _get_batch_embeddings_by_forward_pass_for_layer(
        self, batch: list, layer: int, return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        X_batch_tokenized = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_MODEL_INPUT,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**X_batch_tokenized)
        embeddings = outputs.hidden_states[layer].detach().cpu()
        if return_numpy:
            return embeddings.numpy()
        return embeddings

    def _iter_corpus_and_get_embeddings(
        self, batch_size: int, corpus: list, return_numpy: bool = True, all_layers: bool = False
    ):
        """Generator function to iterate over a given corpus and get batches of embeddings"""
        corpus_size = len(corpus)
        batches = get_batch_amount(corpus_size, batch_size)
        i = 0
        while i < batches:
            batch = corpus[batch_size * i : min(batch_size * (i + 1), corpus_size)]
            if all_layers:
                yield self._get_batch_embeddings_all_layers_by_forward_pass(batch, return_numpy)
            else:
                yield self._get_batch_embeddings_by_forward_pass(batch, return_numpy)
            i += 1

    def _iter_corpus_and_get_embeddings_for_layer(
        self, batch_size: int, corpus: list, layer: int, return_numpy: bool = True
    ):
        """Generator function to iterate over a given corpus and get batches of embeddings"""
        corpus_size = len(corpus)
        batches = get_batch_amount(corpus_size, batch_size)
        i = 0
        while i < batches:
            batch = corpus[batch_size * i : min(batch_size * (i + 1), corpus_size)]
            yield self._get_batch_embeddings_by_forward_pass_for_layer(batch, layer, return_numpy)
            i += 1


    def _debug_embedding_score(self):
        query = {
            156493: "do goldfish grow"
        }
        documents = {
            5203821: "D. Liberalism ............................................................................................................................................. 14. E. Constructivism ...................................................................................................................................... 19. F. The English School ...............................................................................................................................",
            2928707: "Goldfish Only Grow to the Size of Their Enclosure. There is an element of truth to this, but it is not as innocent as it sounds and is related more to water quality than tank size. When properly cared for, goldfish will not stop growing. Most fishes are in fact what are known as indeterminate growers.",
        }
        flat_index_score_non_scaled = 37.02694
        flat_index_score_non_scaled_pyserini = 52.790016
        pyserini_index_score_non_scaled_true = 81.19456
        p_emb_false = self._get_passage_embedding(documents[5203821]).flatten()
        p_emb_true = self._get_passage_embedding(documents[2928707]).flatten()
        q_emb = self._get_query_embedding(query[156493]).flatten()
        q_emb_pyserini = self._get_query_embedding_pyserini(query[156493])

        dot1 = np.dot(p_emb_false, q_emb)
        dot2 = np.dot(p_emb_false, q_emb_pyserini)
        dot3 = np.dot(p_emb_true, q_emb)
        dot4 = np.dot(p_emb_true, q_emb_pyserini)

        assert(math.isclose(flat_index_score_non_scaled, dot1, rel_tol=1e-4))
        assert(math.isclose(flat_index_score_non_scaled_pyserini, dot2, rel_tol=1e-4))
        assert(math.isclose(pyserini_index_score_non_scaled_true, dot3, rel_tol=1e-4))
        assert(math.isclose(pyserini_index_score_non_scaled_true, dot4, rel_tol=1e-4))


    def _debug(self):
        queries = get_queries(MSMARCO_TREC_2019_TEST_QUERIES_PATH)
        # top1000 = get_top_1000_passages(MSMARCO_TEST_43_TOP_1000_PATH)
        # self._init_corpus()
        predictions: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.
        predictions_2: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.
        predictions_3: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.

        for i, row in queries.iterrows():
            qid, query = row[0], row[1]
            # try:
            #     top1_relevant_pid = top1000[qid][0]
            #     top1000_relevant_pids = top1000[qid]
            #     pass
            # except:
            #     continue
            # pass

            # pyserini
            query_emb = self._get_query_embedding(query)
            query_emb_pyserini = self._get_query_embedding_pyserini(query)
            y = self.doc_store_pyserini.search(query, k=1000)
            yy = self.doc_store_pyserini.search(
                query_emb_pyserini.reshape(1, len(query_emb_pyserini)), k=1000
            )
            yyy = self.doc_store_pyserini.search(query_emb, k=1000)
            docs_dict = {doc.docid: doc.score for doc in y}
            predictions[str(qid)] = docs_dict  # type: ignore
            docs_dict = {doc.docid: doc.score for doc in yy}
            predictions_2[str(qid)] = docs_dict  # type: ignore
            docs_dict = {doc.docid: doc.score for doc in yyy}
            predictions_3[str(qid)] = docs_dict  # type: ignore
            pass

        self._trec_eval(predictions, index_name="pyserini_1")
        self._trec_eval(predictions_2, index_name="pyserini_2")
        self._trec_eval(predictions_3, index_name="pyserini_3")
        # if isinstance(self.doc_store, FaissSearcher):
        #     relevant_docs = self.doc_store.search(row[1])
        #     pass
        # elif isinstance(self.doc_store, FAISSDocumentStore):
        #     q_emb_np = self._get_query_embedding(query)
        #     q_emb_pt = torch.from_numpy(q_emb_np).unsqueeze(0)
        #     q_emb_probing_pt = q_emb_pt.mm(self.projection)
        #     q_emb_probing_np = q_emb_probing_pt.squeeze(0).numpy()

        #     relevant_docs = self.doc_store.query_by_embedding(
        #         q_emb_np, index=self.index_name, top_k=1000, return_embedding=True
        #     )
        #     relevant_docs_probing = self.probing_doc_store.query_by_embedding(
        #         q_emb_probing_np, index=self.index_name_probing, top_k=10, return_embedding=True
        # )

        # top1_relevant_doc = self.corpus[self.corpus["pid"] == top1_relevant_pid]
        # passage = top1_relevant_doc["passage"].tolist()[0]
        # p_emb_np = self._get_passage_embedding(passage)
        # p_emb_pt = torch.from_numpy(p_emb_np)
        # for j, doc in enumerate(relevant_docs):
        #     if doc.meta["pid"] == top1_relevant_pid or doc.meta["pid"] == 8434617:
        #         emb_from_doc_store_np = doc.embedding
        #         emb_from_doc_store_pt = torch.from_numpy(emb_from_doc_store_np)
        #         all_close = torch.allclose(p_emb_pt, emb_from_doc_store_pt, rtol=1e-04)
        #         pass

    def _evaluate(
        self,
        doc_store: FAISSDocumentStore,
        index_name: str,
        alter_query_emb: bool = False,
        layer: Optional[int] = None,
        index_suffix: Optional[str] = None,
        recall_at: int = 1000,
    ):
        logging.info(f"Evaluating performance of {index_name}.")
        logging.info(f"Number of docs in index: {doc_store.faiss_indexes[index_name].ntotal}")
        queries = get_queries(MSMARCO_TREC_2019_TEST_QUERIES_PATH)
        predictions: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.

        for i, row in queries.iterrows():
            qid = row[0]
            query = row[1]

            # q_emb_np = self._get_query_embedding(query)
            q_emb_np = self._get_query_embedding_pyserini(query)
            if alter_query_emb:
                q_emb_pt = torch.from_numpy(q_emb_np).unsqueeze(0)
                q_emb_np = q_emb_pt.mm(self.projection).squeeze(0).numpy()

            relevant_docs = doc_store.query_by_embedding(
                q_emb_np, index=index_name, top_k=recall_at, return_embedding=True
            )

            docs_dict = {doc.id: doc.score for doc in relevant_docs}
            predictions[str(qid)] = docs_dict  # type: ignore

        self._trec_eval(predictions, index_name, layer, index_suffix)

    def _evaluate_performance_all_layers(self, recall_at: int = 1000):
        logging.info("Evaluating performance ...")
        queries = get_queries(MSMARCO_TREC_2019_TEST_QUERIES_PATH)

        predictions: dict[
            str, dict[str, float]
        ] = {}  # Query IDs mapped to document IDs mapped to scores.
        predictions_probing: list[dict[str, dict[str, float]]] = [
            {} for _ in range(AMOUNT_LAYERS)
        ]  # list of Query IDs mapped to document IDs mapped to scores. (Ordered by layer number)

        for i, row in queries.iterrows():
            qid = row[0]
            query = row[1]

            q_emb_np = self._get_query_embedding(query)
            q_emb_pt = torch.from_numpy(q_emb_np).unsqueeze(0)
            q_emb_probing_pt = q_emb_pt.mm(self.projection)
            q_emb_probing_np = q_emb_probing_pt.squeeze(0).numpy()
            relevant_docs = self.doc_store.query_by_embedding(
                q_emb_np, index=self.index_name, top_k=recall_at, return_embedding=True
            )
            relevant_docs_probing = {
                layer: self.probing_doc_stores[layer].query_by_embedding(
                    q_emb_probing_np,
                    index=self.index_name_probing,
                    top_k=recall_at,
                    return_embedding=True,
                )
                for layer in range(AMOUNT_LAYERS)
            }
            docs_dict = {doc.id: doc.score for doc in relevant_docs}
            predictions[str(qid)] = docs_dict  # type: ignore
            for layer in range(AMOUNT_LAYERS):
                docs_dict = {doc.id: doc.score for doc in relevant_docs_probing[layer]}
                predictions_probing[layer][str(qid)] = docs_dict  # type: ignore

        self._trec_eval(predictions, self.index_name)
        for layer in range(AMOUNT_LAYERS):
            self._trec_eval(predictions_probing[layer], self.index_name_probing, layer)

    def _trec_eval(
        self,
        predictions,
        index_name: str,
        layer: Optional[int] = None,
        index_suffix: Optional[str] = None,
    ):
        combined_str = f"{'_layer_{layer}' if isinstance(layer, int) else ''}{f'_{index_suffix}' if index_suffix else ''}"
        logging.info(f"Starting official TREC evaluation of {index_name}{combined_str}.")
        out_file_str = f"./logs/results/trec_eval_{index_name}_{self.timestamp}{combined_str}.tsv"
        eval_file_str = f"./logs/trec/trec_eval_{index_name}{combined_str}.tsv"
        out_file = Path(out_file_str)
        write_trec_eval_file(Path(eval_file_str), predictions, self.probing_task)
        trec_evaluation(
            out_file, self.model_choice, MSMARCO_QREL_2019_PATH, TREC_EVAL, eval_file_str, 0
        )
        # if Path(eval_file_str).is_file():
        #     os.remove(eval_file_str)
        logging.info(
            f"TREC evaluation of {index_name}{combined_str} done. Logged results at time {self.timestamp}."
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
    cp = CausalProber(**args)
    cp.run()
