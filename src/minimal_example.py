from pathlib import Path
import logging
import torch
import pandas as pd
import numpy as np
import pickle
import math
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document

from utils import get_corpus, get_qrels, get_queries
from argument_parser import parse_arguments

from rlace import solve_adv_game

DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
# MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection_sample_100.tsv")
MSMARCO_TEST_QUERIES_PATH = Path("./assets/msmarco/queries.dev.tsv")
MSMARCO_TEST_QRELS_PATH = Path("./assets/msmarco/qrels.dev.tsv")
SEED = 12
BATCH_SIZE = 3
EMBEDDING_SIZE = 768


class MinimalExample:
    def __init__(
        self, model_choice: str = "tct_colbert", probing_task: str = "bm25", reindex: bool = False
    ) -> None:
        self.TASKS = {"bm25": self._bm25_probing_task}
        MODEL_CHOICES = {"tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco"}
        self.model_choice_hf_str = MODEL_CHOICES[model_choice]
        self.model_choice = model_choice
        self.probing_task = probing_task
        self.dataset = pd.read_json(DATASET_PATH, orient="records")

        # index names for doc store
        self.index_name = self.model_choice
        self.index_name_probing = f"{self.model_choice}_{self.probing_task}"

        self.train, self.val, self.test = self._train_val_test_split()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # init model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_choice_hf_str)
        self.model = AutoModel.from_pretrained(self.model_choice_hf_str).to(self.device)
        # run probing task
        self.projection = self.TASKS[self.probing_task]()  # projection to remove a concept
        # init document store
        self.document_store = self._init_faiss_document_store(reindex=reindex)

        self._evaluate_performance()

    def _get_X_probing_task(
        self, X_query: np.ndarray, X_passage: np.ndarray, batch_size: int, merging: str
    ):
        merging_options = {"concat", "multiply_elementwise"}
        if merging not in merging_options:
            raise NotImplementedError(f"Merging option must be one of {merging_options}")
        X_file_str = f"./src/X_{self.probing_task}_{merging}.pt"
        if not Path(X_file_str).is_file():
            q_embs = torch.empty((len(X_query), EMBEDDING_SIZE))
            p_embs = torch.empty((len(X_passage), EMBEDDING_SIZE))

            for i in range(int(len(X_query) / batch_size)):
                print(f"{i} / {int(len(X_query) / batch_size)}")
                X_query_tokenized = self.tokenizer(
                    X_query.tolist()[batch_size * i : batch_size * (i + 1)],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                X_passage_tokenized = self.tokenizer(
                    X_passage.tolist()[batch_size * i : batch_size * (i + 1)],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                q_emb = self.model(**X_query_tokenized).pooler_output.detach().cpu()
                p_emb = self.model(**X_passage_tokenized).pooler_output.detach().cpu()
                q_embs[batch_size * i : batch_size * (i + 1), :] = q_emb
                p_embs[batch_size * i : batch_size * (i + 1), :] = p_emb

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
        else:
            X = torch.load(X_file_str)

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
        return torch.eye(X.shape[1], X.shape[1]) - (
            (X.t().mm(y)).mm(y.t().mm(X)) / ((y.t().mm(X)).mm(X.t().mm(y)))
        )

    def _bm25_probing_task(self):
        X_query = np.array([x["query"] for x in self.train["input"].values])
        X_passage = np.array([x["passage"] for x in self.train["input"].values])
        X = self._get_X_probing_task(X_query, X_passage, 1, "multiply_elementwise")
        y = torch.from_numpy(self.train["targets"].apply(lambda x: x[0]["label"]).to_numpy())
        y = y.to(torch.float32).unsqueeze(1)

        P = self.rlace_linear_regression_closed_form(X, y)
        return P

    def _init_faiss_document_store(self, reindex=False, batch_size=3):
        if reindex:
            document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat", index=self.index_name, duplicate_documents="skip"
            )
            corpus_df = get_corpus(MSMARCO_CORPUS_PATH)
            passages = corpus_df["passage"].tolist()
            pids = corpus_df["pid"].tolist()
            docs = []
            docs_probing = []

            batches = (
                math.floor(len(corpus_df) / batch_size) + 1
                if not (len(corpus_df) / batch_size).is_integer()
                else int(len(corpus_df) / batch_size)
            )

            for i in range(batches):
                logging.info(f"{i + 1} / {batches}")
                X_passage_tokenized = self.tokenizer(
                    passages[batch_size * i : min((batch_size * (i + 1)), len(corpus_df))],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                embs = (
                    self.model(**X_passage_tokenized).pooler_output.detach().cpu()
                )  # maybe prepend '[D]'?
                for j in range(embs.shape[0]):
                    emb_probing = (
                        embs[j, :].unsqueeze(0).mm(self.projection).squeeze(0)
                    )  # apply P to the embedding to remove a concept
                    emb = embs[j, :]
                    docs.append(
                        Document(
                            content=passages[i * batch_size + j],
                            embedding=emb,
                            meta={"pid": pids[i * batch_size + j]},
                        )
                    )
                    docs_probing.append(
                        Document(
                            content=passages[i * batch_size + j],
                            embedding=emb_probing,
                            meta={"pid": pids[i * batch_size + j]},
                        )
                    )

            document_store.write_documents(
                documents=docs, duplicate_documents="skip", index=self.index_name, batch_size=300
            )
            document_store.write_documents(
                documents=docs_probing,
                duplicate_documents="skip",
                index=self.index_name_probing,
                batch_size=300,
            )
            with open(f"./{self.index_name}.pickle", "wb+") as faiss_index_file:
                pickle.dump(document_store.faiss_indexes[self.index_name], faiss_index_file)
            with open(f"./{self.index_name_probing}.pickle", "wb+") as faiss_index_file:
                pickle.dump(document_store.faiss_indexes[self.index_name_probing], faiss_index_file)
        else:
            document_store = None
            with open(f"./{self.index_name}.pickle", "rb") as faiss_index_file:
                faiss_index = pickle.load(faiss_index_file)
                document_store = FAISSDocumentStore(
                    faiss_index_factory_str="Flat",
                    index=self.index_name,
                    duplicate_documents="skip",
                    faiss_index=faiss_index,
                )
            # check whether two indicies are possible in document store!
            with open(f"./{self.index_name_probing}.pickle", "rb") as faiss_index_file:
                faiss_index = pickle.load(faiss_index_file)

        return document_store

    # def _add_index_to_doc_store(self, projection):
    #     index_name = f"{self.model_choice}{'_' + self.probing_task if projection else ''}"

    #     self.document_store.write_documents(
    #         documents=docs, duplicate_documents="skip", index=index_name, batch_size=300
    #     )

    def _evaluate_performance(self):
        queries = get_queries(MSMARCO_TEST_QUERIES_PATH)
        qrels = get_qrels(MSMARCO_TEST_QRELS_PATH)
        reciprocal_ranks = []
        reciprocal_ranks_probing = []

        # get query embedding
        for i, row in enumerate(queries.iterrows()):
            logging.debug(f"Evaluating query {i + 1}/{len(queries)}")
            # TODO: what should happen if there is no corresponding qrel to a query? For now: Skip this query
            try:
                relevant_pid = qrels[qrels["qid"] == row[1][0]]["pid"].values[0]
            except:
                continue
            X_query_tokenized = self.tokenizer(
                row[1][1],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            q_emb = self.model(**X_query_tokenized).pooler_output.detach().cpu()
            q_emb_probing = q_emb.mm(self.projection)
            q_emb = q_emb.squeeze(0).numpy()
            q_emb_probing = q_emb_probing.squeeze(0).numpy()
            relevant_docs = self.document_store.query_by_embedding(q_emb, index="tct_colbert")
            relevant_docs_probing = self.document_store.query_by_embedding(q_emb_probing, index="tct_colbert")

            # calculating MRR@10
            reciprocal_rank = 0
            reciprocal_rank_probing = 0
            for j, (doc1, doc2) in enumerate(zip(relevant_docs, relevant_docs_probing)):
                if (
                    doc1.meta["vector_id"] == relevant_pid
                ):  # vectore_id is a counter which corresponds to pid
                    reciprocal_rank = (j + 1) / len(relevant_docs)
                if doc2.meta["vector_id"] == relevant_pid:
                    reciprocal_rank_probing = (j + 1) / len(relevant_docs)
            reciprocal_ranks.append(reciprocal_rank)
            reciprocal_ranks_probing.append(reciprocal_rank_probing)

        mrr = np.mean(reciprocal_ranks)
        mrr_probing = np.mean(reciprocal_ranks_probing)

        return mrr, mrr_probing


def main(args):
    model_choice = args.models.split(",")[0]
    probing_task = args.tasks.split(",")[0]
    min_ex = MinimalExample(model_choice, probing_task, args.reindex)

    # P can be calculated with a closed form, since we are dealing with a linear regression
    # (when we try to linearly predict the BM25 score from the models representation)

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
    logging.getLogger().setLevel(logging.DEBUG)
    args = parse_arguments()
    main(args)
