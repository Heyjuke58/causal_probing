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

from utils import get_corpus
from argument_parser import parse_arguments

from rlace import solve_adv_game

DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
# MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection_sample_100.tsv")
SEED = 12
batch_size = 3
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

        self.train, self.val, self.test = self._train_val_test_split()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_choice_hf_str)
        self.model = AutoModel.from_pretrained(self.model_choice_hf_str).to(self.device)
        self.document_store = self._init_faiss_document_store(reindex=reindex)

    def run(self):
        self.TASKS[self.probing_task]()

    def _get_X_probing_task(self, X_query: np.ndarray, X_passage: np.ndarray, batch_size: int):
        X_file_str = f"./src/X_{self.probing_task}.pt"
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

            # concat embeddings to get a tensor of len(query) x EMBEDDING_SIZE * 2
            # maybe try other possibilities!
            X = torch.cat((q_embs, p_embs), 1)

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
        X = self._get_X_probing_task(X_query, X_passage, 1)
        y = torch.from_numpy(self.train["targets"].apply(lambda x: x[0]["label"]).to_numpy())
        y = y.to(torch.float32).unsqueeze(1)

        P = self.rlace_linear_regression_closed_form(X, y)
        return P

    def _init_faiss_document_store(
        self, projection: Optional[torch.Tensor] = None, reindex=False, batch_size=3
    ):
        index_name = f"{self.model_choice}{'_' + self.probing_task if projection else ''}"
        if reindex:
            document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat", index=index_name, duplicate_documents="skip"
            )
            corpus_df = get_corpus(MSMARCO_CORPUS_PATH)
            passages = corpus_df["passage"].tolist()
            pids = corpus_df["pid"].tolist()
            docs = []

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
                    # apply P to the embedding to remove a concept
                    if projection:
                        emb = embs[j, :].mm(projection)
                    else:
                        emb = embs[j, :]
                    docs.append(Document(content=passages[i * batch_size + j], embedding=emb, meta={"pid": pids[i * batch_size + j]}))

            document_store.write_documents(
                documents=docs, duplicate_documents="skip", index=index_name, batch_size=300
            )
            with open(f"./{index_name}.pickle", "wb+") as faiss_index_file:
                pickle.dump(document_store.faiss_indexes[index_name], faiss_index_file)
        else:
            with open(f"./{index_name}.pickle", "rb") as faiss_index_file:
                faiss_index = pickle.load(faiss_index_file)
                document_store = FAISSDocumentStore(
                    faiss_index_factory_str="Flat",
                    index=index_name,
                    duplicate_documents="skip",
                    faiss_index=faiss_index,
                )

        return document_store


def main(args):
    model_choice = args.models.split(",")[0]
    probing_task = args.tasks.split(",")[0]
    min_ex = MinimalExample(model_choice, probing_task, args.reindex)
    min_ex.run()

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
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    main(args)
