from pathlib import Path
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.schema import Document

from utils import get_corpus
from argument_parser import parse_arguments

from rlace import solve_adv_game

DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
# MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection.tsv")
MSMARCO_CORPUS_PATH = Path("./assets/msmarco/collection_sample.tsv")
SEED = 12
batch_size = 3
EMBEDDING_SIZE = 768


class MinimalExample:
    def __init__(
        self, model_choice: str = "tct_colbert", probing_task: str = "bm25", reindex: bool = False
    ) -> None:
        self.TASKS = {"bm25": self._bm25_probing_task}
        MODEL_CHOICES = {"tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco"}
        self.model_choice = MODEL_CHOICES[model_choice]
        self.probing_task = probing_task
        self.dataset = pd.read_json(DATASET_PATH, orient="records")

        self.train, self.val, self.test = self._train_val_test_split()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_choice)
        self.model = AutoModel.from_pretrained(self.model_choice).to(self.device)
        if reindex:
            self.init_faiss_document_store()

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

    def rlace_linear_regression_closed_form(self, X: torch.Tensor, y: torch.Tensor):
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

    def init_faiss_document_store(self):
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
        corpus_df = get_corpus(MSMARCO_CORPUS_PATH)

        def get_document(row) -> Document:
            X_passage_tokenized = self.tokenizer(
                row[1][1], padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            emb = self.model(**X_passage_tokenized).pooler_output.detach().cpu() # eventuell '[D]' prependen?
            doc = Document(content=row[1][1], embedding=emb, meta={"pid": row[1][0]})
            return doc
        # iterate over passages to index them
        passages = [get_document(row) for row in corpus_df.iterrows()]

        document_store.write_documents(documents=passages)

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
    args = parse_arguments()
    main(args)
