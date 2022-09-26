from pathlib import Path
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

from rlace import solve_adv_game

DATASET_PATH = Path("./datasets/msmarco_bm25_60000_10_2022_04_08-15-40-06.json")
SEED = 12
BATCH_SIZE = 3
EMBEDDING_SIZE = 768
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


def main():
    with open(DATASET_PATH) as dataset_file:
        dataset = pd.read_json(dataset_file, orient="records")
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=SEED),
            [int(0.6 * len(dataset)), int(0.8 * len(dataset))],
        )
        X_query = np.array([x["query"] for x in train["input"].values])
        X_passage = np.array([x["passage"] for x in train["input"].values])
        y = torch.from_numpy(train["targets"].apply(lambda x: x[0]["label"]).to_numpy()).unsqueeze(
            1
        )
    tokenizer = AutoTokenizer.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")
    model = AutoModel.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco").to(DEVICE)
    if not Path("./src/X.pt").is_file():
        q_embs = torch.empty((len(X_query), EMBEDDING_SIZE))
        p_embs = torch.empty((len(X_passage), EMBEDDING_SIZE))

        for i in range(int(len(X_query) / BATCH_SIZE)):
            print(f"{i} / {int(len(X_query) / BATCH_SIZE)}")
            X_query_tokenized = tokenizer(
                X_query.tolist()[BATCH_SIZE * i : BATCH_SIZE * (i + 1)], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(DEVICE)
            X_passage_tokenized = tokenizer(
                X_passage.tolist()[BATCH_SIZE * i : BATCH_SIZE * (i + 1)], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(DEVICE)
            q_emb = model(**X_query_tokenized).pooler_output.detach().cpu()
            p_emb = model(**X_passage_tokenized).pooler_output.detach().cpu()
            q_embs[BATCH_SIZE * i : BATCH_SIZE * (i + 1), :] = q_emb
            p_embs[BATCH_SIZE * i : BATCH_SIZE * (i + 1), :] = p_emb

        # concat embeddings to get a tensor of len(query) x EMBEDDING_SIZE * 2
        X = torch.cat((q_embs, p_embs), 1)

        torch.save(X, "./src/X.pt")
    else:
        X = torch.load("./src/X.pt")
    # P can be calculated with a closed form, since we are dealing with a linear regression
    # (when we try to linearly predict the BM25 score from the models representation)

    P = torch.eye(EMBEDDING_SIZE * 2, EMBEDDING_SIZE * 2) - (
        (X.t().mm(y)).mm(y.t().mm(X)) / ((y.t().mm(X)).mm(X.t().mm(y)))
    )

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
    main()
