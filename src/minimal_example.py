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


def main():
    with open(DATASET_PATH) as dataset_file:
        dataset = pd.read_json(dataset_file, orient="records")
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=SEED),
            [int(0.6 * len(dataset)), int(0.8 * len(dataset))],
        )
        X_text = train["text"].to_numpy()
        y = train["targets"].apply(lambda x: x[0]["label"]).to_numpy()
    tokenizer = AutoTokenizer.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")
    model = AutoModel.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")
    outputs = torch.empty((len(X_text), EMBEDDING_SIZE))
    for i in range(int(len(X_text) / BATCH_SIZE)):
        X_tokenized = tokenizer(
            X_text.tolist()[BATCH_SIZE * i : BATCH_SIZE * (i + 1)],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        output = model(**X_tokenized)
        outputs[BATCH_SIZE * i : BATCH_SIZE * (i + 1), :] = output.pooler_output

    X = outputs
    Ps_rlace, accs_rlace = {}, {}
    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}
    optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}

    output = solve_adv_game(
        X,
        y,
        X,
        y,
        rank=1,
        device="cpu",
        out_iters=50000,
        optimizer_class=optimizer_class,
        optimizer_params_P=optimizer_params_P,
        optimizer_params_predictor=optimizer_params_predictor,
        epsilon=0.002,
        batch_size=128,
    )


if __name__ == "__main__":
    main()
