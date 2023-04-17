import math
import random

import numpy as np
import torch

## SEED
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

## HYPERPARAMETER
BATCH_SIZE_PROBING_MODEL = 10
BATCH_SIZE_LM_MODEL = 12  # TODO: change to 30 again (GPU memory issues)
EMBEDDING_SIZE = 768
CHUNK_SIZE = int(1e6)
CHUNK_AMOUNT = math.ceil(int(8.8e6) / CHUNK_SIZE)  # 8.8M is the corpus size
INDEX_TRAINING_SAMPLE_SIZE = int(1.5e6)
AMOUNT_LAYERS = 13
LAST_LAYER_IDX = AMOUNT_LAYERS - 1
CPU_CORES = 2
MAX_LENGTH_MODEL_INPUT = 512
NUM_BUCKETS_CLASSIFICATION = 10
PROBE_MODEL_RUNS = 20
SUBSPACE_RANK = 20

## MODELS
MODEL_CHOICES = {
    "tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco",
    "tct_colbert_msmarcov2": "castorini/tct_colbert-v2-hnp-msmarco-r2",
    "tct_colbert_v1": "castorini/tct_colbert-msmarco",
}
