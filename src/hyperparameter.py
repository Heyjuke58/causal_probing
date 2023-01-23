import math

## HYPERPARAMETER
SEED = 12
BATCH_SIZE_PROBING_MODEL = 10
BATCH_SIZE_LM_MODEL = 25
EMBEDDING_SIZE = 768
CHUNK_SIZE = int(1e6)
CHUNK_AMOUNT = math.ceil(int(8.8e6) / CHUNK_SIZE)  # 8.8M is the corpus size
INDEX_TRAINING_SAMPLE_SIZE = int(1.5e6)
AMOUNT_LAYERS = 12
LAST_LAYER_IDX = AMOUNT_LAYERS - 1
CPU_CORES = 2
PREPEND_Q_AND_D_TOKEN = True
MAX_LENGTH_MODEL_INPUT = 512

## MODELS
MODEL_CHOICES = {
    "tct_colbert": "castorini/tct_colbert-v2-hnp-msmarco",
    "tct_colbert_msmarcov2": "castorini/tct_colbert-v2-hnp-msmarco-r2",
    "tct_colbert_v1": "castorini/tct_colbert-msmarco",
}