from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProbingTask(Enum):
    BM25 = "bm25"  # BM25
    SEM = "sem"  # Semantic Similarity
    AVG_TI = "avg_ti"  # Average Term Importance
    TI = "ti"  # Term Importance
    BM25_BUCKETIZED = "bm25_bucketized"  # BM25 Bucketized
    SEM_BUCKETIZED = "sem_bucketized"  # Semantic Similarity Bucketized
    AVG_TI_BUCKETIZED = "avg_ti_bucketized"  # Average Term Importance Bucketized
    TI_BUCKETIZED = "ti_bucketized"  # Term Importance Bucketized
    COREF = "coref"  # Coreference Resolution
    NER = "ner"  # Named Entity Recognition
    QC_COARSE = "qc_coarse"  # Question Classification coarse-grained
    QC_FINE = "qc_fine"  # Question Classification fine-grained

    def __str__(self) -> str:
        return self.value


class ProbeModelType(Enum):
    LINEAR = "linear"
    MLP = "mlp"

    def __str__(self) -> str:
        return self.value


@dataclass
class ProbingConfig:
    probing_task: ProbingTask
    probe_model_type: ProbeModelType
    layer: Optional[int]
    rank_subspace: int
    normalize_target: bool
