import logging
import sys
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch

from src.amnesic_probing import create_rand_dir_from_orth_basis_projection
from src.argument_parser import parse_arguments_intervention
from src.evaluate import evaluate
from src.file_locations import *
from src.hyperparameter import BATCH_SIZE_LM_MODEL, EMBEDDING_SIZE, MODEL_CHOICES
from src.model import ModelWrapper
from src.probing import Prober
from src.probing_config import ProbeModelType, ProbingConfig, ProbingTask
from src.utils import get_batch_amount, get_corpus, get_device, get_timestamp


class CausalProber:
    def __init__(
        self,
        probing_task: ProbingTask,
        model_choice: str = "tct_colbert",
        layer: Optional[int] = None,
        device_cpu: bool = False,
        debug: bool = False,
        ablation: Optional[str] = None,
        probe_model_type: ProbeModelType = ProbeModelType.LINEAR,
        eliminated_subspace_rank: int = 1,
        multiple_runs: bool = False,
        reconstruction_both: bool = False,  # only used for regression task that get bucketized
        control_only: bool = False,  # only conduct control experiment
    ) -> None:
        self.device = get_device(device_cpu)

        # Run options
        self.layer = layer
        self.ablation = ablation
        layer_str = f"_layer_{layer}" if type(layer) == int else ""
        suffix_str = f"_{ablation}" if ablation else ""
        self.identification_str = f"{model_choice}_{probing_task}{layer_str}{suffix_str}"

        self.corpus = None

        # Model
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_wrapper = ModelWrapper(model_choice, self.device, layer)

        # Probing task
        self.config: ProbingConfig = ProbingConfig(
            probing_task,
            probe_model_type,
            layer,
            eliminated_subspace_rank,
            normalize_target=False,
        )
        self.prober = Prober(self.config, self.model_wrapper, self.device, debug, reconstruction_both)
        self.projection: torch.Tensor
        self.control_only = control_only
        self.multiple_runs = multiple_runs

        # Debugging
        self.debug = debug
        if debug:
            global MSMARCO_CORPUS_PATH
            global MSMARCO_TREC_2019_TEST_QUERIES_PATH
            MSMARCO_CORPUS_PATH = MSMARCO_TOY_CORPUS_PATH
            # MSMARCO_TREC_2019_TEST_QUERIES_PATH = MSMARCO_TOY_QUERIES_PATH

        # Coref specific
        self.passages_with_corefs_count = 0
        self.absolute_coref_count = 0
        self.tokenization_error_count = 0

        # NER specific
        self.passages_with_ner_count = 0
        self.total_ner_count = 0
        self.ner_tokenization_error_count = 0

    def run(self):
        if self.ablation not in {"control", "subspace_rank"}:
            self.prober.run()
            self.projection = self.prober.projection
        elif self.ablation == "control":
            self.projection = self._get_control_projection()

        if self.ablation == "reconstruct_property":
            self.prober.reconstruction(self.multiple_runs)
        elif self.ablation == "subspace_rank":
            self.prober.determine_subspace_rank(self.control_only)
        else:
            index = self._get_index()
            eval_str = f"{self.identification_str}"
            probing_task = self.prober.config.probing_task
            if self.ablation == "control":
                eval_str = f"control_{self.config.layer}_{self.config.rank_subspace}"
                probing_task = None
            evaluate(
                self.model_wrapper,
                index,
                get_timestamp(),
                self.layer,
                probing_task=probing_task,
                eval_str=eval_str,
                projection=self.projection,
            )

    def _get_control_projection(self):
        control_projection_file_str = f"./cache/projections/control_projection_{self.config.layer}_{self.config.rank_subspace}.pt"
        if Path(control_projection_file_str).is_file():
            projection = torch.load(control_projection_file_str)
            logging.info(f"Control projection read from file {control_projection_file_str}.")
        else:
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
            random_passages = self.corpus.sample(50000)
            rand_size, batches, passages, pids = self._pepare_corpus_iteration(random_passages)
            X = np.zeros((rand_size, EMBEDDING_SIZE))

            for i in range(batches):
                start = BATCH_SIZE_LM_MODEL * i
                end = min(BATCH_SIZE_LM_MODEL * (i + 1), rand_size)
                embs = self.model_wrapper.get_passage_embeddings_pyserini(passages[start:end], self.layer)
                X[start:end] = embs

            projection = torch.from_numpy(create_rand_dir_from_orth_basis_projection(X, self.config.rank_subspace)).to(torch.float32)
            torch.save(projection, control_projection_file_str)
            logging.info(f"Control projection saved to file {control_projection_file_str}.")
        return projection.to(self.device)

    def _get_index(self):
        if self.prober.config.probing_task in {ProbingTask.QC_COARSE, ProbingTask.QC_FINE}:
            # no interventions on passage needed in this task
            index_file_str = f"./cache/reproduction/faiss_index.bin"
        elif self.ablation == "control":
            index_file_str = f"./cache/indexes/control_index_{self.config.rank_subspace}.bin"
            index = self._make_index(index_file_str, cache=False)
            return index
        else:
            index_file_str = f"./cache/indexes/{self.identification_str}.bin"
        if Path(index_file_str).is_file():
            index = faiss.read_index(index_file_str)
            logging.info(f"Index read from file {index_file_str}.")
        else:
            if not isinstance(self.corpus, pd.DataFrame):
                self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
            index = self._make_index(index_file_str)

        return index

    def _make_index(self, index_file_str, cache: bool = True):
        logging.info(f"Making index for {self.identification_str}.")
        index = faiss.IndexIDMap2(faiss.index_factory(EMBEDDING_SIZE, "Flat", faiss.METRIC_INNER_PRODUCT))
        corpus_size, batches, passages, pids = self._pepare_corpus_iteration()

        for i in range(batches):
            start = BATCH_SIZE_LM_MODEL * i
            end = min(BATCH_SIZE_LM_MODEL * (i + 1), corpus_size)
            embs = None
            if self.ablation == "average":
                embs = self.model_wrapper.get_passage_embeddings_pyserini_with_intervention(passages[start:end], self.projection)
            elif isinstance(self.layer, int):
                if (not self.prober.config.probing_task == ProbingTask.QC_COARSE) or self.ablation == "control":
                    embs = self.model_wrapper.get_passage_embeddings_pyserini_with_intervention_at_layer(
                        passages[start:end], self.projection, self.layer
                    )
                elif self.prober.config.probing_task == ProbingTask.QC_COARSE:
                    embs = self.model_wrapper.get_passage_embeddings_pyserini(passages[start:end], self.layer)
                else:
                    raise ValueError(f"Task not implemented for search.")
            else:
                raise ValueError(
                    f"Layer is {self.layer}. Specify differently" if not self.layer else f"ablation {self.ablation} not implemented."
                )

            index.add_with_ids(embs, pids[start:end])
        logging.info(f"Index made.")
        if cache:
            logging.info(f"Saving to file...")
            faiss.write_index(index, index_file_str)
            logging.info(f"Index saved to file: {index_file_str}")
        if self.prober.config.probing_task == ProbingTask.COREF and not self.simple_projection:
            logging.info(f"Passages with corefs: {self.passages_with_corefs_count}")
            logging.info(f"Absolute coref count: {self.absolute_coref_count}")
            logging.info(f"Retokenization errors: {self.tokenization_error_count}")

        return index

    def _pepare_corpus_iteration(self, corpus=None):
        if corpus is None:
            corpus = self.corpus
        corpus_size = len(corpus)
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        passages = corpus["passage"].tolist()
        pids = corpus["pid"].to_numpy()

        return corpus_size, batches, passages, pids


if __name__ == "__main__":
    args = parse_arguments_intervention()

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logging_level = logging.DEBUG
    if not args.debug:
        logging_level = logging.INFO
        file_handler = logging.FileHandler(f"./logs/console/{args.model_choice}_{get_timestamp()}.log")
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
