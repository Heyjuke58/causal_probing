import csv
import logging
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import Normalizer, StandardScaler
from unidecode import unidecode

from src.amnesic_probing import create_rand_dir_from_orth_basis_projection, create_rand_dir_projection
from src.file_locations import DATASET_PATHS
from src.hyperparameter import (
    BATCH_SIZE_LM_MODEL,
    BATCH_SIZE_PROBING_MODEL,
    EMBEDDING_SIZE,
    EPOCHS,
    INITIAL_LR,
    NUM_BUCKETS_CLASSIFICATION,
    PROBE_MODEL_RUNS,
    RERUNS,
    SEED,
)
from src.model import ModelWrapper
from src.nlp_utils import (
    STOPWORDS,
    get_indices_from_span,
    get_indices_from_spans,
    get_spacy_tokenizer,
    preprocess,
    retokenize_span,
    retokenize_spans,
)
from src.probe import TenneyMLP
from src.probing_config import ProbeModelType, ProbingConfig, ProbingTask
from src.rlace import solve_adv_game
from src.utils import get_batch_amount


class Prober:
    def __init__(
        self, config: ProbingConfig, model_wrapper: ModelWrapper, device, debug: bool = False, reconstruction_both: bool = False
    ) -> None:
        self.DATA_PREPROCESSING: dict[ProbingTask, Callable] = {
            ProbingTask.BM25: self._data_preprocessing_regression_task,
            ProbingTask.SEM: self._data_preprocessing_regression_task,
            ProbingTask.AVG_TI: self._data_preprocessing_regression_task,
            ProbingTask.TI: self._data_preprocessing_regression_task_ti,
            ProbingTask.BM25_BUCKETIZED: partial(self._data_preprocessing_bucketized_task, num_buckets=NUM_BUCKETS_CLASSIFICATION),
            ProbingTask.SEM_BUCKETIZED: partial(self._data_preprocessing_bucketized_task, num_buckets=NUM_BUCKETS_CLASSIFICATION),
            ProbingTask.AVG_TI_BUCKETIZED: partial(self._data_preprocessing_bucketized_task, num_buckets=NUM_BUCKETS_CLASSIFICATION),
            ProbingTask.TI_BUCKETIZED: partial(self._data_preprocessing_bucketized_task_ti, num_buckets=NUM_BUCKETS_CLASSIFICATION),
            ProbingTask.COREF: self._data_preprocessing_coref,
            ProbingTask.NER: self._data_preprocessing_ner,
            ProbingTask.QC_COARSE: self._data_preprocessing_qc,
            ProbingTask.QC_FINE: partial(self._data_preprocessing_qc, fine_grained=True),
        }
        self.TASKS_RLACE: dict[ProbingTask, Callable] = {
            ProbingTask.BM25: self.rlace_linear_regression_closed_form,
            ProbingTask.SEM: self.rlace_linear_regression_closed_form,
            ProbingTask.AVG_TI: self.rlace_linear_regression_closed_form,
            ProbingTask.TI: self.rlace_linear_regression_closed_form,
            ProbingTask.BM25_BUCKETIZED: self.rlace,
            ProbingTask.SEM_BUCKETIZED: self.rlace,
            ProbingTask.AVG_TI_BUCKETIZED: self.rlace,
            ProbingTask.TI_BUCKETIZED: self.rlace,
            ProbingTask.COREF: self.rlace,
            ProbingTask.NER: partial(self.rlace, rank=config.rank_subspace),
            ProbingTask.QC_COARSE: partial(self.rlace, rank=config.rank_subspace),
            ProbingTask.QC_FINE: self.rlace,
        }
        self.config: ProbingConfig = config
        self.model_wrapper = model_wrapper
        self.device = device
        if not self.config.probing_task in {ProbingTask.QC_COARSE, ProbingTask.QC_FINE}:
            self.dataset = pd.read_json(DATASET_PATHS[self.config.probing_task], orient="records")
        else:
            self.dataset = pd.read_csv(DATASET_PATHS[self.config.probing_task], sep="\t", header=0)
        self.train, self.test = self._train_test_split()
        layer_str = f"_layer_{self.config.layer}" if type(self.config.layer) == int else ""
        self.normalize_str = f"_normalized_target" if self.config.normalize_target else ""
        self.identification_str = f"{self.config.probing_task}{layer_str}"
        self.logs_dir = "./logs/results/"
        self.debug = debug
        self.reconstruction_both = reconstruction_both

        if self.config.probing_task in {ProbingTask.COREF, ProbingTask.NER}:
            self.spacy_tokenizer = get_spacy_tokenizer()

    def run(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.DATA_PREPROCESSING[self.config.probing_task]()
        self.num_classes = torch.unique(self.y_train).shape[0]

        if self.debug:
            self.projection = torch.rand((EMBEDDING_SIZE, EMBEDDING_SIZE)).to(self.device)
        else:
            self.projection = self.TASKS_RLACE[self.config.probing_task](self.X_train, self.y_train, self.X_test, self.y_test).to(self.device)

    def rlace(self, X: torch.Tensor, y, X_test, y_test, rank: int = 1, subspace_ablation: bool = False, out_iters: int = 50000):
        if self.debug:
            return torch.rand((EMBEDDING_SIZE, EMBEDDING_SIZE))
        # see if cached projection is available
        projection_cache_file = f"./cache/projections/{self.identification_str}_{rank}.pt"
        if Path(projection_cache_file).is_file() and not subspace_ablation:
            P = torch.load(projection_cache_file)
            logging.info(f"Loaded cached projection from {projection_cache_file}")
        else:
            optimizer_class = torch.optim.SGD
            optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}
            optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.0}

            output = solve_adv_game(
                X,
                y,
                X_test,
                y_test,
                rank=rank,
                device=self.device,
                out_iters=out_iters,
                optimizer_class=optimizer_class,
                optimizer_params_P=optimizer_params_P,
                optimizer_params_predictor=optimizer_params_predictor,
                epsilon=0.002,
                batch_size=256,  # was 128 before 29.04.23
            )

            P = torch.from_numpy(output["P"]).float()
            if not subspace_ablation:
                # cache projection
                torch.save(P, projection_cache_file)
                logging.info(f"Saved projection to {projection_cache_file}")

        return P.to(self.device)

    @staticmethod
    def rlace_linear_regression_closed_form(X: torch.Tensor, y: torch.Tensor, X_test, y_test):
        logging.info("Applying RLACE linear regression (closed form).")
        return torch.eye(X.shape[1], X.shape[1]) - ((X.t().mm(y)).mm(y.t().mm(X)) / ((y.t().mm(X)).mm(X.t().mm(y))))

    @staticmethod
    def _standardize_y(y: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        scaler.fit(y)
        return scaler.transform(y)

    @staticmethod
    def _normalize_y(y: np.ndarray) -> np.ndarray:
        normalizer = Normalizer()
        normalizer.fit(y)
        return normalizer.transform(y)

    def _get_y_single_target(self, split, standardize: bool = True):
        y = split["targets"].apply(lambda x: x[0]["label"]).to_numpy().reshape(-1, 1)
        if standardize:
            y = self._standardize_y(y)
        return y.ravel()

    def _get_y_bucketized(self, split, num_buckets: int = NUM_BUCKETS_CLASSIFICATION):
        y = self._get_y_single_target(split)
        return self._bucketize_y(y, num_buckets)

    @staticmethod
    def _bucketize_y(y, num_buckets: int = NUM_BUCKETS_CLASSIFICATION):
        y_between = y[np.where((y <= np.mean(y) + 2 * np.std(y)) & (y >= np.mean(y) - 2 * np.std(y)))]
        boundaries = np.linspace(y_between.min(), y_between.max(), num_buckets)[1:]
        # boundaries = np.linspace(0, 1, num_buckets)[1:]
        return np.digitize(y, boundaries)

    def _y_preprocessing_regression(self, split):
        y = self._get_y_single_target(split).reshape(-1, 1)
        return torch.from_numpy(y).to(torch.float32)

    def _y_preprocessing_bucketized(self, split, num_buckets: int = NUM_BUCKETS_CLASSIFICATION):
        targets = self._get_y_single_target(split, standardize=False)
        y = self._bucketize_y(targets, num_buckets)
        return torch.from_numpy(y).to(torch.float32)

    def _y_preprocessing_ner(self, split):
        labels = []
        spans = []
        entity_texts = []
        for i, row in split.iterrows():
            labels.extend([target["label"] for target in row["targets"]])
            spans.extend([target["span1"] for target in row["targets"]])
            entity_texts.extend([target["text"] for target in row["targets"]])

        y = np.unique(labels, return_inverse=True)
        return torch.from_numpy(y[1]), spans, entity_texts

    def _y_preprocessing_qc(self, split, fine_grained: bool = False):
        labels = split.loc[:, "coarse" if not fine_grained else "fine"].to_numpy()
        y = np.unique(labels, return_inverse=True)
        return torch.from_numpy(y[1])

    def _data_preprocessing_regression_task(self):
        X = self._get_X_without_intervention("train")
        y = self._y_preprocessing_regression(self.train)

        X_test = self._get_X_without_intervention("test")
        y_test = self._y_preprocessing_regression(self.test)

        return X, y, X_test, y_test

    def _data_preprocessing_regression_task_ti(self):
        # decide which terms are selected to be put in the data for training/testing
        # One approach (with sorted list of term importances):
        # 1 important term, 1 unimportant term, 2 term from the middle of the list (quartiles)
        y, self.terms_train = self._trim_ti_dataset_to_get_targets(self.train)
        X = self._get_X_without_intervention_ti("train", self.terms_train)

        y_test, self.terms_test = self._trim_ti_dataset_to_get_targets(self.test)
        X_test = self._get_X_without_intervention_ti("test", self.terms_test)

        return X, y, X_test, y_test

    def _trim_ti_dataset_to_get_targets(self, split):
        split["targets"] = split["targets"].apply(lambda x: sorted(x, key=lambda x: x["label"], reverse=True))

        def is_same_unidecoded(term):
            return term == unidecode(term)

        def find_approriate_token(index, tokens):
            res_index = index
            pos_term = tokens[res_index]["term"]
            while not is_same_unidecoded(pos_term):
                if res_index < 0:
                    res_index -= 1
                else:
                    res_index += 1
                pos_term = tokens[res_index]["term"]

            return res_index, pos_term

        def get_appropriate_terms_and_targets(tokens):
            first_index, term1 = find_approriate_token(0, tokens)
            second_index, term2 = find_approriate_token(int(len(tokens) / 4), tokens)
            third_index, term3 = find_approriate_token(int(len(tokens) / 2), tokens)
            forth_index, term4 = find_approriate_token(-1, tokens)

            return (
                (tokens[first_index]["label"], term1),
                (tokens[second_index]["label"], term2),
                (tokens[third_index]["label"], term3),
                (tokens[forth_index]["label"], term4),
            )

        y_plus_term = split["targets"].apply(get_appropriate_terms_and_targets)

        terms = [(terms[0][1], terms[1][1], terms[2][1], terms[3][1]) for terms in y_plus_term]
        y = np.array([target[0] for subtuple in y_plus_term for target in subtuple]).reshape(-1, 1)
        assert split.shape[0] * 4 == y.shape[0], "Could not get 4 terms per query document pair."
        if self.config.normalize_target:
            y = self._normalize_y(y)
        else:
            y = self._standardize_y(y)
        y = y.ravel().reshape(-1, 1)
        return torch.from_numpy(y).to(torch.float32), terms

    def _data_preprocessing_bucketized_task(self, num_buckets: int = NUM_BUCKETS_CLASSIFICATION):
        X = self._get_X_without_intervention("train")
        y = self._y_preprocessing_bucketized(self.train, num_buckets)

        X_test = self._get_X_without_intervention("test")
        y_test = self._y_preprocessing_bucketized(self.test, num_buckets)

        return X, y, X_test, y_test

    def _data_preprocessing_bucketized_task_ti(self, num_buckets: int = NUM_BUCKETS_CLASSIFICATION):
        y, self.terms_train = self._trim_ti_dataset_to_get_targets(self.train)
        y = y.ravel()
        y = self._bucketize_y(y, num_buckets)
        X = self._get_X_without_intervention_ti("train", self.terms_train)

        y_test, self.terms_test = self._trim_ti_dataset_to_get_targets(self.test)
        y_test = y_test.ravel()
        y_test = self._bucketize_y(y, num_buckets)
        X_test = self._get_X_without_intervention_ti("test", self.terms_test)

        return X, y, X_test, y_test

    def _data_preprocessing_qc(self, fine_grained: bool = False):
        y = self._y_preprocessing_qc(self.train, fine_grained)
        y_test = self._y_preprocessing_qc(self.test, fine_grained)

        X = self._get_X_without_intervention_qc(self.train, "train")
        X_test = self._get_X_without_intervention_qc(self.test, "test")

        return X, y, X_test, y_test

    def _get_X_without_intervention_qc(self, split, split_str):
        X_file_str_orig = f"./cache/probing_task/X_{self.identification_str}_{split_str}.pt"
        if Path(X_file_str_orig).is_file():
            X = torch.load(X_file_str_orig)
            return X

        q_embs = torch.empty((split.shape[0], EMBEDDING_SIZE))
        for i, row in enumerate(split.iterrows()):
            q_emb = self.model_wrapper.get_query_embeddings_pyserini([row[1]["question"]], self.config.layer)
            q_embs[i] = torch.from_numpy(q_emb[0])

        torch.save(q_embs, X_file_str_orig)

        return q_embs

    def _split_preprocessing_qc_with_intervention(self, split, split_str, projection, no_cache: bool = False):
        q_embs = torch.empty((split.shape[0], EMBEDDING_SIZE))
        q_embs_altered = torch.empty((split.shape[0], EMBEDDING_SIZE))
        projection = self.projection if not projection else projection
        for i, row in enumerate(split.iterrows()):
            q_emb = self.model_wrapper.get_query_embeddings_pyserini([row[1]["question"]], self.config.layer)
            q_emb = torch.from_numpy(q_emb[0])
            q_emb_altered = torch.matmul(q_emb, projection.detach().cpu())
            q_embs[i] = q_emb
            q_embs_altered[i] = q_emb_altered

        return q_embs, q_embs_altered

    # yet to implement
    def _data_preprocessing_ner(self):
        y, spans, entitiy_texts = self._y_preprocessing_ner(self.train)
        y_test, spans_test, entity_texts_test = self._y_preprocessing_ner(self.test)

        X = self._get_cached_X_if_it_exists("train")
        if not isinstance(X, torch.Tensor):
            X = self._split_preprocessing_ner(self.train, "train", spans, entitiy_texts)

        X_test = self._get_cached_X_if_it_exists("test")
        if not isinstance(X_test, torch.Tensor):
            X_test = self._split_preprocessing_ner(self.test, "test", spans_test, entity_texts_test)

        return X, y, X_test, y_test

    def _get_ner_embs(self, passage, entity_text, span, passage_embs):
        new_span, correctly_retokenized = retokenize_span(self.model_wrapper, self.spacy_tokenizer, passage, span, entity_text)
        if not correctly_retokenized:
            return
        indices = get_indices_from_span(new_span)

        try:
            ent_embs = torch.mean(torch.index_select(passage_embs, 0, torch.tensor(indices)), dim=0)
            return ent_embs
        except:
            logging.critical(f"{passage} had an error! span: {span} for entity: {entity_text}")

    def _split_preprocessing_ner(self, split, split_str, spans, texts):
        p_embs = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        for i, row in enumerate(split.iterrows()):
            sample = row[1]
            p_emb, _ = self.model_wrapper.get_passage_embeddings_unpooled([sample.text], self.config.layer)
            p_emb = torch.from_numpy(p_emb)[0]
            try:
                p_embs[i * 2, :] = self._get_ner_embs(sample.text, texts[i * 2], spans[i * 2], p_emb)
            except:
                pass
            try:
                p_embs[i * 2 + 1, :] = self._get_ner_embs(sample.text, texts[i * 2 + 1], spans[i * 2 + 1], p_emb)
            except:
                pass
        X_file_str = f"./cache/probing_task/X_{self.identification_str}_{split_str}.pt"
        torch.save(p_embs, X_file_str)
        logging.info(f"Documents processed. Embeddings saved to file {X_file_str}.")

        return p_embs

    def _split_preprocessing_ner_with_intervention(self, split, split_str, projection: Optional[torch.Tensor] = None, no_cache: bool = False):
        X_file_str_orig = f"./cache/probing_task/X_{self.identification_str}_{split_str}.pt"
        X_file_str_altered = f"./cache/probing_task/X_{self.identification_str}_{split_str}_altered.pt"
        projection = self.projection if projection is None else projection

        if Path(X_file_str_orig).is_file() and Path(X_file_str_altered).is_file() and not no_cache:
            X_orig = torch.load(X_file_str_orig)
            X_altered = torch.load(X_file_str_altered)
            logging.info(f"Saved X found locally. Restored X from {X_file_str_orig} and {X_file_str_altered}.")
            return X_orig, X_altered

        _, spans, texts = self._y_preprocessing_ner(split)

        p_embs = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        p_embs_altered = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        for i, row in enumerate(split.iterrows()):
            sample = row[1]
            p_emb, _ = self.model_wrapper.get_passage_embeddings_unpooled([sample.text], self.config.layer)
            p_emb = torch.from_numpy(p_emb)[0]

            ner_emb_1st = self._get_ner_embs(sample.text, texts[i * 2], spans[i * 2], p_emb)
            ner_emb_2nd = self._get_ner_embs(sample.text, texts[i * 2 + 1], spans[i * 2 + 1], p_emb)
            p_embs[i * 2, :] = ner_emb_1st
            p_embs[i * 2 + 1, :] = ner_emb_2nd

            ner_emb_1st_altered = torch.matmul(ner_emb_1st.unsqueeze(0), projection.detach().cpu()).squeeze(0)
            ner_emb_2nd_altered = torch.matmul(ner_emb_2nd.unsqueeze(0), projection.detach().cpu()).squeeze(0)
            p_embs_altered[i * 2, :] = ner_emb_1st_altered
            p_embs_altered[i * 2 + 1, :] = ner_emb_2nd_altered

        if not no_cache:
            torch.save(p_embs, X_file_str_orig)
            torch.save(p_embs_altered, X_file_str_altered)
            logging.info(f"Documents processed for NER. Embeddings saved to file {X_file_str_orig} and {X_file_str_altered}.")
        else:
            logging.info(f"Documents processed for NER. Embeddings not saved to file.")

        return p_embs, p_embs_altered

    def _data_preprocessing_coref(self):
        X = self._get_cached_X_if_it_exists("train")
        if isinstance(X, torch.Tensor):
            y = np.array([1, 0] * len(self.train))
        else:
            X, y = self._split_preprocessing_coref(self.train, "train")

        X_test = self._get_cached_X_if_it_exists("test")
        if isinstance(X_test, torch.Tensor):
            y_test = np.array([1, 0] * len(self.test))
        else:
            X_test, y_test = self._split_preprocessing_coref(self.test, "test")

        return X, y, X_test, y_test

    def _split_preprocessing_coref(self, split, split_str):
        labels = []
        p_embs = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        for i, row in enumerate(split.iterrows()):
            sample = row[1]
            if len(sample[3]) != 2:
                logging.critical(f"{sample[1]} did not have 2 examples, but {len(sample[3])}")
                continue

            # assert len(sample[3]) == 2, "Sample does not contain positive and negative example"
            for j, example in enumerate(sample[3]):
                labels.append(int(example["label"]))
                actual_spans, _, __ = retokenize_spans(
                    self.model_wrapper,
                    self.spacy_tokenizer,
                    sample[1],
                    [example["span1"], example["span2"]],
                    example["text1"],
                    example["text2"],
                )
                indices = get_indices_from_spans(actual_spans)
                p_emb, _ = self.model_wrapper.get_passage_embeddings_unpooled([sample[1]], self.config.layer)
                p_emb = torch.from_numpy(p_emb)[0]
                try:
                    p_embs[i * 2 + j, :] = torch.mean(torch.index_select(p_emb, 0, torch.tensor(indices)), dim=0)
                except:
                    logging.critical(
                        f"{sample[1]} had an error! span1: {example['span1']} span2: {example['span2']} label: {example['label']}"
                    )

        X_file_str = f"./cache/probing_task/X_{self.identification_str}_{split_str}.pt"
        torch.save(p_embs, X_file_str)
        logging.info(f"Documents processed. Embeddings saved to file {X_file_str}.")

        return p_embs, np.array(labels)

    def _split_preprocessing_coref_with_and_without_intervention(self, split, split_str, projection, no_cache=False):
        X_file_str_orig = f"./cache/probing_task/X_{self.identification_str}_{split_str}.pt"
        X_file_str_altered = f"./cache/probing_task/X_{self.identification_str}_{split_str}_altered.pt"
        projection = self.projection if projection is None else projection

        if Path(X_file_str_orig).is_file() and Path(X_file_str_altered).is_file() and not no_cache:
            X_orig = torch.load(X_file_str_orig)
            X_altered = torch.load(X_file_str_altered)
            logging.info(f"Saved X found locally. Restored X from {X_file_str_orig} and {X_file_str_altered}.")
            return X_orig, X_altered

        p_embs = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        p_embs_altered = torch.empty((len(split) * 2, EMBEDDING_SIZE))
        for i, row in enumerate(split.iterrows()):
            sample = row[1]
            assert len(sample[3]) == 2, "Sample does not contain positive and negative example"
            for j, example in enumerate(sample[3]):
                actual_spans, _, __ = retokenize_spans(
                    self.model_wrapper,
                    self.spacy_tokenizer,
                    sample[1],
                    [example["span1"], example["span2"]],
                    example["text1"],
                    example["text2"],
                )
                indices = get_indices_from_spans(actual_spans)

                emb, _ = self.model_wrapper.get_passage_embeddings_unpooled([sample[1]], self.config.layer)
                emb = torch.from_numpy(emb)[0]
                emb = torch.mean(torch.index_select(emb, 0, torch.tensor(indices)), dim=0)
                p_embs[i * 2 + j, :] = emb
                # p_embs_altered[i * 2 + j, :] = torch.einsum("a,ab->a", emb, projection.detach().cpu())
                altered_p_emb = torch.matmul(emb.unsqueeze(0), projection.detach().cpu()).squeeze(0)
                p_embs_altered[i * 2 + j, :] = altered_p_emb

        if not no_cache:
            torch.save(p_embs, X_file_str_orig)
            torch.save(p_embs_altered, X_file_str_altered)
            logging.info(f"Documents processed for coref. Embeddings saved to file {X_file_str_orig} and {X_file_str_altered}.")
        else:
            logging.info(f"Documents processed for coref. Embeddings not saved to file.")

        return p_embs, p_embs_altered

    @staticmethod
    def _get_majority_acc(y):
        if isinstance(y, np.ndarray):
            y = y.tolist()
        c = Counter(y)
        fracts = [v / sum(c.values()) for v in c.values()]
        maj = max(fracts)
        return maj

    def _train_val_test_split(self, ratio_train: float = 0.6, ratio_val_and_test: float = 0.2):
        train, val, test = np.split(
            self.dataset,
            [
                int(ratio_train * len(self.dataset)),
                int((ratio_train + ratio_val_and_test) * len(self.dataset)),
            ],
        )
        train = train.sample(frac=1, random_state=SEED)
        val = val.sample(frac=1, random_state=SEED)
        test = test.sample(frac=1, random_state=SEED)
        return train, val, test

    def _train_test_split(self, ratio_train: float = 0.8):
        train, test, _ = np.split(
            self.dataset,
            [
                int(ratio_train * len(self.dataset)),
                len(self.dataset),
            ],
        )
        train = train.sample(frac=1, random_state=SEED)
        test = test.sample(frac=1, random_state=SEED)
        return train, test

    def _prepare_X_generation(self, split, ti: bool = False):
        if split == "train":
            data = self.train
        elif split == "test":
            data = self.test
        else:
            raise ValueError("Split should be one of [train, val, test].")

        X_query = [x["query"] for x in data["input"].values]  # type: ignore
        X_passage = [x["passage"] for x in data["input"].values]  # type: ignore

        assert len(X_query) == len(X_passage)

        X_size = len(X_query) if not ti else len(X_query) * 4
        X_batch_size = len(X_query)

        q_embs = torch.empty((X_size, EMBEDDING_SIZE))
        p_embs = torch.empty((X_size, EMBEDDING_SIZE))

        batches = get_batch_amount(X_batch_size, BATCH_SIZE_LM_MODEL)

        return X_query, X_passage, q_embs, p_embs, X_batch_size, batches

    def _merge_query_and_passage_embeddings(self, q_embs: torch.Tensor, p_embs: torch.Tensor) -> torch.Tensor:
        """
        Merge query and passage embeddings to construct feature data for training the linear probe.
        Probably can be removed since average seems to be the strategy to go with.
        """
        if self.config.merging_strategy == MergingStrategy.CONCAT:
            # concat embeddings to get a tensor of len(query) x EMBEDDING_SIZE * 2
            X = torch.cat((q_embs, p_embs), 1)
        elif self.config.merging_strategy == MergingStrategy.MULTIPLY_ELEMENTWISE:
            # multiply embeddings elementwise
            X = q_embs * p_embs
        elif self.config.merging_strategy == MergingStrategy.AVERAGE:
            # take average over each embedding dimension
            embs = torch.cat((q_embs.unsqueeze(0), p_embs.unsqueeze(0)), 0)
            X = torch.mean(embs, 0)
        else:
            raise NotImplementedError(f"Merging with strategy {self.config.merging_strategy} not supported.")

        return X

    def _extract_avg_embedding_for_term(self, embs, tokens, term):
        cur_token = ""
        cur_token_split_nr = 1
        resulting_indices = []
        for i, token in enumerate(tokens[4:]):
            token = unidecode(token)  # to convert for example ##° to ##deg

            if token == "[PAD]":
                break

            if not token.isalnum() and not token.startswith("##"):
                cur_token = ""
                continue

            cur_token = cur_token + token if not token.startswith("##") else cur_token + token[2:]
            next_token = tokens[4:][i + 1] if not i == len(tokens[4:]) - 1 else ""
            if next_token.startswith("##"):
                try:
                    next_cur_token = preprocess(cur_token + next_token[2:], stopword_removal=False, toktok=True)[0]
                except IndexError:
                    next_cur_token = cur_token + next_token[2:]
                if next_cur_token == term:
                    for l in range(cur_token_split_nr):
                        resulting_indices.append(i - l + 4)
                    resulting_indices.append(i + 1 + 4)
                    break
                try:
                    next_token = preprocess(next_token[2:], stopword_removal=False, toktok=True)[0]
                except IndexError:
                    next_token = next_token[2:]
                if next_token == term:
                    resulting_indices.append(i + 1 + 4)
                    break
                cur_token_split_nr += 1
                continue
            else:
                try:
                    cur_token_stemmed = preprocess(cur_token, stopword_removal=False, toktok=True)[0]
                except IndexError:
                    cur_token = ""
                    cur_token_split_nr = 1
                    continue
                is_abbreviation = cur_token[:-1] == term and not token.startswith("##")  # isn't, aren't, couldn't, ...
                if cur_token_stemmed == term or is_abbreviation:
                    for l in range(cur_token_split_nr):
                        resulting_indices.append(i - l + 4)  # +4 since we have 4 initial tokens (CLS + [D])
                    break
                else:
                    cur_token = ""
                    cur_token_split_nr = 1

        if not resulting_indices:
            logging.critical(f"tokens: {tokens}, term: {term}")
            return None

        return torch.mean(torch.index_select(embs, 0, torch.tensor(resulting_indices)), dim=0)

    def _get_cached_X_if_it_exists(self, split: str):
        X_file_str = f"./cache/probing_task/X_{self.identification_str}_{split}.pt"

        if Path(X_file_str).is_file():
            X = torch.load(X_file_str)
            logging.info(f"Saved embeddings found locally. Restored embeddings from {X_file_str}.")
            return X

    def _cache_X(self, split: str, X):
        X_file_str = f"./cache/probing_task/X_{self.identification_str}_{split}.pt"
        torch.save(X, X_file_str)
        logging.info(f"All query-document pairs processed. Embeddings saved to file {X_file_str}.")

    def _get_X_without_intervention_ti(self, split: str, terms: List):
        X = self._get_cached_X_if_it_exists(split)

        if isinstance(X, torch.Tensor):
            return X

        logging.info(f"Getting embeddings of query-document pairs of probing task {self.config.probing_task} dataset ({split} split) ...")
        X_query, X_passage, q_embs, p_embs, X_size, batches = self._prepare_X_generation(split, ti=True)
        errors = 0

        for i in range(batches):
            start = BATCH_SIZE_LM_MODEL * i
            end = min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)
            start_res = BATCH_SIZE_LM_MODEL * i * 4

            q_emb = torch.from_numpy(self.model_wrapper.get_query_embeddings_pyserini(X_query[start:end], self.config.layer))
            p_token_embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled(X_passage[start:end], self.config.layer)
            p_token_embs = torch.from_numpy(p_token_embs)
            p_emb = torch.empty((end - start, EMBEDDING_SIZE))
            for j in range(p_emb.shape[0]):
                for k in range(4):
                    index = start_res + j * 4 + k
                    emb = self._extract_avg_embedding_for_term(p_token_embs[j], tokens[j], terms[start + j][k])
                    if not isinstance(emb, torch.Tensor):
                        errors += 1
                        continue
                    q_embs[index] = q_emb[j]
                    p_embs[index] = emb
        logging.info(f"Errors: {errors}")
        X = self._merge_query_and_passage_embeddings(q_embs, p_embs)
        self._cache_X(split, X)

        return X

    def _get_X_without_intervention(self, split: str):
        X = self._get_cached_X_if_it_exists(split)

        if isinstance(X, torch.Tensor):
            return X

        logging.info(f"Getting embeddings of query-document pairs of probing task {self.config.probing_task} dataset ({split} split) ...")
        X_query, X_passage, q_embs, p_embs, X_size, batches = self._prepare_X_generation(split)

        for i in range(batches):
            start = BATCH_SIZE_LM_MODEL * i
            end = min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)

            q_emb = torch.from_numpy(self.model_wrapper.get_query_embeddings_pyserini(X_query[start:end], self.config.layer))
            if not self.config.probing_task in {ProbingTask.AVG_TI, ProbingTask.TI}:
                p_emb = torch.from_numpy(self.model_wrapper.get_passage_embeddings_pyserini(X_passage[start:end], self.config.layer))
            elif self.config.probing_task == ProbingTask.AVG_TI:
                p_token_embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled(X_passage[start:end], self.config.layer)
                p_token_embs = torch.from_numpy(p_token_embs)
                p_emb = torch.empty((end - start, EMBEDDING_SIZE))
                # extract embeddings of non stop words from the passage token embeddings
                for j in range(p_emb.shape[0]):
                    indices_of_stopwords = {0, 1, 2, 3}
                    for k, token in enumerate(tokens[j][4:]):
                        if (token in STOPWORDS or not token.isalnum()) and not token.startswith("##"):
                            indices_of_stopwords.add(k + 4)  # +4 since we have CLS [ D ] at the front
                        if token == "[PAD]":
                            indices_of_stopwords.add(k + 4)

                    all_indices = set(range(len(tokens[j])))
                    indices_non_stopwords = all_indices.difference(indices_of_stopwords)
                    # avg pool over resulting indices
                    p_emb[j, :] = torch.mean(torch.index_select(p_token_embs[j, :], 0, torch.tensor(list(indices_non_stopwords))), dim=0)
            else:
                raise NotImplementedError("Wrong usage of training data generation function.")
            q_embs[start:end, :] = q_emb  # type: ignore
            p_embs[start:end, :] = p_emb  # type: ignore

        X = self._merge_query_and_passage_embeddings(q_embs, p_embs)
        self._cache_X(split, X)

        return X

    def _get_X_with_and_without_intervention_ti(self, split: str, projection: Optional[torch.Tensor] = None, no_cache: bool = False):
        logging.info(
            f"Building X for probing task with and without applied intervention for split {split} with {self.config.merging_strategy} merging strategy."
        )
        X_file_str_orig = f"./cache/probing_task/X_{self.identification_str}_{split}.pt"
        X_file_str_altered = f"./cache/probing_task/X_{self.identification_str}_{split}_altered.pt"
        projection = self.projection if projection is None else projection

        if split == "train":
            terms = self.terms_train
        else:
            terms = self.terms_test

        if Path(X_file_str_orig).is_file() and Path(X_file_str_altered).is_file() and not no_cache:
            X_orig = torch.load(X_file_str_orig)
            X_altered = torch.load(X_file_str_altered)
            logging.info(f"Saved X found locally. Restored X from {X_file_str_orig} and {X_file_str_altered}.")
        else:
            logging.info(f"Getting embeddings of query-document pairs of probing task {self.config.probing_task} dataset ({split} split) ...")
            X_query, X_passage, q_embs, p_embs, X_size, batches = self._prepare_X_generation(split, ti=True)
            q_embs_altered = torch.empty((q_embs.shape[0], EMBEDDING_SIZE))
            p_embs_altered = torch.empty((p_embs.shape[0], EMBEDDING_SIZE))

            for i in range(batches):
                start = BATCH_SIZE_LM_MODEL * i
                end = min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)
                start_res = BATCH_SIZE_LM_MODEL * i * 4

                q_emb, q_emb_altered = self.model_wrapper.get_query_embeddings_pyserini_with_and_without_intervention(
                    X_query[start:end], projection, self.config.layer
                )
                q_emb = torch.from_numpy(q_emb)
                q_emb_altered = torch.from_numpy(q_emb_altered)

                p_token_embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled(X_passage[start:end], self.config.layer)
                p_token_embs = torch.from_numpy(p_token_embs)
                p_emb = torch.empty((end - start, EMBEDDING_SIZE))

                for j in range(p_emb.shape[0]):
                    for k in range(4):
                        index = start_res + j * 4 + k
                        emb = self._extract_avg_embedding_for_term(p_token_embs[j], tokens[j], terms[start + j][k])
                        q_embs[index] = q_emb[j]
                        p_embs[index] = emb
                        q_embs_altered[index] = q_emb_altered[j]
                        altered_p_emb = torch.matmul(emb.unsqueeze(0), projection.detach().cpu()).squeeze(0)
                        p_embs_altered[index] = altered_p_emb

            X_orig = self._merge_query_and_passage_embeddings(q_embs, p_embs)
            X_altered = self._merge_query_and_passage_embeddings(q_embs_altered, p_embs_altered)
            if not no_cache:
                torch.save(X_orig, X_file_str_orig)
                torch.save(X_altered, X_file_str_altered)
                logging.info(f"Building finished. Cached results to {X_file_str_orig} and {X_file_str_altered}.")
            else:
                logging.info(f"Building finished. Did not cache results.")

        return X_orig, X_altered

    def _get_X_with_and_without_intervention(self, split: str, projection: Optional[torch.Tensor] = None, no_cache: bool = False):
        logging.info(
            f"Building X for probing task with and without applied intervention for split {split} with {self.config.merging_strategy} merging strategy."
        )
        X_file_str_orig = f"./cache/probing_task/X_{self.identification_str}_{split}.pt"
        X_file_str_altered = f"./cache/probing_task/X_{self.identification_str}_{split}_altered.pt"
        projection = self.projection if projection is None else projection

        if Path(X_file_str_orig).is_file() and Path(X_file_str_altered).is_file() and not no_cache:
            X_orig = torch.load(X_file_str_orig)
            X_altered = torch.load(X_file_str_altered)
            logging.info(f"Saved X found locally. Restored X from {X_file_str_orig} and {X_file_str_altered}.")
        else:
            X_query, X_passage, q_embs, p_embs, X_size, batches = self._prepare_X_generation(split)
            q_embs_altered = torch.empty((X_size, EMBEDDING_SIZE))
            p_embs_altered = torch.empty((X_size, EMBEDDING_SIZE))

            for i in range(batches):
                start = BATCH_SIZE_LM_MODEL * i
                end = min(BATCH_SIZE_LM_MODEL * (i + 1), X_size)

                q_emb, q_emb_altered = self.model_wrapper.get_query_embeddings_pyserini_with_and_without_intervention(
                    X_query[start:end], projection, self.config.layer
                )
                q_emb = torch.from_numpy(q_emb)
                q_emb_altered = torch.from_numpy(q_emb_altered)

                if not self.config.probing_task in {ProbingTask.AVG_TI, ProbingTask.TI}:
                    p_emb, p_emb_altered = self.model_wrapper.get_passage_embeddings_pyserini_with_and_without_intervention(
                        X_passage[start:end], projection, self.config.layer
                    )
                    p_emb = torch.from_numpy(p_emb)
                    p_emb_altered = torch.from_numpy(p_emb_altered)
                elif self.config.probing_task == ProbingTask.AVG_TI:
                    p_token_embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled(X_passage[start:end], self.config.layer)
                    p_token_embs = torch.from_numpy(p_token_embs)
                    p_emb = torch.empty((end - start, EMBEDDING_SIZE))
                    p_emb_altered = torch.empty((end - start, EMBEDDING_SIZE))
                    # extract embeddings of non stop words from the passage token embeddings
                    for j in range(p_emb.shape[0]):
                        indices_of_stopwords = {0, 1, 2, 3}
                        for k, token in enumerate(tokens[j][4:]):
                            if (token in STOPWORDS or not token.isalnum()) and not token.startswith("##"):
                                indices_of_stopwords.add(k + 4)  # +4 since we have CLS [ D ] at the front
                            if token == "[PAD]":
                                indices_of_stopwords.add(k + 4)

                        all_indices = set(range(len(tokens[j])))
                        indices_non_stopwords = all_indices.difference(indices_of_stopwords)
                        # avg pool over resulting indices
                        p_emb[j, :] = torch.mean(torch.index_select(p_token_embs[j, :], 0, torch.tensor(list(indices_non_stopwords))), dim=0)
                        altered_p_emb = torch.matmul(p_emb[j, :].unsqueeze(0), projection.detach().cpu()).squeeze(0)
                        p_emb_altered[j, :] = altered_p_emb
                        # p_emb_altered[j, :] = torch.einsum("a,ab->a", p_emb[j, :], projection.detach().cpu())
                else:
                    raise NotImplementedError("Wrong usage of training data generation function.")

                q_embs[start:end, :] = q_emb  # type: ignore
                q_embs_altered[start:end, :] = q_emb_altered  # type: ignore
                p_embs[start:end, :] = p_emb  # type: ignore
                p_embs_altered[start:end, :] = p_emb_altered  # type: ignore

            X_orig = self._merge_query_and_passage_embeddings(q_embs, p_embs)
            X_altered = self._merge_query_and_passage_embeddings(q_embs_altered, p_embs_altered)

            if not no_cache:
                torch.save(X_orig, X_file_str_orig)
                torch.save(X_altered, X_file_str_altered)
                logging.info(f"Building finished. Cached results to {X_file_str_orig} and {X_file_str_altered}.")
            else:
                logging.info(f"Building finished. No caching.")

        return X_orig, X_altered

    def train_tenney_mlp(self, probe, X_train, y_train, X_test, y_test) -> float:
        def shuffle_training_data(X, y):
            perm = torch.randperm(X.size(0))
            return X[perm], y[perm]

        # get 10% of training set as validation data
        val_size = int(X_train.size(0) / 10)
        perm = torch.randperm(X_train.size(0))
        idx_val = perm[:val_size]
        idx_train = perm[val_size:]
        X_val = X_train[idx_val]
        y_val = y_train[idx_val]
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        batches_train = get_batch_amount(X_train.size(0), BATCH_SIZE_PROBING_MODEL)
        batches_val = get_batch_amount(X_val.size(0), BATCH_SIZE_PROBING_MODEL)

        best_val_loss = np.inf
        patience = 1

        for epoch in range(EPOCHS):
            if patience >= 10:
                logging.info(f"Early stopping. No improvement in 10 epochs over val loss of {best_val_loss}.")
                break
            logging.info(f"Starting epoch {epoch}.")
            probe.train()
            for i in range(batches_train):
                start = BATCH_SIZE_PROBING_MODEL * i
                end = min(BATCH_SIZE_PROBING_MODEL * (i + 1), X_train.size(0))

                pred = probe(X_train[start:end])
                loss = probe.loss(pred, y_train[start:end])
                loss.backward()
                probe.optimizer.step()
                probe.optimizer.zero_grad()
            X_train, y_train = shuffle_training_data(X_train, y_train)

            logging.info(f"Training epoch {epoch} done. Validating...")
            val_losses = []
            probe.eval()
            for i in range(batches_val):
                start = BATCH_SIZE_PROBING_MODEL * i
                end = min(BATCH_SIZE_PROBING_MODEL * (i + 1), X_val.size(0))

                with torch.no_grad():
                    pred = probe(X_val[start:end])
                    val_losses.append(probe.loss(pred, y_val[start:end]))

            val_loss = np.mean(val_losses, axis=0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 1
            else:
                patience += 1

            best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss

        # evaluate model after training
        batches_test = get_batch_amount(X_test.size(0), BATCH_SIZE_PROBING_MODEL)
        preds = torch.zeros((X_test.size(0)))
        for i in range(batches_test):
            start = BATCH_SIZE_PROBING_MODEL * i
            end = min(BATCH_SIZE_PROBING_MODEL * (i + 1), X_train.size(0))

            with torch.no_grad():
                pred = probe(X_test[start:end])
                preds[start:end] = torch.argmax(pred, dim=1)

        test_acc = float(torch.sum(preds == y_test) / y_test.size(0))

        return test_acc

    #################################################################################################################################
    ### RECONSTRUCTION OF PROPERTY

    def _fit_and_eval_probe(self, probe_cls, probe_kwargs, X_train, y_train, X_test, y_test) -> float:
        probe = probe_cls(**probe_kwargs)
        if probe_cls == TenneyMLP:
            score = self.train_tenney_mlp(probe, X_train, y_train, X_test, y_test)
        else:
            probe.fit(X_train, y_train)
            score = probe.score(X_test, y_test)
        return score

    def _get_probe_class_and_args_classification(self, seed: int = SEED):
        if self.config.probe_model_type == ProbeModelType.LINEAR:
            clf_kwargs = {"loss": "log_loss", "n_jobs": -1, "max_iter": 3000, "random_state": seed, "early_stopping": True}
            clf_model = SGDClassifier
        elif self.config.probe_model_type == ProbeModelType.MLP:
            # clf_kwargs = {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": seed, "early_stopping": True}
            # clf_model = MLPClassifier

            clf_kwargs = {
                "input_dim": EMBEDDING_SIZE,
                "hidden_dim": 256,
                "output_dim": self.num_classes,
                "dropout": 0.3,
                "learning_rate": INITIAL_LR,
            }
            clf_model = TenneyMLP
        else:
            raise NotImplementedError(f"Reconstruction of property not implemented for {self.config.probe_model_type} model type.")
        return clf_model, clf_kwargs

    def _get_probe_class_and_args_regression(self, seed: int = SEED):
        if self.config.probe_model_type == ProbeModelType.LINEAR:
            reg_kwargs = {"random_state": seed, "solver": "svd"}
            reg_model = Ridge
        elif self.config.probe_model_type == ProbeModelType.MLP:
            reg_kwargs = {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": seed, "early_stopping": True}
            reg_model = MLPRegressor
        else:
            raise NotImplementedError(f"Reconstruction of property not implemented for {self.config.probe_model_type} model type.")
        return reg_model, reg_kwargs

    def _get_probe_classes_and_args(self, seed: int = SEED):
        reg_model, reg_kwargs = self._get_probe_class_and_args_regression(seed)
        clf_model, clf_kwargs = self._get_probe_class_and_args_classification(seed)
        return reg_model, reg_kwargs, clf_model, clf_kwargs

    def _get_X_for_probes(self, rank: int = 1, projection: Optional[torch.Tensor] = None, no_cache: bool = False):
        if self.config.probing_task in {ProbingTask.TI, ProbingTask.TI_BUCKETIZED}:
            X_orig_train, X_altered_train = self._get_X_with_and_without_intervention_ti("train", projection, no_cache)
            X_orig_test, X_altered_test = self._get_X_with_and_without_intervention_ti("test", projection, no_cache)
        elif self.config.probing_task == ProbingTask.COREF:
            X_orig_train, X_altered_train = self._split_preprocessing_coref_with_and_without_intervention(
                self.train, "train", projection, no_cache
            )
            X_orig_test, X_altered_test = self._split_preprocessing_coref_with_and_without_intervention(
                self.test, "test", projection, no_cache
            )
        elif self.config.probing_task == ProbingTask.NER:
            X_orig_train, X_altered_train = self._split_preprocessing_ner_with_intervention(self.train, "train", projection, no_cache)
            X_orig_test, X_altered_test = self._split_preprocessing_ner_with_intervention(self.test, "test", projection, no_cache)
        elif self.config.probing_task in {ProbingTask.QC_COARSE, ProbingTask.QC_FINE}:
            X_orig_train, X_altered_train = self._split_preprocessing_qc_with_intervention(self.train, "train", projection, no_cache)
            X_orig_test, X_altered_test = self._split_preprocessing_qc_with_intervention(self.test, "test", projection, no_cache)
        else:
            X_orig_train, X_altered_train = self._get_X_with_and_without_intervention("train", projection, no_cache)
            X_orig_test, X_altered_test = self._get_X_with_and_without_intervention("test", projection, no_cache)

        # creating feature data for control over information ablation
        rand_dir_projection = torch.from_numpy(create_rand_dir_from_orth_basis_projection(X_orig_train, rank)).to(torch.float32)
        X_control_train = torch.einsum("bc,cd->bd", X_orig_train, rand_dir_projection)
        X_control_test = torch.einsum("bc,cd->bd", X_orig_test, rand_dir_projection)

        return X_orig_train, X_altered_train, X_control_train, X_orig_test, X_altered_test, X_control_test

    def _get_X_for_probes_subspace_ablation(self, rank, projection):
        X_file_str_orig_train = f"./cache/probing_task/X_{self.identification_str}_train.pt"
        X_file_str_orig_test = f"./cache/probing_task/X_{self.identification_str}_test.pt"
        if Path(X_file_str_orig_train).is_file() and Path(X_file_str_orig_test).is_file():
            X_orig_train = torch.load(X_file_str_orig_train)
            X_orig_test = torch.load(X_file_str_orig_test)
        else:
            raise Exception("X_orig_train and X_orig_test not found. Please run without subspace ablation first.")

        X_altered_train = torch.einsum("bc,cd->bd", X_orig_train, projection.detach().cpu())
        X_altered_test = torch.einsum("bc,cd->bd", X_orig_test, projection.detach().cpu())

        # creating feature data for control over information ablation
        rand_dir_projection = torch.from_numpy(create_rand_dir_from_orth_basis_projection(X_orig_train, rank)).to(torch.float32)
        X_control_train = torch.einsum("bc,cd->bd", X_orig_train, rand_dir_projection)
        X_control_test = torch.einsum("bc,cd->bd", X_orig_test, rand_dir_projection)

        return X_altered_train, X_control_train, X_altered_test, X_control_test

    def _get_X_for_probes_subspace_ablation_control_only(self, rank):
        X_file_str_orig_train = f"./cache/probing_task/X_{self.identification_str}_train.pt"
        X_file_str_orig_test = f"./cache/probing_task/X_{self.identification_str}_test.pt"
        if Path(X_file_str_orig_train).is_file() and Path(X_file_str_orig_test).is_file():
            X_orig_train = torch.load(X_file_str_orig_train)
            X_orig_test = torch.load(X_file_str_orig_test)
        else:
            raise Exception("X_orig_train and X_orig_test not found. Please run without subspace ablation first.")

        # creating feature data for control over information ablation
        rand_dir_projection = torch.from_numpy(create_rand_dir_from_orth_basis_projection(X_orig_train, rank)).to(torch.float32)
        X_control_train = torch.einsum("bc,cd->bd", X_orig_train, rand_dir_projection)
        X_control_test = torch.einsum("bc,cd->bd", X_orig_test, rand_dir_projection)

        return X_control_train, X_control_test

    def _random_init_X_train_test(self):
        if self.config.probing_task in {ProbingTask.COREF, ProbingTask.NER}:
            train_len = len(self.train) * 2
            test_len = len(self.test) * 2
        else:
            train_len = len(self.train)
            test_len = len(self.test)
        X_train = torch.rand((train_len, EMBEDDING_SIZE)) - 0.5
        X_test = torch.rand((test_len, EMBEDDING_SIZE)) - 0.5
        return X_train, X_test

    def _cache_baseline(self, identification: str, score: float) -> None:
        with open(self.logs_dir + "ablation/" + str(self.config.probing_task) + "/" + identification + "_baseline.log", "w+") as f:
            f.write(f"{score:.3f}")

    def _regressor_baseline(self, reg_cls, reg_kwargs: dict) -> float:
        reg_kwargs.update({"random_state": random.randint(0, int(1e5))})

        X_train, X_test = self._random_init_X_train_test()
        y_train = self._get_y_single_target(self.train)
        y_test = self._get_y_single_target(self.test)

        return self._fit_and_eval_probe(reg_cls, reg_kwargs, X_train, y_train, X_test, y_test)

    def _regressor_baseline_linear(self) -> None:
        r2s = [self._regressor_baseline(Ridge, {}) for _ in range(PROBE_MODEL_RUNS)]
        self._cache_baseline(f"{self.config.probing_task}_regressor_linear{self.normalize_str}", float(np.average(r2s)))

    def _regressor_baseline_mlp(self) -> None:
        reg_kwargs = {"hidden_layer_sizes": (100,), "max_iter": 3000, "early_stopping": True}
        r2s = [self._regressor_baseline(MLPRegressor, reg_kwargs) for _ in range(PROBE_MODEL_RUNS)]
        self._cache_baseline(f"{self.config.probing_task}_regressor_mlp{self.normalize_str}", float(np.average(r2s)))

    def _classifcation_baseline(self, clf_cls, clf_kwargs: dict) -> float:
        clf_kwargs.update({"random_state": random.randint(0, int(1e5))})

        X_train, X_test = self._random_init_X_train_test()
        if self.config.probing_task == ProbingTask.COREF:
            y_train = np.array([1, 0] * len(self.train))
            y_test = np.array([1, 0] * len(self.test))
        elif self.config.probing_task == ProbingTask.NER:
            y_train, _, __ = self._y_preprocessing_ner(self.train)
            y_test, _, __ = self._y_preprocessing_ner(self.test)
        elif self.config.probing_task == ProbingTask.QC_COARSE:
            y_train = self._y_preprocessing_qc(self.train)
            y_test = self._y_preprocessing_qc(self.test)
        elif self.config.probing_task == ProbingTask.QC_FINE:
            y_train = self._y_preprocessing_qc(self.train, True)
            y_test = self._y_preprocessing_qc(self.test, True)
        else:
            y_train = self._get_y_bucketized(self.train)
            y_test = self._get_y_bucketized(self.test)

        return self._fit_and_eval_probe(clf_cls, clf_kwargs, X_train, y_train, X_test, y_test)

    def _classification_baseline_linear(self) -> None:
        clf_kwargs = {"loss": "log_loss", "n_jobs": -1, "max_iter": 3000, "early_stopping": True}
        accs = [self._classifcation_baseline(SGDClassifier, clf_kwargs) for _ in range(PROBE_MODEL_RUNS)]
        self._cache_baseline(f"{self.config.probing_task}_classification_linear{self.normalize_str}", float(np.average(accs)))

    def _classification_baseline_mlp(self) -> None:
        clf_kwargs = {"hidden_layer_sizes": (100,), "max_iter": 3000, "early_stopping": True}
        accs = [self._classifcation_baseline(MLPClassifier, clf_kwargs) for _ in range(PROBE_MODEL_RUNS)]
        self._cache_baseline(f"{self.config.probing_task}_classification_mlp{self.normalize_str}", float(np.average(accs)))

    def reconstruction(self, multiple_runs: bool = False) -> None:
        if self.reconstruction_both:
            reconstruction: dict[ProbingTask, Callable] = {
                ProbingTask.BM25: partial(self.reconstruct_property_both, multiple_runs=multiple_runs),
                ProbingTask.SEM: partial(self.reconstruct_property_both, multiple_runs=multiple_runs),
                ProbingTask.AVG_TI: partial(self.reconstruct_property_both, multiple_runs=multiple_runs),
                ProbingTask.TI: partial(self.reconstruct_property_both, multiple_runs=multiple_runs),
            }
            reconstruction[self.config.probing_task]()
        else:
            reconstruction: dict[ProbingTask, Callable] = {
                ProbingTask.BM25: partial(self.reconstruct_property_regression, multiple_runs=multiple_runs),
                ProbingTask.SEM: partial(self.reconstruct_property_regression, multiple_runs=multiple_runs),
                ProbingTask.AVG_TI: partial(self.reconstruct_property_regression, multiple_runs=multiple_runs),
                ProbingTask.TI: partial(self.reconstruct_property_regression, multiple_runs=multiple_runs),
                ProbingTask.BM25_BUCKETIZED: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.SEM_BUCKETIZED: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.AVG_TI_BUCKETIZED: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.TI_BUCKETIZED: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.COREF: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.NER: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.QC_COARSE: partial(self.reconstruct_property_classification, multiple_runs=multiple_runs),
                ProbingTask.QC_FINE: self.reconstruct_property_classification,
            }
            reconstruction[self.config.probing_task]()

    def reconstruct_property_both(self, multiple_runs: bool = False) -> None:
        """Attempt to predict property from embeddings, which should not contain the property anymore."""
        assert isinstance(self.projection, torch.Tensor), "Probing task should have run before attempting to reconstruct property!"

        logging.info("Calculating baseline performances for reconstruction from random encodings.")
        # self._classification_baseline_linear()
        # self._classification_baseline_mlp()
        # self._regressor_baseline_linear()
        # self._regressor_baseline_mlp()

        seed = SEED
        nr_of_runs = 1
        if multiple_runs:
            nr_of_runs = RERUNS

        for i in range(nr_of_runs):
            reg_model, reg_kwargs, clf_model, clf_kwargs = self._get_probe_classes_and_args(seed)
            seed += 1

            logging.info(f"Attempting recosntruction of property from original, probed and control encodings. {i + 1}/{nr_of_runs}")

            X_orig_train, X_altered_train, X_control_train, X_orig_test, X_altered_test, X_control_test = self._get_X_for_probes(
                no_cache=multiple_runs
            )
            if self.config.probing_task == ProbingTask.TI:
                y_train, _ = self._trim_ti_dataset_to_get_targets(self.train)
                y_test, _ = self._trim_ti_dataset_to_get_targets(self.test)
            else:
                y_train = self._get_y_single_target(self.train)
                y_train_bucketized = self._get_y_single_target(self.train, standardize=False)
                y_test = self._get_y_single_target(self.test)
                y_test_bucketized = self._get_y_single_target(self.test, standardize=False)

            y_train_bucketized = self._bucketize_y(y_train_bucketized)
            y_test_bucketized = self._bucketize_y(y_test_bucketized)

            logging.info(f"Fitting and evaluating regressors and classifiers.")
            r2_orig = self._fit_and_eval_probe(reg_model, reg_kwargs, X_orig_train, y_train, X_orig_test, y_test)
            r2_probed = self._fit_and_eval_probe(reg_model, reg_kwargs, X_altered_train, y_train, X_altered_test, y_test)
            r2_control = self._fit_and_eval_probe(reg_model, reg_kwargs, X_control_train, y_train, X_control_test, y_test)

            acc_orig = self._fit_and_eval_probe(clf_model, clf_kwargs, X_orig_train, y_train_bucketized, X_orig_test, y_test_bucketized)
            acc_probed = self._fit_and_eval_probe(
                clf_model, clf_kwargs, X_altered_train, y_train_bucketized, X_altered_test, y_test_bucketized
            )
            acc_control = self._fit_and_eval_probe(
                clf_model, clf_kwargs, X_control_train, y_train_bucketized, X_control_test, y_test_bucketized
            )

            # maj_acc_train = self._get_majority_acc(y_train_bucketized)
            # maj_acc_test = self._get_majority_acc(y_test_bucketized)

            file_str = (
                self.logs_dir
                + "ablation/"
                + str(self.config.probing_task)
                + "/"
                + self.identification_str
                + self.normalize_str
                + "_reconstruction_both.log"
            )
            with open(file_str, "a+") as f:
                f.write(f"r2_orig\tr2_probed\tr2_control\tr2_diff\tacc_orig\tacc_probed\tacc_control\tacc_diff\tmaj_acc\tmodel_type\n")
                f.write(
                    f"{r2_orig:.3f}\t{r2_probed:.3f}\t{r2_control:.3f}\t{(r2_probed - r2_orig):.3f}\t"
                    + f"{acc_orig:.3f}\t{acc_probed:.3f}\t{acc_control:.3f}\t{(acc_orig - acc_probed):.3f}\t"
                    + f"{0}\t{self.config.probe_model_type}\n"
                )
            logging.info(f"Saved results to {file_str}")

    def reconstruct_property_classification(self, multiple_runs=False):
        # if self.config.probe_model_type == ProbeModelType.LINEAR:
        #     self._classification_baseline_linear()
        # elif self.config.probe_model_type == ProbeModelType.MLP:
        #     self._classification_baseline_mlp()

        seed = SEED
        nr_of_runs = 1
        if multiple_runs:
            nr_of_runs = RERUNS

        for i in range(nr_of_runs):
            clf_model, clf_kwargs = self._get_probe_class_and_args_classification(seed)
            seed += 1

            X_orig_train, X_altered_train, X_control_train, X_orig_test, X_altered_test, X_control_test = self._get_X_for_probes(
                rank=self.config.rank_subspace, no_cache=multiple_runs
            )
            if self.config.probing_task == ProbingTask.TI_BUCKETIZED:
                y_train, _ = self._trim_ti_dataset_to_get_targets(self.train)
                y_train = self._bucketize_y(y_train)
                y_test, _ = self._trim_ti_dataset_to_get_targets(self.test)
                y_test = self._bucketize_y(y_test)
            elif self.config.probing_task == ProbingTask.COREF:
                y_train = np.array([1, 0] * len(self.train))
                y_test = np.array([1, 0] * len(self.test))
            elif self.config.probing_task == ProbingTask.NER:
                y_train, spans_train, texts_train = self._y_preprocessing_ner(self.train)
                y_test, spans_test, texts_test = self._y_preprocessing_ner(self.test)
            elif self.config.probing_task == ProbingTask.QC_COARSE:
                y_train = self._y_preprocessing_qc(self.train)
                y_test = self._y_preprocessing_qc(self.test)
            else:
                y_train = self._get_y_bucketized(self.train)
                y_test = self._get_y_bucketized(self.test)

            acc_orig = self._fit_and_eval_probe(clf_model, clf_kwargs, X_orig_train, y_train, X_orig_test, y_test)
            acc_probed = self._fit_and_eval_probe(clf_model, clf_kwargs, X_altered_train, y_train, X_altered_test, y_test)
            acc_control = self._fit_and_eval_probe(clf_model, clf_kwargs, X_control_train, y_train, X_control_test, y_test)

            file_str = (
                self.logs_dir
                + "ablation/"
                + str(self.config.probing_task)
                + "/"
                + self.identification_str
                + self.normalize_str
                + "_reconstruction_clf.log"
            )
            with open(file_str, "a+") as f:
                f.write("acc_orig\tacc_probed\tacc_control\tmodel_type\n")
                f.write(f"{acc_orig:.3f}\t{acc_probed:.3f}\t{acc_control:.3f}\t{self.config.probe_model_type}\n")

    def reconstruct_property_regression(self, multiple_runs=False):
        if self.config.probe_model_type == ProbeModelType.LINEAR:
            self._regressor_baseline_linear()
        elif self.config.probe_model_type == ProbeModelType.MLP:
            self._regressor_baseline_mlp()

        seed = SEED
        nr_of_runs = 1
        if multiple_runs:
            nr_of_runs = RERUNS

        for i in range(nr_of_runs):
            reg_model, reg_kwargs = self._get_probe_class_and_args_regression(seed)
            seed += 1

            X_orig_train, X_altered_train, X_control_train, X_orig_test, X_altered_test, X_control_test = self._get_X_for_probes(
                no_cache=False  # multiple_runs before test if sgd regressor shoudl be used
            )
            if self.config.probing_task == ProbingTask.TI:
                y_train, _ = self._trim_ti_dataset_to_get_targets(self.train)
                y_test, _ = self._trim_ti_dataset_to_get_targets(self.test)
            else:
                y_train = self._get_y_single_target(self.train)
                y_test = self._get_y_single_target(self.test)

            r2_orig = self._fit_and_eval_probe(reg_model, reg_kwargs, X_orig_train, y_train, X_orig_test, y_test)
            r2_probed = self._fit_and_eval_probe(reg_model, reg_kwargs, X_altered_train, y_train, X_altered_test, y_test)
            r2_control = self._fit_and_eval_probe(reg_model, reg_kwargs, X_control_train, y_train, X_control_test, y_test)

            file_str = (
                self.logs_dir
                + "ablation/"
                + str(self.config.probing_task)
                + "/"
                + self.identification_str
                + self.normalize_str
                + "_reconstruction_reg.log"
            )
            with open(file_str, "a+") as f:
                f.write("r2_orig\tr2_probed\tr2_control\tmodel_type\n")
                f.write(f"{r2_orig:.3f}\t{r2_probed:.3f}\t{r2_control:.3f}\t{self.config.probe_model_type}\n")

    def determine_subspace_rank(self, control_only: bool = False):
        X, y, X_test, y_test = self.DATA_PREPROCESSING[self.config.probing_task]()
        accs_probed = []
        accs_control = []
        for i in list(np.logspace(0, 2.8, num=10, dtype=int)):  # [1, 2, 4, 8, 17, 35, 73, 150, 308, 630]
            # Test removal of multiple ranks for which layer? all layers for now
            clf_model, clf_kwargs = self._get_probe_class_and_args_classification()

            if not control_only:
                projection = self.rlace(X, y, X_test, y_test, rank=i, subspace_ablation=True, out_iters=75000)

                X_altered_train, X_control_train, X_altered_test, X_control_test = self._get_X_for_probes_subspace_ablation(
                    rank=i, projection=projection
                )

            else:
                X_control_train, X_control_test = self._get_X_for_probes_subspace_ablation_control_only(rank=i)

            if self.config.probing_task == ProbingTask.NER:
                y_train, _, __ = self._y_preprocessing_ner(self.train)
                y_test, _, __ = self._y_preprocessing_ner(self.test)
            elif self.config.probing_task == ProbingTask.QC_COARSE:
                y_train = self._y_preprocessing_qc(self.train, False)
                y_test = self._y_preprocessing_qc(self.test, False)
            else:
                y_train = self._get_y_bucketized(self.train)
                y_test = self._get_y_bucketized(self.test)

            if not control_only:
                accs_probed.append(self._fit_and_eval_probe(clf_model, clf_kwargs, X_altered_train, y_train, X_altered_test, y_test))
            accs_control.append(self._fit_and_eval_probe(clf_model, clf_kwargs, X_control_train, y_train, X_control_test, y_test))

        file_str = self.logs_dir + "ablation/subspace/" + self.identification_str + "_subspace_rank.log"
        with open(file_str, "a+") as f:
            tsv_out = csv.writer(f, delimiter="\t")
            if not control_only:
                tsv_out.writerow(accs_probed + [self.config.layer, "probed", self.config.probe_model_type])
            tsv_out.writerow(accs_control + [self.config.layer, "control", self.config.probe_model_type])


#################################################################################################################################
