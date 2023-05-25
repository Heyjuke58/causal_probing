import logging
import re
import sys
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch

from src.amnesic_probing import create_rand_dir_from_orth_basis_projection, create_rand_dir_projection
from src.argument_parser import parse_arguments_intervention
from src.evaluate import evaluate
from src.file_locations import *
from src.hyperparameter import BATCH_SIZE_LM_MODEL, EMBEDDING_SIZE, LAST_LAYER_IDX, MODEL_CHOICES
from src.model import ModelWrapper
from src.nlp_utils import (
    PUNCTUATIONS,
    STOPWORDS,
    get_indices_from_span,
    get_spacy_pipeline,
    get_spacy_pipeline_with_neuralcoref,
    get_spacy_tokenizer,
    retokenize_span,
    retokenize_spans,
)
from src.probing import Prober
from src.probing_config import MergingStrategy, ProbeModelType, ProbingConfig, ProbingTask, PropertyRemoval
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
        alter_query_embedding: bool = False,
        simple_projection: bool = False,
        property_removal: PropertyRemoval = PropertyRemoval.RLACE,
        probe_model_type: ProbeModelType = ProbeModelType.LINEAR,
        merging_strategy: MergingStrategy = MergingStrategy.AVERAGE,
        eliminated_subspace_rank: int = 1,
        multiple_runs: bool = False,
        reconstruction_both: bool = False,  # only used for regression task that get bucketized
        control_only: bool = False,  # only conduct control experiment
    ) -> None:
        self.device = get_device(device_cpu)

        # Run options
        self.layer = layer
        self.ablation = ablation
        self.alter_query_embedding = alter_query_embedding
        self.simple_projection = simple_projection
        layer_str = f"_layer_{layer}" if type(layer) == int else ""
        suffix_str = f"_{ablation}" if ablation else ""
        not_simple_projection_str = "_not_simple_projection" if not simple_projection else ""
        self.identification_str = f"{model_choice}_{probing_task}{layer_str}{suffix_str}{not_simple_projection_str}"

        self.corpus = None

        # Model
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_wrapper = ModelWrapper(model_choice, self.device, layer)

        # Probing task
        self.config: ProbingConfig = ProbingConfig(
            probing_task,
            property_removal,
            merging_strategy,
            probe_model_type,
            layer,
            eliminated_subspace_rank,
            normalize_target=False,
        )
        self.prober = Prober(self.config, self.model_wrapper, self.device, debug, reconstruction_both)
        self.projection: torch.Tensor
        self.control_only = control_only
        self.multiple_runs = multiple_runs

        if probing_task == ProbingTask.COREF and not self.simple_projection:
            self.spacy_tokenizer = get_spacy_tokenizer()
            self.nlp_coref = get_spacy_pipeline_with_neuralcoref()

        if probing_task == ProbingTask.NER and not self.simple_projection:
            self.spacy_tokenizer = get_spacy_tokenizer()
            self.nlp = get_spacy_pipeline()

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
            eval_str = f"{self.identification_str}{'_altered_query_embeddings' if self.alter_query_embedding else ''}"
            probing_task = self.prober.config.probing_task
            if self.ablation == "control":
                eval_str = f"control_{self.config.layer}{'_altered_query_embeddings' if self.alter_query_embedding else ''}_{self.config.rank_subspace}"
                probing_task = None
            evaluate(
                self.model_wrapper,
                index,
                get_timestamp(),
                self.alter_query_embedding,
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
                if (self.simple_projection and not self.prober.config.probing_task == ProbingTask.QC_COARSE) or self.ablation == "control":
                    embs = self.model_wrapper.get_passage_embeddings_pyserini_with_intervention_at_layer(
                        passages[start:end], self.projection, self.layer
                    )
                elif self.prober.config.probing_task == ProbingTask.COREF:
                    embs = [self._get_passage_embeddings_coref(passage) for passage in passages[start:end]]
                    embs = np.asarray(embs)
                elif self.prober.config.probing_task in {ProbingTask.TI, ProbingTask.AVG_TI}:
                    embs = [self._get_passsage_embeddings_ti(passage) for passage in passages[start:end]]
                    embs = np.asarray(embs)
                elif self.prober.config.probing_task == ProbingTask.NER:
                    embs = [self._get_passsage_embeddings_ner(passage) for passage in passages[start:end]]
                    embs = np.asarray(embs)
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

    def _get_passage_embeddings_coref(self, passage):
        passage = re.sub(r"£(\d+)", r"£ \1", passage)
        embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        tokens = tokens[0]  # method is usually used for a batch of passages, but here we only have one

        target_indices = []  # indices of tokens that are part of a coreference and should be projected
        passage_has_coref = False
        passage_had_error = False
        doc = self.nlp_coref(passage)

        for cluster in doc._.coref_clusters:
            passage_has_coref = True
            self.absolute_coref_count += 1
            bert_span, correctly_rejoined_main_ref = retokenize_span(
                self.model_wrapper, self.spacy_tokenizer, passage, [cluster.main.start, cluster.main.end], cluster.main.text
            )
            if not correctly_rejoined_main_ref:
                self.tokenization_error_count += 1
                passage_had_error = True
                continue
            else:
                target_indices.extend(get_indices_from_span(bert_span))

            for mention in cluster.mentions[1:]:
                bert_span, correctly_rejoined_ref = retokenize_span(
                    self.model_wrapper,
                    self.spacy_tokenizer,
                    passage,
                    [mention.start, mention.end],
                    mention.text,
                )
                if not correctly_rejoined_ref:
                    self.tokenization_error_count += 1
                    passage_had_error = True
                    break
                else:
                    target_indices.extend(get_indices_from_span(bert_span))

        if passage_has_coref:
            self.passages_with_corefs_count += 1

        # error in passage or no coref in passage
        if passage_had_error or not target_indices or not passage_has_coref:
            return np.mean(embs[0, 4:], axis=0)

        target_indices = list(dict.fromkeys(target_indices))  # remove duplicate indices (can happen when entities overlap)
        inverted_target_indices = [i for i in range(len(tokens[0])) if i not in target_indices and i not in list(range(4))]

        embs = torch.from_numpy(embs)[0]
        try:
            embs_coref = torch.mean(torch.index_select(embs, 0, torch.tensor(target_indices)), dim=0)
        except:
            logging.critical(f"Index_select error with target_indicese: {passage}, {target_indices}, {len(tokens[0])}")
            if not passage_had_error:
                self.tokenization_error_count += 1
            return torch.mean(embs[4:], dim=0).numpy()
        altered_embs_coref = torch.matmul(embs_coref.unsqueeze(0), self.projection.detach().cpu()).squeeze(0)
        altered_embs_coref = altered_embs_coref.repeat(len(target_indices), 1)

        if inverted_target_indices:
            try:
                embs_non_coref = torch.index_select(embs, 0, torch.tensor(inverted_target_indices))
            except:
                logging.critical(f"Index_select error with inverted_target_indicese: {passage}, {inverted_target_indices}, {len(tokens[0])}")
                if not passage_had_error:
                    self.tokenization_error_count += 1
                return torch.mean(embs[4:], dim=0).numpy()

            embs_concatted = torch.cat((altered_embs_coref, embs_non_coref), dim=0)
        else:
            # in case the whole passage consists of corefs
            embs_concatted = altered_embs_coref

        return torch.mean(embs_concatted, dim=0).numpy()

    def _get_passsage_embeddings_ti(self, passage):
        embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        embs = embs[0]
        tokens = tokens[0]

        # get indices for stopwords and punctuation tokens
        stopword_punctuation_indices = [
            i for i, token in enumerate(tokens) if (token in STOPWORDS or token in PUNCTUATIONS) and not i in list(range(4))
        ]
        target_indices = [i for i in range(len(tokens)) if i not in stopword_punctuation_indices and i not in list(range(4))]
        if not target_indices:
            return np.mean(embs[4:], axis=0)
        embs = torch.from_numpy(embs)
        try:
            embs_non_stopwords_punctuations = torch.mean(torch.index_select(embs, 0, torch.tensor(target_indices)), dim=0)
        except:
            logging.critical(f"Index_select error with target_indicese: {passage}, {target_indices}, {len(tokens)}")
            return torch.mean(embs[4:], dim=0).numpy()
        altered_embs_non_stopwords_punctuations = torch.matmul(
            embs_non_stopwords_punctuations.unsqueeze(0), self.projection.detach().cpu()
        ).squeeze(0)
        altered_embs_non_stopwords_punctuations = altered_embs_non_stopwords_punctuations.repeat(len(target_indices), 1)

        if stopword_punctuation_indices:
            try:
                embs_stopwords_punctuations = torch.index_select(embs, 0, torch.tensor(stopword_punctuation_indices))
            except:
                logging.critical(
                    f"Index_select error with stopword_punctuation_indices: {passage}, {stopword_punctuation_indices}, {len(tokens)}"
                )
                return torch.mean(embs[4:], dim=0).numpy()

            embs_concatted = torch.cat((altered_embs_non_stopwords_punctuations, embs_stopwords_punctuations), dim=0)
        else:
            # in case the whole passage consists of important tokens
            embs_concatted = altered_embs_non_stopwords_punctuations

        return torch.mean(embs_concatted, dim=0).numpy()

    def _get_passsage_embeddings_ner(self, passage):
        passage = re.sub(r"£(\d+)", r"£ \1", passage)
        embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        embs = embs[0]
        tokens = tokens[0]
        passage_has_entity = False

        doc = self.nlp(passage)
        target_indices = []

        for ent in doc.ents:
            passage_has_entity = True
            self.total_ner_count += 1

            bert_span, correctly_tokenized = retokenize_span(
                self.model_wrapper, self.spacy_tokenizer, passage, [ent.start, ent.end], ent.text
            )
            if not correctly_tokenized:
                self.ner_tokenization_error_count += 1
                continue
            target_indices.extend(get_indices_from_span(bert_span))

        if passage_has_entity:
            self.passages_with_ner_count += 1

        if not target_indices:
            return np.mean(embs[4:], axis=0)

        inverted_target_indices = [i for i in range(len(tokens)) if i not in target_indices and i not in list(range(4))]
        embs = torch.from_numpy(embs)
        try:
            ner_embs = torch.mean(torch.index_select(embs, 0, torch.tensor(bert_span)), dim=0)
        except:
            logging.critical(f"Index_select error with target_indicese: {passage}, {target_indices}, {len(tokens)}")
            return torch.mean(embs[4:], dim=0).numpy()
        altered_ner_embs = torch.matmul(ner_embs.unsqueeze(0), self.projection.detach().cpu()).squeeze(0)
        altered_ner_embs = altered_ner_embs.repeat(len(target_indices), 1)

        if inverted_target_indices:
            try:
                embs_non_ner = torch.index_select(embs, 0, torch.tensor(inverted_target_indices))
            except:
                logging.critical(f"Index_select error with inverted_target_indicese: {passage}, {inverted_target_indices}, {len(tokens[0])}")
                return torch.mean(embs[4:], dim=0).numpy()

            embs_concatted = torch.cat((altered_ner_embs, embs_non_ner), dim=0)
        else:
            # in case the whole passage consists of ner
            embs_concatted = altered_ner_embs

        return torch.mean(embs_concatted, dim=0).numpy()


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
