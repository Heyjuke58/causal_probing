import logging
import re
import sys
from typing import Optional

import faiss
import numpy as np
import torch

from src.amnesic_probing import create_rand_dir_projection
from src.argument_parser import parse_arguments_intervention
from src.evaluate import evaluate
from src.file_locations import *
from src.hyperparameter import BATCH_SIZE_LM_MODEL, EMBEDDING_SIZE, LAST_LAYER_IDX, MODEL_CHOICES
from src.model import ModelWrapper
from src.nlp_utils import get_indices_from_span, get_spacy_pipeline, get_spacy_tokenizer, retokenize_spans
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
        property_removal: PropertyRemoval = PropertyRemoval.RLACE,
        probe_model_type: ProbeModelType = ProbeModelType.LINEAR,
        merging_strategy: MergingStrategy = MergingStrategy.AVERAGE,
    ) -> None:
        self.device = get_device(device_cpu)

        # Run options
        self.layer = layer
        self.ablation = ablation
        self.alter_query_embedding = alter_query_embedding
        layer_str = f"_layer_{layer}" if type(layer) == int else ""
        suffix_str = f"_{ablation}" if ablation else ""
        self.identification_str = f"{model_choice}_{probing_task}{layer_str}{suffix_str}"

        # Model
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        self.model_wrapper = ModelWrapper(model_choice, self.device, layer)

        # Probing task
        config = ProbingConfig(probing_task, property_removal, merging_strategy, probe_model_type, layer, normalize_target=False)
        self.prober = Prober(config, self.model_wrapper, self.device, debug)
        self.projection: torch.Tensor

        if probing_task == ProbingTask.COREF:
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

    def run(self):
        if self.ablation not in {"control", "subspace_rank"}:
            self.prober.run()
            self.projection = self.prober.projection
        else:
            self.projection = torch.from_numpy(create_rand_dir_projection(EMBEDDING_SIZE, 1)).to(torch.float32).to(self.device)

        # if self.debug:
        #     self._debug_part_of_bert_enconder()

        if self.ablation == "reconstruct_property":
            self.prober.reconstruction()
        elif self.ablation == "subspace_rank":
            self.prober.determine_subspace_rank()
        else:
            index = self._get_index()
            evaluate(
                self.model_wrapper,
                index,
                get_timestamp(),
                self.alter_query_embedding,
                self.layer,
                probing_task=self.prober.config.probing_task,
                eval_str=f"{self.identification_str}{'_altered_query_embeddings' if self.alter_query_embedding else ''}",
                projection=self.projection,
            )

    def _get_index(self):
        index_file_str = f"./cache/indexes/{self.identification_str}.bin"
        if Path(index_file_str).is_file():
            index = faiss.read_index(index_file_str)
            logging.info(f"Index read from file {index_file_str}.")
        else:
            self.corpus = get_corpus(MSMARCO_CORPUS_PATH)
            index = self._make_index(index_file_str)

        return index

    def _make_index(self, index_file_str):
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
                if self.prober.config.probing_task == ProbingTask.COREF:
                    embs = [self._get_passage_embeddings_coref(passage) for passage in passages[start:end]]
                    embs = np.asarray(embs)
                elif self.prober.config.probing_task in {ProbingTask.TI, ProbingTask.AVG_TI}:
                    embs = [self._get_passsage_embeddings_ti(passage) for passage in passages[start:end]]
                elif self.prober.config.probing_task == ProbingTask.NER:
                    embs = [self._get_passsage_embeddings_ner(passage) for passage in passages[start:end]]
                elif self.prober.config.probing_task == ProbingTask.QC:
                    embs = self.model_wrapper.get_passage_embeddings_pyserini(passages[start:end], self.layer)
                else:
                    embs = self.model_wrapper.get_passage_embeddings_pyserini_with_intervention_at_layer(
                        passages[start:end], self.projection, self.layer
                    )
            else:
                raise ValueError(
                    f"Layer is {self.layer}. Specify differently" if not self.layer else f"ablation {self.ablation} not implemented."
                )

            index.add_with_ids(embs, pids[start:end])
        logging.info(f"Index made. Saving to file...")
        faiss.write_index(index, index_file_str)
        logging.info(f"Index for {self.identification_str} saved to file.")
        if self.prober.config.probing_task == ProbingTask.COREF:
            logging.info(f"Passages with corefs: {self.passages_with_corefs_count}")
            logging.info(f"Absolute coref count: {self.absolute_coref_count}")
            logging.info(f"Retokenization errors: {self.tokenization_error_count}")

        return index

    def _pepare_corpus_iteration(self):
        corpus_size = len(self.corpus)
        batches = get_batch_amount(corpus_size, BATCH_SIZE_LM_MODEL)
        passages = self.corpus["passage"].tolist()
        pids = self.corpus["pid"].to_numpy()

        return corpus_size, batches, passages, pids

    def _get_passage_embeddings_coref(self, passage):
        passage = re.sub(r"£(\d+)", r"£ \1", passage)
        embs, tokens = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        # detect corefenrences in passage and apply projection to them

        target_indices = []  # indices of tokens that are part of a coreference and should be projected
        passage_has_coref = False
        passage_had_error = False
        doc = self.nlp(passage)

        for cluster in doc._.coref_clusters:
            passage_has_coref = True
            self.absolute_coref_count += 1
            added_main_ref = False

            for mention in cluster.mentions[1:]:
                bert_spans, tokenization_error_main_ref, tokenization_error_ref = retokenize_spans(
                    self.model_wrapper,
                    self.spacy_tokenizer,
                    passage,
                    [[cluster.main.start, cluster.main.end], [mention.start, mention.end]],
                    cluster.main.text,
                    mention.text,
                )
                if tokenization_error_main_ref:
                    self.tokenization_error_count += 1
                    passage_had_error = True
                    continue
                elif tokenization_error_ref:
                    self.tokenization_error_count += 1
                    passage_had_error = True
                    if not added_main_ref:
                        added_main_ref = True
                        target_indices.extend(get_indices_from_span(bert_spans[0]))
                else:
                    if not added_main_ref:
                        added_main_ref = True
                        target_indices.extend(get_indices_from_span(bert_spans[0]))
                    target_indices.extend(get_indices_from_span(bert_spans[1]))

        if passage_has_coref:
            self.passages_with_corefs_count += 1

        if target_indices:
            target_indices = list(dict.fromkeys(target_indices))  # remove duplicate indices (can happen when entities overlap)
            inverted_target_indices = [i for i in range(len(tokens[0])) if i not in target_indices and i not in list(range(4))]

            embs = torch.from_numpy(embs)[0]
            try:
                embs_coref = torch.mean(torch.index_select(embs, 0, torch.tensor(target_indices)), dim=0)
            except:
                logging.critical(f"Error in passage: {passage}, {target_indices}, {len(tokens[0])}")
                if not passage_had_error:
                    self.tokenization_error_count += 1
                return torch.mean(embs[4:], dim=0).numpy()
            altered_embs_coref = torch.matmul(embs_coref.unsqueeze(0), self.projection.detach().cpu()).squeeze(0)
            altered_embs_coref = altered_embs_coref.repeat(len(target_indices), 1)

            embs_non_coref = torch.index_select(embs, 0, torch.tensor(inverted_target_indices))
            embs_concatted = torch.cat((altered_embs_coref, embs_non_coref), dim=0)

            return torch.mean(embs_concatted, dim=0).numpy()

        else:
            # passage does not contain coref
            return np.mean(embs[0, 4:], axis=0)

    def _get_passsage_embeddings_ti(self, passage):
        embs = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        # TODO: get non stopword tokens and apply projection to them
        return embs

    def _get_passsage_embeddings_ner(self, passage):
        embs = self.model_wrapper.get_passage_embeddings_unpooled([passage], self.layer)
        # TODO: get entities and apply projection to them
        return embs

    def _debug_part_of_bert_enconder(self):
        query = {156493: "do goldfish grow"}
        documents = {
            5203821: "D. Liberalism ............................................................................................................................................. 14. E. Constructivism ...................................................................................................................................... 19. F. The English School ...............................................................................................................................",
            2928707: "Goldfish Only Grow to the Size of Their Enclosure. There is an element of truth to this, but it is not as innocent as it sounds and is related more to water quality than tank size. When properly cared for, goldfish will not stop growing. Most fishes are in fact what are known as indeterminate growers.",
        }
        passages = list(documents.values())
        embs_altered = self.model_wrapper.get_passage_embeddings_pyserini_with_intervention(passages, self.prober.projection)
        embs, embs_altered2 = self.model_wrapper.get_passage_embeddings_pyserini_with_and_without_intervention(
            passages, self.prober.projection
        )
        embs2 = self.model_wrapper.get_passage_embeddings_pyserini(passages)
        inp, outp = self.model_wrapper._get_in_and_outputs_for_passages(passages)
        embeddings_outp = self.model_wrapper.part_model.embeddings.forward(inp.input_ids, inp.token_type_ids)
        embeddings_outp_pretrained = self.model_wrapper.model.embeddings.forward(inp.input_ids, inp.token_type_ids)
        extended_att_mask = self.model_wrapper.part_model.get_extended_attention_mask(inp.attention_mask, inp.attention_mask.shape)
        out_pob = self.model_wrapper.part_model.encoder.forward(outp.hidden_states[self.layer], attention_mask=extended_att_mask)
        embs_pob = self.model_wrapper.mean_pooling(out_pob.last_hidden_state[:, 4:, :], inp["attention_mask"][:, 4:])
        embs_2 = self.model_wrapper.mean_pooling(outp.last_hidden_state[:, 4:, :], inp["attention_mask"][:, 4:])
        pass


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
