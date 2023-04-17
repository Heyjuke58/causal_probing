from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from src.hyperparameter import LAST_LAYER_IDX, MAX_LENGTH_MODEL_INPUT, MODEL_CHOICES


class ModelWrapper:
    def __init__(self, model_choice: str, device, layer: Optional[int] = None) -> None:
        self.model_choice = model_choice
        self.device = device
        # init model
        self.model_huggingface_str = MODEL_CHOICES[model_choice]
        model_config = get_model_config(self.model_huggingface_str)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_huggingface_str)
        self.model = BertModel.from_pretrained(self.model_huggingface_str, config=model_config).to(self.device)

        # init part of model. Needed when the intervention happens at an intermediate layer to produce a final embedding.
        self.part_model: BertPreTrainedModel = None
        if isinstance(layer, int):
            self.part_model = BertModel.from_pretrained(self.model_huggingface_str, config=model_config).to(self.device)
            for i in range(layer):
                self.part_model.encoder.layer.__delattr__(str(i))

    #######################################################################################################################################
    ### Query Embeddings

    def _get_in_and_outputs_for_queries(self, queries: List[str]):
        max_length = 36  # hardcode for now
        queries = ["[CLS] [Q] " + query + "[MASK]" * max_length for query in queries]
        inputs = self.tokenizer(
            queries,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        return inputs, outputs

    def get_query_embedding_pyserini(self, query: str) -> np.ndarray:
        _, outputs = self._get_in_and_outputs_for_queries([query])
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2).flatten()

    def get_query_embeddings_pyserini(self, queries: List[str], layer: Optional[int] = None) -> np.ndarray:
        _, outputs = self._get_in_and_outputs_for_queries(queries)
        if isinstance(layer, int):
            embeddings = outputs.hidden_states[layer].detach().cpu().numpy()
        else:
            embeddings = outputs.last_hidden_state.detach().cpu().numpy()
        return np.average(embeddings[:, 4:, :], axis=-2)

    def get_query_embeddings_pyserini_with_intervention_at_layer(
        self, queries: List[str], projection: torch.Tensor, layer: int
    ) -> np.ndarray:
        inputs, outputs = self._get_in_and_outputs_for_queries(queries)
        embs_altered = torch.einsum("bac,cd->bad", outputs.hidden_states[layer], projection)
        if layer != LAST_LAYER_IDX:
            extended_att_mask = self.part_model.get_extended_attention_mask(inputs.attention_mask, inputs.attention_mask.shape)
            embs_altered = self.part_model.encoder.forward(embs_altered, extended_att_mask).last_hidden_state.detach().cpu().numpy()
        else:
            embs_altered = embs_altered.detach().cpu().numpy()
        return np.average(embs_altered[:, 4:, :], axis=-2)

    def get_query_embeddings_pyserini_with_and_without_intervention(
        self, queries: List[str], projection: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, outputs = self._get_in_and_outputs_for_queries(queries)
        if isinstance(layer, int):
            embeddings = outputs.hidden_states[layer]
        else:
            embeddings = outputs.last_hidden_state
        embeddings = torch.mean(embeddings[:, 4:, :], dim=-2)
        altered_embeddings = torch.einsum("bc,cd->bd", embeddings, projection)
        return embeddings.detach().cpu().numpy(), altered_embeddings.detach().cpu().numpy()

    def get_query_embeddings_pyserini_with_intervention(self, queries: List[str], projection: torch.Tensor) -> np.ndarray:
        _, outputs = self._get_in_and_outputs_for_queries(queries)
        embeddings = torch.mean(outputs.last_hidden_state[:, 4:, :], dim=-2)
        embeddings = torch.einsum("bc,cd->bd", embeddings, projection)
        return embeddings.detach().cpu().numpy()

    #######################################################################################################################################
    ### Passage Embeddings

    @staticmethod
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_in_and_outputs_for_passages(self, passages: List[str]):
        passages = ["[CLS] [D] " + passage for passage in passages]
        inputs = self.tokenizer(
            passages,
            max_length=MAX_LENGTH_MODEL_INPUT,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        return inputs, outputs

    def get_passage_embeddings_pyserini(self, passages: List[str], layer: Optional[int] = None) -> np.ndarray:
        inputs, outputs = self._get_in_and_outputs_for_passages(passages)
        if isinstance(layer, int):
            hidden_state = outputs.hidden_states[layer]
        else:
            hidden_state = outputs.last_hidden_state
        return self.mean_pooling(hidden_state[:, 4:, :], inputs["attention_mask"][:, 4:]).detach().cpu().numpy()

    def get_passage_embeddings_unpooled(self, passages: List[str], layer: Optional[int] = None) -> np.ndarray:
        inputs, outputs = self._get_in_and_outputs_for_passages(passages)
        if isinstance(layer, int):
            hidden_state = outputs.hidden_states[layer]
        else:
            hidden_state = outputs.last_hidden_state
        tokens = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
        return hidden_state.detach().cpu().numpy(), tokens

    def get_passage_embeddings_pyserini_with_and_without_intervention(
        self, passages: List[str], projection: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """applies intervention on embeddings on average over tokens but returns both the intervened embedding and the original"""
        inputs, outputs = self._get_in_and_outputs_for_passages(passages)
        if isinstance(layer, int):
            hidden_state = outputs.hidden_states[layer]
        else:
            hidden_state = outputs.last_hidden_state
        del outputs
        embeddings = self.mean_pooling(hidden_state[:, 4:, :], inputs["attention_mask"][:, 4:])
        altered_embeddings = torch.einsum("bc,cd->bd", embeddings, projection)
        return embeddings.detach().cpu().numpy(), altered_embeddings.detach().cpu().numpy()

    def get_passage_embeddings_pyserini_with_intervention(self, passages: List[str], projection: torch.Tensor) -> np.ndarray:
        """applies intervention on embeddings on average over tokens"""
        inputs, outputs = self._get_in_and_outputs_for_passages(passages)
        embeddings = self.mean_pooling(outputs["last_hidden_state"][:, 4:, :], inputs["attention_mask"][:, 4:])
        embeddings = torch.einsum("bc,cd->bd", embeddings, projection)
        return embeddings.detach().cpu().numpy()

    def get_passage_embeddings_pyserini_with_intervention_at_layer(
        self,
        passages: List[str],
        projection: torch.Tensor,
        layer: int,
    ) -> np.ndarray:
        """applies intervention on embeddings of each token at specified layer and continues forward pass through model afterwards"""
        inputs, outputs = self._get_in_and_outputs_for_passages(passages)
        embeddings = torch.einsum("bac,cd->bad", outputs.hidden_states[layer], projection)
        del outputs
        if layer != LAST_LAYER_IDX:
            extended_att_mask = self.part_model.get_extended_attention_mask(inputs.attention_mask, inputs.attention_mask.shape)
            embeddings = self.part_model.encoder.forward(embeddings, extended_att_mask).last_hidden_state
        embeddings = self.mean_pooling(embeddings[:, 4:, :], inputs["attention_mask"][:, 4:]).detach().cpu()
        return embeddings.numpy()


def get_model_config(model_huggingface_str: str):
    return BertConfig(
        _name_or_path=model_huggingface_str,
        architectures=["BertModel"],
        attention_probs_dropout_prob=0.1,
        classifier_dropout=None,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        position_embedding_type="absolute",
        transformers_version="4.21.2",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=30522,
        output_hidden_states=True,
    )
