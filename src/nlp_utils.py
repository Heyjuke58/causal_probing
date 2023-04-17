import logging
from typing import List, Optional, Tuple

import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from spacy.attrs import LEMMA, NORM, ORTH, TAG
from unidecode import unidecode

import neuralcoref
from src.model import ModelWrapper


def retokenize_span(model_wrapper: ModelWrapper, spacy_tokenizer, original_text: str, span: list, ref: str) -> Tuple[list, bool]:
    """Given a list of original tokens and a span, recompute new span for the wordpiece tokenizer.

    :param original_text: the original text, where tokens are separated by whitespace.
    :param spans: the original spans w.r.t. the original tokens.
    :return:
    """
    vanilla_tokens = [token.text for token in spacy_tokenizer(original_text)]

    # everything left to the span
    text_left_span = " ".join(vanilla_tokens[: span[0]])
    text_original_span = " ".join(vanilla_tokens[span[0] : span[1]])
    input_ids = model_wrapper.tokenizer(original_text, add_special_tokens=False)["input_ids"]
    decoded = [model_wrapper.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    left_retokenized = model_wrapper.tokenizer(text_left_span, add_special_tokens=False)["input_ids"]
    span_retokenized = model_wrapper.tokenizer(text_original_span, add_special_tokens=False)["input_ids"]

    # shift by 4 for [cls] and [D]
    span = (4 + len(left_retokenized), 4 + len(left_retokenized) + len(span_retokenized))
    rejoined_span, correctly_rejoined_span = rejoin_tokens(decoded, ref, span)

    if not correctly_rejoined_span:
        logging.critical(f"Retokenization failed for reference: {ref}, {span}, {original_text}, {rejoined_span}")

    return span, correctly_rejoined_span


def retokenize_spans(
    model_wrapper: ModelWrapper, spacy_tokenizer, original_text: str, spans: list, main_ref: str, ref: Optional[str] = None
) -> Tuple[list, bool, bool]:
    """Given a list of original tokens and spans, recompute new spans for the wordpiece tokenizer.

    :param original_text: the original text, where tokens are separated by whitespace.
    :param spans: the original spans w.r.t. the original tokens.
    :return:
    """
    vanilla_tokens = [token.text for token in spacy_tokenizer(original_text)]

    # everything left to the span
    text_left_spans = [" ".join(vanilla_tokens[:a]) for a, _ in spans]
    text_original_spans = [" ".join(vanilla_tokens[a:b]) for a, b in spans]
    input_ids = model_wrapper.tokenizer(original_text, add_special_tokens=False)["input_ids"]
    decoded = [model_wrapper.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    left_retokenized = [model_wrapper.tokenizer(left_span, add_special_tokens=False)["input_ids"] for left_span in text_left_spans]
    span_retokenized = [model_wrapper.tokenizer(span, add_special_tokens=False)["input_ids"] for span in text_original_spans]

    new_spans = []
    tokenization_error_main_ref = False
    tokenization_error_ref = False

    for a, b in zip(left_retokenized, span_retokenized):
        # shift by 4 for [cls] and [D]
        span = (4 + len(a), 4 + len(a) + len(b))
        new_spans.append(span)

    rejoined_main_ref, correctly_rejoined_main_ref = rejoin_tokens(decoded, main_ref, new_spans[0])
    if not correctly_rejoined_main_ref:
        logging.critical(f"Retokenization failed for main reference: {main_ref}, {new_spans[0]}, {original_text}, {rejoined_main_ref}")
        tokenization_error_main_ref = True

    if len(new_spans) > 1:
        rejoined_ref, correctly_rejoined_ref = rejoin_tokens(decoded, ref, new_spans[1])

        if not correctly_rejoined_ref:
            logging.critical(f"Retokenization failed for reference: {ref}, {new_spans[1]}, {original_text}, {rejoined_ref}")
            tokenization_error_ref = True

    return new_spans, tokenization_error_main_ref, tokenization_error_ref


def rejoin_tokens(tokens: List, text: str, span: List) -> Tuple[bool, str]:
    """Rejoin tokens to the original text.

    :param tokens: the tokens to rejoin.
    :param text: the original text.
    :param span: the span of the original text.
    :return:
    """
    relevant_tokens = tokens[span[0] - 4 : span[1] - 4]
    if len(relevant_tokens) == 1:
        return relevant_tokens[0] == text.lower(), relevant_tokens[0]
    elif len(relevant_tokens) == 0:
        return False, ""
    relevant_tokens = [
        token + " " if not relevant_tokens[i + 1].startswith("##") else token for i, token in enumerate(relevant_tokens[:-1])
    ] + [relevant_tokens[-1]]
    relevant_tokens = [token.replace("##", "") for token in relevant_tokens]
    rejoined_text = "".join(relevant_tokens).strip()
    return rejoined_text == text.lower(), rejoined_text


def get_indices_from_spans(spans):
    indices_span_1 = list(range(spans[0][0], spans[0][1]))
    indices_span_2 = list(range(spans[1][0], spans[1][1]))
    indices_span_1.extend(indices_span_2)

    return indices_span_1


def get_indices_from_span(span):
    return list(range(span[0], span[1]))


PORTER_STEMMER = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


def preprocess(
    string: str, toktok: bool = False, stemming: bool = True, token_splitting: bool = False, stopword_removal: bool = True
) -> List[str]:
    """
    Tokenize string, remove stopwords, stemming

    Args:
        string (str): The text to preprocess

    Returns:
        List[str]: The resulting list of tokens
    """
    if toktok:  # gonna/gotta one token in toktok tokenizer like we want to have
        tokenizer = ToktokTokenizer()
        x = tokenizer.tokenize(string)
    else:
        x = word_tokenize(string)

    x = [word.lower() for word in x]

    x = [unidecode(word) for word in x]

    if token_splitting:
        # enable that terms splitted into multiple tokens (trans, ##der, ##mal) stay in result
        x = [word for word in x if word.isalnum() or x.startswith("##")]
    else:
        x = [word for word in x if word.isalnum()]

    if stopword_removal:
        x = [word for word in x if not word in STOPWORDS]

    if stemming:
        x = [PORTER_STEMMER.stem(word) for word in x]

    return x


## SPACY TOKENIZER EXCEPTIONS
TOKENIZER_EXCEPTIONS = {
    # do
    "don't": [{ORTH: "don", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Don't": [{ORTH: "Don", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "don’t": [{ORTH: "don", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Don’t": [{ORTH: "Don", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "doesn't": [{ORTH: "doesn", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Doesn't": [{ORTH: "Doesn", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "doesn’t": [{ORTH: "doesn", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "doesn’t": [{ORTH: "Doesn", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "didn't": [{ORTH: "didn", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Didn't": [{ORTH: "Didn", LEMMA: "do"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "didn’t": [{ORTH: "didn", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Didn’t": [{ORTH: "Didn", LEMMA: "do"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    # can
    "can't": [{ORTH: "can", LEMMA: "can"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Can't": [{ORTH: "Can", LEMMA: "can"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "can’t": [{ORTH: "can", LEMMA: "can"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Can’t": [{ORTH: "Can", LEMMA: "can"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "cannot": [{ORTH: "cannot"}],
    "Cannot": [{ORTH: "Cannot"}],
    "couldn't": [{ORTH: "couldn", LEMMA: "can"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Couldn't": [{ORTH: "Couldn", LEMMA: "can"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "couldn’t": [{ORTH: "couldn", LEMMA: "can"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Couldn’t": [{ORTH: "Couldn", LEMMA: "can"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    # have
    "haven't": [{ORTH: "haven", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Haven't": [{ORTH: "Haven", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "haven’t": [{ORTH: "haven", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Haven’t": [{ORTH: "Haven", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hasn't": [{ORTH: "hasn", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Hasn't": [{ORTH: "Hasn", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hasn’t": [{ORTH: "hasn", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Hasn’t": [{ORTH: "Hasn", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hadn't": [{ORTH: "hadn", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Hadn't": [{ORTH: "Hadn", LEMMA: "have"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hadn’t": [{ORTH: "hadn", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Hadn’t": [{ORTH: "Hadn", LEMMA: "have"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    # will/shall/should
    "won't": [{ORTH: "won", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Won't": [{ORTH: "Won", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "won’t": [{ORTH: "won", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Won’t": [{ORTH: "Won", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "willn't": [{ORTH: "willn", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Willn't": [{ORTH: "Willn", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "willn’t": [{ORTH: "willn", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Willn’t": [{ORTH: "Willn", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wouldn't": [{ORTH: "wouldn", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Wouldn't": [{ORTH: "Wouldn", LEMMA: "will"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wouldn’t": [{ORTH: "wouldn", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Wouldn’t": [{ORTH: "Wouldn", LEMMA: "will"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "shouldn't": [{ORTH: "shouldn", LEMMA: "should"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Shouldn't": [{ORTH: "Shouldn", LEMMA: "should"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "shouldn’t": [{ORTH: "shouldn", LEMMA: "should"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Shouldn’t": [{ORTH: "Shouldn", LEMMA: "should"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "shalln't": [{ORTH: "shalln", LEMMA: "shall"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Shalln't": [{ORTH: "Shalln", LEMMA: "shall"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "shalln’t": [{ORTH: "shalln", LEMMA: "shall"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Shalln’t": [{ORTH: "Shalln", LEMMA: "shall"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    # be
    "Im": [{ORTH: "Im"}],
    "im": [{ORTH: "im"}],
    "Id": [{ORTH: "Id"}],  # spacy thinks this is read as I'd but in fact in can be used as Id (identification)
    "id": [{ORTH: "id"}],
    "isn't": [{ORTH: "isn", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Isn't": [{ORTH: "Isn", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "isn’t": [{ORTH: "isn", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Isn’t": [{ORTH: "Isn", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wasn't": [{ORTH: "wasn", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Wasn't": [{ORTH: "Wasn", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wasn’t": [{ORTH: "wasn", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Wasn’t": [{ORTH: "Wasn", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "weren't": [{ORTH: "weren", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Weren't": [{ORTH: "Weren", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "weren’t": [{ORTH: "weren", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Weren’t": [{ORTH: "Weren", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "aren't": [{ORTH: "aren", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Aren't": [{ORTH: "Aren", LEMMA: "be"}, {ORTH: "'t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "aren’t": [{ORTH: "aren", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "Aren’t": [{ORTH: "Aren", LEMMA: "be"}, {ORTH: "’t", LEMMA: "not", NORM: "not", TAG: "RB"}],
    # degree C or F
    "°C": [{ORTH: "°C"}],
    "°c": [{ORTH: "°c"}],
    "°F": [{ORTH: "°F"}],
    "°f": [{ORTH: "°f"}],
    # gonna, wanna, etc.
    "gotta": [{ORTH: "gotta"}],
    "Gotta": [{ORTH: "Gotta"}],
    "gonna": [{ORTH: "gonna"}],
    "Gonna": [{ORTH: "Gonna"}],
    "wanna": [{ORTH: "wanna"}],
    "Wanna": [{ORTH: "Wanna"}],
    "lotta": [{ORTH: "lotta"}],
    "Lotta": [{ORTH: "Lotta"}],
}


def get_spacy_tokenizer():
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm-2.1.0", direct=True)

    import en_core_web_sm

    tokenizer = en_core_web_sm.load().tokenizer

    for key, val in TOKENIZER_EXCEPTIONS.items():
        tokenizer.add_special_case(key, val)

    return tokenizer


def get_spacy_pipeline():
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm-2.1.0", direct=True)

    import en_core_web_sm

    nlp = en_core_web_sm.load()
    neuralcoref.add_to_pipe(nlp)

    for key, val in TOKENIZER_EXCEPTIONS.items():
        nlp.tokenizer.add_special_case(key, val)

    return nlp
