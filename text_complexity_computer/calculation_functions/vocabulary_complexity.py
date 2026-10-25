import json
from functools import cache
from math import log

from spacy.tokens import Doc

from ..tools import resource_path
from .metrics_utils import get_num_words, safe_divide


@cache
def _easy_words(language: str) -> str:
    return resource_path(language, "easy_words.txt").read_text(encoding="utf-8")


@cache
def _word_frequencies(language: str) -> dict[str, float]:
    return json.loads(resource_path(language, "word_frequencies.json").read_text(encoding="utf-8"))


def pa(sp_object: Doc, language: str) -> float:
    """
    Percentage of words not in a reference list (PA) of easy words using the Gougenheim list (Gougenheim et al., 1964).

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        language (str): The language to use for the easy word reference list.

    Returns:
        float: The average number of word not in a reference list.
    """
    # Substring lookup in the raw file content, as done when the bundled model was trained.
    easy_words = _easy_words(language)
    count = sum(1 for token in sp_object if token.lemma_ not in easy_words)
    return safe_divide(count, get_num_words(sp_object))


def nlm(sp_object: Doc) -> float:
    """
    Average size of words (NLM)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: average size of words
    """
    num_word, num_letter = get_num_words(sp_object, get_size=True)
    return safe_divide(num_letter, num_word)


def nmm_8(sp_object: Doc) -> float:
    """
    Average number of words with more than 8 letters (NMM_8)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: Average number of words with more than 8 letters
    """
    return safe_divide(get_num_words(sp_object, min_size=8), get_num_words(sp_object))


def uni_gram_lem(sp_object: Doc, language: str) -> float:
    """
    Uni-gram Model on Lemma (Uni_gram_Lem) (using a uni-gram model) using frequencies
    Lexique383 (New B. et al., 2005).

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        language (str): The language to use for the frequency list.


    Returns:
        float: the average probability value of words according to a frequency list.
    """
    normalized_prob = 0.0
    lexique_dic = _word_frequencies(language)

    for token in sp_object:
        frequency = lexique_dic.get(token.lemma_, 0)
        # Unknown or null frequency words get the lowest log probability of the list.
        normalized_prob += log(frequency) if frequency > 0 else -10

    return safe_divide(normalized_prob, get_num_words(sp_object))
