# -*- coding: utf-8 -*-

import json
import os
import pkgutil
from math import log

from spacy.tokens import Doc

from .metrics_utils import _get_num_words, _safe_divide


def pa(sp_object: Doc, language: str) -> float:
    """Percentage of words not in a reference list (PA) of easy words using the Gougenheim list
    (Gougenheim et al., 1964) in French and the X list in English.

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        language (str): The language to use for the easy word reference list.

    Returns:
        float: The average number of word not in a reference list.
    """
    data = pkgutil.get_data(__name__, os.path.join("resources", language, "easy_words.txt"))
    utf8data = data.decode("utf-8")

    count = 0
    for token in sp_object:
        if token.lemma_ not in utf8data:
            count += 1
    return _safe_divide(count, _get_num_words(sp_object))


def nlm(sp_object: Doc) -> float:
    """Average size of words (NLM)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: average size of words
    """
    num_word, num_letter = _get_num_words(sp_object, get_size=True)
    return _safe_divide(num_letter, num_word)


def nmm_8(sp_object) -> float:
    """Average number of words with more than 8 letters (NMM_8)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: Average number of words with more than 8 letters
    """
    return _safe_divide(_get_num_words(sp_object, min_size=8), _get_num_words(sp_object))


def uni_gram_lem(sp_object: Doc, language: str) -> float:
    """Uni-gram Model on Lemma (Uni_gram_Lem) (using a uni-gram model) using frequencies
    Lexique383 (New B. et al., 2005) for French and SUBTLEX for English (https://osf.io/djpqz/#!).

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        language (str): The language to use for the frequency list.


    Returns:
        float: the average probability value of words according to a frequency list.
    """
    normalized_prob = 0
    data = pkgutil.get_data(__name__, os.path.join("resources", language, "word_frequencies.json"))
    lexique_dic = json.loads(data.decode())
    lexique_keys = list(lexique_dic.keys())

    for token in sp_object:
        if token.lemma_ not in lexique_keys:
            normalized_prob -= 10  # the lowest probability of all the list
        else:
            if lexique_dic[token.lemma_] <= 0:
                normalized_prob -= 10  # the lowest probability of all the list
            else:
                normalized_prob += log(lexique_dic[token.lemma_])

    return _safe_divide(normalized_prob, _get_num_words(sp_object))
