# -*- coding: utf-8 -*-

import json
from math import log
from spacy.tokens import Doc
import pkgutil

from .metrics_utils import _get_num_words, _safe_divide


def pa(sp_object: Doc) -> float:
    """Percentage of words not in a reference list (PA)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of word not in a reference list, here Gougenheim list (Gougenheim et al., 1964).
    """
    data = pkgutil.get_data(__name__, "reference_lists/gougenheim_list.txt")
    gougenheim = data.decode('utf-8')

    count = 0
    for token in sp_object:
        if token.lemma_ not in gougenheim:
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


def uni_gram_lem(sp_object) -> float:
    """Uni-gram Model on Lemma (Uni_gram_Lem)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average probability value of words according to a frequency list,
    here Lexique383 (New B. et al., 2005), using a uni-gram model.
    """
    normalized_prob = 0
    data = pkgutil.get_data(__name__, "reference_lists/lexique383_dic.json")
    lexique383_dic = json.loads(data.decode())
    lexique_keys = list(lexique383_dic.keys())

    for token in sp_object:
        if token.lemma_ not in lexique_keys:
            normalized_prob -= 10  # the lowest probability of all the list
        else:
            if lexique383_dic[token.lemma_] <= 0:
                normalized_prob -= 10  # the lowest probability of all the list
            else:
                normalized_prob += log(lexique383_dic[token.lemma_])

    return _safe_divide(normalized_prob, _get_num_words(sp_object))
