# -*- coding: utf-8 -*-

import re

from spacy.tokens import Doc

from .metrics_utils import (
    get_num_words,
    get_num_sentences,
    get_num_syllables,
    safe_divide,
)
from .syntactic_complexity import mean_length_sentences


def bingui(sp_object: Doc) -> float:
    """
    BINGUI (bingui)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of punctuations [,:;().!?] per sentence
    """
    return safe_divide(len(re.findall(r"[,:;().!?]", sp_object.text)), get_num_sentences(sp_object))


def fk_ease(sp_object: Doc) -> float:
    """
    Flesch-Kincaid Reading Ease score.

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        Flesch-Kincaid Reading Ease score

    Original author: Nicolas Garneau
    """
    num_sentences = get_num_sentences(sp_object)
    num_words = get_num_words(sp_object)
    num_syllables = get_num_syllables(sp_object)
    if num_sentences == 0 or num_words == 0 or num_syllables == 0:
        return 0
    words_per_sent = num_words / num_sentences
    syllables_per_word = num_syllables / num_words
    # The formula below was proposed by Flesch and Kincaid (1948)
    return 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)


def lm_100(sp_object: Doc) -> float:
    """
    Average number of syllable per 100 words (lm_100)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the number of syllable per 100 words
    """
    count = 0
    counts = []
    for i in range(0, len(sp_object), 100):
        for token in sp_object[i : 100 + i]:
            count += get_num_syllables(token)
        counts.append(count)
        count = 0
    return safe_divide(sum(counts), len(counts))


def km_formula(sp_object: Doc) -> float:
    """
    Kandel and Moles formula (KM)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the result of Kales and Moles formula
    """

    # The formula below was proposed by Kandel and Moles (1958) as an adjustment of the
    # Flesch-Kincaid Reading Ease for french
    return 207 - 1.015 * mean_length_sentences(sp_object) - 0.736 * lm_100(sp_object)
