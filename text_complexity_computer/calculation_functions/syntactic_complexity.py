# -*- coding: utf-8 -*-

from typing import Union, Tuple

import numpy as np
from spacy.tokens import Doc, Token

from .metrics_utils import get_num_sentences, get_num_words, safe_divide


def mean_length_sentences(sp_object: Doc) -> float:
    """
    Mean Length of Sentences (MLS)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the mean length of sentences
    """
    return safe_divide(get_num_words(sp_object), get_num_sentences(sp_object))


def nws_90(sp_object: Doc) -> int:
    """
    Length of the 90th percentile sentence (NWS_90)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        int: length of the 90th percentile sentence
    """
    if len(sp_object) == 0:
        # Empty document
        return 0
    sentences = list(range(get_num_sentences(sp_object)))
    if len(sentences) == 0:
        # Fail-case if the SpaCy model could not extract a clear sentence.
        # It happens rarely.
        return 0
    rank = int(np.percentile(sentences, 90))
    return get_num_words(list(sp_object.sents)[rank])


def ps_30(sp_object: Doc):
    """
    Percentage of sentences longer than 30 words (PS_30)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the percentage of sentences longer than 30 words
    """
    sentences_30 = [sentence for sentence in sp_object.sents if get_num_words(sentence) >= 30]
    return safe_divide(len(sentences_30), get_num_sentences(sp_object))


# ==================== I - T-UNITS ==================== #
def check_if_complex(sp_root_token: Token):
    for child in sp_root_token.children:
        # Subordinate in the children of the root
        if child.dep_[:3] == "acl" or child.dep_ in ["ccomp", "orphan", "advcl"]:
            return True
        if check_if_complex(child):
            return True
    return False


def _get_t_units(sp_object: Doc, complex_count: bool = False) -> Union[int, Tuple[int, int]]:
    """T-Units counter (_get_t_units)

    "The following elements were counted as one T-unit: a single clause, a matrix plus subordinate clause, two or more
    phrases in apposition, and fragments of clauses produced by ellipsis. Coordinate clauses were counted as two
    t-units. Elements not counted as t-units include back-channel cues such as mhm and yeah, and discourse boundary
    markers such as okay, thanks or good. False starts were integrated into the following t-unit." (Young 1995:38)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        complex_count (bool, optional): search for Complex T-Unit (Default at False).

    Returns:
        int: the number of T-Units if complex_count=False
        Tuple[int, int]: the number of T-Units and Complex T-Unit if complex_count=True
    """
    # At least one T-unit per sentence
    t_unit_count = get_num_sentences(sp_object)
    complex_t_unit_count = 0

    for sentence in sp_object.sents:
        complex_t_unit_pres = False
        for token in sentence:
            # The coordinate clauses are counted as separate T-Units. (see Docstring for more)
            if token.dep_ == "conj":
                t_unit_count += 1
            if complex_count:
                if token.dep_ == "ROOT":
                    complex_t_unit_pres = check_if_complex(token)
        if complex_count:
            if complex_t_unit_pres:
                complex_t_unit_count += 1

    if complex_count:
        return t_unit_count, complex_t_unit_count
    return t_unit_count


def mean_length_tunit(sp_object: Doc) -> float:
    """
    Mean Length of T-Units (MLT)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average size of T-Units
    """
    t_units = _get_t_units(sp_object)
    return safe_divide(get_num_words(sp_object), t_units)


def tu_s(sp_object: Doc) -> float:
    """
    T-unit per Sentence (TU_S)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of T-Units per sentence
    """
    return safe_divide(_get_t_units(sp_object), get_num_sentences(sp_object))


def ctu_tu(sp_object: Doc) -> float:
    """
    Complex T-units per T-unit (CTU_TU)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Complex T-Units per T-Unit
    """
    tu, ctu = _get_t_units(sp_object, complex_count=True)
    return safe_divide(ctu, tu)


# ====================================================== #


# ==================== II - CLAUSES ==================== #
def _get_clauses(sp_object: Doc, dependent_count: bool = False) -> Union[int, Tuple[int, int]]:
    """
    Clauses Counter (_get_clauses)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        dependent_count (bool, optional): search for Dependent Clauses (Default at False).

    Returns:
        int: the number of Clauses if dependent_count=False
        Tuple[int, int]: the number of Clauses and Dependent Clauses if dependent_count=True
    """
    clause_count = get_num_sentences(sp_object)
    dc_count = 0

    for token in sp_object:
        if token.dep_[:3] == "acl" or token.dep_ in [
            "conj",
            "ccomp",
            "orphan",
            "advcl",
        ]:
            clause_count += 1
            if dependent_count:
                dc_count += 1
    if dependent_count:
        return clause_count, dc_count
    return clause_count


def dc_c(sp_object: Doc) -> float:
    """
    Dependent Clause per Clause (DC_C)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Dependent Clauses per Clause
    """
    clause_count, dc_count = _get_clauses(sp_object, dependent_count=True)
    return safe_divide(dc_count, clause_count)


def c_s(sp_object: Doc) -> float:
    """
    Clauses per Sentences (C_S)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Clauses per phrase
    """
    return safe_divide(_get_clauses(sp_object), get_num_sentences(sp_object))


def c_tu(sp_object: Doc) -> float:
    """
    Clauses per T-Unit (C_TU)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Clauses per T-Unit
    """
    return safe_divide(_get_clauses(sp_object), _get_t_units(sp_object))


# ====================================================== #


# ==================== III - COORDINATE PHRASES ==================== #
def _get_coordinate(sp_object: Doc) -> int:
    """
    Coordinate Counter (_get_coordinate)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        int: the number of Coordinate phrases
    """
    coordinate_counter = 0
    for token in sp_object:
        if token.dep_ == "conj":
            coordinate_counter += 1
    return coordinate_counter


def cp_c(sp_object: Doc) -> float:
    """
    Coordinate phrase per Clause (CP_C)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Coordinate phrases per Clause
    """
    return safe_divide(_get_coordinate(sp_object), _get_clauses(sp_object))


def cp_tu(sp_object: Doc) -> float:
    """
    Coordinate phrase per T-Unit (CP_TU)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Coordinate phrases per T-Unit
    """
    return safe_divide(_get_coordinate(sp_object), _get_t_units(sp_object))


# ====================================================== #