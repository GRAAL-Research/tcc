import numpy as np
from spacy.tokens import Doc, Token

from .metrics_utils import get_num_sentences, get_num_words, safe_divide

DEPENDENT_CLAUSE_DEPS = ("conj", "ccomp", "orphan", "advcl")


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


def ps_30(sp_object: Doc) -> float:
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
def check_if_complex(sp_root_token: Token) -> bool:
    """Return whether a subordinate clause depends, directly or not, on the token."""
    for child in sp_root_token.children:
        # Subordinate in the children of the root
        if child.dep_[:3] == "acl" or child.dep_ in ["ccomp", "orphan", "advcl"]:
            return True
        if check_if_complex(child):
            return True
    return False


def _count_t_units(sp_object: Doc) -> tuple[int, int]:
    """Count the T-units and the complex T-units.

    "The following elements were counted as one T-unit: a single clause, a matrix plus subordinate clause, two or more
    phrases in apposition, and fragments of clauses produced by ellipsis. Coordinate clauses were counted as two
    t-units. Elements not counted as t-units include back-channel cues such as mhm and yeah, and discourse boundary
    markers such as okay, thanks or good. False starts were integrated into the following t-unit." (Young 1995:38)

    Args:
        sp_object: spaCy document of the text.

    Returns:
        The number of T-units and the number of complex T-units.
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
            if token.dep_ == "ROOT":
                complex_t_unit_pres = check_if_complex(token)
        if complex_t_unit_pres:
            complex_t_unit_count += 1

    return t_unit_count, complex_t_unit_count


def _get_t_units(sp_object: Doc) -> int:
    """Count the T-units (see :func:`_count_t_units`) without looking for complex ones."""
    # At least one T-unit per sentence, plus one per coordinate clause.
    return get_num_sentences(sp_object) + sum(1 for token in sp_object if token.dep_ == "conj")


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
    tu, ctu = _count_t_units(sp_object)
    return safe_divide(ctu, tu)


# ====================================================== #


# ==================== II - CLAUSES ==================== #
def _count_clauses(sp_object: Doc) -> tuple[int, int]:
    """Count the clauses and the dependent clauses.

    Args:
        sp_object: spaCy document of the text.

    Returns:
        The number of clauses and the number of dependent clauses.
    """
    dc_count = sum(1 for token in sp_object if token.dep_[:3] == "acl" or token.dep_ in DEPENDENT_CLAUSE_DEPS)
    # At least one clause per sentence
    return get_num_sentences(sp_object) + dc_count, dc_count


def _get_clauses(sp_object: Doc) -> int:
    """Count the clauses (see :func:`_count_clauses`)."""
    return _count_clauses(sp_object)[0]


def dc_c(sp_object: Doc) -> float:
    """
    Dependent Clause per Clause (DC_C)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the average number of Dependent Clauses per Clause
    """
    clause_count, dc_count = _count_clauses(sp_object)
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
