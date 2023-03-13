# -*- coding: utf-8 -*-
from spacy.tokens import Doc, Token
from typing import Union, List

from .metrics_utils import _get_num_words, _safe_divide


def _get_filtered_words(sp_object: Doc, without_stop: bool = False) -> List[Token]:
    """Get a list of all filtered words. What is defined here as filtered is without punctuations and without stopwords.

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        without_stop (bool, optional): does not take into account stopwords. (Default to False)

    Returns:
        List[spacy.tokens.token.Token]: the list of filtered words as spaCy token.
    """
    return [
        word
        for word in sp_object
        if not word.is_punct and "'" not in word.text and (not word.is_stop or not without_stop)
    ]


def ttr(sp_object: Union[Doc, List[Token]]) -> float:
    """Type-Token Ration (TTR)
        (Templin et al., 1957)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the TTR

    """
    num_words, num_unique_words = _get_num_words(sp_object, get_unique=True, without_stop=True)
    return _safe_divide(num_unique_words, num_words)


def msttr(sp_object: Doc, segment_size: int = 50) -> float:
    """Mean Sequential TTR (MSTTR)
        (Johnson et al., 1944)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        segment_size (int, optional): the size of the segement for the sequential mean

    Returns:
        float: the MSTTR
    """
    sttr = list()
    filtered_words = _get_filtered_words(sp_object, without_stop=True)
    num_words = len(filtered_words)
    for i in range(0, num_words - num_words % segment_size, segment_size):
        sttr.append(ttr(filtered_words[i : i + segment_size]))
    return _safe_divide(sum(sttr), len(sttr))


def mattr(sp_object: Doc, window_size: int = 100) -> float:
    """Moving Average TTR (MATTR)
        (Covington and McFall, 2008)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.
        window_size (int, optional): the size of the window for the moving average

    Returns:
        float: the MATTR

    """
    wttr = list()
    filtered_words = _get_filtered_words(sp_object, without_stop=True)
    num_word = len(filtered_words) - window_size
    for i in range(num_word - num_word % window_size + 1):
        wttr.append(ttr(filtered_words[i : i + window_size]))
    return _safe_divide(sum(wttr), len(wttr))


def mtld(sp_object: Doc) -> float:
    """Mesure of Textual Lexical Diversity (MTLD)
        (P.M. McCarthy and S. Jarvis, 2010)

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        float: the MTLD
    """

    def mtld_unidir(sp_object):
        ttr_unidir = 0
        word_bank = set()
        count = 0
        t = 0
        for word in sp_object:
            count += 1
            if word.text not in word_bank:
                word_bank.add(word.text)
                t += 1
            else:
                # The TTR equilibrium point (at 0.720) is described by P.M. McCarthy et al, 2010
                if t / count <= 0.72 and count >= 10:
                    ttr_unidir += 1
                    word_bank = set()
                    t = 0
                    count = 0
        ttr_unidir += (1 - _safe_divide(t, count, returned_value=1)) / 0.28  # 1-0.72
        return _safe_divide(_get_num_words(sp_object), ttr_unidir)

    filtered_words = _get_filtered_words(sp_object, without_stop=True)
    filtered_words_rev = list(reversed(filtered_words))
    return (mtld_unidir(filtered_words) + mtld_unidir(filtered_words_rev)) / 2
