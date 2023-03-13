# -*- coding: utf-8 -*-
import re

import spacy
from spacy.tokens import Doc, Token
from typing import Union, Tuple, List


def _get_num_words(
    sp_object: Union[Doc, List[Token]],
    min_size: int = 0,
    get_size: bool = False,
    get_unique: bool = False,
    without_stop: bool = False,
) -> Union[int, Tuple[int, int]]:
    """Get the number of real words (not punctuation or quotes)

    Args:
        sp_object (Union[spacy.tokens.doc.Doc, list[spacy.tokens.token.Token]]): spaCy object (in case of the Doc type)
    or list of spaCy tokens (in case of the list of Token type) based on the text that will be computed.
        min_size (int, optional): minimum size of the word to be taken account of (Default at 0).
        get_size (bool, optional): get the number of letter in the real words (Default at False).
        get_unique (bool, optional): get the number of unique words (Default at False).
        without_stop (bool, optional): don't count stopwords (Default at False).

    Returns:
        int: the number of real words

    Original author: Nicolas Garneau
    """
    filtered_words = [
        word.text.lower()
        for word in sp_object
        if not word.is_punct
        and "'" not in word.text
        and len(word.text) >= min_size
        and (not word.is_stop or not without_stop)
    ]
    # print(filtered_words)
    if get_size:
        size = len("".join(filtered_words))
        return len(filtered_words), size
    if get_unique:
        return len(filtered_words), len(set(filtered_words))
    return len(filtered_words)


def _get_num_sentences(sp_object: Doc) -> int:
    """Get the number of sentences

    Args:
        sp_object (spacy.tokens.doc.Doc): spaCy object based on the text that will be computed.

    Returns:
        int: the number of sentences

    Original author: Nicolas Garneau
    """
    # The default sentence delimiters in spaCy include the semicolon, which we do not want.
    real_sentences = [sentences for sentences in sp_object.sents if sentences.text[-1] not in [",", ";"]]
    return len(real_sentences)


def _get_num_syllables(sp_object: Union[Doc, Token]) -> int:
    """Get the number of syllables

    Args:
        sp_object (Union[spacy.tokens.doc.Doc, spacy.tokens.token.Token]): spaCy object based on the text
    (in case of the Doc type) or on the word (in case of the Token type) that will be computed.

    Returns:
        int: the number of syllables

    Original author: Nicolas Garneau
    """

    def syllables_estimate(string: str) -> int:
        return len(re.findall(r"[aeiouyœéèëêîïôûüùâàô]+", string))

    # In the case where only a word spaCy object (spacy.tokens.token.Token) is passed.
    if isinstance(sp_object, Token):
        if not sp_object.is_punct and "'" not in sp_object.text:
            return syllables_estimate(sp_object.text)
        else:
            return 0

    filtered_words = [word.text for word in sp_object if not word.is_punct and "'" not in word.text]
    return syllables_estimate(" ".join(filtered_words))


def _safe_divide(x: Union[int, float], y: Union[int, float], returned_value: Union[int, float] = 0.0) -> float:
    """Safe divide x/y

    Args:
        x (Union[int, float]): denominator
        y (Union[int, float]): numerator
        returned_value (Union[int, float], optional): returned number if y = 0

    Returns:
        float: x/y or returns value (default at 0.0) if y = 0
    """
    if y == 0:
        return returned_value
    else:
        return x / y
