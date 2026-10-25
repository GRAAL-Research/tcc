import re
from collections.abc import Iterable

from spacy.tokens import Doc, Token


def get_filtered_words(sp_object: Iterable[Token], min_size: int = 0, without_stop: bool = False) -> list[str]:
    """Return the lowercased real words (not punctuation nor words with an apostrophe).

    Args:
        sp_object: spaCy document, span or list of tokens of the text.
        min_size: Minimum number of letters of a word to keep it.
        without_stop: Whether to drop the stopwords.

    Original author: Nicolas Garneau
    """
    return [
        word.text.lower()
        for word in sp_object
        if not word.is_punct
        and "'" not in word.text
        and len(word.text) >= min_size
        and (not word.is_stop or not without_stop)
    ]


def get_num_words(sp_object: Iterable[Token], min_size: int = 0, without_stop: bool = False) -> int:
    """Return the number of real words (not punctuation nor words with an apostrophe).

    Args:
        sp_object: spaCy document, span or list of tokens of the text.
        min_size: Minimum number of letters of a word to count it.
        without_stop: Whether to ignore the stopwords.

    Original author: Nicolas Garneau
    """
    return len(get_filtered_words(sp_object, min_size=min_size, without_stop=without_stop))


def get_num_words_and_letters(sp_object: Iterable[Token]) -> tuple[int, int]:
    """Return the number of real words and their total number of letters."""
    words = get_filtered_words(sp_object)
    return len(words), len("".join(words))


def get_num_words_and_unique_words(sp_object: Iterable[Token], without_stop: bool = False) -> tuple[int, int]:
    """Return the number of real words and of unique real words."""
    words = get_filtered_words(sp_object, without_stop=without_stop)
    return len(words), len(set(words))


def get_num_sentences(sp_object: Doc) -> int:
    """Return the number of sentences, ignoring the segments ending with a comma.

    Original author: Nicolas Garneau
    """
    return sum(1 for sentence in sp_object.sents if sentence.text[-1] != ",")


def syllables_estimate(string: str) -> int:
    """Estimate the number of syllables as the number of vowel groups."""
    return len(re.findall(r"[aeiouyœéèëêîïôûüùâàô]+", string))


def get_num_syllables(sp_object: Doc | Token) -> int:
    """Return the number of syllables of a document or a single token.

    Punctuation and words with an apostrophe are ignored.

    Original author: Nicolas Garneau
    """
    if isinstance(sp_object, Token):
        if not sp_object.is_punct and "'" not in sp_object.text:
            return syllables_estimate(sp_object.text)
        return 0

    filtered_words = [word.text for word in sp_object if not word.is_punct and "'" not in word.text]
    return syllables_estimate(" ".join(filtered_words))


def safe_divide(x: float, y: float, returned_value: float = 0.0) -> float:
    """Return ``x / y``, or ``returned_value`` when ``y`` is 0."""
    if y == 0:
        return returned_value
    return x / y
